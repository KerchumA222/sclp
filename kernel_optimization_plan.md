# SCLP Kernel Optimization Plan

## Context

**Current performance (Llama-3-8B, RX 7900 XTX gfx1100):**
- Generation (M=1 decode-GEMV): 47.1 t/s vs 50.3 t/s BF16 → **94%** — nearly closed
- Prefill (M>1): 2517 t/s vs 3182 t/s BF16 → **79%** — main remaining gap

The generation path uses `sclp_fused_gemv_kernel` which decodes weights on-the-fly.
The prefill path falls through to a two-pass: `llama_sclp_dispatch` (decode → temp BF16 buffer) then recursive `ggml_cuda_mul_mat` using rocBLAS. The temp buffer is `N×K` uint16_t — for Llama-3-8B's largest layer (4096×14336 = 58M weights), that is **117 MB** of extra VRAM traffic on every prefill matmul.

---

## Task 1: Fused Prefill GEMM Kernel

### Why the current two-pass is slow

For a prefill with sequence length M:
1. **Decode pass**: read 12 bits/weight from VRAM (packed + sm), write 16 bits/weight to temp BF16 buffer.
2. **rocBLAS GEMM**: read 16 bits/weight again, plus 16 bits/activation, write result.

Total VRAM reads per weight element: **28 bits** (12 in + 16 out/in).
A fused kernel reads **12 bits** per weight element once, reusing it for all M input rows.

For M=32 the memory savings factor is ~2.3×; the benefit grows with M.

### Design Approach: Tiled Fused GEMM

#### Thread block / tile layout

```
TILE_M = 8    // output rows per block (adjustable, keep small for register pressure)
TILE_N = 1    // one output column (weight row) per warp
WARPS  = 8    // warps per block → 8 weight rows decoded simultaneously
BLK_X  = 256  // threads per block (WARPS * 64 for gfx1100 wavefront=64)
```

Each block is responsible for a `[TILE_M × WARPS]` tile of the output:
- Dimension N (weight rows): assigned at block level, `WARPS` rows per block.
- Dimension M (input rows / batch): assigned within the block across TILE_M slots.

Each warp computes one output row of the weight matrix dotted against `TILE_M` input rows, accumulating `TILE_M` partial sums simultaneously. The weight is decoded once and multiplied against all `TILE_M` activation values — this is the key reuse.

#### Shared memory layout

```cpp
__shared__ uint8_t s_palette[16];          // 16 bytes — exponent lookup
__shared__ uint8_t s_packed[CHUNK * 1];    // nibble-packed chunk loaded by block
__shared__ uint8_t s_sm[CHUNK];            // sm_stream chunk
__shared__ float   s_x[TILE_M][CHUNK];     // activation tile (TILE_M rows × CHUNK cols)
```

Loading activations into LDS before the inner loop eliminates redundant L2 traffic for each weight row.

#### Kernel pseudocode

```cpp
__launch_bounds__(256, 4)
__global__ void sclp_fused_gemm_kernel(
    const uint8_t*        __restrict__ blob,
    const __hip_bfloat16* __restrict__ X,   // [M × K] activation matrix, row-major
    float*                __restrict__ Y,   // [M × N] output matrix, row-major
    uint32_t N, uint32_t K, uint32_t M
) {
    // --- Header parse (thread 0) ---
    __shared__ uint8_t s_palette_size;
    __shared__ uint8_t s_palette[16];
    if (threadIdx.x == 0) s_palette_size = blob[4];
    __syncthreads();
    if (threadIdx.x < s_palette_size) s_palette[threadIdx.x] = blob[5 + threadIdx.x];
    __syncthreads();

    const uint8_t* packed = blob + 5 + s_palette_size;
    const uint8_t* sm     = packed + ((uint64_t)(N * K + 1) / 2);

    // Block handles weight rows [row_start, row_start + WARPS)
    const int WARPS = blockDim.x / 64;          // wavefront = 64 on gfx1100
    const int warp_id = threadIdx.x / 64;
    const int lane    = threadIdx.x & 63;
    const uint32_t weight_row = blockIdx.x * WARPS + warp_id;
    if (weight_row >= N) return;

    // Block handles output/activation rows [m_start, min(m_start + TILE_M, M))
    // blockIdx.y selects the M tile
    const uint32_t TILE_M = 8;
    const uint32_t m_start = blockIdx.y * TILE_M;
    const uint32_t m_end   = min(m_start + TILE_M, M);
    const uint32_t m_count = m_end - m_start;

    float acc[TILE_M] = {0.0f};   // register accumulators: one per input row

    const uint64_t row_base = (uint64_t)weight_row * K;

    // --- Inner loop: stride over K in chunks of 64*8=512 weights ---
    // Each lane handles 8 consecutive weights in the chunk (vectorized load)
    const uint32_t K8 = (K / 8) * 8;
    for (uint32_t k8 = lane * 8; k8 < K8; k8 += 64 * 8) {
        uint64_t w_base = row_base + k8;

        // Load 4 packed bytes (8 nibbles = 8 weight indices) and 8 sm bytes
        uint32_t p4; __builtin_memcpy(&p4, packed + (w_base >> 1), 4);
        uint64_t sm8; __builtin_memcpy(&sm8, sm + w_base, 8);

        // Pre-decode 8 weights into registers
        __hip_bfloat16 w[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            uint8_t pb    = (uint8_t)(p4  >> ((j >> 1) * 8));
            uint8_t p_idx = (j & 1) ? (pb & 0x0F) : (pb >> 4);
            uint8_t sm_v  = (uint8_t)(sm8 >> (j * 8));
            uint16_t bits = ((uint16_t)(sm_v >> 7) << 15)
                          | ((uint16_t)s_palette[p_idx] << 7)
                          | (sm_v & 0x7F);
            w[j] = *(__hip_bfloat16*)&bits;
        }

        // Accumulate over TILE_M input rows (weights decoded once, reused M times)
        #pragma unroll
        for (uint32_t mi = 0; mi < TILE_M; mi++) {
            if (mi >= m_count) break;
            const __hip_bfloat16* x_row = X + (m_start + mi) * K;
            float partial = 0.0f;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                partial += __bfloat162float(w[j]) * __bfloat162float(x_row[k8 + j]);
            }
            acc[mi] += partial;
        }
    }

    // Scalar tail (K % 8 != 0)
    for (uint32_t k = K8 + lane; k < K; k += 64) {
        uint64_t w_idx = row_base + k;
        uint8_t pb    = packed[w_idx >> 1];
        uint8_t p_idx = (w_idx & 1) ? (pb & 0x0F) : (pb >> 4);
        uint8_t sm_v  = sm[w_idx];
        uint16_t bits = ((uint16_t)(sm_v >> 7) << 15)
                      | ((uint16_t)s_palette[p_idx] << 7)
                      | (sm_v & 0x7F);
        float wf = __bfloat162float(*(__hip_bfloat16*)&bits);
        #pragma unroll
        for (uint32_t mi = 0; mi < TILE_M; mi++) {
            if (mi >= m_count) break;
            acc[mi] += wf * __bfloat162float(X[(m_start + mi) * K + k]);
        }
    }

    // Warp reduction: reduce each acc[mi] across 64 lanes
    #pragma unroll
    for (int mi = 0; mi < TILE_M; mi++) {
        // Full 64-lane reduction using __shfl_down (two steps of 32, 16, 8, 4, 2, 1)
        for (int offset = 32; offset > 0; offset >>= 1)
            acc[mi] += __shfl_down(acc[mi], offset);
        if (lane == 0 && mi < (int)m_count) {
            Y[(m_start + mi) * N + weight_row] = acc[mi];
        }
    }
}
```

#### Launch configuration

```cpp
inline void llama_sclp_fused_gemm(
    const void*   blob_ptr,
    const float*  src_f32,      // [M × K] F32 activations
    float*        dst_f32,      // [M × N] F32 output
    uint32_t N, uint32_t K, uint32_t M,
    void*         tmp_bf16,     // [M × K] __hip_bfloat16 scratch
    hipStream_t   stream
) {
    // Convert activations F32 → BF16
    dim3 cvt_block(256);
    dim3 cvt_grid((M * K + 255) / 256);
    f32_to_bf16_kernel<<<cvt_grid, cvt_block, 0, stream>>>(
        src_f32, (__hip_bfloat16*)tmp_bf16, M * K);

    constexpr int WARPS      = 4;    // weight rows per block
    constexpr int TILE_M     = 8;    // activation rows per block
    constexpr int THREADS    = WARPS * 64;
    dim3 block(THREADS);
    dim3 grid(
        (N + WARPS - 1) / WARPS,          // over weight rows
        (M + TILE_M - 1) / TILE_M         // over activation rows
    );
    sclp_fused_gemm_kernel<<<grid, block, 0, stream>>>(
        (const uint8_t*)blob_ptr,
        (const __hip_bfloat16*)tmp_bf16,
        dst_f32, N, K, M);

    // Sidecar fixup still needed — run decode-only sidecar kernel
    // (reads sidecar from blob, patches dst directly — needs M-aware version, see below)
}
```

**Note on sidecar fixup for prefill**: the current `sclp_fixup_sidecar_kernel` patches a flat BF16 decode buffer. For the fused GEMM, sidecar weights affect the dot products — each sidecar weight at position `(row, col)` contributes to all M output rows at `Y[mi * N + row]` by `(sidecar_val - palette_approx_val) * X[mi * K + col]`. This requires a dedicated `sclp_fused_gemm_sidecar_fixup_kernel`:

```cpp
__global__ void sclp_fused_gemm_sidecar_fixup_kernel(
    const uint8_t*        __restrict__ blob,
    const __hip_bfloat16* __restrict__ X,   // [M × K]
    float*                __restrict__ Y,   // [M × N]
    uint32_t N, uint32_t K, uint32_t M
) {
    // Parse header on thread 0
    __shared__ uint8_t s_palette_size;
    __shared__ uint32_t s_sidecar_count;
    if (threadIdx.x == 0) s_palette_size = blob[4];
    __syncthreads();

    const uint8_t* packed       = blob + 5 + s_palette_size;
    const uint8_t* sm           = packed + ((uint64_t)(N * K + 1) / 2);
    const uint8_t* sidecar_base = sm + N * K;

    if (threadIdx.x == 0) {
        uint32_t sc;
        __builtin_memcpy(&sc, sidecar_base, 4);
        s_sidecar_count = sc;
    }
    __syncthreads();
    if (s_sidecar_count == 0) return;

    const uint8_t* idx_base = sidecar_base + 4;
    const uint8_t* val_base = idx_base + (uint64_t)s_sidecar_count * 4;

    // Each thread handles one sidecar weight, across all M activation rows
    uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < s_sidecar_count; i += stride) {
        uint32_t flat_idx;
        uint16_t val_bits;
        __builtin_memcpy(&flat_idx, idx_base + i * 4, 4);
        __builtin_memcpy(&val_bits, val_base + i * 2, 2);

        // flat_idx = row * K + col in the weight matrix (row∈[0,N), col∈[0,K))
        uint32_t weight_row = flat_idx / K;
        uint32_t weight_col = flat_idx % K;

        // What did the palette approximation decode to for this position?
        uint8_t pb    = packed[flat_idx >> 1];
        uint8_t p_idx = (flat_idx & 1) ? (pb & 0x0F) : (pb >> 4);
        // s_palette is not in shared here — re-read from blob (rare, only sidecar)
        uint8_t exp   = blob[5 + p_idx];
        uint8_t sm_v  = sm[flat_idx];
        uint16_t approx_bits = ((uint16_t)(sm_v >> 7) << 15)
                             | ((uint16_t)exp << 7)
                             | (sm_v & 0x7F);

        float delta = __bfloat162float(*(__hip_bfloat16*)&val_bits)
                    - __bfloat162float(*(__hip_bfloat16*)&approx_bits);

        // Apply correction to all M output rows
        for (uint32_t mi = 0; mi < M; mi++) {
            float x_val = __bfloat162float(X[mi * K + weight_col]);
            atomicAdd(&Y[mi * N + weight_row], delta * x_val);
        }
    }
}
```

Because sidecar count is typically 0.01–0.03% of weights, this kernel does negligible work and the inner loop over M is short compared to K.

#### Integration in ggml_cuda_mul_mat

Replace the two-pass fallback in `ggml-cuda.cu` (lines 2414–2423):

```cpp
// Before (two-pass):
const int64_t num_weights = ggml_nelements(src0);
ggml_cuda_pool_alloc<uint16_t> decoded(ctx.pool(), num_weights);
llama_sclp_dispatch(src0->data, decoded.get(), num_weights, stream);
ggml_tensor src0_bf16 = *src0;
src0_bf16.type = GGML_TYPE_BF16;
src0_bf16.data = decoded.get();
ggml_cuda_mul_mat(ctx, &src0_bf16, src1, dst);

// After (fused GEMM path):
if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
    ggml_cuda_pool_alloc<__hip_bfloat16> x_bf16(ctx.pool(), (size_t)(M * K));
    llama_sclp_fused_gemm(
        src0->data,
        (const float*)src1->data,
        (float*)dst->data,
        (uint32_t)N, (uint32_t)K, (uint32_t)M,
        x_bf16.get(), stream);
    return;
}
// ... fallback for other types
```

#### Expected performance impact

For M=32 (common prefill batch):
- Current: decode N×K×12bits from VRAM + write N×K×16bits + read N×K×16bits again = 44 bits/weight
- Fused: decode N×K×12bits once + read M×K×16bits activations = 12 + 16M/N bits/weight effective

For Llama-3-8B mlp.gate_proj (N=14336, K=4096): M activations column reuse ratio = 14336/4096 ≈ 3.5×. At M=32 the benefit is maximal — weight stream touched once, activations fit in L2.

Theoretical speedup for M=32: ~2.0–2.5× over two-pass on the weight-VRAM-bandwidth-bound path.

---

## Task 2: Other Optimization Opportunities (Prioritized)

### 1. [HIGH IMPACT] 256-entry LUT for packed → two BF16 values

**Current hot path per weight pair** (in `sclp_fused_gemv_kernel`):
1. Load `packed_byte`
2. Unpack two nibbles (`>> 4`, `& 0x0F`)
3. Index `s_palette` twice
4. Load two `sm` bytes
5. Assemble two BF16 words via bit arithmetic

**Proposed**: precompute a 512-entry `uint32_t` LUT in LDS where `lut[packed_byte]` encodes the combined exponent bits for both weights (or a 256×2 layout). The nibble unpack and palette lookup collapse into a single LDS read. Combined with the sm stream this halves the decode arithmetic per weight pair.

```cpp
// LUT construction: run once at block start
__shared__ uint32_t s_exp_lut[256];  // 1KB LDS
if (threadIdx.x < 256) {
    uint8_t hi = s_palette[threadIdx.x >> 4];
    uint8_t lo = s_palette[threadIdx.x & 0xF];
    // Pack: high16 = exp0 field bits, low16 = exp1 field bits
    s_exp_lut[threadIdx.x] = ((uint32_t)hi << 23) | ((uint32_t)lo << 7);
    // Shift chosen so that ORing sm bits gives the final BF16 word directly
}
__syncthreads();

// Inner loop decode (one packed byte → two weights):
uint32_t exp_pair = s_exp_lut[packed_byte];
uint16_t bits0 = (uint16_t)(exp_pair >> 16) | ((uint16_t)(sm0 & 0x7F)) | ((uint16_t)(sm0 >> 7) << 15);
uint16_t bits1 = (uint16_t)(exp_pair & 0xFFFF) | ((uint16_t)(sm1 & 0x7F)) | ((uint16_t)(sm1 >> 7) << 15);
```

**Estimated impact**: reduces arithmetic in the inner loop by ~3 instructions per weight pair. For bandwidth-bound kernels this is noise; for compute-bound kernels at large M it may matter. LDS usage increases by 1KB per block, reducing occupancy slightly — profile before committing.

### 2. [HIGH IMPACT] Sidecar fixup merged into GEMV kernel

**Current**: `sclp_fixup_sidecar_kernel` runs as a separate kernel launch after decode, adding kernel-launch latency (~5–10µs on HIP) and a full round-trip through the output buffer. For inference with many small matmuls (Llama-3-8B has ~224 SCLP tensors), this is ~2.24 ms/token of pure latency overhead even if the kernels do trivial work.

**Proposed**: inline the sidecar fixup at the end of `sclp_fused_gemv_kernel`. Since sidecar count is stored in the blob and can be read by one thread into LDS, the main warp handles all sidecar weights in a grid-stride loop after its normal reduction.

```cpp
// At the end of sclp_fused_gemv_kernel, after y[row] = acc:
// Thread 0 of each block reads sidecar_count into LDS once
// All warps grid-stride over sidecar entries with atomicAdd to y[]
```

The atomicAdd to F32 is safe (single scatter per sidecar entry) and the sidecar count is so small (~0.01%) that contention is negligible.

**Estimated impact**: eliminates 224 extra kernel launches per token → ~2ms/token saved, or ~2–3 t/s at 47 t/s baseline. Also eliminates one extra output-buffer read-write pass during sidecar fixup.

### 3. [MEDIUM IMPACT] Blob header parsing: eliminate second `__syncthreads`

**Current `sclp_decode_blob_kernel`** has two `__syncthreads` barriers:
1. After thread 0 reads `palette_size_s`.
2. After all threads load palette entries.

The second barrier is necessary. The first can be eliminated if thread 0 reads `palette_size` and all palette entries atomically before any other thread needs the data. Since palette size ≤ 16, all 16 palette bytes can be read by the first 16 threads (threadIdx.x < 16) without needing `palette_size_s` — just read up to 16 bytes and let the kernel use a conservative bound.

```cpp
// Alternative: read all 16 possible palette bytes unconditionally
// (safe — bytes beyond palette_size are never used since packed indices < palette_size)
if (threadIdx.x < 16) s_palette[threadIdx.x] = blob[5 + threadIdx.x];
__syncthreads();  // Only ONE barrier needed
```

This trades one `__syncthreads` + one LDS write broadcast for 16 unconditional byte loads. On gfx1100 each `__syncthreads` costs ~4 cycles plus potential stall. Saving one barrier per kernel launch saves nothing at large K but eliminates a hazard at small K.

**Estimated impact**: Tiny in absolute terms. Worth doing for correctness/cleanliness regardless.

### 4. [MEDIUM IMPACT] GEMV grid sizing for gfx1100

**Current** (`llama_sclp_fused_gemv`):
```cpp
constexpr int WARPS_PER_BLOCK = 16;
dim3 gemv_block(WARPS_PER_BLOCK * 32);  // 512 threads — but wavefront = 64 on gfx1100!
```

**Problem**: gfx1100 (RDNA3) uses 64-thread wavefronts, not 32-thread warps. A block of 512 threads = 8 wavefronts. `__shfl_down` with offset ≤ 31 works within a 64-lane wavefront, but the current warp reduction uses `offset = 16` as the max, which only reduces within a 32-lane warp. On RDNA3, threads 32–63 within a wavefront are never reduced against threads 0–31, so **the reduction is incorrect on gfx1100** — each wavefront produces two independent partial sums in lanes 0 and 32, but only lane 0's result is written.

**Fix**: change the reduction to use offsets up to 32:
```cpp
for (int offset = 32; offset > 0; offset >>= 1)
    acc += __shfl_down(acc, offset);
if (lane == 0) y[row] = acc;
```

And redefine `warp_id` and `lane` based on wavefront size 64:
```cpp
const int warp_id = threadIdx.x / 64;
const int lane    = threadIdx.x & 63;
```

**Estimated impact**: This is a correctness bug on gfx1100 producing silently wrong output (approximately halved dot products). Fixing it is the highest priority correctness fix. The performance impact after fixing may be neutral (same number of reduction steps) but the arithmetic will be correct.

**Verification**: compare `sclp_fused_gemv_kernel` output against the two-pass decode path for a known weight + activation vector. If they diverge by ~50%, this is the cause.

### 5. [LOW IMPACT] Use `__builtin_amdgcn_ds_bpermute` for palette broadcast

Instead of LDS for the palette, use wavefront-level `__builtin_amdgcn_ds_bpermute` (RDNA3 DS_BPERMUTE instruction) to broadcast palette entries across the 64 lanes without LDS allocation. Each lane holds one palette entry in a register, and any lane can read another lane's register via the permute instruction.

```cpp
// Each lane i holds s_palette[i] in a register (lanes 0..15 used, 16..63 unused)
uint8_t my_palette_entry = (lane < palette_size) ? blob[5 + lane] : 0;
// To decode: p_exp = __builtin_amdgcn_ds_bpermute(p_idx * 4, my_palette_entry)
```

This eliminates 16 bytes of LDS usage per wavefront and removes the LDS bank conflict risk during palette lookups. On gfx1100 the DS_BPERMUTE latency is ~24 cycles vs ~4 cycles for LDS broadcast — this is likely a regression for high-palette-utilization cases.

**Estimated impact**: Probably neutral or slightly negative unless occupancy is LDS-bound. Profile first.

### 6. [LOW IMPACT] Pack sm + palette decode into a single wide load

The `sm_stream` and `packed_indices` arrays are accessed in predictable strides (sm is byte-indexed sequentially, packed is nibble-indexed). Combining them into an interleaved layout would enable a single 128-bit load per 8 weights instead of two separate 64-bit + 32-bit loads. However, this requires changing the on-disk GGUF blob format — a significant compatibility break.

**Estimated impact**: At most 10–15% bandwidth reduction on the decode pass. Not worth the format break.

---

## Unknowns Requiring Hardware Profiling

1. **Wavefront reduction bug (item 4)**: Confirm whether `sclp_fused_gemv_kernel` actually produces correct output on gfx1100. Run `python3 tests/test_hip_module.py` comparing GEMV against decode+dot product reference — a ~50% systematic error would confirm the 32 vs 64 wavefront mismatch.

2. **Prefill bottleneck**: Profile the two-pass path with `rocprof --hsa-trace` to determine what fraction of time is decode vs rocBLAS GEMM. If rocBLAS is already efficient (L2 cache hits on the decoded buffer), the fused GEMM may give only modest gains for M≥64 where rocBLAS's tensor core path dominates.

3. **Register pressure in fused GEMM**: The `acc[TILE_M]` array of 8 F32 accumulators + 8 BF16 weight values + loop variables may spill to scratch memory on gfx1100 (256 VGPR limit per wavefront). Profile VGPR usage with `llvm-readobj --elf-output-style=GNU -S <kernel.o>` or `rocm-smi --showmemuse`. If spilling, reduce TILE_M to 4.

4. **LUT approach (item 1)**: The 1KB LDS cost for the 256-entry LUT may reduce occupancy from 4 to 3 blocks/CU on gfx1100. Profile occupancy with rocprof before adding the LUT.

5. **Sidecar fixup latency (item 2)**: Measure actual kernel launch overhead empirically. On ROCm 6.x this is typically 5–15µs per launch. For 224 tensors × 2 kernels × 47 t/s = ~21,000 kernel launches/second, the overhead could be 100–300ms/second — a meaningful fraction if at the high end.

---

## Recommended Implementation Order

| Priority | Item | Risk | Estimated Gain |
|----------|------|------|---------------|
| 1 | Fix 32 vs 64 wavefront reduction bug | Low (correctness fix) | Correctness + up to 2× GEMV accuracy |
| 2 | Fused prefill GEMM kernel (this doc) | Medium (new kernel) | ~20–30% prefill speedup |
| 3 | Merge sidecar fixup into GEMV | Low | ~2–3 t/s generation |
| 4 | Single-barrier header parse | Very low | Negligible but clean |
| 5 | LUT for packed decode | Medium (LDS tradeoff) | Profile first |
| 6 | DS_BPERMUTE palette | High (arch-specific) | Likely neutral |
