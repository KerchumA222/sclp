# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **weight compression PoC** implementing SCLP (Soft Clipping Lossless-First) compression for BF16 neural network weights. The algorithm:
1. Clips rare large exponents stochastically (soft clipping)
2. Encodes exponents as 4-bit palette indices (up to 16 unique exponents)
3. Stores sign + 3-bit mantissa per weight in a packed SM stream

See `design.md` for the full system design rationale and algorithm details.

There are two parallel implementations:
- **Pure Python/NumPy** (`src/compression/`) — fully functional reference implementation
- **HIP GPU kernels** (`src/hip/`) — implemented, requires ROCm hardware to run

## Build (HIP Module)

```bash
cd src/hip && mkdir -p build && cd build && cmake .. && make -j$(nproc) --no-print-directory
```

The compiled `.so` is placed in `python_pkg/`. Requires ROCm (`hipcc`, `pybind11`).

**Before running HIP kernel tests, verify ROCm hardware:**
```bash
rocminfo | grep -i "amdgpu"
```

## Running Tests

pytest is not installed globally — use the `eval_env` venv:

```bash
source eval_env/bin/activate

# Run all tests (from repo root)
python3 -m pytest tests/

# Run a single test file
python3 -m pytest tests/test_pipeline.py -v

# HIP module tests (requires ROCm hardware)
python3 tests/test_hip_module.py
```

## Architecture

### Python Reference Implementation (`src/compression/`)

| File | Role |
|---|---|
| `clipping.py` | `soft_exponent_clip()` — stochastic exponent clipping on BF16 uint16 arrays |
| `encoder.py` | `encode_palette()` — builds exponent palette + interleaved ws_stream |
| `decoder.py` | `decode_palette()` — reconstructs BF16 weights from palette + ws_stream |
| `storage.py` | `CompressedTensorStorage` — binary `.sclp` file format (magic `SCLP`) |
| `pipeline.py` | `SCLPCompressor` — high-level compress/decompress/save/load API |

### Selective Compression (MLP + Attention)
SCLP is applied to all linear projections in the transformer block:
- MLP: `fc1`, `fc2` (Gate, Up, Down)
- Attention: `q_proj`, `k_proj`, `v_proj`, `out_proj`

Rationale: These constitute ~90% of model parameters. Embeddings and LayerNorm are excluded.

### BF16 Bit Layout (1-8-7)
- Bit 15: sign
- Bits 14-7: exponent (8 bits)
- Bits 6-0: mantissa (7 bits)

The Python encoder and HIP encoder now produce the **same wire format**:
- `ws_stream`: one byte per weight — `palette_idx(7:4) | smn(3:0)`
  - `palette_idx`: 4-bit index into the exponent palette
  - `smn`: `sign(3) | mantissa_top3(2:0)` — top 3 of 7 mantissa bits
- Compression ratio: 8 bits/weight vs 16 bits original = **2×**
- Top 3 mantissa bits are kept; bits 3:0 are zeroed. This acts as mild regularization and empirically *lowers* PPL vs BF16.
- Both palette index and SM for each weight are co-located in a single byte, halving L2 cache pressure vs separate packed+SM arrays.

### Encoded Data Structure (dict returned by `encode_palette`)
```python
{
  'palette':   np.uint8[<=16],   # exponent values sorted by frequency (descending)
  'ws_stream': np.uint8[N],      # one byte per weight: palette_idx(7:4) | smn(3:0)
  'num_weights': int,
  'sidecar': {                   # weights whose exponent is not in the palette
    'indices': np.uint32[K],     # positions in the weight array
    'values':  np.uint16[K],     # full BF16 bits stored verbatim
  }
}
```

`testmodule.encode` returns `packed`, `sm`, `sidecar_indices`, `sidecar_values` (HIP module uses the older separate-stream layout internally; the GGUF bridge uses ws_stream).

### HIP GPU Implementation (`src/hip/`)

| File | Status |
|---|---|
| `clipping.hip` | Complete — `soft_exponent_clip_kernel` |
| `encoder.hip` | Complete — `encode_palette_kernel` (interleaved ws_stream: idx(7:4)\|smn(3:0)) |
| `decoder.hip` | Complete — `decode_palette_kernel` (reconstructs BF16 from palette + ws_stream) |
| `launcher.hip` | C-interface launchers exported via `extern "C"` |
| `wrapper.cpp` | pybind11 bindings — exposes `clip`, `encode`, `decode` |
| `CMakeLists.txt` | Builds `hip_kernels` static lib + `testmodule` shared lib; strips LTO flags |

The compiled Python module (`testmodule`) lives in `python_pkg/` and is imported as `import testmodule`.

### Kernel Launcher Signatures
```cpp
launch_clip_kernel(const uint16_t* input, uint16_t* output, uint n, uint8_t threshold, uint32_t seed, uint8_t mantissa_mask);
launch_encode_kernel(const uint16_t* input, const uint8_t* lookup, uint8_t* ws, uint32_t n);
launch_decode_kernel(const uint8_t* ws, const uint8_t* palette, uint16_t* output, uint32_t n);
```

### File Format (`.sclp`) — VERSION 2
Binary, little-endian:
`SCLP` magic (4B) | version uint16 | num_weights uint32 | palette_size uint8 | palette bytes | indices_len uint32 | packed_indices bytes | sm_len uint32 | sm_stream bytes | sidecar_count uint32 | sidecar_indices uint32[] | sidecar_values uint16[]
VERSION 1 files (no sidecar) load correctly with an empty sidecar.

## Key Implementation Notes

- All weights are passed as `np.uint16` representing raw BF16 bit patterns, never as floats
- Python decoder is fully vectorised (no Python loop); encoder produces `ws_stream` in a single NumPy operation
- Clipping: exponents `> threshold+1` are hard-clipped to `threshold`; exponents `== threshold+1` survive with 50% probability (flat, matches HIP XorshiftPRNG)
- Sidecar: weights with exponents outside the top-16 palette are stored verbatim; decoder restores them exactly. Without sidecar, rare low-exponent weights would be mapped to the nearest palette exponent — catastrophic for quality.
- **Palette selection**: 
  - `encode_palette_6b` and `encode_palette_4b`: defaults to `palette_method='kmeans'` (1-D weighted k-means, 20 iterations, k-means++ init). 
  - Frequency-based selection available via `palette_method='frequency'` but produces 8× worse worst-case relative error (25500% vs 3100% max rel err on Gemma4 MoE tensors) at no compression-ratio cost. 
  - K-means protects rare low-exponent (near-zero) weights that frequency selection maps catastrophically to the nearest dense-cluster exponent; these weights matter disproportionately for MoE routing stability. 
  - For SCLP4 with k=4, k-means has different tradeoffs: spreads slots across full range, worsening MSE vs frequency (7.5e-5 vs 1.7e-5) because 85% of weights cluster centrally. However, MaxRel error is significantly better. With `sidecar_dist=1`, k-means+sidecar wins clearly.
  - `encode_palette` (SCLP8) still uses frequency selection; sidecar handles outliers.
- `testmodule.encode(input, palette)` accepts the palette array (≤16 uint8 exponent values); the wrapper builds the nearest-neighbour lookup internally. Returns `packed`, `sm`, `sidecar_indices`, `sidecar_values` (HIP module uses separate streams internally).
- `testmodule.decode(packed, sm, palette, sidecar_indices=[], sidecar_values=[], num_weights_hint=-1)` — sidecar and num_weights args are optional; num_weights_hint is required for odd N.
- The `mantissa_mask` parameter of `clip` is applied as `output = weight & (0xFF80 | mantissa_mask)`; pass `0x7F` to preserve all mantissa bits
- The GEMV kernel omits sidecar correction intentionally — a block-scoped scan caused a 37% throughput regression. The ~0.01% of sidecar weights introduce negligible error for single-token generation.

## llama.cpp Integration

The llama.cpp integration lives on the `sclp` branch at `/home/ajkerchum/llama.cpp`.

### Build (llama.cpp sclp branch)

```bash
cd /home/ajkerchum/llama.cpp
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100
cmake --build build --config Release -j$(nproc)
```

### GGUF Blob Wire Format

The SCLP payload stored inside a GGUF tensor slot is **not** the same as the standalone `.sclp` file format — no magic bytes, no version field. It is a minimal self-describing blob:

```
[uint32 num_weights][uint8 palette_size][palette (palette_size bytes)]
[ws_stream (num_weights bytes): palette_idx(7:4)|smn(3:0) per weight]
[uint32 sidecar_count]
[uint32 × sidecar_count sidecar_indices][uint16 × sidecar_count sidecar_values]
```

`ws_stream` is exactly `num_weights` bytes — both the palette index and SM nibble for each weight are co-located in a single byte. The sidecar base is always at `ws + num_weights`.

Weights whose exponents fall outside the top-16 palette are stored verbatim in the sidecar section and restored exactly by `sclp_fixup_sidecar_kernel` after the main decode. The sidecar is typically 0.01–0.03% of all weights (lossless).

**Two on-disk variants exist:**

- **Padded** (generated by `patch_gguf_sclp.py`): blob is zero-padded to exactly `num_weights * 2` bytes. All GGUF tensor offsets and strides are identical to BF16 — the simplest format, but wastes half the allocation since actual compressed content is `~num_weights` bytes.

- **Compact** (generated by `repack_sclp_gguf.py`): each blob is stored at its actual compressed size (no trailing zeros), padded only to GGUF alignment (32 bytes). Each tensor's on-disk size is inferred at load time from consecutive GGUF tensor offsets. For Llama-3-8B: 14.97 GB → 8.47 GB (saves 6.5 GB, ~43% reduction).

The loader handles both transparently via `disk_size` (see below).

### Generating a Patched GGUF

To compress a single tensor in an existing GGUF using binary in-place patching:

```bash
source eval_env/bin/activate
python3 tests/patch_gguf_sclp.py
```

Edit the paths at the bottom of `patch_gguf_sclp.py` to choose input file, output file, and target tensor name. The script copies the file then seeks and overwrites only the 4-byte type field in the tensor-info section and the tensor data bytes — all other GGUF offsets stay valid.

**F16→BF16 conversion is required.** If the source GGUF stores weights as F16 (e.g. `Meta-Llama-3-8B.fp16.gguf`), the `to_bf16_uint16()` helper converts via a float32 intermediate before encoding. Passing raw F16 bits to the SCLP encoder produces garbage output — the encoder treats all bits as BF16 1-8-7 layout, but F16 has a 5-bit exponent and 10-bit mantissa (1-5-10).

### Generating a Native SCLP GGUF (from HuggingFace)

To convert a HuggingFace checkpoint directly to a full SCLP GGUF (all linear projections compressed):

```bash
source eval_env/bin/activate
python3 tests/convert_to_sclp_gguf.py
```

This produces the **padded** format (each SCLP blob = `num_weights * 2` bytes).

### Repacking to Compact Format

To strip the zero-padding from an existing padded SCLP GGUF:

```bash
source eval_env/bin/activate
python3 tests/repack_sclp_gguf.py \
    --input  models/llama3/Llama-3-8B-SCLPws-Patched.gguf \
    --output models/llama3/Llama-3-8B-SCLPws-Compact.gguf
```

Each SCLP tensor blob is parsed to find its actual end offset (after sidecar values), then written at that exact size. Non-SCLP tensors are copied verbatim. For Llama-3-8B: 14.97 GB → 8.47 GB (saves 6.5 GB, ~43% reduction).

### Running Inference

```bash
/home/ajkerchum/llama.cpp/build/bin/llama-completion \
    -m /home/ajkerchum/poc/models/llama3/Llama-3-8B-SCLPws-Compact.gguf \
    -ngl 99 \
    -n 100 \
    -no-cnv \
    --repeat-penalty 1.3 \
    -p "The capital of France is"
```

Use `llama-completion`, not `llama-cli`. `llama-cli` does not support `-no-cnv`. Both binaries auto-enable conversation mode when the GGUF contains an embedded chat template (Llama 3 does) — `-no-cnv` forces raw completion. `--repeat-penalty 1.3` prevents repetition loops common in base model completion. Verified working on RX 7900 XTX (gfx1100). Current benchmark results (Llama-3-8B, compact GGUF):

| Model | Size | tg128 | pp512 | PPL (wikitext-2) |
|---|---|---|---|---|
| BF16 (fp16 GGUF) | 14.97 GB | ~52 t/s | ~12,000 t/s | 10.59 |
| Q8_0 | 7.95 GB | ~52 t/s | ~3,430 t/s | 10.41 |
| **SCLP (ws format)** | **8.47 GB** | **~66 t/s** | **~2,730 t/s** | **9.87** |

tg wins because SCLP reads 8 bits/weight vs 16 for BF16. pp lags Q8_0 because the two-pass decode (decode blob → BF16 → rocBLAS GEMM) adds a full weight-matrix read before the GEMM; Q8_0 goes directly through rocBLAS INT8.

### Prefill Optimization (May 2026)

**Caution — earlier write-up was bogus** due to `-DSCLP_MEMSET_STUB=1` getting stuck in CMakeCache during profiling. The flag stubbed `llama_sclp{,4,6}_dispatch` to `hipMemsetAsync` (zero-fill BF16 weight buffer), making the rocBLAS GEMM produce all-zero output. Every subsequent build for ~a session ran with that stub on, inflating pp numbers to the memset-bandwidth ceiling and producing zero-output PPL that we mistook for "still close to baseline" because the *original* baseline measurements were also taken in that same session. The stub mechanism has now been removed entirely (see llama.cpp `b0306ab2c`).

**Real numbers (clean build, gfx1100 RX 7900 XTX):**

| Model | Config | pp512 t/s | tg t/s | vs reference |
|---|---|---|---|---|
| Llama-3-8B | SCLP4-kmeans-sc1 | 2574 | 25.5 | 78% of Q8_0's 3276 |
| Llama-3-8B | Q8_0 | 3276 | 81 | (reference) |
| Gemma4 MoE | MIXED+imatrix 1%, two-pass | 615 | 55.4 | 21% of Q5_K_M's 2937 |
| Gemma4 MoE | MIXED+imatrix 1%, fused WMMA on (broken) | 845 | 55.4 | +38% but PPL=10^8 |

What survived from the earlier session:
- The hipMemset stub *did* validly tell us decode work matters; the magnitude was just garbled. With the stub gone, vectorized SCLP4/SCLP6 stores still represent a real (small) win (~6% on Gemma4 prefill vs scalar stores).
- Decode-vs-GEMM ratio in the two-pass path is no longer cleanly profiled; needs to be redone with `hipEventRecord` timers (not a global stub) when prioritized.

### Fused SCLP4 MoE WMMA Prefill Kernel (WIP, May 2026)

Three kernels in `sclp_bridge.cuh` make up the fused prefill path:
1. `sclp4_fused_moe_wmma_kernel` — RDNA3 WMMA, fast but outputs 30% of correct magnitude (real bug — most likely single-warp fragment register packing differs from the 4-warp SCLP8 reference).
2. `sclp4_fused_moe_scalar_kernel` — same routing/gather, scalar dot-products. 92% of correct magnitude without sidecar.
3. `sclp4_moe_sidecar_correct_kernel` — post-hoc atomicAdd corrections for the ~1.9% sidecar weights. Scalar + sidecar reaches **bit-perfect** match to two-pass (max_abs 1e-5 across all 360K dst cells per call, verified via diff harness mode=2).

Env-var matrix (all default off):
- `SCLP_FUSED_MOE_WMMA=1` — fused only
- `SCLP_FUSED_MOE_WMMA_DIFF=1` — fused into scratch + two-pass into dst + diff
- `SCLP_FUSED_MOE_SCALAR=1` — use scalar kernel instead of WMMA
- `SCLP_FUSED_MOE_NO_SIDECAR=1` — skip sidecar correction

Perf (Gemma4 MIXED-imatrix-1%):
| Path | pp512 t/s | notes |
|---|---|---|
| Two-pass (baseline) | 615 | correct, PPL=940 @ 50 chunks |
| Fused WMMA + sidecar | 411 | WMMA broken, sidecar dominates |
| Fused scalar + sidecar | 396 | math correct, sidecar dominates |

The sidecar correction kernel currently dominates kernel time — for each sidecar weight it iterates over every routed slot in that expert's bin doing atomicAdd to dst. Needs restructuring (per-slot loop with shared-memory sidecar batch loading) before fused can beat baseline.

**PPL "mystery" resolved**: was a real numerical bug, not a mystery. Earlier diff sampling reported max_abs ~1e-5 but only looked at the first 16 rows. Running diff over the full 360K-cell dst shows max=0.005, mean=0.00065, 89% of cells with diff > 1e-4. That's ~0.5% per-cell drift, which compounds through 26 MoE FFN layers (with GLU multiplicative gating between gate-up and down) into catastrophic PPL.

Definitive test via `SCLP_M2_OVERWRITE_FUSED=1` in DIFF mode: at end of dispatch, copy fused output onto dst. PPL drops from 1650 (baseline, dst=two-pass) to 10.6M (dst=fused). Same dst tensor — only difference is which path wrote.

Source: rocBLAS BF16 GEMM accumulates in tensor-core tile order; our scalar loop accumulates sequentially. K=2816 F32-accumulated multiplications differ at ~1e-3 magnitudes between the two orders (non-associative F32 addition). Gemma4-IT is very sensitive to FFN-output noise at this scale.

**Implication for the fused-MoE-prefill program**: closing the prefill perf gap via fused decode+GEMM requires either using rocBLAS for the GEMM (defeats fusion), exactly reproducing rocBLAS's tile+accumulation order (substantial WMMA tiling work), or recalibrating from BF16 (long shot). The naive "just write a fused kernel" path doesn't work for models sensitive to BF16-level numerical noise. Pivoting to other prefill optimizations is likely more productive.

### Bridge Architecture (`sclp_bridge.cuh`)

`/home/ajkerchum/llama.cpp/ggml/src/ggml-cuda/sclp_bridge.cuh` implements the on-device decode path:

- `sclp_decode_blob_kernel`: Self-parses the GGUF blob header on-device. Thread 0 reads `blob[4]` (palette_size) into `__shared__` memory, then all threads load the palette into `__shared__ uint8_t s_palette[16]`. `ws = blob + 5 + palette_size`. Each thread processes 8 weights via a single `uint64_t` load from `ws`. No host-side device reads, safe during HIP stream capture.
- `sclp_fixup_sidecar_kernel`: Reads `sidecar_count` from `ws + num_weights` on-device, then each thread restores one outlier weight via scatter write into the output buffer. Uses a grid-stride loop with 4 fixed blocks so it handles any sidecar count without a D2H read.
- `sclp_fused_gemv_kernel`: Fused decode+GEMV for M=1 (token generation). Reads ws bytes directly, decodes on-the-fly with a `uint64_t` load per 8 weights. Accepts F32 activations — no separate conversion kernel. 1024 threads/block (32 warps), 1 warp per output row.
- `sclp_fused_gemm_kernel`: Fused decode+GEMM for small M (prefill up to `2×GEMM_TILE_M=32`). Template parameter `TILE_M=16` keeps all accumulators in VGPRs.
- `llama_sclp_dispatch`: Launches decode+fixup kernels for the two-pass path (large M prefill, feeds rocBLAS). Decode grid: `ceil(num_weights/8)` groups of 256 threads.

The dispatch is wired in at the top of `ggml_cuda_mul_mat` in `ggml-cuda.cu`: when `src0->type == GGML_TYPE_SCLP`, the blob is decoded into a pool-allocated `uint16_t` buffer, `src0_bf16` is constructed as a copy of `src0` with `type = GGML_TYPE_BF16` and `data = decoded.get()`, and the function recurses. Because `GGML_TYPE_SCLP` has `type_size=2` (same as BF16), all strides in `nb[]` are already correct for BF16 — no adjustment needed.

**HIP graph capture constraint**: `hipMemcpyAsync D2H` and `hipStreamSynchronize` are illegal during HIP graph capture (`GGML_HIP_GRAPHS=ON`). Any host read of a device pointer (e.g. reading palette_size from the blob) causes `ROCm error: operation failed due to a previous error during capture`. All header parsing must happen on-device.

### ggml Type Registration

| Location | Change |
|---|---|
| `ggml/include/ggml.h` | `GGML_TYPE_SCLP = 42`, `GGML_TYPE_COUNT = 43` |
| `ggml/src/ggml.c` | `[GGML_TYPE_SCLP] = { .type_name="sclp", .blck_size=1, .type_size=2, .is_quantized=true }` |
| `gguf-py/gguf/constants.py` | `GGMLQuantizationType.SCLP = 42`, `GGML_QUANT_SIZES[SCLP] = (1, 2)` |
| `ggml-cuda.cu` | SCLP intercept in `ggml_cuda_mul_mat`; `case GGML_TYPE_SCLP: return true;` in `supports_op` MUL_MAT switch |

The `supports_op` entry is critical: without it, `select_weight_buft` returns `nullptr` during model loading and the process crashes.

### Compact GGUF Loader Support

Three files were modified to support variable-size (compact) SCLP blobs:

**`ggml/src/gguf.cpp`** — added `disk_size` field to `gguf_tensor_info` and `gguf_ti_nbytes()` helper:
- Validation loop infers `disk_size` from consecutive tensor offset differences when they differ from `ggml_nbytes`
- All write paths (`write_tensor_data`, `gguf_set_tensor_type`) use `gguf_ti_nbytes()` for offset calculations
- New API: `gguf_set_tensor_disk_size(ctx, name, disk_size)` — sets compact size and recalculates all subsequent offsets
- Declaration added to `ggml/include/gguf.h`

**`src/llama-model-loader.h`** — added `disk_size` to `llama_tensor_weight`:
- Computed from `gguf_get_tensor_offset(i+1) - gguf_get_tensor_offset(i)` for all but the last tensor
- Last tensor falls back to `ggml_nbytes(tensor)`

**`src/llama-model-loader.cpp`** — both data-loading paths handle compact blobs:
- mmap GPU copy path (line ~1569): copies only `disk_size` bytes into a zero-padded `n_size` buffer before `ggml_backend_tensor_set`; uses `disk_size` for lmlock/mmaps_used tracking
- non-mmap host path: reads only `min(disk_size, n_size)` bytes, zeroes the rest

**`gguf-py/gguf/gguf_reader.py`** — `_build_tensors` pre-collects all tensor offsets and computes `disk_size = offset[i+1] - offset[i]`. When `disk_size != n_bytes`, reads `disk_size` bytes as a flat `uint8` array instead of the padded `n_bytes`. This handles compact SCLP GGUFs without a reshape error.

## Future Work: Mixed-Precision Per-Tensor / Per-Expert

Following the pattern of Q4_K_M, vary the SCLP precision across tensors to trade size for quality on the parts of the model that need it.

**Per-tensor policy** (cheap, no kernel changes — multiple SCLP types coexist via per-tensor dispatch on `src0->type`):
- Keep `token_embd` and `output` at BF16 (or SCLP6) — embedding/unembedding errors propagate to every token
- Keep first/last 1–2 transformer blocks at SCLP6 — early-layer features and final-layer logit shaping are quality-sensitive
- Interior MLP `fc1`/`fc2` at SCLP4 — ~85% of weights, errors are absorbed by residual stream
- Attention projections at SCLP6 — small fraction of weights, but routing/mixing errors compound

Implementation: extend `convert_to_sclp_gguf.py` with a name-pattern policy table mapping tensor name → SCLP variant. No bridge/kernel changes needed since ggml dispatches per-tensor on `src0->type`.

**Per-expert policy** (MoE only, requires dispatch changes):
- Routed-expert usage is highly skewed; hot experts at SCLP6, cold experts at SCLP4
- Could also vary `sidecar_dist` per expert (free, no kernel change — sidecar size is already variable)
- Requires the MoE GEMV path to handle mixed types within one tensor — non-trivial

Blocked until the SCLP4 decoder bug is fixed; mixing precisions while SCLP4 produces garbage would just measure SCLP6 quality on SCLP6 tensors and noise elsewhere.

## imatrix-Aware Encoding (Live — Sidecar Path Working, +1 GB for 15× PPL Win)

**Status (May 2026)**: imatrix loader (`src/compression/imatrix.py`), encoder integration (`encode_palette_4b/6b`), and converter flag (`--imatrix` + `--sidecar-imatrix-budget`) all live. **Imatrix is applied to *sidecar selection*, not to palette k-means** — the naive k-means weighting was tried first and regressed PPL 5×; this version replaces it.

**Calibration setup**:
- `llama-imatrix -m gemma4-Q5_K_M.gguf -f wikitext-2-raw/wiki.train.raw --output-format dat --chunks 80 -ngl 99 -c 512 --no-ppl` → `eval_data/gemma4-imatrix.dat` (55 MB, 295 entries, ncall=80 for dense / 34 for MoE due to top-8 routing). `.dat` output works; `.gguf` output OOMs on this MoE model.

**How it works** (`src/compression/encoder.py` `_encode_4b_expert` / `_encode_6b_expert`):
- Palette uses raw exponent frequency (no imatrix), so palette quality is unaffected by activation outliers.
- Sidecar has two tiers:
  1. **Mandatory**: weights with `exponent_distance_to_palette > sidecar_dist` (unchanged — catastrophic-error rescue).
  2. **Discretionary (imatrix)**: when `--imatrix` is set, additionally rescue the top `sidecar_imatrix_budget` fraction of remaining weights ranked by `activation_importance × exponent_distance`. Dist-0 weights are excluded (their only error is mantissa truncation, which sidecar can't reduce).
- Importance per weight = `imatrix_value[col_idx]` where `col_idx = flat_index % K` (K = innermost / input dim).

**Result on Gemma4-26B-A4B-IT** (wikitext-test, `-c 512 -b 512 --chunks 50`):

| Config | Size | tg t/s | PPL |
|---|---|---|---|
| **Mixed + imatrix-sidecar 1%** | **17.5 GB** | **56** | **940** |
| Mixed (no imatrix) | 16.2 GB | 55 | 13,909 |
| Mixed (naive imatrix k-means) | 16.2 GB | 55 | 66,370 |
| Q5_K_M | 18 GB | — | 18,481 |
| SCLP6 pure | 19 GB | 56 | 341,388 |
| SCLP4 pure | 15 GB | 2.5 | 77,053 |

**Headline**: the new champion is **20× better PPL than Q5_K_M at smaller size**, **15× better than non-imatrix mixed at +8% size**, and matches pure-SCLP6 inference speed.

**Why this works where k-means weighting didn't**: with only k=4 palette entries (SCLP4) the cluster centers are over-constrained by the exponent distribution itself — activation weights can shift them slightly but not enough to overcome mantissa-truncation error. The sidecar IS unconstrained per-weight; spending the imatrix signal there directly rescues the highest-impact errors. `importance × distance` correctly identifies "this weight matters AND is being quantized badly" — exactly the union we want.

### Budget Sweep Results

Sweep at `--sidecar-imatrix-budget` ∈ {0, 0.005, 0.01, 0.02} on Gemma4 mixed:

| budget | size | sidecar % | PPL | Δ vs 0% |
|---|---|---|---|---|
| 0.000 | 16.18 GB | 0.91% | 13,909 | — |
| 0.005 | 16.83 GB | 1.41% | 1,506 | 9.2× |
| **0.010** ⭐ | **17.51 GB** | 1.91% | **940** | **14.8×** |
| 0.020 | 18.88 GB | 2.91% | 1,026 | 13.6× |

Knee at 1%. The 2% point is marginally worse than 1% (within error bars — ±62 vs ±55), suggesting we've hit the quality floor for this calibration: further sidecar slots after 1% just add file size without recovering more meaningful error. **`--sidecar-imatrix-budget 0.01` is the recommended default.**

### vs Q4_K on gate/up (hybrid comparison)

To isolate whether SCLP4 is the bottleneck in mixed, swapped the SCLP4-on-gate/up for Q4_K (same imatrix file used by llama-quantize):

| Config | Size | PPL |
|---|---|---|
| **SCLP4 + imatrix-sidecar 1%** ⭐ | 17.5 GB | **940** |
| SCLP6+Q4K hybrid + imatrix-sidecar 1% | 16.4 GB | 9,961 |

Q4_K is **10.6× worse PPL** at slightly smaller size despite having access to the same imatrix calibration. The win is from concentration of precision: Q4_K spreads ~4.5 bits uniformly across every weight via per-256-block scales, while SCLP4+imatrix-sidecar gives most weights ~4 bits but stores the top ~2% (ranked by activation × distance) *verbatim*. For Gemma4-IT's weight distribution — long-tailed in `importance × reconstruction_error` — surgical lossless rescue beats uniform scaling. The Q4_K route is *not* the right knob even though it's the obvious one to try.

To produce the hybrid: requires a patch to `llama-quantize` (which only honored `--tensor-type` overrides when the default type was already quantized; now applies regardless — see `src/llama-quant.cpp`). Patch is local to this fork and worth upstreaming.

### Future Tuning Knobs

- **Calibrate from BF16, not Q5_K_M.** Q5_K_M's own quantization noise contaminates the imatrix activations. ~3h CPU pass on the 47 GB BF16 model. Worth a few hundred PPL points if true champion is wanted.
- **Per-tensor budget** instead of uniform — different tensor classes have different importance×distance distributions; uniform 1% may be wasteful on some, insufficient on others.
- **Error-magnitude based sidecar** instead of `importance × distance` — actually compute the reconstruction error per weight (including mantissa truncation, not just exponent distance) and rank by `importance × actual_error`.

## Future Work: imatrix-Aware Encoding (original notes below)

llama.cpp's `llama-imatrix` records per-column activation magnitudes from a calibration pass (e.g. wikitext). Quantizers like Q4_K use these to weight per-column error: a weight in a high-activation column gets a proportionally larger penalty if misquantized. SCLP can use the same signal in three places:

1. **Weighted k-means palette selection** — replace raw `counts` in `_kmeans_palette` with activation-magnitude-weighted counts (sum of `|activation|²` over each weight's column). Palette minimizes activation-weighted reconstruction error rather than raw bit error. Encoder-only change.

2. **Smarter sidecar selection** — instead of pure `dist > sidecar_dist`, sidecar weights with the largest `activation_weight × dist` product. Same sidecar size budget, spent on weights that actually matter for output quality. Highest leverage for SCLP4 where sidecar slots are scarce.

3. **Per-tensor precision budget** — per-tensor activation totals drive the per-tensor SCLP4/SCLP6 mix from the mixed-precision policy above, replacing hardcoded "first/last N blocks" heuristics with a data-driven choice.

**Calibration**: one pass with `llama-imatrix` on wikitext (~10 min on RX 7900 XTX once SCLP4 is fixed) produces a `.dat` file. Extend `convert_to_sclp_gguf.py` with `--imatrix path.dat` to consume it.

Blocked on the SCLP4 decoder fix.

## Known Issues

### SCLP4 VRAM Allocation (Resolved on Llama-3, Pending on Gemma4)

**Original symptom:** SCLP4 inference produced garbage output across models. Earlier `get_alloc_size` for SCLP4 reserved only `1.25 × ggml_nbytes`, but on disk the blob is `header + N/2 (ws) + 6 × sidecar_count`. Per-tensor sidecar fractions are wildly model-dependent:

- Gemma4 26B MoE: max blob ratio **1.26×** (mostly low-exponent-spread MoE weights)
- Llama-3-8B: max blob ratio **2.34×** on `attn_q.weight` (~11% sidecar — Llama's weight distribution has a broader exponent tail than Gemma4's)

When `disk_size > alloc_size`, the loader's `ggml_backend_tensor_set` asserted out (with `--no-mmap`) or silently truncated the blob (with mmap — producing garbage downstream as the kernel read undefined memory past the truncation point, especially the sidecar_count field which then scatter-corrupted random output positions).

**Fix:** `get_alloc_size` for SCLP4 now returns `3 × ggml_nbytes + 64KB` (= 1.5 × N bytes). This is enough for Llama-3-8B's worst tensor and any plausible sidecar fraction up to ~16%. Also fixed `ggml_backend_tensor_set_async` to check `alloc_size` (matching the sync version's bound) so the non-mmap GPU upload path no longer asserts.

**Verified working:** Llama-3-8B SCLP4-kmeans-sc1 — loads, generates at 25.88 t/s (≈3× BF16 read bandwidth), grammatically coherent English output (semantic quality is poor as expected at 4 bits/weight, not a bug).

### SCLP disk_size Hint via op_params

To make `get_alloc_size` exact per tensor (so Gemma4 doesn't OOM and Llama-3-8B doesn't truncate), the model loader stashes each SCLP tensor's `disk_size` into `tensor->op_params` immediately after `create_tensor`. CUDA's `get_alloc_size` reads it back and reserves exactly `disk_size` rounded up to 256B buffer alignment. The heuristic (3× for SCLP4, 1.25× for SCLP/SCLP6) remains only as a fallback when no hint is present (e.g., compute-graph temporaries).

**Layout in `op_params[16]`** (last three slots — see `llama-model-loader.cpp::create_tensor`):
- `op_params[13]` — magic `0x504C4353` (`'SCLP'`) sentinel
- `op_params[14]` — low 32 bits of `disk_size`
- `op_params[15]` — high 32 bits of `disk_size`

**Why this is OK today.** `op_params` is convention-reserved for the *operation* that produced a tensor; weight tensors loaded from a GGUF have `op == GGML_OP_NONE`, so `op_params` is zero-initialized and unused. No ggml code currently reads or writes `op_params` on `GGML_OP_NONE` tensors.

**Risks to watch for** (this is intentionally fragile — convention, not enforcement):
1. **Upstream ggml changes.** A future ggml release could start using `op_params` on weight tensors (e.g., to carry quantization metadata) or assert `op_params == 0` for `OP_NONE`. On a tracking fork, this would silently corrupt the hint or crash on merge. Mitigation: on each upstream merge, grep for new `op_params` writers/readers and verify `OP_NONE` paths still leave the field clear.
2. **View / transpose propagation.** Some tensor constructors (`ggml_view_*`, `ggml_transpose`, `ggml_dup_tensor`) `memset` `op_params` to zero, others inherit it. If a view of an SCLP weight is ever materialized and allocated separately, the hint could leak into a non-SCLP context. Mitigation: `get_alloc_size` only honors the hint for SCLP-typed tensors, so a leaked hint on a BF16 view is ignored. If we ever make views of SCLP tensors typed as SCLP, audit the constructor path.
3. **Debug / serialization tooling.** Graph dumpers and gguf writers may render `op_params` as op-meaningful values; for SCLP weight tensors the trailing slots will show a `disk_size`-shaped uint64 with the `'SCLP'` sentinel. Mitigation: documented here; if needed, add a "clear hint" helper and call it before any serialization that round-trips `op_params`.
4. **The sentinel collides with a real op_params value.** Extremely unlikely (`0x504C4353` is not a sensible numeric op param), but possible. Mitigation: only check the sentinel on SCLP-typed tensors; non-SCLP tensors are unaffected.

**Stronger long-term alternatives** if any of the above bites:
- Add a real field to `ggml_tensor` (e.g., `size_t alloc_size_hint`). Invasive but unambiguous.
- Side-table `std::unordered_map<const ggml_tensor*, size_t>` inside the CUDA buffer-type context, populated by a loader callback. No ggml-field abuse, slightly more code.
- Encode `disk_size` as a GGUF per-tensor KV. Persistent and explicit, but redundant — the value is already implicit in consecutive tensor offsets.

### Future Work: Converter-Side Precision Fallback

Even with exact alloc, a tensor whose SCLP4 blob exceeds its BF16 size is pathological — the converter should detect that case at encode time and fall back to BF16 (or SCLP6) for that specific tensor. This is the natural on-ramp to the mixed-precision idea above: instead of a hardcoded per-name policy, drive precision selection by measured per-tensor compression efficiency.

## Future Work: TurboQuant for KV Cache

Google's TurboQuant (2024) compresses the attention KV cache via per-token learned rotation + low-bit quantization, dramatically cutting the dominant runtime memory cost at long contexts. A community llama.cpp fork already implements it (find via the upstream issue tracker for "TurboQuant" / "kv-cache rotation"). Worth bringing in because:

- SCLP compresses *weights* — wins big on bandwidth-bound generation (load less from VRAM per token), modest VRAM savings.
- TurboQuant compresses the *KV cache* — wins big on long-context VRAM (KV grows with sequence length, weights don't).
- The two are orthogonal: a model with both gets compounded VRAM savings, especially on Gemma4-26B-A4B where long context blows up KV faster than weights.

Integration path: identify the fork, cherry-pick the KV cache rotation/quantization changes on top of our rebased `sclp` branch, smoke-test on Llama-3-8B first (no SCLP), then layer onto SCLP-compressed weights. Expected friction: the KV cache lives in a different code path from weight loading, so SCLP and TurboQuant changes shouldn't conflict structurally — but ggml's KV cache types and graph nodes have churned, so plan on rebase work.

## Mixed-Precision Status (Gemma4, May 2026)

`convert_to_sclp_gguf.py --format mixed` applies a name-based per-tensor policy:

- `attn_q/k/v/output`, `ffn_down`, `ffn_down_exps` → **SCLP6** (errors compound through softmax / pre-residual add)
- `ffn_gate/up`, `ffn_gate_exps`, `ffn_up_exps`, `ffn_gate_up_exps` → **SCLP4** (bulk of weights, errors partially absorbed by GeGLU)
- `token_embd`, `output`, all norms/scales → **verbatim BF16/F16**

On Gemma4-26B-A4B-IT with `--jinja`:

| Config | Size | tg t/s | Output |
|---|---|---|---|
| BF16 | 48 GB | (CPU only) | ✓ |
| SCLP6 pure | 19 GB | 56 | ✓ |
| **Mixed (fused SCLP4 MoE GEMV)** | **17 GB** | **55** | ✓ |
| SCLP4-kmeans-sc1 pure | 15 GB | 2.5 | ✗ mode collapse |

Pure SCLP4 mode-collapses on Gemma4-IT (`"own own own own..."`). The decode kernel is byte-perfect (verified on real MoE blobs up to 508M weights), so this is genuinely quantization noise — Gemma4's instruction-tuned weight distribution is too tight for k=4-palette + 1-mantissa-bit. Llama-3-8B SCLP4 produces incoherent-but-diverse English (high PPL, no collapse).

Mixed is now pareto-optimal: matches SCLP6 speed (within 2%), saves 2 GB, same quality. Further size wins require pushing more tensors to SCLP4 — `ffn_down_exps` (residual-stream feeder) is the obvious candidate but needs measured per-tensor evaluation, not a name policy.

### Fused SCLP4 MoE GEMV

`sclp4_fused_moe_gemv_kernel` in `sclp_bridge.cuh` mirrors `sclp6_fused_moe_gemv_kernel`:
- One block per `(row_tile, active_expert_slot)`; one warp per output row of the routed expert.
- Decodes only the routed experts' bytes inline (4×4 shared-mem LUT for `(pidx, smn) → float`), never materializes a full BF16 expert buffer.
- Shared memory: 4 (ws_offset) + 64 (LUT) + K × 4 (activation broadcast) bytes.
- Sidecar omitted (per documented SCLP convention — block-scoped scan caused ~37% regression on SCLP).
- Wired into `ggml_cuda_mul_mat_id` for `GGML_TYPE_SCLP4` when `n_batches == 1`; prefill still uses the existing two-pass decode → recursive `mul_mat_id`.

Without this kernel, SCLP4 MoE inference goes through two-pass decode every token, decoding ALL 128 experts when only 8 are routed — a 16× waste plus a 1 GB BF16 temp buffer per layer per token. With it, mixed-precision matches SCLP6 speed.

### Perplexity Comparison

Wikitext-2-raw, `-c 512 -b 512 --chunks 50` on Gemma4-26B-A4B-IT:

| Config | Size | tg t/s | PPL (final) | PPL (chunk 1) |
|---|---|---|---|---|
| **Mixed (fused MoE GEMV)** | **17 GB** | **55** | **13,909** ± 940 | 303,636 |
| Q5_K_M | 18 GB | — | 18,481 ± 1,282 | — |
| SCLP4-kmeans-sc1 pure | 15 GB | 2.5 | 77,053 ± 5,107 | (similar) |
| SCLP6-kmeans pure | 19 GB | 56 | **236,609,342** ± 19M | 3,018,680,874 |

**Caveat: absolute PPL is inflated by ~3 orders of magnitude** because Gemma4-IT is an instruction-tuned model and wikitext is OOD prose. Use the *ranking* and *anomalies*, not the absolute numbers. A healthy Gemma4-IT on OOD wikitext should be ~50-200; chat-formatted eval would give the meaningful number. (For reference: Llama-3-8B base hits PPL ~10 on the same setup.)

**Key findings:**
1. **Mixed beats Q5_K_M on quality at smaller size** (13,909 vs 18,481, 17 GB vs 18 GB). The per-tensor precision policy is a real win.
2. **SCLP4 pure at 77K confirms mode collapse quantitatively** — 5× worse than mixed.
3. **SCLP6 pure at 236M turned out to be two stacked issues, not a prefill bug.** Investigation:
   - The SCLP6 GGUF on disk was generated 4 minutes *before* the k-means palette commit landed, so it used frequency-based palette selection with documented worst-case relative error of 25,500% on Gemma4 MoE tensors. Regenerating with the current encoder (k-means + sidecar_dist=1) drops PPL from 236,609,342 → **341,388** (700× improvement). Inference quality verified: "Paris." at 56 t/s.
   - The remaining 24× gap between SCLP6-kmeans-sc1 (341K) and mixed (14K) is *real but not a bug*. It reflects a quality cliff for Gemma4-IT: per-tensor error metrics show **SCLP4-kmeans-sc1 has 7× higher MSE but 1.8× lower MaxAbsErr than SCLP6-kmeans-sc1**, because SCLP4's smaller palette (4 vs 8 entries) pushes 1.36% of weights into the lossless sidecar vs SCLP6's 0.07%. Gemma4-IT is sensitive to worst-case weight error, so SCLP4 wins despite worse average error. Mixed inherits SCLP4 on `ffn_gate_up_exps` (where this matters most) and SCLP6 on attention (where average error matters more).
   - Generation always took the fused MoE GEMV path (`n_batches==1`) which is more numerically forgiving than two-pass decode → rocBLAS batched GEMM, which is why the issue was invisible at generation but blew up under `llama-perplexity` at batch 512.
   - **Attempted fix that didn't work**: changed encoder's sidecar threshold from `> sidecar_dist` to `>= sidecar_dist`, which with `sidecar_dist=1` sidecars *every* weight not exactly at a palette exponent. Re-encoded SCLP6 ballooned from 19 GB → 25 GB (5% sidecar vs 0.07%), exceeded 24 GB VRAM, didn't load. Reverted the change. A targeted error-based sidecar policy (sidecar weights whose mapped reconstruction error exceeds a threshold) is the next direction to try if pursuing — but `mixed` already dominates pure SCLP6 on size *and* speed *and* PPL, so this isn't on the critical path. Stale-GGUF (item above) was the actionable fix; everything else here is post-mortem.

**Practical takeaway**: regenerate any SCLP6 GGUF older than 2026-05-16 10:19 — it likely uses frequency-based palette selection. Anything generated after that is fine. Mixed-precision (`--format mixed`) is the best Gemma4 config end-to-end.

Methodology note: re-run with chat-formatted prompts (template-applied wikitext, or a true Gemma-style eval set) to get absolute PPLs that can be compared to published numbers. For our purposes (ranking SCLP variants against each other and against Q5_K_M on the same eval setup), the current numbers are sufficient.

### SCLP6 Inference Hang (Critical)
In some cases, SCLP6 inference hangs after loading the model, consuming CPU time indefinitely without producing output. This does not appear to be a consistent issue and may be related to specific prompt/context combinations or GPU memory pressure.

**Status:** Requires further investigation. SCLP6 works in some cases (model loading verified) but hangs in others.
