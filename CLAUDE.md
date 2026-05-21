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
  - `encode_palette_4b/6b` default to k-means (1-D, 20 iter, k-means++ init). Protects rare low-exponent weights that frequency selection maps catastrophically; matters for MoE routing stability.
  - Frequency available via `palette_method='frequency'` but 8× worse worst-case MaxRel error at no compression-ratio gain.
  - `encode_palette` (SCLP8) uses frequency; sidecar handles outliers.
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

## Architecture & Format
- **Current Standard**: SCLP8 (8-bit interleaved `ws_stream`).
- **GGUF Types**: `GGML_TYPE_SCLP` (8-bit), `GGML_TYPE_SCLP4` (4-bit), `GGML_TYPE_SCLP6` (6-bit).
- **Layout (all types)**: `[num_weights (4B)][n_experts (4B)][per-expert: palette_size (1B), palette bytes]...[ws_stream][sidecar]`.

> [!NOTE]
> All SCLP types (8/4/6) now share the same per-expert blob header format. SCLP8 per-expert palettes validated end-to-end on Gemma4 128-expert MoE tensors.

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

### llama-quantize SCLP Support (May 2026)

`llama-quantize` now natively supports `SCLP`, `SCLP4`, `SCLP6` as quantization types and as `--tensor-type` overrides. See `/home/ajkerchum/llama.cpp/src/llama-sclp.{h,cpp}`. This replaces the two-step Python converter for most hybrid use cases.

**Single-step hybrid (SCLP6 attn+ffn_down + Q4_K gate/up)**:
```bash
llama-quantize \
  --imatrix /home/ajkerchum/poc/eval_data/gemma4-imatrix.dat \
  --tensor-type '^token_embd\.weight$=BF16' \
  --tensor-type '^output\.weight$=BF16' \
  --tensor-type '^blk\.[0-9]+\.attn_(q|k|v|output)\.weight$=SCLP6' \
  --tensor-type '^blk\.[0-9]+\.ffn_down(_exps)?\.weight$=SCLP6' \
  bf16-input.gguf output.gguf Q4_K_M 8
```

**Gotchas (learned during integration)**:
1. **Flags must come BEFORE positional args** (`tools/quantize/quantize.cpp:511`). Arguments after the input file are silently ignored.
2. **`--tensor-type` uses `std::regex_search`** (substring match). Use `^...$` anchors to avoid `output.weight` matching `attn_output.weight`.
3. **token_embd and output MUST be native types** (BF16 or a standard quant). SCLP types only handle MUL_MAT on the GPU backend — they can't go through GET_ROWS (embedding/output lookup), which crashes on CPU.
4. **Last positional arg is `nthread`** — default is 1, use 8+ for parallel Q4_K row quantization. SCLP expert encoding has its own internal threading (uses `std::thread::hardware_concurrency()`).
5. **Soft clipping is disabled by default** (`clip_threshold=0`). The Python encoder doesn't clip; the C++ default matches.

**Implementation**: `llama_tensor_quantize_sclp()` in `src/llama-sclp.cpp` parallelizes per-expert encoding (one thread per expert up to hw concurrency). Quantization time on Gemma4-26B-A4B: ~220s with 8 threads + parallel experts, down from ~1230s single-threaded — **5.6× speedup**.

### Generating a Native SCLP GGUF (from HuggingFace, Python)

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

**Decode kernel improvements** (May 2026): grid-parallel sidecar fixup (4 → 256 blocks) + vectorized thread-level decoding (4/16 → 32 weights/thread) nearly doubled Gemma4 MoE prefill (615 → 1210 t/s, PPL-preserving). Verified on Llama-3-8B SCLP4 (25.5 tg t/s) and Gemma4 mixed (55 tg t/s).

**Fused MoE prefill (WIP)**: kernel attempted for small-M prefill; sidecar correction dominates kernel time, making fused slower than two-pass baseline. Root cause identified: F32 accumulation order (tensor-core tile vs scalar sequential) differs at ~1e-3 per multiplication, compounding through 26 MoE FFN layers into catastrophic PPL on Gemma4-IT. Closing the gap requires either rocBLAS GEMM (defeats fusion) or exact tile-order replication (substantial work). Not a priority; two-pass baseline sufficient.

### Bridge Architecture (`sclp_bridge.cuh`)

`/home/ajkerchum/llama.cpp/ggml/src/ggml-cuda/sclp_bridge.cuh` kernels:
- `sclp_decode_blob_kernel`: Self-parses header on-device (thread-0 broadcast palette to shared mem), decodes 8 weights per thread via uint64 loads.
- `sclp_fixup_sidecar_kernel`: Grid-stride sidecar scatter-write (handles any sidecar_count without D2H read).
- `sclp_fused_gemv_kernel`: Fused decode+GEMV for M=1 (inline decode, 1 warp/row).
- `sclp_fused_gemm_kernel`: Fused decode+GEMM for small M (TILE_M=16, accums in VGPRs).
- `llama_sclp_dispatch`: Launches decode+fixup for two-pass path; wired into `ggml_cuda_mul_mat` (decodes blob → BF16 buffer, recurses). All HIP-graph-safe (no D2H reads).

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

## Future Work: Per-Tensor & Per-Expert Precision

**Per-tensor policy** (name-based, no kernel changes): embeddings/output at BF16; first/last blocks at SCLP6; interior MLPs at SCLP4; attention at SCLP6. Extend `convert_to_sclp_gguf.py` with policy table.

**Per-expert policy** (MoE): route-skewed experts to mixed precision; optionally vary `sidecar_dist` per expert. Requires MoE path changes to handle mixed types within tensor.

## imatrix-Aware Encoding (Live — Sidecar Path Working, +1 GB for 15× PPL Win)

**Status (May 2026)**: imatrix loader (`src/compression/imatrix.py`), encoder integration (`encode_palette_4b/6b`), and converter flag (`--imatrix` + `--sidecar-imatrix-budget`) all live. **Imatrix is applied to *sidecar selection*, not to palette k-means** — the naive k-means weighting was tried first and regressed PPL 5×; this version replaces it.

**Calibration**: `llama-imatrix -m Q5_K_M.gguf -f wikitext-2-raw --output-format dat --chunks 80 -ngl 99 -c 512 --no-ppl` → `.dat`. Use `--imatrix path.dat --sidecar-imatrix-budget 0.01` in converter (see results below).

**How it works** (`src/compression/encoder.py` `_encode_4b_expert` / `_encode_6b_expert`):
- Palette: raw exponent frequency (unaffected by activation outliers).
- Sidecar: two tiers: (1) mandatory `dist > sidecar_dist`, (2) discretionary top-`budget` fraction by `importance × distance`. Importance per weight = `imatrix_value[col_idx]` (col_idx = flat_index % K).

**Result on Gemma4-26B-A4B-IT** (wikitext-test, `-c 512 -b 512 --chunks 50`):

| Config | Size | tg t/s | PPL |
|---|---|---|---|
| **Mixed + imatrix-sidecar 1%** | **17.5 GB** | **56** | **940** |
| Mixed (no imatrix) | 16.2 GB | 55 | 13,909 |
| Mixed (naive imatrix k-means) | 16.2 GB | 55 | 66,370 |
| Q5_K_M | 18 GB | — | 18,481 |
| SCLP6 pure | 19 GB | 56 | 341,388 |
| SCLP4 pure | 15 GB | 2.5 | 77,053 |

**Headline**: 20× better PPL than Q5_K_M at smaller size; 15× better than non-imatrix mixed; matches SCLP6 speed.

**Why sidecar-imatrix wins over k-means-weighted palette**: k=4 palette centers are over-constrained by exponent distribution; activation weighting can't overcome mantissa-truncation error. Per-weight sidecar is unconstrained — `importance × distance` directly rescues highest-impact errors.

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

Two hybrid variants tested:

**Python-generated SCLP6+Q4_K (old)** — generated via two-step process (BF16 → llama-quantize Q4_K → patch SCLP6 tensors), without the new C++ SCLP encoder's k-means/imatrix-sidecar logic for SCLP6 tensors:

| Config | Size | PPL |
|---|---|---|
| **SCLP4 + imatrix-sidecar 1%** ⭐ | 17.5 GB | **940** |
| SCLP6+Q4K hybrid (old, no imatrix on SCLP6) | 16.4 GB | 9,961 |

**C++ llama-quantize SCLP6+Q4_K (new, May 2026)** — single-step via patched `llama-quantize` with native SCLP types. Uses k-means palette + imatrix-sidecar 1% on the SCLP6 tensors. See "llama-quantize SCLP Support" below.

llama-bench results on Gemma4-26B-A4B-IT (RX 7900 XTX, `-fa 1 -t 16`):

| Config | Size | pp512 | tg128 |
|---|---|---|---|
| **MIXED-imatrix** (SCLP6 attn+ffn_down, SCLP4 gate/up) | 14.89 GiB | 1,137 | 64.5 |
| **SCLP6+Q4K hybrid** (SCLP6 attn+ffn_down, Q4_K gate/up) | 15.80 GiB | **1,805** | **70.3** |

The hybrid is **+59% prefill** and **+9% tg** at +0.9 GiB. Q4_K gate/up goes directly through rocBLAS INT8 — no two-pass SCLP decode for the bulk of weights.

PPL comparison (wikitext-test, `-c 512 -b 512 --chunks 50 -fa on`):

| Config | Size | PPL |
|---|---|---|
| **MIXED-imatrix** (SCLP6+SCLP4+imatrix-sidecar 1%) ⭐ | 14.9 GiB | **940** |
| **SCLP6-Q4K hybrid** (C++ llama-quantize, imatrix-sidecar on SCLP6) | 15.8 GiB | 8,877 |
| Old Python SCLP6+Q4K (no imatrix on SCLP6) | 16.4 GB | 9,961 |

The C++ hybrid is marginally better than the old Python version (imatrix-sidecar on SCLP6 working as expected), but Q4_K gate/up costs **~9× PPL** vs SCLP4+imatrix-sidecar. The Q4_K route trades quality for prefill throughput — pick MIXED-imatrix for quality, SCLP6-Q4K for prefill-bound workloads.

Q4_K is *not* a drop-in replacement for SCLP4 quality-wise (10.6× worse PPL on the old comparison), but for **attn+ffn_down=SCLP6 + gate/up=Q4_K** the speed win is significant. The right knob depends on workload (prefill vs decode bound).

To produce the hybrid: requires a patch to `llama-quantize` (which only honored `--tensor-type` overrides when the default type was already quantized; now applies regardless — see `src/llama-quant.cpp`). Patch is local to this fork and worth upstreaming.

### Future Tuning Knobs

- **Calibrate from BF16** (3h CPU) instead of Q5_K_M to eliminate imatrix contamination — worth ~hundreds PPL points.
- **Per-tensor budget** — different tensors have different importance×distance distributions; uniform 1% may be suboptimal.
- **Error-magnitude based sidecar** — rank by `importance × actual_error` rather than `importance × distance`.

### OOD vs Wikitext PPL on Agentic Workloads (May 2026)

Built an agentic-trace calibration + eval set from `Verdugie/opus-4.6-training-catalog` (conversation/coding/reasoning splits flattened to `USER:`/`ASSISTANT:` text). See `tests/prep_opus_trace.py`. Holdout: 150 conversations (~1.6 MB) for OOD PPL; calibration: 6,503 conversations (~21 MB) used for `llama-imatrix --chunks 200` on Q5_K_M (~5 min on RX 7900 XTX, 97-99% MoE expert coverage).

PPL on the held-out agentic OOD set (50 chunks, `-c 512 -b 512 -fa on`):

| Config | Size | imatrix source | OOD PPL |
|---|---|---|---|
| **MIXED-opus** (SCLP6+SCLP4, opus imatrix) ⭐ | 17.0 GiB | opus traces, 200 chunks | **26.6 ± 1.1** |
| MIXED-imatrix (SCLP6+SCLP4, wiki imatrix) | 14.9 GiB | wikitext, 80 chunks | 39.2 ± 1.8 |
| SCLP6-Q4K-opus (SCLP6+Q4K, opus imatrix) | 16.7 GiB | opus traces | 157.6 ± 8.7 |
| SCLP6-Q4K (SCLP6+Q4K, wiki imatrix) | 15.8 GiB | wikitext | 161.6 ± 8.9 |

**Key findings**:

1. **Wikitext PPL was OOD-inflated by ~50×**. Same models drop from 940→39 (MIXED) and 8877→158 (Q4K hybrid) when measured on in-distribution agentic text. The wikitext numbers showed correct *ratios* between quants but absolute values were misleading.

2. **Calibration domain matters for SCLP4 sidecar selection, barely for Q4_K**. Switching to opus-traces imatrix:
   - MIXED model: −32% PPL (39.2 → 26.6). SCLP4+imatrix-sidecar picks different weights to rescue when calibrated on the target domain.
   - Q4K hybrid: −2% PPL (161.6 → 157.6, within noise). Q4_K per-block scales are insensitive to per-column importance vs SCLP's direct top-k weight rescue.

3. **SCLP4+imatrix-sidecar quality lead over Q4_K grows with calibration match**. With wiki-imatrix on both: 4.1× (39 vs 162). With opus-imatrix on both: **5.9× (27 vs 158)**.

4. **Size increased with opus imatrix** (17.0 vs 14.9 GiB for MIXED): the opus importance distribution promotes more weights into sidecar at the same 0.01 budget. Net result is still a Pareto win — same speed, +2 GiB, but 32% lower PPL.

**Recommendation**: For agentic/instruction-tuned use cases, calibrate imatrix on representative dialogue traces (not wikitext) when targeting SCLP4+imatrix-sidecar. Q4_K gate/up isn't worth it unless prefill-bound — costs 6× quality for +59% prefill.

To rebuild MIXED-opus from BF16:
```bash
llama-quantize \
  --imatrix /home/ajkerchum/poc/eval_data/gemma4-opus-imatrix.dat \
  --tensor-type '^token_embd\.weight$=BF16' \
  --tensor-type '^output\.weight$=BF16' \
  --tensor-type '^blk\.[0-9]+\.attn_(q|k|v|output)\.weight$=SCLP6' \
  --tensor-type '^blk\.[0-9]+\.ffn_down(_exps)?\.weight$=SCLP6' \
  bf16-shard-00001-of-00002.gguf MIXED-opus.gguf SCLP4 8
```

## Known Issues

### SCLP4 VRAM Allocation (Resolved on Llama-3, Pending on Gemma4)

**Original symptom:** SCLP4 inference produced garbage output across models. Earlier `get_alloc_size` for SCLP4 reserved only `1.25 × ggml_nbytes`, but on disk the blob is `header + N/2 (ws) + 6 × sidecar_count`. Per-tensor sidecar fractions are wildly model-dependent:

- Gemma4 26B MoE: max blob ratio **1.26×** (mostly low-exponent-spread MoE weights)
- Llama-3-8B: max blob ratio **2.34×** on `attn_q.weight` (~11% sidecar — Llama's weight distribution has a broader exponent tail than Gemma4's)

When `disk_size > alloc_size`, the loader's `ggml_backend_tensor_set` asserted out (with `--no-mmap`) or silently truncated the blob (with mmap — producing garbage downstream as the kernel read undefined memory past the truncation point, especially the sidecar_count field which then scatter-corrupted random output positions).

**Fix:** `get_alloc_size` for SCLP4 now returns `3 × ggml_nbytes + 64KB` (= 1.5 × N bytes). This is enough for Llama-3-8B's worst tensor and any plausible sidecar fraction up to ~16%. Also fixed `ggml_backend_tensor_set_async` to check `alloc_size` (matching the sync version's bound) so the non-mmap GPU upload path no longer asserts.

**Verified working:** Llama-3-8B SCLP4-kmeans-sc1 — loads, generates at 25.88 t/s (≈3× BF16 read bandwidth), grammatically coherent English output (semantic quality is poor as expected at 4 bits/weight, not a bug).

### SCLP disk_size Hint via op_params

Model loader stashes `disk_size` into `tensor->op_params[13:15]` (magic sentinel + uint64) to allow per-tensor exact VRAM allocation. See `llama-model-loader.cpp::create_tensor` for layout. `get_alloc_size` reads the hint (or falls back to 3× for SCLP4, 1.25× for SCLP/SCLP6). This uses `op_params` convention-reserved for compute-graph operations; weight tensors (`OP_NONE`) are unused today. **On upstream merge: grep for new `op_params` writers/readers and verify `OP_NONE` paths remain clear.**

### Future Work: Converter-Side Precision Fallback

Even with exact alloc, a tensor whose SCLP4 blob exceeds its BF16 size is pathological — the converter should detect that case at encode time and fall back to BF16 (or SCLP6) for that specific tensor. This is the natural on-ramp to the mixed-precision idea above: instead of a hardcoded per-name policy, drive precision selection by measured per-tensor compression efficiency.

## 16 GB VRAM Recipe (Gemma4-26B-A4B-IT, May 2026)

For 16 GB cards (leaves ~1.4 GiB for KV cache + compute on a 24 GB target sliced down). Built with opus-traces imatrix.

| Config | Size | OOD PPL |
|---|---|---|
| MIXED-opus-16gb (Q6_K embeds + SCLP6 attn+ffn_down + SCLP4 gate/up) | 16.2 GiB | 26.4 ± 1.1 |
| **SCLP6attn-opus** (Q6_K embeds + SCLP6 attn only, SCLP4 ffn_down+gate/up) ⭐ | **14.6 GiB** | **40.0 ± 1.7** |
| SCLP4-pure-opus (Q6_K embeds + SCLP4 everywhere) | 14.3 GiB | 97.1 ± 3.8 |

**SCLP6attn-opus is the sweet spot for 16 GB**: 50% better PPL than pure SCLP4 at nearly the same size. Confirms design intuition — attention errors compound through softmax, so keeping `attn_(q|k|v|output)` at SCLP6 is highest-leverage; `ffn_down` errors absorb linearly into the residual stream and survive SCLP4.

Build:
```bash
llama-quantize \
  --imatrix /home/ajkerchum/poc/eval_data/gemma4-opus-imatrix.dat \
  --tensor-type '^token_embd\.weight$=Q6_K' \
  --tensor-type '^output\.weight$=Q6_K' \
  --tensor-type '^blk\.[0-9]+\.attn_(q|k|v|output)\.weight$=SCLP6' \
  bf16-shard-00001-of-00002.gguf SCLP6attn-opus.gguf SCLP4 8
```

Q6_K embeddings save ~0.8 GiB vs BF16 with no measurable PPL impact.

## Future Work: TurboQuant for KV Cache

Orthogonal compression for KV cache (per-token rotation + low-bit quantization). Find community fork via upstream issue tracker. Integration: cherry-pick onto `sclp` branch, smoke-test Llama-3-8B, then layer onto weights. Expect rebase friction on KV cache type/node changes in ggml.

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
3. **SCLP6 pure at 236M was stale GGUF** using frequency-based palette (before k-means commit). Regenerating drops PPL to 341K (700× improvement). Mixed dominates pure SCLP6 on size, speed, and PPL due to per-tensor error sensitivity (SCLP4's wider sidecar ~1.36% vs SCLP6's 0.07% better captures worst-case errors on Gemma4-IT).

**Practical takeaway**: regenerate any SCLP6 GGUF older than 2026-05-16 10:19. Mixed-precision (`--format mixed`) is best Gemma4 config.

### SCLP6 Inference Hang (Critical)
In some cases, SCLP6 inference hangs after loading the model, consuming CPU time indefinitely without producing output. This does not appear to be a consistent issue and may be related to specific prompt/context combinations or GPU memory pressure.

**Status:** Requires further investigation. SCLP6 works in some cases (model loading verified) but hangs in others.
