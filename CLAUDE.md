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
| `encoder.py` | `encode_palette()` — builds exponent palette + 4-bit indices + SM stream |
| `decoder.py` | `decode_palette()` — reconstructs BF16 weights from palette + SM stream |
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
- `packed_indices`: nibble-packed, 2 weights per byte — `(idx_even << 4) | idx_odd`
- `sm_stream`: one byte per weight — `sign(7) | mantissa(6:0)` — full 7 bits, lossless
- Compression ratio: 12 bits/weight vs 16 bits original = **1.333×** on the compressed streams

### Encoded Data Structure (dict returned by `encode_palette` and `testmodule.encode`)
```python
{
  'palette':        np.uint8[<=16],        # exponent values sorted by frequency (descending)
  'packed_indices': np.uint8[ceil(N/2)],   # nibble-packed: high nibble = even weight
  'sm_stream':      np.uint8[N],           # sign(7) | mantissa(6:0) per weight
  'num_weights':    int,
  'sidecar': {                             # weights whose exponent is not in the palette
    'indices': np.uint32[K],               # positions in the weight array
    'values':  np.uint16[K],              # full BF16 bits stored verbatim
  }
}
```

`testmodule.encode` returns the same fields with keys `packed`, `sm`, `sidecar_indices`, `sidecar_values`.

### HIP GPU Implementation (`src/hip/`)

| File | Status |
|---|---|
| `clipping.hip` | Complete — `soft_exponent_clip_kernel` |
| `encoder.hip` | Complete — `encode_palette_kernel` (nibble-packed indices + SM stream) |
| `decoder.hip` | Complete — `decode_palette_kernel` (reconstructs BF16 from palette + SM) |
| `launcher.hip` | C-interface launchers exported via `extern "C"` |
| `wrapper.cpp` | pybind11 bindings — exposes `clip`, `encode`, `decode` |
| `CMakeLists.txt` | Builds `hip_kernels` static lib + `testmodule` shared lib; strips LTO flags |

The compiled Python module (`testmodule`) lives in `python_pkg/` and is imported as `import testmodule`.

### Kernel Launcher Signatures
```cpp
launch_clip_kernel(const uint16_t* input, uint16_t* output, uint n, uint8_t threshold, uint32_t seed, uint8_t mantissa_mask);
launch_encode_kernel(const uint16_t* input, const uint8_t* lookup, uint8_t* packed, uint8_t* sm, uint32_t n);
launch_decode_kernel(const uint8_t* packed, const uint8_t* sm, const uint8_t* palette, uint16_t* output, uint32_t n);
```

### File Format (`.sclp`) — VERSION 2
Binary, little-endian:
`SCLP` magic (4B) | version uint16 | num_weights uint32 | palette_size uint8 | palette bytes | indices_len uint32 | packed_indices bytes | sm_len uint32 | sm_stream bytes | sidecar_count uint32 | sidecar_indices uint32[] | sidecar_values uint16[]
VERSION 1 files (no sidecar) load correctly with an empty sidecar.

## Key Implementation Notes

- All weights are passed as `np.uint16` representing raw BF16 bit patterns, never as floats
- Python and HIP encoders produce identical wire format; Python decoder is fully vectorised (no Python loop)
- Clipping: exponents `> threshold+1` are hard-clipped to `threshold`; exponents `== threshold+1` survive with 50% probability (flat, matches HIP XorshiftPRNG)
- Sidecar: weights with exponents outside the top-16 palette are stored verbatim; decoder restores them exactly. Without sidecar, rare low-exponent weights would be inflated to the nearest palette exponent — catastrophic for quality.
- `testmodule.encode(input, palette)` accepts the palette array (≤16 uint8 exponent values); the wrapper builds the nearest-neighbour lookup internally
- `testmodule.decode(packed, sm, palette, sidecar_indices=[], sidecar_values=[])` — sidecar args are optional
- The `mantissa_mask` parameter of `clip` is applied as `output = weight & (0xFF80 | mantissa_mask)`; pass `0x7F` to preserve all mantissa bits

## llama.cpp Integration

The llama.cpp integration lives on the `sclp` branch at `/home/ajkerchum/llama.cpp`.

### Build (llama.cpp sclp branch)

```bash
cd /home/ajkerchum/llama.cpp
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100
cmake --build build --config Release -j$(nproc)
```

### GGUF Blob Wire Format

The SCLP payload stored inside a GGUF tensor slot is **not** the same as the standalone `.sclp` file format — no magic bytes, no version field. It is a minimal self-describing blob sized to exactly `num_weights * 2` bytes (identical to BF16), so all GGUF tensor offsets and strides are unchanged:

```
[uint32 num_weights][uint8 palette_size][palette (palette_size bytes)]
[packed_indices (ceil(num_weights/2) bytes)][sm_stream (num_weights bytes)]
[uint32 sidecar_count]
[uint32 × sidecar_count sidecar_indices][uint16 × sidecar_count sidecar_values]
[zero padding to fill num_weights*2 bytes total]
```

Weights whose exponents fall outside the top-16 palette are stored verbatim in the sidecar section and restored exactly by `sclp_fixup_sidecar_kernel` after the main decode. The sidecar is typically 0.01–0.03% of all weights (lossless). The GGUF blob is sized to `num_weights * 2` regardless of sidecar count — the sidecar data fits within the natural padding between the compressed streams and the allocation boundary.

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

### Running Inference

```bash
/home/ajkerchum/llama.cpp/build/bin/llama-completion \
    -m /home/ajkerchum/poc/models/llama3/Llama-3-8B-SCLP-Patched.gguf \
    -ngl 99 \
    -n 100 \
    -no-cnv \
    --repeat-penalty 1.3 \
    -p "The capital of France is"
```

Use `llama-completion`, not `llama-cli`. `llama-cli` does not support `-no-cnv`. Both binaries auto-enable conversation mode when the GGUF contains an embedded chat template (Llama 3 does) — `-no-cnv` forces raw completion. `--repeat-penalty 1.3` prevents repetition loops common in base model completion. Verified working on RX 7900 XTX (gfx1100) at ~16 t/s compressed / ~52 t/s FP16 baseline. The 3x gap is transcoder bridge overhead (extra decode + memory pass per matmul); eliminated by the fused decode-GEMM path.

### Bridge Architecture (`sclp_bridge.cuh`)

`/home/ajkerchum/llama.cpp/ggml/src/ggml-cuda/sclp_bridge.cuh` implements the on-device decode path:

- `sclp_decode_blob_kernel`: Self-parses the GGUF blob header on-device. Thread 0 reads `blob[4]` (palette_size) into `__shared__` memory, then all threads load the palette into `__shared__ uint8_t s_palette[16]`. Derives `packed` and `sm` pointer offsets from palette_size. No host-side device reads, safe during HIP stream capture.
- `sclp_fixup_sidecar_kernel`: Reads `sidecar_count` from the blob on-device (after sm_stream), then each thread restores one outlier weight via scatter write into the output buffer. Uses a grid-stride loop with 4 fixed blocks so it handles any sidecar count without a D2H read. Threads where `i >= sidecar_count` return after one shared-memory read.
- `llama_sclp_dispatch`: Launches both kernels sequentially on the same HIP stream. Decode grid is proportional to `num_weights`; fixup grid is fixed at 4 blocks.

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
