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
