# SCLP: Soft-Clipping Lossless-First Compression for LLM Weights

SCLP is a lossless-first compression scheme for BF16 neural network weights that achieves a consistent **1.333× compression ratio** on compressed streams with near-zero quality loss on large models.

## Key Results (Llama-3-8B)

| Metric | Value |
|---|---|
| Compression ratio (SCLP streams) | **1.333×** (12 bits/weight vs 16) |
| Model-wide compression (87% of params) | **1.278×** |
| File size (padded GGUF) | 14.97 GB |
| File size (compact GGUF) | **11.72 GB** (−22%) |
| PPL delta vs BF16 baseline | **−0.09%** (within noise floor) |
| Inference speed, fused decode-GEMV | **~49 t/s** vs ~52 t/s FP16 (94% of baseline) |

## How It Works

BF16 weights have an 8-bit exponent field. In practice, any given weight matrix uses only **10–16 distinct exponent values** covering 99.9%+ of its weights. SCLP exploits this:

1. **Palette**: Record the top ≤16 exponent values by frequency
2. **Pack indices**: Store each weight's palette index as a 4-bit nibble (2 per byte)
3. **SM stream**: Store sign + full 7-bit mantissa (1 byte/weight, lossless)
4. **Sidecar**: Store the rare outlier weights (0.01–0.03%) verbatim as BF16

Decoding is: `reconstruct(palette[index], sm_byte)` — a single lookup + bitwise combine per weight. This maps directly to a fused decode-GEMV GPU kernel with negligible overhead.

### Compression Ratio

- Input: 16 bits/weight (BF16)
- Palette index: 4 bits/weight
- SM stream: 8 bits/weight
- **Total: 12 bits/weight = 1.333×**

## Repository Layout

```
src/compression/      Python reference implementation
  clipping.py         Stochastic exponent clipping
  encoder.py          Palette builder + nibble packing
  decoder.py          Vectorised BF16 reconstruction
  storage.py          .sclp binary file format
  pipeline.py         High-level compress/decompress API

src/hip/              HIP/ROCm GPU kernels
  clipping.hip        Soft-clip kernel
  encoder.hip         Palette encode kernel
  decoder.hip         Palette decode kernel
  launcher.hip        C-interface launchers
  wrapper.cpp         pybind11 Python bindings

tests/
  convert_to_sclp_gguf.py    HuggingFace → SCLP GGUF converter
  repack_sclp_gguf.py        Strip zero-padding → compact GGUF
  patch_gguf_sclp.py         Patch individual tensors in an existing GGUF
  analyze_sidecar_cost.py    Rank tensors by sidecar drop cost
  convert_selective_sidecar.py  Selective sidecar removal tool
  test_pipeline.py           Core correctness tests
  test_hip_module.py         HIP kernel tests (requires ROCm)

design.md             Algorithm design rationale
experimental_results.md  All benchmark results
CLAUDE.md             Full implementation guide (llama.cpp integration, wire formats)
```

## llama.cpp Integration

SCLP-compressed GGUFs can be run directly with our fork of llama.cpp:
**[github.com/KerchumA222/llama.cpp](https://github.com/KerchumA222/llama.cpp) — branch `sclp`**

Changes on top of upstream llama.cpp:

- `GGML_TYPE_SCLP = 42` registered in `ggml.h` / `ggml.c`
- `GGMLQuantizationType.SCLP = 42` in `gguf-py`
- On-device decode via `sclp_bridge.cuh` — self-parses blob header in shared memory, safe during HIP graph capture
- Fused decode-GEMV kernel for single-token (M=1) inference path (~49 t/s vs ~52 t/s FP16 on RX 7900 XTX)
- GGUF loader supports both padded and compact blob formats via `disk_size` field

## Quick Start

### Prerequisites

- Python 3.10+, numpy, torch, transformers
- `gguf-py` from the llama.cpp fork (set `LLAMA_CPP_DIR` env var, or clone sibling to this repo as `../llama.cpp`)
- ROCm (optional, for HIP kernels and llama.cpp inference)

### Setup

```bash
# Clone this repo
git clone https://github.com/KerchumA222/sclp sclp-research
cd sclp-research

# Clone the llama.cpp fork (sclp branch) as a sibling directory
git clone -b sclp https://github.com/KerchumA222/llama.cpp llama.cpp

# Create a venv and install Python dependencies
python3 -m venv eval_env
source eval_env/bin/activate
pip install numpy torch transformers
```

### Run Python tests (no GPU needed)

```bash
source eval_env/bin/activate
python3 -m pytest tests/test_pipeline.py tests/test_distributions.py -v
```

### Build the HIP kernel module (requires ROCm)

```bash
cd src/hip && mkdir -p build && cd build
cmake .. && make -j$(nproc) --no-print-directory
# Produces python_pkg/testmodule.so
```

### Convert a model

```bash
source eval_env/bin/activate

# Convert HuggingFace BF16 checkpoint → padded SCLP GGUF
python3 tests/convert_to_sclp_gguf.py \
    --input  /path/to/Meta-Llama-3-8B.fp16.gguf \
    --output /path/to/Llama-3-8B-SCLP.gguf

# Repack padded → compact (saves ~22% on Llama-3-8B)
python3 tests/repack_sclp_gguf.py \
    --input  /path/to/Llama-3-8B-SCLP.gguf \
    --output /path/to/Llama-3-8B-SCLP-Compact.gguf
```

### Build llama.cpp and run inference

```bash
cd ../llama.cpp
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100
cmake --build build --config Release -j$(nproc)

build/bin/llama-completion \
    -m /path/to/Llama-3-8B-SCLP-Compact.gguf \
    -ngl 99 -n 100 -no-cnv --repeat-penalty 1.3 \
    -p "The capital of France is"
```

## GGUF Blob Format

```
[uint32 num_weights]
[uint8  palette_size]
[uint8  × palette_size  palette]
[uint8  × ceil(N/2)     packed_indices]   ← 4 bits/weight, nibble-packed
[uint8  × N             sm_stream]        ← sign(7) | mantissa(6:0)
[uint32 sidecar_count]
[uint32 × K             sidecar_indices]  ← positions of outlier weights
[uint16 × K             sidecar_values]   ← full BF16 bits verbatim
```

**Padded format**: zero-padded to `num_weights × 2` bytes (same as BF16 allocation).  
**Compact format**: stored at actual blob end, padded only to 32-byte GGUF alignment.

See `CLAUDE.md` for the full implementation guide including the llama.cpp loader changes.
