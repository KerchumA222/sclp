# SCLP: Soft-Clipping Lossless-First Compression for LLM Weights

SCLP is a lossless-first compression scheme for BF16 neural network weights that achieves **2× compression** per stream (8 bits/weight) with quality that matches or exceeds BF16 on large models. Three precision tiers (SCLP8/6/4) and mixed-precision policies enable fine-grained size–quality trade-offs, with imatrix-aware sidecar selection for instruction-tuned and MoE models.

## Key Results

### Llama-3-8B (SCLP8, RX 7900 XTX)

| Metric | Value |
|---|---|
| Compression ratio (SCLP streams) | **2.0×** (8 bits/weight vs 16) |
| File size (compact GGUF) | **8.47 GB** (vs 14.97 GB BF16) |
| PPL (wikitext-2) | **9.87** (vs 10.59 BF16) |
| Generation speed | **~66 t/s** (vs ~52 t/s BF16) |
| Prefill speed | **~2,730 t/s** (vs ~12,000 t/s BF16) |

### Gemma4-26B-A4B-IT (Mixed Precision, RX 7900 XTX)

| Config | Size | tg128 | OOD PPL |
|---|---|---|---|
| MIXED-opus (SCLP6 attn+ffn_down, SCLP4 gate/up, imatrix-sidecar 1%) | 17.0 GiB | 55 t/s | **26.6** |
| SCLP6attn-opus (16 GB recipe: SCLP6 attn, SCLP4 rest) | 14.6 GiB | ~55 t/s | 40.0 |
| Q5_K_M | 18 GB | — | 18,481 (wikitext) |

## How It Works

BF16 weights have an 8-bit exponent field. In practice, any given weight matrix uses only **10–16 distinct exponent values** covering 99.9%+ of its weights. SCLP exploits this:

1. **Palette**: Record the top ≤16 exponent values (k-means by default; frequency available).
2. **Interleaved stream (ws_stream)**: Store each weight in a single byte:
   - **High nibble (4 bits)**: Index into the exponent palette.
   - **Low nibble (4 bits)**: Sign bit + Top 3 mantissa bits (SMN).
3. **Sidecar**: Store rare outlier weights (0.01–3%) verbatim as BF16. Optionally expanded via imatrix-aware selection for instruction-tuned models.

Decoding is: `reconstruct(palette[index], smn_nibble)` — a single lookup + bitwise combine per weight. This maps directly to a fused decode-GEMV GPU kernel with negligible overhead.

### Precision Tiers

| Type | Bits/weight | Palette bits | Mantissa bits | Use case |
|---|---|---|---|---|
| **SCLP8** | 8 | 4 (16 entries) | 3 | Default, near-lossless |
| **SCLP6** | 6 | 4 (16 entries) | 1 | Attention layers, quality-sensitive |
| **SCLP4** | 4 | 4 (16 entries) | 0 (sign only) | MLP gate/up, size-optimized |

Mixed-precision policies assign different tiers per tensor type (e.g., SCLP6 for attention, SCLP4 for gate/up projections).

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

- `GGML_TYPE_SCLP = 42`, `GGML_TYPE_SCLP4`, `GGML_TYPE_SCLP6` registered in `ggml.h` / `ggml.c`
- `llama-quantize` natively supports `SCLP`, `SCLP4`, `SCLP6` as quantization types and `--tensor-type` overrides
- On-device decode via `sclp_bridge.cuh` — self-parses blob header in shared memory, safe during HIP graph capture
- Fused decode-GEMV kernel for single-token inference (~66 t/s vs ~52 t/s BF16 on RX 7900 XTX)
- Fused MoE GEMV kernels for SCLP4/SCLP6 (decode only routed experts inline, no full buffer materialization)
- GGUF loader supports compact blob format via `disk_size` field

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

**Option A: llama-quantize (recommended for mixed-precision)**
```bash
# Build llama.cpp first (see below), then:
llama-quantize \
    --imatrix /path/to/imatrix.dat \
    --tensor-type '^token_embd\.weight$=BF16' \
    --tensor-type '^output\.weight$=BF16' \
    --tensor-type '^blk\.[0-9]+\.attn_(q|k|v|output)\.weight$=SCLP6' \
    --tensor-type '^blk\.[0-9]+\.ffn_down(_exps)?\.weight$=SCLP6' \
    bf16-input.gguf output.gguf SCLP4 8
```

**Option B: Python converter**
```bash
source eval_env/bin/activate
python3 tests/convert_to_sclp_gguf.py \
    --input  /path/to/bf16-model.gguf \
    --output /path/to/model-SCLP.gguf \
    --format mixed
```

### Build llama.cpp and run inference

```bash
cd ../llama.cpp
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100
cmake --build build --config Release -j$(nproc)

build/bin/llama-completion \
    -m /path/to/model-SCLP.gguf \
    -ngl 99 -n 100 -no-cnv --repeat-penalty 1.3 \
    -p "The capital of France is"
```

## GGUF Blob Format

All SCLP types (8/4/6) share the same per-expert blob header:

```
[uint32 num_weights]
[uint32 n_experts]
[per-expert: uint8 palette_size, uint8 × palette_size palette] ...
[uint8  × num_weights   ws_stream]        ← palette_idx(7:4) | smn(3:0)
[uint32 sidecar_count]
[uint32 × K             sidecar_indices]  ← positions of outlier weights
[uint16 × K             sidecar_values]   ← full BF16 bits verbatim
```

Each blob is stored at its actual compressed size, padded only to 32-byte GGUF alignment. The loader infers `disk_size` from consecutive tensor offsets.

See `CLAUDE.md` for the full implementation guide including the llama.cpp loader changes.
