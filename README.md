# SCLP: Soft-Clipping Lossless-First Compression for LLM Weights

SCLP is an exponent-palette compression scheme for BF16 neural network weights. Four precision tiers (SCLP8/6/5/4) and mixed-precision policies enable fine-grained size-quality trade-offs, with imatrix-aware sidecar selection for instruction-tuned and MoE models.

This repository contains the **reference implementation** (Python + HIP kernels). Production inference runs through our [llama.cpp fork](https://github.com/KerchumA222/llama.cpp) (branch `sclp`).

## Key Results

### Llama-3-8B (dense, RX 7900 XTX)

| Config | Size | tg128 (t/s) | pp512 (t/s) |
|--------|------|-------------|-------------|
| BF16 | 14.97 GB | ~52 | ~12,000 |
| SCLP8 | 7.92 GB | 43 | ~2,650 |
| SCLP6 | 6.71 GB | 38 | ~2,780 |
| SCLP5 | 5.72 GB | 40 | ~2,650 |
| SCLP4 | 4.86 GB | 34 | ~2,640 |

### Gemma4-26B-A4B-IT (MoE, RX 7900 XTX)

OOD perplexity on held-out agentic traces (opus-trace, 50 chunks):

| Config | Size | tg (t/s) | OOD PPL |
|--------|------|----------|---------|
| MIXED-opus (SCLP6 attn+down, SCLP4 gate/up, imatrix) | 17.0 GiB | 55 | **26.6** |
| SCLP6attn-opus (16 GB recipe) | 14.6 GiB | ~55 | 40.0 |
| MIXED-bpal (per-block palette SCLP4) | 14.9 GiB | 55 | 132.6 |

## How It Works

BF16 weights have an 8-bit exponent field. In practice, any given weight matrix uses only 10-16 distinct exponent values covering 99.9%+ of its weights. SCLP exploits this:

1. **Palette**: K-means clustering selects the top exponent values (16 for SCLP8, 8 for SCLP6, 4 per block for SCLP4/5).
2. **Packed stream**: Each weight is encoded as `palette_index | sign | mantissa_top_bits` at the type's bit width.
3. **Sidecar**: Outlier weights (0.01-3%) are stored verbatim as BF16. Optionally expanded via imatrix-aware selection.

Decoding is a single lookup + bitwise combine per weight, mapping directly to a fused decode-GEMV GPU kernel.

### Precision Tiers

| Type | Bits/weight | Palette | Mantissa bits | Notes |
|------|-------------|---------|---------------|-------|
| **SCLP8** | 8 | 16 entries, global | 3 | Near-lossless, per-block BF16 scaling |
| **SCLP6** | 6 | 8 entries, global | 2 | Attention layers, per-block BF16 scaling |
| **SCLP5** | 5 | 4 entries, per-block | 2 | Pareto-dominated by SCLP4+imatrix |
| **SCLP4** | 4 | 4 entries, per-block | 1 | Bulk MLP weights, best compression |

SCLP6/8 use per-block scaling (a BF16 scale per 32 weights). SCLP4/5 use per-block palettes (4 exponents per 256 weights) instead — per-block scaling degenerates at k=4 (concentrates all exponents near 126-127).

## Repository Layout

```
src/compression/      Python reference implementation
  clipping.py         Stochastic exponent clipping
  encoder.py          Palette builder + nibble packing (SCLP4/6/8)
  decoder.py          Vectorised BF16 reconstruction
  storage.py          .sclp binary file format
  pipeline.py         High-level compress/decompress API
  imatrix.py          Importance matrix loader for sidecar selection

src/hip/              HIP/ROCm GPU kernels
  clipping.hip        Soft-clip kernel
  encoder.hip         Palette encode kernel
  decoder.hip         Palette decode kernel
  launcher.hip        C-interface launchers
  wrapper.cpp         pybind11 Python bindings

tests/
  convert_to_sclp_gguf.py       HuggingFace -> SCLP GGUF converter
  patch_gguf_sclp.py            Patch individual tensors in existing GGUF
  prep_opus_trace.py            Build agentic-trace calibration/eval sets
  test_pipeline.py              Core correctness tests
  test_hip_module.py            HIP kernel tests (requires ROCm)

design.md                       Algorithm design rationale
experimental_results.md         Benchmark results and analysis
plans/sclp4_vs_q4k_improvement.md  SCLP4 quality gap analysis
```

## llama.cpp Integration

SCLP-compressed GGUFs run directly with our [llama.cpp fork](https://github.com/KerchumA222/llama.cpp):

| Branch | Contents |
|--------|----------|
| `sclp` | SCLP types + kernels on top of upstream llama.cpp |
| `sclp-turboquant` | SCLP merged with [TurboQuant](https://github.com/TheTom/llama-cpp-turboquant) KV cache compression |

Key integration points:

- `GGML_TYPE_SCLP8 = 47`, `SCLP6 = 48`, `SCLP4 = 49`, `SCLP5 = 50` registered in ggml
- `llama-quantize` natively supports `SCLP4`, `SCLP5`, `SCLP6`, `SCLP8` as quantization types and `--tensor-type` overrides
- Per-type GPU kernels in `ggml/src/ggml-cuda/sclp_bridge_sclp{4,5,6,8}.cu`: two-pass decode, fused GEMV (M=1), fused MoE GEMV, MoE WMMA prefill
- Folded sidecar correction in fused GEMV (sorted sidecar, binary search per row, no atomics)
- K-tiled shared memory for high occupancy (4 blocks/CU on SCLP4/5/6)
- Compact GGUF storage — loader infers `disk_size` from tensor offsets

See [docs/sclp.md](https://github.com/KerchumA222/llama.cpp/blob/sclp/docs/sclp.md) in the llama.cpp fork for the full inference guide.

## Quick Start

### Prerequisites

- Python 3.10+, numpy, torch, transformers
- ROCm (optional, for HIP kernels and llama.cpp inference)

### Setup

```bash
git clone https://github.com/KerchumA222/sclp
cd sclp

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

### Quantize and run a model

```bash
# Clone and build the llama.cpp fork
git clone -b sclp https://github.com/KerchumA222/llama.cpp
cd llama.cpp
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100
cmake --build build --config Release -j$(nproc)

# Quantize (mixed precision, recommended)
build/bin/llama-quantize \
    --imatrix /path/to/imatrix.dat \
    --tensor-type '^token_embd\.weight$=BF16' \
    --tensor-type '^output\.weight$=BF16' \
    --tensor-type '^blk\.[0-9]+\.attn_(q|k|v|output)\.weight$=SCLP6' \
    --tensor-type '^blk\.[0-9]+\.ffn_down(_exps)?\.weight$=SCLP6' \
    bf16-input.gguf output.gguf SCLP4 8

# Run inference
build/bin/llama-completion \
    -m output.gguf -ngl 99 -n 200 -no-cnv --repeat-penalty 1.3 \
    -p "The capital of France is"
```

## GGUF Blob Format

SCLP6/SCLP8 (per-block scaling):
```
[uint32 num_weights][uint32 n_experts]
[per-expert: uint8 palette_size, uint8[] palette] ...
[BF16 scales: ceil(N/32) per expert]
[ws_stream]
[uint32 sidecar_count][uint32[] indices][uint16[] values]
```

SCLP4/SCLP5 (per-block palette):
```
[uint32 num_weights][uint32 n_experts]
[per-expert: uint8 palette_size=0] ...
[block_palettes: 4 x uint8 per 256-weight block]
[ws_stream]
[uint32 sidecar_count][uint32[] indices][uint16[] values]
```

Each blob is stored at its actual compressed size, padded only to 32-byte GGUF alignment.
