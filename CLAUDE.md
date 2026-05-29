# CLAUDE.md

Guidance for Claude Code working in this repository.

## Project Overview

A **weight compression PoC** implementing SCLP (Soft Clipping Lossless-First) compression for BF16 neural-network weights:
1. Clips rare large exponents stochastically (soft clipping)
2. Encodes exponents as 4-bit palette indices (≤16 unique exponents)
3. Stores sign + 3-bit mantissa per weight in a packed stream

See `design.md` for full rationale. Two implementations: **Pure Python/NumPy** (`src/compression/`, reference) and **HIP GPU kernels** (`src/hip/`, requires ROCm). The production path is the **llama.cpp integration** on the `sclp` branch of the fork, checked out alongside this repo at `$LLAMA_CPP` (defaults to `../llama.cpp`).

## Build & Test

```bash
# HIP module (compiled .so → python_pkg/, imported as `import testmodule`). Requires ROCm.
cd src/hip && mkdir -p build && cd build && cmake .. && make -j$(nproc) --no-print-directory
rocminfo | grep -i "amdgpu"          # verify ROCm hardware before HIP tests

# llama.cpp sclp branch
cd "$LLAMA_CPP"   # ../llama.cpp by default
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100
cmake --build build --config Release -j$(nproc)

# Python tests (pytest not global — use the venv)
source eval_env/bin/activate
python3 -m pytest tests/                    # all
python3 -m pytest tests/test_pipeline.py -v # single file
python3 tests/test_hip_module.py            # HIP module (requires ROCm)
```

## Architecture (Python reference, `src/compression/`)

| File | Role |
|---|---|
| `clipping.py` | `soft_exponent_clip()` — stochastic exponent clipping on BF16 uint16 arrays |
| `encoder.py` | `encode_palette()` — builds exponent palette + interleaved ws_stream |
| `decoder.py` | `decode_palette()` — reconstructs BF16 from palette + ws_stream |
| `storage.py` | `CompressedTensorStorage` — `.sclp` file format (magic `SCLP`) |
| `pipeline.py` | `SCLPCompressor` — high-level compress/decompress/save/load |
| `imatrix.py` | imatrix loader for activation-aware sidecar selection |

HIP kernels in `src/hip/` (`clipping.hip`, `encoder.hip`, `decoder.hip`, `launcher.hip` extern-C launchers, `wrapper.cpp` pybind11). All complete; require ROCm to run.

**Scope:** SCLP applies to all linear projections (MLP gate/up/down, attention q/k/v/out) — ~90% of params. Embeddings, output, LayerNorm excluded (kept native).

### BF16 bit layout (1-8-7) & wire format
- Bit 15 sign · bits 14-7 exponent (8b) · bits 6-0 mantissa (7b).
- All weights passed as raw `np.uint16` BF16 bit patterns, never floats.
- `ws_stream`: one byte per weight = `palette_idx(7:4) | smn(3:0)`, where `smn = sign(3) | mantissa_top3(2:0)`. 8 bits/weight = **2×** vs BF16.
- Bottom 4 mantissa bits zeroed — acts as mild regularization and empirically *lowers* PPL vs BF16. Co-locating index+SM in one byte halves L2 pressure vs separate streams.

`encode_palette` returns `{palette: uint8[≤16], ws_stream: uint8[N], num_weights, sidecar: {indices: uint32[K], values: uint16[K]}}`. Sidecar = weights whose exponent is outside the palette, stored as verbatim BF16 and restored exactly (without it, rare exponents map to nearest palette entry — catastrophic). Decoder is fully vectorised.

**Palette selection:** `encode_palette_4b/6b` default to k-means (1-D, 20 iter, k-means++ init) — protects rare low-exponent weights that frequency selection maps catastrophically (matters for MoE routing). `palette_method='frequency'` available but 8× worse worst-case MaxRel. `encode_palette` (SCLP8) uses frequency + sidecar.

**HIP module API:** `testmodule.encode(input, palette)` → `packed, sm, sidecar_indices, sidecar_values` (separate streams internally; GGUF bridge uses ws_stream). `testmodule.decode(packed, sm, palette, sidecar_indices=[], sidecar_values=[], num_weights_hint=-1)`. `clip` mantissa_mask applied as `weight & (0xFF80 | mantissa_mask)` — pass `0x7F` to keep all mantissa bits.

## llama.cpp Integration

### GGUF types & registration (verified against fork)

| Type | enum | blck_size | type_size |
|---|---|---|---|
| `GGML_TYPE_SCLP8` (8-bit, **current standard**) | 47 | 1 | 1 |
| `GGML_TYPE_SCLP6` (6-bit) | 48 | 4 | 3 |
| `GGML_TYPE_SCLP4` (4-bit) | 49 | 2 | 1 |
| `GGML_TYPE_SCLP5` (5-bit) | 50 | 8 | 5 |

`GGML_TYPE_COUNT = 51`. Registered in `ggml/include/ggml.h`, `ggml/src/ggml.c` (`.type_name="sclp8"` etc., `.is_quantized=true`), `gguf-py/gguf/constants.py` (SCLP8/6/4 only — SCLP5 is C++-only). `ggml-cuda.cu` adds the MUL_MAT intercept in `ggml_cuda_mul_mat` and `case GGML_TYPE_SCLP8: return true;` in `supports_op`. **The `supports_op` entry is critical** — without it `select_weight_buft` returns nullptr at load and crashes.

### Blob layout (compact: stored at actual compressed size, padded to 32-byte GGUF alignment; size inferred from consecutive tensor offsets)

**SCLP6/SCLP8** (per-block scaling, PBS — BF16 scale per `QK_SCLP=32` weights):
```
[u32 num_weights][u32 n_experts]
[per-expert: u8 palette_size, u8[] palette] ...
[BF16 scales: ceil(expert_nw/32) per expert, contiguous]
[ws_stream]
[u32 sidecar_count][u32×count indices][u16×count values]
```
**SCLP4** (per-block palette — each `QK_SCLP4=256` block has its own 4-entry k-means palette; `palette_size=0` in header signals this mode; no scale multiply):
```
[u32 num_weights][u32 n_experts]
[per-expert: u8 palette_size=0] ...
[block_palettes: 4×u8 per block, ceil(expert_nw/256) blocks per expert]
[ws_stream]
[u32 sidecar_count][u32×count indices][u16×count values]
```
`ws_stream` size: `num_weights` (SCLP8), `ceil(N/2)` (SCLP4), `ceil(N/4)*3` (SCLP6). Sidecar values are raw BF16 of the original (unscaled) weight; `sclp_fixup_sidecar_kernel` writes them directly. Sidecar typically 0.01–0.03% (SCLP8) up to 1–5% (SCLP6 attn). For Llama-3-8B SCLP8: 8.47 GB vs 14.97 GB BF16 (−43%).

### Bridge kernels (`ggml/src/ggml-cuda/sclp_bridge.cuh`)
- `sclp_decode_blob_kernel` — self-parses header on-device, decodes 8 weights/thread via uint64 loads.
- `sclp_fixup_sidecar_kernel` — grid-stride sidecar scatter-write (no D2H read).
- `sclp_fused_gemv_kernel` / `sclp_fused_gemm_kernel` — fused decode+GEMV (M=1) / decode+GEMM (small M).
- `sclp4_fused_moe_gemv_kernel` / `sclp6_fused_moe_gemv_kernel` — fused MoE GEMV; one block per (row_tile, active_expert), decodes only routed experts inline (no full BF16 expert buffer). Wired into `ggml_cuda_mul_mat_id` when `n_batches==1`; prefill uses two-pass.
- `llama_sclp_dispatch` — two-pass decode+fixup, wired into `ggml_cuda_mul_mat`. All HIP-graph-safe.

**Two-pass vs fused:** generation (M=1) uses fused decode+GEMV (folds sidecar in — see below). Prefill (M>1) uses two-pass: decode blob → BF16, `sclp_fixup_sidecar_kernel` overwrites sidecar, then rocBLAS GEMM. Fusing decode into prefill GEMM was tried but F32 accumulation order (tensor-core tile vs scalar) diverges ~1e-3/mul, compounding to catastrophic PPL through MoE layers; not worth it for a ~20% prefill win.

### Compact GGUF loader support (4 files)
- `ggml/src/gguf.cpp` — `disk_size` field + `gguf_ti_nbytes()` helper; infers disk_size from offset deltas; new `gguf_set_tensor_disk_size()` API (decl in `ggml/include/gguf.h`).
- `src/llama-model-loader.h` — `disk_size` per weight from offset deltas; **last tensor derived from `file->size() - data_offset - tensor_offset`** (the `ggml_nbytes` fallback truncated the last compact blob → sidecar fixup read garbage `sc_count` → GPU hang; was root cause of SCLP4 M>1 prefill + llama-bench stalls).
- `src/llama-model-loader.cpp` — both load paths zero-pad to `n_size` and copy only `disk_size` bytes.
- `gguf-py/gguf/gguf_reader.py` — `_build_tensors` reads `disk_size` bytes as flat uint8 when it differs from `n_bytes`.

**disk_size hint:** loader stashes disk_size into `tensor->op_params[13:15]` (magic sentinel + u64) for exact VRAM alloc. `get_alloc_size` reads the hint, else falls back to 3× for SCLP4, 1.25× for SCLP8/6. Uses op_params reserved for compute ops; weight tensors (`OP_NONE`) are unused today — **on upstream merge, verify no new op_params writers collide.**

## Generating Models

### llama-quantize (native SCLP — preferred)
`llama-quantize` supports `SCLP`/`SCLP4`/`SCLP6` as quant types and `--tensor-type` overrides (`src/llama-sclp.{h,cpp}`). Replaces the old two-step Python converter for hybrids.

```bash
# Current default MIXED build (SCLP6 attn+ffn_down, SCLP4 per-block-palette gate/up):
llama-quantize \
  --imatrix eval_data/gemma4-opus-imatrix.dat \
  --tensor-type '^token_embd\.weight$=BF16' \
  --tensor-type '^output\.weight$=BF16' \
  --tensor-type '^blk\.[0-9]+\.attn_(q|k|v|output)\.weight$=SCLP6' \
  --tensor-type '^blk\.[0-9]+\.ffn_down(_exps)?\.weight$=SCLP6' \
  bf16-shard-00001-of-00002.gguf MIXED-bpal.gguf SCLP4 8
```

**Gotchas:**
1. **Flags go BEFORE positional args** (`tools/quantize/quantize.cpp:511`); args after the input file are silently ignored.
2. `--tensor-type` uses `std::regex_search` (substring) — anchor with `^...$` so `output.weight` doesn't match `attn_output.weight`.
3. **token_embd & output MUST be native** (BF16 or standard quant). SCLP only does MUL_MAT on GPU — GET_ROWS (embed/output lookup) crashes on CPU.
4. **Last positional arg is `nthread`** (default 1; use 8+). SCLP also threads per-expert internally.
5. Soft clipping disabled by default (`clip_threshold=0`), matching the Python encoder.

`llama_tensor_quantize_sclp()` parallelizes per-expert encoding: Gemma4-26B ~220s @ 8 threads (5.6× over single-threaded). Requires a local fork patch so `--tensor-type` overrides apply even when the base type isn't quantized (`src/llama-quant.cpp`) — worth upstreaming.

### Python converter (full SCLP from HuggingFace)
```bash
source eval_env/bin/activate
python3 tests/convert_to_sclp_gguf.py        # --format mixed for per-tensor policy
```
**F16→BF16 required:** if the source stores F16, `to_bf16_uint16()` converts via float32 first. Passing raw F16 bits to the encoder produces garbage (encoder assumes 1-8-7; F16 is 1-5-10).

`--format mixed` policy: attn_q/k/v/out + ffn_down(_exps) → SCLP6 (errors compound through softmax / pre-residual); ffn_gate/up(_exps) → SCLP4 (bulk, errors absorbed by GeGLU); token_embd/output/norms → BF16.

### imatrix (activation-aware sidecar)
Applied to **sidecar selection, not palette k-means** (naive k-means weighting regressed PPL 5×). Two sidecar tiers: (1) mandatory `dist > sidecar_dist`, (2) discretionary top-`budget` fraction by `importance × distance` (importance = `imatrix_value[flat_index % K]`).

```bash
llama-imatrix -m Q5_K_M.gguf -f wikitext-2-raw --output-format dat --chunks 80 -ngl 99 -c 512 --no-ppl
# then: --imatrix path.dat --sidecar-imatrix-budget 0.01   (knee at 1%; recommended default)
```
Budget sweep on Gemma4 mixed: 0→13,909 PPL, 0.5%→1,506, **1%→940**, 2%→1,026 (2% within error of 1% — quality floor). Calibrate from BF16 (not Q5_K_M) to avoid imatrix contamination (~hundreds PPL).

### Running inference
```bash
"$LLAMA_CPP"/build/bin/llama-completion \
    -m model.gguf -ngl 99 -n 100 -no-cnv --repeat-penalty 1.3 \
    -p "The capital of France is"
```
Use `llama-completion`, not `llama-cli` (no `-no-cnv` support). Both auto-enable conversation mode when a chat template is embedded; `-no-cnv` forces raw completion. `--repeat-penalty 1.3` avoids base-model loops. Verified on RX 7900 XTX (gfx1100).

## Results

**Llama-3-8B SCLP8** (wikitext-2): 8.47 GB / tg ~66 t/s* / pp ~2,730 t/s / PPL 9.87 (vs BF16 14.97 GB, 10.59; Q8_0 7.95 GB, 10.41). tg wins (8 vs 16 bits/weight); pp lags Q8_0 (two-pass decode adds a weight-matrix read before GEMM; Q8_0 goes direct through rocBLAS INT8).
*tg with folded sidecar is ~30 t/s (see below); 66 was pre-sidecar.

**Gemma4-26B-A4B-IT mixed** (`--jinja`): 17 GB / tg 55 t/s / coherent. Pure SCLP6 19 GB / 56 t/s ✓. Pure SCLP4 mode-collapses on this IT model (`"own own own..."`) — decode is byte-perfect (verified to 508M weights), so it's genuine k=4-palette quantization noise; Llama-3-8B SCLP4 is incoherent-but-diverse (no collapse). Without the fused SCLP4 MoE GEMV, MoE decodes all 128 experts/token (16× waste + 1 GB temp); with it, mixed matches SCLP6 speed.

**OOD PPL** (opus-trace holdout, 50 chunks, `-fa on`, opus imatrix) — the decision-relevant numbers:

| Config | Size | OOD PPL |
|---|---|---|
| MIXED-opus (SCLP6+SCLP4 **global** palette) | 17.0 GiB | 26.6 ± 1.1 |
| **MIXED-bpal** (SCLP6+SCLP4 **per-block** palette) ⭐ | 14.9 GiB | 132.6 ± 7.1 |
| MIXED-bpal (wikitext imatrix) | 14.9 GiB | 39.2 ± 1.8 |
| SCLP6-Q4K hybrid (opus imatrix) | 15.8 GiB | 290.4 ± 17.0 |

Findings: (1) per-block-palette SCLP4 beats Q4_K by **2.2× at 0.9 GiB smaller**; Q4_K trades +60% pp / +19% tg (justified only for prefill-bound work). (2) Global palette is better PPL but +2.1 GiB (almost all imatrix sidecar — per-block palette covers more exponents locally, so the same budget rescues fewer extra weights; per-byte it's more efficient). (3) Calibration domain matters for SCLP4 sidecar (−32% with opus traces), barely for Q4_K. (4) **Wikitext PPL is OOD-inflated ~50×** — use rankings, not absolutes (Gemma4-IT healthy OOD wiki ~50-200; Llama-3-8B base ~10).

**16 GB VRAM recipe** (Gemma4-26B, opus imatrix): **SCLP6attn-opus** (Q6_K embeds + SCLP6 attn only + SCLP4 ffn_down/gate/up) = 14.6 GiB / OOD PPL 40.0 ⭐ — 50% better than pure SCLP4 (14.3 GiB, 97.1) at ~same size. MIXED-opus-16gb (adds SCLP6 ffn_down) = 16.2 GiB / 26.4. Q6_K embeds save ~0.8 GiB with no measurable PPL hit.
```bash
llama-quantize --imatrix .../gemma4-opus-imatrix.dat \
  --tensor-type '^token_embd\.weight$=Q6_K' --tensor-type '^output\.weight$=Q6_K' \
  --tensor-type '^blk\.[0-9]+\.attn_(q|k|v|output)\.weight$=SCLP6' \
  bf16-shard-00001-of-00002.gguf SCLP6attn-opus.gguf SCLP4 8
```

**Recommendation:** per-block-palette SCLP4 is the default for new MIXED builds (smaller, no PBS degeneration). Global-palette SCLP4 + high imatrix budget wins on quality when size is unconstrained (+2 GiB). Q4_K gate/up only for prefill-bound workloads. **Regenerate any SCLP6 GGUF older than 2026-05-16 10:19** (pre-k-means).

## Folded Sidecar in Fused GEMV (the recurring sidecar-vs-fused perf problem, solved)

The encoder **sorts the sidecar by weight index** (`src/llama-sclp.cpp`, all SCLP types). Since `gidx = row*K + col`, each row's outliers form one contiguous range. The dense fused GEMV (M=1) folds the correction in: one warp/row binary-searches its range and adds `(true - palette_approx) * x[col]` using `x` already in smem — no atomics, no second kernel. Replaces the old per-entry atomicAdd (slower than two-pass) and the older "omit sidecar" workaround (a block-scoped scan caused a 37% regression).

Sorting is order-irrelevant to two-pass fixup and CPU decode, so **prefill is unaffected** (M>1 already overwrites sidecar before GEMM — folding there would be redundant). **Old SCLP GGUFs must be regenerated** — a fused GEMV on an *unsorted* sidecar melts down (~0.16 t/s); no back-compat guard.

tg (Llama-3-8B, RX 7900 XTX, all coherent): SCLP4 26.8 · SCLP5 28.9 · **SCLP6 34.2 (fused GEMV RE-ENABLED)** · SCLP8 ~30.

### Per-Block Scaling (PBS) — SCLP6/SCLP8 only
BF16 scale per `QK_SCLP=32` weights; encoder normalizes block by max-abs before encoding, decoder multiplies back. Overhead +6.25% (SCLP8) / +8.3% (SCLP6). Works (Gemma4-31B dense & 26B MoE coherent with `-ctk/-ctv turbo4`).

**PBS is fundamentally broken for SCLP4:** normalization concentrates exponents near 126-127, the 4-entry palette degenerates to ~2-bit scalar quant → mode collapse. **Solution = per-block palette** (each 256-weight block gets its own 4-entry k-means palette; exploits local exponent variation instead of destroying it). Overhead 1.56%. Llama-3-8B SCLP4 wikitext: per-block palette **102.4 PPL** vs 117.6 (global) vs 209 (PBS QK=256). Per-block palette is now the **only** SCLP4 mode — old SCLP4 GGUFs incompatible.

### SCLP5 (5-bit: idx2 | sign1 | mant2)
SCLP4's per-block palette + a 2nd mantissa bit; 8 weights → 5 bytes (eight 5-bit codes MSB-first in a 40-bit big-endian field). Full type implemented (encoder/dequant, GPU decode/sidecar/fused-GEMV, CPU decode, gates, ftype plumbing). **Pareto-dominated by SCLP4 + imatrix-sidecar** (uniform extra mantissa bit is less bit-efficient than targeted sidecar; no tg advantage). Kept selectable. Details: `plans/sclp4_vs_q4k_improvement.md`.

## Implementation Gotchas
- Scales stored as **BF16** not FP16 (kernels reinterpret as `__hip_bfloat16`; FP16 bits → garbage).
- `float → BF16` output = **top** 16 bits (`uint32 >> 16`), not bottom (`reinterpret_cast<uint16_t*>` takes low bytes on LE — wrong).
- BF16→float via `__bfloat162float()`, never `__low2float()` (that's for FP16 `__half2`).
- Sidecar stores the original *unscaled* BF16; fixup writes directly (no scale multiply).

## Known Issues
- **GPU wedge from `kill -9` mid-kernel** (WSL2/ROCm) → everything drops to ~0.16 t/s; needs `wsl --shutdown`. Bound GPU jobs with `timeout`; don't kill mid-kernel. (ROCm uses `/dev/dxg`, not `/dev/kfd`.)
- **SCLP6/SCLP4 fused GEMV K-alignment (latent):** kernels assume per-row group padding `row*ceil(K/group_size)` but the encoder packs flat. Misfires only when K isn't a multiple of the group size (4/2) — never on real LLM dims (multiples of 128). Fix if re-enabling.

## Resolved Issues (lessons)
- **SCLP4 prefill / llama-bench stalls** — last-tensor `disk_size` fell back to `ggml_nbytes` (smaller than compact blob), truncating the upload → sidecar fixup read garbage `sc_count` → GPU hang. Fixed in `llama-model-loader.h` (derive from file boundary).
- **SCLP4 VRAM** — `get_alloc_size` reserved only 1.25× but blobs reach 2.34× on Llama-3 attn_q (~11% sidecar; Gemma4 max 1.26×). Now 3×+64KB for SCLP4; also fixed `ggml_backend_tensor_set_async` bound.
- **SCLP6 fused GEMV garble** — was sidecar omission (SCLP6 has 1-5% sidecar vs SCLP8 ~0.02%); folded sidecar fixed and re-enabled it.

## Future Work
- **Converter-side precision fallback:** detect at encode time when a SCLP4 blob exceeds BF16 size and fall back to SCLP6/BF16 — natural on-ramp to measured (not name-based) per-tensor precision.
- **Per-tensor / per-expert precision & budget**; **error-magnitude sidecar** (rank by `importance × actual_error`).
- **TurboQuant for KV cache** (orthogonal: per-token rotation + low-bit). Cherry-pick onto `sclp`, expect rebase friction on KV cache types.
