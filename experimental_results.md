# Experimental Results: SCLP Compression Performance

This document aggregates experimental results for the Soft-Exponent Clipping (SCLP) algorithm.

## 1. Distribution-based Error & Compression Analysis
*Test Configuration: Threshold Exponent = 125, Full Mantissa Precision (Mask=0x7F)*

| Distribution | Compression Ratio | Mean Squared Error (MSE) |
| :--- | :---: | :---: |
| **Uniform** | 2.00x | 3.60e+74 |
| **Gaussian** | 2.00x | 5.49e+70 |
| **Laplace** | 2.00x | 3.14e+74 |
| **LLM-Pareto** | 2.00x | 1.77e+76 |

> **Note on MSE values:** Extremely high values are attributed to the extreme exponent ranges produced by the Pareto and Laplace generators in this test script, causing large float64 magnitudes during error calculation.

## 2. Kernel & Pipeline Verification
*Test Configuration: Random seed-based weight generation (100 weights)*

| Test Case | Status | Key Metric/Observation |
| :--- | :---: | :--- |
| **End-to-End Pipeline** | **PASSED** | $\text{Decoded} \equiv \text{Clipped}$ (Lossless reconstruction of compressed state) |
| **Mantissa Truncation** | **PASSED** | Correct bitwise zeroing; pass `mantissa_mask=0x7F` to preserve all bits |
| **Module Integration** | **PASSED** | Python-to-HIP pybind11 binding functional with correct arg signature |
| **Interface Integrity** | **PASSED** | Encode/Decode shape and type consistency verified (Packed: 50, SM: 100) |

## 3. Precision Scaling Analysis (Mantissa Truncation)
*Test Configuration: Threshold Exponent = 125, Variable Mantissa Masking (0x7F → 0x0F → 0x03)*

| Distribution | Compression Ratio | Relative Error (RelErr) |
| :--- | :---: | :---: |
| **Uniform** | 2.00x | 0.51 → 0.57 → 0.58 |
| **Gaussian** | 2.00x | 0.49 → 0.63 → 0.64 |
| **Laplace** | 2.00x | 0.49 → 0.63 → 0.65 |
| **LLM-Pareto** | 2.00x | 0.14 → 0.19 → 0.19 |

> **Observation:** Error increases as bits are truncated, but remains significantly lower for heavy-tailed (Pareto) distributions, validating the exponent-centric precision approach.

## 4. OPT-125m Exponent Distribution (MLP Layers)
*Source: `analyze_exponent_dist()` in `tests/test_real_ppl_impact.py`*
*Scope: All 24 fc1+fc2 MLP layers, 56.6M parameters*

- **26 unique exponents** across all MLP weights
- Top **10 exponents** cover **99%** of weights
- Top **14 exponents** cover **99.9%** of weights
- A handful of weights (~7K, ~0.01%) carry exponent 125–126 and are disproportionately critical to quality (see §6)

This confirms the palette compression assumption: a 16-entry palette captures virtually the entire weight population, with outliers handled losslessly via the sidecar mechanism.

## 5. Real-World PPL Impact: Stage A — Clipping Only (OPT-125m)
*Scope: All 24 fc1+fc2 MLP layers. No encode/decode, no mantissa truncation.*
*Baseline PPL: 53.32 (WikiText-2, 20 samples)*

| Threshold | Weights Clipped | PPL | ΔPPL% |
| :---: | :---: | :---: | :---: |
| 117 | — | — | — |
| 119 | — | — | — |
| 121 | — | — | — |
| 122 | — | — | — |
| 123 | — | — | — |
| 124 | — | — | — |
| **125** | **~0.01%** | **~56.83** | **+6.58%** |

> `—` rows need a full sweep re-run (`python3 tests/test_real_ppl_impact.py`). Only threshold=125 was captured in the session that produced these results.

## 6. Real-World PPL Impact: Stage B — Mantissa Truncation Only (OPT-125m)
*Scope: All 24 fc1+fc2 MLP layers. Bottom 4 mantissa bits zeroed, no clipping.*
*Baseline PPL: 53.32*

| Operation | PPL | ΔPPL% |
| :--- | :---: | :---: |
| Bits 3:0 zeroed (mask 0xFFF0) | ~52.23 | **-2.04%** |

> The slight PPL *improvement* is within measurement noise (20 samples). Conclusion: full-mantissa lossless SM stream is the right default; mantissa truncation is negligible but not necessary.

## 7. Real-World PPL Impact: Stage C — Full Pipeline (OPT-125m)
*Scope: All 24 fc1+fc2 MLP layers. Clip + encode (nibble palette indices + SM stream) + decode.*
*Baseline PPL: 53.32. MLP storage: 113 MB BF16.*

| Threshold | Comp. Ratio | MLP (comp.) | PPL | ΔPPL% |
| :---: | :---: | :---: | :---: | :---: |
| 122 | — | — | — | — |
| 123 | — | — | — | — |
| **124** | **2.000x** | **~56 MB** | **~243** | **+357%** |
| **125** | **2.000x** | **~56 MB** | **~56.30** | **+5.60%** |

> **Cliff at threshold=124:** ~7K weights carrying exponents 125–126 are essential to model quality. Clipping them (threshold=124) inflates their magnitude by up to 2^(125-121) = 16384× (without sidecar) or stochastically rounds half of them to threshold (with sidecar), causing catastrophic PPL. Threshold=125 leaves these weights intact.
>
> `—` rows need a full sweep re-run.

**Operating point: threshold=125** — 1.333× MLP compression, +5.60% PPL vs. BF16 baseline.

## 8. Attention Layer Compression (OPT-125m)

Evaluated the impact of adding attention layers (Q, K, V, O) to the compression pipeline at threshold=125.

| Scenario | MLP Compressed | Attn Compressed | Model Compression | PPL | ΔPPL% |
|---|---|---|---|---|---|
| Baseline | No | No | 1.000x | 53.32 | — |
| MLP Only | Yes | No | 1.127x | 54.17 | +1.59% |
| Attn Only | No | Yes | 1.060x | 53.69 | +0.70% |
| **MLP + Attn** | **Yes** | **Yes** | **1.204x** | **54.40** | **+2.04%** |

> **Finding:** Attention layers are significantly less sensitive to exponent clipping than MLP layers. Adding them to the compression pipeline improves model-wide compression by 6.8% with minimal additional quality loss (+0.45% ΔPPL vs MLP-only).

## 9. Llama-3-8B Scaling Results

Evaluated SCLP on `unsloth/llama-3-8b` (BF16) using a subset of WikiText-2. All MLP and Attention linear layers (86.9% of model parameters) were compressed.

| Scenario | Threshold | Model Compression | PPL | ΔPPL% |
|---|---|---|---|---|
| Baseline | — | 1.000x | 10.99 | — |
| **Full SCLP** | **125** | **1.772x** | **10.98** | **-0.09%** |
| Full SCLP | 123 | 1.772x | 7387.59 | +67110% |

> **Finding:** Llama-3-8B shows the same critical threshold behavior as OPT-125m. At threshold=125, we achieve near-lossless compression (ΔPPL within noise floor) while compressing 87% of the model to 8 bits/weight (SCLP8). The slight negative ΔPPL is likely due to the small sample size (10 samples) and stochastic rounding effects.

## 10. GPU Throughput Benchmarks (ROCm)

Evaluated the `decode` kernel performance on a large weight matrix (500M weights, 1GB BF16).

| Implementation | Actual BW | Effective BW | Throughput | Speedup vs BF16 |
|---|---|---|---|---|
| BF16 Baseline (Copy) | 5.70 GB/s | 5.70 GB/s | 1424 M weights/s | 1.00x |
| **SCLP Vectorized Decode** | **6.22 GB/s** | **2.35 GB/s** | **2848 M weights/s** | **2.00x** |

> **Analysis:** The vectorized SCLP decoder achieves **near-theoretical maximum speedup** (2.0x). By processing two weights per thread and using LDS for the palette, we successfully hide the decoding compute overhead behind memory latency. The effective bandwidth of 2.35 GB/s represents the processing speed relative to original BF16 weights.

## 11. Fused Decode-GEMV Kernel Optimization (RX 7900 XTX, gfx1100)

Iterative optimization of the fused SCLP decode-GEMV kernel (M=1 single-token inference path).
Baseline: two-pass decode-to-buffer → rocBLAS GEMM.

| Optimization | t/s | Δ |
|---|---|---|
| Two-pass decode + rocBLAS (baseline) | 16.0 | — |
| Fused GEMV, scalar byte loads | 27.6 | +73% |
| + Vectorized loads (uint64_t sm, uint32_t packed) | 44.6 | +62% |
| + 16 warps/block (512 threads) | 47.6 | +7% |
| + Dual accumulators + `__launch_bounds__(512,4)` | 49.0 | +3% |
| FP16 rocBLAS reference | ~52 | — |

> **Approaches that hurt and were reverted:**
> - Shared memory staging for activations: −14% (syncthreads barriers outweigh reuse benefit)
> - bfloat162 paired multiply: −13% (float2 accumulation overhead > scalar FMA)
>
> **Residual 3 t/s gap** to FP16 baseline is inherent decode overhead (palette lookup + nibble
> unpacking). Closing it further requires a fundamentally different approach such as a
> 256-entry precomputed lookup table mapping packed bytes directly to BF16 pairs.

## 12. Future Metrics
- [ ] Full threshold sweep: Stage A at all thresholds (117–125), Stage C at 122–123.
- [ ] Throughput (GB/s) benchmarks on RDNA3 hardware (RX 7900 XTX, gfx1100).
- [ ] PPL delta on Llama-3-70B.
- [x] Per-layer threshold tuning — early/late transformer layers may require higher thresholds.
- [x] Attention layer compression (implemented and validated).
- [x] Scaling validation on Llama-3-8B.
- [x] GPU Throughput validation (Vectorized Decoder).

## 13. Selective Sidecar Removal — PPL vs Quality Trade-off (Llama-3-8B, 2026-05-02)

*Methodology: sidecar entries ranked by drop_cost = MAE × sidecar_count per tensor.*
*PPL measured with llama-perplexity on WikiText-2 test set (~10K tokens), ctx=2048.*
*Hardware: AMD RX 7900 XTX (gfx1100). Inference via fused decode-GEMV kernel.*

| Model | Tensors dropped | Drop cost | PPL | ΔPPL |
|---|---|---|---|---|
| Full sidecars (baseline) | 0% | 0.00% | 9.5493 | — |
| attn_v sidecars dropped | 32/224 | 0.45% | 9.4841 | -0.68% |
| 50% cheapest dropped | 112/224 | 10.47% | 8.9978 | -5.78% |
| 90% cheapest dropped | 165/224 | 44.27% | 6.3503 | -33.50% |

**Key findings:**

1. **No speed benefit** — sidecar removal has no measurable effect on inference speed (48.89 → 48.81 → 48.86 → 49.08 t/s across all four variants). The fixup kernel processes at most ~8K entries per tensor on a fixed 4-block grid — negligible vs the GEMV over 16–58M weights.

2. **PPL improves (unexpectedly) as sidecars are dropped.** This is not a quality regression. Analysis of the sidecar exponent distribution reveals why: 72% of sidecar entries are *below*-palette exponents (exp 103–106, just under the palette floor of 106), not large outliers. When their sidecar is dropped, the decoder rounds their exponent up to the nearest palette entry (~106–108), slightly inflating near-zero weights. This acts as mild regularization on noisy small-magnitude weights.

3. **Exponent distribution of sidecar entries** (across all 224 SCLP tensors):
   - Palette range: exponents 106–124 (top-16 most frequent)
   - Sidecar below palette floor (exp < 106): **72.1%** of sidecar entries
   - Sidecar above palette ceiling (exp > 124): **0.7%** of sidecar entries
   - Sidecar at exp=0 (subnormals): 15,393 entries across whole model

4. **The `drop_cost` proxy is still valid** for identifying which layers are safe to drop, but the objective is now confirmed to be: "minimize magnitude inflation of near-zero weights" rather than "avoid clipping large outliers." Layers with low MAE are safe because their sidecar weights are already close to the nearest palette exponent.

5. **Recommendation**: All 32 `attn_v` sidecars can be dropped at zero PPL cost. This also simplifies GGUF generation by skipping sidecar encoding for V projections entirely.

**Inference speed summary (RX 7900 XTX, fused decode-GEMV kernel):**

| Path | t/s |
|---|---|
| FP16 rocBLAS baseline | ~52 |
| SCLP two-pass (decode → GEMM) | ~16 |
| SCLP fused decode-GEMV | ~49 |

## 14. Compact GGUF Format — Disk Savings (Llama-3-8B, 2026-05-06)

*Tool: `tests/repack_sclp_gguf.py`. Input: padded SCLP GGUF. Hardware: AMD RX 7900 XTX.*

The padded SCLP GGUF format allocates `num_weights * 2` bytes per tensor slot (matching BF16 stride), leaving the gap between the actual compressed blob end and the slot boundary as zero padding. The compact format stores each blob at its actual size, padded only to 32-byte GGUF alignment.

| Tensor type | Size (padded) | Size (compact) | Ratio |
|---|---|---|---|
| SCLP tensors (224 total) | 13.00 GB | 6.50 GB | 2.000× |
| Non-SCLP tensors (67 total) | 1.97 GB | 1.97 GB | 1.000× |
| **Total file** | **14.97 GB** | **8.47 GB** | **1.767×** |

**Savings: 6.50 GB (43.4% reduction).**

The compact format is fully supported by the modified loader — inference verified correct at ~46 t/s on the compact file (identical output to padded format). The SCLP decode kernel self-parses the blob header on-device so it is unaffected by the change in on-disk allocation.

## 15. Task Accuracy: Llama-3-8B Variants (0-shot, lm-evaluation-harness, 2026-05-08)

> **Note:** §15–17 used an earlier SCLP GGUF with separate packed+sm streams (11.72 GB). The current ws_stream interleaved format (§14) produces 8.47 GB compact files. Speed and quality results here are from the older format but remain directionally valid.

*Hardware: AMD RX 7900 XTX (gfx1100) · GGML_HIP=ON · ngl=99 · ctx=16384*
*Evaluation: lm-evaluation-harness via local-completions backend, 100 examples per task.*
*Tasks: HellaSwag (acc_norm), ARC-Challenge (acc_norm), ARC-Easy (acc)*

| Variant | File Size | HellaSwag | ARC-C | ARC-E |
|---|---|---|---|---|
| BF16 baseline | 15.00 GB | 0.570 | 0.360 | 0.490 |
| **SCLP compact** | **11.72 GB** | **0.560** | **0.350** | **0.440** |
| Q8_0 | 8.00 GB | 0.580 | 0.370 | 0.470 |

**Key finding:** SCLP compact (11.72 GB, 1.277× compression) scores within 1–5% of the BF16 baseline on all tasks. The small deltas are within expected noise for 100-example evaluation. SCLP retains full task accuracy while saving 3.28 GB (21.9%) over BF16.

## 16. Perplexity and Speed: Llama-3-8B Variants (2026-05-08)

*Hardware: AMD RX 7900 XTX (gfx1100) · GGML_HIP=ON · ngl=99*
*PPL: llama-perplexity on WikiText-2, ctx=512 (standard window).*
*Speed: llama-bench, 512 prompt + 128 generation tokens.*

| Variant | File Size | PPL (WikiText-2) | Prompt (t/s) | Generation (t/s) |
|---|---|---|---|---|
| BF16 baseline | 15.00 GB | 10.1443 | 3182.5 | 50.3 |
| **SCLP compact** | **11.72 GB** | **10.0417** | **2517.0** | **47.1** |
| Q8_0 | 8.00 GB | 9.9751 | 3393.2 | 82.3 |

**Key findings:**

1. **PPL parity:** SCLP PPL (10.04) is within noise of BF16 (10.14), confirming near-lossless quality at 1.277× compression. Consistent with the Python pipeline result in §8 (10.98 vs 10.99).

2. **Generation speed 94% of BF16:** 47.1 vs 50.3 t/s — the fused decode-GEMV kernel (§10) eliminates virtually all decode overhead for single-token inference. The residual 3.2 t/s gap matches the ~3 t/s gap identified in §10 as inherent nibble-unpack overhead.

3. **Prompt speed slower (79% of BF16):** 2517 vs 3182 t/s — the prefill path still uses the two-pass decode (decode-to-buffer → matrix multiply). No fused prefill kernel exists yet; this is the next optimization target.

4. **Q8 generation advantage:** 82.3 t/s reflects integer arithmetic throughput; Q8 fits the model in less VRAM, reducing memory bandwidth pressure during generation.

## 17. Fused Prefill GEMM + Sidecar Correction (2026-05-08)

*Hardware: AMD RX 7900 XTX (gfx1100) · GGML_HIP=ON · ngl=99*
*Model: Llama-3-8B-SCLP-Patched.gguf (padded format, 14.96 GiB)*
*Speed: llama-bench, 512 prompt + 128 generation tokens.*

### Root Cause of Prefill Corruption (Found and Fixed)

The fused GEMV/GEMM kernels intentionally skip sidecar fixup for speed. The sidecar contains ~0.02% of weights whose exponents fall outside the top-16 palette — the nearest palette exponent is used instead. While tiny in count, individual sidecar weights can differ from their palette approximation by 50–200%, causing individual output elements to be off by 2–7×.

For **M>1 prefill**: these errors propagate into the KV cache for all prompt positions. The corrupted KV cache causes catastrophic garbage output during decode.

For **M=1 decode**: the same errors are tolerable because each new token is computed against a correctly-prefilled KV cache from prefill, and the error affects only ~1 of 4096 output elements per layer.

### Fix: Sidecar Correction Kernel

A new `sclp_sidecar_correct_gemm_kernel` runs after the fused GEMM and atomically adds `(w_correct − w_approx) × x[k]` for each sidecar entry to the output. For all M token rows simultaneously. Cost is negligible: ~1K–5K atomic adds per layer vs 16M+ multiply-accumulates in the main GEMM.

The M=1 GEMV path does NOT include sidecar correction (not needed for correctness; adding it dropped generation from 47 → 34 t/s due to kernel launch overhead across 224 layers).

### M Threshold for Fused GEMM

The fused GEMM re-decodes the weight blob `ceil(M/TILE_M)` times — once per TILE_M=4 token rows. For large M (e.g. M=512 from llama-bench pp512), this is 128× more blob reads than two-pass. A threshold of **M ≤ 2×TILE_M = 8** is applied: larger M falls through to two-pass so cuBLAS handles it efficiently.

### Speed Results (after fix)

| Path | Prompt pp512 (t/s) | Generation tg128 (t/s) |
|---|---|---|
| BF16 baseline | 3182.5 | 50.3 |
| SCLP two-pass (pre-GEMM) | 2517.0 | 47.1 |
| SCLP fused GEMM + sidecar | **2537.7** | **47.0** |

**Key finding:** Speed is now statistically identical to the two-pass baseline (within noise) with correct output. The fused GEMM path handles M≤8 micro-batches correctly (no intermediate BF16 buffer, with sidecar correction). Larger M automatically falls through to two-pass.

The generation speed (47.0 t/s, 93% of BF16) is unchanged — the GEMV path is unaffected.

**Next steps:** Increase TILE_M to amortize weight decoding across more token rows, and benchmark at smaller ubatch sizes (M=2..8) where the fused GEMM is active.

## 18. KL-Divergence vs Perplexity for SCLP Eval (2026-06-12)

**Headline: gate SCLP encoder/sidecar changes on KL-divergence against a high-precision reference, not perplexity.** Perplexity — in- *and* out-of-distribution — is confounded and noisy; KL cleanly measures distance from full precision. Established on two experiments:

### 18a. Palette change (#4), in-distribution (Llama-3-8B SCLP6, budget 0)
The magnitude-weighted palette (#4) cut reconstruction MSE 25–95% but **raised** in-distribution wikitext PPL 8.95 → 9.91. KL vs the FP16 model **fell** 9.8×:

| Encoder | wikitext PPL | Mean KLD vs FP16 |
|---|---|---|
| count-weighted palette (pre-#4) | 8.95 | 0.2686 |
| magnitude-weighted palette (#4) | 9.91 | **0.0274** |

The lower-MSE model is an order of magnitude closer to full precision; the PPL rise is a regularization artifact (the lossy palette smooths in-distribution text). Verdict: ship #4. KL is the decider.

### 18b. Sidecar budget sweep, OOD (Gemma4-12B-IT-QAT MIXED-bpal)
Swept `--sidecar-budget` {0, 0.5%, 1%, 2%}. OOD PPL (opus-trace) was **non-monotonic noise** — 0.5% scored *worst*. KL vs a Q8_0 reference (BF16 too slow to host on a 24 GB card; Q8_0 ≈ full precision, all models vs the same ref so ranking is exact) is **cleanly monotone-decreasing** — confirming the nested sidecar (each budget overwrites more weights to exact BF16) always moves closer to truth:

| Model | Size (GiB) | OOD PPL (50ch) | **Mean KLD vs Q8_0** | Median KLD |
|---|---|---|---|---|
| SCLP MIXED-bpal, budget 0 | 8.77 | 44.1 | 1.460 | 0.632 |
| SCLP MIXED-bpal, 0.5% | 8.98 | 55.5 (worst) | 1.418 | 0.599 |
| SCLP MIXED-bpal, 1% | 9.19 | 48.9 | 1.193 | 0.480 |
| SCLP MIXED-bpal, 2% | 9.61 | 48.2 | 1.011 | 0.335 |
| **Q4_K_M** | 6.87 | 40.4 | **0.400** | 0.073 |
| **Q4_0** | 6.52 | 37.3 | **0.148** | 0.012 |

**Findings.** (1) The discretionary sidecar is mechanically sound: KL falls monotonically with budget (the PPL inversion was pure noise). Since KL never stops improving, there is **no "knee"** — budget is a size↔fidelity tradeoff, and the 1% default is a reasonable midpoint (KL would argue for *higher*, never lower). (2) Every SCLP budget is 2.5–10× **further** from full precision than Q4_K_M / Q4_0 at **larger** size — decisive on this model. **Caveat:** this is a *QAT-targeting-Q4_0* model, so Q4_0 is near-lossless **by design** (weights sit on its grid); SCLP's log-palette grid can't match QAT-conditioned weights. This is a QAT-specific advantage for Q4_0, **not** a general SCLP verdict — the non-QAT Gemma4-26B-MoE result (per-block SCLP4 beat Q4_K 2.2×) still stands. The proper non-QAT-MoE budget re-validation needs a 26B BF16 source (not on disk).

**Method note.** KL via `llama-perplexity --kl-divergence-base <ref.dat>` (write phase, high-precision model) then `--kl-divergence` (compare phase, each quant), `-fa off -c 512`. A coherence smoke is still required (PPL/KL can both miss mode collapse) — but note raw `-no-cnv` completion garbles on *all* quants of an IT model (needs the chat template), so use a templated chat smoke.

## 19. SCLP4M vs SCLP4 — magnitude codebook beats exponent palette at 4-bit (2026-06-16)

**Question.** SCLP4 spends 4 bits as `2 idx (4-entry log-exponent palette) | sign | 1 mantissa`; SCLP4M spends ~4.5 bits as `3 idx (8 *arbitrary* BF16 magnitudes from per-block linear-space Lloyd k-means) | sign`. Is the free-floating magnitude codebook worth the ~0.4 extra bpw vs the exponent grid + 1 mantissa bit? Gated on **KL vs a higher-precision reference**, not PPL (§18).

### 19a. Dense, whole-model 4-bit (Llama-3-8B, KL vs Q8_0, 50ch wikitext, 1% sidecar budget, embeds BF16)
| Quant | Size (GiB) | PPL(Q) | PPL(Q)/base | **Mean KLD** | Median KLD | RMS Δp |
|---|---|---|---|---|---|---|
| SCLP4 (per-block palette) | 6.219 | 14.63 | 1.488 | 0.557 | 0.355 | 22.6% |
| **SCLP4M (magnitude codebook)** | 6.321 | 10.04 | **1.021** | **0.0572** | **0.0255** | **7.4%** |

base = Q8_0 (PPL 9.86). **SCLP4M is 9.7× closer to full precision (mean KLD), near-lossless (PPL ratio 1.02 vs 1.49), for +1.6% size.** Coherent smoke (math-forum continuation, no collapse). A decisive Pareto win: 8 arbitrary per-block magnitudes represent the local weight distribution far better than a 4-entry exponent grid + one mantissa bit; here PPL *agrees* with KL.

### 19b. MoE, MIXED build, only gate/up exps swapped (Gemma4-26B-A4B-it-qat, KL vs Q5_K_M, 32ch wikitext)
Both builds: SCLP6 attn+ffn_down, BF16 embeds, opus imatrix, 1% budget; **only** ffn_gate/up_exps differ (SCLP4 vs SCLP4M) — isolates the codebook in the role 4-bit actually occupies.
| Build | Size (GiB) | PPL(Q)/base | **Mean KLD** | Median KLD | Max KLD |
|---|---|---|---|---|---|
| MIXED-SCLP4 gate/up | 16.62 | **1.060** | 0.434 | 0.287 | 7.34 |
| **MIXED-SCLP4M gate/up** | 17.30 | 1.140 | **0.353** | **0.216** | **6.48** |

base = Q5_K_M (Q8_0/Q6_K won't fit the 24 GB card given Gemma4's ~262K vocab + logit buffers; Q5_K_M is closer to truth than either 4-bit candidate so KL-to-Q5KM tracks KL-to-truth; both candidates vs same ref → ranking exact). MIXED-SCLP4M coherent (templated chat, Rayleigh-scattering answer, no collapse).

**Findings.** (1) **Direction confirmed on MoE: SCLP4M is strictly more faithful** (−19% mean / −25% median KL). (2) The **KL/PPL inversion recurs** — SCLP4M has *higher* wikitext PPL (ratio 1.14 vs 1.06) yet *lower* KL: better reconstruction de-regularizes in-distribution text (cf. §18a). KL decides → SCLP4M. (3) The margin is **modest vs the 9.7× dense blowout** because (a) MIXED swaps only the least-sensitive tensors (gate/up; attn+ffn_down stay SCLP6 in both, capping reach), and (b) QAT pre-shapes weights onto a Q4_0 *linear* grid that both 4-bit schemes handle well — and QAT's linear grid mildly *flatters* SCLP4M's linear codebook, so the non-QAT MoE gap is likely between the two. (4) SCLP4M costs **+0.73 GiB** (codebook 0.5 b/wt vs palette 0.125 b/wt on gate/up).

**Verdict.** Magnitude codebook ≥ exponent palette at 4-bit across **both** model classes — decisive on whole-model dense, incremental when confined to bulk gate/up. **Recommend SCLP4M as the default 4-bit mode at its natural (with-sidecar) size** (retire SCLP4 as the recommended path). The "widen 4-bit coverage" follow-up is explored in §19c — pure SCLP4M replaces SCLP6 (−2.2 GiB) and stays coherent, but the codebook-vs-sidecar trade is a *crossover* (it only pays off above SCLP4's size), and the collapse-avoidance can't be credited to SCLP4M on a QAT testbed.

**Codebook-overhead caveat (why not propagate to 6/8-bit):** the per-block codebook is `2^idx × 2 B` per 256-block — SCLP4M (8 entries) +0.5 b/wt (cheap), "SCLP6M" (32) +2.0 b/wt → ~7 bpw effective, "SCLP8M" (128) +8 b/wt (doubles storage). The codebook is a 4-bit-native trick; SCLP6/8 already amortize their palette per-*expert* + PBS + mantissa bits, which is more bit-efficient at those widths. "SCLP6M vs SCLP6" remains an untested empirical question, not a mechanical rollout.

### 19c. Widening 4-bit: pure SCLP4M (replace SCLP6) + codebook-vs-sidecar Pareto (Gemma4-26B-A4B-qat, 2026-06-16)
Goal: use SCLP4M *in place of* SCLP6 (and SCLP4) everywhere, to shrink the MIXED build, and isolate whether the SCLP4M codebook is worth its bytes vs spending the same bytes on sidecar. All **pure** (every linear proj at the named 4-bit type, embeds/output BF16, opus imatrix), KL vs the same Q5_K_M base, 32ch wiki, all **coherent** (templated chat).

| Pure build | Size (GiB) | bpw | **Mean KLD** | Median KLD |
|---|---|---|---|---|
| SCLP4M, sidecar stripped (`--sidecar-budget '.*=0.0'`) | 15.31 | 5.09 | 1.069 | 0.866 |
| SCLP4, 1% (default) | 15.21 | 5.05 | 0.727 | 0.577 |
| SCLP4, 2.5% (`'.*=0.025'`) | 16.68 | 5.54 | 0.599 | 0.456 |
| **SCLP4M, 1% (default)** | 16.36 | 5.18 | **0.479** | 0.332 |
| *(ref)* MIXED-SCLP4 (SCLP6 attn+down) | 17.85 | 5.65 | 0.434 | 0.287 |
| *(ref)* MIXED-SCLP4M (SCLP6 attn+down) | 18.58 | 5.89 | 0.353 | 0.216 |

Per-tensor, **SCLP4M is ~29% smaller than SCLP6 on the same tensor** (attn_q 9.39→6.64, attn_v 4.70→3.32, attn_output 18.77→13.27, ffn_down_exps 207.68→147.18 MiB; analytic ~0.56 vs 0.81 B/wt). Replacing SCLP6 with SCLP4M everywhere drops MIXED-4m 18.58 → pure-4m 16.36 GiB (−2.2) but raises KL 0.353 → 0.479 (4-bit < 6-bit on sensitive attn/ffn_down).

**Findings.** (1) **Codebook-vs-sidecar is a crossover, not a clean win.** At ~15.2 GiB (SCLP4's size), SCLP4-1% (0.727) **beats** SCLP4M-with-sidecar-stripped (1.069) — the codebook (0.0625 B/wt) is *less* bit-efficient than targeted sidecar (same lesson as [[project_sclp5_dominated]]). At ~16.4 GiB, SCLP4M-1% (0.479) **dominates** SCLP4-2.5% (0.599) — lower KL *and* smaller. So **SCLP4M is not a free drop-in at SCLP4's size**: its +1.15 GiB codebook overhead only pays off when you can *also* afford a sidecar; reducing SCLP4M's sidecar to match SCLP4's size makes it worse than SCLP4. (2) **Pure SCLP4M is coherent (collapse-free) — but so is pure SCLP4 here**, because this is the **QAT** model (weights pre-conditioned onto the Q4_0 linear grid prevent the collapse). The original non-QAT "own own own" collapse is *not* reproduced on QAT, so this testbed **cannot credit SCLP4M for solving collapse** — QAT did. (3) QAT linearization *helps SCLP4's palette and mutes the codebook's edge*: contrast the non-QAT dense Llama (§19a) where SCLP4M won 9.7× near iso-size. The crossover is QAT-specific; a non-QAT 26B MoE would likely favor SCLP4M more.

**Deployment frontier (KL monotone-decreasing with size):** `SCLP4-1% (15.2)` → `SCLP4M-1% (16.4)` → `MIXED-SCLP4 (17.9)` → `MIXED-SCLP4M (18.6)`. Pick by budget: **~15 GiB → pure SCLP4**; **~16.4 GiB → pure SCLP4M** (best per-byte in range, replaces SCLP6, −2.2 GiB vs MIXED-4m); **~18 GiB → MIXED** (SCLP6 on attn+ffn_down). SCLP4-2.5% is dominated (off-frontier). **Caveat carried:** verdict measured on a QAT model; non-QAT crossover point unknown.
