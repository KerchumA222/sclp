# Plan: F16 intermediate for SCLP prefill GEMM

## Goal
Replace the BF16 intermediate buffer in the SCLP two-pass prefill path with F16, and feed the resulting `src0_bf16`-equivalent tensor (typed F16) into `ggml_cuda_mul_mat_id`. Hypothesis: rocBLAS's F16 GEMM on RDNA3 (gfx1100) outperforms its BF16 GEMM for our shape (K=2816, N=1408, M variable, ne_experts=128 with ~32 routed slots each), closing some of the 1210 → 2937 t/s gap to Q5_K_M.

This is a one-knob test. If F16 wins on perf and PPL stays in the same band, ship it; otherwise revert and move to lever #3 (Q8_0 on attention) or #4 (hipBLASLt).

## Why this might work
- rocBLAS BF16 GEMM is sometimes a thinner code path than F16 on RDNA3; F16 has a longer optimization history.
- F16 and BF16 are both 16-bit, so memory traffic is identical — purely a kernel-quality test.
- WMMA on RDNA3 supports both formats natively, so we're not falling off a hardware tier.

## Why it might not
- F16 has 5-bit exponent (range ~[-14, +15]), BF16 has 8-bit (~[-126, +127]). **Any SCLP palette exponent outside F16's range silently produces garbage** (overflow to inf or underflow to zero).
- F16 subnormals behave differently from BF16 — could shift numerics enough to hurt PPL even if range fits.
- rocBLAS may take the same kernel path for both types, in which case this is a no-op perf-wise.

## Pre-flight check (do this first, before any code)
Build a one-off Python script that walks `models/gemma4/google_gemma-4-26B-A4B-it-MIXED-imat-sidecar1pct.gguf` and `models/llama3/Llama-3-8B-SCLPws-Compact.gguf`, reads every SCLP tensor's palette, and reports:
- Per-tensor: min/max palette exponent (BF16 8-bit biased: 0–255).
- Aggregate: count of tensors where any palette exponent is **>142** (= 15 + 127, F16 overflow) or **<113** (= -14 + 127, F16 underflow into subnormals).
- Same check for sidecar values: decode each sidecar uint16 BF16, extract exponent byte, compare.

**Abort criterion**: if any non-trivial fraction (>1% of weights by count) sits outside F16 normal range, F16 is unsafe without scaling. Document and bail.

Reuse existing palette dumpers in `src/compression/encoder.py` rather than re-implementing.

## Implementation (if pre-flight passes)

### Step 1 — Add F16 variant of decode kernels
In `llama.cpp/ggml/src/ggml-cuda/sclp_bridge.cuh`, add F16 outputs to:
- `sclp_decode_blob_kernel` (SCLP8)
- `sclp4_decode_blob_kernel`
- `sclp6_decode_blob_kernel`
- `sclp4_fixup_sidecar_kernel`, `sclp_fixup_sidecar_kernel`, `sclp6_fixup_sidecar_kernel`

Simplest approach: template the output type, or copy each kernel to a `*_f16` variant. The bit-assembly changes from:
```cpp
uint16_t bits = (sign << 15) | (exp_bf16 << 7) | (mant_top1 << 6);
```
to:
```cpp
// F16: sign(15) | exp_f16(14:10) | mant(9:0)
uint8_t exp_f16 = exp_bf16 - 112;   // bias rebias 127 → 15
uint16_t bits = (sign << 15) | (exp_f16 << 10) | (mant_top1 << 9);
```
(Mantissa top bit migrates from BF16 position 6 to F16 position 9; remaining mantissa bits zero.)

Sidecar fixup also needs BF16→F16 conversion of the sidecar values stored in the blob (those are full BF16 uint16; convert per cell during fixup write).

### Step 2 — Wire dispatch
In `ggml-cuda.cu` SCLP intercept (around the `src0_bf16` construction):
- Allocate `decoded` as `uint16_t` (size unchanged).
- Call new F16 decode + F16 sidecar fixup kernels.
- Set `src0_bf16.type = GGML_TYPE_F16` (rename the local for clarity — `src0_inter`).
- nb computation already correct (sizeof(F16) == sizeof(BF16) == 2).

Apply to SCLP, SCLP4, SCLP6 paths.

### Step 3 — Gate behind env var initially
`SCLP_F16_INTER=1` env var → use F16 path; default (unset) keeps BF16. Lets us A/B without recompiling.

### Step 4 — Test PPL
With env var on, run perplexity for 50 chunks on Gemma4 MIXED and Llama-3-8B SCLP4. Compare to documented baselines:
- Gemma4 MIXED-imat-sidecar1pct (BF16 inter): 940 ± 55
- Llama-3-8B SCLP4-kmeans-sc1 (BF16 inter): need to look up

**Abort criterion**: PPL drift > 5% on either model. F16 is unsafe.

### Step 5 — Benchmark prefill
`llama-bench` with `-p 512 -n 0` on the same two models, with and without `SCLP_F16_INTER=1`. Three runs each, take median.

**Success criterion**: prefill t/s improves by ≥10% on Gemma4 (target: 1210 → ~1330+) AND tg t/s does not regress more than 5%.

### Step 6 — Decide
- Win: remove env gate, make F16 the default. Update CLAUDE.md numbers.
- Loss: revert the wiring change, keep F16 kernels behind the gate as a documented experiment, move on to lever #3.

## Out of scope
- Changing the GGUF on-disk format (still SCLP).
- Touching the fused MoE prefill kernels (separate dead-end debugging).
- Per-tensor scaling to make out-of-range exponents fit (deferred; if pre-flight finds out-of-range exps, that's its own project).

## Estimated effort
- Pre-flight script: ~30 min.
- F16 decode kernels (3 of them + sidecar variants): ~2 hours (mostly mechanical).
- Dispatch wiring + env gate: ~30 min.
- PPL + prefill measurements: ~30 min wall time, can run in background.

Total: half a day, including the abort paths.

## Done definition
Either:
1. F16 path is default, CLAUDE.md updated with new prefill number and PPL re-verified, OR
2. F16 path is reverted (or env-gated and documented as failed), with the failure mode written into the relevant memory note so we don't re-try it.
