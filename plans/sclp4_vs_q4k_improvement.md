# SCLP4 vs Q4_K — measured gap and improvement plan (2026-05-24)

## Goal
Make SCLP4 match or exceed Q4_K on quality AND throughput (TG prioritized, then load time).

## Measured baseline (Llama-3-8B, wikitext-2 test, RX 7900 XTX, -ngl 99 -fa on)

| Model | Size | bpw | PPL (40 chunks) | tg t/s | prefill | Output quality |
|---|---|---|---|---|---|---|
| Q4_K_M | 4.69 GB | 4.89 | **9.99 ± 0.29** | ~52 | fast | coherent |
| SCLP4 (per-block palette, Q6_K embeds) | 4.86 GB | 5.19 | ~102 (doc) | 31 | **~11 t/s (pathological)** | degenerates after ~3 tokens |

Two independent problems, both confirmed this session:
1. **Quality**: SCLP4 PPL ~10× worse than Q4_K at comparable size. Generation degenerates
   ("The capital of France is Paris :.?, and... (s), that to save the by each two million0002...").
2. **Prefill throughput**: SCLP4 dense prefill ~11 t/s (4 PPL chunks > 180s). M=1 generation is
   fine (31 t/s) — the two-pass decode at M=512 (decode whole matrix → BF16 → rocBLAS) is the
   bottleneck. Q4_K goes straight through rocBLAS INT8.

(Note: full SCLP4 PPL runs take ~30 min because of #2 — need a faster eval loop, e.g. 10 chunks,
or fix #2 first so iteration is tractable.)

## Root cause of the quality gap
SCLP4 nibble = `idx(2b) | sign(1b) | mant_top1(1b)` (llama-sclp.cpp:13).
- 4-entry **exponent** palette per 256-block + **1 mantissa bit** → values snap to {1.0, 1.5}×2^exp.
- 16 representable values per block, **exponentially** spaced (e.g. ±{1,1.5,2,3,4,6,8,12}×2^e).
- Q4_K: 16 **linearly** spaced levels per 32-block via affine scale+min. For ~Gaussian weights,
  linear spacing concentrates resolution near zero where the mass is; exponential spacing wastes it
  on the tail. The single mantissa bit (50% gaps) is the dominant error source.

## Proposed experiment A — rebalance bits: 1 exp-idx + 2 mantissa
Change nibble to `idx(1b) | sign(1b) | mant_top2(2b)`:
- Per-block palette: k-means **k=2** exponents (keep 4-byte block-palette storage, use [0],[1], so
  all `ws_start = bpal_start + (nw/256)*4` offset math is unchanged — minimal kernel churn).
- mant = `(w>>5)&3` placed at output bits 6:5; stochastic-round `drop_bits` 6→5.
- Representable: 2 exps × {1.0,1.25,1.5,1.75} × ± = 16 values spanning ~1.8 octaves, **4× finer
  mantissa**. Hypothesis: wins on tight distributions (ffn_down / residual feeders, gate/up),
  may lose on wide (attention) — but SCLP4 is only used on FFN where distributions are tighter.

Files to change (encoder + decoders must stay bit-exact):
- `src/llama-sclp.cpp`: `_encode_4b` block-palette k=2; nibble layout; drop_bits.
- `ggml/src/ggml-cuda/sclp_bridge.cuh`: `sclp4_decode_blob_kernel` (two-pass, used by PPL/prefill),
  `sclp4_fused_gemv_kernel` (M=1 dense tg), and for MoE: `sclp4_fused_moe_gemv_kernel`,
  `sclp4_fused_moe_wmma_kernel`, `sclp4_fused_moe_scalar_kernel`, `sclp4_moe_sidecar_correct*`.
- `ggml/src/ggml-cpu/ggml-cpu.c`: `sclp_decode_to_bf16_cpu` SCLP4 branch.
- Validate dense path first (Llama-3-8B: encoder + two-pass decode + fused gemv + CPU), then MoE.

Format-breaking: old SCLP4 GGUFs incompatible (already the policy). Verify via generation coherence
+ short PPL before propagating to MoE kernels.

## Proposed experiment B — close the prefill throughput gap
SCLP4 dense prefill two-pass decodes the full weight matrix to BF16 every mul_mat. Options:
- Profile whether decode or the BF16 GEMM dominates (SCLP_TIME_TWOPASS=1 exists for MoE; add for dense).
- A fused decode+GEMM (WMMA) for dense small-M, mirroring the MoE WMMA scaffold.
- Or accept two-pass but ensure the decode kernel isn't the bottleneck (vectorize like the MoE decode
  improvements that nearly doubled Gemma4 prefill).

## Recommended order
A first (quality is the bigger differentiator and the user's stated bar is "match/exceed"), with a
faster eval loop (10 chunks). Then B. Both are format/kernel changes — worth a review checkpoint
before the MoE-kernel propagation since that's the high-churn, hard-to-verify part.

---

## Experiment results (2026-05-24, autonomous session)

### Experiment A (rebalance to 1 exp-idx + 2 mantissa bits) — REJECTED
Reducing the per-block palette to k=2 exponents exploded the sidecar: the encoder's
`distance > 1 -> mandatory sidecar` rule sent most weights to verbatim BF16. Llama-3-8B SCLP4
went 4.86 GB -> 10.7 GB (5.19 -> 11.21 bpw). **The 4-entry exponent palette is necessary for
exponent coverage in a 256-block; the bit budget cannot be traded toward mantissa.** Reverted.

### Experiment C (optimal per-weight (idx,mant) encoding) — PROMISING
Pure encoder change, NO format/kernel change, identical file size. Instead of nearest-exponent +
top-mantissa-bit, pick the (palette_idx, mantissa) pair minimizing |reconstruct - original|.
A neighbouring palette exponent often reconstructs closer (1.8*2^e -> 1.0*2^(e+1) beats 1.5*2^e).
Llama-3-8B generation went from degenerate gibberish ("Paris :.?, and... (s)...") to coherent
English ("Paris. France's largest city is located at 14.33 miles to the northwest..."), same
4.86 GB / 31 tg t/s. PPL measurement in progress. This is the lead candidate — provably >= nearest
per weight, zero format risk, and should also help SCLP6/8 (their encoder uses nearest too).

### Experiment C RESULT — LANDED (commit ec890de5a)
PPL (Llama-3-8B wikitext, -c 64 -fa off, 12 chunks; short-context so absolute values inflated,
use the ratios):
| Model | PPL |
|---|---|
| SCLP4 nearest (old) | 220.9 +/- 58 |
| SCLP4 optimal-enc (new) | **64.0 +/- 16** |
| Q4_K_M | 40.8 +/- 10 |
Optimal encoding = **3.4x PPL improvement**, gap to Q4_K cut from 5.4x to 1.6x. Encoder-only,
no format change. Committed. Next lever to fully match Q4_K: still ~1.6x behind — candidates are
the dense two-pass prefill speed (also fixes the slow/"hung" PPL eval) and/or SCLP5 (5-bit:
idx2|sign|mant2, 4-entry palette) for the extra mantissa bit without exploding sidecar.

NOTE: full -c 512 PPL eval is blocked by the SCLP4 two-pass-prefill slowness amplified by
logits_all (+flash-attn) — see project_sclp_perplexity_hang. -c 64 -fa off is the working (slow)
eval regime. Fixing dense prefill throughput would unblock standard -c 512 eval AND help real prefill.

### SCLP5 (5-bit: idx2|sign1|mant2, 8w/5B) RESULT — works but PARETO-DOMINATED
Implemented full new type (ggml enum/traits, encoder, GPU two-pass decode+sidecar, dense intercept,
supports_op, alloc, loader/gguf compact gates, llama-quantize). Llama-3-8B, -c64 -fa off 12 chunks:
| Config | PPL | BPW |
|---|---|---|
| SCLP4 +1% | 64 | 5.19 |
| SCLP5 +0% | 64 | 5.64 |
| SCLP4 +2% | 55 | 5.61 |
| SCLP5 +1% | 55 | 6.06 |
| Q4_K | 41 | 4.89 |
SCLP5 ties SCLP4-at-higher-budget on PPL but is always LARGER -> dominated. A uniform extra mantissa
bit is less bit-efficient than targeted imatrix-sidecar (which only pays for high-impact weights).
SCLP5 decodes correctly (coherent generation). Verdict: not worth keeping; SCLP4+imatrix is the better
SCLP lever. Whole family still trails Q4_K on in-distribution wikitext (structural: exponential vs
linear quant spacing). Caveat: -c64 PPL variance is high (+/-13-15); size deltas are exact.

### SCLP4 vs SCLP5 runtime (Llama-3-8B, RX 7900 XTX, -fa on, via llama-completion timers)
| Model | Size | pp (480-tok prefill) | tg (127 tok) |
| SCLP4 optenc | 4.86 GB | 1452 t/s | 30.9 t/s |
| SCLP5 | 5.67 GB | 1438 t/s | 14.8 t/s |
tg: SCLP4 is 2.1x faster — SCLP4 has a fused M=1 GEMV; SCLP5 lacks one so every token runs the
two-pass decode (whole weight matrix -> BF16). prefill identical (both two-pass at M>1). Implementing
sclp5_fused_gemv would close the tg gap. NOTE: llama-bench STALLS on SCLP models (498% CPU, never
completes) for both pp and tg, while llama-completion works — llama-bench-specific bug; use completion
timers for SCLP perf.

### SCLP5 fused GEMV (committed ca3344121) — sidecar kills the speed win
Built sclp5_fused_gemv + sclp5_sidecar_correct_gemv. fused-no-sidecar = 34.9 tg (2.4x two-pass) but
WRONG: SCLP5's 4-entry palette sends high-magnitude outliers to sidecar (A/B diff: ffn_up ~0 sidecar
matched to 1e-6, others off by up to 5.8). Required sidecar correction (~1M scattered atomicAdds/mul_mat)
→ 13.3 tg, slower than two-pass (14.9). Two-pass stays default; fused behind SCLP5_FUSED_GEMV=1.
Net: SCLP5 has NO tg advantage (still ~2x slower than SCLP4's sidecar-free fused gemv) AND is
quality-dominated by SCLP4+imatrix. Same sidecar tension that disabled SCLP6 fused GEMV.

### Folded sidecar applied to ALL fused GEMVs (commits 641a3ae1e, ab22e4bbd, 4aa56016d)
General fix for the recurring sidecar-vs-fused-GEMV perf problem: encoder sorts sidecar by index
(gidx=row*K+col → each row's outliers are a contiguous range); fused GEMV binary-searches the row's
range and folds (true-approx)*x[col] using smem x — no atomics, no 2nd kernel. Llama-3-8B tg (coherent):
| Type | tg t/s | notes |
| SCLP4 | 26.8 | was sidecar-less in tg (quality win) |
| SCLP5 | 28.9 | was 14.8 two-pass / 13.3 atomic-sidecar |
| SCLP6 | 34.2 | fused GEMV RE-ENABLED (was if(false)) |
| SCLP8 | ~30  | now applies sidecar |
Prefill (M>1) untouched: two-pass already scatter-applies sidecar before rocBLAS. Folded pattern is
M=1-only. Design Q answered: prefill does NOT need folding (materialize+scatter is correct & faster).
OPEN: SCLP4 M=480 two-pass PREFILL stalls (SCLP5/6/8 prefill fast) — separate pre-existing decode-kernel
issue, task #6.
