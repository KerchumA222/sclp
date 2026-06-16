# SCLP Algorithm Review & Recommendations (2026-06-10)

Review of design.md, the Python reference (`src/compression/`), and the production
encoder (`$LLAMA_CPP/src/llama-sclp.cpp`). Recommendations ordered by expected
quality-per-effort.

## Overall assessment

Sound architecture with several genuinely good pieces: the lossless sidecar
(verbatim BF16 rescue) is the right answer to exponent outliers, the fixed-width
byte-aligned stream is the right SIMT call, the per-block palette for SCLP4
correctly diagnoses why PBS degenerates in log domain, and the sorted-sidecar fold
into the fused GEMV is elegant.

The structural weakness is at the low-bit end: constraining codewords to
`{palette exponent} × {1 or 2 mantissa levels}` is a log-domain grid, and at
4 bits/weight a log grid is a poor fit to roughly-Gaussian weights compared to
Q4_K's affine grid — that, not implementation detail, is the main source of the
SCLP4 ≈ 102 vs Q4_K ≈ 10 PPL gap on Llama-3-8B. Meanwhile the perf value
proposition has eroded (SCLP8 tg fell 66 → ~30 t/s with folded sidecar, below
BF16's 52), so SCLP's defensible niche today is MoE + OOD robustness.

## Quality wins, encoder-only (no format change; regenerate GGUFs)

### 1. Sidecar priority = importance × actual squared error  ✅ IMPLEMENTED 2026-06-10
Replace `importance × exponent_distance` with `importance × err²` (matching the
output-MSE contribution `E[x²]·Δw²`).

Problems with the old priority (`llama-sclp.cpp` per-block ~404, global ~460):
- Exponent distance is log-domain error multiplied by linear-domain importance —
  a dist-1 miss at exponent 130 is orders of magnitude more absolute error than
  at exponent 100, but ranked equally.
- Distance-0 weights were excluded entirely, yet at SCLP4 an in-palette weight
  still carries up to ~25% relative error from the 1-mantissa-bit grid. The
  justifying comment in `encoder.py` ("mantissa truncation, which sidecar can't
  avoid") was factually wrong: the sidecar stores verbatim BF16 and restores the
  full mantissa.

Implementation: per-block path reuses `best_err` from the joint search (was
computed and discarded); global path reconstructs the decoder output (palette
exponent + copied mantissa bits, × PBS block scale) and ranks by
`imatrix[col] × err²`. Mandatory tier (`distance > 1`) unchanged. Python
reference synced. **TODO: re-sweep the budget knee (0.5%/1%/2%) — it may move.**

### 2. Port joint (idx, mantissa) min-error search to SCLP6/SCLP8
The per-block path exhaustively picks the best codeword
(`llama-sclp.cpp:381-395`; `1.8·2^e → 1.0·2^(e+1)` beats nearest-exponent), but
the global path still does nearest-exponent + copied mantissa bits. SCLP6 has
8 × 4 = 32 candidates per weight — trivial offline cost — and SCLP6 attention
tensors are precisely where 1–5% of weights land on mapped exponents. Compare
against the PBS-scaled value, since that's what the decoder reconstructs.

### 3. A/B round-to-nearest vs stochastic rounding (SCLP6/8 mantissas)
Stochastic rounding (`llama-sclp.cpp:59`) is unbiased but has 2× the MSE of
round-to-nearest; its advantage applies to *accumulating* training gradients,
not static inference weights. SR is largely a no-op on the SCLP4/5 path (the
joint search overrides the mantissa), so this is really about SCLP6/8. Today's
env toggle only offers SR vs biased truncation — add an RTN mode and A/B it.

### 4. Align palette-selection objective with assignment objective
Per-block palettes are chosen by count-weighted k-means in *exponent* space, but
codes are assigned by *linear* reconstruction error. With ~10–20 unique
exponents per 256-weight block, score candidate palettes by the actual
quantity: `Σ min over codewords (w − r)²` (computable from the per-exponent
histogram), or run the k-means in linear magnitude space.

## Format-level (bigger wins; breaking, regenerate GGUFs)

### 5. Free per-block magnitude codebook — structural fix for the Q4_K gap
Generalize SCLP4's "4 exponents × 2 mantissa levels" (8 constrained magnitudes)
to 8 *arbitrary* BF16 magnitudes per 256-block: 3-bit index + 1 sign bit,
codebook = 16 B/block = 0.5 bpw overhead → 4.5 bpw, exactly Q4_K's budget.
A Lloyd-Max (k-means) scalar codebook is at least as good as any affine grid at
equal rate, so this should match or beat Q4_K quality while keeping everything
that makes SCLP worth having: fixed-width LUT decode (kernel shape unchanged),
the sidecar, and the fused per-expert MoE GEMV that Q4_K lacks. design.md §4.4
already anticipates this. Given per-block SCLP4 already beats Q4_K 2.2× on
Gemma4 OOD *despite* the constrained grid, this is the highest-ceiling change.

### 6. Compress sidecar indices: per-row offset table + u16 columns
Sidecar entries cost 6 B each (u32 flat index + u16 value); at a 1% budget
that's ~0.5 bpw — 12% of SCLP4's stream — and the OOD table attributes almost
all of the global-palette build's +2.1 GiB to sidecar. Entries are already
sorted and `gidx = row·K + col`, so store a per-row offset table plus u16 column
per entry: 4 B/entry, −33%. Bonus: the fused GEMV's per-warp binary search
becomes a direct row-range lookup. Same budget → 1.5× the rescued weights, or
same quality at smaller size.

## Performance

### 7. Profile the folded-sidecar GEMV regression (66 → ~30 t/s on SCLP8)
The sidecar is ~0.02% of weights on SCLP8 — three orders of magnitude less data
than the main stream cannot legitimately cost 2× throughput. Suspects:
divergence around the binary search, occupancy drop from extra registers/smem,
or the correction path serializing the reduction. At 30 t/s SCLP8 is slower
than BF16 and the dense value prop is negative. Item 6's row-offset table is
one candidate fix.

## Hygiene

- **Deprecate soft clipping (stage 1).** Off by default, superseded by the
  sidecar (lossless rescue strictly dominates lossy clipping of the same
  outliers), implicated in earlier IT-MoE bugs, and the implementation
  (hard-clip everything above threshold+1) doesn't match the design doc's "map
  rare exponents to nearest *common* exponent."
- **Latent crash in Python k-means** (`encoder.py` dead-cluster branch): builds
  a set of unhashable numpy arrays → TypeError the first time a cluster dies.
- **Python reference drift:** truncates mantissas (no SR), lacks the joint
  search, silently drops trailing weights when `num_weights % n_experts != 0`.
  Either sync it or demote it explicitly to format documentation.
- **design.md is stale:** §3 rejects MX-style block normalization but PBS
  shipped; §8 still claims 66 t/s / "27% faster"; §4.2 calls mantissa truncation
  optional when it's inherent to every wire format.
- **Eval gate:** make KL-divergence vs BF16 logits
  (`llama-perplexity --kl-divergence`) the primary metric alongside the
  200-token chat smoke test. Wikitext PPL is OOD-inflated ~50× on Gemma4-IT and
  PPL alone missed a mode collapse once already.
