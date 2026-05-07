# Adaptive Lossless-First LLM Weight Compression

## System Design Summary
*Primary Target: Consumer AMD RDNA3/4 GPUs | Secondary: NVIDIA Ada, Apple M-series*

---

## 1. Motivation and Design Goals

Standard quantization (INT4/INT8) reduces model size but introduces unpredictable quality loss and alters model behaviour in ways existing evaluation metrics may not capture. The goal of this system is to achieve significant memory footprint reduction on consumer GPUs while keeping quality loss small, bounded, and predictable.

The system combines three ideas into a unified pipeline:

- Mild lossy preprocessing (soft exponent clipping) to increase the compressibility of the weight distribution
- Fixed-width lossless palette compression applied on top, exploiting the amplified skew in the exponent distribution
- Self-speculative decoding to mitigate the small-batch-size overhead inherent to decode-fused inference kernels

This is deliberately not a quantization system. The loss is introduced only to improve lossless compressibility, not as a primary size reduction mechanism. The total quality loss is expected to be substantially smaller than INT8 quantization.

---

## 2. Background: Why BF16 Weights Are Compressible

Every LLM weight is stored as a BF16 value comprising three fields:

| Field | Description |
|---|---|
| Sign (1 bit) | Positive or negative. Near-random distribution across weights — incompressible. |
| Exponent (8 bits) | Encodes magnitude. Highly skewed: top 16 values cover ~99% of weights in a typical layer. Low entropy (~2.6 bits effective). Primary compression target. |
| Mantissa (7 bits) | Precise value within the magnitude. Effectively random — lossless compression yields near-zero gain. Mild lossy truncation is the only lever here. |

The exponent's skewed distribution is the key insight shared by ZipServ (arxiv:2603.17435) and Cloudflare's Unweight system. Both exploit it differently. This system builds on ZipServ's approach, extended with selective compression and a lossy preprocessing stage.

---

## 3. Relationship to Prior Work

| System | Approach | Relevance to This Design |
|---|---|---|
| ZipServ (AS_PLOS '26) | Fixed-length TCA-TBE bitmap encoding, fused ZipGEMM kernel. Targets consumer GDPR GPUs. ~30% compression, up to 2.21x kernel speedup over cuBLAS. | Direct foundation. TCA-TBE encoding and fused kernel design are the starting point. Extended here with soft clipping, selective compression, and ROCm adaptation. |
| Cloudflare Unweight | Huffman coding on exponents, four adaptive pipelines, autotuner, selective MLP-only compression. Targets H100 HBM exclusively. 13–22% model reduction, 30–40% throughput overhead currently. | Contributes: selective MLP-only compression strategy, autotuner concept, layer pipelining approach. Huffman encoding is NOT adopted — variable-length decoding is incompatible with SIMT on consumer GPUs. |
| MX Formats (OCP) | Shared block exponent normalises weights within blocks of 16–32 values. Reduces per-stored-exponent entropy. | Explicitly rejected for this design. Block normalisation spreads residual exponents more uniformly, increasing per-stored-exponent entropy and making lossless compression less effective. |

---

## 4. Compression Pipeline

The compression pipeline is entirely offline (CPU-side at model load time or as a one-time preprocessing step). The GPU kernel sees only the compressed representation.

### 4.1 Stage 1: Soft Exponent Clipping (Lossy)

The exponent distribution already has low entropy, but contains a long tail of rare values that force the palette to remain large. Soft clipping maps rare exponents to the nearest high-frequency exponent, shrinking the effective palette from ~16 values to ~8 or fewer.

Key design decisions:

- **Soft clipping (not hard clipping):** rare exponents are mapped to the nearest common exponent rather than clamped to a fixed boundary. This bounds the per-weight magnitude error and avoids the systematic bias introduced by hard clipping.
- **Stochastic rounding at the clip boundary:** weights at the clip boundary are rounded randomly rather than deterministically. This preserves expected value and reduces error accumulation across layers.
- **Per-layer tunable clipping strength:** early and late transformer layers tend to have higher weight variance and tolerate clipping less well than middle layers. Clipping aggressiveness is a per-layer parameter set at preprocessing time, not hardcoded.
- **Error is bounded and predictable:** unlike quantization, the error introduced is a known function of the clipping threshold and the weight's original exponent, making quality impact analysable before deployment.

### 4.2 Stage 2: Mantissa Truncation (Lossy, Optional)

Dropping the lowest 2–3 bits of the mantissa makes the mantissa stream more compressible losslessly, and reduces per-weight storage slightly. This stage is optional and should be validated per model before enabling.

- Error per weight is bounded by `2^(exponent) × 2^(-mantissa_bits)`, which is small for typical weight magnitudes after clipping has normalised the exponent distribution.
- Combined with soft exponent clipping, total per-weight error remains small and bounded.
- The mantissa is otherwise left structurally unchanged — no codebook or learned representation is used in v1.

### 4.3 Stage 3: Fixed-Width 4-Bit Palette Encoding (Lossless)

After clipping, the exponent distribution is highly concentrated. A palette of ≤8 (or up to 16) dominant exponent values is built per tensor. Each weight's exponent is replaced by a 4-bit index into this palette.

- **Fixed-width encoding:** every index is exactly 4 bits regardless of value. This is the critical SIMT compatibility property — no variable-length bitstreams, no control-flow divergence, no lane stalls.
- **Palette fits in registers:** 8–16 BF16 values = 16–32 bytes. Easily held in thread registers or a small LDS allocation, avoiding global memory lookups in the hot path.
- Two 4-bit indices are packed per byte in the weight stream.
- Sign and mantissa bytes are stored in a separate stream (or interleaved — layout is a tunable parameter determined by profiling for optimal coalescing on the target GPU).
- Rows containing weights with exponents outside the palette ('outlier rows') are stored verbatim in BF16 as a sidecar. In v1 this is simplified further by soft clipping eliminating most outliers beforehand, potentially removing the need for a sidecar entirely at the cost of a small additional quality trade-off.

**Why not Huffman or ANS?** Both achieve better compression ratios than fixed-width palette encoding, but both produce variable-length output that causes SIMT divergence on GPU wavefronts. On AMD RDNA3 with 64-lane wavefronts, this divergence is more expensive than on NVIDIA. The palette approach sacrifices ~3–5% compression ratio for fully branchless, constant-time decode — the correct trade-off for consumer GPU inference.

### 4.4 Selective Compression (MLP + Attention Weights)

Following Unweight's insight and empirical validation, compression is applied to the following weight matrices:

- **MLP Layers:**
  - Gate projection (W_gate)
  - Up projection (W_up)
  - Down projection (W_down)
- **Attention Layers:**
  - Query projection (W_q)
  - Key projection (W_k)
  - Value projection (W_v)
  - Output projection (W_o)

Rationale: MLP and Attention projections constitute roughly 90% of total model parameters (67.8% in OPT-125m) and dominate memory traffic during token generation. Embeddings and layer norms are left uncompressed due to their higher sensitivity and smaller footprint.

### 4.5 Optional: Combination with INT8 Quantization

The pipeline can be preceded by INT8 quantization. In this case the palette operates on full 8-bit integer values rather than BF16 exponents — the 16 most common INT8 values are stored in the codebook and each weight is replaced by a 4-bit index. The combination makes most sense at INT8; at INT4 the quantization has already consumed most compressible entropy and the palette adds marginal value. Error from quantization and clipping accumulates in the residual stream and must be validated empirically rather than assumed to be additive.

A learned codebook (k-means / vector quantization) is a natural upgrade over the frequency-based palette: k-means minimises reconstruction error directly rather than approximating it via frequency, at the cost of a more expensive offline preprocessing step. The kernel decode logic is identical — a 4-bit index lookup — so the upgrade does not affect kernel complexity.

---

## 5. GPU Inference Kernel

### 5.2 Fused Decode-GEMM (Research Path)

To achieve maximum efficiency on consumer hardware (RDNA3), we are developing a Fused SCLP-GEMM approach.

- **Option A: Transcoder Bridge (Current Baseline):** Transcode compressed weights to BF16 via a highly-optimized vectorized HIP kernel, then pass to `rocBLAS` or a standard GEMM library. This minimizes the compute overhead of decompression.
- **Option B: Fused Tiled-GEMM (Research):** Custom tiled-GEMM kernel where tiles of compressed SCLP data are loaded into Shared Memory (LDS), transcoded to BF16 on-the-fly in registers, and passed to hardware matrix cores (WMMA).

**Research Plan:**
1. Develop a high-speed Transcoder Bridge kernel.
2. If memory bandwidth remains the bottleneck (as suggested by our current 94% throughput efficiency), focus on maximizing bridge performance.
3. If compute becomes the bottleneck, implement the Fused Tiled-GEMM using WMMA.

Per-weight operations inside the kernel (all fixed-width, branchless):

1. Load 4-bit palette index from packed index stream (fixed-width load)
2. Decode exponent via palette lookup → recover clipped exponent (single table lookup, palette in registers)
3. Recombine exponent + sign + mantissa → BF16 value (bitwise OR of packed bytes, ~3–5 ALU instructions)
4. Feed reconstructed BF16 value into MFMA accumulation (RDNA3) or Tensor Core (N/A for consumer NVIDIA)

The reconstruct-then-MFMA sequence is designed so that decode ALU instructions are issued during memory latency of the next tile load, keeping the matrix units fully fed.

### 5.2 Target Architecture Details

| Property | AMD RDNA3 | AMD CDNA3 (MI300X) | NVIDIA Ada (RTX 4090) | Apple M-series |
|---|---|---|---|---|
| Matrix op | WMMA/MFMA 16×16 | MFMA 16×16 | Tensor Core 16×16 | AMX (CPU-side) |
| Wavefront/Warp width | Wave64 (64 lanes) | Wave64 | Warp32 (32 lanes) | N/A |
| LDS / Shared memory | 64 KB per CU | 64 KB per CU | 100 KB per SM | Unified pool |
| Memory type | GDDR6X ~9* GB/s | HBM3 ~5.3 TB/s | GDDR6X ~1008 GB/s | Unified ~400 GB/s |
| Primary bottleneck | Memory BW | Compute | Memory BW | CPU/GPU sharing |

**Note on INT8 compute:** INT8 and BF16 MFMA use the same physical Matrix Core hardware on RDNA3, reconfigured per instruction. INT8 is ~2x faster in raw throughput but the advantage only materialises when the workload is compute-bound. Consumer single-user inference at batch size 1 is memory-bandwidth-bound, so the INT8 throughput advantage is largely irrelevant in that regime. The primary benefit of INT8 in this system is smaller weights (fewer bytes loaded from VRAM), not faster arithmetic.

### 5.3 Known Kernel Bottlenecks

| Bottleneck | Detail and Mitigation |
|---|---|
| LDS pressure | Palette table (32 bytes) + MFMA tile buffers must fit in 64 KB LDS. If combined usage forces lower wavefront occupancy, latency hiding degrades. Mitigation: keep palette in thread registers rather than LDS where possible; profile occupancy and tunable tiles-per-wavefront accordingly. |
| Memory coalescing | 4-bit packed indices require two indices per byte, complicating coalesced loads. Whether index and sign+mantissa streams are interleaved or separate affects coalescing differently depending on tile traversal order. Storage layout must be designed around AMD tile access patterns (MFMA 16×16), not ported from ZipServ tile layout. |
| MFMA tile alignment | ZipServ's TCA-TBE tiling (8×8, 16×64, 64×64) was designed for NVIDIA Tensor Core dimensions. RDNA3 MFMA tiles are 16×16×16. Tiling hierarchy must be redesigned AMD-first. |
| Small batch size overhead | At batch size 1–4, GEMM is too small to amortise fixed decode overhead. Mitigated by autotuner fallback path and speculative decoding. |
| Register pressure | Fusing decode into GEMM increases registers per thread. High register use limits simultaneous wavefront count, reducing latency hiding. |

---

## 6. Small Batch Size Mitigation Strategy

### 6.1 Autotuner with Fallback Path

At load time, a lightweight autotuner measures end-to-end throughput for two execution paths per layer:
- **Path A (fused):** decode-GEMM kernel with palette lookup fused into matrix multiply
- **Path B (deintense-first):** decompress weights once, run native rocBLAS/cuBLAS GEMM on uncompressed data

The crossover batch size is hardware- and layer-specific. The autotuner result is cached as a per-model config file. At inference time, the runtime performs a single lookup to select the correct path per layer.

### 6.2 Speculative Decoding

Self-speculative decoding converts effective batch size 1 into batch size K from the base model's perspective, without requiring a separate smaller model. The base model weights are always frozen — compression applies without retraining interaction.

| Method | Description |
|---|---|
| EAGLE-2 (recommended) | Autoregressive draft network conditioned on base model hidden states. High acceptance rate. |
| Medusa | Multiple small MLP heads attached to base model's final hidden state. Simpler training, lower acceptance. |
| Lookahead decoding | Training-free. Maintains n-gram cache from previous context. Zero extra parameters. |

### 6.3 Layer Pipelining (Decode-Ahead)

While the GPU computes layer N's GEMM, a separate HIP stream decodes layer N+1's weights into a staging buffer. Double-buffering ensures decode output is never overwritten while still being consumed.

---

## 					

## 7. Expected Compression Ratios and Quality

| Metric | Expected Value |
|---|---|
| Exponent compression (lossless only, no clipping) | ~30% on exponent stream |
| Exponent compression (with soft clipping, palette ≤8 values) | ~35–45% on exponent stream |
| Overall model size reduction (MLP weights only, no mantissa truncation) | ~20–30% of total model size |
| Overall model size reduction (MLP weights + mantissa truncation, 2 bits dropped) | ~25–35% of total model size |
| Quality loss vs. baseline BF16 | Expected substantially less than INT8 quantization. Bounded and predictable. |

---

## 8. Key Risks and Open Questions

- **Clipping Sensitivity:** Per-layer clipping sensitivity is theoretically bounded but empirically unknown for specific models.
- **RDNA3 MFMA occupancy:** The combined LDS requirements of the palette buffer and MFMA tile buffers may force lower wavefront occupancy than expected.
- **Speculative decoding acceptance rate variability:** Effectiveness depends on the match between draft network and base model distribution.
- **llama.cpp integration complexity:** Adding a new quant type that only benefits CUDA/ROCm paths might face resistance from llama.cpp maintainers who prioritise cross-backend simplicity.

---

## 9. Key Dependencies and References

| Item | Detail |
|---|---|
| ZipServ source | github.com/scitix/ZipServ_BF16 — Primary algorithmic reference (ASPLOS '26). |
| Unweight kernels | github.com/cloudflareresearch/unweight-kernels — Reference for autotuner design and selective compression strategy. |
| rocWMMA | AMD's C++ interface for RDNA/CDNA matrix operations. Primary API for MFMA tile operations in the fused kernel. |
| llama.cpp | Target deployment framework (GGUF format, ROCm/HIP backend). |

---

*End of summary document.*
