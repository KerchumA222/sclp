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

### 4.3 Stage 3: Fixed-Width 8-Bit Interleaved Encoding (Lossless-First)

After clipping, the exponent distribution is highly concentrated. A palette of ≤16 dominant exponent values is built per tensor. Each weight is encoded into a single 8-bit byte:

- **High nibble (4 bits)**: Index into the 16-entry exponent palette.
- **Low nibble (4 bits)**: Sign bit (1 bit) + Top 3 mantissa bits (3 bits).

Design highlights:
- **Fixed-width encoding:** every weight is exactly 8 bits. This is the critical SIMT compatibility property — no variable-length bitstreams, no control-flow divergence, no lane stalls.
- **Palette fits in registers:** 16 BF16 values = 32 bytes. Easily held in thread registers or a small LDS allocation, avoiding global memory lookups in the hot path.
- **Interleaved layout:** Palette index and sign/mantissa bits are co-located in the same byte. This halves L2 cache pressure and improves memory coalescing compared to separate index/SM streams.
- **Sidecar rescue:** Weights with exponents outside the top-16 palette are stored verbatim in a sidecar section. A grid-parallel fixup kernel restores them exactly, making the scheme functionally lossless for the most important outliers.

**Why 8-bit?** 8-bit (2.0×) is the "sweet spot" for RDNA3 hardware. It allows for simple byte-aligned loads and maps cleanly to 16x16 matrix operations. Lower bit-widths (SCLP4/SCLP6) are supported via bit-packing but introduce additional unpacking overhead.

### 4.4 Stage 4: SCLP5 (Draft) — Interleaved Bit-Planes

To bridge the performance gap between SCLP4 (~1200 t/s) and SCLP6 (~1400 t/s) while supporting 32-entry palettes (reducing MoE sidecars), a 5-bit **Interleaved Bit-Plane** strategy is in draft. This format packs 32 weights into 20-byte blocks, allowing for coalesced 128-bit loads and efficient Bit Field Extract (BFE) decoding on RDNA3.

---

## 5. Mixture of Experts (MoE) Support

SCLP provides tiered support for MoE layers (e.g., DeepSeek, Mixtral):

- **All types (SCLP4/6/8)**: Full **per-expert palette** support. Each expert weight matrix is encoded with its own optimized exponent palette. All types share the same blob header: `[num_weights (4B)][n_experts (4B)][per-expert: palette_size, palette bytes]...[ws_stream][sidecar]`.

### 5.1 Selective Compression (MLP + Attention Weights)

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

### 5.1 Transcoder Bridge (Implemented — llama.cpp Integration)

The Transcoder Bridge approach is implemented and working end-to-end in the llama.cpp `sclp` branch on RX 7900 XTX (gfx1100). All 224 linear projections of Llama 3 8B are compressed. Observed: ~16 t/s compressed / ~52 t/s FP16 baseline. The 3x overhead is expected for the bridge (extra decode + memory pass per matmul) and will be eliminated by the fused decode-GEMM path.

**Architecture:**
- `sclp_bridge.cuh` contains two on-device kernels and a dispatch function:
  - `sclp_decode_blob_kernel`: reads blob header on-device (palette_size at `blob[4]`, palette at `blob[5..]`), decodes packed indices + sm_stream to BF16, writes to a pool-allocated output buffer. All fixed-width, branchless.
  - `sclp_fixup_sidecar_kernel`: reads `sidecar_count` from `blob[sm_end..]` on-device, then scatter-writes each outlier weight's exact BF16 value into the output buffer. Grid-stride loop with 4 fixed blocks handles any sidecar count without a D2H read.
  - `llama_sclp_dispatch`: sizes decode grid proportional to `num_weights`, launches fixup with 4 fixed blocks.
- The dispatch intercepts at the top of `ggml_cuda_mul_mat`: when `src0->type == GGML_TYPE_SCLP`, the blob is decoded into a pool-allocated `uint16_t` buffer, the tensor type is patched to `GGML_TYPE_BF16` in a stack copy, and the function recurses through the standard BF16 matmul path (rocBLAS / hipBLAS).
- Because `GGML_TYPE_SCLP` has `type_size=2` (same as BF16), all tensor strides are correct for BF16 without adjustment.

Per-weight decode operations inside `sclp_decode_blob_kernel` (fixed-width, branchless):
1. Load 4-bit palette index from packed index stream
2. Decode exponent via shared-memory palette lookup (palette cached in `__shared__ uint8_t s_palette[16]`)
3. Reconstruct BF16: `((sign >> 7) << 15) | (exp << 7) | (mantissa & 0x7F)`
4. (After main decode) Sidecar fixup restores ~0.012% of weights that had exponents outside top-16 palette

**HIP graph capture constraint**: All header parsing must happen on-device. `hipMemcpyAsync D2H` and `hipStreamSynchronize` are forbidden during HIP graph stream capture (`GGML_HIP_GRAPHS=ON`) and cause `ROCm error: operation failed due to a previous error during capture`. Both kernels use shared-memory reads for all blob metadata — no host-side device reads, graph-safe.

### 5.2 Fused Decode-GEMV/GEMM (Implemented)

To achieve maximum efficiency on consumer hardware (RDNA3), SCLP utilizes fused kernels that eliminate intermediate BF16 allocations:

- **Fused GEMV (M=1)**: `sclp_fused_gemv_kernel` transcodes SCLP weights directly into registers during the dot-product reduction. It uses an LDS-backed LUT for the palette to minimize ALU cost.
- **Fused WMMA (M>1)**: `sclp_fused_wmma_kernel` (and its scalar equivalent) decodes SCLP weights on-the-fly and feeds them directly into RDNA3 WMMA matrix instructions. This eliminates the "double trip" to memory and the large scratch buffer required by the two-pass approach.

These kernels are designed so that decode ALU instructions are issued during memory latency of the next tile load, keeping the matrix units fully fed.

### 5.3 Target Architecture Details (RDNA3 Primary)

| Property | AMD RDNA3 | AMD CDNA3 (MI300X) | NVIDIA Ada (RTX 4090) | Apple M-series |
|---|---|---|---|---|
| Matrix op | WMMA/MFMA 16×16 | MFMA 16×16 | Tensor Core 16×16 | AMX (CPU-side) |
| Wavefront/Warp width | Wave64 (64 lanes) | Wave64 | Warp32 (32 lanes) | N/A |
| LDS / Shared memory | 64 KB per CU | 64 KB per CU | 100 KB per SM | Unified pool |
| Memory type | GDDR6X ~9* GB/s | HBM3 ~5.3 TB/s | GDDR6X ~1008 GB/s | Unified ~400 GB/s |
| Primary bottleneck | Memory BW | Compute | Memory BW | CPU/GPU sharing |

**Note on INT8 compute:** INT8 and BF16 MFMA use the same physical Matrix Core hardware on RDNA3, reconfigured per instruction. INT8 is ~2x faster in raw throughput but the advantage only materialises when the workload is compute-bound. Consumer single-user inference at batch size 1 is memory-bandwidth-bound, so the INT8 throughput advantage is largely irrelevant in that regime. The primary benefit of INT8 in this system is smaller weights (fewer bytes loaded from VRAM), not faster arithmetic.

### 5.4 Known Kernel Bottlenecks

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
| Exponent compression (SCLP8, interleaved) | 50% (2.0x) on streams |
| Exponent compression (with soft clipping, SCLP4) | 75% (4.0x) on streams |
| Overall model size reduction (Llama-3-8B, SCLP8) | ~43% reduction (14.97 -> 8.47 GB) |
| Quality loss vs. baseline BF16 | Expected substantially less than INT8 quantization. Bounded and predictable. |

---

## 8. Key Risks and Open Questions

- **Clipping Sensitivity:** Per-layer clipping sensitivity is theoretically bounded but empirically unknown for specific models.
- **RDNA3 MFMA occupancy:** The combined LDS requirements of the palette buffer and MFMA tile buffers may force lower wavefront occupancy than expected.
- **Speculative decoding acceptance rate variability:** Effectiveness depends on the match between draft network and base model distribution.
- ~~**llama.cpp integration complexity:** Adding a new quant type that only benefits CUDA/ROCm paths might face resistance from llama.cpp maintainers who prioritise cross-backend simplicity.~~ **Resolved.** SCLP type registered as `GGML_TYPE_SCLP = 42`, bridge wired into `ggml-cuda.cu`, inference verified end-to-end on RX 7900 XTX. Key lessons: (1) `supports_op` switch must include SCLP for MUL_MAT or model load crashes; (2) HIP graph capture forbids any D2H memcpy — kernel must be fully self-parsing on-device.
- ~~**GGUF blob format lacks sidecar:**~~ **Resolved.** Sidecar is now embedded after the sm_stream in the GGUF blob (`[uint32 sidecar_count][uint32[] indices][uint16[] values]`). The GPU kernel (`sclp_fixup_sidecar_kernel`) restores outlier weights exactly via on-device scatter writes. Observed sidecar fraction: 0.012% of weights across Llama 3 8B. Compression is now fully lossless. Key lesson: typical BF16 LLM tensors have 17–25 unique exponents; the 1–9 per-tensor outlier exponents would cause cumulative quality degradation (up to 64× weight error per exponent step) that collapses model output — sidecar is required for correctness, not just precision.

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
