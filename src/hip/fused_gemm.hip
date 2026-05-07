#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

// Fused decode-GEMV: decodes SCLP blob weights on-the-fly while accumulating
// the dot product.  One warp computes one output row; palette lives in shared
// memory (16 bytes) so the 16-entry lookup costs a single shared-mem read.
//
// Sidecar weights (0.01–0.03% of all weights, near-zero exponents) are skipped
// in this path: the palette nearest-neighbour substitution introduces negligible
// error for outliers that are already close to zero.  Use the two-pass path
// (sclp_bridge.cuh) when exact losslessness is required.
//
// Input layout:
//   blob   — SCLP blob, device pointer (see CLAUDE.md wire format)
//   x      — activation vector [K], BF16
//   y      — output vector [N], F32
//   N, K   — logical matrix dimensions (weight matrix is N×K, row-major)
// Optimizations applied:
//   1. Vectorized loads: 8 sm bytes (uint64_t) + 4 packed bytes (uint32_t) per iter.
//   2. Dual accumulators for ILP — each lane processes two independent 8-weight chunks.
//   3. __launch_bounds__(512, 4) for occupancy hint on gfx1100 (RDNA3).
__launch_bounds__(512, 4)
__global__ void sclp_fused_gemv_kernel(
    const uint8_t*        __restrict__ blob,
    const __hip_bfloat16* __restrict__ x,
    float*                __restrict__ y,
    uint32_t N,
    uint32_t K
) {
    __shared__ uint8_t s_palette_size;
    __shared__ uint8_t s_palette[16];

    if (threadIdx.x == 0) {
        s_palette_size = blob[4];
    }
    __syncthreads();

    if (threadIdx.x < (uint32_t)s_palette_size) {
        s_palette[threadIdx.x] = blob[5 + threadIdx.x];
    }
    __syncthreads();

    // Derive compressed stream base pointers from blob header.
    const uint8_t* packed = blob + 5 + s_palette_size;
    const uint8_t* sm     = packed + ((uint64_t)(N * K + 1) / 2);

    const int warps_per_block = blockDim.x / 32;
    const int warp_id  = threadIdx.x / 32;
    const int lane     = threadIdx.x & 31;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp_id;

    if (row >= N) return;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const uint64_t row_base = (uint64_t)row * K;

    // --- Vectorized loop: process 16 weights per iteration (2×8), dual accumulator ---
    const uint32_t K16 = (K / 16) * 16;

    for (uint32_t k16 = lane * 16; k16 < K16; k16 += 32 * 16) {
        // Chunk A: weights [k16 .. k16+7]
        {
            uint64_t w_base = row_base + k16;
            uint32_t p4; __builtin_memcpy(&p4, packed + (w_base >> 1), sizeof(uint32_t));
            uint64_t sm8; __builtin_memcpy(&sm8, sm + w_base, sizeof(uint64_t));
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                uint8_t pb    = (uint8_t)(p4  >> ((j >> 1) * 8));
                uint8_t p_idx = (j & 1) ? (pb & 0x0F) : (pb >> 4);
                uint8_t sm_val = (uint8_t)(sm8 >> (j * 8));
                uint16_t bits = ((uint16_t)(sm_val >> 7) << 15)
                              | ((uint16_t)s_palette[p_idx] << 7)
                              | (sm_val & 0x7F);
                acc0 += __bfloat162float(*(__hip_bfloat16*)&bits)
                      * __bfloat162float(x[k16 + j]);
            }
        }
        // Chunk B: weights [k16+8 .. k16+15]
        {
            uint64_t w_base = row_base + k16 + 8;
            uint32_t p4; __builtin_memcpy(&p4, packed + (w_base >> 1), sizeof(uint32_t));
            uint64_t sm8; __builtin_memcpy(&sm8, sm + w_base, sizeof(uint64_t));
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                uint8_t pb    = (uint8_t)(p4  >> ((j >> 1) * 8));
                uint8_t p_idx = (j & 1) ? (pb & 0x0F) : (pb >> 4);
                uint8_t sm_val = (uint8_t)(sm8 >> (j * 8));
                uint16_t bits = ((uint16_t)(sm_val >> 7) << 15)
                              | ((uint16_t)s_palette[p_idx] << 7)
                              | (sm_val & 0x7F);
                acc1 += __bfloat162float(*(__hip_bfloat16*)&bits)
                      * __bfloat162float(x[k16 + 8 + j]);
            }
        }
    }

    float acc = acc0 + acc1;

    // --- Scalar tail ---
    for (uint32_t k = K16 + lane; k < K; k += 32) {
        uint64_t w_idx = row_base + k;

        uint8_t packed_byte = packed[w_idx >> 1];
        uint8_t p_idx  = (w_idx & 1) ? (packed_byte & 0x0F) : (packed_byte >> 4);
        uint8_t sm_val = sm[w_idx];

        uint16_t bits = ((uint16_t)(sm_val >> 7) << 15)
                      | ((uint16_t)s_palette[p_idx] << 7)
                      | (sm_val & 0x7F);

        acc += __bfloat162float(*(__hip_bfloat16*)&bits)
             * __bfloat162float(x[k]);
    }

    // Warp-level reduction across the 32 lanes.
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset);
    }

    if (lane == 0) {
        y[row] = acc;
    }
}
