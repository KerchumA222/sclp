#include <hip/hip_runtime.h>
#include <cstdint>

__global__ void decode_palette_kernel(
    const uint8_t* __restrict__ packed_indices,
    const uint8_t* __restrict__ sm_stream,
    const uint8_t* __restrict__ palette,
    uint16_t* __restrict__ output,
    uint32_t num_weights
) {
    // 1. Load palette into LDS
    __shared__ uint8_t s_palette[16];
    if (threadIdx.x < 16) {
        s_palette[threadIdx.x] = palette[threadIdx.x];
    }
    __syncthreads();

    // Each thread processes a pair of weights (one packed byte + one SM byte).
    // SM byte: smn0(high nibble) | smn1(low nibble), smn = sign(3) | mantissa_top3(2:0)
    uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t base_idx = pair_idx << 1;

    if (base_idx >= num_weights) return;

    uint8_t packed_byte = packed_indices[pair_idx];
    uint8_t sm_byte     = sm_stream[pair_idx];
    uint8_t exp0 = s_palette[packed_byte >> 4];
    uint8_t exp1 = s_palette[packed_byte & 0x0F];
    uint8_t smn0 = sm_byte >> 4;
    uint8_t smn1 = sm_byte & 0x0F;

    // smn = sign(3) | mant_top3(2:0) → BF16: sign(15) | exp(14:7) | mant_top3(6:4) | 0000
    output[base_idx] = ((uint16_t)(smn0 >> 3) << 15) | ((uint16_t)exp0 << 7) | ((uint16_t)(smn0 & 0x7) << 4);

    if (base_idx + 1 < num_weights)
        output[base_idx + 1] = ((uint16_t)(smn1 >> 3) << 15) | ((uint16_t)exp1 << 7) | ((uint16_t)(smn1 & 0x7) << 4);
}
