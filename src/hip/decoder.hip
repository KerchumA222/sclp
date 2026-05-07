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

    // Each thread processes a pair of weights (one packed byte)
    uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t base_idx = pair_idx << 1;
    
    if (base_idx >= num_weights) return;

    // 2. Read one byte of packed indices (2 weights)
    uint8_t packed_byte = packed_indices[pair_idx];
    uint8_t exp0 = s_palette[packed_byte >> 4];
    uint8_t exp1 = s_palette[packed_byte & 0x0F];

    // 3. Process first weight
    uint8_t sm0 = sm_stream[base_idx];
    output[base_idx] = ((uint16_t)(sm0 >> 7) << 15) | ((uint16_t)exp0 << 7) | (sm0 & 0x7F);

    // 4. Process second weight if it exists
    if (base_idx + 1 < num_weights) {
        uint8_t sm1 = sm_stream[base_idx + 1];
        output[base_idx + 1] = ((uint16_t)(sm1 >> 7) << 15) | ((uint16_t)exp1 << 7) | (sm1 & 0x7F);
    }
}
