#include <hip/hip_runtime.h>
#include <cstdint>

__global__ void decode_palette_kernel(
    const uint8_t* __restrict__ packed_indices,
    const uint8_t* __restrict__ sm_stream,
    const uint8_t* __restrict__ palette,
    uint16_t* __restrict__ output,
    uint32_t num_weights
) {
    // Each thread processes one weight (one reconstructed BF16 value). 
    // Since each packed index byte contains TWO weights, the number of threads equals num_weights.
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_weights) return;

    // 1. Extract the 4-bit index from the packed_indices stream
    // Each byte contains two 4-bit indices: [high nibble | low nibble]
    uint32_t byte_idx = idx / 2;
    uint8_t packed_byte = packed_indices[byte_idx];
    
    uint8_t index_in_byte = idx % 2; // 0 for high nibble, 1 for low
    uint8_t palette_idx = (index_in_byte == 0) ? (packed_byte >> 4) : (packed_byte & 0x0F);

    // 2. Get the exponent from the palette (the shared scale factor for this index)
    uint8_t exponent = palette[palette_idx];

    // 3. Extract sign and mantissa from the sm_stream byte
    // The SM stream stores one byte per weight: [1-bit Sign | 7-bit Mantissa]
    uint8_t sm_val = sm_stream[idx];
    uint8_t sign = (sm_val >> 7) & 0x01; // Extract highest bit as sign
    uint8_t mantissa = sm_val & 0x7F;     // Extract remaining 7 bits as mantissa

    // 4. Reconstruct the BF16 pattern: [Sign (15)] [Exponent (14-7)] [Mantissa (6-0)]
    // | S | EEEEEEEE | MMMMMMM |  (total 1+8+7 = 16 bits)
    uint16_t reconstructed = static_cast<uint16_t>((static_cast<uint16_t>(sign) << 15) | (static_cast<uint16_t>(exponent) << 7) | mantissa);

    output[idx] = reconstructed;
}
