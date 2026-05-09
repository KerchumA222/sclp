#include <hip/hip_runtime.h>
#include <cstdint>

// Each thread encodes TWO weights into one packed byte and one SM byte.
// packed byte: idx0(high nibble) | idx1(low nibble)
// SM byte:     smn0(high nibble) | smn1(low nibble)
//   where smn = sign(3) | mantissa_top3(2:0)  — top 3 of 7 mantissa bits
// Output pointers each receive ceil(num_weights/2) bytes.
__global__ void encode_palette_kernel(const uint16_t* input, const uint8_t* lookup, uint8_t* packed, uint8_t* sm, uint32_t num_weights) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (num_weights + 1) / 2) return;

    auto encode_weight = [&](uint16_t w, uint8_t& out_idx, uint8_t& out_smn) {
        uint8_t exp      = static_cast<uint8_t>((w >> 7) & 0xFF);
        out_idx          = lookup[exp] & 0x0F;
        uint8_t sign     = static_cast<uint8_t>((w >> 15) & 0x1);
        uint8_t mant_top3 = static_cast<uint8_t>((w >> 4) & 0x7);  // bits 6:4
        out_smn          = static_cast<uint8_t>((sign << 3) | mant_top3);
    };

    uint32_t i0 = idx * 2;
    uint8_t idx0, smn0, idx1 = 0, smn1 = 0;
    encode_weight(input[i0], idx0, smn0);
    if (i0 + 1 < num_weights)
        encode_weight(input[i0 + 1], idx1, smn1);

    packed[idx] = static_cast<uint8_t>((idx0 << 4) | (idx1 & 0x0F));
    sm[idx]     = static_cast<uint8_t>((smn0 << 4) | (smn1 & 0x0F));
}
