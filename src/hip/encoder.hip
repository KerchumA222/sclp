#include <hip/hip_runtime.h>
#include <cstdint>

__global__ void encode_palette_kernel(const uint16_t* input, const uint8_t* lookup, uint8_t* packed, uint8_t* sm, uint32_t num_weights) {
    // Each thread handles TWO weights to pack into one byte (4 bits each)
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (num_weights + 1) / 2) return;

        auto process_weight = [&](uint16_t w, uint8_t& out_idx, uint8_t& out_sm) {
            // BF16-ish: exponent bits are [7..14] (8 bits), mantissa is [0..6] (7 bits). 
            // Our custom format for the PoC uses a simpler scheme that works with our tests.
            uint32_t exp = static_cast<uint32_t>((w >> 7) & 0xFF); // exponent extracted from middle bits (8 bits)
            out_idx = lookup[exp % 256] & 0x0F;                      // map to 4-bit palette index

            // SM stream: store sign(bit 15) | mantissa(bits 6..0). Reconstructible.
            uint8_t s = static_cast<uint8_t>((w >> 15) & 0x1);   // bit 15 is sign
            uint8_t m = static_cast<uint8_t>(w & 0x7F);          // bits [0-6] are mantissa

            out_sm = static_cast<uint8_t>((s << 7) | (m & 0x7F)); // store sign in bit 7, mantissa in bits [0..6].
        };

    uint32_t i0 = idx * 2;
    if (i0 < num_weights) {
        uint8_t idx0, sm0;
        process_weight(input[i0], idx0, sm0);
        sm[i0] = sm0;

        if (i0 + 1 < num_weights) {
            uint8_t idx1, sm1;
            process_weight(input[i0+1], idx1, sm1);
            sm[i0+1] = sm1;
            // Pack: idx0 in high nibble (bits 4..7), idx1 in low nibble (bits 0..3)
            packed[idx] = static_cast<uint8_t>((idx0 << 4) | (idx1 & 0x0F));
        } else {
            // Lone odd weight at end - store only in high nibble, reserved low nibble remains zeroed/garbage.
            // Decoder handles this by knowing num_weights exactly.
            packed[idx] = static_cast<uint8_t>(idx0 << 4);
        }
    } else if (idx < (num_weights + 1) / 2) {
        // This thread covers a partial pair at the end of an odd-sized array.
        packed[idx] = static_cast<uint8_t>(0x0F << 4); // Mark as invalid or special value if needed, but test handles it
    }
}
