#include <hip/hip_runtime.h>
#include <cstdint>

extern __global__ void encode_palette_kernel(const uint16_t* input, const uint8_t* lookup, uint8_t* packed, uint8_t* sm, uint32_t num_weights);
extern __global__ void decode_palette_kernel(const uint8_t* packed, const uint8_t* sm, const uint8_t* palette, uint16_t* output, uint32_t num_weights);
extern __global__ void soft_exponent_clip_kernel(const uint16_t* input, uint16_t* output, unsigned int num_weights, unsigned char threshold, unsigned int seed, uint8_t mantissa_mask);

// --- launch functions in extern "C" for pybind11 visibility ---
extern "C" {

void launch_encode_kernel(const uint16_t* input, const uint8_t* lookup, uint8_t* packed, uint8_t* sm, uint32_t num_weights) {
    dim3 block(256);
    // Each thread handles TWO weights to pack into one byte. 
    uint32_t threads_needed = (num_weights + 1) / 2;
    dim3 grid((threads_needed + 255) / 256);

    encode_palette_kernel<<<grid, block>>>(input, lookup, packed, sm, num_weights);
}

void launch_decode_kernel(const uint8_t* packed, const uint8_t* sm, const uint8_t* palette, uint16_t* output, uint32_t num_weights) {
    dim3 block(256);
    // Each thread handles ONE reconstructed weight (one element per thread).
    uint32_t grid_size = (num_weights + 255) / 256;
    dim3 grid(grid_size);

    decode_palette_kernel<<<grid, block>>>(packed, sm, palette, output, num_weights);
}

// Stub GEMM to allow pipeline verification on non-ROCm systems if needed
void launch_gemm_cuda(const float* A, const float* B, float* C, int M, int N, int K) {
    printf("  -> Launching gemm stub: %dx%d x %dx%d -> %dx%d\n", M, K, K, N, M, N);
}

void launch_clip_kernel(const uint16_t* input, uint16_t* output, unsigned int num_weights, unsigned char threshold, unsigned int seed, uint8_t mantissa_mask) {
    dim3 block(256);
    uint32_t grid_size = (num_weights + 255) / 256;
    dim3 grid(grid_size);

    soft_exponent_clip_kernel<<<grid, block>>>(input, output, num_weights, threshold, seed, mantissa_mask);
}

} // end extern "C"

