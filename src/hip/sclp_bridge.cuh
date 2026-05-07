#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

// This bridge handles the dispatch of SCLP decompression 
// to our highly optimized vectorized HIP kernel.

struct sclp_header {
    uint32_t num_weights;
    uint8_t  palette_size;
};

// Vectorized decoder implementation integrated from our POC
__global__ void sclp_decode_kernel(
    const uint8_t* __restrict__ packed_indices,
    const uint8_t* __restrict__ sm_stream,
    const uint8_t* __restrict__ palette,
    __half* __restrict__ output,
    uint32_t num_weights
) {
    __shared__ uint8_t s_palette[16];
    if (threadIdx.x < 16) {
        s_palette[threadIdx.x] = palette[threadIdx.x];
    }
    __syncthreads();

    uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t base_idx = pair_idx << 1;
    
    if (base_idx >= num_weights) return;

    uint8_t packed_byte = packed_indices[pair_idx];
    uint8_t exp0 = s_palette[packed_byte >> 4];
    uint8_t exp1 = s_palette[packed_byte & 0x0F];

    uint8_t sm0 = sm_stream[base_idx];
    output[base_idx] = ((uint16_t)(sm0 >> 7) << 15) | ((uint16_t)exp0 << 7) | (sm0 & 0x7F);

    if (base_idx + 1 < num_weights) {
        uint8_t sm1 = sm_stream[base_idx + 1];
        output[base_idx + 1] = ((uint16_t)(sm1 >> 7) << 15) | ((uint16_t)exp1 << 7) | (sm1 & 0x7F);
    }
}

// Entry point for llama.cpp HIP backend
inline void llama_sclp_dispatch(const void* sclp_data, __half* output, hipStream_t stream) {
    const uint8_t* data = (const uint8_t*)sclp_data;
    
    // Parse header
    const sclp_header* hdr = (const sclp_header*)data;
    const uint8_t* palette = data + sizeof(sclp_header);
    const uint8_t* packed  = palette + hdr->palette_size;
    const uint8_t* sm      = packed + ((hdr->num_weights + 1) / 2);
    
    // Launch optimized decoder
    uint32_t threads_needed = (hdr->num_weights + 1) / 2;
    dim3 block(256);
    dim3 grid((threads_needed + 255) / 256);
    sclp_decode_kernel<<<grid, block, 0, stream>>>(packed, sm, palette, output, hdr->num_weights);
}
