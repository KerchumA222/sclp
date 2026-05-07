#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdint>

using namespace rocwmma;

__global__ void fused_sclp_gemm_kernel(
    const __half* __restrict__ A, 
    const uint8_t* __restrict__ packed_indices,
    const uint8_t* __restrict__ sm_stream,
    const uint8_t* __restrict__ palette,
    __half* __restrict__ C, 
    int M, int N, int K
) {
    // WMMA supports 16x16x16 tiles
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    // Allocate fragments
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, __half> c_frag;

    fill_fragment(c_frag, (__half)0.0f);

    int row = (blockIdx.y * WMMA_M);
    int col = (blockIdx.x * WMMA_N);

    for (int k = 0; k < K; k += WMMA_K) {
        // 1. Decode SCLP tile into local buffer (transcoding to BF16)
        // Note: rocWMMA matrix_b (col_major) expects data in a specific layout
        // We use a flat array that mimics the layout rocWMMA expects for a col_major 16x16 fragment
        __half b_tile[WMMA_K * WMMA_N];
        for (int i = 0; i < WMMA_K; i++) {
            for (int j = 0; j < WMMA_N; j++) {
                int b_row = k + i;
                int b_col = col + j;
                int weight_idx = b_row * N + b_col;
                
                uint8_t packed_byte = packed_indices[weight_idx >> 1];
                uint8_t p_idx = (weight_idx & 1) ? (packed_byte & 0x0F) : (packed_byte >> 4);
                uint8_t sm_val = sm_stream[weight_idx];
                
                uint16_t bits = ((uint16_t)(sm_val >> 7) << 15) | ((uint16_t)palette[p_idx] << 7) | (sm_val & 0x7F);
                // Col-major layout: [col * WMMA_K + row]
                b_tile[j * WMMA_K + i] = *(__half*)&bits;
            }
        }

        // 2. Load fragments
        load_matrix_sync(a_frag, A + (row * K + k), K);
        load_matrix_sync(b_frag, b_tile, WMMA_K);

        // 3. WMMA FMA
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 4. Store result
    store_matrix_sync(C + (row * N + col), c_frag, N, mem_row_major);
}
