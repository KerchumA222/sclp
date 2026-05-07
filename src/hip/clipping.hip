#include <hip/hip_runtime.h>
#include <cstdint>
#include <algorithm>

// Simple Xorshift PRNG for branchless stochastic rounding within the kernel
struct XorshiftPRNG {
    unsigned int state;
    __device__ unsigned int next() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }
};

__global__ void soft_exponent_clip_kernel(
    const uint16_t* __restrict__ input,
    uint16_t* __restrict__ output,
    unsigned int num_weights,
    unsigned char threshold,
    unsigned int seed,
    uint8_t mantissa_mask
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_weights) return;

    XorshiftPRNG prng;
    prng.state = seed ^ idx;

    uint16_t weight = input[idx];
    uint16_t exp = (weight >> 7) & 0xFF;

    if (exp > (uint16_t)threshold + 1) {
        // Hard clip
        weight &= 0x80FF;
        weight |= ((uint16_t)threshold << 7);
    } else if (exp == (uint16_t)threshold + 1) {
        // Stochastic round: 50% chance to keep threshold + 1, 50% to clip to threshold
        if ((prng.next() & 0x1) == 0) {
            weight &= 0x80FF;
            weight |= ((uint16_t)threshold << 7);
        }
    }

    // Apply mantissa mask: zero out mantissa bits not covered by the mask
    output[idx] = weight & (0xFF80 | (uint16_t)mantissa_mask);
}
