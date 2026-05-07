#define __HIP_PLATFORM_AMD__
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

// --- Macro for HIP error checking ---
#define CHECK_HIP(cmd)                                                                           \
    do {                                                                                       \
        hipError_t error = cmd;                                                                \
        if (error != hipSuccess) {                                                             \
            throw std::runtime_error(std::string("HIP Error: ") + hipGetErrorString(error));     \
        }                                                                                      \
    } while (0)

// --- External Kernel Launchers ---
extern "C" {
    void launch_clip_kernel(const uint16_t* input, uint16_t* output, unsigned int num_weights, unsigned char threshold, unsigned int seed, uint8_t mantissa_mask);
    void launch_encode_kernel(const uint16_t* input, const uint8_t* lookup, uint8_t* packed, uint8_t* sm, uint32_t num_weights);
    void launch_decode_kernel(const uint8_t* packed, const uint8_t* sm, const uint8_t* palette, uint16_t* output, uint32_t num_weights);
    void launch_fused_gemm(const uint16_t* A, const uint8_t* packed, const uint8_t* sm, const uint8_t* palette, uint16_t* C, int M, int N, int K);
}

PYBIND11_MODULE(testmodule, m) {
    m.doc() = "HIP-based weight compression module";

    // 1. Clipping Kernel: input (uint16) -> output (uint16)
    m.def("clip", [](py::array_t<uint16_t> input, uint8_t threshold, uint32_t seed, uint8_t mantissa_mask) {
        auto buf_in = input.request();
        const uint16_t* h_input = static_cast<const uint16_t*>(buf_in.ptr);
        uint32_t num_weights = static_cast<uint32_t>(buf_in.size);

        py::array_t<uint16_t> output(num_weights);
        auto buf_out = output.request();
        uint16_t* h_output = static_cast<uint16_t*>(buf_out.ptr);

        uint16_t *d_input, *d_output;
        CHECK_HIP(hipMalloc(&d_input, num_weights * sizeof(uint16_t)));
        CHECK_HIP(hipMalloc(&d_output, num_weights * sizeof(uint16_t)));

        CHECK_HIP(hipMemcpy(d_input, h_input, num_weights * sizeof(uint16_t), hipMemcpyHostToDevice));
        launch_clip_kernel(d_input, d_output, num_weights, threshold, seed, mantissa_mask);
        CHECK_HIP(hipMemcpy(h_output, d_output, num_weights * sizeof(uint16_t), hipMemcpyDeviceToHost));

        CHECK_HIP(hipFree(d_input));
        CHECK_HIP(hipFree(d_output));
        return output;
    }, "Standard BF16 weight clipping kernel");

    // 2. Encode Kernel
    //    input:   uint16[N]     — clipped BF16 weights
    //    palette: uint8[<=16]   — exponent values (sorted by frequency); the wrapper
    //                             builds a nearest-neighbour lookup internally
    //    returns: dict {
    //      'packed':          uint8[ceil(N/2)] — nibble-packed palette indices
    //      'sm':              uint8[N]         — sign(7)|mantissa(6:0) per weight
    //      'sidecar_indices': uint32[K]        — positions of outlier weights
    //      'sidecar_values':  uint16[K]        — full BF16 bits for those weights
    //    }
    m.def("encode", [](py::array_t<uint16_t> input, py::array_t<uint8_t> palette_arr) {
        auto buf_in  = input.request();
        uint32_t num_weights = static_cast<uint32_t>(buf_in.size);
        const uint16_t* h_input = static_cast<const uint16_t*>(buf_in.ptr);

        auto buf_pal = palette_arr.request();
        const uint8_t* h_palette = static_cast<const uint8_t*>(buf_pal.ptr);
        uint32_t palette_size = static_cast<uint32_t>(buf_pal.size);

        // Build nearest-neighbour lookup (uint8[256]) from palette
        uint8_t h_lookup[256];
        for (int ev = 0; ev < 256; ev++) {
            int best_idx = 0, best_dist = 257;
            for (uint32_t pi = 0; pi < palette_size; pi++) {
                int d = ev - (int)h_palette[pi];
                if (d < 0) d = -d;
                if (d < best_dist) { best_dist = d; best_idx = (int)pi; }
            }
            h_lookup[ev] = static_cast<uint8_t>(best_idx);
        }

        // Build palette set for sidecar detection (O(1) lookup)
        std::unordered_set<uint8_t> palette_set(h_palette, h_palette + palette_size);

        uint32_t packed_size = (num_weights + 1) / 2;

        py::array_t<uint8_t> packed_out(packed_size);
        py::array_t<uint8_t> sm_out(num_weights);
        uint8_t* h_packed = static_cast<uint8_t*>(packed_out.request().ptr);
        uint8_t* h_sm     = static_cast<uint8_t*>(sm_out.request().ptr);

        uint16_t *d_input;
        uint8_t  *d_lookup, *d_packed, *d_sm;
        CHECK_HIP(hipMalloc(&d_input,  num_weights * sizeof(uint16_t)));
        CHECK_HIP(hipMalloc(&d_lookup, 256));
        CHECK_HIP(hipMalloc(&d_packed, packed_size));
        CHECK_HIP(hipMalloc(&d_sm,     num_weights));

        CHECK_HIP(hipMemcpy(d_input,  h_input,  num_weights * sizeof(uint16_t), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_lookup, h_lookup, 256,                            hipMemcpyHostToDevice));

        launch_encode_kernel(d_input, d_lookup, d_packed, d_sm, num_weights);

        CHECK_HIP(hipMemcpy(h_packed, d_packed, packed_size,  hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(h_sm,     d_sm,     num_weights,  hipMemcpyDeviceToHost));

        CHECK_HIP(hipFree(d_input));
        CHECK_HIP(hipFree(d_lookup));
        CHECK_HIP(hipFree(d_packed));
        CHECK_HIP(hipFree(d_sm));

        // Compute sidecar on CPU: weights whose exponent is not in the palette
        std::vector<uint32_t> sc_idx_vec;
        std::vector<uint16_t> sc_val_vec;
        for (uint32_t i = 0; i < num_weights; i++) {
            uint8_t exp = static_cast<uint8_t>((h_input[i] >> 7) & 0xFF);
            if (palette_set.find(exp) == palette_set.end()) {
                sc_idx_vec.push_back(i);
                sc_val_vec.push_back(h_input[i]);
            }
        }

        py::array_t<uint32_t> sidecar_indices(sc_idx_vec.size());
        py::array_t<uint16_t> sidecar_values(sc_val_vec.size());
        if (!sc_idx_vec.empty()) {
            std::memcpy(sidecar_indices.request().ptr, sc_idx_vec.data(),
                        sc_idx_vec.size() * sizeof(uint32_t));
            std::memcpy(sidecar_values.request().ptr, sc_val_vec.data(),
                        sc_val_vec.size() * sizeof(uint16_t));
        }

        py::dict result;
        result["packed"]          = packed_out;
        result["sm"]              = sm_out;
        result["sidecar_indices"] = sidecar_indices;
        result["sidecar_values"]  = sidecar_values;
        return result;
    }, "Encode BF16 weights to nibble-packed palette indices and SM stream, with sidecar for outliers");

    // 3. Decode Kernel
    //    packed:          uint8[ceil(N/2)] — nibble-packed palette indices
    //    sm:              uint8[N]         — sign(7)|mantissa(6:0) per weight
    //    palette:         uint8[<=16]      — exponent values
    //    sidecar_indices: uint32[K]        — optional outlier positions
    //    sidecar_values:  uint16[K]        — optional outlier BF16 values
    //    returns: uint16[N]
    m.def("decode", [](py::array_t<uint8_t> packed,
                       py::array_t<uint8_t> sm,
                       py::array_t<uint8_t> palette,
                       py::array_t<uint32_t> sidecar_indices,
                       py::array_t<uint16_t> sidecar_values) {
        auto buf_sm = sm.request();
        uint32_t num_weights = static_cast<uint32_t>(buf_sm.size);
        const uint8_t* h_sm = static_cast<const uint8_t*>(buf_sm.ptr);

        auto buf_packed = packed.request();
        const uint8_t* h_packed = static_cast<const uint8_t*>(buf_packed.ptr);

        auto buf_palette = palette.request();
        const uint8_t* h_palette = static_cast<const uint8_t*>(buf_palette.ptr);

        py::array_t<uint16_t> output(num_weights);
        uint16_t* h_output = static_cast<uint16_t*>(output.request().ptr);

        uint32_t packed_size = (num_weights + 1) / 2;
        uint8_t  *d_packed, *d_sm, *d_palette;
        uint16_t *d_output;
        CHECK_HIP(hipMalloc(&d_packed,  packed_size));
        CHECK_HIP(hipMalloc(&d_sm,      num_weights));
        CHECK_HIP(hipMalloc(&d_palette, buf_palette.size));
        CHECK_HIP(hipMalloc(&d_output,  num_weights * sizeof(uint16_t)));

        CHECK_HIP(hipMemcpy(d_packed,  h_packed,  packed_size,          hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_sm,      h_sm,      num_weights,           hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_palette, h_palette, buf_palette.size,      hipMemcpyHostToDevice));

        launch_decode_kernel(d_packed, d_sm, d_palette, d_output, num_weights);

        CHECK_HIP(hipMemcpy(h_output, d_output, num_weights * sizeof(uint16_t), hipMemcpyDeviceToHost));

        CHECK_HIP(hipFree(d_packed));
        CHECK_HIP(hipFree(d_sm));
        CHECK_HIP(hipFree(d_palette));
        CHECK_HIP(hipFree(d_output));

        // Apply sidecar overrides on CPU (restores outlier weights exactly)
        auto buf_sc_idx = sidecar_indices.request();
        auto buf_sc_val = sidecar_values.request();
        if (buf_sc_idx.size > 0) {
            const uint32_t* h_sc_idx = static_cast<const uint32_t*>(buf_sc_idx.ptr);
            const uint16_t* h_sc_val = static_cast<const uint16_t*>(buf_sc_val.ptr);
            for (py::ssize_t i = 0; i < buf_sc_idx.size; i++) {
                h_output[h_sc_idx[i]] = h_sc_val[i];
            }
        }

        return output;
    },
    py::arg("packed"),
    py::arg("sm"),
    py::arg("palette"),
    py::arg("sidecar_indices") = py::array_t<uint32_t>(0),
    py::arg("sidecar_values")  = py::array_t<uint16_t>(0),
    "Decode nibble-packed palette indices and SM stream back to BF16 weights, applying sidecar");

    m.def("fused_gemm", [](py::array_t<uint16_t> A, py::array_t<uint8_t> packed, py::array_t<uint8_t> sm, py::array_t<uint8_t> palette, int M, int N, int K) {
        auto buf_a = A.request();
        auto buf_packed = packed.request();
        auto buf_sm = sm.request();
        auto buf_pal = palette.request();

        py::array_t<uint16_t> C(M * N);
        auto buf_c = C.request();

        uint16_t *d_A, *d_C;
        uint8_t *d_packed, *d_sm, *d_palette;
        CHECK_HIP(hipMalloc(&d_A, M * K * sizeof(uint16_t)));
        CHECK_HIP(hipMalloc(&d_C, M * N * sizeof(uint16_t)));
        CHECK_HIP(hipMalloc(&d_packed, packed.size()));
        CHECK_HIP(hipMalloc(&d_sm, sm.size()));
        CHECK_HIP(hipMalloc(&d_palette, palette.size()));

        CHECK_HIP(hipMemcpy(d_A, buf_a.ptr, M * K * sizeof(uint16_t), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_packed, buf_packed.ptr, packed.size(), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_sm, buf_sm.ptr, sm.size(), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_palette, buf_pal.ptr, palette.size(), hipMemcpyHostToDevice));

        launch_fused_gemm((const uint16_t*)d_A, (const uint8_t*)d_packed, (const uint8_t*)d_sm, (const uint8_t*)d_palette, (uint16_t*)d_C, M, N, K);

        CHECK_HIP(hipMemcpy(buf_c.ptr, d_C, M * N * sizeof(uint16_t), hipMemcpyDeviceToHost));

        CHECK_HIP(hipFree(d_A));
        CHECK_HIP(hipFree(d_C));
        CHECK_HIP(hipFree(d_packed));
        CHECK_HIP(hipFree(d_sm));
        CHECK_HIP(hipFree(d_palette));
        return C;
    }, "Fused SCLP-GEMM kernel");
}
