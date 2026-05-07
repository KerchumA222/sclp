# SCLP GGUF Tensor Header Specification

To ensure robustness in `llama.cpp`'s tensor loading, SCLP tensors will be written with a formal binary header prefix before the payload.

### Memory Layout
All SCLP tensors follow this structure:

| Offset | Size | Name | Description |
| :--- | :--- | :--- | :--- |
| 0 | 4 bytes | `num_weights` | Total number of BF16 weights |
| 4 | 1 byte | `palette_size` | Number of entries in palette (max 16) |
| 5 | `palette_size` | `palette` | Exponent values (uint8) |
| 5+P | `(num_weights+1)//2`| `packed_indices` | 4-bit palette indices |
| ... | `num_weights` | `sm_stream` | Sign + Mantissa bytes |

### Implementation Detail
- **Total Payload Size:** $5 + \text{palette\_size} + ((N+1)/2) + N$ bytes.
- **Loading:** The `ggml_hip` dispatcher will read this header to reconstruct the pointers for our GPU decoder kernels.

---
# Llama.cpp Integration: HIP Dispatch Hook

To trigger SCLP decompression, the backend will be modified as follows:

1. **`ggml-cuda/mmq.cu`**: Add `case GGML_TYPE_SCLP:` to `ggml_cuda_mul_mat_q_switch_type`.
2. **Decompression Logic**:
   - The kernel will read the header bytes from the `data` pointer.
   - It will derive the pointers for `packed_indices`, `sm_stream`, and `palette`.
   - It will perform an in-place or auxiliary-buffer decode (to FP16/BF16) and then immediately proceed with the GEMM.
