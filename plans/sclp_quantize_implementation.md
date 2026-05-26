# Plan: Add SCLP Support to llama-quantize

This plan outlines the steps to integrate SCLP (Soft Clipping Lossless-First) compression into the `llama-quantize` tool within the `llama.cpp` repository.

## 1. Plumbing & CLI Exposure

- **`include/llama.h`**:
    - Define new `llama_ftype` constants:
        - `LLAMA_FTYPE_MOSTLY_SCLP = 42`
        - `LLAMA_FTYPE_MOSTLY_SCLP4 = 43`
        - `LLAMA_FTYPE_MOSTLY_SCLP6 = 44`
- **`tools/quantize/quantize.cpp`**:
    - Add `SCLP`, `SCLP4`, and `SCLP6` to the `QUANT_OPTIONS` vector.
    - Provide appropriate descriptions (e.g., "8-bit SCLP", "4-bit SCLP", "6-bit SCLP").
- **`src/llama-quant.cpp`**:
    - Update `llama_ftype_get_default_type` to map the new ftypes to `GGML_TYPE_SCLP`, `GGML_TYPE_SCLP4`, and `GGML_TYPE_SCLP6`.

## 2. Core Quantization Logic (Porting from PoC)

The actual quantization (encoding) logic needs to be ported from Python/HIP to C++.

- **`ggml/src/ggml-quants.c`** (or a new `ggml-sclp.cpp` if appropriate):
    - **Palette Generation**: Port the 1-D weighted k-means and frequency-based palette generation for exponents.
    - **Encoding/Packing**:
        - `quantize_row_sclp`: 8-bit interleaved (4-bit idx, 4-bit SM).
        - `quantize_row_sclp4`: 4-bit.
        - `quantize_row_sclp6`: 6-bit.
    - **Sidecar Generation**: Logic to identify outliers (based on exponent distance or imatrix importance) and store them in the lossless sidecar section.
- **`ggml/src/ggml.c`**:
    - Update `ggml_quantize_chunk` to dispatch to the new SCLP quantization functions.

## 3. GGUF Format & Variable Size Handling

SCLP blobs are variable-sized due to the sidecar and per-expert palettes.

- **`src/llama-quant.cpp`**:
    - Modify the quantization loop to handle the fact that `new_size` for SCLP is not just `nrows * row_size`.
    - It must include the header (num_weights, palette) and the trailing sidecar.
    - Use `gguf_set_tensor_disk_size` to correctly update the GGUF metadata when saving.

## 4. Importance Matrix (imatrix) Integration

- Port the imatrix-aware sidecar selection:
    - Instead of just hard clipping, rank weights by `importance * distance` and use the sidecar budget (e.g., 1%) to rescue the most critical weights.
    - This is essential for maintaining quality at 4-bit (SCLP4).

## 5. Validation & Testing

- Compare the output of `llama-quantize --tensor-type ...=sclp` with the existing Python-based converters (`convert_to_sclp_gguf.py`).
- Verify bit-level compatibility for the `ws_stream` and sidecar sections.
- Run perplexity tests on the resulting GGUFs to ensure no regression.
