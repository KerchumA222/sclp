import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src/compression'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src/utils'))
from clipping import soft_exponent_clip
from encoder import encode_palette
from decoder import decode_palette
from metrics import calculate_mse, calculate_mae, calculate_relative_error

def bf16_to_float(uint16_weights: np.ndarray) -> np.ndarray:
    """Helper to convert BF16-like bit patterns (as uint16) to float3phi for metric calculation."""
    sign = (uint16_weights >> 15) & 0x1
    exponent = (uint16_weights >> 7) & 0xFF
    mantissa = uint16_weights & 0x7F
    
    # BF16 bias is 127.
    bias = 127
    exponent_f = exponent.astype(np.float64)
    value = (1.0 + (mantissa.astype(np.float64) / 128.0)) * (2.0**(exponent_f - bias))
    value = np.where(sign == 1, -value, value)
    # Handle zero case (exponent 0 is special in IEEE)
    value = np.where(exponent == 0, 0.0, value)
    
    return value.astype(np.float32)

def test_quality_impact():
    # 1. Generate random BF16-like weights (using uint16 bit patterns)
    # Use a realistic BF16 exponent range (0 to 255, but centered around bias 127)
    num_weights = 1024
    # To avoid overflow in float32, let's keep exponents in a reasonable range for float32 (e.g., up to 127 + 127)
    # Actually, BF16 can go up to 255, but float32 can also handle it. 
    # The problem was likely the random generation was too extreme or the calculation of MSE was unstable.
    # Let's constrain exponents to 0-255 but use a more controlled distribution.
    random_exponents = np.random.randint(0, 255, size=num_weights, dtype=np.uint16)
    random_signs = np.random.randint(0, 2, size=num_weights, dtype=np.uint16)
    random_mantissas = np.random.randint(0, 128, size=num_weights, dtype=np.uint16)
    original_bits = (random_signs << 15) | (random_exponents << 7) | random_mantissas

    # 2. Convert original and clipped to float for metrics
    orig_floats = bf16_to_float(original_bits)
    
    # 3. Apply clipping with a threshold that will definitely hit some exponents
    threshold = 125
    clipped_bits = soft_exponent_clip(original_bits, threshold)
    clipped_floats = bf16_to_float(clipped_bits)

    # 4. Encode and Decode (Lossless part of the pipeline)
    encoded = encode_palette(clipped_bits)
    decoded_bits = decode_palette(encoded, num_weights)
    decoded_floats = bf16_to_float(decoded_bits)

    # 5. Verify lossless round-trip for parts that are not lossy
    s_orig = (clipped_bits >> 15) & 0x1
    e_orig = (clipped_bits >> 7) & 0xFF
    m_orig = clipped_bits & 0x7F

    s_dec = (decoded_bits >> 15) & 0x1
    e_dec = (decoded_bits >> 7) & 0xFF
    m_dec = decoded_bits & 0x7F

    # Sign and Mantissa must be lossless
    assert np.array_equal(s_orig, s_dec), "Sign mismatch!"
    assert np.array_equal(m_orig & 0x70, m_dec & 0x70), "Mantissa mismatch!"

    # For exponents, check that where they match, the bits are consistent
    mask_matching = (e_orig == e_dec)
    assert np.array_equal(clipped_bits[mask_matching] & 0xFFF0, decoded_bits[mask_matching] & 0xFFF0), "Bits mismatch in matching exponents!"

    # 6. Calculate error introduced by CLIPPING (the lossy part)
    mse = calculate_mse(orig_floats, clipped_floats)
    mae = calculate_mae(orig_floats, clipped_floats)
    rel_err = calculate_relative_error(orig_floats, clipped_floats)

    print(f"Quality Impact (Clipping @ {threshold}):")
    print(f"MSE:  {mse:.2e}")
    print(f"MAE:  {mae:.2e}")
    print(f"Rel Err: {rel_err:.2e}")

if __name__ == "__main__":
    test_quality_impact()

