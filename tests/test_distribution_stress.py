import numpy as np
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/compression'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/utils'))

from clipping import soft_exponent_clip
from encoder import encode_palette
from decoder import decode_palette
from metrics import calculate_mse, calculate_mae, calculate_relative_error

def run_distribution_test(name, weights_bf16):
    num_weights = len(weights_bf16)
    threshold = 125
    
    # Clipping
    clipped_bits = soft_exponent_clip(weights_bf16, threshold)
    
    # Encoding
    encoded = encode_palette(clipped_bits)
    
    # Decoding
    decoded_bits = decode_palette(encoded, num_weights)
    
    # Metrics calculation (requires float conversion)
    def bf16_to_float(uint16_weights):
        sign = (uint16_weights >> 15) & 0x1
        exponent = (uint16_weights >> 7) & 0xFF
        mantissa = uint16_weights & 0x7F
        bias = 127
        exp_f = exponent.astype(np.float64)
        val = (1.0 + (mantissa.astype(np.float64) / 128.0)) * (2.0**(exp_f - bias))
        val = np.where(sign == 1, -val, val)
        val = np.where(exponent == 0, 0.0, val)
        return val.astype(np.float32)

    orig_floats = bf16_to_float(weights_bf16)
    clipped_floats = bf16_to_float(clipped_bits)
    
    mse = calculate_mse(orig_floats, clipped_floats)
    rel_err = calculate_relative_error(orig_floats, clipped_floats)
    
    # Compression ratio calculation (rough estimate based on bits)
    original_size = num_weights * 16 # 16 bits per weight
    compressed_size = (len(encoded['palette']) + 
                       len(encoded['packed_indices']) * 4 + 
                       len(encoded['sm_stream']) * 8)
    ratio = original_size / compressed_size

    print(f"[{name:15}] Ratio: {ratio:5.2f}x | MSE: {mse:6.2e} | RelErr: {rel_err:6.2e}")

if __name__ == "__main__":
    num_elements = 4096
    print(f"Stress Testing SCLP with {num_elements} weights per distribution...\n")
    print(f"{'Distribution':15} | {'Ratio':7} | {'MSE':10} | {'RelErr':8}")
    print("-" * 55)

    # 1. Uniform Distribution (High entropy, harder for palette)
    unif_exp = np.random.randint(0, 255, size=num_elements, dtype=np.uint16)
    unif_bits = (np.zeros(num_elements, dtype=np.uint16) << 15) | (unif_exp << 7) | np.random.randint(0, 128, size=num_elements, dtype=np.uint16)
    run_distribution_test("Uniform", unif_bits)

    # 2. Normal Distribution (Clustered, easier for palette)
    norm_vals = np.random.normal(loc=127, scale=30, size=num_elements).astype(np.int32)
    norm_exp = np.clip(norm_vals, 0, 254).astype(np.uint16)
    norm_bits = (np.zeros(num_elements, dtype=np.uint16) << 15) | (norm_exp << 7) | np.random.randint(0, 128, size=num_elements, dtype=np.uint16)
    run_distribution_test("Normal", norm_bits)

    # 3. Log-Normal (Skewed, similar to real weights)
    lognorm_vals = np.random.lognormal(mean=4.0, sigma=0.5, size=num_elements)
    lognorm_exp = np.clip(lognorm_vals, 0, 254).astype(np.uint16)
    lognorm_bits = (np.zeros(num_elements, dtype=np.uint16) << 15) | (lognorm_exp << 7) | np.random.randint(0, 128, size=num_elements, dtype=np.uint16)
    run_distribution_test("Log-Normal", lognorm_bits)
