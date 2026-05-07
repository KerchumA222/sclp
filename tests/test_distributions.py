import numpy as np
import os
import sys

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/compression')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/utils')))

from encoder import encode_palette
from decoder import decode_palette
from clipping import soft_exponent_clip

def bf16_to_float64_real(uint16_weights: np.ndarray) -> np.ndarray:
    sign = (uint16_weights >> 15) & 0x1
    exponent = (uint16_weights >> 7) & 0xFF
    mantissa = uint16_weights & 0x7F
    bias = 127
    exponent_f = exponent.astype(np.float64)
    # Use float64 to prevent overflow during calculation
    value = (1.0 + (mantissa.astype(np.float64) / 128.0)) * (2.0**(exponent_f - bias))
    value = np.where(sign == 1, -value, value)
    value = np.where(exponent == 0, 0.0, value)
    return value


def run_distribution_test(name, weights_bf16, threshold=125, mask=0x7F):
    num_weights = len(weights_bf16)
    original_size = num_weights * 2
    
    # Pipeline
    clipped = soft_exponent_clip(weights_bf16, threshold, mantissa_mask=mask)
    encoded = encode_palette(clipped)
    decoded = decode_palette(encoded, num_weights)
    
    # Metrics
    compressed_size = (len(encoded['palette']) + 
                       len(encoded['packed_indices']) + 
                       len(encoded['sm_stream']))
    ratio = original_size / compressed_size
    
    # Use float64 for error metrics to avoid overflow
    orig_f = bf16_to_float64_real(weights_bf16)
    clip_f = bf16_to_float64_real(clipped)
    
    # Calculate Relative Error: mean(|orig - clip| / orig)
    rel_error = np.mean(np.abs(orig_f - clip_f) / (np.abs(orig_f) + 1e-20))
    
    mask_name = hex(mask)
    print(f"[{name:^12} | {mask_name:^5}] Ratio: {ratio:5.2f}x | RelErr: {rel_error:.2e}")


def generate_distribution(name, num_weights):
    if name == "Uniform":
        exponents = np.random.randint(0, 255, size=num_weights, dtype=np.uint16)
    elif name == "Gaussian":
        exponents = np.int32(np.random.normal(127, 30, size=num_weights))
        exponents = np.clip(exponents, 0, 255).astype(np.uint16)
    elif name == "Laplace":
        exponents = np.int32(np.random.laplace(127, 20, size=num_weights))
        exponents = np.clip(exponents, 0, 255).astype(np.uint16)
    else:
        exponents = np.random.randint(0, 255, size=num_weights, dtype=np.uint16)
        
    signs = np.random.randint(0, 2, size=num_weights, dtype=np.uint16)
    mantissas = np.random.randint(0, 128, size=num_weights, dtype=np.uint16)
    return (signs << 15) | (exponents << 7) | mantissas

if __name__ == "__main__":
    num_weights = 10000
    distributions = ["Uniform", "Gaussian", "Laplace", "LLM-Pareto"]
    masks = [0x7F, 0x0F, 0x03] # Full, 4-bit, 2-bit
    
    print(f"{'Distribution':^12} | {'Mask':^5} | {'Ratio':^8} | {'RelErr':^16}")
    print("-" * 60)
    
    for dist in distributions:
        if dist == "LLM-Pareto":
            alpha = 1.5
            exponents = np.random.pareto(alpha, num_weights) * 50
            exponents = np.clip(exponents, 0, 255).astype(np.uint16)
            signs = np.int32(np.random.randint(0, 2, size=num_weights))
            mantissas = np.random.randint(0, 128, size=num_weights, dtype=np.uint16)
            weights = (signs.astype(np.uint16) << 15) | (exponents << 7) | mantissas
        else:
            weights = generate_distribution(dist, num_weights)
            
        for mask in masks:
            run_distribution_test(dist, weights, threshold=125, mask=mask)
