import numpy as np
import time
import os
import sys

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/compression')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/utils')))

from clipping import soft_exponent_clip
from encoder import encode_palette
from decoder import decode_palette

def generate_llm_like_weights(num_weights):
    """Generates weights following a power-law distribution to simulate LLM weights."""
    # 1. Generate exponents using a Pareto-like distribution (skewed towards small values)
    alpha = 1.5
    exponents = np.random.pareto(alpha, num_weights) * 50 
    exponents = np.clip(exponents, 0, 255).astype(np.uint16)
    
    # 2. Generate signs and mantissas
    signs = np.random.randint(0, 2, size=num_weights, dtype=np.uint16)
    mantissas = np.random.randint(0, 128, size=num_weights, dtype=np.uint16)
    
    # 3. Pack into BF16-like bits
    bits = (signs << 15) | (exponents << 7) | mantissas
    return bits

def benchmark_pipeline(num_weights):
    # 1. Setup Data (LLM-like)
    original_bits = generate_llm_like_weights(num_weights)
    original_size_bytes = num_weights * 2 

    threshold = 125

    # 2. Measure Clipping
    start = time.perf_counter()
    clipped_bits = soft_exponent_clip(original_bits, threshold)
    clipping_time = time.perf_counter() - start

    # 3. Measure Encoding
    start = time.perf_counter()
    encoded_data = encode_palette(clipped_bits)
    encoding_time = time.perf_counter() - start

    # Calculate Compressed Size
    compressed_size_bytes = (len(encoded_data['palette']) + 
                             len(encoded_data['packed_indices']) + 
                             len(encoded_data['sm_stream']))

    # 4. Measure Decoding
    start = time.perf_counter()
    decoded_bits = decode_palette(encoded_data, num_weights)
    decoding_time = time.perf_counter() - start

    # Calculate Throughput (GB/s)
    throughput_gb_s = (original_size_bytes / 1e9) / decoding_time if decoding_time > 0 else 0
    compression_ratio = original_size_bytes / compressed_size_bytes
    
    # Memory bandwidth utilization (GB/s)
    total_bytes_transferred = compressed_size_bytes + original_size_bytes
    bandwidth_gb_s = (total_bytes_transferred / 1e9) / decoding_time if decoding_time > 0 else 0

    return {
        'num_weights': num_weights,
        'clipping_ms': clipping_time * 1000,
        'encoding_ms': encoding_time * 1000,
        'decoding_ms': decoding_time * 1000,
        'throughput_gb_s': throughput_gb_s,
        'compression_ratio': compression_ratio,
        'bandwidth_gb_s': bandwidth_gb_s
    }

if __name__ == "__main__":
    sizes = [2**10, 2**12, 2**14, 2**16, 2**18]
    print(f"{'Weights':>10} | {'Ratio':>8} | {'Dec (ms)':>10} | {'Thr (GB/s)':>10} | {'BW (GB/s)':>10}")
    print("-" * 75)

    for size in sizes:
        try:
            res = benchmark_pipeline(size)
            print(
                f"{res['num_weights']:10d} | "
                f"{res['compression_ratio']:8.2f}x | "
                f"{res['decoding_ms']:10.4f} | "
                f"{res['throughput_gb_s']:10.4f} | "
                f"{res['bandwidth_gb_s']:10.4f}"
            )
        except Exception as e:
            print(f"Error benchmarking {size}: {e}")


