import numpy as np
import time
import os
import sys

# Add the package directory to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../python_pkg")))

try:
    import testmodule
except ImportError as e:
    print(f"Failed to import testmodule: {error}")
    sys.exit(1)

def benchmark_hip_extension(num_weights):
    """Benchmarks the HIP extension performance."""
    # 1. Setup Data
    valid_exps = np.arange(1, 11, dtype=np.uint16)
    
    weights_uint16 = np.zeros(num_weights, dtype=np.uint16)
    for i in range(num_weights):
        exp = valid_exps[i % 10]
        mantissa = i % 128
        sign = 0 if i % 2 == 0 else 1
        weights_uint16[i] = (sign << 15) | (exp << 7) | mantissa

    palette = np.arange(10, dtype=np.uint8)
    lookup = np.zeros(256, dtype=np.uint8)
    for i, e in enumerate(valid_exps):
        lookup[e] = i

    threshold = 10

    # 2. Measure Clipping (HIP)
    start = time.perf_counter()
    clipped = testmodule.clip(weights_uint16, threshold, 42)
    clipping_time = time.perf_counter() - start

    # 3. Measure Encoding (HIP)
    start = time.perf_counter()
    encoded = testmodule.encode(clipped, lookup)
    encoding_time = time.perf_counter() - start
    packed = encoded["packed"]
    sm_stream = encoded["sm"]

    # 4. Measure Decoding (HIP)
    start = time.perf_counter()
    decoded = testmodule.decode(packed, sm_stream, palette)
    decoding_time = time.perf_counter() - start

    # Stats
    original_size_bytes = num_weights * 2
    compressed_size_bytes = len(packed) + len(sm_stream)
    throughput_gb_s = (original_size_bytes / 1e9) / decoding_time if decoding_time > 0 else 0
    compression_ratio = original_size_bytes / compressed_size_bytes

    return {
        'num_weights': num_weights,
        'clipping_ms': clipping_time * 1000,
        'encoding_ms': encoding_time * 1000,
        'decoding_ms': decoding_time * 1000,
        'throughput_gb_s': throughput_gb_s,
        'compression_ratio': compression_ratio,
    }

if __name__ == "__main__":
    size = 2**20 
    print(f"Benchmarking HIP module with {size} weights...")
    try:
        res = benchmark_hip_extension(size)
        print(f"{'Weights':>10} | {'Ratio':>8} | {'Dec (ms)':>10} | {'Thr (GB/s)':>10}")
        print("-" * 55)
        print(f"{res['num_weights']:10d} | "
              f"{res['compression_ratio']:8.2f}x | "
              f"{res['decoding_ms']:10.4f} | "
              f"{res['throughput_gb_s']:10.4f}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
