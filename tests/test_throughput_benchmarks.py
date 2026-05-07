import time
import numpy as np
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.compression.pipeline import SCLPCompressor

def benchmark_throughput(tensor_sizes=[10**4, 10**5, 10**6, 10**7]):
    """Benchmarks compression and decompression throughput (elements/sec)."""
    compressor = SCLPCompressor(threshold_exponent=125)
    
    print(f"{'Size (Elements)':<18} | {'Comp. (M/param/s)':<18} | {'Decomp. (M/param/s)':<18}")
    print("-" * 60)

    for N in tensor_sizes:
        # Create dummy BF16-like data (as uint16)
        weights = np.random.randint(0, 65535, size=N, dtype=np.uint16)
        
        # Benchmark Compression
        start_c = time.time()
        compressed_package = compressor.compress(weights)
        end_c = time.time()
        comp_duration = end_c - start_c
        comp_speed = (N / comp_duration) / 1e6 if comp_duration > 0 else 0

        # Benchmark Decompression
        start_d = time.time()
        reconstructed = compressor.decompress(compressed_package)
        end_d = time.time()
        decomp_duration = end_d - start_d
        decomp_speed = (N / decomp_duration) / 1e6 if decomp_duration > 0 else 0

        print(f"{N:<18,} | {comp_speed:<18.4f} | {decomp_speed:<18.4f}")

if __name__ == "__main__":
    benchmark_throughput()
