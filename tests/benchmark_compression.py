import time
import numpy as np
import sys
import os

# Add the built python package to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../python_pkg')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import testmodule
from src.compression.encoder import encode_palette

def benchmark_kernels(num_weights=10_000_000):
    print(f"Starting Benchmark: {num_weights:,} weights\n")
    
    rng = np.random.default_rng(42)
    # Generate random BF16 weights
    h_weights = rng.integers(0x4000, 0x7FFF, size=num_weights, dtype=np.uint16)
    
    # --- 1. Baseline Benchmark (Standard BF16) ---
    # We need to move data to GPU for a fair comparison
    # Note: The wrapper handles hipMalloc/hipMemcpy, so we just pass the array
    
    # Prepare threshold and seed
    threshold = 0x45
    seed = 98765
    
    # Warmup
    _ = testmodule.clip(h_weights.copy(), threshold, seed, 0x07)
    
    start_time = time.perf_counter()
    # We use a copy to ensure we aren't measuring Python overhead of the same object
    _ = testmodule.clip(h_weights.copy(), threshold, seed, 0x07)
    end_time = time.perf_counter()
    
    baseline_duration = end_time - start_time
    baseline_throughput = num_weights / baseline_duration
    baseline_bandwidth = (num_weights * 2) / (baseline_duration * 1e9) # GB/s (2 bytes per weight)

    print(f"--- Baseline (BF16) ---")
    print(f"Time: {baseline_duration:.4f}s")
    print(f"Throughput: {baseline_throughput/1e6:.2f} M weights/s")
    print(f"Bandwidth: {baseline_bandwidth:.2f} GB/s\n")

    # --- 2. Compressed Benchmark (4-bit) ---
    # First, perform the encoding (Python side)
    encoded = encode_palette(h_weights)
    sm = encoded['sm_stream'].astype(np.uint8)
    packed = encoded['packed_indices'].astype(np.uint8)
    palette = encoded['palette'].astype(np.uint8)
    
    # Warmup
    _ = testmodule.decode(packed, sm, palette)
    
    start_time = time.perf_counter()
    _ = testmodule.decode(packed, sm, palette)
    end_time = time.perf_counter()
    
    compressed_duration = end_time - start_time
    compressed_throughput = num_weights / compressed_duration
    # Bandwidth is lower because we read fewer bytes
    bytes_read = sm.nbytes + packed.nbytes + palette.nbytes
    compressed_bandwidth = bytes_read / (compressed_duration * 1e9)

    print(f"--- Compressed (4-bit Decode) ---")
    print(f"Time: {compressed_duration:.4f}s")
    print(f"Throughput: {compressed_throughput/1e6:.2f} M weights/s")
    print(f"Bandwidth: {compressed_bandwidth:.2f} GB/s\n")

    # --- 3. Comparison ---
    speedup = baseline_duration / compressed_duration
    print(f"Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    benchmark_kernels()
