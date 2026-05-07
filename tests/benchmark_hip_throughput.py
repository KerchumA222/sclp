"""
HIP Throughput Benchmark for SCLP Compression.

Measures:
1. Raw BF16 copy bandwidth (Memory baseline)
2. SCLP Decode throughput (Compressed -> BF16)
3. Effective Bandwidth (Original size / Decode time)

Usage:
    source eval_env/bin/activate
    python3 tests/benchmark_hip_throughput.py
"""
import numpy as np
import time
import os
import sys

# Add python_pkg to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../python_pkg')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

try:
    import testmodule
    from compression.encoder import encode_palette
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def benchmark_throughput():
    # Large size to saturate memory bandwidth
    num_weights = 500_000_000 
    print(f"Benchmarking with {num_weights:,} weights (~{num_weights*2/1e6:.1f} MB BF16)")
    
    # 1. Prepare Data
    rng = np.random.default_rng(42)
    # Generate random BF16 bits
    h_weights = rng.integers(0x0000, 0xFFFF, size=num_weights, dtype=np.uint16)
    
    # Encode for SCLP (Reference Python encoder)
    print("Encoding weights for SCLP...")
    encoded = encode_palette(h_weights)
    packed = encoded['packed_indices']
    sm = encoded['sm_stream']
    palette = encoded['palette']
    
    # 2. Warmup
    print("Warming up kernels...")
    for _ in range(3):
        _ = testmodule.clip(h_weights, 125, 42, 0x7F)
        _ = testmodule.decode(packed, sm, palette)

    # 3. Benchmark: SCLP Decode
    print("Running SCLP Decode Benchmark...")
    iters = 10
    start = time.perf_counter()
    for _ in range(iters):
        _ = testmodule.decode(packed, sm, palette)
    end = time.perf_counter()
    
    decode_time = (end - start) / iters
    decode_throughput = num_weights / decode_time
    
    # Bytes read: packed_indices (4 bits/w) + sm_stream (8 bits/w) + palette (negligible)
    # total = N*0.5 + N*1.0 = 1.5 * N bytes
    bytes_read = len(packed) + len(sm) + len(palette)
    # Bytes written: N * 2 (BF16)
    bytes_written = num_weights * 2
    actual_bandwidth = (bytes_read + bytes_written) / (decode_time * 1e9)
    effective_throughput_gb = (num_weights * 2) / (decode_time * 1e9)

    print(f"\n--- SCLP Decode Results ---")
    print(f"Avg Time:      {decode_time*1000:.3f} ms")
    print(f"Throughput:    {decode_throughput/1e6:.2f} M weights/s")
    print(f"Actual BW:     {actual_bandwidth:.2f} GB/s (Physical read+write)")
    print(f"Effective BW:  {effective_throughput_gb:.2f} GB/s (Equiv. BF16 bandwidth)")

    # 4. Benchmark: Baseline (Copy/Clip as proxy for raw bandwidth)
    # We use 'clip' with high threshold and full mask as a proxy for a simple read-modify-write kernel
    print("\nRunning Baseline (Clip/Copy) Benchmark...")
    start = time.perf_counter()
    for _ in range(iters):
        _ = testmodule.clip(h_weights, 0xFF, 42, 0x7F)
    end = time.perf_counter()
    
    baseline_time = (end - start) / iters
    baseline_bw = (num_weights * 2 * 2) / (baseline_time * 1e9) # Read 2 bytes, write 2 bytes
    
    print(f"--- Baseline Results ---")
    print(f"Avg Time:      {baseline_time*1000:.3f} ms")
    print(f"Baseline BW:   {baseline_bw:.2f} GB/s (Physical read+write)")

    # 5. Speedup Analysis
    speedup = baseline_time / decode_time
    print(f"\n--- Summary ---")
    print(f"SCLP vs BF16 Speedup: {speedup:.2f}x")
    print(f"Theoretical Gain:    {1.333:.2f}x (16 bits / 12 bits)")

if __name__ == "__main__":
    benchmark_throughput()
