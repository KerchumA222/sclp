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

def benchmark_compare():
    # Matrix dimensions (N=K=M)
    # Using a size that is friendly to tiling
    size = 1024 
    print(f"Benchmarking Fused GEMM vs Transcoder Bridge ({size}x{size}x{size})...")
    
    # 1. Setup Data
    A = np.random.randint(0, 0xFFFF, size=(size, size), dtype=np.uint16)
    B_raw = np.random.randint(0, 0xFFFF, size=(size, size), dtype=np.uint16)
    encoded = encode_palette(B_raw.flatten())
    
    packed = encoded['packed_indices'].astype(np.uint8)
    sm = encoded['sm_stream'].astype(np.uint8)
    palette = encoded['palette'].astype(np.uint8)
    
    # 2. Benchmark Bridge: Decode then GEMM (Mocking GEMM with a dummy operation)
    print("Benchmarking Transcoder Bridge (Decode + GEMM)...")
    start = time.perf_counter()
    # Decode
    decoded = testmodule.decode(packed, sm, palette)
    # Simulate GEMM (just a memory copy/touch as proxy)
    _ = decoded.copy()
    bridge_time = time.perf_counter() - start
    
    # 3. Benchmark Fused GEMM
    print("Benchmarking Fused GEMM...")
    start = time.perf_counter()
    _ = testmodule.fused_gemm(A, packed, sm, palette, size, size, size)
    fused_time = time.perf_counter() - start
    
    print(f"\n--- Performance Results ---")
    print(f"Bridge Time: {bridge_time*1000:.2f} ms")
    print(f"Fused Time:  {fused_time*1000:.2f} ms")
    print(f"Speedup:     {bridge_time/fused_time:.2f}x")

if __name__ == "__main__":
    benchmark_compare()
