import time
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.abspath("../../src"))
sys.path.append(os.path.abspath("../../src/compression"))
sys.path.append(os.path.abspath("../../src/utils"))

try:
    from compression.pipeline import SCLPCompressor
except ImportError as e:
    print(f"Failed to import pipeline: {numpy_error_handler(e)}")
    SCLPCompressor = None

def numpy_error_handler(e):
    return str(e)

class HIPAutotuner:
    """
    A framework for benchmarking Path A (Fused) vs Path B (Decompress-first).
    Designed to be extended with real ROCm calls on GPU hardware.
    """
    def __init__(self, module=None, compressor=None):
        self.module = module # This would be the 'testmodule' C++ extension
        self.compressor = compressor # SCLPCompressor (CPU reference)
        self.results = {}

    def benchmark_path_a(self, weights, query, batch_size):
        """Simulates Path A: Fused Decode-GEMM."""
        if self.module is None:
            # CPU Proxy/Fallback for benchmarking logic validation
            time.sleep(0.01) # Simulate latency
            return 10.0 / batch_size 
        
        start = time.perf_counter()
        try:
            # In real usage: self.module.fused_gemm(...)
            _ = self.module.clip(weights, 1.0, 42) # Simplified for simulation
            _ = self.module.encode(weights, np.arange(256, dtype=np.uint8))
        except Exception:
            pass
        end = time.perf_counter()
        return (end - start) / batch_size

    def benchmark_path_b(self, weights, query, batch_size):
        """Simulates Path B: Decompress-to-buffer + rocBLAS GEMM."""
        if self.compressor is None:
            # Pure simulation fallback if no compressor provided
            time.sleep(0.05) 
            return 50.0 / batch_size
        
        start = time.perf_counter()
        try:
            # Real Path B logic (CPU-side): Decompress weights, then run dense GEMM
            compressed_pkg = self.compressor.compress(weights)
            reconstructed = self.compressor.decompress(compressed_pkg)
            
            # Simulate dense matrix multiplication on reconstructed weights
            # In real usage: rocblas_sgemm(...)
            # Reconstructed is the flattened weights. We reshape it to match query dimension for matmul.
            # If query is (batch, N), then reconstructed should be (N, K) or similar.
            # Here we assume query is (B, N) and we want result (B, 1).
            # So we reshape reconstructed to (N, 1).
            reconstructed_reshaped = reconstructed.reshape(-1, 1)
            _ = np.matmul(query, reconstructed_reshaped)
        except Exception as e:
            print(f"Path B error: {e}")
            return float('inf')
            
        end = time.perf_counter()
        return (end - start) / batch_size

    def run_benchmark(self, weights, query, batch_sizes=[1, 4, 8, 16]):
        print(f"{'Batch Size':<12} | {'Path A (ms)':<15} | {'Path B (ms)':<15} | {'Winner':<10}")
        print("-" * 60)
        
        for bs in batch_sizes:
            # For simulation, we need to scale query size with batch size
            q = query.repeat(bs, axis=0) if bs > 1 else query
            t_a = self.benchmark_path_a(weights, q, bs)
            t_b = self.benchmark_path_b(weights, q, bs)
            winner = "Path A" if t_a < t_b else "Path B"
            print(f"{bs:<12} | {t_a:<15.4f} | {t_b:<15.4f} | {winner:<10}")
            self.results[bs] = winner

        return self.results

if __name__ == "__main__":
    # Dummy data for testing the autotuner framework
    weights_bf16 = (np.random.rand(1024) * 65535).astype(np.uint16)
    query_matrix = np.random.rand(1, 1024).astype(np.float32)
    
    compressor_ref = SCLPCompressor() if SCLPCompressor else None

    # Test 1: Pure Simulation (No module, No compressor)
    print("Running Autotuner Simulation (Pure Mock)...")
    autotoper_sim = HIPAutotuner(module=None, compressor=None)
    autotoper_sim.run_benchmark(weights_bf16, query_matrix)

    # Test 2: Realistic Path B simulation using the real CPU Compressor
    print("\nRunning Auttuner with Real CPU Compression (Path B Simulation)...")
    autotoper_ref = HIPAutotuner(module=None, compressor=compressor_ref)
    autotoper_ref.run_benchmark(weights_bf16, query_matrix)

    # Test 3: Attempt real module if available
    try:
        import testmodule
        print("\nAttempting Real Hardware Benchmark (Requires GPU)...")
        autotoper_real = HIPAutotuner(module=testmodule, compressor=compressor_ref)
        autotoper_real.run_benchmark(weights_bf16, query_matrix)
    except Exception as e:
        print(f"\nReal hardware benchmark skipped or failed: {e}")


