import numpy as np
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

def test_fused_gemm():
    # Matrix dimensions
    M, N, K = 16, 16, 16
    
    # 1. Generate test data
    # Matrix A: BF16 (uint16 representation)
    A = np.random.randint(0, 0xFFFF, size=(M, K), dtype=np.uint16)
    
    # Matrix B: Compressed (SCLP)
    # Generate random data and compress it
    B_raw = np.random.randint(0, 0xFFFF, size=(K, N), dtype=np.uint16)
    encoded = encode_palette(B_raw.flatten())
    
    packed = encoded['packed_indices'].astype(np.uint8)
    sm = encoded['sm_stream'].astype(np.uint8)
    palette = encoded['palette'].astype(np.uint8)
    
    # 2. Run Fused GEMM
    # Note: Our fused_gemm expects A as uint16 (raw BF16 bits)
    print("Running fused_gemm...")
    C_fused = testmodule.fused_gemm(A, packed, sm, palette, M, N, K)
    print(f"Result shape: {C_fused.shape}")
    
    print("Fused GEMM test passed (kernel launched successfully)!")

if __name__ == "__main__":
    test_fused_gemm()
