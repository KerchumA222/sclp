import sys
import os
import numpy as np

sys.path.append('/home/ajkerchum/poc/python_pkg')

try:
    import testmodule
    print("Successfully imported testmodule")
except ImportError as e:
    print(f"Failed to import testmodule: {e}")
    sys.exit(1)

def test_pipeline_interface():
    """Verify the Python interface for encode and decode (shapes/types)."""
    print("Running interface test...")
    
    num_weights = 100
    # Create dummy input weights following our custom format: [Sign bit][Exp bits 7-14][Mantissa bits 0-6]
    input_data = np.zeros(num_weights, dtype=np.uint16)
    for i in range(num_weights):
        # Construct a value: Sign=0, Exp=128, Mantissa=1
        val = (0 << 15) | (128 << 7) | 1
        input_data[i] = val

    # Mock lookup table (256 entries)
    lookup = np.arange(256, dtype=np.uint8)

    try:
        # 1. Test Encode Interface
        print("Testing encode interface...")
        result = testmodule.encode(input_data, lookup)
        
        print(f"  - Packed shape: {result['packed'].shape}")
        expected_packed = (num_weights + 1) // 2
        print(f"  - Expected packed shape: ({expected_packed},)")
        assert result["packed"].shape == (expected_packed,)
        print(f"  - SM shape: {result['sm'].shape}")
        expected_sm = (num_weights,)
        print(f"  - Expected SM shape: {expected_sm}")
        assert result["sm"].shape == expected_sm
        print("  - Encode interface: OK")

        # 2. Test Decode Interface
        print("Testing decode interface...")
        packed = result["packed"]
        sm = result["sm"]
        palette = lookup # use same as lookup for simplicity in this test
        
        output_data = testmodule.decode(packed, sm, palette)
        print(f"  - Output shape: {output_data.shape}")
        assert output_data.shape == (num_weights,)
        assert output_data.dtype == np.uint16
        print("  - Decode interface: OK")

    except Exception as e:
        # We expect failure on non-GPU systems, but we want to catch it specifically
        if "no ROCm-capable device" in str(e):
            print(f"  - Interface test skipped (No GPU found): {e}")
        else:
            print(f"  - Interface test FAILED: {e}")
            import traceback
            traceback.print_exc()
            raise e

if __name__ == "__main__":
    test_pipeline_interface()
