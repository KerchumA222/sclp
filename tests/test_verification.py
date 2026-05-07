import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _setup_paths  # noqa: F401

try:
    import testmodule
    print("Successfully imported testmodule")
except ImportError as e:
    print(f"Failed to import testmodule: {e}")
    sys.exit(1)

def test_pipeline_verification():
    """Verify the Python interface and actual kernel computation."""
    print("Running verification test...")
    
    num_weights = 100
    # Create dummy input weights following our custom format: [Sign bit][Exp bits 7-16][Mantissa bits 0-6]
    # We use a specific value so we can check if it survives the roundtrip.
    input_data = np.zeros(num_weights, dtype=np.uint16)
    for i in range(num_weights):
        # Construct a value: Sign=0, Exp=128, Mantissa=1
        val = (0 << 15) | (128 << 7) | 1
        input_data[i] = val

    # Palette: exponent values present in the input (encode accepts palette, not
    # a raw lookup — the wrapper builds the nearest-neighbour lookup internally)
    palette = np.array([128], dtype=np.uint8)

    try:
        # 1. Test Encode Interface
        print("Testing encode interface...")
        result = testmodule.encode(input_data, palette)
        print(f"  - Packed bytes (first 5): {result['packed'][:5].tolist()}")

        packed     = result["packed"]
        sm         = result["sm"]
        sc_indices = result["sidecar_indices"]
        sc_values  = result["sidecar_values"]

        print(f"  - Packed shape: {packed.shape}")
        expected_packed = (num_weights + 1) // 2
        print(f"  - Expected packed shape: ({expected_packed},)")
        assert packed.shape == (expected_packed,)
        print(f"  - SM shape: {sm.shape}")
        expected_sm = (num_weights,)
        print(f"  - Expected SM shape: {expected_sm}")
        assert sm.shape == expected_sm
        print(f"  - Sidecar: {len(sc_indices)} outlier weight(s)")
        print("  - Encode interface: OK")

        # 2. Test Decode Interface
        print("Testing decode interface...")
        output_data = testmodule.decode(packed, sm, palette, sc_indices, sc_values)
        
        assert output_data.shape == (num_weights,)
        assert output_data.dtype == np.uint16
        print("  - Decode interface: OK")

        # 3. Verification of Values
        print("Verifying kernel computation results...")
        # Check if the reconstructed values match our input exactly
        if np.array_equal(input_data, output_data):
            print("  - [SUCCESS] Kernel roundtrip: Input == Output")
        else:
            mismatches = np.where(input_data != output_data)[0]
            print(f"  - [FAILURE] Kernel mismatch detected!")
            print(f"    First 5 mismatches:")
            for idx in mismatches[:5]:
                print(f"      Idx {idx}: Input={input_data[idx]:016b}, Output={output_data[idx]:016b}")
            raise ValueError("Kernel computation error: Roundtrip failed.")

    except Exception as e:
        if "no ROCm-capable device" in str(e):
            print(f"  - Interface test skipped (No GPU found): {e}")
        else:
            print(f"  - Verification FAILED: {e}")
            import traceback
            traceback.print_exc()
            raise e

if __name__ == "__main__":
    test_pipeline_verification()
