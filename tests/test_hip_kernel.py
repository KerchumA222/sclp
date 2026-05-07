import sys
import os
import numpy as np

# Add the directory containing the module to sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'python_pkg')))

try:
    import testmodule
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_hip_clipping_with_mask():
    # Prepare dummy weights
    # 0x40FF -> sign=0, exp=64, mantissa=0x7F
    # 0xC0FF -> sign=1, exp=64, mantissa=0x7F
    # 0x4280 -> sign=0, exp=65, mantissa=0x00
    weights = np.array([0x40FF, 0xC0FF, 0x4280], dtype=np.uint16)
    threshold = 0xFF  # Set threshold high so exponent doesn't change
    seed = 42
    mantissa_mask = 0x0F # Truncate mantissa to 4 bits
    
    print(f"Input weights: {[hex(x) for x in weights]}")
    print(f"Threshold: {hex(threshold)}, Mask: {hex(mantissa_mask)}")

    # Call the HIP kernel via pybind11 wrapper
    output = testmodule.clip(weights, threshold, seed, mantissa_mask)
    
    # Expected outputs (since threshold is high, only mantissa is truncated):
    # 0x40FF & 0xF -> sign=0, exp=0x81, mantissa=0x0F -> 0x408F
    # 0xC0FF & 0xF -> sign=1, exp=0x81, mantissa=0x0F -> 0xC08F
    # 0x4280 & 0xF -> sign=0, exp=0x85, mantissa=0x00 -> 0x4280
    expected = np.array([0x408F, 0xC08F, 0x4280], dtype=np.uint16)
    
    print(f"Output weights: {[hex(x) for x in output]}")
    print(f"Expected weights: {[hex(x) for x in expected]}")
    
    assert np.array_equal(output, expected), "Kernel output does not match expectation!"
    print("HIP Kernel verification SUCCESSFUL!")

if __name__ == "__main__":
    test_hip_clipping_with_mask()
