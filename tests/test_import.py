import sys
import os
sys.path.append('/home/ajkerchum/poc/python_pkg')

try:
    import testmodule
    print("Successfully imported testmodule")
    # Try calling one of the functions
    print("Testing clip function...")
    import numpy as np
    input_data = np.arange(10, dtype=np.uint16)
    threshold = 5
    seed = 42
    mask = 0x7F # Default mask
    output = testmodule.clip(input_data, threshold, seed, mask)
    print("Clip function executed (at enough arguments to satisfy pybind11)")

except Exception as e:
    print(f"Error during testing: {e}")
