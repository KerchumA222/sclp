import numpy as np
import os
import sys

# Add python_pkg to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../python_pkg')))

import testmodule

def test_hip_clip():
    num_weights = 1024
    # Create random weights
    rng = np.random.default_rng(42)
    weights = rng.integers(0, 65535, size=num_weights, dtype=np.uint16)
    threshold = 64
    seed = 42
    mask = 0x0F # Truncate mantissa to 4 bits

    print(f"Testing HIP clip with threshold={threshold} and mask={hex(mask)}")
    
    # Call the new HIP kernel via wrapper
    clipped = testmodule.clip(weights, threshold, seed, mask)
    
    # Verify truncation: all mantissa bits above 0x0F should be zeroed
    for i in range(len(clipped)):
        new_mantissa = clipped[i] & 0x7F
        # Check if any bit in the range [0x10, 0x7F] is set
        if (new_mantissa & 0x70) != 0:
            print(f"Error: Mantissa truncation failed at index {i}!")
            print(f"Weight before: {weights[i]:016b}")
            print(f"Weight after:  {clipped[int(i)]:016b}")
            sys.exit(1)
    
    print("Success: Mantissa truncation verified in HIP kernel!")

if __name__ == "__main__":
    test_hip_clip()
