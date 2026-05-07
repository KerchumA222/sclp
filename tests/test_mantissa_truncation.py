import numpy as np
import os
import sys

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/compression')))

from clipping import soft_exponent_clip

def test_mantissa_truncation_logic():
    num_weights = 100
        # Create weights with random sign, exponent, and mantissa
    signs = np.random.randint(0, 2, size=num_weights, dtype=np.uint16)
    exponents = np.random.randint(0, 255, size=num_weights, dtype=np.uint16)
    mantissas = np.random.randint(0, 128, size=num_weights, dtype=np.uint16)
    weights = (signs << 15) | (exponents << 7) | mantissas
 # Wait, I messed up the variable name in my head... let's fix it.
    # Re-do properly:
    weights = (signs << 15) | (exponents << 7) | mantissas
    
    threshold = 127
    # Test with no truncation (full mask)
    clipped_full = soft_exponent_clip(weights, threshold, mantissa_mask=0x7F)
    
    # Force some weights to be above threshold to trigger stochastic rounding
    weights[0] |= (1 << 7) # Set exponent bit to something large
    # Since we cannot easily modify the array in place without side effects on the test...
    # Let's just create a fresh set where we KNOW some are above.
    test_weights = np.array([0x4000, 0x3E00], dtype=np.uint16) # Exp: 64 (above threshold if threshold is small), 62
    # Actually, let's use a low threshold for testing
    threshold = 60
    clipped_full = soft_exponent_clip(test_weights, threshold, mantissa_mask=0x7F)
    assert len(clipped_full) == 2

    # Test with extreme truncation (zeroing out mantissa)
    clipped_truncated = soft_exponent_clip(test_weights, threshold, mantissa_mask=0x00)
    # All weights should now have 0 mantissa
    assert np.all((clipped_truncated & 0x7F) == 0)

if __name__ == "__main__":
    test_mantissa_truncation_logic()
    print("Mantissa truncation logic test passed!")
