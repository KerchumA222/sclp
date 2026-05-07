import numpy as np
import os
import sys

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/compression')))

from clipping import soft_exponent_clip
from encoder import encode_palette
from decoder import decode_palette

def test_end_to_end():
    # 1. Generate random BF16-like weights (using uint16 bit patterns)
    num_weights = 1024
    # Create some pattern in exponents to make compression interesting
    # Exponents between 60 and 70
    random_exponents = np.random.randint(60, 71, size=num_weights, dtype=np.uint16)
    random_signs = np.random.randint(0, 2, size=num_weights, dtype=np.uint16)
    random_mantissas = np.random.randint(0, 128, size=num_weights, dtype=np.uint16)
    
    original_weights = (random_signs << 15) | (random_exponents << 7) | random_mantissas

    # 2. Apply Soft Exponent Clipping
    threshold = 65
    clipped_weights = soft_exponent_clip(original_weights, threshold)

    # 3. Encode
    encoded = encode_palette(clipped_weights)

    # 4. Decode
    decoded_weights = decode_palette(encoded, num_weights)

    # 5. Verify
    # Note: Since clipping is lossy, we can't check for equality with original,
    # but we CAN check if the decoded weights match the clipped weights.
    success = np.array_equal(clipped_weights, decoded_weights)
    
    # Calculate error relative to original (for info)
    # Convert bit patterns to float-like values for comparison? 
    # For simplicity in this test, let's just look at the difference in exponents
    orig_exp = (original_weights >> 7) & 0xFF
    clip_exp = (clipped_weights >> 7) & 0xFF
    dec_exp  = (decoded_weights >> 7) & 0xFF
    
    error_count = np.sum(orig_exp != dec_exp)

    print(f"Test End-to-End: {'PASSED' if success else 'FAILED'}")
    print(f"Number of weights with exponent error: {error_count} / {num_weights}")
    print(f"Compression Ratio (approx): {num_weights * 2 / (len(encoded['packed_indices']) + len(encoded['sm_stream'])):.2f}x")

if __name__ == "__main__":
    test_end_to_end()
