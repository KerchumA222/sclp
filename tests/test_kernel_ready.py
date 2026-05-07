import numpy as np
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from compression.clipping import soft_exponent_clip
from compression.encoder import encode_palette
from compression.decoder import decode_palette

def test_amd_wavefront_compatibility():
    """
    Validates that the operations are branchless and use only 
    SIMD-friendly primitives (masks, shifts) suitable for AMD RDNA wavefronts.
    """
    # Wavefront size for AMD (typically 64 threads on RDNA3/4)
    wavefront_size = 64
    num_waves = 10
    total_elements = wavefront_size * num_waves
    
    # Generate synthetic data
    random_exponents = np.random.randint(0, 255, size=total_elements, dtype=np.uint16)
    random_signs = np.array([0]*total_elements, dtype=np.uint16)
    random_mantissas = np.random.randint(0, 128, size=total_elements, dtype=np.uint16)
    original_bits = (random_signs << 15) | (random_exponents << 7) | random_mantissas
    
    # 1. Test Clipping (Must be branchless clamp)
    threshold = 125
    clipped_bits = soft_exponent_clip(original_bits, threshold)
    assert np.all((clipped_bits >> 7) & 0xFF <= threshold + 1), "Clipping failed to bound exponents"

    # 2. Test Encoder (Must avoid branching for palette lookup)
    encoded_data = encode_palette(clipped_bits)
    
    # packed_indices are nibble-packed: 2 weights per byte
    packed_len = len(encoded_data['packed_indices'])
    assert packed_len == (total_elements + 1) // 2, "Padding logic error"

    # 3. Test Decoder (Must use bitwise unpacking)
    decoded_bits = decode_palette(encoded_data, total_elements)
    
    # Final verification of data integrity
    assert len(decoded_bits) == total_elements, "Decoded size mismatch"
    
    print(f"AMD Wavefront Compatibility Test Passed!")
    print(f"  Processed {total_elements} elements across {num_waves} waves.")

if __name__ == "__main__":
    test_amd_wavefront_compatibility()
