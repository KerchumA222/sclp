import os
import sys
import numpy as np

# Add project root to sys.path to allow absolute imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.compression.pipeline import SCLPCompressor

def run_integrity_test():

    print("Starting SCLPCompressor Bit-Integrity Test...")
    
    # 1. Initialize Compressor
    compressor = SCLPCompressor(threshold_exponent=125)
    
    # 2. Create random test data (uint16 bits)
    # We'll use a large enough array to see if compression works
    size = 1000000
    original_data = np.random.randint(0, 65535, size=size, dtype=np.uint16)
    
    print(f"Original data size: {size} elements")
    
    # 3. Perform Compression/Decompression
    print("Compressing...")
    compressed_package = compressor.compress(original_data.flatten())
    
    print("Decompressing...")
    reconstructed_data_flat = compressor.decompress(compressed_package)
    
    # 4. Verify
    reconstructed_data = reconstructed_data_flat.reshape(original_data.shape)
    
    # Check if all elements are identical (after masking out the bits we don't preserve)
    # We preserve: Sign (bit 15), Exponent (bits 14-7), and Mantissa bits 6, 5, 4.
    # So we mask out bits 3, 2, 1, 0.
    mask = 0xFFF0 
    is_identical = np.all((original_data & mask) == (reconstructed_data & mask))
    
    print(f"Mask used: {hex(mask)}")
    
    if is_identical:
        print("Bit-integrity check (for preserved bits): PASSED")
    else:
        print("Bit-integrity check (for preserved bits): FAILED")
        # Find first mismatch
        mismatch_idx = np.where((original_data & mask) != (reconstructed_data & mask))[0][0]
        print(f"First mismatch at index {mismatch_idx}")
        print(f"Original (masked): {hex(original_data[mismatch_idx] & mask)}")
        print(f"Reconstructed (masked): {hex(reconstructed_data[mismatch_idx] & mask)}")


if __name__ == "__main__":
    run_integrity_test()
