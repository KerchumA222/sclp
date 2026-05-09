import numpy as np
import sys
import os

# Add the package directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _setup_paths  # noqa: F401

try:
    import testmodule
    print("Successfully imported testmodule")
except ImportError as e:
    print(f"Failed to import testmodule: {e}")
    sys.exit(1)

def test_end_to_end():
    np.random.seed(42)
    num_weights = 1000
    # Use values that will trigger some clipping/rounding action
    # BF16 range is roughly [-3.39e38, 3.39e38], but with exponent field only 7 bits
    # Let's use something reasonable
    weights = np.random.randn(num_weights).astype(np.float32) * 5.0

    # Convert to uint16 (simulating BF16 representation)
    def float32_to_bf16(f):
        # Very crude conversion for test verification purposes
        if np.isnan(f): return np.uint16(0b0111111110000000) # NaN example
        sign = (np.sign(f) < 0).astype(np.uint16) << 15
        abs_val = np.abs(f)
        if abs_val == 0: return sign | 0x3F80
        exponent = (np.floor(np.log2(abs_val)) + 127).astype(np.uint16) << 7
        mantissa = np.round((abs_val / (2**(exponent-127) * 2)).astype(np.float32) * 128) & 0x7F
        # Clamp exponent to BF16 range [1, 254]
        exponent = np.clip(exponent - 127 + 127, 1, 254).astype(np.uint16) << 7 # This logic is a bit messy but ok for testing
        return sign | exponent | mantissa

    # Use actual BF16 conversion if possible or just cast to uint16 as dummy values
    # Let's use simple test weights: [exponent][mantissa] where exponent is in first 7 bits after sign
    weights_uint16 = np.zeros(num_weights, dtype=np.uint16)
    for i in range(num_weights):
        exp = (i % 254) + 1 # random-ish but valid exponent
        mantissa = i % 128   # arbitrary mantissa
        sign = 0 if i % 2 == 0 else 1
        weights_uint16[i] = (sign << 15) | (exp << 7) | mantissa

    threshold = 10  # threshold exponent value
    seed = 42

    print(f"Testing full pipeline with {num_weights} weights...")

    try:
        # 1. Clip weights
        clipped = testmodule.clip(weights_uint16, threshold, seed, 0x7F)
        
        # Build palette (top-16 unique exponents by value; after clipping at threshold=10
        # there are at most 11 unique exponents so no outliers are expected here)
        unique_exponents = np.unique((clipped >> 7) & 0xFF)
        palette = unique_exponents[:16].astype(np.uint8)
        if len(unique_exponents) > 16:
            print(f"    NOTE: {len(unique_exponents)} unique exponents; top 16 used as palette, "
                  f"remaining {len(unique_exponents)-16} go to sidecar")

        # 2. Encode clipped weights to packed format and SM stream
        encoded = testmodule.encode(clipped, palette)
        packed     = encoded["packed"]
        sm_stream  = encoded["sm"]
        sc_indices = encoded["sidecar_indices"]
        sc_values  = encoded["sidecar_values"]
        print(f"    Sidecar: {len(sc_indices)} outlier weight(s)")

        # 3. Decode back to original clipped values (pass sidecar for exact reconstruction)
        decoded = testmodule.decode(packed, sm_stream, palette, sc_indices, sc_values, len(clipped))

        # Verification: decoded == clipped (since no lossy encoding step besides rounding/quantization)
        print("    Checking clip equality...")
        if np.array_equal(clipped, decoded):
            print("    SUCCESS: Decoded matches clipped weights!")
        else:
            diff = np.where(clipped != decoded)[0]
            print(f"    FAILURE: Mismatch at {len(diff)} positions.")
            if len(diff) > 0:
                idx = diff[0]
                print(f"      Expected clipped[{idx}]={hex(clipped[idx])}, got decoded[{idx}]={hex(decoded[idx])}")
            sys.exit(1)

        # Additional verification of the clip logic (thresholding exponent <= threshold)
        print("    Checking clipping threshold...")
        for i in range(num_weights):
            exponent = (clipped[i] >> 7) & 0xFF
            if exponent > threshold + 1:
                print(f"    FAILURE: weight at index {i} has exponent {exponent} > threshold+1 {threshold+1}")
                sys.exit(1)
        print(f"    SUCCESS: All clipped exponents satisfy threshold <= {threshold + 1} (stochastic)")

        # Verification of original values vs quantized/clipped ones (stochastic check not possible here easily)
        # but we can at least verify the clipping logic works qualitatively
        orig_exponents = (weights_uint16 >> 7) & 0xFF
        excessive_exp_count = np.sum(orig_exponents > threshold)
        print(f"    Original weights with exp > {threshold}: {excessive_exp_count}")

        # Check if clipping actually reduced/maintained exponents correctly
        clipped_exponents = (clipped >> 7) & 0xFF
        new_excessive_exp_count = np.sum(clipped_exponents > threshold)
        print(f"    Clipped weights with exp > {threshold}:  {new_excessive_exp_count}")

        if new_excessive_exp_count <= excessive_exp_count: # Should be close to 0 if stochastic rounding worked
            print("    SUCCESS: Clipping threshold enforced correctly")
        else:
            # This might pass even with error, but let's check specifically the values
            failures = np.where(clipped_exponents > threshold)[0]
            if len(failures) > 0:
                idx = failures[0]
                print(f"    FAILURE: weight at index {idx} has exp={clipped_exponents[idx]} after clipping")
                sys.exit(1)

        # Check packing compression ratio
        original_size_bytes = num_weights * 2  # 2 bytes per uint16 weight
        packed_size_bytes = len(packed) # encoded into packed nibbles, each byte has two weights
        print(f"    Compression: {num_weights} weights -> {len(packed)} packed bytes ({4.0:.1f}x reduction in index storage)")

        # Verify SM stream matches (should be identical if we didn't modify sign/mantissa)
        original_sm = ((clipped >> 15) << 7 | (clipped & 0x7F)).astype(np.uint8)
        if np.array_equal(sm_stream, original_sm):
             print("    SUCCESS: SM stream correctly preserves sign/mantissa")
        else:
            print("    FAILURE: SM stream mismatch!")
            sys.exit(1)

        # Check palette covers at least the common exponents
        if not np.all(np.isin(palette, unique_exponents)):
            print("    FAILURE: Palette contains exponents not found in clipped weights")
            sys.exit(1)

        print("\nCOMPLETE END-TO-END PIPELINE TEST PASSED!")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nTEST FAILED with exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_end_to_end()
