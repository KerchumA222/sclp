import numpy as np
import sys
import os

# Add project root to sys.path to allow absolute imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.compression.encoder import encode_palette
from src.compression.decoder import decode_palette

def test_e2e():
    # Test weights: 0x4080, 0x4280, 0x4080, 0x4480, 0x4000 (5 weights)
    weights = np.array([0x4080, 0x4280, 0x4080, 0x4480, 0x4000], dtype=np.uint16)
    print(f"Original: {[hex(x) for x in weights]}")

    # Encode
    encoded = encode_palette(weights)
    print("Encoded SM Stream (hex):", [hex(b) for b in encoded['sm_stream']])
    
    # Decode
    decoded = decode_palette(encoded, len(weights))
    print(f"Decoded:  {[hex(x) for x in decoded]}")

    # Verify
    assert np.array_equal(weights, decoded), "E2E Test Failed!"
    print("E2E Test Passed!")

if __name__ == "__main__":
    test_e2e()
