import numpy as np


def decode_palette(encoded_data: dict, num_weights: int) -> np.ndarray:
    """
    Decode SCLP compressed weights back to BF16 uint16 bit patterns.

    Expects the format produced by encode_palette():
      packed_indices: uint8[ceil(N/2)] — nibble-packed, high nibble = even weight
      sm_stream:      uint8[ceil(N/2)] — nibble per weight: sign(3) | mantissa_top3(2:0), nibble-packed
      palette:        uint8[<=16]      — exponent values
      sidecar:        optional dict    — {indices uint32[], values uint16[]}
    """
    palette       = encoded_data['palette']
    packed        = encoded_data['packed_indices']
    sm_stream     = encoded_data['sm_stream']

    # Unpack 4-bit indices from nibble-packed array
    pos     = np.arange(num_weights)
    is_odd  = (pos % 2).astype(bool)
    pb      = packed[pos // 2].astype(np.uint8)
    nibbles = np.where(is_odd, pb & 0x0F, pb >> 4).astype(np.intp)

    # Look up exponents from palette
    exponents = palette[np.clip(nibbles, 0, len(palette) - 1)].astype(np.uint16)

    # Unpack sign and top-3-bit mantissa from nibble-packed SM stream
    sm_pos       = np.arange(num_weights)
    sm_is_odd    = (sm_pos % 2).astype(bool)
    sm_pb        = sm_stream[sm_pos // 2].astype(np.uint8)
    sm_nibbles   = np.where(sm_is_odd, sm_pb & 0x0F, sm_pb >> 4).astype(np.uint16)
    sign         = (sm_nibbles >> 3) & 0x1
    mantissa     = (sm_nibbles & 0x7) << 4  # restore to bits 6:4, zeros at 3:0

    # Reconstruct BF16 bit pattern: sign(15) | exponent(14:7) | mantissa(6:0)
    weights = ((sign << 15) | (exponents << 7) | mantissa).astype(np.uint16)

    # Restore outlier weights stored verbatim in the sidecar
    sidecar = encoded_data.get('sidecar')
    if sidecar is not None and len(sidecar['indices']) > 0:
        weights[sidecar['indices']] = sidecar['values']

    return weights


if __name__ == "__main__":
    from src.compression.encoder import encode_palette
    test_weights = np.array([0xC001, 0x4002, 0xE003, 0x2004, 0x6005], dtype=np.uint16)
    encoded = encode_palette(test_weights)
    decoded = decode_palette(encoded, len(test_weights))
    print("Original: ", [hex(x) for x in test_weights])
    print("Decoded:  ", [hex(x) for x in decoded])
    assert np.array_equal(test_weights, decoded), "E2E Test Failed!"
    print("Decoder internal test passed!")
