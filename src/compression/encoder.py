import numpy as np


def encode_palette(clipped_weights_bf16: np.ndarray) -> dict:
    """
    Encode BF16 weights (as uint16 bit patterns) into the SCLP compressed format.

    Output format matches the HIP GPU encoder:
      packed_indices: uint8[ceil(N/2)]  — nibble-packed palette indices,
                                          high nibble = even weight, low nibble = odd weight
      sm_stream:      uint8[ceil(N/2)]  — nibble per weight: sign(3) | mantissa_top3(2:0), nibble-packed
      palette:        uint8[<=16]       — exponent values sorted by frequency (descending)
      sidecar:        {indices uint32[], values uint16[]}
                      — weights whose exponent is not in the palette, stored verbatim.
                        Nearest-neighbour palette entry is used as the packed placeholder;
                        the decoder restores the exact original via the sidecar.
    """
    weights = clipped_weights_bf16.flatten().astype(np.uint16)
    num_weights = len(weights)

    # 1. Exponent palette: top 16 unique exponents by frequency
    exponents = ((weights >> 7) & 0xFF).astype(np.uint8)
    unique_exponents, counts = np.unique(exponents, return_counts=True)
    sorted_indices = np.argsort(-counts)
    palette = unique_exponents[sorted_indices][:16].astype(np.uint8)

    # 2. Nearest-neighbour exponent → palette index lookup (all 256 values)
    #    Weights outside the palette get the closest entry as a placeholder in
    #    packed_indices; the decoder will restore them via the sidecar.
    exp_lookup = np.argmin(
        np.abs(np.arange(256, dtype=np.int16)[:, None] -
               palette.astype(np.int16)[None, :]),
        axis=1
    ).astype(np.uint8)
    indices = exp_lookup[exponents].astype(np.uint8)

    # 3. Sidecar: positions where the original exponent is not in the palette
    in_palette = np.isin(exponents, palette)
    outlier_positions = np.where(~in_palette)[0].astype(np.uint32)
    sidecar = {
        'indices': outlier_positions,
        'values':  weights[outlier_positions].astype(np.uint16),
    }

    # 4. Nibble-pack indices: 2 per byte, high nibble = even weight, low nibble = odd
    packed_size = (num_weights + 1) // 2
    packed_nibbles = np.zeros(packed_size, dtype=np.uint8)
    even_idx = np.arange(0, num_weights, 2)
    odd_idx  = np.arange(1, num_weights, 2)
    packed_nibbles[:len(even_idx)] |= ((indices[even_idx] & 0x0F) << 4).astype(np.uint8)
    if len(odd_idx) > 0:
        packed_nibbles[:len(odd_idx)] |= (indices[odd_idx] & 0x0F).astype(np.uint8)

    # 5. SM stream: nibble per weight — sign(3) | mantissa_top3(2:0)
    #    Keeps the 3 most significant mantissa bits; zeros bits 3:0.
    #    Nibble-packed: high nibble = even weight, low nibble = odd weight.
    #    Halves SM bandwidth vs the old byte-per-weight encoding.
    sign          = ((weights >> 15) & 0x1).astype(np.uint8)
    mantissa_top3 = ((weights >> 4) & 0x7).astype(np.uint8)  # bits 6:4
    sm_nibbles    = ((sign << 3) | mantissa_top3).astype(np.uint8)

    sm_size   = (num_weights + 1) // 2
    sm_stream = np.zeros(sm_size, dtype=np.uint8)
    even_w = np.arange(0, num_weights, 2)
    odd_w  = np.arange(1, num_weights, 2)
    sm_stream[:len(even_w)] |= (sm_nibbles[even_w] << 4).astype(np.uint8)
    if len(odd_w) > 0:
        sm_stream[:len(odd_w)] |= sm_nibbles[odd_w].astype(np.uint8)

    return {
        'palette':        palette,
        'packed_indices': packed_nibbles,
        'sm_stream':      sm_stream,
        'num_weights':    num_weights,
        'sidecar':        sidecar,
    }


if __name__ == "__main__":
    from src.compression.decoder import decode_palette
    test_weights = np.array([0xC001, 0x4002, 0xE003, 0x2004, 0x6005], dtype=np.uint16)
    encoded = encode_palette(test_weights)
    decoded = decode_palette(encoded, len(test_weights))
    print("Original: ", [hex(x) for x in test_weights])
    print("Decoded:  ", [hex(x) for x in decoded])
    assert np.array_equal(test_weights, decoded), "E2E Test Failed!"
    print("Encoder/decoder internal test passed!")
