import numpy as np


def decode_palette(encoded_data: dict, num_weights: int) -> np.ndarray:
    """
    Decode SCLP compressed weights back to BF16 uint16 bit patterns.

    Expects the format produced by encode_palette():
      ws_stream:  uint8[N]    — byte per weight: palette_idx(7:4) | smn(3:0)
                                smn = sign(3) | mantissa_top3(2:0)
      palette:    uint8[<=16] — exponent values
      sidecar:    optional dict — {indices uint32[], values uint16[]}
    """
    palette   = encoded_data['palette']
    ws_stream = encoded_data['ws_stream']

    ws = ws_stream[:num_weights].astype(np.uint16)

    # Unpack palette index (high nibble) and SM nibble (low nibble)
    p_idx  = (ws >> 4) & 0xF
    smn    = ws & 0xF

    # Look up exponents from palette
    exponents = palette[np.clip(p_idx, 0, len(palette) - 1)].astype(np.uint16)

    # Unpack sign and top-3-bit mantissa from SM nibble
    sign     = (smn >> 3) & 0x1
    mantissa = (smn & 0x7) << 4  # restore to bits 6:4, zeros at 3:0

    # Reconstruct BF16: sign(15) | exponent(14:7) | mantissa_top3(6:4) | 0000
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
    print("Decoder internal test passed!")
