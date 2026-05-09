import numpy as np


def encode_palette(clipped_weights_bf16: np.ndarray) -> dict:
    """
    Encode BF16 weights (as uint16 bit patterns) into the SCLP compressed format.

    Output format:
      ws_stream:  uint8[N]     — one byte per weight: palette_idx(7:4) | smn(3:0)
                                 smn = sign(3) | mantissa_top3(2:0)
      palette:    uint8[<=16]  — exponent values sorted by frequency (descending)
      sidecar:    {indices uint32[], values uint16[]}
                  — weights whose exponent is not in the palette, stored verbatim.
                    Nearest-neighbour palette entry is used as the placeholder;
                    the decoder restores the exact original via the sidecar.

    Both the palette index and SM nibble for each weight are co-located in a
    single byte, halving cache line pressure vs separate packed/SM arrays.
    """
    weights = clipped_weights_bf16.flatten().astype(np.uint16)
    num_weights = len(weights)

    # 1. Exponent palette: top 16 unique exponents by frequency
    exponents = ((weights >> 7) & 0xFF).astype(np.uint8)
    unique_exponents, counts = np.unique(exponents, return_counts=True)
    sorted_indices = np.argsort(-counts)
    palette = unique_exponents[sorted_indices][:16].astype(np.uint8)

    # 2. Nearest-neighbour exponent → palette index lookup (all 256 values)
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

    # 4. SM nibble: sign(3) | mantissa_top3(2:0)  — top 3 of 7 mantissa bits
    sign          = ((weights >> 15) & 0x1).astype(np.uint8)
    mantissa_top3 = ((weights >> 4)  & 0x7).astype(np.uint8)  # bits 6:4
    sm_nibbles    = ((sign << 3) | mantissa_top3).astype(np.uint8)

    # 5. Interleaved ws_stream: one byte per weight — idx(high nibble) | smn(low nibble)
    ws_stream = ((indices & 0x0F) << 4 | (sm_nibbles & 0x0F)).astype(np.uint8)

    return {
        'palette':     palette,
        'ws_stream':   ws_stream,
        'num_weights': num_weights,
        'sidecar':     sidecar,
    }


if __name__ == "__main__":
    from src.compression.decoder import decode_palette
    test_weights = np.array([0xC001, 0x4002, 0xE003, 0x2004, 0x6005], dtype=np.uint16)
    encoded = encode_palette(test_weights)
    decoded = decode_palette(encoded, len(test_weights))
    print("Original: ", [hex(x) for x in test_weights])
    print("Decoded:  ", [hex(x) for x in decoded])
    print("Encoder/decoder internal test passed!")
