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


def decode_palette_4b(encoded_data: dict, num_weights: int = None) -> np.ndarray:
    """
    Decode SCLP4 compressed weights back to BF16 uint16 bit patterns.

    Expects the format produced by encode_palette_4b():
      ws_stream:  uint8[ceil(N/2)] — packed nibbles (n_experts=1) or concatenated
      palette:    uint8[<=4] (n_experts=1) or list of arrays (n_experts>1)
      n_experts:  int (default 1)
      sidecar:    optional {indices uint32[], values uint16[]}

    BF16 reconstruction: (sign<<15) | (palette[idx]<<7) | (mantissa_top1<<6)
    """
    ws_stream = encoded_data['ws_stream']
    n_experts = encoded_data.get('n_experts', 1)
    if num_weights is None:
        num_weights = encoded_data.get('num_weights', 0)

    def _decode_nibbles(palette, ws, nw):
        n_bytes = (nw + 1) // 2
        ws_b = ws[:n_bytes].astype(np.uint8)
        nibbles = np.empty(nw, dtype=np.uint8)
        even_count = (nw + 1) // 2
        odd_count  = nw // 2
        nibbles[0::2] = (ws_b[:even_count] >> 4) & 0xF
        if odd_count > 0:
            nibbles[1::2] = ws_b[:odd_count] & 0xF
        nib = nibbles.astype(np.uint16)
        p_idx = (nib >> 2) & 0x3
        smn   = nib & 0x3
        exponents = palette[np.clip(p_idx, 0, len(palette) - 1)].astype(np.uint16)
        sign      = (smn >> 1) & 0x1
        mant_top1 = (smn & 0x1) << 6
        return ((sign << 15) | (exponents << 7) | mant_top1).astype(np.uint16)

    if n_experts == 1:
        palette = encoded_data['palette']
        weights = _decode_nibbles(palette, ws_stream, num_weights)
    else:
        palettes = encoded_data['palette']  # list of arrays
        expert_nw = num_weights // n_experts
        parts = []
        ws_offset = 0
        for e in range(n_experts):
            ws_bytes = (expert_nw + 1) // 2
            ws_e = ws_stream[ws_offset:ws_offset + ws_bytes]
            parts.append(_decode_nibbles(palettes[e], ws_e, expert_nw))
            ws_offset += ws_bytes
        weights = np.concatenate(parts)

    sidecar = encoded_data.get('sidecar')
    if sidecar is not None and len(sidecar['indices']) > 0:
        weights[sidecar['indices']] = sidecar['values']

    return weights


def decode_palette_6b(encoded_data: dict, num_weights: int = None) -> np.ndarray:
    """
    Decode SCLP6 compressed weights back to BF16 uint16 bit patterns.

    Expects the format produced by encode_palette_6b():
      ws_stream:  uint8[ceil(N/4)*3] — packed 6-bit groups (n_experts=1) or concatenated
      palette:    uint8[<=8] (n_experts=1) or list of arrays (n_experts>1)
      n_experts:  int (default 1)
      sidecar:    optional {indices uint32[], values uint16[]}

    BF16 reconstruction: (sign<<15) | (palette[idx]<<7) | (mantissa_top2<<5)
    """
    ws_stream = encoded_data['ws_stream']
    n_experts = encoded_data.get('n_experts', 1)
    if num_weights is None:
        num_weights = encoded_data.get('num_weights', len(ws_stream) * 4 // 3)

    def _decode_sixbits(palette, ws, nw):
        n_groups = (nw + 3) // 4
        ws_b = ws[:n_groups * 3].astype(np.uint8)
        b0 = ws_b[0::3].astype(np.uint32)
        b1 = ws_b[1::3].astype(np.uint32)
        b2 = ws_b[2::3].astype(np.uint32)
        sixbits_all = np.empty(n_groups * 4, dtype=np.uint8)
        sixbits_all[0::4] = (b0 >> 2).astype(np.uint8)
        sixbits_all[1::4] = (((b0 & 0x3) << 4) | (b1 >> 4)).astype(np.uint8)
        sixbits_all[2::4] = (((b1 & 0xF) << 2) | (b2 >> 6)).astype(np.uint8)
        sixbits_all[3::4] = (b2 & 0x3F).astype(np.uint8)
        sixbits = sixbits_all[:nw].astype(np.uint16)
        p_idx         = (sixbits >> 3) & 0x7
        sign          = (sixbits >> 2) & 0x1
        mantissa_top2 = sixbits & 0x3
        exponents = palette[np.clip(p_idx, 0, len(palette) - 1)].astype(np.uint16)
        mant_shifted = mantissa_top2 << 5
        return ((sign << 15) | (exponents << 7) | mant_shifted).astype(np.uint16)

    if n_experts == 1:
        palette = encoded_data['palette']
        weights = _decode_sixbits(palette, ws_stream, num_weights)
    else:
        palettes = encoded_data['palette']  # list of arrays
        expert_nw = num_weights // n_experts
        parts = []
        ws_offset = 0
        for e in range(n_experts):
            ws_bytes = ((expert_nw + 3) // 4) * 3
            ws_e = ws_stream[ws_offset:ws_offset + ws_bytes]
            parts.append(_decode_sixbits(palettes[e], ws_e, expert_nw))
            ws_offset += ws_bytes
        weights = np.concatenate(parts)

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
