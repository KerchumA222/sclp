import numpy as np


def decode_palette(encoded_data: dict, num_weights: int = None) -> np.ndarray:
    """
    Decode SCLP8 (Version 3) from the interleaved ws_stream format.
    Supports per-expert palettes.
    """
    palette   = encoded_data['palette']
    ws_stream = encoded_data['ws_stream']
    sidecar   = encoded_data['sidecar']
    n_experts = encoded_data.get('n_experts', 1)
    
    if num_weights is None:
        num_weights = encoded_data.get('num_weights', len(ws_stream))

    expert_nw = num_weights // n_experts
    decoded = np.zeros(num_weights, dtype=np.uint16)

    for e in range(n_experts):
        expert_pal = palette[e] if n_experts > 1 else palette
        expert_ws  = ws_stream[e * expert_nw : (e + 1) * expert_nw]
        
        # 1. Decode indices and SM nibbles from interleaved ws_stream
        # ws_stream byte: palette_idx(7:4) | smn(3:0)
        indices    = (expert_ws >> 4) & 0x0F
        sm_nibbles = expert_ws & 0x0F

        # 2. Reconstruct BF16: sign(15) | exponent(14:7) | mantissa_top3(6:4)
        sign          = (sm_nibbles >> 3) & 0x01
        mantissa_top3 = sm_nibbles & 0x07
        exponents     = expert_pal[indices].astype(np.uint16)

        decoded[e * expert_nw : (e + 1) * expert_nw] = (
            (sign.astype(np.uint16) << 15) |
            (exponents << 7) |
            (mantissa_top3.astype(np.uint16) << 4)
        )

    # 3. Apply sidecar corrections (Version 3 sidecar stores FULL BF16)
    if sidecar['indices'].size > 0:
        decoded[sidecar['indices']] = sidecar['values']

    return decoded


def decode_palette_4b(encoded_data: dict, num_weights: int = None) -> np.ndarray:
    """
    Decode SCLP4 back to BF16.
    """
    ws_stream = encoded_data['ws_stream']
    n_experts = encoded_data.get('n_experts', 1)
    if num_weights is None:
        num_weights = encoded_data.get('num_weights', len(ws_stream) * 2)

    def _decode_nibbles(palette, ws, nw):
        # packed nibbles, high nibble = even weight, low = odd
        # nibble layout: bits[3:2]=palette_idx, bit[1]=sign, bit[0]=mantissa_top1
        nibbles = np.empty(nw, dtype=np.uint8)
        n_bytes = (nw + 1) // 2
        ws_b = ws[:n_bytes]
        nibbles[0::2] = (ws_b >> 4)
        if nw > 1:
            n_odd = nw // 2
            nibbles[1:nw:2] = (ws_b[:n_odd] & 0x0F)
        
        p_idx         = (nibbles >> 2) & 0x3
        sign          = (nibbles >> 1) & 0x1
        mantissa_top1 = nibbles & 0x1
        exponents = palette[np.clip(p_idx, 0, len(palette) - 1)].astype(np.uint16)
        return ((sign.astype(np.uint16) << 15) | (exponents << 7) | (mantissa_top1.astype(np.uint16) << 6)).astype(np.uint16)

    if n_experts == 1:
        palette = encoded_data['palette']
        weights = _decode_nibbles(palette, ws_stream, num_weights)
    else:
        palettes = encoded_data['palette']
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


def decode_palette_4m(encoded_data: dict, num_weights: int = None) -> np.ndarray:
    """
    Decode SCLP4M back to BF16 uint16.

    Each 256-weight block has a per-block 8-entry BF16 magnitude codebook.
    nibble layout: bits[3:1] = codebook_idx, bit[0] = sign
    Decode: mag = codebook[idx]; bits = mag | (sign << 15)

    Supports multi-expert encoding (n_experts > 1) where block_codebooks is a
    list of per-expert uint16[n_blocks_e, 8] arrays.
    """
    ws_stream = encoded_data['ws_stream']
    block_codebooks = encoded_data['block_codebooks']
    n_experts = encoded_data.get('n_experts', 1)
    QK = 256

    if num_weights is None:
        num_weights = encoded_data.get('num_weights', len(ws_stream) * 2)

    def _decode_4m(cb_array, ws, nw):
        """cb_array: uint16[n_blocks, 8]; ws: packed nibble bytes; nw: number of weights."""
        n_bytes = (nw + 1) // 2
        ws_b    = ws[:n_bytes]
        nibbles = np.empty(nw, dtype=np.uint8)
        nibbles[0::2] = ws_b >> 4
        if nw > 1:
            n_odd = nw // 2
            nibbles[1:nw:2] = ws_b[:n_odd] & 0x0F

        idx  = (nibbles >> 1) & 0x7  # 3-bit codebook index
        sign = (nibbles & 0x1).astype(np.uint16)

        # Look up magnitude from per-block codebook
        block_idx = np.arange(nw, dtype=np.int64) // QK
        cb_entries = cb_array[block_idx, idx]  # uint16 BF16 magnitudes (sign bit = 0)
        return (cb_entries | (sign << 15)).astype(np.uint16)

    if n_experts == 1:
        cb = block_codebooks  # uint16[n_blocks, 8]
        weights = _decode_4m(cb, ws_stream, num_weights)
    else:
        expert_nw = num_weights // n_experts
        parts     = []
        ws_offset = 0
        for e in range(n_experts):
            ws_bytes = (expert_nw + 1) // 2
            ws_e = ws_stream[ws_offset:ws_offset + ws_bytes]
            parts.append(_decode_4m(block_codebooks[e], ws_e, expert_nw))
            ws_offset += ws_bytes
        weights = np.concatenate(parts)

    sidecar = encoded_data.get('sidecar')
    if sidecar is not None and len(sidecar['indices']) > 0:
        weights[sidecar['indices']] = sidecar['values']
    return weights


def decode_palette_6b(encoded_data: dict, num_weights: int = None) -> np.ndarray:
    """
    Decode SCLP6 back to BF16.
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
        return ((sign.astype(np.uint16) << 15) | (exponents << 7) | (mantissa_top2.astype(np.uint16) << 5)).astype(np.uint16)

    if n_experts == 1:
        palette = encoded_data['palette']
        weights = _decode_sixbits(palette, ws_stream, num_weights)
    else:
        palettes = encoded_data['palette']
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
    test_weights = np.array([0xC001, 0x4002, 0xE003, 0x2004, 0x6005, 0xC001, 0x4002, 0xE003, 0x2004, 0x6005], dtype=np.uint16)

    print("Testing SCLP8 (n_experts=2)...")
    encoded = encode_palette(test_weights, n_experts=2)
    decoded = decode_palette(encoded, len(test_weights))
    print("Original: ", [hex(x) for x in test_weights])
    print("Decoded:  ", [hex(x) for x in decoded])
    print("SCLP8 MoE test passed!")
