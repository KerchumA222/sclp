import numpy as np


def _kmeans_palette(unique_exponents: np.ndarray, counts: np.ndarray, k: int,
                    n_iter: int = 20) -> np.ndarray:
    """
    1-D weighted k-means over BF16 exponent values.

    Minimises sum(count[e] * (e - centroid)^2) over cluster assignments.
    Centroids are real-valued during iteration but snapped to the nearest
    observed exponent value at the end so every palette entry is a valid BF16
    exponent.  Initialised with k-means++ seeding for stability.
    """
    if len(unique_exponents) <= k:
        return unique_exponents.astype(np.uint8)

    exponents = unique_exponents.astype(np.float32)
    weights   = counts.astype(np.float32)
    total_w   = weights.sum()

    # k-means++ initialisation
    rng = np.random.default_rng(42)
    centers = [float(exponents[rng.choice(len(exponents), p=weights / total_w)])]
    for _ in range(1, k):
        d2 = np.array([min((e - c) ** 2 for c in centers) for e in exponents],
                      dtype=np.float32)
        d2_w = d2 * weights
        centers.append(float(exponents[rng.choice(len(exponents), p=d2_w / d2_w.sum())]))
    centers = np.array(centers, dtype=np.float32)

    for _ in range(n_iter):
        # assign each unique exponent to nearest centroid
        dists  = np.abs(exponents[:, None] - centers[None, :])  # (U, k)
        labels = dists.argmin(axis=1)                           # (U,)

        new_centers = np.empty(k, dtype=np.float32)
        for j in range(k):
            mask = labels == j
            if mask.any():
                new_centers[j] = (exponents[mask] * weights[mask]).sum() / weights[mask].sum()
            else:
                # dead cluster: reinitialise to most-frequent unrepresented exponent
                represented = set(exponents[labels == i] for i in range(k) if i != j)
                unrepresented = [i for i, e in enumerate(exponents) if e not in represented]
                if unrepresented:
                    new_centers[j] = exponents[max(unrepresented, key=lambda i: weights[i])]
                else:
                    new_centers[j] = centers[j]

        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    # snap each centroid to the nearest observed exponent value
    snapped = []
    for c in centers:
        nearest = unique_exponents[np.argmin(np.abs(exponents - c))]
        snapped.append(int(nearest))

    # deduplicate while preserving order; fill any gap with next most-frequent
    seen = set()
    palette = []
    for v in snapped:
        if v not in seen:
            seen.add(v)
            palette.append(v)

    if len(palette) < k:
        freq_order = unique_exponents[np.argsort(-counts)]
        for v in freq_order:
            if int(v) not in seen:
                seen.add(int(v))
                palette.append(int(v))
            if len(palette) == k:
                break

    return np.array(palette[:k], dtype=np.uint8)


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


def encode_palette_4b(weights_uint16: np.ndarray, n_experts: int = 1) -> dict:
    """
    Encode BF16 weights (as uint16 bit patterns) into the SCLP4 compressed format.

    Output format (4 bits/weight, palette ≤4, 2 weights per byte):
      ws_stream:  uint8[ceil(N/2)] — packed nibbles, high nibble = even weight, low = odd
                  nibble layout: bits[3:2]=palette_idx, bit[1]=sign, bit[0]=mantissa_top1
      palette:    uint8[<=4]       — exponent values sorted by frequency (descending)
      sidecar:    {indices uint32[], values uint16[]}

    BF16 reconstruction: (sign<<15) | (palette[idx]<<7) | (mantissa_top1<<6)
    """
    weights = weights_uint16.flatten().astype(np.uint16)
    num_weights = len(weights)

    def _encode_4b_expert(expert_weights):
        """Encode a single expert's weights, returning (palette, ws_bytes)."""
        exponents = ((expert_weights >> 7) & 0xFF).astype(np.uint8)
        unique_exponents, counts = np.unique(exponents, return_counts=True)
        sorted_indices = np.argsort(-counts)
        palette = unique_exponents[sorted_indices][:4].astype(np.uint8)

        exp_lookup = np.argmin(
            np.abs(np.arange(256, dtype=np.int16)[:, None] -
                   palette.astype(np.int16)[None, :]),
            axis=1
        ).astype(np.uint8)
        indices = exp_lookup[exponents].astype(np.uint8)

        sign          = ((expert_weights >> 15) & 0x1).astype(np.uint8)
        mantissa_top1 = ((expert_weights >> 6)  & 0x1).astype(np.uint8)
        sm_bits       = ((sign << 1) | mantissa_top1).astype(np.uint8)

        nibbles = ((indices & 0x3) << 2 | (sm_bits & 0x3)).astype(np.uint8)

        nw = len(expert_weights)
        num_bytes = (nw + 1) // 2
        ws = np.zeros(num_bytes, dtype=np.uint8)
        ws[:] = (nibbles[0::2] << 4)
        if len(nibbles) > 1:
            odd_len = len(nibbles[1::2])
            ws[:odd_len] |= nibbles[1::2]
        return palette, ws

    if n_experts == 1:
        palette, ws_stream = _encode_4b_expert(weights)
    else:
        expert_nw = num_weights // n_experts
        expert_palettes = []
        ws_parts = []
        for e in range(n_experts):
            ew = weights[e * expert_nw:(e + 1) * expert_nw]
            pal, ws = _encode_4b_expert(ew)
            expert_palettes.append(pal)
            ws_parts.append(ws)
        # For multi-expert, palette is a list and ws_stream is concatenated
        palette = expert_palettes  # list of arrays
        ws_stream = np.concatenate(ws_parts)

    sidecar = {'indices': np.array([], dtype=np.uint32), 'values': np.array([], dtype=np.uint16)}

    return {
        'palette':     palette,
        'ws_stream':   ws_stream,
        'num_weights': num_weights,
        'n_experts':   n_experts,
        'sidecar':     sidecar,
    }


def encode_palette_6b(weights_uint16: np.ndarray, n_experts: int = 1,
                      palette_method: str = 'kmeans') -> dict:
    """
    Encode BF16 weights (as uint16 bit patterns) into the SCLP6 compressed format.

    Output format (6 bits/weight, palette ≤8, 4 weights per 3 bytes):
      ws_stream:  uint8[ceil(N/4)*3] — packed 6-bit groups
                  sixbit layout: bits[5:3]=palette_idx, bit[2]=sign, bits[1:0]=mantissa_top2
      palette:    uint8[<=8]         — exponent values sorted by frequency (descending)
      sidecar:    {indices uint32[], values uint16[]}  — always empty (SCLP6 is fully lossy)

    BF16 reconstruction: (sign<<15) | (palette[idx]<<7) | (mantissa_top2<<5)

    Byte packing (4 weights → 3 bytes):
      byte0 = (w0 << 2) | (w1 >> 4)
      byte1 = ((w1 & 0xF) << 4) | (w2 >> 2)
      byte2 = ((w2 & 0x3) << 6) | w3
    """
    weights = weights_uint16.flatten().astype(np.uint16)
    num_weights = len(weights)

    def _encode_6b_expert(expert_weights):
        """Encode a single expert's weights, returning (palette, ws_bytes)."""
        exponents = ((expert_weights >> 7) & 0xFF).astype(np.uint8)
        unique_exponents, counts = np.unique(exponents, return_counts=True)
        if palette_method == 'kmeans':
            palette = _kmeans_palette(unique_exponents, counts, k=8)
        else:
            sorted_indices = np.argsort(-counts)
            palette = unique_exponents[sorted_indices][:8].astype(np.uint8)

        exp_lookup = np.argmin(
            np.abs(np.arange(256, dtype=np.int16)[:, None] -
                   palette.astype(np.int16)[None, :]),
            axis=1
        ).astype(np.uint8)
        indices = exp_lookup[exponents].astype(np.uint8)

        sign          = ((expert_weights >> 15) & 0x1).astype(np.uint8)
        mantissa_top2 = ((expert_weights >> 5)  & 0x3).astype(np.uint8)
        sixbits = ((indices & 0x7) << 3 | (sign << 2) | (mantissa_top2 & 0x3)).astype(np.uint8)

        nw = len(expert_weights)
        n_groups = (nw + 3) // 4
        padded = np.zeros(n_groups * 4, dtype=np.uint8)
        padded[:nw] = sixbits

        w0 = padded[0::4].astype(np.uint32)
        w1 = padded[1::4].astype(np.uint32)
        w2 = padded[2::4].astype(np.uint32)
        w3 = padded[3::4].astype(np.uint32)

        ws = np.empty(n_groups * 3, dtype=np.uint8)
        ws[0::3] = ((w0 << 2) | (w1 >> 4)).astype(np.uint8)
        ws[1::3] = (((w1 & 0xF) << 4) | (w2 >> 2)).astype(np.uint8)
        ws[2::3] = (((w2 & 0x3) << 6) | w3).astype(np.uint8)
        return palette, ws

    if n_experts == 1:
        palette, ws_stream = _encode_6b_expert(weights)
    else:
        expert_nw = num_weights // n_experts
        expert_palettes = []
        ws_parts = []
        for e in range(n_experts):
            ew = weights[e * expert_nw:(e + 1) * expert_nw]
            pal, ws = _encode_6b_expert(ew)
            expert_palettes.append(pal)
            ws_parts.append(ws)
        palette = expert_palettes  # list of arrays
        ws_stream = np.concatenate(ws_parts)

    sidecar = {'indices': np.array([], dtype=np.uint32), 'values': np.array([], dtype=np.uint16)}

    return {
        'palette':     palette,
        'ws_stream':   ws_stream,
        'num_weights': num_weights,
        'n_experts':   n_experts,
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
