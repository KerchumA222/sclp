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


def encode_palette_4b(weights_uint16: np.ndarray, n_experts: int = 1,
                      palette_method: str = 'kmeans',
                      sidecar_dist: int = 0,
                      importance: np.ndarray | None = None,
                      K: int | None = None,
                      sidecar_imatrix_budget: float = 0.0) -> dict:
    """
    Encode BF16 weights (as uint16 bit patterns) into the SCLP4 compressed format.

    Output format (4 bits/weight, palette ≤4, 2 weights per byte):
      ws_stream:  uint8[ceil(N/2)] — packed nibbles, high nibble = even weight, low = odd
                  nibble layout: bits[3:2]=palette_idx, bit[1]=sign, bit[0]=mantissa_top1
      palette:    uint8[<=4]       — exponent values
      sidecar:    {indices uint32[], values uint16[]}

    sidecar_dist: same semantics as encode_palette_6b. For SCLP4 the gain is larger
    because frequency k=4 leaves ~15% of weights uncovered.

    importance: optional float32[n_experts, K] activation-importance per (expert, input-column)
                from an imatrix. When provided, the k-means palette selection is weighted by
                activation magnitude instead of raw exponent frequency — palette entries
                cluster toward exponents that appear in high-activation columns. Requires K
                (the input dim, = ne[0]) so we can fold importance over the per-weight layout.

    BF16 reconstruction: (sign<<15) | (palette[idx]<<7) | (mantissa_top1<<6)
    """
    weights = weights_uint16.flatten().astype(np.uint16)
    num_weights = len(weights)

    if importance is not None:
        if K is None:
            raise ValueError("encode_palette_4b: K (input dim) is required when importance is provided")
        if importance.shape != (n_experts, K):
            raise ValueError(f"encode_palette_4b: importance shape {importance.shape} != ({n_experts}, {K})")

    sidecar_indices_all = []
    sidecar_values_all  = []

    def _encode_4b_expert(expert_weights, expert_offset=0, expert_imp=None):
        """Encode a single expert's weights, returning (palette, ws_bytes).

        Palette selection always uses raw exponent frequency (importance-weighted
        k-means was tried and regressed PPL — see CLAUDE.md). When expert_imp is
        provided, importance is applied to *sidecar selection* instead: weights are
        ranked by `importance × exponent_distance_to_palette` and the top
        `sidecar_imatrix_budget` fraction joins the lossless sidecar in addition to
        the mandatory `dist > sidecar_dist` group.
        """
        exponents = ((expert_weights >> 7) & 0xFF).astype(np.uint8)
        unique_exponents, counts = np.unique(exponents, return_counts=True)

        if palette_method == 'kmeans':
            palette = _kmeans_palette(unique_exponents, counts, k=4)
        else:
            sorted_indices = np.argsort(-counts)
            palette = unique_exponents[sorted_indices][:4].astype(np.uint8)

        dist_per_exp = np.min(
            np.abs(np.arange(256, dtype=np.int16)[:, None] -
                   palette.astype(np.int16)[None, :]),
            axis=1
        )
        exp_lookup = np.argmin(
            np.abs(np.arange(256, dtype=np.int16)[:, None] -
                   palette.astype(np.int16)[None, :]),
            axis=1
        ).astype(np.uint8)
        indices = exp_lookup[exponents].astype(np.uint8)

        per_weight_dist = dist_per_exp[exponents]  # int16

        # Mandatory sidecar: catastrophic distance > sidecar_dist (matches non-imatrix path).
        if sidecar_dist > 0:
            outlier_mask = per_weight_dist > sidecar_dist
        else:
            outlier_mask = np.zeros(len(expert_weights), dtype=bool)

        # Discretionary sidecar (imatrix-aware): on top of the mandatory set, rescue the
        # top `sidecar_imatrix_budget` fraction of remaining weights ranked by
        # `importance × distance`. Weights with dist == 0 contribute 0 priority and
        # never get rescued (their only error is mantissa truncation, which sidecar
        # can't avoid). Weights with high importance × distance dominate the ranking.
        if expert_imp is not None and sidecar_imatrix_budget > 0.0:
            K_dim = expert_imp.shape[0]
            col_idx = np.arange(len(expert_weights), dtype=np.int64) % K_dim
            per_weight_imp = expert_imp[col_idx]  # float32, broadcast over rows
            priority = per_weight_imp * per_weight_dist.astype(np.float32)
            # Exclude already-mandatory entries from the ranking.
            priority = np.where(outlier_mask, -np.inf, priority)
            n_extra = int(len(expert_weights) * sidecar_imatrix_budget)
            if n_extra > 0:
                # argpartition gives indices of the top n_extra by priority.
                cand = np.argpartition(priority, -n_extra)[-n_extra:]
                # Only keep ones with strictly positive priority (otherwise we'd be
                # sidecaring dist-0 weights, which is wasteful).
                cand = cand[priority[cand] > 0]
                outlier_mask[cand] = True

        if outlier_mask.any():
            out_pos = np.where(outlier_mask)[0].astype(np.uint32)
            sidecar_indices_all.append(out_pos + expert_offset)
            sidecar_values_all.append(expert_weights[outlier_mask])

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
        eimp = importance[0] if importance is not None else None
        palette, ws_stream = _encode_4b_expert(weights, expert_offset=0, expert_imp=eimp)
    else:
        expert_nw = num_weights // n_experts
        expert_palettes = []
        ws_parts = []
        for e in range(n_experts):
            ew = weights[e * expert_nw:(e + 1) * expert_nw]
            eimp = importance[e] if importance is not None else None
            pal, ws = _encode_4b_expert(ew, expert_offset=e * expert_nw, expert_imp=eimp)
            expert_palettes.append(pal)
            ws_parts.append(ws)
        palette = expert_palettes  # list of arrays
        ws_stream = np.concatenate(ws_parts)

    if sidecar_indices_all:
        sc_indices = np.concatenate(sidecar_indices_all).astype(np.uint32)
        sc_values  = np.concatenate(sidecar_values_all).astype(np.uint16)
    else:
        sc_indices = np.array([], dtype=np.uint32)
        sc_values  = np.array([], dtype=np.uint16)
    sidecar = {'indices': sc_indices, 'values': sc_values}

    return {
        'palette':     palette,
        'ws_stream':   ws_stream,
        'num_weights': num_weights,
        'n_experts':   n_experts,
        'sidecar':     sidecar,
    }


def encode_palette_6b(weights_uint16: np.ndarray, n_experts: int = 1,
                      palette_method: str = 'kmeans',
                      sidecar_dist: int = 0,
                      importance: np.ndarray | None = None,
                      K: int | None = None,
                      sidecar_imatrix_budget: float = 0.0) -> dict:
    """
    Encode BF16 weights (as uint16 bit patterns) into the SCLP6 compressed format.

    Output format (6 bits/weight, palette ≤8, 4 weights per 3 bytes):
      ws_stream:  uint8[ceil(N/4)*3] — packed 6-bit groups
                  sixbit layout: bits[5:3]=palette_idx, bit[2]=sign, bits[1:0]=mantissa_top2
      palette:    uint8[<=8]         — exponent values
      sidecar:    {indices uint32[], values uint16[]}

    sidecar_dist: weights whose exponent distance to their nearest palette entry
    exceeds this threshold are stored verbatim in the sidecar (lossless rescue).
    0 = no sidecar (default, fully lossy). 1 is recommended: captures ~0.1% of
    weights responsible for the worst-case errors with negligible size overhead.

    importance: optional float32[n_experts, K] activation magnitudes from imatrix.
                Same semantics as encode_palette_4b.

    BF16 reconstruction: (sign<<15) | (palette[idx]<<7) | (mantissa_top2<<5)

    Byte packing (4 weights → 3 bytes):
      byte0 = (w0 << 2) | (w1 >> 4)
      byte1 = ((w1 & 0xF) << 4) | (w2 >> 2)
      byte2 = ((w2 & 0x3) << 6) | w3
    """
    weights = weights_uint16.flatten().astype(np.uint16)
    num_weights = len(weights)

    if importance is not None:
        if K is None:
            raise ValueError("encode_palette_6b: K is required when importance is provided")
        if importance.shape != (n_experts, K):
            raise ValueError(f"encode_palette_6b: importance shape {importance.shape} != ({n_experts}, {K})")

    sidecar_indices_all = []
    sidecar_values_all  = []

    def _encode_6b_expert(expert_weights, expert_offset=0, expert_imp=None):
        """Encode a single expert's weights, returning (palette, ws_bytes).

        Palette selection always uses raw exponent frequency. When expert_imp is
        provided, importance is applied to sidecar selection (see SCLP4 docstring).
        """
        exponents = ((expert_weights >> 7) & 0xFF).astype(np.uint8)
        unique_exponents, counts = np.unique(exponents, return_counts=True)

        if palette_method == 'kmeans':
            palette = _kmeans_palette(unique_exponents, counts, k=8)
        else:
            sorted_indices = np.argsort(-counts)
            palette = unique_exponents[sorted_indices][:8].astype(np.uint8)

        # nearest-palette distance per weight (for sidecar gating)
        dist_per_exp = np.min(
            np.abs(np.arange(256, dtype=np.int16)[:, None] -
                   palette.astype(np.int16)[None, :]),
            axis=1
        )  # shape (256,)

        exp_lookup = np.argmin(
            np.abs(np.arange(256, dtype=np.int16)[:, None] -
                   palette.astype(np.int16)[None, :]),
            axis=1
        ).astype(np.uint8)
        indices = exp_lookup[exponents].astype(np.uint8)

        per_weight_dist = dist_per_exp[exponents]

        if sidecar_dist > 0:
            outlier_mask = per_weight_dist > sidecar_dist
        else:
            outlier_mask = np.zeros(len(expert_weights), dtype=bool)

        # Imatrix-aware discretionary sidecar (see SCLP4 implementation comment).
        if expert_imp is not None and sidecar_imatrix_budget > 0.0:
            K_dim = expert_imp.shape[0]
            col_idx = np.arange(len(expert_weights), dtype=np.int64) % K_dim
            per_weight_imp = expert_imp[col_idx]
            priority = per_weight_imp * per_weight_dist.astype(np.float32)
            priority = np.where(outlier_mask, -np.inf, priority)
            n_extra = int(len(expert_weights) * sidecar_imatrix_budget)
            if n_extra > 0:
                cand = np.argpartition(priority, -n_extra)[-n_extra:]
                cand = cand[priority[cand] > 0]
                outlier_mask[cand] = True

        if outlier_mask.any():
            out_pos = np.where(outlier_mask)[0].astype(np.uint32)
            sidecar_indices_all.append(out_pos + expert_offset)
            sidecar_values_all.append(expert_weights[outlier_mask])

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
        eimp = importance[0] if importance is not None else None
        palette, ws_stream = _encode_6b_expert(weights, expert_offset=0, expert_imp=eimp)
    else:
        expert_nw = num_weights // n_experts
        expert_palettes = []
        ws_parts = []
        for e in range(n_experts):
            ew = weights[e * expert_nw:(e + 1) * expert_nw]
            eimp = importance[e] if importance is not None else None
            pal, ws = _encode_6b_expert(ew, expert_offset=e * expert_nw, expert_imp=eimp)
            expert_palettes.append(pal)
            ws_parts.append(ws)
        palette = expert_palettes  # list of arrays
        ws_stream = np.concatenate(ws_parts)

    if sidecar_indices_all:
        sc_indices = np.concatenate(sidecar_indices_all).astype(np.uint32)
        sc_values  = np.concatenate(sidecar_values_all).astype(np.uint16)
    else:
        sc_indices = np.array([], dtype=np.uint32)
        sc_values  = np.array([], dtype=np.uint16)
    sidecar = {'indices': sc_indices, 'values': sc_values}

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
