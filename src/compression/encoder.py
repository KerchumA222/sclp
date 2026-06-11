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


def encode_palette(weights_uint16: np.ndarray, n_experts: int = 1) -> dict:
    """
    Encode BF16 weights (as uint16 bit patterns) into the SCLP compressed format (SCLP8).

    Output format (8 bits/weight, palette ≤16):
      ws_stream:  uint8[N]     — one byte per weight: palette_idx(7:4) | smn(3:0)
                                 smn = sign(3) | mantissa_top3(2:0)
      palette:    uint8[<=16]  — exponent values (per-expert if n_experts > 1)
      sidecar:    {indices uint32[], values uint16[]}
    """
    weights = weights_uint16.flatten().astype(np.uint16)
    num_weights = len(weights)

    sidecar_indices_all = []
    sidecar_values_all  = []

    def _encode_8b_expert(expert_weights, expert_offset=0):
        # 1. Exponent palette: top 16 unique exponents by frequency
        exponents = ((expert_weights >> 7) & 0xFF).astype(np.uint8)
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
        outlier_mask = ~in_palette
        if outlier_mask.any():
            out_pos = np.where(outlier_mask)[0].astype(np.uint32)
            sidecar_indices_all.append(out_pos + expert_offset)
            sidecar_values_all.append(expert_weights[outlier_mask])

        # 4. SM nibble: sign(3) | mantissa_top3(2:0)  — top 3 of 7 mantissa bits
        sign          = ((expert_weights >> 15) & 0x1).astype(np.uint8)
        mantissa_top3 = ((expert_weights >> 4)  & 0x7).astype(np.uint8)  # bits 6:4
        sm_nibbles    = ((sign << 3) | mantissa_top3).astype(np.uint8)

        # 5. Interleaved ws_stream: one byte per weight — idx(high nibble) | smn(low nibble)
        ws_stream = ((indices & 0x0F) << 4 | (sm_nibbles & 0x0F)).astype(np.uint8)
        return palette, ws_stream

    expert_nw = num_weights // n_experts
    ws_parts = []
    palettes = []
    for e in range(n_experts):
        ew = weights[e * expert_nw:(e + 1) * expert_nw]
        pal, ws = _encode_8b_expert(ew, expert_offset=e * expert_nw)
        palettes.append(pal)
        ws_parts.append(ws)
    
    ws_stream = np.concatenate(ws_parts)
    
    if sidecar_indices_all:
        sc_indices = np.concatenate(sidecar_indices_all).astype(np.uint32)
        sc_values  = np.concatenate(sidecar_values_all).astype(np.uint16)
    else:
        sc_indices = np.array([], dtype=np.uint32)
        sc_values  = np.array([], dtype=np.uint16)
    
    return {
        'palette':     palettes if n_experts > 1 else palettes[0],
        'ws_stream':   ws_stream,
        'num_weights': num_weights,
        'n_experts':   n_experts,
        'sidecar':     {'indices': sc_indices, 'values': sc_values}
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
        # `importance × squared_reconstruction_error`. The sidecar stores verbatim BF16
        # so it fixes both exponent error AND mantissa truncation — dist-0 weights are
        # valid candidates (their mantissa error is real and rescued by the sidecar).
        # Reconstruction: (sign<<15) | (palette[idx]<<7) | (mantissa_top1<<6)
        if expert_imp is not None and sidecar_imatrix_budget > 0.0:
            K_dim = expert_imp.shape[0]
            col_idx = np.arange(len(expert_weights), dtype=np.int64) % K_dim
            per_weight_imp = expert_imp[col_idx]  # float32, broadcast over rows

            # Compute reconstruction error for every weight.
            sign_bits    = ((expert_weights >> 15) & 0x1).astype(np.uint16)
            mant_top1    = ((expert_weights >> 6)  & 0x1).astype(np.uint16)
            recon_bits   = ((sign_bits << 15)
                            | (palette[indices].astype(np.uint16) << 7)
                            | (mant_top1 << 6)).astype(np.uint16)
            # Convert uint16 BF16 bits → float32 via uint32 view.
            orig_f32  = (expert_weights.astype(np.uint32) << 16).view(np.float32)
            recon_f32 = (recon_bits.astype(np.uint32) << 16).view(np.float32)
            sq_err    = (orig_f32 - recon_f32) ** 2

            priority = per_weight_imp * sq_err
            # Exclude already-mandatory entries from the ranking.
            priority = np.where(outlier_mask, -np.inf, priority)
            n_extra = int(len(expert_weights) * sidecar_imatrix_budget)
            if n_extra > 0:
                # argpartition gives indices of the top n_extra by priority.
                cand = np.argpartition(priority, -n_extra)[-n_extra:]
                # Keep only entries with strictly positive priority (zero sq_err → no gain).
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


def encode_palette_4m(weights_uint16: np.ndarray, n_experts: int = 1,
                      importance: np.ndarray | None = None,
                      K: int | None = None,
                      sidecar_imatrix_budget: float = 0.0) -> dict:
    """
    Encode BF16 weights (as uint16 bit patterns) into the SCLP4M compressed format.

    SCLP4M = per-block 8-entry BF16 magnitude codebook (idx3|sign1, 4.5 bpw).
    Each 256-weight block has its own codebook of 8 arbitrary BF16 magnitudes derived
    from 1-D Lloyd k-means over |w| in linear space.  Sign is stored separately.
    No exponent palette, no mantissa bits, no per-block scale.

    Output format (4 bits/weight, 2 weights per byte):
      ws_stream:        uint8[ceil(N/2)]  — nibble = (idx<<1)|sign; high nibble = even weight
      block_codebooks:  uint16[n_blocks, 8] — BF16 magnitude bit-patterns (sign bit 0)
      sidecar:          {indices uint32[], values uint16[]}

    Decode: mag = codebook[idx]; bits = mag | (sign<<15); weight_bf16 = bits

    Deviation from C++ reference:
      - k-means init: C++ uses exact quantile positions q=(2j+1)*n/16 on sorted magnitudes.
        Python does the same (see _kmeans_magnitude_codebook_block below).
      - Convergence: 15 Lloyd iterations (same as C++); midpoint-boundary assignment
        over sorted magnitudes (same algorithm, vectorised with numpy).
      - No stochastic rounding (SCLP4M has no mantissa grid, so it's inapplicable).

    importance: optional float32[n_experts, K] imatrix values. When provided, the
                discretionary sidecar rescues the top `sidecar_imatrix_budget` fraction
                of weights ranked by importance × err². No mandatory sidecar tier
                (the adaptive codebook handles outliers; mandatory tier would be redundant).
    """
    weights = weights_uint16.flatten().astype(np.uint16)
    num_weights = len(weights)
    QK = 256  # block size — matches QK_SCLP4 in C++

    if importance is not None:
        if K is None:
            raise ValueError("encode_palette_4m: K (input dim) required when importance is provided")
        if importance.shape != (n_experts, K):
            raise ValueError(f"encode_palette_4m: importance shape {importance.shape} != ({n_experts}, {K})")

    sidecar_indices_all = []
    sidecar_values_all  = []

    def _kmeans_magnitude_codebook_block(block_float: np.ndarray) -> np.ndarray:
        """1-D Lloyd k-means over |w| for one block → 8 BF16 magnitude bit-patterns.

        Matches kmeans_magnitude_codebook() in llama-sclp.cpp:
          - quantile init: c[j] = sorted_mags[(2j+1)*n/16]
          - 15 Lloyd iterations with midpoint-boundary assignment on sorted mags
          - centroids stored as float_to_bf16(c[j]) & 0x7FFF (sign bit cleared)
        """
        mags = np.abs(block_float).astype(np.float32)
        sorted_mags = np.sort(mags)
        n = len(sorted_mags)
        k = 8

        # Quantile init (same as C++)
        c = np.empty(k, dtype=np.float32)
        for j in range(k):
            q = (2 * j + 1) * n // 16
            if q >= n:
                q = n - 1
            c[j] = sorted_mags[q]

        for _ in range(15):
            # Midpoint-boundary assignment on sorted magnitudes (matches C++ inner loop)
            # Boundaries between cluster j and j+1 are at 0.5*(c[j]+c[j+1])
            boundaries = 0.5 * (c[:-1] + c[1:])  # shape (7,)
            labels = np.searchsorted(boundaries, sorted_mags)  # shape (n,)

            new_c = np.empty(k, dtype=np.float64)
            converged = True
            for j in range(k):
                mask = labels == j
                if mask.any():
                    nc = float(sorted_mags[mask].mean())
                    if abs(nc - float(c[j])) > 1e-8:
                        converged = False
                    new_c[j] = nc
                else:
                    new_c[j] = c[j]
            c = new_c.astype(np.float32)
            if converged:
                break

        # Store as BF16 magnitude bit-patterns (sign bit cleared), using RNE
        cb_f32 = c.astype(np.float32)
        cb_u32 = cb_f32.view(np.uint32)
        lsb = (cb_u32 >> 16) & 1
        bias = np.uint32(0x7FFF) + lsb
        cb_bf16 = ((cb_u32.astype(np.uint64) + bias.astype(np.uint64)) >> 16).astype(np.uint16)
        cb_bf16 = cb_bf16 & np.uint16(0x7FFF)  # clear sign bit — magnitude only
        return cb_bf16  # uint16[8]

    def _encode_4m_expert(expert_weights_u16, expert_float, expert_offset=0, expert_imp=None):
        """Encode a single expert.  Returns (block_codebooks_u16, ws_bytes)."""
        nw = len(expert_weights_u16)
        n_blocks = (nw + QK - 1) // QK

        block_codebooks = np.zeros((n_blocks, 8), dtype=np.uint16)
        indices   = np.zeros(nw, dtype=np.uint8)
        sm_nibbles = np.zeros(nw, dtype=np.uint8)  # sign bit only

        for b in range(n_blocks):
            b_start = b * QK
            b_end   = min(b_start + QK, nw)
            blk_f   = expert_float[b_start:b_end]

            cb = _kmeans_magnitude_codebook_block(blk_f)
            block_codebooks[b] = cb
            cb_f = (cb.astype(np.uint32) << 16).view(np.float32)  # BF16 → float32

            mags = np.abs(blk_f).astype(np.float32)
            # Index search: pick cb entry minimising |cb[j] - |w||
            errs = np.abs(mags[:, None] - cb_f[None, :])  # (blk, 8)
            best_idx = errs.argmin(axis=1).astype(np.uint8)
            indices[b_start:b_end]    = best_idx
            sm_nibbles[b_start:b_end] = (blk_f < 0).astype(np.uint8)  # sign

        # Discretionary imatrix sidecar
        if expert_imp is not None and sidecar_imatrix_budget > 0.0:
            K_dim = expert_imp.shape[0]
            col_idx = np.arange(nw, dtype=np.int64) % K_dim
            per_weight_imp = expert_imp[col_idx]

            # Reconstruction error: |bf16(codebook[idx]) - |w||
            block_idx_per_w = np.arange(nw, dtype=np.int64) // QK
            cb_flat = block_codebooks[block_idx_per_w, indices]  # uint16 BF16 magnitudes
            recon_f = (cb_flat.astype(np.uint32) << 16).view(np.float32)
            orig_f  = expert_float.astype(np.float32)
            err2    = (recon_f - np.abs(orig_f)) ** 2

            priority = per_weight_imp * err2
            n_extra  = int(nw * sidecar_imatrix_budget)
            outlier_mask = np.zeros(nw, dtype=bool)
            if n_extra > 0:
                cand = np.argpartition(priority, -n_extra)[-n_extra:]
                cand = cand[priority[cand] > 0]
                outlier_mask[cand] = True
        else:
            outlier_mask = np.zeros(nw, dtype=bool)

        if outlier_mask.any():
            out_pos = np.where(outlier_mask)[0].astype(np.uint32)
            sidecar_indices_all.append(out_pos + expert_offset)
            sidecar_values_all.append(expert_weights_u16[outlier_mask])

        # Pack nibbles: nibble = (idx<<1)|sign; high nibble = even weight
        nibbles = ((indices & 0x7) << 1 | sm_nibbles).astype(np.uint8)
        num_bytes = (nw + 1) // 2
        ws = np.zeros(num_bytes, dtype=np.uint8)
        ws[:] = (nibbles[0::2] << 4)
        if nw > 1:
            n_odd = nw // 2
            ws[:n_odd] |= nibbles[1::2]

        return block_codebooks, ws

    # Convert weights to float32 for magnitude operations
    weights_f32 = (weights.astype(np.uint32) << 16).view(np.float32)

    if n_experts == 1:
        eimp = importance[0] if importance is not None else None
        block_codebooks, ws_stream = _encode_4m_expert(weights, weights_f32,
                                                        expert_offset=0, expert_imp=eimp)
    else:
        expert_nw = num_weights // n_experts
        all_cb    = []
        ws_parts  = []
        for e in range(n_experts):
            ew   = weights[e * expert_nw:(e + 1) * expert_nw]
            ewf  = weights_f32[e * expert_nw:(e + 1) * expert_nw]
            eimp = importance[e] if importance is not None else None
            cb, ws = _encode_4m_expert(ew, ewf, expert_offset=e * expert_nw, expert_imp=eimp)
            all_cb.append(cb)
            ws_parts.append(ws)
        # block_codebooks is a list of arrays per expert when n_experts > 1
        block_codebooks = all_cb
        ws_stream = np.concatenate(ws_parts)

    if sidecar_indices_all:
        sc_indices = np.concatenate(sidecar_indices_all).astype(np.uint32)
        sc_values  = np.concatenate(sidecar_values_all).astype(np.uint16)
    else:
        sc_indices = np.array([], dtype=np.uint32)
        sc_values  = np.array([], dtype=np.uint16)

    return {
        'block_codebooks': block_codebooks,  # uint16[n_blocks, 8] or list[uint16[n_blocks_e, 8]]
        'ws_stream':       ws_stream,
        'num_weights':     num_weights,
        'n_experts':       n_experts,
        'sidecar':         {'indices': sc_indices, 'values': sc_values},
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

        # Imatrix-aware discretionary sidecar: ranked by importance × squared
        # reconstruction error (same logic as SCLP4; see that block for rationale).
        # Reconstruction: (sign<<15) | (palette[idx]<<7) | (mantissa_top2<<5)
        if expert_imp is not None and sidecar_imatrix_budget > 0.0:
            K_dim = expert_imp.shape[0]
            col_idx = np.arange(len(expert_weights), dtype=np.int64) % K_dim
            per_weight_imp = expert_imp[col_idx]

            # Compute reconstruction error for every weight.
            sign_bits    = ((expert_weights >> 15) & 0x1).astype(np.uint16)
            mant_top2    = ((expert_weights >> 5)  & 0x3).astype(np.uint16)
            recon_bits   = ((sign_bits << 15)
                            | (palette[indices].astype(np.uint16) << 7)
                            | (mant_top2 << 5)).astype(np.uint16)
            orig_f32  = (expert_weights.astype(np.uint32) << 16).view(np.float32)
            recon_f32 = (recon_bits.astype(np.uint32) << 16).view(np.float32)
            sq_err    = (orig_f32 - recon_f32) ** 2

            priority = per_weight_imp * sq_err
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


