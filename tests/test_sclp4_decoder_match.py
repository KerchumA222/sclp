"""
Verify the SCLP4 CUDA decoder logic by reimplementing it in Python
and comparing against the original input.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from compression.encoder import encode_palette_4b


def cuda_decode_blob(blob: bytes, num_weights: int) -> np.ndarray:
    """Mimic the sclp4_decode_blob_kernel logic in Python."""
    # Parse header
    import struct
    blob_nw = struct.unpack_from('<I', blob, 0)[0]
    n_experts = struct.unpack_from('<I', blob, 4)[0]
    print(f"  blob num_weights={blob_nw}, n_experts={n_experts}")

    # Read palettes
    pos = 8
    palettes = []
    palette_sizes = []
    for e in range(n_experts):
        ps = blob[pos]
        palette_sizes.append(ps)
        palettes.append(list(blob[pos+1:pos+1+ps]))
        pos += 1 + ps
    ws_start = pos

    print(f"  palette_sizes={palette_sizes[:3]}..., palettes[0]={palettes[0]}")
    print(f"  ws_start={ws_start}")

    expert_nw = num_weights // n_experts
    expert_nibble_bytes = (expert_nw + 1) // 2

    output = np.zeros(num_weights, dtype=np.uint16)

    for base_idx in range(0, num_weights, 16):
        e = base_idx // expert_nw
        local_base = base_idx - e * expert_nw
        if e >= n_experts:
            continue
        remaining = expert_nw - local_base
        n_this = min(16, remaining)

        # Read up to 8 bytes from ws
        byte_offset = ws_start + e * expert_nibble_bytes + local_base // 2
        n_bytes = min(8, (n_this + 1) // 2)
        ws8 = blob[byte_offset:byte_offset+n_bytes]

        for i in range(n_this):
            byte = ws8[i // 2] if i // 2 < len(ws8) else 0
            nibble = (byte >> 4) if i % 2 == 0 else (byte & 0xF)
            pidx = nibble >> 2
            smn = nibble & 0x3
            exp = palettes[e][pidx] if pidx < palette_sizes[e] else 0
            sign = (smn >> 1) & 1
            mant = smn & 1
            output[base_idx + i] = (sign << 15) | (exp << 7) | (mant << 6)

    # Sidecar fixup
    sc_count = struct.unpack_from('<I', blob, ws_start + n_experts * expert_nibble_bytes)[0]
    print(f"  sidecar_count={sc_count}")
    if sc_count > 0:
        sc_base = ws_start + n_experts * expert_nibble_bytes
        idx_base = sc_base + 4
        val_base = sc_base + 4 + sc_count * 4
        for i in range(sc_count):
            idx = struct.unpack_from('<I', blob, idx_base + i * 4)[0]
            val = struct.unpack_from('<H', blob, val_base + i * 2)[0]
            output[idx] = val

    return output


def main():
    # Small test tensor
    np.random.seed(42)
    N = 64  # small dense tensor
    weights = np.random.randint(0, 65535, N, dtype=np.uint16)
    # Use realistic BF16 distribution: most exponents around 120
    weights = (weights & 0x807F) | (np.random.randint(118, 124, N).astype(np.uint16) << 7)

    encoded = encode_palette_4b(weights, n_experts=1, palette_method='kmeans', sidecar_dist=1)
    print(f"Input: {N} weights")
    print(f"Palette: {encoded['palette']}")
    print(f"WS bytes: {len(encoded['ws_stream'])}")
    print(f"Sidecar: {len(encoded['sidecar']['indices'])} entries")

    # Build blob the same way convert_to_sclp_gguf.py does
    import struct
    pal = encoded['palette'].astype(np.uint8)
    palette_header = bytes([len(pal)]) + pal.tobytes()
    blob = (
        struct.pack('<II', N, 1)
        + palette_header
        + encoded['ws_stream'].astype(np.uint8).tobytes()
        + struct.pack('<I', len(encoded['sidecar']['indices']))
        + encoded['sidecar']['indices'].astype(np.uint32).tobytes()
        + encoded['sidecar']['values'].astype(np.uint16).tobytes()
    )

    print(f"\nBlob size: {len(blob)} bytes")
    print("\nDecoding with mimicked CUDA logic...")
    decoded = cuda_decode_blob(blob, N)

    # Compare
    print("\n--- Comparison ---")
    orig_exp = (weights >> 7) & 0xFF
    dec_exp = (decoded >> 7) & 0xFF
    print(f"Original first 16 exponents: {orig_exp[:16].tolist()}")
    print(f"Decoded  first 16 exponents: {dec_exp[:16].tolist()}")

    # Show full BF16 values comparison
    orig_signs = (weights >> 15) & 1
    dec_signs = (decoded >> 15) & 1
    print(f"Original signs:  {orig_signs[:16].tolist()}")
    print(f"Decoded  signs:  {dec_signs[:16].tolist()}")

    diff = np.abs(weights.view(np.int16).astype(np.int32) - decoded.view(np.int16).astype(np.int32))
    print(f"\nMax bit difference: {diff.max()}, mean: {diff.mean():.2f}")
    print(f"Exact matches: {(weights == decoded).sum()}/{N}")


if __name__ == '__main__':
    main()
