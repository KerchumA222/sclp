#!/usr/bin/env python3
"""
Produce a copy of an SCLP GGUF with sidecars selectively zeroed out.

Layers are ranked by drop_cost (mae × sidecar_count). The cheapest N% are
candidates for sidecar removal.  A --keep-types override forces certain tensor
types to always retain their sidecar regardless of cost ranking.

Usage examples:
    # Drop cheapest 50% of layers by cost, always keep attn_q sidecars
    python3 tests/convert_selective_sidecar.py --drop-pct 50 --keep-types attn_q

    # Drop only attn_v and attn_k (by type, ignoring pct threshold)
    python3 tests/convert_selective_sidecar.py --drop-types attn_v,attn_k

    # Drop cheapest 25%, keeping early layers (blk.0, blk.1) regardless
    python3 tests/convert_selective_sidecar.py --drop-pct 25 --keep-early 2
"""

import sys, os, struct, shutil, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _setup_paths  # noqa: F401

from gguf import GGUFReader, GGMLQuantizationType

ATTN_KEYS = ('attn_q', 'attn_k', 'attn_v', 'attn_output')
FFN_KEYS  = ('ffn_gate', 'ffn_up', 'ffn_down')


def classify(name: str) -> str:
    for k in ATTN_KEYS + FFN_KEYS:
        if k in name:
            return k
    return 'other'


def block_index(name: str) -> int:
    """Extract block index from tensor name, or -1 for non-block tensors."""
    import re
    m = re.search(r'blk\.(\d+)\.', name)
    return int(m.group(1)) if m else -1


def parse_blob_offsets(blob: bytes):
    """Return (num_w, sidecar_count, sidecar_count_offset_in_blob)."""
    num_w,   = struct.unpack_from('<I', blob, 0)
    pal_size = blob[4]
    packed_off = 5 + pal_size
    sm_off     = packed_off + (num_w + 1) // 2
    sc_off     = sm_off + num_w           # offset of uint32 sidecar_count
    sc_count,  = struct.unpack_from('<I', blob, sc_off)
    return num_w, sc_count, sc_off


def compute_drop_cost(blob: bytes) -> float:
    num_w,   = struct.unpack_from('<I', blob, 0)
    pal_size = blob[4]
    palette  = np.frombuffer(blob, dtype=np.uint8, count=pal_size, offset=5)

    packed_off = 5 + pal_size
    sm_off     = packed_off + (num_w + 1) // 2
    sc_off     = sm_off + num_w
    sc_count,  = struct.unpack_from('<I', blob, sc_off)

    if sc_count == 0:
        return 0.0

    idx_off = sc_off + 4
    val_off = idx_off + sc_count * 4
    sc_val  = np.frombuffer(blob, dtype=np.uint16, count=sc_count, offset=val_off)

    exponents  = ((sc_val >> 7) & 0xFF).astype(np.uint8)
    dist       = np.abs(exponents[:, None].astype(np.int16) - palette.astype(np.int16))
    nearest_e  = palette[np.argmin(dist, axis=1)].astype(np.uint16)
    approx_val = (sc_val & 0x807F) | (nearest_e << 7)

    def to_f32(b):
        return (b.astype(np.uint32) << 16).view(np.float32)

    mae = float(np.mean(np.abs(to_f32(sc_val) - to_f32(approx_val))))
    return mae * sc_count


def zero_sidecar(blob: bytearray, sc_off: int) -> None:
    """Overwrite sidecar_count with 0; sidecar data bytes become dead padding."""
    struct.pack_into('<I', blob, sc_off, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gguf',        required=True, help='Input SCLP GGUF file')
    parser.add_argument('--out',         default=None, help='Output path (default: auto)')
    parser.add_argument('--drop-pct',    type=float, default=None,
                        help='Drop cheapest N%% of tensors by drop_cost')
    parser.add_argument('--drop-types',  default=None,
                        help='Comma-separated tensor types to always drop (e.g. attn_v,attn_k)')
    parser.add_argument('--keep-types',  default=None,
                        help='Comma-separated tensor types to always keep')
    parser.add_argument('--keep-early',  type=int, default=0,
                        help='Always keep sidecars for the first N transformer blocks')
    args = parser.parse_args()

    drop_types  = set(args.drop_types.split(','))  if args.drop_types  else set()
    keep_types  = set(args.keep_types.split(','))  if args.keep_types  else set()

    if args.out is None:
        base, ext = os.path.splitext(args.gguf)
        tag = f'_nosidecar'
        if args.drop_pct is not None:
            tag += f'_{int(args.drop_pct)}pct'
        if drop_types:
            tag += '_' + '_'.join(sorted(drop_types))
        args.out = base + tag + ext

    print(f"Input:  {args.gguf}")
    print(f"Output: {args.out}")

    reader  = GGUFReader(args.gguf, mode='r')
    targets = [t for t in reader.tensors if t.tensor_type == GGMLQuantizationType.SCLP]
    print(f"SCLP tensors: {len(targets)}")

    # ── score every tensor ────────────────────────────────────────────────────
    scored = []
    for t in targets:
        blob  = bytes(t.data)
        cost  = compute_drop_cost(blob)
        scored.append({'tensor': t, 'cost': cost,
                       'ttype': classify(t.name), 'blk': block_index(t.name)})

    scored.sort(key=lambda x: x['cost'])

    # ── build the drop set ────────────────────────────────────────────────────
    to_drop = set()

    # type-based forced drops
    for s in scored:
        if s['ttype'] in drop_types:
            to_drop.add(s['tensor'].name)

    # pct-based drops (cheapest N%)
    if args.drop_pct is not None:
        cutoff = int(len(scored) * args.drop_pct / 100)
        for s in scored[:cutoff]:
            to_drop.add(s['tensor'].name)

    # apply keep constraints
    for s in scored:
        name = s['tensor'].name
        if s['ttype'] in keep_types:
            to_drop.discard(name)
        if args.keep_early > 0 and 0 <= s['blk'] < args.keep_early:
            to_drop.discard(name)

    total_sidecar_before = sum(
        parse_blob_offsets(bytes(s['tensor'].data))[1] for s in scored)
    total_sidecar_after  = sum(
        0 if s['tensor'].name in to_drop
        else parse_blob_offsets(bytes(s['tensor'].data))[1]
        for s in scored)

    dropped_cost  = sum(s['cost'] for s in scored if s['tensor'].name in to_drop)
    total_cost    = sum(s['cost'] for s in scored)
    pct_cost_lost = 100 * dropped_cost / total_cost if total_cost > 0 else 0

    print(f"\nTensors to drop sidecar: {len(to_drop)} / {len(targets)}")
    print(f"Sidecar entries before:  {total_sidecar_before:,}")
    print(f"Sidecar entries after:   {total_sidecar_after:,}")
    print(f"Drop cost incurred:      {dropped_cost:.4f} / {total_cost:.4f} "
          f"({pct_cost_lost:.2f}% of total)")

    # break down drops by type
    from collections import Counter
    dropped_types = Counter(s['ttype'] for s in scored if s['tensor'].name in to_drop)
    print("\nDropped by tensor type:")
    for ttype, count in sorted(dropped_types.items()):
        print(f"  {ttype:<15} {count}")

    # ── write patched copy ────────────────────────────────────────────────────
    print(f"\nCopying to {args.out} …")
    shutil.copy2(args.gguf, args.out)

    patched = 0
    with open(args.out, 'r+b') as f:
        for s in scored:
            t = s['tensor']
            if t.name not in to_drop:
                continue
            blob     = bytearray(t.data)
            _, _, sc_off = parse_blob_offsets(bytes(blob))
            zero_sidecar(blob, sc_off)
            f.seek(t.data_offset)
            f.write(bytes(blob))
            patched += 1

    print(f"Patched {patched} tensors.")
    print(f"\nRun inference with:")
    print(f"  llama-completion -m {args.out} -ngl 99 -n 100 -no-cnv "
          f"--repeat-penalty 1.3 -p \"The capital of France is\"")


if __name__ == '__main__':
    main()
