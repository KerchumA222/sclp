#!/usr/bin/env python3
"""
Rank SCLP tensors by the cost of dropping their sidecar.

For each SCLP tensor the script computes:
  - sidecar_count      : number of verbatim-stored outlier weights
  - sidecar_frac       : sidecar_count / num_weights
  - mae                : mean |true_weight - palette_approx_weight| for sidecar entries
  - drop_cost          : mae * sidecar_count  (proxy for PPL impact)
  - tensor_type        : attn_q/k/v/o  |  ffn_gate/up/down

Layers with low drop_cost are safe candidates for sidecar removal.

Usage:
    python3 tests/analyze_sidecar_cost.py [--gguf PATH] [--top N] [--csv PATH]
"""

import sys, os, struct, argparse
import numpy as np

sys.path.append(os.path.abspath('/home/ajkerchum/llama.cpp/gguf-py'))
sys.path.append(os.path.abspath('/home/ajkerchum/poc/src'))

from gguf import GGUFReader, GGMLQuantizationType


# ── tensor classification ─────────────────────────────────────────────────────

ATTN_KEYS  = ('attn_q', 'attn_k', 'attn_v', 'attn_output')
FFN_KEYS   = ('ffn_gate', 'ffn_up', 'ffn_down')

def classify(name: str) -> str:
    for k in ATTN_KEYS:
        if k in name:
            return k
    for k in FFN_KEYS:
        if k in name:
            return k
    return 'other'


# ── blob parsing ──────────────────────────────────────────────────────────────

def parse_blob(blob: bytes):
    """Return (palette, packed_bytes, sm_bytes, sidecar_indices, sidecar_values)."""
    num_w,   = struct.unpack_from('<I', blob, 0)
    pal_size = blob[4]
    palette  = np.frombuffer(blob, dtype=np.uint8, count=pal_size, offset=5)

    packed_off = 5 + pal_size
    sm_off     = packed_off + (num_w + 1) // 2
    sc_off     = sm_off + num_w
    sc_count,  = struct.unpack_from('<I', blob, sc_off)

    idx_off = sc_off + 4
    val_off = idx_off + sc_count * 4

    sidecar_idx = np.frombuffer(blob, dtype=np.uint32, count=sc_count, offset=idx_off)
    sidecar_val = np.frombuffer(blob, dtype=np.uint16, count=sc_count, offset=val_off)

    return num_w, palette, sidecar_idx, sidecar_val


def palette_approx(true_bits: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """
    For each BF16 weight (uint16), return the BF16 bits the decoder would produce
    if it used the nearest palette entry for that weight's exponent.

    Nearest = palette entry with smallest |exp_true - exp_palette|.
    sign and mantissa are preserved from the original weight.
    """
    exponents = ((true_bits >> 7) & 0xFF).astype(np.uint8)          # 8-bit exponent
    # distance from each weight's exponent to every palette entry
    # shape: (n_sidecar, palette_size)
    dist = np.abs(exponents[:, None].astype(np.int16) -
                  palette.astype(np.int16))
    nearest_idx = np.argmin(dist, axis=1)
    nearest_exp = palette[nearest_idx].astype(np.uint16)

    approx_bits = (true_bits & 0x807F) | (nearest_exp << 7)
    return approx_bits


def bits_to_float(bits: np.ndarray) -> np.ndarray:
    """BF16 uint16 → float32 via view trick."""
    u32 = bits.astype(np.uint32) << 16
    return u32.view(np.float32)


def compute_mae(true_bits: np.ndarray, approx_bits: np.ndarray) -> float:
    return float(np.mean(np.abs(bits_to_float(true_bits) - bits_to_float(approx_bits))))


# ── main ──────────────────────────────────────────────────────────────────────

def analyse(gguf_path: str, top_n: int, csv_path: str | None):
    print(f"Reading {gguf_path} …")
    reader = GGUFReader(gguf_path, mode='r')
    tensors = [t for t in reader.tensors if t.tensor_type == GGMLQuantizationType.SCLP]
    print(f"Found {len(tensors)} SCLP tensors\n")

    rows = []
    for t in tensors:
        blob = bytes(t.data)
        num_w, palette, sc_idx, sc_val = parse_blob(blob)

        sc_count = len(sc_idx)
        sc_frac  = sc_count / num_w if num_w > 0 else 0.0

        if sc_count > 0:
            approx = palette_approx(sc_val, palette)
            mae    = compute_mae(sc_val, approx)
        else:
            mae = 0.0

        drop_cost = mae * sc_count
        rows.append({
            'name':        t.name,
            'type':        classify(t.name),
            'num_weights': num_w,
            'sidecar':     sc_count,
            'frac_pct':    sc_frac * 100,
            'mae':         mae,
            'drop_cost':   drop_cost,
        })

    rows.sort(key=lambda r: r['drop_cost'])

    # ── per-type summary ──────────────────────────────────────────────────────
    from collections import defaultdict
    by_type = defaultdict(list)
    for r in rows:
        by_type[r['type']].append(r)

    print(f"{'Type':<15} {'Tensors':>7} {'Avg sidecar':>12} {'Avg MAE':>12} {'Avg cost':>12}")
    print("-" * 62)
    for ttype in sorted(by_type):
        grp = by_type[ttype]
        print(f"{ttype:<15} {len(grp):>7} "
              f"{np.mean([r['sidecar'] for r in grp]):>12.1f} "
              f"{np.mean([r['mae']     for r in grp]):>12.6f} "
              f"{np.mean([r['drop_cost'] for r in grp]):>12.4f}")

    # ── cheapest layers (safest to drop sidecar) ─────────────────────────────
    print(f"\n── {top_n} cheapest layers to drop sidecar (lowest drop_cost) ──")
    print(f"{'Rank':<5} {'Name':<40} {'Type':<12} {'Sidecar':>8} {'MAE':>12} {'Cost':>12}")
    print("-" * 93)
    for i, r in enumerate(rows[:top_n]):
        print(f"{i+1:<5} {r['name']:<40} {r['type']:<12} "
              f"{r['sidecar']:>8} {r['mae']:>12.6f} {r['drop_cost']:>12.4f}")

    # ── most expensive layers (keep sidecar) ─────────────────────────────────
    print(f"\n── {top_n} most expensive layers (keep sidecar) ──")
    print(f"{'Rank':<5} {'Name':<40} {'Type':<12} {'Sidecar':>8} {'MAE':>12} {'Cost':>12}")
    print("-" * 93)
    for i, r in enumerate(rows[-top_n:][::-1]):
        print(f"{i+1:<5} {r['name']:<40} {r['type']:<12} "
              f"{r['sidecar']:>8} {r['mae']:>12.6f} {r['drop_cost']:>12.4f}")

    # ── cumulative cost curve ─────────────────────────────────────────────────
    total_cost = sum(r['drop_cost'] for r in rows)
    print(f"\n── Cumulative drop_cost by % of layers dropped ──")
    print(f"{'Layers dropped':>16} {'Pct layers':>12} {'Cumulative cost':>18} {'% of total':>12}")
    print("-" * 62)
    for pct in (10, 25, 50, 75, 90, 100):
        n = max(1, int(len(rows) * pct / 100))
        cum = sum(r['drop_cost'] for r in rows[:n])
        print(f"{n:>16} {pct:>11}% {cum:>18.4f} {100*cum/total_cost if total_cost>0 else 0:>11.1f}%")

    if csv_path:
        import csv
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"\nFull results written to {csv_path}")

    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gguf', default='/home/ajkerchum/poc/models/llama3/Llama-3-8B-SCLP-Patched.gguf')
    parser.add_argument('--top',  type=int, default=15)
    parser.add_argument('--csv',  default=None)
    args = parser.parse_args()
    analyse(args.gguf, args.top, args.csv)
