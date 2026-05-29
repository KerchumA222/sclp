"""
Analyze BF16 exponent and mantissa distributions across real model tensors.
Reads a sample of MoE and attention tensors from the BF16 shards.
"""
import sys, os
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LLAMA_CPP = os.environ.get('LLAMA_CPP', os.path.join(_REPO, '..', 'llama.cpp'))
sys.path.insert(0, os.path.join(_REPO, 'src'))
sys.path.insert(0, os.path.join(_LLAMA_CPP, 'gguf-py'))
import gguf

# Set MODEL_DIR to the directory holding the BF16 GGUF shards.
_MODEL_DIR = os.environ.get(
    'MODEL_DIR',
    os.path.join(_REPO, 'models/gemma4/google_gemma-4-26B-A4B-it-bf16'),
)
SHARDS = [
    os.path.join(_MODEL_DIR, 'google_gemma-4-26B-A4B-it-bf16-00001-of-00002.gguf'),
    os.path.join(_MODEL_DIR, 'google_gemma-4-26B-A4B-it-bf16-00002-of-00002.gguf'),
]

SAMPLE_TENSORS = [
    'blk.0.ffn_down_exps.weight',   # MoE expert (large, many experts)
    'blk.0.attn_q.weight',           # dense attention
    'blk.15.ffn_gate_up_exps.weight', # mid-layer MoE
    'blk.29.ffn_down_exps.weight',   # late-layer MoE (if exists)
]


def bf16_parts(u16: np.ndarray):
    exponents = ((u16 >> 7) & 0xFF).astype(np.uint8)
    mantissas = (u16 & 0x7F).astype(np.uint8)   # full 7 mantissa bits
    signs     = ((u16 >> 15) & 1).astype(np.uint8)
    return exponents, mantissas, signs


def analyze_tensor(name: str, u16: np.ndarray):
    exponents, mantissas, signs = bf16_parts(u16.flatten())

    print(f"\n{'─'*60}")
    print(f"Tensor: {name}  ({len(u16.flatten()):,} weights)")

    # --- Exponent distribution ---
    uniq_exp, cnt_exp = np.unique(exponents, return_counts=True)
    order = np.argsort(-cnt_exp)
    print(f"\nExponents: {len(uniq_exp)} unique values  "
          f"[{int(uniq_exp.min())}–{int(uniq_exp.max())}]")
    print(f"  Top-10 by frequency:")
    for i in order[:10]:
        pct = 100.0 * cnt_exp[i] / len(exponents)
        bar = '█' * int(pct / 2)
        print(f"    exp={uniq_exp[i]:3d}  {cnt_exp[i]:>10,}  ({pct:5.2f}%)  {bar}")

    cum = np.cumsum(cnt_exp[order]) / len(exponents)
    slots_90 = int(np.searchsorted(cum, 0.90)) + 1
    slots_99 = int(np.searchsorted(cum, 0.99)) + 1
    slots_999 = int(np.searchsorted(cum, 0.999)) + 1
    print(f"  Coverage:  90% in {slots_90} palette slots, "
          f"99% in {slots_99}, 99.9% in {slots_999}")

    # Fraction that would hit sidecar at palette sizes 4, 8
    for k in (4, 8):
        top_k_exps = set(uniq_exp[order[:k]].tolist())
        sidecar_pct = 100.0 * np.sum(~np.isin(exponents, list(top_k_exps))) / len(exponents)
        print(f"  Sidecar fraction (k={k} frequency palette): {sidecar_pct:.4f}%")

    # With k-means: how many weights have exponent distance > 2 from nearest palette entry?
    from compression.encoder import _kmeans_palette
    km_pal = _kmeans_palette(uniq_exp, cnt_exp, k=8)
    dists  = np.min(np.abs(exponents.astype(np.int16)[:, None] -
                           km_pal.astype(np.int16)[None, :]), axis=1)
    for threshold in (1, 2, 4):
        outlier_pct = 100.0 * np.mean(dists > threshold)
        print(f"  k-means k=8: weights with nearest-palette dist > {threshold}: {outlier_pct:.4f}%")

    # --- Mantissa distribution ---
    print(f"\nMantissa (7-bit, 0–127): mean={mantissas.mean():.1f}  "
          f"std={mantissas.std():.1f}  median={np.median(mantissas):.0f}")

    # Distribution by top-3 bits (bits 6:5:4 of mantissa — what SCLP6 currently discards)
    top3 = (mantissas >> 4).astype(np.uint8)   # bits 6:4
    top2 = (mantissas >> 5).astype(np.uint8)   # bits 6:5 (what SCLP6 stores)
    top1 = (mantissas >> 6).astype(np.uint8)   # bit 6 only

    for label, vals in [('top-1 bit (SCLP4 stores)', top1),
                        ('top-2 bits (SCLP6 stores)', top2),
                        ('top-3 bits', top3)]:
        uniq_m, cnt_m = np.unique(vals, return_counts=True)
        entropy = -np.sum((cnt_m/cnt_m.sum()) * np.log2(cnt_m/cnt_m.sum() + 1e-12))
        max_entropy = np.log2(len(uniq_m))
        print(f"  {label}: {len(uniq_m)} values, "
              f"entropy={entropy:.3f}/{max_entropy:.3f} bits "
              f"({'near-uniform' if entropy/max_entropy > 0.95 else 'clustered'})")
        pcts = 100.0 * cnt_m / cnt_m.sum()
        print(f"    distribution: " + "  ".join(f"{v}:{p:.1f}%" for v, p in zip(uniq_m, pcts)))

    # Mantissa uniformity per exponent bucket (are mantissas uniform within each exponent?)
    print(f"\n  Mantissa uniformity within top-5 exponent buckets:")
    for exp_val in uniq_exp[order[:5]]:
        mask = exponents == exp_val
        m_in_bucket = mantissas[mask]
        _, cnt_b = np.unique(m_in_bucket, return_counts=True)
        ent = -np.sum((cnt_b/cnt_b.sum()) * np.log2(cnt_b/cnt_b.sum() + 1e-12))
        max_ent = np.log2(128)   # uniform over 0-127
        print(f"    exp={exp_val}: {len(m_in_bucket):,} weights, "
              f"mantissa entropy={ent:.2f}/{max_ent:.2f} ({100*ent/max_ent:.0f}% of uniform)")


def main():
    print("Loading BF16 shards...")
    readers = [gguf.GGUFReader(p) for p in SHARDS]
    tensor_map = {}
    for r in readers:
        for t in r.tensors:
            tensor_map[t.name] = t

    found = [n for n in SAMPLE_TENSORS if n in tensor_map]
    missing = [n for n in SAMPLE_TENSORS if n not in tensor_map]
    if missing:
        # fall back to available tensors of same type
        moe   = [t.name for t in tensor_map.values() if 'exps' in t.name][:2]
        dense = [t.name for t in tensor_map.values() if 'attn_q' in t.name][:1]
        found = list(dict.fromkeys(found + moe + dense))[:4]

    for name in found:
        t = tensor_map[name]
        raw = t.data
        if raw.dtype != np.uint16:
            raw = raw.view(np.uint16)
        analyze_tensor(name, raw)


if __name__ == '__main__':
    main()
