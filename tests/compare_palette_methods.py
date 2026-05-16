"""
Compare frequency-based vs k-means palette selection for SCLP6.
Generates synthetic BF16 weight tensors matching Gemma4's MoE expert dimensions
and measures per-method reconstruction error.
"""
import sys
import numpy as np

sys.path.insert(0, '/home/ajkerchum/poc/src')
from compression.encoder import encode_palette_6b


def float32_to_bf16_u16(f32: np.ndarray) -> np.ndarray:
    u32 = f32.view(np.uint32)
    # round-to-nearest-even
    rounding = ((u32 >> 16) & 1) + 0x7FFF
    return ((u32 + rounding) >> 16).astype(np.uint16)


def bf16_to_float32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def decode_sclp6_expert(palette: np.ndarray, ws: np.ndarray, n_weights: int) -> np.ndarray:
    n_groups = (n_weights + 3) // 4
    b0 = ws[0::3].astype(np.uint32)
    b1 = ws[1::3].astype(np.uint32)
    b2 = ws[2::3].astype(np.uint32)

    sixbits = np.empty(n_groups * 4, dtype=np.uint8)
    sixbits[0::4] = (b0 >> 2).astype(np.uint8)
    sixbits[1::4] = (((b0 & 0x3) << 4) | (b1 >> 4)).astype(np.uint8)
    sixbits[2::4] = (((b1 & 0xF) << 2) | (b2 >> 6)).astype(np.uint8)
    sixbits[3::4] = (b2 & 0x3F).astype(np.uint8)
    sixbits = sixbits[:n_weights]

    pidx = (sixbits >> 3) & 0x7
    smn  = sixbits & 0x7
    sign = (smn >> 2) & 1
    mant = smn & 0x3

    pal_safe = np.clip(pidx, 0, len(palette) - 1)
    exp_vals = palette[pal_safe]
    bits = (sign.astype(np.uint16) << 15) | (exp_vals.astype(np.uint16) << 7) | (mant.astype(np.uint16) << 5)
    return bf16_to_float32(bits)


def encode_and_measure(u16: np.ndarray, n_experts: int, method: str):
    result   = encode_palette_6b(u16, n_experts=n_experts, palette_method=method)
    orig_f32 = bf16_to_float32(u16.flatten())

    expert_nw = len(u16.flatten()) // n_experts
    ws_all    = result['ws_stream']
    palettes  = result['palette'] if n_experts > 1 else [result['palette']]

    recon = np.empty_like(orig_f32)
    for e in range(n_experts):
        pal    = palettes[e]
        n_grps = (expert_nw + 3) // 4
        ws     = ws_all[e * n_grps * 3 : (e + 1) * n_grps * 3]
        recon[e * expert_nw : (e + 1) * expert_nw] = decode_sclp6_expert(pal, ws, expert_nw)

    err  = recon - orig_f32
    mse  = float(np.mean(err ** 2))
    mask = np.abs(orig_f32) > 1e-6
    rel  = np.abs(err[mask]) / np.abs(orig_f32[mask])
    return mse, float(rel.mean()), float(rel.max())


def make_moe_expert_tensor(rng, n_experts=128, N=2816, K=704):
    """
    Synthetic BF16 MoE expert weights matching Gemma4 26B-A4B dimensions.
    Normal(0, 0.02) clipped to ±0.5 — typical initialisation range for MLP experts.
    """
    f32  = rng.normal(0, 0.02, size=(n_experts * N * K,)).astype(np.float32)
    f32  = np.clip(f32, -0.5, 0.5)
    return float32_to_bf16_u16(f32)


def make_attention_tensor(rng, K=2816, N=2816):
    """Dense attention projection — slightly tighter distribution."""
    f32 = rng.normal(0, 0.01, size=(K * N,)).astype(np.float32)
    return float32_to_bf16_u16(f32)


def main():
    rng = np.random.default_rng(0)

    cases = [
        ('MoE gate/up   [128×2816×704]', make_moe_expert_tensor(rng, 128, 2816, 704), 128),
        ('MoE down      [128×704×2816]', make_moe_expert_tensor(rng, 128,  704, 2816), 128),
        ('Attn proj     [2816×2816]',    make_attention_tensor(rng, 2816, 2816),        1),
        ('MoE gate/up 2 [128×2816×704]', make_moe_expert_tensor(rng, 128, 2816, 704), 128),
    ]

    hdr = f"{'Tensor':<34} {'Method':<10} {'MSE':>12} {'MeanRel%':>10} {'MaxRel%':>10} {'Δ MSE%':>10}"
    print(hdr)
    print('─' * len(hdr))

    for label, u16, n_exp in cases:
        results = {}
        for method in ('frequency', 'kmeans'):
            mse, mr, xr = encode_and_measure(u16, n_exp, method)
            results[method] = (mse, mr, xr)

        for method in ('frequency', 'kmeans'):
            mse, mr, xr = results[method]
            if method == 'kmeans':
                delta = 100.0 * (results['kmeans'][0] - results['frequency'][0]) / results['frequency'][0]
                delta_str = f"{delta:>+10.2f}"
            else:
                delta_str = f"{'(baseline)':>10}"
            print(f"{label:<34} {method:<10} {mse:>12.4e} {100*mr:>9.4f}% {100*xr:>9.2f}% {delta_str}")
        print()

    # also show palette contents for one small example
    print("── Palette comparison (MoE gate, expert 0) ──")
    rng2 = np.random.default_rng(0)
    u16  = make_moe_expert_tensor(rng2, 128, 2816, 704)
    expert_nw = u16.size // 128
    ex0 = u16[:expert_nw]

    from compression.encoder import _kmeans_palette
    exponents = ((ex0 >> 7) & 0xFF).astype(np.uint8)
    uniq, cnts = np.unique(exponents, return_counts=True)
    freq_pal   = uniq[np.argsort(-cnts)][:8]
    km_pal     = _kmeans_palette(uniq, cnts, k=8)

    print(f"  Frequency palette: {sorted(freq_pal.tolist())}")
    print(f"  K-means   palette: {sorted(km_pal.tolist())}")
    print(f"  All unique exponents ({len(uniq)}): min={uniq.min()}, max={uniq.max()}, "
          f"top-3 by count: {uniq[np.argsort(-cnts)[:3]].tolist()}")


if __name__ == '__main__':
    main()
