"""Test SCLP4 encode-decode roundtrip on real model weights."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, '/home/ajkerchum/llama.cpp/gguf-py')

import numpy as np
from gguf import GGUFReader
from compression.encoder import encode_palette_4b, encode_palette_6b
from compression.decoder import decode_palette_4b, decode_palette_6b


def compute_metrics(orig: np.ndarray, decoded: np.ndarray, name: str):
    """Compute reconstruction quality metrics."""
    orig_f32 = orig.view(np.int16).astype(np.int32)  # use int for diff
    dec_f32 = decoded.view(np.int16).astype(np.int32)

    # Convert BF16 bits to float32 for accurate comparison
    orig_u32 = orig.astype(np.uint32) << 16
    dec_u32 = decoded.astype(np.uint32) << 16
    orig_fp = orig_u32.view(np.float32)
    dec_fp = dec_u32.view(np.float32)

    eps = 1e-10
    abs_err = np.abs(dec_fp - orig_fp)
    rel_err = abs_err / (np.abs(orig_fp) + eps)

    print(f"\n{name}:")
    print(f"  MSE: {(abs_err**2).mean():.2e}")
    print(f"  MaxAbsErr: {abs_err.max():.4e}")
    print(f"  MaxRelErr: {rel_err.max():.4e}")
    print(f"  Mean orig magnitude: {np.abs(orig_fp).mean():.4e}")
    # Check NaN/Inf
    n_bad = np.sum(np.isnan(dec_fp) | np.isinf(dec_fp))
    print(f"  NaN/Inf in decoded: {n_bad}")


def main():
    SHARD = '/home/ajkerchum/poc/models/gemma4/google_gemma-4-26B-A4B-it-bf16/google_gemma-4-26B-A4B-it-bf16-00001-of-00002.gguf'
    reader = GGUFReader(SHARD)

    # Test on a dense attention tensor (n_experts=1)
    for t in reader.tensors:
        if t.name == 'blk.0.attn_q.weight':
            print(f"=== Testing: {t.name} ===")
            print(f"Shape: {t.shape.tolist()}, type: {t.tensor_type}")
            data = t.data.view(np.uint16).flatten() if t.data.dtype != np.uint16 else t.data.flatten()
            print(f"Total weights: {len(data)}")

            # SCLP4 encode-decode roundtrip
            enc4 = encode_palette_4b(data, n_experts=1, palette_method='kmeans', sidecar_dist=1)
            print(f"SCLP4 palette: {enc4['palette']}")
            print(f"SCLP4 sidecar: {len(enc4['sidecar']['indices'])} ({100*len(enc4['sidecar']['indices'])/len(data):.2f}%)")
            dec4 = decode_palette_4b(enc4, num_weights=len(data))
            compute_metrics(data, dec4, "SCLP4 (k-means, sidecar_dist=1)")

            # SCLP6 for comparison
            enc6 = encode_palette_6b(data, n_experts=1, palette_method='kmeans', sidecar_dist=1)
            print(f"\nSCLP6 palette: {enc6['palette']}")
            print(f"SCLP6 sidecar: {len(enc6['sidecar']['indices'])} ({100*len(enc6['sidecar']['indices'])/len(data):.2f}%)")
            dec6 = decode_palette_6b(enc6, num_weights=len(data))
            compute_metrics(data, dec6, "SCLP6 (k-means, sidecar_dist=1)")
            break


if __name__ == '__main__':
    main()
