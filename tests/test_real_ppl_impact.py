"""
SCLP compression quality evaluation on OPT-125m.

Tests three stages independently to isolate quality impact of each component:
  Stage A - Clipping only: soft_exponent_clip, no encode/decode
  Stage B - Mantissa truncation only: drop bottom 4 mantissa bits (optional lossy
             stage from design.md §4.2; the encoder no longer does this by default)
  Stage C - Full pipeline: clip + encode + decode (clipping + mantissa truncation)

Also reports the exponent distribution and compression ratios.

Usage:
    source eval_env/bin/activate
    python3 tests/test_real_ppl_impact.py
"""
import torch
import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.compression.clipping import soft_exponent_clip
from src.compression.encoder import encode_palette
from src.compression.decoder import decode_palette

torch.set_num_threads(16)


# ---------------------------------------------------------------------------
# BF16 tensor helpers
# ---------------------------------------------------------------------------
def get_bf16_bits(tensor: torch.Tensor) -> np.ndarray:
    """Return raw uint16 bit patterns from a bfloat16 weight tensor."""
    return tensor.detach().contiguous().cpu().view(torch.int16).numpy().view(np.uint16)


def set_bf16_bits(module: torch.nn.Module, bits: np.ndarray, shape):
    """Re-inject uint16 BF16 bit patterns into a Linear layer's weight."""
    t = torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16).reshape(shape)
    module.weight.data.copy_(t)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def get_mlp_layers(model):
    """Return (name, module) for all fc1/fc2 MLP layers in OPT."""
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and (name.endswith('.fc1') or name.endswith('.fc2'))
    ]


def compute_ppl(model, tokenizer, samples, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for text in samples:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            ids = inputs["input_ids"]
            if ids.shape[1] < 5:
                continue
            loss = model(ids, labels=ids).loss.item()
            total_loss += loss * ids.shape[1]
            total_tokens += ids.shape[1]
    return float(np.exp(total_loss / max(total_tokens, 1)))


# ---------------------------------------------------------------------------
# Weight injection helpers for each stage
# ---------------------------------------------------------------------------
def save_all_mlp(mlp_layers):
    return {name: (get_bf16_bits(m.weight).copy(), m.weight.shape)
            for name, m in mlp_layers}


def restore_all_mlp(mlp_layers, saved):
    for name, module in mlp_layers:
        bits, shape = saved[name]
        set_bf16_bits(module, bits, shape)


def apply_clip_only(mlp_layers, threshold):
    """Stage A: clip exponents, inject directly (no encode/decode, no mantissa loss)."""
    for _, module in mlp_layers:
        bits = get_bf16_bits(module.weight).flatten()
        clipped = soft_exponent_clip(bits, threshold)
        set_bf16_bits(module, clipped, module.weight.shape)


def apply_mantissa_trunc_only(mlp_layers):
    """Stage B: drop bottom 4 mantissa bits (optional design.md §4.2 step), no clipping."""
    for _, module in mlp_layers:
        bits = get_bf16_bits(module.weight).flatten()
        truncated = bits & np.uint16(0xFFF0)  # zero out bits 3:0 of mantissa
        set_bf16_bits(module, truncated, module.weight.shape)


def apply_full_pipeline(mlp_layers, threshold):
    """Stage C: clip + encode + decode (both lossy stages combined)."""
    total_orig, total_comp = 0, 0
    for _, module in mlp_layers:
        bits = get_bf16_bits(module.weight).flatten()
        clipped = soft_exponent_clip(bits, threshold)
        encoded = encode_palette(clipped)
        total_orig += bits.nbytes
        total_comp += (len(encoded['palette']) +
                       len(encoded['packed_indices']) +
                       len(encoded['sm_stream']))
        decoded = decode_palette(encoded, bits.size)
        set_bf16_bits(module, decoded, module.weight.shape)
    return total_orig, total_comp


# ---------------------------------------------------------------------------
# Exponent distribution analysis
# ---------------------------------------------------------------------------
def analyze_exponent_dist(mlp_layers):
    all_exp = np.concatenate([
        ((get_bf16_bits(m.weight).flatten() >> 7) & 0xFF)
        for _, m in mlp_layers
    ])
    unique, counts = np.unique(all_exp, return_counts=True)
    total = len(all_exp)
    order = np.argsort(-counts)

    print(f"\nMLP exponent distribution  ({total:,} weights, {len(mlp_layers)} layers)")
    print(f"  {'Exp':>5}  {'Count':>12}  {'Pct%':>7}  {'Cumul%':>8}")
    cumul = 0
    for i in order[:20]:
        cumul += counts[i]
        print(f"  {unique[i]:>5}  {counts[i]:>12,}  "
              f"{100*counts[i]/total:>6.2f}%  {100*cumul/total:>7.2f}%")
    print(f"  ({len(unique)} unique exponents total)")

    sorted_counts = np.sort(counts)[::-1]
    cumul_pct = np.cumsum(sorted_counts) / total
    n99  = int(np.searchsorted(cumul_pct, 0.99))  + 1
    n999 = int(np.searchsorted(cumul_pct, 0.999)) + 1
    print(f"  Exponents covering 99%: {n99},  99.9%: {n999}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    model_id = "facebook/opt-125m"
    device = "cpu"

    print(f"Loading {model_id} (bfloat16, CPU)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16
    ).to(device)

    dtype = next(model.parameters()).dtype
    print(f"Weight dtype: {dtype}")
    if dtype != torch.bfloat16:
        print(f"WARNING: expected bfloat16, got {dtype}. BF16 bit layout assumed; results may be inaccurate.")

    mlp_layers = get_mlp_layers(model)
    total_mlp_params = sum(m.weight.numel() for _, m in mlp_layers)
    total_model_params = sum(p.numel() for p in model.parameters())
    print(f"MLP layers: {len(mlp_layers)}  "
          f"({total_mlp_params:,} params, {total_mlp_params*2/1e6:.1f} MB BF16, "
          f"{100*total_mlp_params/total_model_params:.1f}% of model)")

    analyze_exponent_dist(mlp_layers)

    print("\nLoading WikiText-2 test samples...")
    test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    samples = [t for t in test_data["text"] if len(t) > 200][:20]
    print(f"Using {len(samples)} samples")

    # --- Baseline ---
    print("\n--- Baseline ---")
    t0 = time.perf_counter()
    baseline_ppl = compute_ppl(model, tokenizer, samples, device)
    print(f"PPL: {baseline_ppl:.4f}  ({time.perf_counter()-t0:.1f}s)")

    # --- Stage A: Clipping only (no mantissa loss) ---
    print("\n--- Stage A: Clipping only (no encode/decode, no mantissa truncation) ---")
    print(f"{'Thresh':>7}  {'Clipped%':>9}  {'PPL':>9}  {'ΔPPL%':>8}")
    print("-" * 40)

    original_bits = save_all_mlp(mlp_layers)

    # Compute fraction clipped at each threshold from the distribution
    all_exp = np.concatenate([
        ((get_bf16_bits(m.weight).flatten() >> 7) & 0xFF)
        for _, m in mlp_layers
    ])

    thresholds = [117, 119, 121, 122, 123, 124, 125]
    for threshold in thresholds:
        pct_clipped = 100.0 * np.mean(all_exp > threshold)
        apply_clip_only(mlp_layers, threshold)
        ppl = compute_ppl(model, tokenizer, samples, device)
        delta = (ppl - baseline_ppl) / baseline_ppl * 100
        print(f"{threshold:>7}  {pct_clipped:>8.2f}%  {ppl:>9.4f}  {delta:>+7.2f}%")
        restore_all_mlp(mlp_layers, original_bits)

    # --- Stage B: Mantissa truncation only (bottom 4 bits zeroed) ---
    print("\n--- Stage B: Mantissa truncation only (bits 3:0 zeroed, no clipping) ---")
    apply_mantissa_trunc_only(mlp_layers)
    ppl_trunc = compute_ppl(model, tokenizer, samples, device)
    delta_trunc = (ppl_trunc - baseline_ppl) / baseline_ppl * 100
    print(f"PPL: {ppl_trunc:.4f}  ΔPPL: {delta_trunc:+.2f}%")
    restore_all_mlp(mlp_layers, original_bits)

    # --- Stage C: Full pipeline (clip + encode/decode) ---
    print("\n--- Stage C: Full pipeline (clip + encode/decode, both lossy stages) ---")
    print(f"{'Thresh':>7}  {'Ratio':>7}  {'Comp MB':>8}  {'PPL':>9}  {'ΔPPL%':>8}")
    print("-" * 50)

    for threshold in [122, 123, 124, 125]:
        orig_bytes, comp_bytes = apply_full_pipeline(mlp_layers, threshold)
        ppl = compute_ppl(model, tokenizer, samples, device)
        delta = (ppl - baseline_ppl) / baseline_ppl * 100
        ratio = orig_bytes / comp_bytes
        print(f"{threshold:>7}  {ratio:>7.3f}x  {comp_bytes/1e6:>8.1f}  {ppl:>9.4f}  {delta:>+7.2f}%")
        restore_all_mlp(mlp_layers, original_bits)

    print(f"\nOriginal MLP: {total_mlp_params*2/1e6:.1f} MB  |  "
          f"Total model: {total_model_params*2/1e6:.1f} MB")


if __name__ == "__main__":
    main()
