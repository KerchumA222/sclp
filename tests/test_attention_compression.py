"""
SCLP compression quality evaluation on OPT-125m for Attention layers.

Compares quality impact of SCLP compression on:
1. MLP layers only (baseline for this PoC)
2. Attention layers only (q, k, v, out projections)
3. Both MLP and Attention layers

Usage:
    source eval_env/bin/activate
    python3 tests/test_attention_compression.py
"""
import torch
import numpy as np
import os
import sys
import time

# Seed RNG for reproducibility
np.random.seed(42)
torch.manual_seed(42)

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

def get_attn_layers(model):
    """Return (name, module) for all q, k, v, out projection layers in OPT."""
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and any(name.endswith(suffix) for suffix in ['.k_proj', '.v_proj', '.q_proj', '.out_proj'])
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
# Weight injection helpers
# ---------------------------------------------------------------------------
def save_layers(layers):
    return {name: (get_bf16_bits(m.weight).copy(), m.weight.shape)
            for name, m in layers}


def restore_layers(layers, saved):
    for name, module in layers:
        bits, shape = saved[name]
        set_bf16_bits(module, bits, shape)


def apply_full_pipeline(layers, threshold):
    """Clip + encode + decode."""
    total_orig, total_comp = 0, 0
    for _, module in layers:
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
def analyze_exponent_dist(name, layers, threshold=125):
    all_exp = np.concatenate([
        ((get_bf16_bits(m.weight).flatten() >> 7) & 0xFF)
        for _, m in layers
    ])
    unique, counts = np.unique(all_exp, return_counts=True)
    total = len(all_exp)
    order = np.argsort(-counts)

    print(f"\n{name} exponent distribution  ({total:,} weights, {len(layers)} layers)")
    print(f"  {'Exp':>5}  {'Count':>12}  {'Pct%':>7}  {'Cumul%':>8}")
    cumul = 0
    for i in order[:10]:
        cumul += counts[i]
        print(f"  {unique[i]:>5}  {counts[i]:>12,}  "
              f"{100*counts[i]/total:>6.2f}%  {100*cumul/total:>7.2f}%")
    
    num_clipped = np.sum(all_exp > threshold)
    print(f"  Weights with exp > {threshold}: {num_clipped:,} ({100*num_clipped/total:.4f}%)")

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

    mlp_layers = get_mlp_layers(model)
    attn_layers = get_attn_layers(model)
    
    total_model_params = sum(p.numel() for p in model.parameters())
    
    print(f"\nModel Summary:")
    print(f"Total params: {total_model_params:,}")
    
    mlp_params = sum(m.weight.numel() for _, m in mlp_layers)
    print(f"MLP layers:   {len(mlp_layers)} layers, {mlp_params:,} params ({100*mlp_params/total_model_params:.1f}%)")
    
    attn_params = sum(m.weight.numel() for _, m in attn_layers)
    print(f"Attn layers:  {len(attn_layers)} layers, {attn_params:,} params ({100*attn_params/total_model_params:.1f}%)")

    threshold = 125
    analyze_exponent_dist("MLP", mlp_layers, threshold)
    analyze_exponent_dist("Attn", attn_layers, threshold)

    print("\nLoading WikiText-2 test samples...")
    test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    samples = [t for t in test_data["text"] if len(t) > 200][:20]
    print(f"Using {len(samples)} samples")

    # --- Baseline ---
    print("\n--- Baseline ---")
    baseline_ppl = compute_ppl(model, tokenizer, samples, device)
    print(f"PPL: {baseline_ppl:.4f}")

    original_mlp = save_layers(mlp_layers)
    original_attn = save_layers(attn_layers)

    # --- Scenario 1: MLP Only ---
    print(f"\n--- Scenario 1: MLP Only (Threshold={threshold}) ---")
    orig_b, comp_b = apply_full_pipeline(mlp_layers, threshold)
    ppl_mlp = compute_ppl(model, tokenizer, samples, device)
    ratio_mlp = (total_model_params * 2) / ((total_model_params - mlp_params) * 2 + comp_b)
    print(f"PPL: {ppl_mlp:.4f}  (ΔPPL: {(ppl_mlp-baseline_ppl)/baseline_ppl*100:+.2f}%)")
    print(f"Model Compression Ratio: {ratio_mlp:.3f}x")
    restore_layers(mlp_layers, original_mlp)

    # --- Scenario 2: Attn Only ---
    print(f"\n--- Scenario 2: Attn Only (Threshold={threshold}) ---")
    orig_b, comp_b = apply_full_pipeline(attn_layers, threshold)
    ppl_attn = compute_ppl(model, tokenizer, samples, device)
    ratio_attn = (total_model_params * 2) / ((total_model_params - attn_params) * 2 + comp_b)
    print(f"PPL: {ppl_attn:.4f}  (ΔPPL: {(ppl_attn-baseline_ppl)/baseline_ppl*100:+.2f}%)")
    print(f"Model Compression Ratio: {ratio_attn:.3f}x")
    restore_layers(attn_layers, original_attn)

    # --- Scenario 3: MLP + Attn Sweep ---
    print(f"\n--- Scenario 3: MLP + Attn Threshold Sweep ---")
    print(f"{'Thresh':>7}  {'PPL':>9}  {'ΔPPL%':>8}  {'Ratio':>8}")
    print("-" * 45)
    
    for t in [121, 122, 123, 124, 125]:
        o1, c1 = apply_full_pipeline(mlp_layers, t)
        o2, c2 = apply_full_pipeline(attn_layers, t)
        ppl_both = compute_ppl(model, tokenizer, samples, device)
        total_comp_b = c1 + c2
        ratio_both = (total_model_params * 2) / ((total_model_params - mlp_params - attn_params) * 2 + total_comp_b)
        print(f"{t:>7}  {ppl_both:>9.4f}  {(ppl_both-baseline_ppl)/baseline_ppl*100:>+7.2f}%  {ratio_both:>7.3f}x")
        
        restore_layers(mlp_layers, original_mlp)
        restore_layers(attn_layers, original_attn)

if __name__ == "__main__":
    main()
