"""
SCLP compression quality evaluation on Llama-3-8B.

Compares quality impact of SCLP compression on all linear layers:
MLP (gate, up, down) and Attention (q, k, v, o).

Usage:
    source eval_env/bin/activate
    python3 tests/test_llama3_ppl.py
"""
import torch
import numpy as np
import os
import sys
import time
import gc

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
def get_target_layers(model):
    """Return (name, module) for all MLP and Attention linear projections in Llama-3."""
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Llama-3 layer names usually contain these strings
            if any(suffix in name for suffix in [
                'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj',
                'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj'
            ]):
                targets.append((name, module))
    return targets


def compute_ppl(model, tokenizer, samples, device, max_length=128):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    print(f"  Computing PPL for {len(samples)} samples...", end="", flush=True)
    with torch.no_grad():
        for i, text in enumerate(samples):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            ids = inputs["input_ids"]
            if ids.shape[1] < 5:
                continue
            outputs = model(ids, labels=ids)
            loss = outputs.loss.item()
            total_loss += loss * ids.shape[1]
            total_tokens += ids.shape[1]
            
            # Frequent GC to keep memory low
            del inputs, ids, outputs
            if i % 5 == 0:
                print(".", end="", flush=True)
                gc.collect()
                
    print(" Done.")
    return float(np.exp(total_loss / max(total_tokens, 1)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    model_id = "unsloth/llama-3-8b"
    device = "cpu"

    print(f"Loading {model_id} (bfloat16, CPU)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Use low_cpu_mem_usage=True and load in bfloat16 directly
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).to(device)

    target_layers = get_target_layers(model)
    total_model_params = sum(p.numel() for p in model.parameters())
    target_params = sum(m.weight.numel() for _, m in target_layers)
    
    print(f"\nModel Summary:")
    print(f"Total params:  {total_model_params:,}")
    print(f"Target layers: {len(target_layers)} layers, {target_params:,} params ({100*target_params/total_model_params:.1f}%)")

    # Exponent distribution analysis (on first layer only for speed)
    print("\nAnalyzing exponent distribution (first layer)...")
    name, module = target_layers[0]
    all_exp = (get_bf16_bits(module.weight).flatten() >> 7) & 0xFF
    unique, counts = np.unique(all_exp, return_counts=True)
    order = np.argsort(-counts)
    total = len(all_exp)
    print(f"  {'Exp':>5}  {'Count':>12}  {'Pct%':>7}")
    for i in order[:10]:
        print(f"  {unique[i]:>5}  {counts[i]:>12,}  {100*counts[i]/total:>6.2f}%")
    
    num_clipped = np.sum(all_exp > 125)
    print(f"  Weights with exp > 125: {num_clipped:,} ({100*num_clipped/total:.4f}%)")
    del all_exp

    print("\nLoading WikiText-2 test samples...")
    test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # Reduced samples and context for speed
    samples = [t for t in test_data["text"] if len(t) > 200][:10]
    print(f"Using {len(samples)} samples")

    # --- Baseline ---
    print("\n--- Baseline ---")
    t0 = time.perf_counter()
    baseline_ppl = compute_ppl(model, tokenizer, samples, device)
    print(f"Baseline PPL: {baseline_ppl:.4f}  (Time: {time.perf_counter()-t0:.1f}s)")

    # --- SCLP Compression Sweep ---
    thresholds = [123, 125] # Reduced sweep for speed
    print(f"\n--- SCLP Compression Threshold Sweep ---")
    print(f"{'Thresh':>7}  {'PPL':>9}  {'ΔPPL%':>8}  {'LayerRatio':>12}")
    print("-" * 45)

    original_weights = {}
    for name, module in target_layers:
        original_weights[name] = get_bf16_bits(module.weight).copy()

    for t in thresholds:
        total_orig_bytes = 0
        total_comp_bytes = 0
        
        # print(f"  Compressing layers at threshold {t}...", flush=True)
        for name, module in target_layers:
            bits = original_weights[name].flatten()
            clipped = soft_exponent_clip(bits, t)
            encoded = encode_palette(clipped)
            
            total_orig_bytes += bits.nbytes
            total_comp_bytes += (len(encoded['palette']) +
                                len(encoded['packed_indices']) +
                                len(encoded['sm_stream']))
            
            decoded = decode_palette(encoded, bits.size)
            set_bf16_bits(module, decoded, module.weight.shape)
            
            del clipped, encoded, decoded
        
        comp_ppl = compute_ppl(model, tokenizer, samples, device)
        delta_ppl = (comp_ppl - baseline_ppl) / baseline_ppl * 100
        layer_ratio = total_orig_bytes / total_comp_bytes
        
        print(f"{t:>7}  {comp_ppl:>9.4f}  {delta_ppl:>+7.2f}%  {layer_ratio:>10.3f}x")
        
        # Restore for next threshold
        for name, module in target_layers:
            set_bf16_bits(module, original_weights[name], module.weight.shape)
        
        gc.collect()

    print("\nSweep complete.")

if __name__ == "__main__":
    main()
