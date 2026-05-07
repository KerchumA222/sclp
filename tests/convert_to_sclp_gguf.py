import sys
import os
import struct
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoConfig

sys.path.append("/home/ajkerchum/llama.cpp/gguf-py")
sys.path.append("/home/ajkerchum/poc/src")

from gguf import GGUFWriter, GGMLQuantizationType
from compression.encoder import encode_palette

# Layers targeted for SCLP compression (linear projections only)
SCLP_TARGETS = [
    'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj',
    'self_attn.q_proj', 'self_attn.k_proj',
    'self_attn.v_proj', 'self_attn.o_proj',
]

# HuggingFace → GGUF tensor name mapping for Llama architecture
HF_TO_GGUF = {
    'model.embed_tokens.weight':                   'token_embd.weight',
    'model.norm.weight':                           'output_norm.weight',
    'lm_head.weight':                              'output.weight',
}

def hf_layer_to_gguf(name):
    """Convert HuggingFace layer name to GGUF tensor name."""
    # model.layers.{i}.self_attn.q_proj.weight → blk.{i}.attn_q.weight
    attn_map = {
        'self_attn.q_proj': 'attn_q',
        'self_attn.k_proj': 'attn_k',
        'self_attn.v_proj': 'attn_v',
        'self_attn.o_proj': 'attn_output',
    }
    mlp_map = {
        'mlp.gate_proj': 'ffn_gate',
        'mlp.up_proj':   'ffn_up',
        'mlp.down_proj': 'ffn_down',
    }
    norm_map = {
        'input_layernorm':     'attn_norm',
        'post_attention_layernorm': 'ffn_norm',
    }
    import re
    m = re.match(r'model\.layers\.(\d+)\.(.*?)\.weight$', name)
    if m:
        idx, sub = m.group(1), m.group(2)
        for hf_key, gguf_key in {**attn_map, **mlp_map, **norm_map}.items():
            if sub == hf_key:
                return f'blk.{idx}.{gguf_key}.weight'
    return HF_TO_GGUF.get(name, name)  # fallback: return as-is


def bf16_param_to_uint16(param):
    """Return BF16 weight bits as a flattened uint16 numpy array."""
    # Keep as bfloat16, view int16 bits, then cast to uint16 (same bit pattern)
    return param.detach().bfloat16().view(torch.int16).numpy().view(np.uint16).flatten()


def build_sclp_payload(data_u16):
    """Encode uint16 BF16 weights as SCLP payload bytes."""
    encoded = encode_palette(data_u16)
    num_weights = len(data_u16)
    palette = encoded['palette'].astype(np.uint8)
    packed  = encoded['packed_indices'].astype(np.uint8)
    sm      = encoded['sm_stream'].astype(np.uint8)
    header  = struct.pack("<IB", num_weights, len(palette))
    return header + palette.tobytes() + packed.tobytes() + sm.tobytes()


def convert_to_sclp(model_id, output_path):
    print(f"Loading {model_id}...")
    config = AutoConfig.from_pretrained(model_id)
    model  = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = GGUFWriter(output_path, "llama")

    # Required KV metadata for llama.cpp to load the model
    writer.add_uint32("general.file_type", 1)  # F16 (non-SCLP tensors are F32 here)
    writer.add_uint32("llama.context_length",       config.max_position_embeddings)
    writer.add_uint32("llama.embedding_length",     config.hidden_size)
    writer.add_uint32("llama.block_count",          config.num_hidden_layers)
    writer.add_uint32("llama.feed_forward_length",  config.intermediate_size)
    writer.add_uint32("llama.rope.dimension_count",
                      config.hidden_size // config.num_attention_heads)
    writer.add_uint32("llama.attention.head_count",    config.num_attention_heads)
    writer.add_uint32("llama.attention.head_count_kv",
                      getattr(config, 'num_key_value_heads', config.num_attention_heads))
    writer.add_float32("llama.attention.layer_norm_rms_epsilon", config.rms_norm_eps)
    writer.add_uint32("llama.vocab_size", config.vocab_size)

    for hf_name, param in model.named_parameters():
        gguf_name = hf_layer_to_gguf(hf_name)
        is_sclp = any(t in hf_name for t in SCLP_TARGETS)

        if is_sclp:
            print(f"  SCLP  {hf_name} → {gguf_name}")
            data = bf16_param_to_uint16(param)
            payload = build_sclp_payload(data)

            # SCLP type_size = 2 (same as BF16), so logical shape = param.shape
            # Pad payload to num_weights * 2 bytes so GGUF offsets are aligned
            logical_bytes = data.size * 2
            if len(payload) < logical_bytes:
                payload += b'\x00' * (logical_bytes - len(payload))
            elif len(payload) > logical_bytes:
                raise ValueError(
                    f"SCLP payload ({len(payload)} B) > slot ({logical_bytes} B)"
                )

            np_payload = np.frombuffer(payload, dtype=np.uint8)
            writer.add_tensor(gguf_name, np_payload,
                              raw_dtype=GGMLQuantizationType.SCLP,
                              raw_shape=list(param.shape))
        else:
            print(f"  BF16  {hf_name} → {gguf_name}")
            data = bf16_param_to_uint16(param)
            np_bf16 = data.reshape(param.shape)
            writer.add_tensor(gguf_name, np_bf16,
                              raw_dtype=GGMLQuantizationType.BF16,
                              raw_shape=list(param.shape))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Written: {output_path}")


if __name__ == "__main__":
    convert_to_sclp(
        "unsloth/Meta-Llama-3-8B",
        "/home/ajkerchum/poc/models/llama3/Llama-3-8B-Native-SCLP.gguf"
    )
