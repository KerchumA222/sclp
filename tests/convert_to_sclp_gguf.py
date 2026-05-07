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
    """Encode uint16 BF16 weights as a padded SCLP blob.

    Padded to exactly num_weights*2 bytes so GGUF offset arithmetic (which
    derives tensor size from shape × type_size) stays correct.
    """
    encoded = encode_palette(data_u16)
    num_weights = len(data_u16)
    palette = encoded['palette'].astype(np.uint8)
    packed  = encoded['packed_indices'].astype(np.uint8)
    sm      = encoded['sm_stream'].astype(np.uint8)
    sidecar_indices = encoded['sidecar']['indices'].astype(np.uint32)
    sidecar_values  = encoded['sidecar']['values'].astype(np.uint16)

    header = struct.pack("<IB", num_weights, len(palette))
    sidecar_hdr = struct.pack("<I", len(sidecar_indices))
    blob = (
        header
        + palette.tobytes()
        + packed.tobytes()
        + sm.tobytes()
        + sidecar_hdr
        + sidecar_indices.tobytes()
        + sidecar_values.tobytes()
    )
    target_size = num_weights * 2
    assert len(blob) <= target_size, (
        f"SCLP blob {len(blob)} > BF16 size {target_size} — sidecar too large"
    )
    return blob + b'\x00' * (target_size - len(blob))


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
    # Use existing FP16 GGUF and patch it in-place
    import sys
    import os
    import struct
    import shutil
    import numpy as np
    sys.path.append(os.path.abspath('/home/ajkerchum/llama.cpp/gguf-py/'))
    sys.path.append(os.path.abspath('/home/ajkerchum/poc/src'))

    from gguf import GGUFReader, GGMLQuantizationType
    from compression.encoder import encode_palette

    SCLP_TARGET_SUFFIXES = [
        'attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'attn_output.weight',
        'ffn_gate.weight', 'ffn_up.weight', 'ffn_down.weight',
    ]

    def is_sclp_target(name: str) -> bool:
        return any(name.endswith(s) for s in SCLP_TARGET_SUFFIXES)

    def to_bf16_uint16(tensor_data: np.ndarray, tensor_type: GGMLQuantizationType) -> np.ndarray:
        raw = tensor_data.flatten()
        if tensor_type == GGMLQuantizationType.BF16:
            return raw.view(np.uint16)
        if tensor_type == GGMLQuantizationType.F16:
            f32 = raw.view(np.float16).astype(np.float32)
            return (f32.view(np.uint32) >> 16).astype(np.uint16)
        if tensor_type == GGMLQuantizationType.F32:
            return (raw.view(np.float32).view(np.uint32) >> 16).astype(np.uint16)
        raise ValueError(f"Unsupported source type: {tensor_type}")

    input_path = "/home/ajkerchum/poc/models/llama3/Meta-Llama-3-8B.fp16.gguf"
    output_path = "/home/ajkerchum/poc/models/llama3/Llama-3-8B-SCLP-Compressed.gguf"

    print(f"Reading {input_path}...")
    reader = GGUFReader(input_path, mode='r')

    targets = [t for t in reader.tensors if is_sclp_target(t.name)]
    print(f"Found {len(targets)} SCLP targets out of {len(reader.tensors)} tensors.")

    print(f"Copying {input_path} → {output_path}...")
    shutil.copy2(input_path, output_path)

    total_original_bytes = 0
    total_payload_bytes = 0
    total_sidecar = 0

    with open(output_path, 'r+b') as f:
        for i, tensor in enumerate(targets):
            data = to_bf16_uint16(tensor.data, tensor.tensor_type)
            encoded = encode_palette(data)

            num_weights = len(data)
            palette = encoded['palette'].astype(np.uint8)
            packed = encoded['packed_indices'].astype(np.uint8)
            sm = encoded['sm_stream'].astype(np.uint8)
            sidecar_indices = encoded['sidecar']['indices'].astype(np.uint32)
            sidecar_values = encoded['sidecar']['values'].astype(np.uint16)
            sidecar_count = len(sidecar_indices)

            # Build payload with ACTUAL compressed size (no padding)
            header = struct.pack("<IB", num_weights, len(palette))
            sidecar_hdr = struct.pack("<I", sidecar_count)
            payload = bytearray(
                header
                + palette.tobytes()
                + packed.tobytes()
                + sm.tobytes()
                + sidecar_hdr
                + sidecar_indices.tobytes()
                + sidecar_values.tobytes()
            )

            payload_bytes = len(payload)

            if payload_bytes > tensor.n_bytes:
                raise ValueError(
                    f"SCLP payload ({payload_bytes} B) exceeds original ({tensor.n_bytes} B)"
                )

            # Update tensor type to SCLP (42)
            field = tensor.field
            type_field_offset = (field.offset
                         + field.parts[0].nbytes
                         + field.parts[1].nbytes
                         + field.parts[2].nbytes
                         + field.parts[3].nbytes)

            f.seek(type_field_offset)
            f.write(struct.pack("<I", int(GGMLQuantizationType.SCLP)))

            # Write compressed payload (not padded)
            f.seek(tensor.data_offset)
            f.write(payload)

            ratio = tensor.n_bytes / payload_bytes
            total_original_bytes += tensor.n_bytes
            total_payload_bytes += payload_bytes
            total_sidecar += sidecar_count
            sc_note = f" ({sidecar_count} sidecar)" if sidecar_count else ""
            print(f"  [{i+1:3d}/{len(targets)}] {tensor.name}: {ratio:.3f}x{sc_note}")

    overall_ratio = total_original_bytes / total_payload_bytes
    savings_mb = (total_original_bytes - total_payload_bytes) / (1024 ** 2)
    total_weights = total_original_bytes // 2
    sidecar_pct = 100.0 * total_sidecar / total_weights if total_weights else 0
    print(f"\nDone. {len(targets)} tensors compressed.")
    print(f"Overall ratio: {overall_ratio:.3f}x ({savings_mb:.1f} MiB saved)")
    print(f"Sidecar entries: {total_sidecar:,} weights ({sidecar_pct:.4f}% of total)")
