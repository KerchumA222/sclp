import sys
import os
import struct
import shutil
import numpy as np

sys.path.append(os.path.abspath('/home/ajkerchum/llama.cpp/gguf-py/'))
sys.path.append(os.path.abspath('/home/ajkerchum/poc/src'))

from gguf import GGUFReader, GGMLQuantizationType
from compression.encoder import encode_palette

# GGUF tensor name suffixes targeted for SCLP compression (all linear projections)
SCLP_TARGET_SUFFIXES = [
    'attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'attn_output.weight',
    'ffn_gate.weight', 'ffn_up.weight', 'ffn_down.weight',
]

def is_sclp_target(name: str) -> bool:
    return any(name.endswith(s) for s in SCLP_TARGET_SUFFIXES)


def to_bf16_uint16(tensor_data: np.ndarray, tensor_type: GGMLQuantizationType) -> np.ndarray:
    """Return BF16 bit patterns as uint16, converting from any input dtype."""
    raw = tensor_data.flatten()
    if tensor_type == GGMLQuantizationType.BF16:
        return raw.view(np.uint16)
    if tensor_type == GGMLQuantizationType.F16:
        f32 = raw.view(np.float16).astype(np.float32)
        return (f32.view(np.uint32) >> 16).astype(np.uint16)
    if tensor_type == GGMLQuantizationType.F32:
        return (raw.view(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    raise ValueError(f"Unsupported source type for SCLP encoding: {tensor_type}")


def patch_gguf_with_sclp(input_path, output_path):
    print(f"Reading {input_path}...")
    reader = GGUFReader(input_path, mode='r')

    targets = [t for t in reader.tensors if is_sclp_target(t.name)]
    total = len(reader.tensors)
    print(f"Found {len(targets)} SCLP targets out of {total} tensors.")

    print(f"Copying {input_path} → {output_path}...")
    shutil.copy2(input_path, output_path)

    total_original_bytes = 0
    total_payload_bytes  = 0
    total_sidecar        = 0

    with open(output_path, 'r+b') as f:
        for i, tensor in enumerate(targets, 1):
            data = to_bf16_uint16(tensor.data, tensor.tensor_type)
            encoded = encode_palette(data)

            num_weights     = len(data)
            palette         = encoded['palette'].astype(np.uint8)
            packed          = encoded['packed_indices'].astype(np.uint8)
            sm              = encoded['sm_stream'].astype(np.uint8)
            sidecar_indices = encoded['sidecar']['indices'].astype(np.uint32)
            sidecar_values  = encoded['sidecar']['values'].astype(np.uint16)
            sidecar_count   = len(sidecar_indices)

            # Blob layout:
            #   [uint32 num_weights][uint8 palette_size][palette]
            #   [packed_indices][sm_stream]
            #   [uint32 sidecar_count][uint32[] indices][uint16[] values]
            #   [zero padding to num_weights*2 bytes]
            header        = struct.pack("<IB", num_weights, len(palette))
            sidecar_hdr   = struct.pack("<I", sidecar_count)
            sclp_payload  = bytearray(
                header
                + palette.tobytes()
                + packed.tobytes()
                + sm.tobytes()
                + sidecar_hdr
                + sidecar_indices.tobytes()
                + sidecar_values.tobytes()
            )

            payload_bytes = len(sclp_payload)

            if payload_bytes > tensor.n_bytes:
                raise ValueError(
                    f"SCLP payload ({payload_bytes} B) exceeds original tensor "
                    f"({tensor.n_bytes} B) for {tensor.name} — cannot fit in-place."
                )
            sclp_payload += b'\x00' * (tensor.n_bytes - payload_bytes)

            # field.parts = [name_len, name_data, n_dims, dims, raw_dtype, offset_tensor]
            field = tensor.field
            type_field_offset = (field.offset
                                 + field.parts[0].nbytes   # name length (uint64)
                                 + field.parts[1].nbytes   # name bytes
                                 + field.parts[2].nbytes   # n_dims (uint32)
                                 + field.parts[3].nbytes)  # dims (n_dims × uint64)

            f.seek(type_field_offset)
            f.write(struct.pack("<I", int(GGMLQuantizationType.SCLP)))

            f.seek(tensor.data_offset)
            f.write(sclp_payload)

            ratio = tensor.n_bytes / payload_bytes
            total_original_bytes += tensor.n_bytes
            total_payload_bytes  += payload_bytes
            total_sidecar        += sidecar_count
            sc_note = f" ({sidecar_count} sidecar)" if sidecar_count else ""
            print(f"  [{i:3d}/{len(targets)}] {tensor.name}: {ratio:.3f}x{sc_note}")

    overall_ratio = total_original_bytes / total_payload_bytes
    savings_mb = (total_original_bytes - total_payload_bytes) / (1024 ** 2)
    total_weights = total_original_bytes // 2
    sidecar_pct = 100.0 * total_sidecar / total_weights if total_weights else 0
    print(f"\nDone. {len(targets)} tensors compressed.")
    print(f"Overall ratio:   {overall_ratio:.3f}x  ({savings_mb:.1f} MiB saved)")
    print(f"Sidecar entries: {total_sidecar:,} weights ({sidecar_pct:.4f}% of total)")


if __name__ == "__main__":
    patch_gguf_with_sclp(
        "/home/ajkerchum/poc/models/llama3/Meta-Llama-3-8B.fp16.gguf",
        "/home/ajkerchum/poc/models/llama3/Llama-3-8B-SCLP-Patched.gguf",
    )
