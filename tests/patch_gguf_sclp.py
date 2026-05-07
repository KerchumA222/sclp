import sys
import os
import struct
import shutil
import numpy as np

sys.path.append(os.path.abspath('/home/ajkerchum/llama.cpp/gguf-py/'))
sys.path.append(os.path.abspath('/home/ajkerchum/poc/src'))

from gguf import GGUFReader, GGMLQuantizationType
from compression.encoder import encode_palette


def to_bf16_uint16(tensor_data: np.ndarray, tensor_type: GGMLQuantizationType) -> np.ndarray:
    """Return BF16 bit patterns as uint16, converting from any input dtype."""
    raw = tensor_data.flatten()
    if tensor_type == GGMLQuantizationType.BF16:
        return raw.view(np.uint16)
    if tensor_type == GGMLQuantizationType.F16:
        # F16 → float32 → BF16 (take upper 16 bits of float32 bits)
        f32 = raw.view(np.float16).astype(np.float32)
        return (f32.view(np.uint32) >> 16).astype(np.uint16)
    if tensor_type == GGMLQuantizationType.F32:
        return (raw.view(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    raise ValueError(f"Unsupported source type for SCLP encoding: {tensor_type}")


def patch_gguf_with_sclp(input_path, output_path, target_tensor_name):
    print(f"Reading {input_path}...")
    reader = GGUFReader(input_path, mode='r')

    tensor = None
    for t in reader.tensors:
        if t.name == target_tensor_name:
            tensor = t
            break

    if tensor is None:
        print(f"Tensor '{target_tensor_name}' not found in GGUF.")
        return

    print(f"Compressing {target_tensor_name} ({tensor.n_bytes} bytes, "
          f"{tensor.n_elements} weights)...")

    # Convert source data to BF16 uint16 bit patterns (F16/BF16/F32 all handled)
    data = to_bf16_uint16(tensor.data, tensor.tensor_type)
    encoded = encode_palette(data)

    num_weights = len(data)
    palette      = encoded['palette'].astype(np.uint8)
    packed       = encoded['packed_indices'].astype(np.uint8)
    sm           = encoded['sm_stream'].astype(np.uint8)

    header = struct.pack("<IB", num_weights, len(palette))
    sclp_payload = bytearray(header + palette.tobytes() + packed.tobytes() + sm.tobytes())

    if len(sclp_payload) > tensor.n_bytes:
        raise ValueError(
            f"SCLP payload ({len(sclp_payload)} B) exceeds original tensor "
            f"({tensor.n_bytes} B) — cannot fit in-place."
        )
    # Zero-pad to exact original size so all GGUF tensor offsets stay valid
    sclp_payload += b'\x00' * (tensor.n_bytes - len(sclp_payload))

    # Compute file offset of the tensor's type uint32 in the tensor-info section.
    # field.parts = [name_len, name_data, n_dims, dims, raw_dtype, offset_tensor]
    field = tensor.field
    type_field_offset = (field.offset
                         + field.parts[0].nbytes   # name length (uint64)
                         + field.parts[1].nbytes   # name bytes
                         + field.parts[2].nbytes   # n_dims (uint32)
                         + field.parts[3].nbytes)  # dims (n_dims × uint64)

    print(f"Copying {input_path} → {output_path}...")
    shutil.copy2(input_path, output_path)

    with open(output_path, 'r+b') as f:
        # Patch: change tensor type to SCLP (42)
        f.seek(type_field_offset)
        f.write(struct.pack("<I", int(GGMLQuantizationType.SCLP)))

        # Patch: overwrite tensor data with SCLP payload
        f.seek(tensor.data_offset)
        f.write(sclp_payload)

    ratio = tensor.n_bytes / (len(header) + len(palette) + len(packed) + len(sm))
    print(f"Done. Compression ratio: {ratio:.3f}x "
          f"({len(header)+len(palette)+len(packed)+len(sm)} B payload, "
          f"{tensor.n_bytes} B slot)")


if __name__ == "__main__":
    patch_gguf_with_sclp(
        "/home/ajkerchum/poc/models/llama3/Meta-Llama-3-8B.fp16.gguf",
        "/home/ajkerchum/poc/models/llama3/Llama-3-8B-SCLP-Patched.gguf",
        "blk.0.ffn_up.weight"
    )
