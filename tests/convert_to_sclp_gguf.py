#!/usr/bin/env python3
"""
Convert a BF16/F16/F32 GGUF to a compact SCLP GGUF in a single pass.

Linear projection tensors are encoded with SCLP and written at their actual
compressed size (no zero-padding). All other tensors are copied verbatim.

Usage:
    python3 tests/convert_to_sclp_gguf.py --input model.bf16.gguf --output model.sclp.gguf
"""
import argparse
import struct
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import _setup_paths  # noqa: F401

from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType
from gguf.constants import GGUFValueType
from compression.encoder import encode_palette

SCLP_TARGET_SUFFIXES = [
    'attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'attn_output.weight',
    'ffn_gate.weight', 'ffn_up.weight', 'ffn_down.weight',
    # MoE stacked expert tensors (Gemma 4, DeepSeek, etc.)
    'ffn_gate_exps.weight', 'ffn_up_exps.weight', 'ffn_down_exps.weight',
]


def is_sclp_target(name: str) -> bool:
    return any(name.endswith(s) for s in SCLP_TARGET_SUFFIXES)


def to_bf16_uint16(tensor_data: np.ndarray, tensor_type: GGMLQuantizationType) -> np.ndarray:
    """Return BF16 bit patterns as uint16, converting from any supported input dtype."""
    raw = tensor_data.flatten()
    if tensor_type == GGMLQuantizationType.BF16:
        return raw.view(np.uint16)
    if tensor_type == GGMLQuantizationType.F16:
        f32 = raw.view(np.float16).astype(np.float32)
        return (f32.view(np.uint32) >> 16).astype(np.uint16)
    if tensor_type == GGMLQuantizationType.F32:
        return (raw.view(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    raise ValueError(f"Unsupported source type for SCLP encoding: {tensor_type}")


def build_sclp_blob(data_u16: np.ndarray) -> bytes:
    """Encode uint16 BF16 weights into a compact SCLP blob (no zero-padding)."""
    encoded = encode_palette(data_u16)
    num_weights     = len(data_u16)
    palette         = encoded['palette'].astype(np.uint8)
    ws              = encoded['ws_stream'].astype(np.uint8)
    sidecar_indices = encoded['sidecar']['indices'].astype(np.uint32)
    sidecar_values  = encoded['sidecar']['values'].astype(np.uint16)

    blob = (
        struct.pack('<IB', num_weights, len(palette))
        + palette.tobytes()
        + ws.tobytes()
        + struct.pack('<I', len(sidecar_indices))
        + sidecar_indices.tobytes()
        + sidecar_values.tobytes()
    )
    return blob


def copy_kv(writer: GGUFWriter, reader: GGUFReader) -> None:
    for key, field in reader.fields.items():
        if key.startswith('GGUF.'):
            continue
        if key == 'general.architecture':
            continue  # GGUFWriter adds this via constructor
        main_type = field.types[0]
        if main_type == GGUFValueType.ARRAY:
            sub_type = field.types[-1]
            writer.add_key_value(key, field.contents(), main_type, sub_type)
        else:
            writer.add_key_value(key, field.contents(), main_type)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True, help='Input BF16/F16 GGUF file')
    parser.add_argument('--output', default=None,  help='Output compact SCLP GGUF file')
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = base + '-SCLP' + ext

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")

    reader = GGUFReader(args.input, mode='r')

    arch_field = reader.fields.get('general.architecture')
    arch = arch_field.contents() if arch_field else 'llama'

    writer = GGUFWriter(args.output, arch=arch)
    copy_kv(writer, reader)

    total_original = 0
    total_compact   = 0
    n_compressed    = 0
    total_sidecar   = 0

    for t in reader.tensors:
        shape = list(reversed(t.shape.tolist()))  # GGUF stores fastest-dim first

        if is_sclp_target(t.name):
            data = to_bf16_uint16(t.data, t.tensor_type)
            blob = build_sclp_blob(data)

            # Pad to even byte count for uint16 view
            if len(blob) % 2 != 0:
                blob += b'\x00'
            np_blob = np.frombuffer(blob, dtype=np.uint16).copy()

            writer.add_tensor(t.name, np_blob,
                              raw_shape=shape,
                              raw_dtype=GGMLQuantizationType.SCLP)

            sc_count = struct.unpack_from('<I', blob, 5 + blob[4] + len(data))[0]
            ratio = t.n_bytes / len(blob)
            total_original += t.n_bytes
            total_compact  += len(blob)
            total_sidecar  += sc_count
            n_compressed   += 1
            sc_note = f" ({sc_count} sidecar)" if sc_count else ""
            print(f"  SCLP [{n_compressed:3d}] {t.name}: {ratio:.3f}x{sc_note}")
        else:
            np_data = t.data.copy()
            writer.add_tensor(t.name, np_data,
                              raw_shape=shape,
                              raw_dtype=t.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    if total_original > 0:
        orig_gb = total_original / (1024**3)
        comp_gb = total_compact  / (1024**3)
        total_weights = total_original // 2
        sc_pct = 100.0 * total_sidecar / total_weights if total_weights else 0
        print(f"\n{n_compressed} tensors compressed:")
        print(f"  SCLP data: {orig_gb:.2f} GB → {comp_gb:.2f} GB ({orig_gb - comp_gb:.2f} GB saved, {total_original/total_compact:.3f}x)")
        print(f"  Sidecar:   {total_sidecar:,} weights ({sc_pct:.4f}%)")

    input_size  = os.path.getsize(args.input)  / (1024**3)
    output_size = os.path.getsize(args.output) / (1024**3)
    print(f"File:   {input_size:.2f} GB → {output_size:.2f} GB (saved {input_size - output_size:.2f} GB)")
    print(f"Written: {args.output}")


if __name__ == '__main__':
    main()
