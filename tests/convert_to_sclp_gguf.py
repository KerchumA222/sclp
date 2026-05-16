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
from gguf.constants import GGUFValueType, GGML_QUANT_SIZES
from compression.encoder import encode_palette, encode_palette_4b, encode_palette_6b

SCLP_TARGET_SUFFIXES = [
    'attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'attn_output.weight',
    'ffn_gate.weight', 'ffn_up.weight', 'ffn_down.weight',
    # MoE stacked expert tensors (separate gate/up, e.g. DeepSeek)
    'ffn_gate_exps.weight', 'ffn_up_exps.weight', 'ffn_down_exps.weight',
    # MoE fused gate+up experts (Gemma 4)
    'ffn_gate_up_exps.weight',
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
    """Encode uint16 BF16 weights into a compact SCLP (8-bit) blob (no zero-padding)."""
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


def build_sclp4_blob(data_u16: np.ndarray, shape: list = None) -> bytes:
    """Encode uint16 BF16 weights into a compact SCLP4 (4-bit) blob (no zero-padding).

    For MoE tensors with shape [n_experts, N, K] (slowest-first after GGUF reversal),
    each expert gets its own palette.
    New header: [uint32 num_weights][uint32 n_experts][per-expert: uint8 palette_size, palette]
    """
    # shape is reversed GGUF order: [n_experts, N, K] (slowest to fastest)
    n_experts = shape[0] if (shape is not None and len(shape) >= 3) else 1
    encoded = encode_palette_4b(data_u16, n_experts=n_experts)
    num_weights = len(data_u16)
    ws = encoded['ws_stream'].astype(np.uint8)
    sidecar_indices = encoded['sidecar']['indices'].astype(np.uint32)
    sidecar_values  = encoded['sidecar']['values'].astype(np.uint16)

    # Build per-expert palette header section
    palettes = encoded['palette'] if n_experts > 1 else [encoded['palette']]
    palette_header = b''
    for pal in palettes:
        pal = pal.astype(np.uint8)
        palette_header += bytes([len(pal)]) + pal.tobytes()

    blob = (
        struct.pack('<II', num_weights, n_experts)
        + palette_header
        + ws.tobytes()
        + struct.pack('<I', len(sidecar_indices))
        + sidecar_indices.tobytes()
        + sidecar_values.tobytes()
    )
    return blob


def build_sclp6_blob(data_u16: np.ndarray, shape: list = None) -> bytes:
    """Encode uint16 BF16 weights into a compact SCLP6 (6-bit) blob (no zero-padding).

    For MoE tensors with shape [n_experts, N, K] (slowest-first after GGUF reversal),
    each expert gets its own palette.
    New header: [uint32 num_weights][uint32 n_experts][per-expert: uint8 palette_size, palette]
    """
    # shape is reversed GGUF order: [n_experts, N, K] (slowest to fastest)
    n_experts = shape[0] if (shape is not None and len(shape) >= 3) else 1
    encoded = encode_palette_6b(data_u16, n_experts=n_experts)
    num_weights = len(data_u16)
    ws = encoded['ws_stream'].astype(np.uint8)
    sidecar_indices = encoded['sidecar']['indices'].astype(np.uint32)
    sidecar_values  = encoded['sidecar']['values'].astype(np.uint16)

    # Build per-expert palette header section
    palettes = encoded['palette'] if n_experts > 1 else [encoded['palette']]
    palette_header = b''
    for pal in palettes:
        pal = pal.astype(np.uint8)
        palette_header += bytes([len(pal)]) + pal.tobytes()

    blob = (
        struct.pack('<II', num_weights, n_experts)
        + palette_header
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
        if key.startswith('split.'):
            continue  # strip split-shard metadata — output is a single file
        main_type = field.types[0]
        if main_type == GGUFValueType.ARRAY:
            sub_type = field.types[-1]
            writer.add_key_value(key, field.contents(), main_type, sub_type)
        else:
            writer.add_key_value(key, field.contents(), main_type)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True, nargs='+',
                        help='Input BF16/F16 GGUF file(s). For split GGUFs, pass all shards in order.')
    parser.add_argument('--output', default=None,  help='Output compact SCLP GGUF file')
    parser.add_argument('--format', choices=['sclp8', 'sclp6', 'sclp4'], default='sclp8',
                        help='Output quantization format: sclp8 (default, 8 bits/weight), sclp6 (6 bits/weight), or sclp4 (4 bits/weight)')
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input[0])
        # Strip trailing shard suffix if present (e.g. -00001-of-00002)
        import re
        base = re.sub(r'-\d{5}-of-\d{5}$', '', base)
        args.output = base + '-SCLP' + ext

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")

    readers = [GGUFReader(p, mode='r') for p in args.input]
    reader  = readers[0]  # KV metadata comes from shard 0

    arch_field = reader.fields.get('general.architecture')
    arch = arch_field.contents() if arch_field else 'llama'

    writer = GGUFWriter(args.output, arch=arch)
    copy_kv(writer, reader)

    total_original = 0
    total_compact   = 0
    n_compressed    = 0
    total_sidecar   = 0

    all_tensors = [t for r in readers for t in r.tensors]
    for t in all_tensors:
        shape = list(reversed(t.shape.tolist()))  # GGUF stores fastest-dim first

        if is_sclp_target(t.name):
            data = to_bf16_uint16(t.data, t.tensor_type)
            if args.format == 'sclp4':
                blob = build_sclp4_blob(data, shape=shape)
                gguf_dtype = GGMLQuantizationType.SCLP4
                # Use int8 (not uint8) to bypass GGUFWriter's quant_shape_from_byte_shape
                # which would incorrectly double the last dimension for blck_size=2 types.
                np_blob = np.frombuffer(blob, dtype=np.int8).copy()
            elif args.format == 'sclp6':
                blob = build_sclp6_blob(data, shape=shape)
                gguf_dtype = GGMLQuantizationType.SCLP6
                # Use int8 (not uint8) to bypass GGUFWriter's quant_shape_from_byte_shape
                # which would incorrectly transform shape for blck_size=4 types.
                np_blob = np.frombuffer(blob, dtype=np.int8).copy()
            else:
                blob = build_sclp_blob(data)
                gguf_dtype = GGMLQuantizationType.SCLP
                # Pad to even byte count for uint16 view
                if len(blob) % 2 != 0:
                    blob += b'\x00'
                np_blob = np.frombuffer(blob, dtype=np.uint16).copy()

            writer.add_tensor(t.name, np_blob,
                              raw_shape=shape,
                              raw_dtype=gguf_dtype)

            # Compute ws_offset from the new multi-palette header
            n_exp = struct.unpack_from('<I', blob, 4)[0]
            pos = 8
            for _ in range(n_exp):
                pos += 1 + blob[pos]  # skip palette_size + palette bytes
            ws_offset = pos
            if args.format == 'sclp4':
                ws_len = (len(data) + 1) // 2
            elif args.format == 'sclp6':
                ws_len = ((len(data) + 3) // 4) * 3
            else:
                ws_len = len(data)
            sc_count = struct.unpack_from('<I', blob, ws_offset + ws_len)[0]
            ratio = t.n_bytes / len(blob)
            total_original += t.n_bytes
            total_compact  += len(blob)
            total_sidecar  += sc_count
            n_compressed   += 1
            sc_note = f" ({sc_count} sidecar)" if sc_count else ""
            fmt_label = args.format.upper()
            print(f"  {fmt_label} [{n_compressed:3d}] {t.name}: {ratio:.3f}x{sc_note}")
        else:
            # GGUFWriter applies quant_shape_from_byte_shape when tensor.dtype==uint8,
            # which would halve the shape. View as the native element type to avoid that.
            raw = t.data
            if raw.dtype == np.uint8:
                block_size, type_size = GGML_QUANT_SIZES[t.tensor_type]
                if type_size == 2 and block_size == 1:
                    raw = raw.view(np.uint16)
                # Other quantized types (Q4, Q8 etc.) should not appear in a BF16 source GGUF.
            writer.add_tensor(t.name, raw.copy(),
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

    input_size  = sum(os.path.getsize(p) for p in args.input) / (1024**3)
    output_size = os.path.getsize(args.output) / (1024**3)
    print(f"File:   {input_size:.2f} GB → {output_size:.2f} GB (saved {input_size - output_size:.2f} GB)")
    print(f"Written: {args.output}")


if __name__ == '__main__':
    main()
