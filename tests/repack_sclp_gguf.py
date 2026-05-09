#!/usr/bin/env python3
"""
Repack an existing padded SCLP GGUF into a compact GGUF where each SCLP
tensor blob is stored at its actual compressed size (no trailing zero padding).

All non-SCLP tensors are copied verbatim.

Usage:
    python3 tests/repack_sclp_gguf.py [--input PATH] [--output PATH]
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


def sclp_blob_actual_size(data: np.ndarray) -> int:
    """Return the number of meaningful bytes in a padded SCLP blob."""
    blob = bytes(data)
    num_w,    = struct.unpack_from('<I', blob, 0)
    pal_size  = blob[4]
    packed_off = 5 + pal_size
    sm_off    = packed_off + (num_w + 1) // 2
    sc_off    = sm_off + (num_w + 1) // 2  # SM is nibble-packed: ceil(N/2) bytes
    sc_count, = struct.unpack_from('<I', blob, sc_off)
    idx_off   = sc_off + 4
    val_off   = idx_off + sc_count * 4
    end       = val_off + sc_count * 2
    return end


def copy_kv(writer: GGUFWriter, reader: GGUFReader) -> None:
    """Copy all key-value metadata from reader to writer."""
    for key, field in reader.fields.items():
        if key.startswith('GGUF.'):
            continue  # internal fields (version, counts)
        if key == 'general.architecture':
            continue  # GGUFWriter already added this via constructor

        main_type = field.types[0]

        if main_type == GGUFValueType.ARRAY:
            sub_type = field.types[-1]
            vals = field.contents()
            writer.add_key_value(key, vals, main_type, sub_type)
        else:
            val = field.contents()
            writer.add_key_value(key, val, main_type)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True, help='Input padded SCLP GGUF file')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = base + '-Compact' + ext

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")

    reader = GGUFReader(args.input, mode='r')

    # Determine architecture for GGUFWriter constructor
    arch_field = reader.fields.get('general.architecture')
    arch = arch_field.contents() if arch_field else 'llama'

    writer = GGUFWriter(args.output, arch=arch)
    copy_kv(writer, reader)

    total_original = 0
    total_compact   = 0

    for t in reader.tensors:
        # GGUF stores shape in reverse order (fastest-changing dim first)
        shape = list(reversed(t.shape.tolist()))

        if t.tensor_type == GGMLQuantizationType.SCLP:
            actual = sclp_blob_actual_size(t.data)
            blob = bytes(t.data)[:actual]
            # Pad to even byte count for uint16 view
            if len(blob) % 2 != 0:
                blob += b'\x00'
            np_blob = np.frombuffer(blob, dtype=np.uint16).copy()
            # Pass weight shape via raw_shape; uint16 dtype bypasses quant_shape_from_byte_shape
            writer.add_tensor(t.name, np_blob,
                              raw_shape=shape,
                              raw_dtype=GGMLQuantizationType.SCLP)
            total_original += t.n_bytes
            total_compact  += len(blob)
            ratio = t.n_bytes / len(blob)
            print(f"  SCLP {t.name}: {t.n_bytes} → {len(blob)} bytes ({ratio:.3f}x)")
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
        savings = (total_original - total_compact) / (1024**3)
        orig_gb = total_original / (1024**3)
        comp_gb = total_compact / (1024**3)
        print(f"\nSCLP tensor savings: {savings:.2f} GB  ({orig_gb:.2f} GB → {comp_gb:.2f} GB)")

    input_size  = os.path.getsize(args.input)  / (1024**3)
    output_size = os.path.getsize(args.output) / (1024**3)
    print(f"File size: {input_size:.2f} GB → {output_size:.2f} GB  "
          f"(saved {input_size - output_size:.2f} GB)")
    print(f"Written: {args.output}")


if __name__ == '__main__':
    main()
