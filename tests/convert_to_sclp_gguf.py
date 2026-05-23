#!/usr/bin/env python3
"""
Convert a BF16/F16/F32 GGUF to a compact SCLP GGUF in a single pass.

Linear projection tensors are encoded with SCLP and written at their actual
compressed size (compact format). All other tensors are copied verbatim.

Usage:
    python3 tests/convert_to_sclp_gguf.py --input model.bf16.gguf --output model.sclp.gguf
"""
import argparse
import struct
import sys
import os
import tempfile
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import _setup_paths  # noqa: F401

from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType
from gguf.constants import GGUFValueType, GGML_QUANT_SIZES
from compression.encoder import encode_palette, encode_palette_4b, encode_palette_6b
from compression.imatrix import load_imatrix, per_column_importance

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


# --- mixed-precision policy ------------------------------------------------
# Returned strings: 'sclp4' | 'sclp6' | 'sclp8' | 'none' (copy verbatim).
# Rationale per group:
#   - attn_*           SCLP6: small fraction of weights, errors compound through
#                      softmax/normalization; cheap to keep at higher precision.
#   - ffn_down*        SCLP6: feeds the residual stream directly (no following
#                      non-linearity), so reconstruction error survives downstream.
#   - ffn_gate/up*     SCLP4: feed the gated MLP non-linearity, errors are
#                      partially absorbed by SiLU/GeGLU; ~85% of total weights so
#                      this is where the size win is.
# Token-embd / output / norms / scales fall through to 'none' (copied verbatim).
_MIXED_POLICY = [
    # (suffix, precision) — first match wins
    ('attn_q.weight',           'sclp6'),
    ('attn_k.weight',           'sclp6'),
    ('attn_v.weight',           'sclp6'),
    ('attn_output.weight',      'sclp6'),
    ('ffn_down.weight',         'sclp6'),
    ('ffn_down_exps.weight',    'sclp6'),
    ('ffn_gate.weight',         'sclp4'),
    ('ffn_up.weight',           'sclp4'),
    ('ffn_gate_exps.weight',    'sclp4'),
    ('ffn_up_exps.weight',      'sclp4'),
    ('ffn_gate_up_exps.weight', 'sclp4'),
]

_MIXED_Q4K_POLICY = [
    # SCLP6 on attention + ffn_down; gate/up left as verbatim BF16 for
    # subsequent llama-quantize Q4_K pass.
    ('attn_q.weight',           'sclp6'),
    ('attn_k.weight',           'sclp6'),
    ('attn_v.weight',           'sclp6'),
    ('attn_output.weight',      'sclp6'),
    ('ffn_down.weight',         'sclp6'),
    ('ffn_down_exps.weight',    'sclp6'),
]


def precision_for_tensor(name: str, default_format: str) -> str:
    """Return the SCLP variant ('sclp4'|'sclp6'|'sclp8') to use for a tensor name,
    or 'none' if it should be copied verbatim.

    For uniform --format sclpN runs, every is_sclp_target tensor uses N and all
    others are copied. For --format mixed, attention + ffn_down go SCLP6 and
    ffn_gate/up go SCLP4 (token_embd, output, norms copied verbatim)."""
    if default_format == 'mixed':
        policy = _MIXED_POLICY
    elif default_format == 'mixed-q4k':
        policy = _MIXED_Q4K_POLICY
    else:
        return default_format if is_sclp_target(name) else 'none'
    for suffix, prec in policy:
        if name.endswith(suffix):
            return prec
    return 'none'


def _f32_to_bf16_rne(f32: np.ndarray) -> np.ndarray:
    """Convert F32 to BF16 using round-to-nearest-even (matches C++ float_to_bf16)."""
    i = f32.view(np.uint32)
    lsb = (i >> 16) & 1
    bias = np.uint32(0x7FFF) + lsb
    return ((i.astype(np.uint64) + bias.astype(np.uint64)) >> 16).astype(np.uint16)


def to_bf16_uint16(tensor_data: np.ndarray, tensor_type: GGMLQuantizationType) -> np.ndarray:
    """Return BF16 bit patterns as uint16, converting from any supported input dtype."""
    raw = tensor_data.flatten()
    if tensor_type == GGMLQuantizationType.BF16:
        return raw.view(np.uint16)
    if tensor_type == GGMLQuantizationType.F16:
        f32 = raw.view(np.float16).astype(np.float32)
        return _f32_to_bf16_rne(f32)
    if tensor_type == GGMLQuantizationType.F32:
        return _f32_to_bf16_rne(raw.view(np.float32))
    raise ValueError(f"Unsupported source type for SCLP encoding: {tensor_type}")


def build_sclp_blob(data_u16: np.ndarray, n_experts: int = 1) -> bytes:
    """Encode uint16 BF16 weights into a compact SCLP (8-bit) blob (compact format)."""
    encoded = encode_palette(data_u16, n_experts=n_experts)
    num_weights     = len(data_u16)
    ws              = encoded['ws_stream'].astype(np.uint8)
    sidecar_indices = encoded['sidecar']['indices'].astype(np.uint32)
    sidecar_values  = encoded['sidecar']['values'].astype(np.uint16)

    palettes = encoded['palette']
    if n_experts == 1:
        palette_header = struct.pack('<B', len(palettes)) + palettes.astype(np.uint8).tobytes()
    else:
        palette_header = b""
        for p in palettes:
            palette_header += struct.pack('<B', len(p)) + p.astype(np.uint8).tobytes()

    blob = (
        struct.pack('<II', num_weights, n_experts)
        + palette_header
        + ws.tobytes()
        + struct.pack('<I', len(sidecar_indices))
        + sidecar_indices.tobytes()
        + sidecar_values.tobytes()
    )
    return blob


def build_sclp4_blob(data_u16: np.ndarray, shape: list = None, sidecar_dist: int = 0,
                     importance: np.ndarray | None = None,
                     sidecar_imatrix_budget: float = 0.0) -> bytes:
    """Encode uint16 BF16 weights into a compact SCLP4 (4-bit) blob (compact format).

    For MoE tensors with shape [n_experts, N, K] (slowest-first after GGUF reversal),
    each expert gets its own palette.
    New header: [uint32 num_weights][uint32 n_experts][per-expert: uint8 palette_size, palette]
    """
    # shape is reversed GGUF order: [n_experts, N, K] (slowest to fastest)
    n_experts = shape[0] if (shape is not None and len(shape) >= 3) else 1
    # K = innermost (fastest-changing) dim — that's shape[-1] in slowest-first order.
    K = int(shape[-1]) if shape is not None and len(shape) >= 1 else None
    encoded = encode_palette_4b(data_u16, n_experts=n_experts, sidecar_dist=sidecar_dist,
                                 importance=importance, K=K if importance is not None else None,
                                 sidecar_imatrix_budget=sidecar_imatrix_budget)
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


def build_sclp6_blob(data_u16: np.ndarray, shape: list = None, sidecar_dist: int = 0,
                     importance: np.ndarray | None = None,
                     sidecar_imatrix_budget: float = 0.0) -> bytes:
    """Encode uint16 BF16 weights into a compact SCLP6 (6-bit) blob (compact format).

    For MoE tensors with shape [n_experts, N, K] (slowest-first after GGUF reversal),
    each expert gets its own palette.
    New header: [uint32 num_weights][uint32 n_experts][per-expert: uint8 palette_size, palette]
    """
    # shape is reversed GGUF order: [n_experts, N, K] (slowest to fastest)
    n_experts = shape[0] if (shape is not None and len(shape) >= 3) else 1
    K = int(shape[-1]) if shape is not None and len(shape) >= 1 else None
    encoded = encode_palette_6b(data_u16, n_experts=n_experts, sidecar_dist=sidecar_dist,
                                 importance=importance, K=K if importance is not None else None,
                                 sidecar_imatrix_budget=sidecar_imatrix_budget)
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


def _load_tensor_from_mmap(gguf_path: str, byte_offset: int, n_elements: int,
                           tensor_type_val: int) -> np.ndarray:
    """mmap a tensor directly from the GGUF file and convert to BF16 uint16."""
    tt = GGMLQuantizationType(tensor_type_val)
    if tt == GGMLQuantizationType.BF16:
        mm = np.memmap(gguf_path, dtype=np.uint16, mode='r',
                       offset=byte_offset, shape=(n_elements,))
        return np.array(mm)
    elif tt == GGMLQuantizationType.F16:
        mm = np.memmap(gguf_path, dtype=np.float16, mode='r',
                       offset=byte_offset, shape=(n_elements,))
        f32 = np.array(mm, dtype=np.float32)
        return (f32.view(np.uint32) >> 16).astype(np.uint16)
    elif tt == GGMLQuantizationType.F32:
        mm = np.memmap(gguf_path, dtype=np.float32, mode='r',
                       offset=byte_offset, shape=(n_elements,))
        return (np.array(mm).view(np.uint32) >> 16).astype(np.uint16)
    else:
        raise ValueError(f"Unsupported tensor type: {tt}")


def _encode_worker(prec: str, gguf_path: str, byte_offset: int, n_elements: int,
                   tensor_type_val: int, shape: list, sidecar_dist: int,
                   importance: np.ndarray | None, sidecar_imatrix_budget: float,
                   tmp_path: str, n_experts: int) -> tuple:
    """Worker: mmap tensor from GGUF, encode, write blob to tmp_path, return summary."""
    data_u16 = _load_tensor_from_mmap(gguf_path, byte_offset, n_elements, tensor_type_val)
    if prec == 'sclp4':
        blob = build_sclp4_blob(data_u16, shape=shape, sidecar_dist=sidecar_dist,
                                importance=importance,
                                sidecar_imatrix_budget=sidecar_imatrix_budget)
        ws_len = (len(data_u16) + 1) // 2
    elif prec == 'sclp6':
        blob = build_sclp6_blob(data_u16, shape=shape, sidecar_dist=sidecar_dist,
                                importance=importance,
                                sidecar_imatrix_budget=sidecar_imatrix_budget)
        ws_len = ((len(data_u16) + 3) // 4) * 3
    else:  # sclp8
        blob = build_sclp_blob(data_u16, n_experts=n_experts)
        if len(blob) % 2 != 0:
            blob += b'\x00'
        ws_len = len(data_u16)

    with open(tmp_path, 'wb') as f:
        f.write(blob)

    # Compute offsets from blob bytes
    n_exp = struct.unpack_from('<I', blob, 4)[0]
    pos = 8
    for _ in range(n_exp):
        pos += 1 + blob[pos]
    ws_offset = pos
    sc_count = struct.unpack_from('<I', blob, ws_offset + ws_len)[0]
    return (len(blob), sc_count, ws_offset, ws_len)


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
    parser.add_argument('--format', choices=['sclp8', 'sclp6', 'sclp4', 'mixed', 'mixed-q4k'], default='sclp8',
                        help='Output quantization format. sclp8/6/4: uniform N-bit on every linear projection. '
                             'mixed: per-tensor policy (attn_*+ffn_down* → SCLP6, ffn_gate/up* → SCLP4). '
                             'mixed-q4k: SCLP6 on attn+ffn_down (from --bf16-source), Q4_K gate/up copied verbatim from --input.')
    parser.add_argument('--bf16-source', nargs='+', default=None, metavar='PATH',
                        help='BF16/F16 GGUF shard(s) to read unquantized weights from. '
                             'Required for --format mixed-q4k when --input is a quantized GGUF: '
                             'tensors targeted for SCLP encoding are read from here.')
    parser.add_argument('--sidecar-dist', type=int, default=0, metavar='D',
                        help='SCLP4/6: store weights verbatim in sidecar if nearest palette exponent distance > D. '
                             '0=off (default). 1 recommended: fixes ~0.1%% of worst-case errors at <0.3%% size overhead.')
    parser.add_argument('--imatrix', default=None, metavar='PATH',
                        help='Optional path to a llama.cpp legacy imatrix .dat file. When provided, per-tensor '
                             'activation importance can rescue extra high-impact weights into the sidecar — '
                             'controlled by --sidecar-imatrix-budget. (Imatrix is no longer applied to k-means '
                             'palette selection; that variant regressed PPL — see CLAUDE.md.)')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) // 2),
                        metavar='N',
                        help='Parallel encoding workers (default: cpu_count/2). '
                             'Each worker holds one tensor in memory; tune down if RAM-bound.')
    parser.add_argument('--sidecar-imatrix-budget', type=float, default=0.0, metavar='FRAC',
                        help='When --imatrix is set, sidecar an additional FRAC fraction of each tensor\'s '
                             'weights ranked by importance × exponent_distance_to_palette. 0=off (default). '
                             '0.005-0.02 is the range to try; higher inflates file size, lower limits the '
                             'quality lift.')
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input[0])
        # Strip trailing shard suffix if present (e.g. -00001-of-00002)
        import re
        base = re.sub(r'-\d{5}-of-\d{5}$', '', base)
        args.output = base + '-SCLP' + ext

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    if args.imatrix:
        print(f"Imatrix: {args.imatrix}")
        imatrix_data = load_imatrix(args.imatrix)
        print(f"  loaded {len(imatrix_data)} imatrix entries")
    else:
        imatrix_data = None

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
    per_format_counts = {'sclp4': 0, 'sclp6': 0, 'sclp8': 0, 'none': 0}

    # Tensor types we can encode (we read BF16 bit patterns; F16/F32 round through f32→bf16).
    encodable_types = {GGMLQuantizationType.BF16, GGMLQuantizationType.F16, GGMLQuantizationType.F32}

    # When --bf16-source is provided (for mixed-q4k), build a lookup so SCLP-targeted
    # tensors can be encoded from unquantized weights even when --input is quantized.
    bf16_lookup = {}  # tensor_name -> (gguf_path, data_offset, n_elements, tensor_type_val)
    if args.bf16_source:
        print(f"BF16 source: {args.bf16_source}")
        bf16_readers = [GGUFReader(p, mode='r') for p in args.bf16_source]
        for br, bp in zip(bf16_readers, args.bf16_source):
            for bt in br.tensors:
                bf16_lookup[bt.name] = (bp, bt.data_offset, int(np.prod(bt.shape)), int(bt.tensor_type))
        print(f"  indexed {len(bf16_lookup)} tensors from BF16 source")
        del bf16_readers  # close mmaps; workers will re-mmap individual tensors

    all_tensors = []
    tensor_gguf_path = []  # which file each tensor lives in
    for r, path in zip(readers, args.input):
        for t in r.tensors:
            all_tensors.append(t)
            tensor_gguf_path.append(path)

    # Precompute per-tensor plan so workers can be dispatched without re-deriving policy.
    plans = []  # list of dicts: name, prec, shape, importance, encodable
    for t in all_tensors:
        shape = list(reversed(t.shape.tolist()))
        prec  = precision_for_tensor(t.name, args.format)
        if prec != 'none' and t.tensor_type not in encodable_types:
            if t.name in bf16_lookup:
                pass  # will encode from BF16 source
            else:
                print(f"  passthru: {t.name} is {t.tensor_type.name} (not BF16/F16/F32) — copying verbatim")
                prec = 'none'

        n_experts = 1
        if any(x in t.name for x in ['.ffn_gate_exps.', '.ffn_up_exps.', '.ffn_down_exps.', '.ffn_gate_up_exps.']):
            # For expert tensors, shape is [n_experts, rows, cols]
            if len(shape) == 3:
                n_experts = shape[0]
        
        importance = None
        if prec != 'none' and imatrix_data is not None:
            entry = imatrix_data.get(t.name)
            if entry is not None and entry['nval'] > 0:
                K = int(shape[-1])
                try:
                    importance = per_column_importance(entry, K)
                except ValueError as e:
                    print(f"  ! imatrix shape mismatch for {t.name}: {e} — skipping importance")
                    importance = None
        plans.append({'shape': shape, 'prec': prec, 'importance': importance, 'n_experts': n_experts})
        per_format_counts[prec] += 1

    n_sclp = sum(1 for p in plans if p['prec'] != 'none')
    workers = max(1, min(args.workers, n_sclp)) if n_sclp else 1
    print(f"Parallel encode: {workers} worker(s) for {n_sclp} SCLP tensors")

    # Worker tempdir lives next to the output so blob mmap reads stay on the same filesystem.
    output_dir = os.path.dirname(os.path.abspath(args.output)) or '.'
    tmpdir_ctx = tempfile.TemporaryDirectory(prefix='sclp_blobs_', dir=output_dir)
    tmpdir = tmpdir_ctx.name

    # Bounded sliding window: at most `lookahead` SCLP futures in flight at any time.
    # Keeps peak memory ~ lookahead × (input + blob) per worker.
    lookahead = workers * 2
    futures = {}  # tensor_idx -> Future

    ctx = mp.get_context('spawn')  # avoid fork(): GGUF mmaps + reader state aren't fork-safe
    ex = ProcessPoolExecutor(max_workers=workers, mp_context=ctx) if n_sclp else None

    def submit_idx(i: int):
        t = all_tensors[i]
        p = plans[i]
        if p['prec'] == 'none':
            return
        # Use BF16 source when the primary input isn't encodable (e.g. Q4_K_M input)
        if t.tensor_type not in encodable_types and t.name in bf16_lookup:
            gguf_path, byte_offset, n_elements, tensor_type_val = bf16_lookup[t.name]
        else:
            gguf_path = tensor_gguf_path[i]
            n_elements = int(np.prod(t.shape))
            byte_offset = t.data_offset
            tensor_type_val = int(t.tensor_type)
        tmp_path = os.path.join(tmpdir, f"{i:05d}.blob")
        futures[i] = ex.submit(
            _encode_worker, p['prec'], gguf_path, byte_offset, n_elements,
            tensor_type_val, p['shape'],
            args.sidecar_dist, p['importance'], args.sidecar_imatrix_budget, tmp_path, p['n_experts']
        )

    try:
        # Prime the pipeline
        for i in range(min(lookahead, len(all_tensors))):
            submit_idx(i)

        for i, t in enumerate(all_tensors):
            p = plans[i]
            prec = p['prec']
            shape = p['shape']

            if prec != 'none':
                blob_size, sc_count, ws_offset, ws_len = futures.pop(i).result()
                tmp_path = os.path.join(tmpdir, f"{i:05d}.blob")
                if prec == 'sclp4':
                    gguf_dtype = GGMLQuantizationType.SCLP4
                    blob_dtype = np.int8
                elif prec == 'sclp6':
                    gguf_dtype = GGMLQuantizationType.SCLP6
                    blob_dtype = np.int8
                else:
                    gguf_dtype = GGMLQuantizationType.SCLP
                    blob_dtype = np.uint16

                # mmap the worker's blob file. Copy to detach from the file so the
                # tempdir can be cleaned up at the end; writer.add_tensor keeps refs.
                if blob_dtype == np.uint16:
                    assert blob_size % 2 == 0, "sclp8 blob must be uint16-aligned"
                    np_blob = np.memmap(tmp_path, dtype=np.uint16, mode='r',
                                        shape=(blob_size // 2,))
                else:
                    np_blob = np.memmap(tmp_path, dtype=np.int8, mode='r',
                                        shape=(blob_size,))
                np_blob = np.array(np_blob)  # detach from mmap (free to unlink tmp file)
                os.unlink(tmp_path)

                writer.add_tensor(t.name, np_blob,
                                  raw_shape=shape,
                                  raw_dtype=gguf_dtype)

                ratio = t.n_bytes / blob_size
                total_original += t.n_bytes
                total_compact  += blob_size
                total_sidecar  += sc_count
                n_compressed   += 1
                sc_note = f" ({sc_count} sidecar)" if sc_count else ""
                print(f"  {prec.upper():>5} [{n_compressed:3d}] {t.name}: {ratio:.3f}x{sc_note}")
            else:
                # GGUFWriter expects shape semantics that differ by element type:
                #   - F-types and BF16-as-uint16: pass element shape (fastest-first).
                #   - K-quants (uint8 byte arrays): pass byte shape; writer recovers
                #     element shape internally via quant_shape_from_byte_shape.
                raw = t.data
                if raw.dtype == np.uint8:
                    block_size, type_size = GGML_QUANT_SIZES[t.tensor_type]
                    if type_size == 2 and block_size == 1:
                        raw = raw.view(np.uint16)
                        writer.add_tensor(t.name, raw.copy(),
                                          raw_shape=shape,
                                          raw_dtype=t.tensor_type)
                    else:
                        writer.add_tensor(t.name, raw.copy(),
                                          raw_shape=list(raw.shape),
                                          raw_dtype=t.tensor_type)
                else:
                    writer.add_tensor(t.name, raw.copy(),
                                      raw_shape=shape,
                                      raw_dtype=t.tensor_type)

            # Slide the window: submit the tensor `lookahead` ahead of the just-consumed one.
            nxt = i + lookahead
            if nxt < len(all_tensors):
                submit_idx(nxt)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
    finally:
        if ex is not None:
            ex.shutdown(wait=True)
        tmpdir_ctx.cleanup()

    if total_original > 0:
        orig_gb = total_original / (1024**3)
        comp_gb = total_compact  / (1024**3)
        total_weights = total_original // 2
        sc_pct = 100.0 * total_sidecar / total_weights if total_weights else 0
        print(f"\n{n_compressed} tensors compressed:")
        if args.format == 'mixed':
            print(f"  per-format counts: SCLP6={per_format_counts['sclp6']} "
                  f"SCLP4={per_format_counts['sclp4']} "
                  f"verbatim={per_format_counts['none']}")
        print(f"  SCLP data: {orig_gb:.2f} GB → {comp_gb:.2f} GB ({orig_gb - comp_gb:.2f} GB saved, {total_original/total_compact:.3f}x)")
        print(f"  Sidecar:   {total_sidecar:,} weights ({sc_pct:.4f}%)")

    input_size  = sum(os.path.getsize(p) for p in args.input) / (1024**3)
    output_size = os.path.getsize(args.output) / (1024**3)
    print(f"File:   {input_size:.2f} GB → {output_size:.2f} GB (saved {input_size - output_size:.2f} GB)")
    print(f"Written: {args.output}")


if __name__ == '__main__':
    main()
