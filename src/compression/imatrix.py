"""Read llama.cpp legacy imatrix .dat files.

The .dat format (see llama.cpp tools/imatrix/imatrix.cpp::save_imatrix_legacy):

    int32 n_entries
    for each entry:
        int32 name_len
        utf8  name[name_len]
        int32 ncall           # forward passes that touched this tensor
        int32 nval            # total number of stored floats
        float values[nval]    # normalized per-column activation magnitude
                              # values[i] = (sum_sq[i] / counts[expert_of(i)]) * ncall
                              # layout for MoE: [e0_k0..e0_kK-1, e1_k0..e1_kK-1, ...]
                              # layout for dense: [k0..kK-1] (single "expert")
    int32 m_last_chunk
    int32 dataset_filename_len
    utf8  dataset_filename[dataset_filename_len]

We expose:
    load_imatrix(path) -> dict
        {
            tensor_name: {
                'values':    np.float32[n_experts, K],   # E[x²] per (expert, input column)
                'ncall':     int,                          # forward passes
                'n_experts': int,
                'K':         int,
            },
            ...
        }
"""
from __future__ import annotations
import struct
from pathlib import Path
import numpy as np


def load_imatrix(path: str | Path) -> dict:
    p = Path(path)
    raw = p.read_bytes()
    off = 0

    def read(fmt):
        nonlocal off
        sz = struct.calcsize(fmt)
        out = struct.unpack_from(fmt, raw, off)
        off += sz
        return out

    (n_entries,) = read('<i')
    entries: dict = {}

    for _ in range(n_entries):
        (name_len,) = read('<i')
        name = raw[off:off + name_len].decode('utf-8')
        off += name_len

        (ncall, nval) = read('<ii')

        if nval == 0:
            entries[name] = {'values': np.zeros((0, 0), dtype=np.float32),
                             'ncall': ncall, 'n_experts': 0, 'K': 0}
            continue

        values = np.frombuffer(raw, dtype=np.float32, count=nval, offset=off).copy()
        off += nval * 4

        # llama.cpp normalises by `nmat = counts.size()`; we don't have nmat in the file
        # directly, but for MoE tensors nval is a multiple of K and nmat=n_experts.
        # We infer it later when we know K for a given tensor (caller passes the input dim).
        entries[name] = {
            'values':    values,           # flat float32[nval]
            'ncall':     ncall,
            'nval':      nval,
        }

    return entries


def per_column_importance(entry: dict, K: int) -> np.ndarray:
    """Return shape [n_experts, K] activation importance for a tensor.

    K is the input dimension (ne[0] / fastest-changing). For dense tensors n_experts=1."""
    values = entry['values']
    if values.size == 0:
        return np.zeros((1, K), dtype=np.float32)
    nval = values.size
    if nval % K != 0:
        raise ValueError(f"imatrix nval={nval} not divisible by K={K} for entry")
    n_experts = nval // K
    return values.reshape(n_experts, K)
