"""
Shared sys.path setup for SCLP scripts and tests.

Adds to sys.path (idempotent):
  - <repo>/src              — Python reference implementation
  - <repo>/python_pkg       — compiled HIP testmodule (if present)
  - <llama.cpp>/gguf-py     — gguf read/write library

Resolution order for llama.cpp directory:
  1. LLAMA_CPP_DIR environment variable
  2. Sibling directory ../llama.cpp relative to the repo root
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _add(p: Path) -> None:
    s = str(p)
    if p.exists() and s not in sys.path:
        sys.path.insert(0, s)


# src/ — Python reference implementation
_add(REPO_ROOT / 'src')

# python_pkg/ — compiled HIP module (optional)
_add(REPO_ROOT / 'python_pkg')

# gguf-py — from llama.cpp fork
_llama_cpp_dir = os.environ.get('LLAMA_CPP_DIR')
if _llama_cpp_dir:
    _gguf_py = Path(_llama_cpp_dir) / 'gguf-py'
else:
    _gguf_py = REPO_ROOT.parent / 'llama.cpp' / 'gguf-py'

_add(_gguf_py)
