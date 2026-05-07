
import sys
import os
import numpy as np

# Mocking parts of gguf-py structure for local testing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _setup_paths  # noqa: F401
from gguf import GGUFWriter, GGMLQuantizationType

def test_sclp_gguf_writing():
    # SCLP data structure
    # A single SCLP tensor will contain the combined metadata and data blobs
    # We must treat this as a raw byte array in GGUF
    
    filename = "test_sclp.gguf"
    writer = GGUFWriter(filename, "test_arch")
    
    # 1. Define SCLP metadata
    # We pack palette + indices + SM into a single byte stream
    palette = np.arange(16, dtype=np.uint8)
    packed_indices = np.zeros(100, dtype=np.uint8)
    sm = np.zeros(200, dtype=np.uint8)
    
    # Concatenated payload
    payload = np.concatenate([palette, packed_indices, sm])
    
    # Register tensor with our custom type
    # Use SCLP type we added to constants.py
    writer.add_tensor("test_sclp_tensor", payload, raw_dtype=GGMLQuantizationType.SCLP)
    
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    print(f"Successfully wrote {filename} with SCLP tensor.")

if __name__ == "__main__":
    test_sclp_gguf_writing()
