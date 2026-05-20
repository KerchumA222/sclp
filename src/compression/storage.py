import numpy as np
import struct
import os

class CompressedTensorStorage:
    """
    Binary storage format for compressed BF16 tensors.

    VERSION 3 layout (Interleaved):
      Magic (4B)  Version uint16  NumWeights uint32
      PaletteSize uint8   Palette[N] uint8
      WSStreamLen uint32  WSStream[NumWeights] uint8
      SidecarCount uint32
      SidecarIndices[SidecarCount] uint32
      SidecarValues[SidecarCount]  uint16
    """
    
    MAGIC = b'SCLP'
    VERSION = 3

    def __init__(self, filepath: str):
        self.filepath = filepath

    def save(self, encoded_data: dict, num_weights: int):
        palette = encoded_data['palette'].astype(np.uint8)
        ws_stream = encoded_data['ws_stream'].astype(np.uint8)
        sidecar = encoded_data.get('sidecar', {'indices': np.array([], dtype=np.uint32),
                                                'values':  np.array([], dtype=np.uint16)})

        with open(self.filepath, 'wb') as f:
            f.write(self.MAGIC)
            f.write(struct.pack('<H', self.VERSION))
            f.write(struct.pack('<I', num_weights))
            f.write(struct.pack('<B', len(palette)))
            f.write(palette.tobytes())
            f.write(struct.pack('<I', len(ws_stream)))
            f.write(ws_stream.tobytes())
            
            sc_indices = sidecar['indices'].astype(np.uint32)
            sc_values  = sidecar['values'].astype(np.uint16)
            f.write(struct.pack('<I', len(sc_indices)))
            f.write(sc_indices.tobytes())
            f.write(sc_values.tobytes())

    def load(self) -> tuple:
        with open(self.filepath, 'rb') as f:
            magic = f.read(4)
            if magic != self.MAGIC:
                raise ValueError("Invalid file format")
            
            version = struct.unpack('<H', f.read(2))[0]
            num_weights = struct.unpack('<I', f.read(4))[0]
            palette_size = struct.unpack('<B', f.read(1))[0]

            palette = np.frombuffer(f.read(palette_size), dtype=np.uint8).copy()

            if version == 3:
                ws_len = struct.unpack('<I', f.read(4))[0]
                ws_stream = np.frombuffer(f.read(ws_len), dtype=np.uint8).copy()
                packed_indices = None
                sm_stream = None
            else:
                # Legacy Version 1/2
                indices_len = struct.unpack('<I', f.read(4))[0]
                packed_indices = np.frombuffer(f.read(indices_len), dtype=np.uint8).copy()
                sm_len = struct.unpack('<I', f.read(4))[0]
                sm_stream = np.frombuffer(f.read(sm_len), dtype=np.uint8).copy()
                ws_stream = None

            sidecar = {'indices': np.array([], dtype=np.uint32),
                       'values':  np.array([], dtype=np.uint16)}
            if version >= 2:
                sc_count_bytes = f.read(4)
                if sc_count_bytes:
                    sc_count = struct.unpack('<I', sc_count_bytes)[0]
                    if sc_count > 0:
                        sc_idx = np.frombuffer(f.read(sc_count * 4), dtype=np.uint32).copy()
                        sc_val = np.frombuffer(f.read(sc_count * 2), dtype=np.uint16).copy()
                        sidecar = {'indices': sc_idx, 'values': sc_val}

            encoded_data = {
                'palette':        palette,
                'ws_stream':      ws_stream,
                'packed_indices': packed_indices,
                'sm_stream':      sm_stream,
                'sidecar':        sidecar,
            }
            return encoded_data, num_weights

if __name__ == "__main__":
    # Test storage
    import numpy as np
    from encoder import encode_palette
    
    test_weights = np.array([0x4080, 0x4280, 0x4080, 0x4480], dtype=np.uint16)
    encoded = encode_palette(test_weights)
    
    storage = CompressedTensorStorage("poc/data/test_tensor.sclp")
    storage.save(encoded, len(test_weights))
    print("Saved compressed tensor to poc/data/test_tensor.sclp")
    
    loaded_encoded, loaded_num_weights = storage.load()
    print(f"Loaded num weights: {loaded_num_weights}")
    print(f"Palette match: {np.array_equal(encoded['palette'], loaded_encoded['palette'])}")
    print(f"Indices match: {np.array_equal(encoded['packed_indices'], loaded_encoded['packed_indices'])}")
    print(f"SM stream match: {np.array_equal(encoded['sm_stream'], loaded_encoded['sm_stream'])}")
