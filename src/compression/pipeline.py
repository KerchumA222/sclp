import numpy as np
from .clipping import soft_exponent_clip
from .encoder import encode_palette
from .decoder import decode_palette
from .storage import CompressedTensorStorage

class SCLPCompressor:
    """
    High-level API for the SCLP (Soft Clipping Lossless-First) compression pipeline.
    Encapsulates clipping, encoding, and decoding.
    """
    def __init__(self, threshold_exponent: int = 125):
        self.threshold = threshold_exponent

    def compress(self, weights_bf16: np.ndarray) -> dict:
        """
        Full compression pipeline: Clip -> Encode.
        
        Args:
            weights_bf16: Input weights as numpy array of uint16 (BF16 bits).
            
        Returns:
            A dictionary containing the compressed data and metadata.
        """
        num_weights = len(weights_bf16)
        clipped_bits = soft_exponent_clip(weights_bf16, self.threshold)
        encoded_data = encode_palette(clipped_bits)
        return {
            'data': encoded_data,
            'num_weights': num_weights
        }

    def decompress(self, compressed_package: dict) -> np.ndarray:
        """
        Full decompression pipeline: Decode bits.
        
        Args:
            compressed_package: The dictionary returned by `compress`.
            
        Returns:
            Reconstructed weights as numpy array of uint16 (BF16 bits).
        """
        encoded_data = compressed_package['data']
        num_weights = compressed_package['num_weights']
        return decode_palette(encoded_data, num_weights)

    def save_to_file(self, weights_bf16: np.ndarray, filepath: str):
        """Helper to compress and save directly to a file."""
        package = self.compress(weights_bf16)
        storage = CompressedTensorStorage(filepath)
        storage.save(package['data'], package['num_weights'])

    def load_from_file(self, filepath: str) -> np.ndarray:
        """Helper to load and decompress directly from a file."""
        storage = CompressedTensorStorage(filepath)
        encoded_data, num_weights = storage.load()
        return decode_palette(encoded_data, num_weights)
