import numpy as np

def soft_exponent_clip(weights_bf16: np.ndarray, threshold_exponent: int, mantissa_mask: int = 0x7F) -> np.ndarray:
    """
    Performs soft exponent clipping on BF16 weights with stochastic rounding.
    
    Args:
        weights_bf16: Input weights as numpy array of uint16 (representing BF16 bits).
        threshold_exponent: The exponent value above which rare exponents are mapped 
                            to this threshold.
        mantissa_mask: Bit mask for the mantissa to simulate truncation.
                                 
    Returns:
        Clipped weights as a numpy array of uint16.
    """
    # Extract components from BF16 bit representation
    # BF16 layout (1-8-7): [Sign: 15] [Exponent: 14-7] [Mantissa: 6-0]
    
    sign = (weights_bf16 >> 15) & 0x1
    exponent = (weights_bf16 >> 7) & 0xFF
    mantissa = weights_bf16 & 0x7F
    
    # Apply mantissa mask to simulate truncation
    mantissa &= mantissa_mask

    new_exponent = exponent.copy()

    # Hard-clip: exp > threshold+1 always maps to threshold (matches HIP kernel)
    mask_hard = exponent > (threshold_exponent + 1)
    new_exponent[mask_hard] = threshold_exponent

    # Stochastic zone: exp == threshold+1 survives with 50% probability (flat,
    # matching the HIP XorshiftPRNG behaviour rather than mantissa-proportional)
    mask_stoch = exponent == (threshold_exponent + 1)
    if np.any(mask_stoch):
        r = np.random.rand(np.sum(mask_stoch))
        new_exponent[mask_stoch] = np.where(
            r < 0.5, threshold_exponent + 1, threshold_exponent
        )
    
    # Reconstruct the BF16 bit pattern
    clipped_weights = (sign.astype(np.uint16) << 15) | (new_exponent.astype(np.uint16) << 7) | mantissa.astype(np.uint16)
    
    return clipped_weights


if __name__ == "__main__":
    # Test with dummy data
    test_weights = np.array([0x4000, 0x4200, 0x4400, 0x3E00], dtype=np.uint16) # Exponents: 64, 66, 68, 62
    threshold = 65
    clipped = soft_exponent_clip(test_weights, threshold)
    print(f"Original: {test_weights}")
    print(f"Clipped:  {clipped}")


