import numpy as np

def calculate_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calculates Mean Squared Error using float64 to prevent overflow."""
    # Convert to float64 immediately to handle large differences and avoid precision loss
    diff = np.abs(original.astype(np.float64) - reconstructed.astype(np.float64))
    # Cap the difference at a value whose square fits in float64 (~1e150^2 = 1e300)
    return float(np.mean(np.square(np.clip(diff, 0, 1e150))))

def calculate_mae(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calculates Mean Absolute Error between two weight tensors."""
    return float(np.mean(np.abs(original.astype(np.float64) - reconstructed.astype(np.float64))))

def calculate_relative_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calculates relative error (MAE / mean magnitude)."""
    mae = calculate_mae(original, reconstructed)
    mean_mag = float(np.mean(np.abs(original.astype(np.float64))))
    return mae / mean_mag if mean_mag != 0 else 0.0

