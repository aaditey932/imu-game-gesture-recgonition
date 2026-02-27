"""
Feature extraction for IMU gesture recognition.
Converts a window of (ax, ay, az, gx, gy, gz) samples into a fixed-size feature vector.
"""

import math
import numpy as np

# Channel names for the 6-axis window
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]


def _magnitude_3d(x, y, z):
    """Magnitude of (x, y, z)."""
    return math.sqrt(x * x + y * y + z * z)


# Total feature count: 6*6 + 6 + 6 + 2 = 50
N_FEATURES = 6 * 6 + 6 + 6 + 2


def _zero_crossing_count(arr):
    """Count sign changes between consecutive samples (zero-crossings)."""
    if len(arr) < 2:
        return 0
    return int(np.sum((arr[1:] * arr[:-1]) < 0))


def window_to_features(window):
    """
    Convert a window of 6-axis samples into a numeric feature vector.

    Window: list of 6-tuples (ax, ay, az, gx, gy, gz).
    Returns: 1D numpy array of shape (n_features,) with N_FEATURES (50) elements.

    Feature order:
      - Per channel (ax, ay, az, gx, gy, gz), 6 stats each: mean, std, min, max,
        range (peak-to-peak), energy (mean of squared values). Total 36.
      - Accel magnitude (per sample): mean, std, min, max, range, energy. Total 6.
      - Gyro magnitude (per sample): mean, std, min, max, range, energy. Total 6.
      - Zero-crossing count for accel magnitude and gyro magnitude. Total 2.
    """
    if not window:
        return np.zeros(N_FEATURES, dtype=float)

    arr = np.asarray(window, dtype=float)
    # arr shape: (n_samples, 6) -> columns 0..5 are ax, ay, az, gx, gy, gz
    n_samples = arr.shape[0]
    features = []

    # Per channel: mean, std, min, max, range, energy (mean of squares for scale stability)
    for c in range(6):
        col = arr[:, c]
        mean_val = float(np.mean(col))
        std_val = float(np.std(col)) if n_samples > 1 else 0.0
        min_val = float(np.min(col))
        max_val = float(np.max(col))
        range_val = max_val - min_val
        energy_val = float(np.mean(col ** 2))
        features.extend([mean_val, std_val, min_val, max_val, range_val, energy_val])

    # Accel magnitude per sample: sqrt(ax^2 + ay^2 + az^2)
    accel_mag = np.array(
        [_magnitude_3d(arr[i, 0], arr[i, 1], arr[i, 2]) for i in range(n_samples)],
        dtype=float,
    )
    features.append(float(np.mean(accel_mag)))
    features.append(float(np.std(accel_mag)) if n_samples > 1 else 0.0)
    features.append(float(np.min(accel_mag)))
    features.append(float(np.max(accel_mag)))
    features.append(float(np.max(accel_mag) - np.min(accel_mag)))
    features.append(float(np.mean(accel_mag ** 2)))

    # Gyro magnitude per sample: sqrt(gx^2 + gy^2 + gz^2)
    gyro_mag = np.array(
        [_magnitude_3d(arr[i, 3], arr[i, 4], arr[i, 5]) for i in range(n_samples)],
        dtype=float,
    )
    features.append(float(np.mean(gyro_mag)))
    features.append(float(np.std(gyro_mag)) if n_samples > 1 else 0.0)
    features.append(float(np.min(gyro_mag)))
    features.append(float(np.max(gyro_mag)))
    features.append(float(np.max(gyro_mag) - np.min(gyro_mag)))
    features.append(float(np.mean(gyro_mag ** 2)))

    # Zero-crossings (Option B): accel_mag and gyro_mag only
    features.append(float(_zero_crossing_count(accel_mag)))
    features.append(float(_zero_crossing_count(gyro_mag)))

    return np.array(features, dtype=float)
