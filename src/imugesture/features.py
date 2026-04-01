"""
Feature extraction for IMU + flex gesture recognition.
Converts a window of (ax, ay, az, gx, gy, gz, f0, …) samples into a fixed-size feature vector.
"""

import math
import numpy as np

# Must match flex_reader.N_FLEX_CHANNELS and WINDOW_SIZE in collect_data / train / run_live.
N_FLEX_CHANNELS = 2


def _magnitude_3d(x, y, z):
    """Magnitude of (x, y, z)."""
    return math.sqrt(x * x + y * y + z * z)


# Per IMU axis: 6 stats; per flex channel: 6 stats; accel_mag: 6; gyro_mag: 6; zc: 2
N_FEATURES = 6 * 6 + 6 * N_FLEX_CHANNELS + 6 + 6 + 2


def _zero_crossing_count(arr):
    """Count sign changes between consecutive samples (zero-crossings)."""
    if len(arr) < 2:
        return 0
    return int(np.sum((arr[1:] * arr[:-1]) < 0))


def window_to_features(window):
    """
    Convert a window of samples into a numeric feature vector.

    Window: list of tuples of length 6 + N_FLEX_CHANNELS:
      (ax, ay, az, gx, gy, gz, f0, [f1, …]).
    Returns: 1D numpy array of shape (n_features,) with N_FEATURES elements.

    Feature order:
      - Per IMU channel (ax…gz), 6 stats each: mean, std, min, max, range, energy. Total 36.
      - Per flex channel, same 6 stats. Total 6 * N_FLEX_CHANNELS.
      - Accel magnitude (per sample): mean, std, min, max, range, energy. Total 6.
      - Gyro magnitude (per sample): mean, std, min, max, range, energy. Total 6.
      - Zero-crossing count for accel magnitude and gyro magnitude. Total 2.
    """
    if not window:
        return np.zeros(N_FEATURES, dtype=float)

    arr = np.asarray(window, dtype=float)
    n_samples = arr.shape[0]
    n_cols = 6 + N_FLEX_CHANNELS
    if arr.shape[1] != n_cols:
        raise ValueError(
            f"window must have {n_cols} columns (6 IMU + {N_FLEX_CHANNELS} flex), got {arr.shape[1]}"
        )

    features = []

    # Per IMU channel: mean, std, min, max, range, energy
    for c in range(6):
        col = arr[:, c]
        mean_val = float(np.mean(col))
        std_val = float(np.std(col)) if n_samples > 1 else 0.0
        min_val = float(np.min(col))
        max_val = float(np.max(col))
        range_val = max_val - min_val
        energy_val = float(np.mean(col ** 2))
        features.extend([mean_val, std_val, min_val, max_val, range_val, energy_val])

    # Per flex channel: same six statistics
    for c in range(N_FLEX_CHANNELS):
        col = arr[:, 6 + c]
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

    # Zero-crossings: accel_mag and gyro_mag only
    features.append(float(_zero_crossing_count(accel_mag)))
    features.append(float(_zero_crossing_count(gyro_mag)))

    return np.array(features, dtype=float)
