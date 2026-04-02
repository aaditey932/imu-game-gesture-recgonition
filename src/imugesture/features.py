"""
Feature extraction for IMU + flex gesture recognition.
Converts a window of (ax, ay, az, gx, gy, gz, f0, …) samples into a fixed-size feature vector.

Schema v2 (99 features) extends v1 (62) with:
  - Dynamic accel magnitude after removing window mean from ax,ay,az (gravity-robust proxy).
  - Per-axis first-difference stats; gyro-magnitude delta stats; normalized peak index of ||gyro||.
  - Minimal spectral features on mean-centered gyro magnitude (FFT power bands).
  - Flex first-difference stats and f0−f1 shape.

Must match flex_reader.N_FLEX_CHANNELS and WINDOW_SIZE in collect_data / train / run_live.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

N_FLEX_CHANNELS = 2
FEATURE_SCHEMA = "v2"

_IMU_NAMES = ("ax", "ay", "az", "gx", "gy", "gz")


def _magnitude_3d(x, y, z) -> float:
    """Magnitude of (x, y, z)."""
    return math.sqrt(x * x + y * y + z * z)


def _six_stats(col: np.ndarray) -> List[float]:
    """mean, std, min, max, range, mean-square energy."""
    col = np.asarray(col, dtype=float).ravel()
    n = col.size
    if n == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    mean_val = float(np.mean(col))
    std_val = float(np.std(col)) if n > 1 else 0.0
    min_val = float(np.min(col))
    max_val = float(np.max(col))
    range_val = max_val - min_val
    energy_val = float(np.mean(col**2))
    return [mean_val, std_val, min_val, max_val, range_val, energy_val]


def _zero_crossing_count(arr: np.ndarray) -> int:
    """Count sign changes between consecutive samples (zero-crossings)."""
    if len(arr) < 2:
        return 0
    return int(np.sum((arr[1:] * arr[:-1]) < 0))


def _diff_mean_abs_std_max(col: np.ndarray) -> List[float]:
    """First differences: mean(|Δ|), std(Δ), max(|Δ|)."""
    col = np.asarray(col, dtype=float).ravel()
    if col.size < 2:
        return [0.0, 0.0, 0.0]
    d = np.diff(col)
    return [float(np.mean(np.abs(d))), float(np.std(d)), float(np.max(np.abs(d)))]


def _spectral_gyro_mag_features(gyro_mag: np.ndarray) -> List[float]:
    """
    Minimal FFT features on mean-centered gyro magnitude.
    Returns: dominant_bin_norm, low_band_ratio, mid_band_ratio (excludes DC).
    """
    x = np.asarray(gyro_mag, dtype=float).ravel() - float(np.mean(gyro_mag))
    n = x.size
    if n < 2:
        return [0.0, 0.0, 0.0]
    power = np.abs(np.fft.rfft(x)) ** 2
    # Drop DC (index 0); use remaining bins
    bins = power[1:].astype(float)
    s = float(np.sum(bins))
    if s < 1e-18 or bins.size == 0:
        return [0.0, 0.0, 0.0]
    dom_idx = int(np.argmax(bins))
    dom_norm = float(dom_idx / max(1, bins.size - 1))
    low_end = min(2, bins.size)
    low = float(np.sum(bins[:low_end]) / s)
    mid_start = low_end
    mid_end = min(mid_start + 3, bins.size)
    mid = float(np.sum(bins[mid_start:mid_end]) / s) if mid_start < bins.size else 0.0
    return [dom_norm, low, mid]


def _build_feature_names() -> List[str]:
    """Ordered names matching window_to_features output (v2)."""
    names: List[str] = []
    for ch in _IMU_NAMES:
        for s in ("mean", "std", "min", "max", "range", "energy"):
            names.append(f"imu_{ch}_{s}")
    for j in range(N_FLEX_CHANNELS):
        for s in ("mean", "std", "min", "max", "range", "energy"):
            names.append(f"flex{j}_{s}")
    for s in ("mean", "std", "min", "max", "range", "energy"):
        names.append(f"accel_mag_{s}")
    for s in ("mean", "std", "min", "max", "range", "energy"):
        names.append(f"gyro_mag_{s}")
    names.extend(["zc_accel_mag", "zc_gyro_mag"])
    # v2
    for s in ("mean", "std", "min", "max", "range", "energy"):
        names.append(f"dyn_accel_mag_{s}")
    for ch in _IMU_NAMES:
        names.extend(
            [
                f"d_{ch}_mean_abs",
                f"d_{ch}_std",
                f"d_{ch}_max_abs",
            ]
        )
    names.extend(
        [
            "d_gyro_mag_mean_abs",
            "d_gyro_mag_std",
            "d_gyro_mag_max_abs",
            "gyro_mag_peak_index_norm",
        ]
    )
    names.extend(
        [
            "spec_gyro_mag_dom_bin_norm",
            "spec_gyro_mag_low_ratio",
            "spec_gyro_mag_mid_ratio",
        ]
    )
    names.extend(
        [
            "flex0_diff_mean_abs",
            "flex0_diff_std",
            "flex1_diff_mean_abs",
            "flex1_diff_std",
            "flex0_minus_f1_mean",
            "flex0_minus_f1_std",
        ]
    )
    return names


FEATURE_NAMES: List[str] = _build_feature_names()
N_FEATURES = len(FEATURE_NAMES)


def window_to_features(window) -> np.ndarray:
    """
    Convert a window of samples into a numeric feature vector (schema v2).

    Window: list of tuples of length 6 + N_FLEX_CHANNELS:
      (ax, ay, az, gx, gy, gz, f0, [f1, …]).
    Returns: 1D numpy array of shape (N_FEATURES,).

    v1 block (62): IMU per-axis stats, flex stats, accel/gyro magnitude stats, ZC counts.
    v2 block (+37): dynamic accel magnitude, per-axis deltas, gyro_mag deltas, peak index,
                    spectral gyro_mag, flex deltas and f0−f1.
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

    features: List[float] = []

    # --- v1: Per IMU channel ---
    for c in range(6):
        features.extend(_six_stats(arr[:, c]))

    # --- v1: Per flex channel ---
    for c in range(N_FLEX_CHANNELS):
        features.extend(_six_stats(arr[:, 6 + c]))

    # --- v1: Accel magnitude ---
    accel_mag = np.sqrt(np.sum(arr[:, 0:3] ** 2, axis=1))
    features.extend(_six_stats(accel_mag))

    # --- v1: Gyro magnitude ---
    gyro_mag = np.sqrt(np.sum(arr[:, 3:6] ** 2, axis=1))
    features.extend(_six_stats(gyro_mag))

    features.append(float(_zero_crossing_count(accel_mag)))
    features.append(float(_zero_crossing_count(gyro_mag)))

    # --- v2: Dynamic acceleration magnitude (subtract mean accel per axis) ---
    mean_a = np.mean(arr[:, 0:3], axis=0)
    dyn = arr[:, 0:3] - mean_a
    dyn_mag = np.sqrt(np.sum(dyn**2, axis=1))
    features.extend(_six_stats(dyn_mag))

    # --- v2: Per-axis first differences (IMU) ---
    for c in range(6):
        features.extend(_diff_mean_abs_std_max(arr[:, c]))

    # --- v2: Gyro magnitude series differences ---
    features.extend(_diff_mean_abs_std_max(gyro_mag))

    # --- v2: Normalized peak index of gyro magnitude ---
    peak_norm = float(np.argmax(gyro_mag)) / max(1.0, float(n_samples - 1))
    features.append(peak_norm)

    # --- v2: Spectral (gyro_mag) ---
    features.extend(_spectral_gyro_mag_features(gyro_mag))

    # --- v2: Flex deltas and cross-channel ---
    f0 = arr[:, 6]
    f1 = arr[:, 7]
    features.extend(_diff_mean_abs_std_max(f0)[:2])  # mean_abs, std
    features.extend(_diff_mean_abs_std_max(f1)[:2])
    diff_f = f0 - f1
    features.append(float(np.mean(diff_f)))
    features.append(float(np.std(diff_f)) if n_samples > 1 else 0.0)

    out = np.array(features, dtype=float)
    if out.size != N_FEATURES:
        raise RuntimeError(
            f"Feature count mismatch: built {out.size}, expected {N_FEATURES}"
        )
    return out
