# IMU Game Gesture Recognition

Real-time gesture recognition from an IMU (MPU6050) and flex sensors for game controls (e.g. **Punch Kick Duck**): six strike classes plus idle, with motion-gated live inference and optional UDP key injection on a PC.

## Project Purpose

**Goal:** Use a wrist-worn or handheld IMU (plus ADS1x15 flex) to trigger game actions via distinct gestures, with one-trigger-per-gesture behavior suitable for game input.

**Approach:** Record labeled windows from the sensors, extract a fixed-size feature vector per window, train a classifier, then run live inference with a motion-gated state machine. Hysteresis and cooldown reduce jitter and double-fires.

## Configuration

Tunable values live next to the code that uses them: **`scripts/collect_data.py`**, **`scripts/train.py`**, **`scripts/run_live.py`** (motion gating, debounce, confidence thresholds, and UDP host/port at the top of that file), **`scripts/augment_csv.py`** (noise sigmas at the top), and **`N_FLEX_CHANNELS`** in **`src/imugesture/features.py`** and **`src/imugesture/flex_reader.py`** (keep those two in sync). Nothing is read from a `.env` file at runtime.

## Layout

| Path | Purpose |
|------|---------|
| `src/imugesture/` | Package: `mpu_reader`, `flex_reader`, `features`, `paths` (data/model file locations) |
| `scripts/` | `collect_data.py`, `augment_csv.py`, `train.py`, `run_live.py` |
| `data/` | `gesture_data.csv` (created by collection) |
| `models/` | `model.joblib` (written by training) |
| `pc/` | `action_receiver.py` for UDP → keyboard on the gaming PC |

## Hardware & Setup

- **IMU:** MPU6050 on I2C (Raspberry Pi with CircuitPython Blinka).
- **Flex:** ADS1015/ADS1115 (see `imugesture.flex_reader`).

**Install** (from repo root on the Pi):

```bash
pip install -r requirements.txt
pip install -e .
```

This installs NumPy, scikit-learn, joblib, Blinka, MPU6050, and ADS1x15 libraries, then registers the `imugesture` package in editable mode so `scripts/` can import it from any working directory.

Use a **virtual environment** on Debian/Raspberry Pi OS if `pip install` is blocked (PEP 668). You can still run `python scripts/collect_data.py` (and the other scripts) from the repo root without installing the package: each script prepends `src/` to `sys.path`. For ad-hoc imports in a REPL, `export PYTHONPATH=src` still works.

## How to Run

Run the pipeline in this order (always from the **repository root** so paths resolve):

1. **Collect data** — Interactive recording to `data/gesture_data.csv`:
   ```bash
   python scripts/collect_data.py
   ```
   Keys **1–7**: Idle, punch_left, kick_left, duck_left, punch_right, kick_right, duck_right (**q** quit). See the script docstring for session targets.

2. **Augment data (optional)** — Append synthetic rows by resampling per class and adding Gaussian noise (same CSV schema; preserves each parent row’s `take_id` for group splits):
   ```bash
   python scripts/augment_csv.py
   ```
   Use `--per-class N`, `--seed`, and optional `--input` / `--output` paths; defaults append to `data/gesture_data.csv`. Tune `SIGMA_*` in the script if needed. Running multiple times adds more rows each run.

3. **Train** — Build the model:
   ```bash
   python scripts/train.py
   ```
   Reads `data/gesture_data.csv`, writes `models/model.joblib`. Needs at least two classes.

4. **Run live** — Stream sensors and fire gestures (optional UDP to `pc/action_receiver.py`):
   ```bash
   python scripts/run_live.py
   ```
   Loads `models/model.joblib`. Edit `ACTION_RECEIVER_HOST` / `ACTION_RECEIVER_PORT` at the top of `run_live.py` (empty host disables UDP).

**Dry runs:** `python scripts/collect_data.py --dry-run`, `python scripts/train.py --dry-run`, `python scripts/run_live.py --dry-run`.

## ML Model

- **Classifier:** scikit-learn **Random Forest** in `scripts/train.py` (300 trees, `class_weight="balanced"`, `oob_score=True`).
- **Input:** Feature vector from one time window (`imugesture.features.window_to_features`). Window = 15 samples at 0.04 s (match constants in `scripts/collect_data.py`, `scripts/train.py`, `scripts/run_live.py`).
- **Features:** Per-channel statistics on IMU axes, flex channels, accel/gyro magnitude, and zero-crossing counts (dimension matches `N_FEATURES` in `imugesture.features`).

## Training Pipeline

- **Data:** `data/gesture_data.csv` — label, flattened samples per timestep, `timestamp_iso`, `take_id`.
- **Augmentation:** Optional extra rows from `scripts/augment_csv.py` (noise on IMU/flex columns; new `timestamp_iso`; inherited `take_id` from the sampled source row).
- **Split:** **GroupShuffleSplit** by `take_id` when possible; else stratified **train_test_split**.
- **Artifact:** `models/model.joblib` holds `"model"` and `"label_encoder"` for `scripts/run_live.py`.

## Live Inference

- **Motion gating:** Segment starts when gyro energy exceeds a threshold; ends after consecutive quiet frames.
- **Triggering:** Stable prediction + confidence → one UDP action (or console print) per segment; cooldown and intra-segment re-trigger rules apply.

## Constants

Keep `WINDOW_SIZE`, `SAMPLE_DELAY_SEC`, and `N_FLEX_CHANNELS` consistent across `collect_data.py`, `train.py`, `run_live.py`, `features.py`, and `flex_reader.py` when you change them.

## PC receiver

On the gaming machine:

```bash
pip install -r pc/requirements.txt
python pc/action_receiver.py
```

Default key map matches Punch Kick Duck (see `pc/action_receiver.py` docstring).
