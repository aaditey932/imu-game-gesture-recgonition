# IMU Game Gesture Recognition

Real-time gesture recognition from an IMU (MPU6050) for game controls. Recognizes ATTACK, BLOCK, MAGIC, and IDLE from 6-axis accelerometer and gyroscope data.

## Project Purpose

**Goal:** Use a wrist-worn or handheld IMU to trigger game actions (e.g. attack, block, magic) via distinct gestures, with a clean one-trigger-per-gesture behavior suitable for game input.

**Approach:** Record labeled 6-axis (accel + gyro) windows from the MPU6050, extract a fixed-size feature vector per window, train a classifier, then run live inference with a motion-gated state machine so each gesture produces a single trigger. Hysteresis and cooldown reduce jitter and double-fires.

## Hardware & Setup

- **IMU:** MPU6050 connected over I2C (e.g. Raspberry Pi using `board` / `busio` from CircuitPython).
- **Optional:** I2C LCD (16×2 at address 39) for status during data collection and live run. The scripts run without it; LCD is used only when the `lcd_i2c` module is available.

**Requirements:** Python 3 and the dependencies in `requirements.txt`:

```bash
pip install -r requirements.txt
```

This installs `numpy`, `scikit-learn`, `joblib`, and `adafruit-circuitpython-mpu6050`. The optional LCD uses a local/custom `lcd_i2c` module (not in requirements).

## How to Run

Run the pipeline in this order:

1. **Collect data** — Record labeled gesture windows:
   ```bash
   python collect_data.py
   ```
   Press **1** = IDLE, **2** = ATTACK, **3** = BLOCK, **4** = MAGIC. Follow the in-script guidance for balanced sessions. Data is appended to `gesture_data.csv`. You need this file before training.

2. **Train** — Build the model from the CSV:
   ```bash
   python train.py
   ```
   Reads `gesture_data.csv` and writes `model.joblib`. Requires at least two gesture classes.

3. **Run live** — Stream IMU data and output gesture triggers:
   ```bash
   python run_live.py
   ```
   Loads `model.joblib`, streams from the MPU6050, and prints triggered gestures (and shows them on the LCD if present). Requires `model.joblib` from step 2.

**Data quality:** For best results, follow the collector’s session plan: collect enough IDLE (including near-misses like half-attacks and fidgets), balance ATTACK and BLOCK, and do 2–3 sessions on different days. See the docstring in `collect_data.py` for target window counts and session structure.

## ML Model

- **Classifier:** scikit-learn **Random Forest** (`RandomForestClassifier` with 300 trees, `class_weight="balanced"`, `oob_score=True`). Implemented in `train.py`.
- **Input:** Each prediction uses a **50-dimensional feature vector** derived from one time window (see `features.py`). A window is 15 samples × 6 axes (ax, ay, az, gx, gy, gz) at 0.04 s per sample (~0.6 s total).
- **Features (50 total):** For each of the 6 channels: mean, std, min, max, range, energy (mean of squares). Same 6 stats for accel magnitude and for gyro magnitude. Plus zero-crossing counts for accel magnitude and gyro magnitude.

## Training Pipeline

- **Data:** `gesture_data.csv`. Each row is one labeled window: `label` plus flattened samples (ax_0…gz_14), and optionally `timestamp_iso` and `take_id`.
- **Steps:** Load CSV → build windows → convert each window to features with `window_to_features()` from `features.py` → encode labels with `LabelEncoder` → split data:
  - If there are at least 2 distinct `take_id`s: **GroupShuffleSplit** (20% test) so the same take is not in both train and test.
  - Otherwise: stratified **train_test_split** (20% test).
- **Training:** Fit the Random Forest on the training set. Script prints train accuracy, test accuracy, OOB score, classification report, and confusion matrix. Saves to `model.joblib` a dict with `"model"` and `"label_encoder"` for use in `run_live.py`.

## Data Collection Tips

- Aim for **IDLE** windows ≥ ATTACK + BLOCK combined; label near-misses (half-attacks, twitches, resets, fidgets) as IDLE.
- Typical targets: ~150–300 windows each for ATTACK and BLOCK, ~400–800 for IDLE. Do 2–3 sessions on different days for variety.

## Live Inference Behavior

- **Motion gating:** A segment starts when gyro-based motion energy exceeds a threshold and ends after several consecutive quiet frames. Only segments (not every frame) can trigger a gesture.
- **Triggering:** A gesture is fired when the same class is predicted for a small number of consecutive frames (with minimum confidence). One gesture produces one trigger; a short pulse and cooldown avoid repeats. A different gesture in the same segment can re-trigger after a short delay.

## Constants

Window size (15 samples) and sample delay (0.04 s) are defined in both `train.py` and `run_live.py`. If you change them, update both files so training and live inference use the same window.
