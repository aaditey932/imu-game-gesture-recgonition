# IMU Game Gesture Recognition

Wrist-worn or handheld **MPU6050** plus **ADS1x15** flex → six strike classes + Idle for games (e.g. **Punch Kick Duck**). **Flow:** label windows in `collect_data`, train a **Random Forest**, run **motion-gated** live inference with optional **UDP → keyboard** on a PC. Hysteresis and cooldown keep input game-friendly.

## Configuration

Tunables sit next to their code: `scripts/collect_data.py`, `train.py`, `run_live.py`, `augment_csv.py`, and `N_FLEX_CHANNELS` in `src/imugesture/features.py` + `flex_reader.py` (keep those two in sync). The repo does **not** load `.env` automatically; export variables in the shell or your service file. For the optional LLM agent: **`OPENAI_API_KEY`**, optional **`OPENAI_BASE_URL`**.

## Layout

| Path | Purpose |
|------|---------|
| `src/imugesture/` | `mpu_reader`, `flex_reader`, `features`, `paths` |
| `src/imugesture/tuning_agent.py` | Optional OpenAI Responses API loop for live threshold tuning |
| `scripts/` | `collect_data.py`, `augment_csv.py`, `train.py`, `run_live.py` |
| `data/` | `gesture_data.csv` |
| `models/` | `model.joblib` |
| `pc/` | `action_receiver.py` — UDP → keys on the gaming PC |

## Hardware & Setup

- **IMU:** MPU6050 (I2C, CircuitPython Blinka). **Flex:** ADS1015/ADS1115.

From repo root on the Pi:

```bash
pip install -r requirements.txt
pip install -e .
```

Uses a venv if PEP 668 blocks system installs. Scripts prepend `src/` to `sys.path`; for a bare REPL, `export PYTHONPATH=src`.

## How to Run

Always from **repository root**:

1. **Collect** → `data/gesture_data.csv`: `python scripts/collect_data.py` — keys **1–7** (Idle + six gestures), **q** quit; see script docstring for targets.
2. **Augment (optional):** `python scripts/augment_csv.py` — `--per-class`, `--seed`, optional `--input`/`--output`; tune `SIGMA_*` in script.
3. **Train:** `python scripts/train.py` → `models/model.joblib` (needs ≥2 classes).
4. **Live:** `python scripts/run_live.py` — set `ACTION_RECEIVER_HOST` / `ACTION_RECEIVER_PORT` in `run_live.py` (empty host = no UDP).

**Dry runs:** `python scripts/collect_data.py --dry-run`, `python scripts/train.py --dry-run`, `python scripts/run_live.py --dry-run`.

## ML Model

- **Classifier:** scikit-learn Random Forest in `train.py` (300 trees, `class_weight="balanced"`, `oob_score=True`).
- **Window:** 15 samples × 0.04 s — same constants in `collect_data.py`, `train.py`, `run_live.py`.
- **Features:** Per-channel stats on IMU, flex, magnitudes, zero-crossings (`N_FEATURES` in `imugesture.features`).

## Training Pipeline

- **CSV:** label, flattened samples, `timestamp_iso`, `take_id`.
- **Augment:** noise + inherited `take_id` for group splits.
- **Split:** `GroupShuffleSplit` by `take_id` when possible; else stratified `train_test_split`.
- **Artifact:** `model.joblib` has `model` + `label_encoder`.

## Live Inference

- **Motion gating:** Segment opens when gyro energy passes `start_energy`; closes after `quiet_frames` below `end_energy`.
- **Triggering:** Stable non-Idle class + `min_confidence` → one fire (re-triggers allowed with cooldown / intra-segment rules).

**Optional LLM agent:** `python scripts/run_live.py --enable-agent` (needs `openai` from `requirements.txt`). Uses OpenAI **Responses** API; set `OPENAI_API_KEY` and `OPENAI_BASE_URL` if required. Adjusts **`min_confidence`**, **`start_energy`**, **`end_energy`** with server-side clamps — flags **`--agent-*`**, **`--agent-energy-*`**, details in [`src/imugesture/tuning_agent.py`](src/imugesture/tuning_agent.py).

## Constants

Keep `WINDOW_SIZE`, `SAMPLE_DELAY_SEC`, and `N_FLEX_CHANNELS` aligned across `collect_data.py`, `train.py`, `run_live.py`, `features.py`, and `flex_reader.py`.

## PC receiver

```bash
pip install -r pc/requirements.txt
python pc/action_receiver.py
```

Default key map: Punch Kick Duck (`pc/action_receiver.py` docstring).
