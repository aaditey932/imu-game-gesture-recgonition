"""
Collect labeled gesture data from the MPU6050 and flex sensors (ADS1x15).
Interactive: press 1/2/3 (IDLE/ATTACK/BLOCK) to start recording; countdown, lead-in, then
multiple windows per take. Data appended to CSV with timestamp and take_id.

Session targets (aim for balanced data):
  - ATTACK: 150–300 windows (vary speed, amplitude, wrist angle, posture).
  - BLOCK: 150–300 windows (same).
  - IDLE: 400–800 windows (at least as many as ATTACK+BLOCK; include near-misses).
  - Near-miss IDLE: half-attacks, wrist twitches, resets, fidgets, scratching, strap
    adjustment, wrist rotation without the gesture — label all as IDLE.

Session plan (20–30 min): warm-up 30 IDLE → ATTACK (normal/fast/small/different wrist) →
BLOCK (same) → 150–300 IDLE near-misses. Do 2–3 sessions on different days.

Timing: the 0.3 s lead-in after "GO" puts the gesture peak roughly in the middle of the
window. For quicker event gestures you can try WINDOW_SIZE=15–20 and SAMPLE_DELAY_SEC=0.03–0.04
(edit the constants below; keep them aligned with train.py and run_live.py).
"""

import sys
from pathlib import Path

_repo_src = Path(__file__).resolve().parents[1] / "src"
if _repo_src.is_dir():
    sys.path.insert(0, str(_repo_src))

import csv
import time
import os
import re
import argparse
from datetime import datetime, timezone

from imugesture.paths import DATA_CSV, ensure_data_dirs

WINDOW_SIZE = 15
SAMPLE_DELAY_SEC = 0.04
N_FLEX_CHANNELS = 2

WINDOWS_PER_TAKE = 100
# New take_id (and a fresh timestamp_iso) every N windows within one recording run.
# Must divide WINDOWS_PER_TAKE evenly so each recording run uses an integer number of chunks.
TAKE_ID_EVERY_N_WINDOWS = 20
PAUSE_BETWEEN_WINDOWS_SEC = 1.0
LEAD_IN_SEC = 0.3
COUNTDOWN_SEC = 1.0

# 1..7 mapped to the gesture classes below.
GESTURE_LABELS = [
    "Idle",
    "punch_left",
    "kick_left",
    "duck_left",
    "punch_right",
    "kick_right",
    "duck_right",
]
KEY_TO_LABEL = {
    "1": "Idle",
    "2": "punch_left",
    "3": "kick_left",
    "4": "duck_left",
    "5": "punch_right",
    "6": "kick_right",
    "7": "duck_right",
}


def build_csv_header():
    """Header: label + flattened window columns (IMU + flex per timestep) + timestamp_iso + take_id."""
    header = ["label"]
    for i in range(WINDOW_SIZE):
        for ch in ["ax", "ay", "az", "gx", "gy", "gz"]:
            header.append(f"{ch}_{i}")
        for j in range(N_FLEX_CHANNELS):
            header.append(f"f{j}_{i}")
    header.append("timestamp_iso")
    header.append("take_id")
    return header


def ensure_csv_exists():
    """Create CSV with header if it does not exist."""
    ensure_data_dirs()
    if not os.path.exists(DATA_CSV):
        with open(DATA_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(build_csv_header())
        return
    migrate_csv_if_needed()


def _detect_flex_count_from_header(header):
    max_idx = -1
    pat = re.compile(r"^f(\d+)_\d+$")
    for col in header:
        m = pat.match(col)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1 if max_idx >= 0 else 0


def _base_header_with_flex_count(flex_count):
    base = ["label"]
    for i in range(WINDOW_SIZE):
        for ch in ["ax", "ay", "az", "gx", "gy", "gz"]:
            base.append(f"{ch}_{i}")
        for j in range(flex_count):
            base.append(f"f{j}_{i}")
    return base


def _normalize_row_width(row, width):
    if len(row) < width:
        return row + [""] * (width - len(row))
    if len(row) > width:
        return row[:width]
    return row


def _upgrade_row_old_flex_to_new_flex(row, old_flex_count, has_ts_take):
    old_sample_width = 6 + old_flex_count
    new_sample_width = 6 + N_FLEX_CHANNELS
    old_flat_len = WINDOW_SIZE * old_sample_width
    new_flat_len = WINDOW_SIZE * new_sample_width

    row = _normalize_row_width(row, 1 + old_flat_len + (2 if has_ts_take else 0))
    label = row[0] if row else ""
    old_flat = row[1 : 1 + old_flat_len]
    ts_take = row[1 + old_flat_len : 1 + old_flat_len + 2] if has_ts_take else []

    new_flat = []
    for i in range(WINDOW_SIZE):
        start = i * old_sample_width
        sample = old_flat[start : start + old_sample_width]
        sample = _normalize_row_width(sample, old_sample_width)
        imu = sample[:6]
        flex_old = sample[6:]
        flex_new = flex_old + ["0.0"] * max(0, N_FLEX_CHANNELS - old_flex_count)
        flex_new = flex_new[:N_FLEX_CHANNELS]
        new_flat.extend(imu + flex_new)

    if has_ts_take:
        ts_take = _normalize_row_width(ts_take, 2)
    else:
        ts_take = ["", ""]
    upgraded = [label] + new_flat + ts_take
    return _normalize_row_width(upgraded, 1 + new_flat_len + 2)


def migrate_csv_if_needed():
    """In-place migrate older CSV schemas to current header (2 flex channels + timestamp/take_id)."""
    if not os.path.exists(DATA_CSV):
        return
    expected = build_csv_header()
    with open(DATA_CSV, "r", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        with open(DATA_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(expected)
        return

    header = rows[0]
    if header == expected:
        return

    n_exp = len(expected)
    # Training/export CSVs may append derived feature columns after take_id; collection only stores raw window + meta.
    if len(header) > n_exp and header[:n_exp] == expected:
        migrated = [expected]
        for row in rows[1:]:
            if not row:
                continue
            migrated.append(_normalize_row_width(row[:n_exp], n_exp))
        tmp_file = DATA_CSV.parent / (DATA_CSV.name + ".tmp")
        with open(tmp_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(migrated)
        os.replace(tmp_file, DATA_CSV)
        return

    old_flex_count = _detect_flex_count_from_header(header)
    if old_flex_count <= 0:
        raise ValueError(
            f"{DATA_CSV} header is incompatible with current schema; cannot auto-migrate."
        )
    has_ts_take = "timestamp_iso" in header and "take_id" in header
    compatible_old = _base_header_with_flex_count(old_flex_count) + (
        ["timestamp_iso", "take_id"] if has_ts_take else []
    )
    if header != compatible_old:
        raise ValueError(
            f"{DATA_CSV} header is incompatible with current schema; cannot auto-migrate."
        )

    migrated = [expected]
    for row in rows[1:]:
        if not row:
            continue
        migrated.append(_upgrade_row_old_flex_to_new_flex(row, old_flex_count, has_ts_take))

    tmp_file = DATA_CSV.parent / (DATA_CSV.name + ".tmp")
    with open(tmp_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(migrated)
    os.replace(tmp_file, DATA_CSV)


def record_one_window(read_accel_fn, read_gyro_fn, read_flex_fn):
    """
    Record exactly WINDOW_SIZE samples (accel + gyro + flex each time).
    Returns list of (6 + N_FLEX_CHANNELS)-tuples per sample.
    """
    window = []
    for _ in range(WINDOW_SIZE):
        ax, ay, az = read_accel_fn()
        gx, gy, gz = read_gyro_fn()
        flex_vals = read_flex_fn()
        if len(flex_vals) != N_FLEX_CHANNELS:
            raise ValueError(
                f"read_flex_fn() must return {N_FLEX_CHANNELS} values, got {len(flex_vals)}"
            )
        sample = (float(ax), float(ay), float(az), float(gx), float(gy), float(gz))
        sample = sample + tuple(float(x) for x in flex_vals)
        window.append(sample)
        time.sleep(SAMPLE_DELAY_SEC)
    return window


def flatten_window(window):
    """Flatten window: per sample ax, ay, az, gx, gy, gz, f0, … then next timestep."""
    row = []
    for sample in window:
        row.extend(sample)
    return row


def read_max_take_id(csv_path):
    """Largest take_id in existing CSV (0 if missing/empty). Used so restarts do not reuse ids."""
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "take_id" not in reader.fieldnames:
            return 0
        m = 0
        for row in reader:
            v = (row.get("take_id") or "").strip()
            if not v:
                continue
            try:
                m = max(m, int(v))
            except ValueError:
                pass
        return m


def append_row(label, window, timestamp_iso, take_id):
    """Append one labeled window as a row to gesture_data.csv. Uses new columns only if file already has them."""
    ensure_csv_exists()
    row = [label] + flatten_window(window)
    row = row + [timestamp_iso, take_id]
    with open(DATA_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def beep():
    """Terminal bell so you don't need to watch the screen."""
    print("\a", flush=True)


def get_single_key():
    """Read a single key (1, 2, 3, or q) without requiring Enter. Falls back to input() if raw mode unavailable."""
    if sys.stdin.isatty():
        try:
            import termios
            import tty
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
                return ch.lower() if ch else ""
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass
    line = input().strip()
    return line[0].lower() if line else ""


def run_countdown_and_lead_in(label):
    """Print Ready, then 3, 2, 1, GO with COUNTDOWN_SEC between; beep on GO; wait LEAD_IN_SEC."""
    print("Ready…")
    for s in ["3", "2", "1", "GO"]:
        time.sleep(COUNTDOWN_SEC)
        print(s)
        if s == "GO":
            beep()
    time.sleep(LEAD_IN_SEC)


def parse_args():
    p = argparse.ArgumentParser(
        description="Collect labeled IMU + flex gesture windows into data/gesture_data.csv."
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate CSV schema setup and exit without starting interactive capture.",
    )
    return p.parse_args()


def main():
    from imugesture.mpu_reader import setup_mpu, read_accel, read_gyro
    from imugesture.flex_reader import setup_flex, read_flex

    args = parse_args()

    print("IMU + flex gesture data collection")
    print(
        "Labels: 1=Idle 2=punch_left 3=kick_left 4=duck_left "
        "5=punch_right 6=kick_right 7=duck_right"
    )
    print("Press key to record immediately.")
    print(
        f"Window: {WINDOW_SIZE} samples, {SAMPLE_DELAY_SEC}s between; "
        f"{WINDOWS_PER_TAKE} windows per run; take_id changes every {TAKE_ID_EVERY_N_WINDOWS} windows."
    )
    print(f"Data saved to: {DATA_CSV}\n")

    try:
        ensure_csv_exists()
    except Exception as e:
        print(f"Error preparing CSV schema: {e}")
        return
    if args.dry_run:
        print(f"Dry run OK: CSV schema ready at {DATA_CSV}")
        return

    try:
        setup_mpu()
    except Exception as e:
        print(f"Error setting up MPU6050: {e}")
        return

    try:
        setup_flex()
    except Exception as e:
        print(f"Error setting up ADS1x15 (flex): {e}")
        return

    if WINDOWS_PER_TAKE % TAKE_ID_EVERY_N_WINDOWS != 0:
        print(
            "Error: WINDOWS_PER_TAKE must be a multiple of TAKE_ID_EVERY_N_WINDOWS "
            "so each recording run emits equal-sized chunks."
        )
        return

    take_id = read_max_take_id(str(DATA_CSV))
    if take_id > 0:
        print(f"Resuming take_id from CSV (next chunk will use ids starting at {take_id + 1}).")

    while True:
        print(
            "\nPress 1=Idle 2=punch_left 3=kick_left 4=duck_left "
            "5=punch_right 6=kick_right 7=duck_right (or q to quit):"
        )
        choice = get_single_key()
        if choice == "q" or choice == "":
            break
        if choice not in KEY_TO_LABEL:
            print("Use 1..7 (or q to quit).")
            continue
        label = KEY_TO_LABEL[choice]

        run_countdown_and_lead_in(label)

        first_take_this_run = None
        last_take_this_run = None
        n_chunks = (WINDOWS_PER_TAKE + TAKE_ID_EVERY_N_WINDOWS - 1) // TAKE_ID_EVERY_N_WINDOWS

        for i in range(WINDOWS_PER_TAKE):
            if i % TAKE_ID_EVERY_N_WINDOWS == 0:
                take_id += 1
                last_take_this_run = take_id
                if first_take_this_run is None:
                    first_take_this_run = take_id
                timestamp_iso = datetime.now(timezone.utc).isoformat()

            chunk = i // TAKE_ID_EVERY_N_WINDOWS + 1
            print(
                f"  Recording window {i + 1}/{WINDOWS_PER_TAKE} "
                f"(take_id={take_id}, chunk {chunk}/{n_chunks})…"
            )
            window = record_one_window(read_accel, read_gyro, read_flex)
            append_row(label, window, timestamp_iso, take_id)
            if i < WINDOWS_PER_TAKE - 1:
                time.sleep(PAUSE_BETWEEN_WINDOWS_SEC)

        print(
            f"Saved {WINDOWS_PER_TAKE} windows for '{label}' "
            f"(take_id {first_take_this_run}–{last_take_this_run})."
        )


if __name__ == "__main__":
    main()
