"""
Collect labeled gesture data from the MPU6050.
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
(update train.py and run_live.py to match).
"""

import csv
import sys
import time
import os
from datetime import datetime, timezone

# Window: 15 samples × 0.04 s = 0.6 s (tuned for quicker event gestures).
WINDOW_SIZE = 15
SAMPLE_DELAY_SEC = 0.04
WINDOWS_PER_TAKE = 10
PAUSE_BETWEEN_WINDOWS_SEC = 1.0
LEAD_IN_SEC = 0.3
COUNTDOWN_SEC = 1.0

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(_DATA_DIR, "gesture_data.csv")

# 1=IDLE, 2=ATTACK, 3=BLOCK, 4=MAGIC
GESTURE_LABELS = ["IDLE", "ATTACK", "BLOCK", "MAGIC"]
KEY_TO_LABEL = {"1": "IDLE", "2": "ATTACK", "3": "BLOCK", "4": "MAGIC"}


def build_csv_header():
    """Header: label + flattened window columns + timestamp_iso + take_id."""
    header = ["label"]
    for i in range(WINDOW_SIZE):
        for ch in ["ax", "ay", "az", "gx", "gy", "gz"]:
            header.append(f"{ch}_{i}")
    header.append("timestamp_iso")
    header.append("take_id")
    return header


def ensure_csv_exists():
    """Create CSV with header if it does not exist."""
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(build_csv_header())


def _csv_has_new_format():
    """True if existing CSV has timestamp_iso and take_id columns (so we can append in new format)."""
    if not os.path.exists(DATA_FILE):
        return True
    with open(DATA_FILE, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
    return header is not None and "timestamp_iso" in header and "take_id" in header


def record_one_window(read_accel_fn, read_gyro_fn):
    """
    Record exactly WINDOW_SIZE samples (accel + gyro each time).
    Returns list of 6-tuples (ax, ay, az, gx, gy, gz).
    """
    window = []
    for _ in range(WINDOW_SIZE):
        ax, ay, az = read_accel_fn()
        gx, gy, gz = read_gyro_fn()
        window.append((float(ax), float(ay), float(az), float(gx), float(gy), float(gz)))
        time.sleep(SAMPLE_DELAY_SEC)
    return window


def flatten_window(window):
    """Flatten window to a list of numbers in order ax_0, ay_0, az_0, gx_0, gy_0, gz_0, ax_1, ..."""
    row = []
    for sample in window:
        row.extend(sample)
    return row


def append_row(label, window, timestamp_iso, take_id):
    """Append one labeled window as a row to gesture_data.csv. Uses new columns only if file already has them."""
    ensure_csv_exists()
    row = [label] + flatten_window(window)
    if _csv_has_new_format():
        row = row + [timestamp_iso, take_id]
    with open(DATA_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def beep():
    """Terminal bell so you don't need to watch the screen."""
    print("\a", flush=True)


def _lcd_show(lcd, line0: str, line1: str = ""):
    """Update LCD if available."""
    if lcd is None:
        return
    try:
        lcd.cursor.setPos(0, 0)
        lcd.write_text(line0[:16].ljust(16))
        lcd.cursor.setPos(1, 0)
        lcd.write_text(line1[:16].ljust(16))
    except Exception:
        pass


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


def run_countdown_and_lead_in(label, lcd):
    """Print and show on LCD: Ready… then 3, 2, 1, GO with COUNTDOWN_SEC between; beep on GO; wait LEAD_IN_SEC."""
    print("Ready…")
    _lcd_show(lcd, "Ready…", label)
    for s in ["3", "2", "1", "GO"]:
        time.sleep(COUNTDOWN_SEC)
        print(s)
        _lcd_show(lcd, s, label)
        if s == "GO":
            beep()
    time.sleep(LEAD_IN_SEC)


def main():
    from mpu_reader import setup_mpu, read_accel, read_gyro

    lcd = None
    try:
        from lcd_i2c import LCD_I2C
        lcd = LCD_I2C(39, 16, 2)
        lcd.backlight.on()
    except Exception:
        pass

    print("IMU Gesture Data Collection")
    print("Labels: 1=IDLE  2=ATTACK  3=BLOCK  4=MAGIC  (press key to record immediately)")
    print("IDLE >= ATTACK+BLOCK; include near-misses as IDLE.")
    print(f"Window: {WINDOW_SIZE} samples, {SAMPLE_DELAY_SEC}s between; {WINDOWS_PER_TAKE} windows per take.")
    print(f"Data saved to: {DATA_FILE}\n")

    try:
        setup_mpu()
    except Exception as e:
        print(f"Error setting up MPU6050: {e}")
        return

    take_id = 0
    while True:
        print("\nPress 1=IDLE  2=ATTACK  3=BLOCK  4=MAGIC  (or q to quit):")
        choice = get_single_key()
        if choice == "q" or choice == "":
            break
        if choice not in KEY_TO_LABEL:
            print("Use 1, 2, 3, or 4 (or q to quit).")
            continue
        label = KEY_TO_LABEL[choice]
        take_id += 1
        timestamp_iso = datetime.now(timezone.utc).isoformat()

        run_countdown_and_lead_in(label, lcd)

        for i in range(WINDOWS_PER_TAKE):
            print(f"  Recording window {i + 1}/{WINDOWS_PER_TAKE}…")
            _lcd_show(lcd, label, f"{i + 1}/{WINDOWS_PER_TAKE}")
            window = record_one_window(read_accel, read_gyro)
            append_row(label, window, timestamp_iso, take_id)
            if i < WINDOWS_PER_TAKE - 1:
                time.sleep(PAUSE_BETWEEN_WINDOWS_SEC)
        print(f"Saved {WINDOWS_PER_TAKE} windows for '{label}' (take_id={take_id}).")


if __name__ == "__main__":
    main()
