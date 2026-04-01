"""
Load trained model and run continuous gesture prediction.
State machine: motion-gated segments → classify → single pulse per segment with hysteresis-based re-arming.
Designed for clean idle → attack → idle → attack … for game controls.
"""

import sys
from pathlib import Path

_repo_src = Path(__file__).resolve().parents[1] / "src"
if _repo_src.is_dir():
    sys.path.insert(0, str(_repo_src))

import json
import os
import socket
import time
import argparse

from imugesture.paths import MODEL_JOBLIB

WINDOW_SIZE = 15
SAMPLE_DELAY_SEC = 0.04

# Motion gating / live detection
START_ENERGY_THRESH = 1.0
END_ENERGY_THRESH = 0.3
QUIET_FRAMES_TO_END = 5

GESTURE_STABLE_FRAMES = 2
PULSE_SEC = 1.0
COOLDOWN_SEC = 0.25
MIN_CONFIDENCE = 0.5
INTRA_SEGMENT_RETRIGGER_SEC = 0.25

# PC running pc/action_receiver.py — set host to "" to disable UDP
ACTION_RECEIVER_HOST = "192.168.1.218"
ACTION_RECEIVER_PORT = 5005

# Trained classes. Any unknown label is treated as Idle.
IDLE_LABEL = "Idle"
GESTURE_LABELS = {
    IDLE_LABEL,
    "punch_left",
    "kick_left",
    "duck_left",
    "punch_right",
    "kick_right",
    "duck_right",
}
TRIGGER_GESTURES = tuple(sorted(l for l in GESTURE_LABELS if l != IDLE_LABEL))


def parse_args():
    p = argparse.ArgumentParser(description="Run live gesture inference from IMU + flex sensors.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Load model/classes and exit without initializing sensors or running live loop.",
    )
    return p.parse_args()


def main():
    import joblib
    from imugesture.mpu_reader import setup_mpu, read_accel, read_gyro
    from imugesture.flex_reader import setup_flex, read_flex
    from imugesture.features import window_to_features

    args = parse_args()

    if not os.path.exists(MODEL_JOBLIB):
        print(
            f"{MODEL_JOBLIB} not found. Run scripts/train.py first after collecting data."
        )
        return

    package = joblib.load(MODEL_JOBLIB)
    model = package["model"]
    label_encoder = package["label_encoder"]
    has_proba = hasattr(model, "predict_proba")
    model_classes = set(str(x) for x in getattr(label_encoder, "classes_", []))

    if args.dry_run:
        print("Dry run OK: model loaded.")
        print(f"Model classes: {sorted(model_classes)}")
        unknown = sorted(model_classes - GESTURE_LABELS)
        if unknown:
            print(f"Warning: model has classes not in live labels: {unknown}")
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

    action_receiver_host = ACTION_RECEIVER_HOST.strip()
    action_sock = None
    action_addr = None
    if action_receiver_host:
        action_port = ACTION_RECEIVER_PORT
        action_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        action_sock.setblocking(False)
        action_addr = (action_receiver_host, action_port)
        print(f"Action UDP -> {action_addr[0]}:{action_addr[1]}")

    def notify_action_pc(gesture_label: str, confidence: float) -> None:
        if action_sock is None or gesture_label not in TRIGGER_GESTURES:
            return
        payload = json.dumps(
            {"action": gesture_label.lower(), "confidence": float(confidence)}
        ).encode("utf-8")
        try:
            action_sock.sendto(payload, action_addr)
        except OSError as e:
            print(f"action UDP send failed: {e!r}")

    def show_state(state: str):
        print(f"state={state}")

    window = []

    # Display and timing
    current_display = IDLE_LABEL
    cooldown_until = 0.0
    pulse_until = 0.0

    # Motion-gated segment state
    in_segment = False
    segment_fired = False
    segment_label = None
    segment_pred_streak = 0
    quiet_frames = 0
    segment_age_frames = 0

    # Gesture firing history (for combos)
    last_fired_label = IDLE_LABEL
    last_fire_time = 0.0
    retrigger_label = None
    retrigger_pred_streak = 0

    show_state(IDLE_LABEL)

    try:
        while True:
            ax, ay, az = read_accel()
            gx, gy, gz = read_gyro()
            flex_vals = read_flex()
            motion_energy = (gx * gx + gy * gy + gz * gz) ** 0.5
            print(f"energy={motion_energy:.1f}", end="\r")
            window.append(
                (float(ax), float(ay), float(az), float(gx), float(gy), float(gz)) + tuple(flex_vals)
            )
            if len(window) > WINDOW_SIZE:
                window.pop(0)

            if len(window) == WINDOW_SIZE:
                feats = window_to_features(window).reshape(1, -1)
                pred_enc = model.predict(feats)[0]
                label = label_encoder.inverse_transform([pred_enc])[0]
                display = label if label in GESTURE_LABELS else IDLE_LABEL
                print(f"raw_pred={display}")
                now = time.monotonic()

                # End pulse -> go back to idle display
                if current_display != IDLE_LABEL and now >= pulse_until:
                    current_display = IDLE_LABEL
                    show_state(IDLE_LABEL)

                # Classifier confidence (if available)
                max_prob = 1.0
                if has_proba:
                    try:
                        probs = model.predict_proba(feats)[0]
                        max_prob = float(probs.max())
                    except Exception:
                        max_prob = 1.0

                # Motion-based segment start
                if (not in_segment) and now >= cooldown_until and motion_energy > START_ENERGY_THRESH:
                    in_segment = True
                    segment_fired = False
                    segment_label = None
                    segment_pred_streak = 0
                    quiet_frames = 0
                    segment_age_frames = 0
                    # Reset per-segment gesture history
                    last_fired_label = IDLE_LABEL
                    last_fire_time = 0.0
                    retrigger_label = None
                    retrigger_pred_streak = 0

                # Track quiet frames and end segment when motion settles
                if in_segment:
                    segment_age_frames += 1
                    if motion_energy < END_ENERGY_THRESH:
                        quiet_frames += 1
                    else:
                        quiet_frames = 0

                    if quiet_frames >= QUIET_FRAMES_TO_END:
                        in_segment = False
                        segment_fired = False
                        segment_label = None
                        segment_pred_streak = 0
                        quiet_frames = 0

                # First gesture detection within an active segment
                # Wait a few frames after segment start so we latch on the
                # main pose/motion, not the initial transition.
                if in_segment and not segment_fired and segment_age_frames >= 3:
                    candidate = display if display in TRIGGER_GESTURES else None

                    if candidate and max_prob >= MIN_CONFIDENCE:
                        if candidate == segment_label:
                            segment_pred_streak += 1
                        else:
                            segment_label = candidate
                            segment_pred_streak = 1
                    else:
                        segment_label = None
                        segment_pred_streak = 0

                    if segment_label and segment_pred_streak >= GESTURE_STABLE_FRAMES:
                        gesture = segment_label
                        print(gesture)
                        notify_action_pc(gesture, max_prob)
                        show_state(gesture)
                        current_display = gesture
                        pulse_until = now + PULSE_SEC
                        cooldown_until = now + COOLDOWN_SEC
                        segment_fired = True
                        last_fired_label = gesture
                        last_fire_time = now
                        # Reset for possible future re-trigger
                        retrigger_label = None
                        retrigger_pred_streak = 0

                # Gesture-change re-trigger within an active segment
                if in_segment and segment_fired:
                    candidate2 = display if display in TRIGGER_GESTURES else None

                    can_consider_new = (
                        candidate2 is not None
                        and candidate2 != last_fired_label
                        and max_prob >= MIN_CONFIDENCE
                        and now >= cooldown_until
                        and (now - last_fire_time) >= INTRA_SEGMENT_RETRIGGER_SEC
                    )

                    if can_consider_new:
                        if candidate2 == retrigger_label:
                            retrigger_pred_streak += 1
                        else:
                            retrigger_label = candidate2
                            retrigger_pred_streak = 1
                    else:
                        retrigger_label = None
                        retrigger_pred_streak = 0

                    if retrigger_label and retrigger_pred_streak >= GESTURE_STABLE_FRAMES:
                        gesture = retrigger_label
                        print(gesture)
                        notify_action_pc(gesture, max_prob)
                        show_state(gesture)
                        current_display = gesture
                        pulse_until = now + PULSE_SEC
                        cooldown_until = now + COOLDOWN_SEC
                        last_fired_label = gesture
                        last_fire_time = now
                        retrigger_label = None
                        retrigger_pred_streak = 0

            time.sleep(SAMPLE_DELAY_SEC)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
