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

# Motion gating / live detection (conservative defaults; use CLI for snappier tuning)
START_ENERGY_THRESH = 0.65
END_ENERGY_THRESH = 0.32
QUIET_FRAMES_TO_END = 3

# Wait this many frames after segment start before latching (skip transition noise).
MIN_SEGMENT_FRAMES_BEFORE_LATCH = 3

GESTURE_STABLE_FRAMES = 2
PULSE_SEC = 0.5
COOLDOWN_SEC = 0.25
MIN_CONFIDENCE = 0.45
INTRA_SEGMENT_RETRIGGER_SEC = 0.25

# Min time between two fires of the same class within one motion segment.
SAME_GESTURE_MIN_GAP_SEC = 0.65

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
    p.add_argument("--start-energy", type=float, default=None, help="Override START_ENERGY_THRESH.")
    p.add_argument("--end-energy", type=float, default=None, help="Override END_ENERGY_THRESH.")
    p.add_argument(
        "--quiet-frames", type=int, default=None, help="Override QUIET_FRAMES_TO_END."
    )
    p.add_argument(
        "--min-confidence", type=float, default=None, help="Override MIN_CONFIDENCE."
    )
    p.add_argument(
        "--stable-frames", type=int, default=None, help="Override GESTURE_STABLE_FRAMES."
    )
    p.add_argument(
        "--min-segment-frames",
        type=int,
        default=None,
        help="Override MIN_SEGMENT_FRAMES_BEFORE_LATCH.",
    )
    p.add_argument("--cooldown", type=float, default=None, help="Override COOLDOWN_SEC.")
    p.add_argument(
        "--intra-retrigger",
        type=float,
        default=None,
        help="Override INTRA_SEGMENT_RETRIGGER_SEC (different class within segment).",
    )
    p.add_argument(
        "--same-gesture-gap",
        type=float,
        default=None,
        help="Override SAME_GESTURE_MIN_GAP_SEC.",
    )
    return p.parse_args()


def _live_config(args: argparse.Namespace):
    """Resolve thresholds: CLI overrides, else module defaults."""
    return {
        "start_energy": args.start_energy if args.start_energy is not None else START_ENERGY_THRESH,
        "end_energy": args.end_energy if args.end_energy is not None else END_ENERGY_THRESH,
        "quiet_frames": args.quiet_frames if args.quiet_frames is not None else QUIET_FRAMES_TO_END,
        "min_confidence": args.min_confidence if args.min_confidence is not None else MIN_CONFIDENCE,
        "stable_frames": args.stable_frames if args.stable_frames is not None else GESTURE_STABLE_FRAMES,
        "min_segment_frames": args.min_segment_frames
        if args.min_segment_frames is not None
        else MIN_SEGMENT_FRAMES_BEFORE_LATCH,
        "cooldown_sec": args.cooldown if args.cooldown is not None else COOLDOWN_SEC,
        "intra_retrigger_sec": args.intra_retrigger
        if args.intra_retrigger is not None
        else INTRA_SEGMENT_RETRIGGER_SEC,
        "same_gesture_gap_sec": args.same_gesture_gap
        if args.same_gesture_gap is not None
        else SAME_GESTURE_MIN_GAP_SEC,
    }


def main():
    import joblib
    from imugesture.mpu_reader import setup_mpu, read_accel, read_gyro
    from imugesture.flex_reader import setup_flex, read_flex
    from imugesture.features import FEATURE_SCHEMA as EXPECTED_FEATURE_SCHEMA
    from imugesture.features import window_to_features

    args = parse_args()
    cfg = _live_config(args)

    if not os.path.exists(MODEL_JOBLIB):
        print(
            f"{MODEL_JOBLIB} not found. Run scripts/train.py first after collecting data."
        )
        return

    package = joblib.load(MODEL_JOBLIB)
    model = package["model"]
    label_encoder = package["label_encoder"]
    loaded_schema = package.get("feature_schema")
    if loaded_schema != EXPECTED_FEATURE_SCHEMA:
        print(
            f"Warning: model feature_schema={loaded_schema!r}, "
            f"expected {EXPECTED_FEATURE_SCHEMA!r}. Retrain with scripts/train.py if inference fails."
        )
    has_proba = hasattr(model, "predict_proba")
    model_classes = set(str(x) for x in getattr(label_encoder, "classes_", []))

    if args.dry_run:
        print("Dry run OK: model loaded.")
        print(f"feature_schema (model): {loaded_schema!r}, expected: {EXPECTED_FEATURE_SCHEMA!r}")
        print(f"Model classes: {sorted(model_classes)}")
        unknown = sorted(model_classes - GESTURE_LABELS)
        if unknown:
            print(f"Warning: model has classes not in live labels: {unknown}")
        print(
            "Live thresholds:",
            f"start_energy={cfg['start_energy']}",
            f"end_energy={cfg['end_energy']}",
            f"quiet_frames={cfg['quiet_frames']}",
            f"min_confidence={cfg['min_confidence']}",
            f"stable_frames={cfg['stable_frames']}",
            f"min_segment_frames={cfg['min_segment_frames']}",
            f"cooldown_sec={cfg['cooldown_sec']}",
            f"intra_retrigger_sec={cfg['intra_retrigger_sec']}",
            f"same_gesture_gap_sec={cfg['same_gesture_gap_sec']}",
        )
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
    print(
        "Live thresholds:",
        f"start_energy={cfg['start_energy']}",
        f"end_energy={cfg['end_energy']}",
        f"quiet_frames={cfg['quiet_frames']}",
        f"min_confidence={cfg['min_confidence']}",
        f"stable_frames={cfg['stable_frames']}",
        f"min_segment_frames={cfg['min_segment_frames']}",
        f"same_gesture_gap_sec={cfg['same_gesture_gap_sec']}",
    )

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
    cooldown_until = 0.5
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
                max_prob = 1.0
                if has_proba:
                    try:
                        probs = model.predict_proba(feats)[0]
                        max_prob = float(probs.max())
                    except Exception:
                        max_prob = 1.0
                print(f"raw_pred={display} conf={max_prob:.3f}")
                now = time.monotonic()

                # End pulse -> go back to idle display
                if current_display != IDLE_LABEL and now >= pulse_until:
                    current_display = IDLE_LABEL
                    show_state(IDLE_LABEL)

                # Motion-based segment start
                if (not in_segment) and now >= cooldown_until and motion_energy > cfg["start_energy"]:
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
                    if motion_energy < cfg["end_energy"]:
                        quiet_frames += 1
                    else:
                        quiet_frames = 0

                    if quiet_frames >= cfg["quiet_frames"]:
                        in_segment = False
                        segment_fired = False
                        segment_label = None
                        segment_pred_streak = 0
                        quiet_frames = 0

                # First gesture detection within an active segment
                # Wait a few frames after segment start so we latch on the
                # main pose/motion, not the initial transition.
                if in_segment and not segment_fired and segment_age_frames >= cfg["min_segment_frames"]:
                    candidate = display if display in TRIGGER_GESTURES else None

                    if candidate and max_prob >= cfg["min_confidence"]:
                        if candidate == segment_label:
                            segment_pred_streak += 1
                        else:
                            segment_label = candidate
                            segment_pred_streak = 1
                    else:
                        segment_label = None
                        segment_pred_streak = 0

                    if segment_label and segment_pred_streak >= cfg["stable_frames"]:
                        gesture = segment_label
                        print(gesture)
                        notify_action_pc(gesture, max_prob)
                        show_state(gesture)
                        current_display = gesture
                        pulse_until = now + PULSE_SEC
                        cooldown_until = now + cfg["cooldown_sec"]
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

                    if retrigger_label and retrigger_pred_streak >= cfg["stable_frames"]:
                        gesture = retrigger_label
                        print(gesture)
                        notify_action_pc(gesture, max_prob)
                        show_state(gesture)
                        current_display = gesture
                        pulse_until = now + PULSE_SEC
                        cooldown_until = now + cfg["cooldown_sec"]
                        last_fired_label = gesture
                        last_fire_time = now
                        retrigger_label = None
                        retrigger_pred_streak = 0

            time.sleep(SAMPLE_DELAY_SEC)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
