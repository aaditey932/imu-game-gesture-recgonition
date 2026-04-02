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

import argparse
import json
import os
import queue
import socket
import threading
import time

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
    p.add_argument(
        "--enable-agent",
        action="store_true",
        help="Background LLM agent tunes min_confidence and motion energy thresholds from segment metrics.",
    )
    p.add_argument(
        "--agent-interval",
        type=float,
        default=20.0,
        help="Seconds between agent API calls when --enable-agent.",
    )
    p.add_argument(
        "--agent-model",
        type=str,
        default="gpt-5.4",
        help="Model for OpenAI Responses API (client.responses.create).",
    )
    p.add_argument(
        "--openai-base-url",
        type=str,
        default=None,
        help="e.g. http://localhost:11434/v1 for Ollama; default uses OPENAI_BASE_URL env or OpenAI cloud.",
    )
    p.add_argument(
        "--agent-clamp-min",
        type=float,
        default=0.30,
        help="Lower bound for agent-tuned min_confidence.",
    )
    p.add_argument(
        "--agent-clamp-max",
        type=float,
        default=0.60,
        help="Upper bound for agent-tuned min_confidence.",
    )
    p.add_argument(
        "--agent-max-delta",
        type=float,
        default=0.03,
        help="Max absolute change per agent tool call.",
    )
    p.add_argument(
        "--agent-cooldown",
        type=float,
        default=15.0,
        help="Min seconds between applied min_confidence changes.",
    )
    p.add_argument(
        "--agent-energy-start-min",
        type=float,
        default=0.35,
        help="Agent lower clamp for start_energy.",
    )
    p.add_argument(
        "--agent-energy-start-max",
        type=float,
        default=0.95,
        help="Agent upper clamp for start_energy.",
    )
    p.add_argument(
        "--agent-energy-end-min",
        type=float,
        default=0.10,
        help="Agent lower clamp for end_energy.",
    )
    p.add_argument(
        "--agent-energy-end-max",
        type=float,
        default=0.55,
        help="Agent upper clamp for end_energy.",
    )
    p.add_argument(
        "--agent-energy-max-delta",
        type=float,
        default=0.06,
        help="Max absolute change per set_start_energy / set_end_energy call.",
    )
    p.add_argument(
        "--agent-energy-cooldown",
        type=float,
        default=15.0,
        help="Min seconds between applied start_energy or end_energy changes.",
    )
    p.add_argument(
        "--agent-energy-min-gap",
        type=float,
        default=0.05,
        help="Required gap: start_energy - end_energy after agent updates.",
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
    cfg_lock = threading.Lock()

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
        if args.enable_agent:
            print(
                "Agent (dry-run): --enable-agent would use OPENAI_API_KEY, "
                f"interval={args.agent_interval}s model={args.agent_model}"
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

    stop_event = threading.Event()
    metric_queue = None
    agent_thread = None
    if args.enable_agent:
        from imugesture.tuning_agent import AgentSettings, enqueue_segment_metric, run_agent_loop

        if not os.environ.get("OPENAI_API_KEY"):
            print(
                "Warning: OPENAI_API_KEY unset. Cloud APIs need it; "
                "for Ollama set OPENAI_BASE_URL (e.g. http://localhost:11434/v1) and use a dummy key if required."
            )
        metric_queue = queue.Queue(maxsize=32)
        agent_settings = AgentSettings(
            interval_sec=args.agent_interval,
            model=args.agent_model,
            base_url=args.openai_base_url or os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            clamp_min=args.agent_clamp_min,
            clamp_max=args.agent_clamp_max,
            max_delta=args.agent_max_delta,
            cooldown_sec=args.agent_cooldown,
            energy_start_clamp_min=args.agent_energy_start_min,
            energy_start_clamp_max=args.agent_energy_start_max,
            energy_end_clamp_min=args.agent_energy_end_min,
            energy_end_clamp_max=args.agent_energy_end_max,
            energy_max_delta=args.agent_energy_max_delta,
            energy_cooldown_sec=args.agent_energy_cooldown,
            energy_min_gap=args.agent_energy_min_gap,
        )
        agent_thread = threading.Thread(
            target=run_agent_loop,
            args=(metric_queue, cfg, cfg_lock, stop_event, agent_settings),
            daemon=True,
            name="tuning_agent",
        )
        agent_thread.start()
        print(
            f"Tuning agent thread started (interval={args.agent_interval}s, model={args.agent_model})"
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

    # Per-segment metrics for tuning agent (reset on segment start)
    seg_max_energy = 0.0
    seg_max_max_prob = 0.0
    seg_frames_high_conf = 0
    seg_best_label = None
    seg_best_prob = 0.0
    seg_fire_count = 0
    seg_idle_high_conf_before_start = 0
    accum_idle_high_conf = 0

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
                with cfg_lock:
                    c = dict(cfg)
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

                # Idle-phase: trigger-class + above min_confidence while not in a segment and
                # below segment-open energy (avoids counting the opening frame as idle).
                if (
                    not in_segment
                    and motion_energy <= c["start_energy"]
                ):
                    cand_idle = display if display in TRIGGER_GESTURES else None
                    if cand_idle is not None and max_prob >= c["min_confidence"]:
                        accum_idle_high_conf += 1

                # Motion-based segment start
                if (not in_segment) and now >= cooldown_until and motion_energy > c["start_energy"]:
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
                    seg_idle_high_conf_before_start = accum_idle_high_conf
                    accum_idle_high_conf = 0
                    seg_fire_count = 0
                    seg_max_energy = float(motion_energy)
                    seg_max_max_prob = float(max_prob)
                    seg_frames_high_conf = 0
                    seg_best_label = None
                    seg_best_prob = 0.0
                    if display in TRIGGER_GESTURES:
                        seg_best_label = display
                        seg_best_prob = max_prob

                # Track quiet frames and end segment when motion settles
                if in_segment:
                    seg_max_energy = max(seg_max_energy, float(motion_energy))
                    seg_max_max_prob = max(seg_max_max_prob, float(max_prob))
                    candidate_roll = display if display in TRIGGER_GESTURES else None
                    if candidate_roll is not None and max_prob > seg_best_prob:
                        seg_best_label = candidate_roll
                        seg_best_prob = float(max_prob)
                    if (
                        not segment_fired
                        and candidate_roll is not None
                        and max_prob >= c["min_confidence"]
                    ):
                        seg_frames_high_conf += 1

                    segment_age_frames += 1
                    if motion_energy < c["end_energy"]:
                        quiet_frames += 1
                    else:
                        quiet_frames = 0

                    if quiet_frames >= c["quiet_frames"]:
                        if args.enable_agent and metric_queue is not None:
                            record = {
                                "t_mono": time.monotonic(),
                                "segment_frames": segment_age_frames,
                                "max_energy": seg_max_energy,
                                "max_energy_above_start": float(
                                    seg_max_energy - c["start_energy"]
                                ),
                                "fired": segment_fired,
                                "fire_count": seg_fire_count,
                                "max_max_prob": seg_max_max_prob,
                                "max_prob_above_min": float(
                                    seg_max_max_prob - c["min_confidence"]
                                ),
                                "best_non_idle": {
                                    "label": seg_best_label,
                                    "prob": seg_best_prob,
                                },
                                "frames_high_conf_non_idle": seg_frames_high_conf,
                                "idle_high_conf_frames_before_segment": (
                                    seg_idle_high_conf_before_start
                                ),
                                "thresholds_at_segment_end": {
                                    "start_energy": float(c["start_energy"]),
                                    "end_energy": float(c["end_energy"]),
                                    "min_confidence": float(c["min_confidence"]),
                                    "quiet_frames": int(c["quiet_frames"]),
                                    "stable_frames": int(c["stable_frames"]),
                                    "min_segment_frames": int(c["min_segment_frames"]),
                                },
                            }
                            enqueue_segment_metric(metric_queue, record)
                        in_segment = False
                        segment_fired = False
                        segment_label = None
                        segment_pred_streak = 0
                        quiet_frames = 0

                # First gesture detection within an active segment
                # Wait a few frames after segment start so we latch on the
                # main pose/motion, not the initial transition.
                if in_segment and not segment_fired and segment_age_frames >= c["min_segment_frames"]:
                    candidate = display if display in TRIGGER_GESTURES else None

                    if candidate and max_prob >= c["min_confidence"]:
                        if candidate == segment_label:
                            segment_pred_streak += 1
                        else:
                            segment_label = candidate
                            segment_pred_streak = 1
                    else:
                        segment_label = None
                        segment_pred_streak = 0

                    if segment_label and segment_pred_streak >= c["stable_frames"]:
                        gesture = segment_label
                        print(gesture)
                        notify_action_pc(gesture, max_prob)
                        seg_fire_count += 1
                        show_state(gesture)
                        current_display = gesture
                        pulse_until = now + PULSE_SEC
                        cooldown_until = now + c["cooldown_sec"]
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
                        and max_prob >= c["min_confidence"]
                        and now >= cooldown_until
                        and (now - last_fire_time) >= c["intra_retrigger_sec"]
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

                    if retrigger_label and retrigger_pred_streak >= c["stable_frames"]:
                        gesture = retrigger_label
                        print(gesture)
                        notify_action_pc(gesture, max_prob)
                        seg_fire_count += 1
                        show_state(gesture)
                        current_display = gesture
                        pulse_until = now + PULSE_SEC
                        cooldown_until = now + c["cooldown_sec"]
                        last_fired_label = gesture
                        last_fire_time = now
                        retrigger_label = None
                        retrigger_pred_streak = 0

            time.sleep(SAMPLE_DELAY_SEC)
    except KeyboardInterrupt:
        pass
    finally:
        if args.enable_agent and agent_thread is not None:
            stop_event.set()
            agent_thread.join(timeout=5.0)


if __name__ == "__main__":
    main()
