"""
Load trained model and run continuous gesture prediction.
State machine: motion-gated segments → classify → single pulse per segment with hysteresis-based re-arming.
Designed for clean idle → attack → idle → attack … for game controls.
"""

import os
import time

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(_DATA_DIR, "model.joblib")
WINDOW_SIZE = 15
SAMPLE_DELAY_SEC = 0.04

# Motion energy gate (gyro magnitude-based)
# Thresholds tuned from recorded data (gyro magnitude stats in gesture_data.csv):
# - IDLE max_mag:   p75 ≈ 1.06, max ≈ 4.18
# - Gestures (ATTACK/BLOCK/MAGIC) max_mag: p25 ≈ 2.7–3.7, median ≈ 3.2–4.4
START_ENERGY_THRESH = 1.0    # start segment when energy rises above this
END_ENERGY_THRESH = 0.3      # end segment when energy stays below this
QUIET_FRAMES_TO_END = 5      # consecutive quiet frames to consider motion ended

# Gesture detection parameters
GESTURE_STABLE_FRAMES = 2       # consecutive frames of same gesture to trigger
PULSE_SEC = 1.0                 # show ATTACK/BLOCK/MAGIC this long
COOLDOWN_SEC = 0.25             # min time before next trigger
MIN_CONFIDENCE = 0.5            # minimum classifier confidence (if available)
INTRA_SEGMENT_RETRIGGER_SEC = 0.25  # min gap before a different gesture can re-trigger within same segment

# States we display; anything else is treated as IDLE
GESTURE_LABELS = {"ATTACK", "BLOCK", "MAGIC", "IDLE"}
TRIGGER_GESTURES = ("ATTACK", "BLOCK", "MAGIC")


def main():
    import joblib
    from mpu_reader import setup_mpu, read_accel, read_gyro
    from features import window_to_features
    from lcd_i2c import LCD_I2C

    # I2C LCD on address 39 (16x2), same as Week 1 lab.
    lcd = LCD_I2C(39, 16, 2)
    lcd.backlight.on()
    lcd.blink.off()  # no blink so transitions look clean

    try:
        lcd.cursor.setPos(0, 0)
        lcd.write_text("Starting...".ljust(16))
        lcd.cursor.setPos(1, 0)
        lcd.write_text("".ljust(16))
    except Exception:
        pass

    if not os.path.exists(MODEL_FILE):
        print(f"{MODEL_FILE} not found. Run train.py first after collecting data.")
        try:
            lcd.cursor.setPos(0, 0)
            lcd.write_text("No model found".ljust(16))
            lcd.cursor.setPos(1, 0)
            lcd.write_text("Run train.py".ljust(16))
        except Exception:
            pass
        return

    package = joblib.load(MODEL_FILE)
    model = package["model"]
    label_encoder = package["label_encoder"]
    has_proba = hasattr(model, "predict_proba")

    try:
        setup_mpu()
    except Exception as e:
        print(f"Error setting up MPU6050: {e}")
        try:
            lcd.cursor.setPos(0, 0)
            lcd.write_text("MPU6050 error".ljust(16))
            lcd.cursor.setPos(1, 0)
            lcd.write_text("Check wiring".ljust(16))
        except Exception:
            pass
        return

    def show_state(state: str):
        try:
            lcd.cursor.setPos(0, 0)
            lcd.write_text("Gesture:".ljust(16))
            lcd.cursor.setPos(1, 0)
            lcd.write_text(state.ljust(16))
        except Exception:
            pass

    window = []

    # Display and timing
    current_display = "IDLE"
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
    last_fired_label = "IDLE"
    last_fire_time = 0.0
    retrigger_label = None
    retrigger_pred_streak = 0

    show_state("IDLE")

    try:
        while True:
            ax, ay, az = read_accel()
            gx, gy, gz = read_gyro()
            motion_energy = (gx * gx + gy * gy + gz * gz) ** 0.5
            print(f"energy={motion_energy:.1f}", end="\r")
            window.append((float(ax), float(ay), float(az), float(gx), float(gy), float(gz)))
            if len(window) > WINDOW_SIZE:
                window.pop(0)

            if len(window) == WINDOW_SIZE:
                feats = window_to_features(window).reshape(1, -1)
                pred_enc = model.predict(feats)[0]
                label = label_encoder.inverse_transform([pred_enc])[0]
                display = label if label in GESTURE_LABELS else "IDLE"
                print(f"raw_pred={display}")
                now = time.monotonic()

                # End pulse -> go back to IDLE display
                if current_display != "IDLE" and now >= pulse_until:
                    current_display = "IDLE"
                    show_state("IDLE")

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
                    last_fired_label = "IDLE"
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
