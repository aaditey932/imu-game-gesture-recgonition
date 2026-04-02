"""Unit tests for tuning_agent validation (no API calls)."""

import threading
import unittest

from imugesture.tuning_agent import (
    AgentSettings,
    apply_tool_call,
    fix_energy_hysteresis,
    validate_and_apply_min_confidence,
    validate_energy_scalar,
)


def _full_cfg():
    return {
        "min_confidence": 0.45,
        "start_energy": 0.65,
        "end_energy": 0.32,
    }


class TestTuningAgent(unittest.TestCase):
    def test_validate_clamps_absolute_bounds(self):
        # Large max_delta so one step can reach clamp edges from a mid current.
        s = AgentSettings(clamp_min=0.3, clamp_max=0.6, max_delta=1.0)
        v, _ = validate_and_apply_min_confidence(0.1, 0.45, s)
        self.assertEqual(v, 0.3)
        v2, _ = validate_and_apply_min_confidence(0.99, 0.45, s)
        self.assertEqual(v2, 0.6)

    def test_validate_caps_delta(self):
        s = AgentSettings(clamp_min=0.3, clamp_max=0.6, max_delta=0.03)
        v, note = validate_and_apply_min_confidence(0.55, 0.45, s)
        self.assertLessEqual(abs(v - 0.45), 0.03 + 1e-6)
        self.assertIn("capped", note)

    def test_validate_energy_scalar_delta(self):
        v, note = validate_energy_scalar(0.9, 0.65, 0.2, 1.0, 0.06)
        self.assertLessEqual(abs(v - 0.65), 0.06 + 1e-6)
        self.assertIn("capped", note)

    def test_fix_energy_hysteresis(self):
        cfg = {"start_energy": 0.40, "end_energy": 0.38}
        s = AgentSettings(
            energy_start_clamp_min=0.35,
            energy_start_clamp_max=0.95,
            energy_end_clamp_min=0.10,
            energy_end_clamp_max=0.55,
            energy_min_gap=0.05,
        )
        fix_energy_hysteresis(cfg, s)
        self.assertGreaterEqual(cfg["start_energy"] - cfg["end_energy"], s.energy_min_gap - 1e-9)

    def test_apply_tool_noop(self):
        cfg = _full_cfg()
        lock = threading.Lock()
        s = AgentSettings()
        out = apply_tool_call("noop", '{"reason": "test"}', cfg, lock, s, [], [])
        self.assertIn("noop", out.lower())
        self.assertEqual(cfg["min_confidence"], 0.45)

    def test_apply_set_min_confidence(self):
        cfg = _full_cfg()
        lock = threading.Lock()
        s = AgentSettings(clamp_min=0.3, clamp_max=0.6, max_delta=0.05, cooldown_sec=0.0)
        last: list = []
        last_e: list = []
        out = apply_tool_call(
            "set_min_confidence",
            '{"value": 0.50, "reason": "raise"}',
            cfg,
            lock,
            s,
            last,
            last_e,
        )
        self.assertEqual(cfg["min_confidence"], 0.50)
        self.assertIn("0.5", out)

    def test_apply_set_start_energy_preserves_gap(self):
        cfg = _full_cfg()
        lock = threading.Lock()
        s = AgentSettings(
            energy_start_clamp_min=0.20,
            energy_start_clamp_max=0.95,
            energy_end_clamp_min=0.10,
            energy_end_clamp_max=0.55,
            energy_max_delta=0.30,
            energy_cooldown_sec=0.0,
            energy_min_gap=0.05,
        )
        last: list = []
        last_e: list = []
        apply_tool_call(
            "set_start_energy",
            '{"value": 0.40, "reason": "lower start"}',
            cfg,
            lock,
            s,
            last,
            last_e,
        )
        self.assertGreaterEqual(cfg["start_energy"] - cfg["end_energy"], s.energy_min_gap - 1e-9)

    def test_apply_set_end_energy_raises_start_if_needed(self):
        cfg = {"min_confidence": 0.45, "start_energy": 0.36, "end_energy": 0.32}
        lock = threading.Lock()
        s = AgentSettings(
            energy_start_clamp_min=0.35,
            energy_start_clamp_max=0.95,
            energy_end_clamp_min=0.10,
            energy_end_clamp_max=0.55,
            energy_max_delta=0.10,
            energy_cooldown_sec=0.0,
            energy_min_gap=0.05,
        )
        last: list = []
        last_e: list = []
        apply_tool_call(
            "set_end_energy",
            '{"value": 0.34, "reason": "raise end"}',
            cfg,
            lock,
            s,
            last,
            last_e,
        )
        self.assertGreaterEqual(cfg["start_energy"] - cfg["end_energy"], s.energy_min_gap - 1e-9)


if __name__ == "__main__":
    unittest.main()
