"""
LLM tool-calling loop for live tuning of min_confidence (OpenAI Responses API).

Environment:
  OPENAI_API_KEY — API key.
  OPENAI_BASE_URL — optional override for the API base URL.

Uses client.responses.create() with function tools (set_min_confidence, set_start_energy,
set_end_energy, noop), then function_call_output items and a follow-up responses.create.
See: https://developers.openai.com/api/docs/guides/function-calling
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Lazy import so dry-run without openai still works until pip install
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


@dataclass
class AgentSettings:
    interval_sec: float = 20.0
    model: str = "gpt-5.4"
    base_url: Optional[str] = None
    api_key: str = ""
    clamp_min: float = 0.30
    clamp_max: float = 0.60
    max_delta: float = 0.03
    cooldown_sec: float = 15.0
    queue_maxsize: int = 32
    # Motion gating (gyro energy scale, same units as run_live START/END_ENERGY_THRESH)
    energy_start_clamp_min: float = 0.35
    energy_start_clamp_max: float = 0.95
    energy_end_clamp_min: float = 0.10
    energy_end_clamp_max: float = 0.55
    energy_max_delta: float = 0.06
    energy_cooldown_sec: float = 15.0
    energy_min_gap: float = 0.05


def tool_definitions() -> List[Dict[str, Any]]:
    """Responses API shape: type/name/description/parameters (not nested under function)."""
    return [
        {
            "type": "function",
            "name": "set_min_confidence",
            "description": (
                "Adjust the minimum classifier probability required to accept a gesture. "
                "Lower slightly if segments show motion and plausible class probabilities but no fire; "
                "raise slightly if idle false positives are likely. Respect safety bounds."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "New min_confidence in [0,1].",
                    },
                    "reason": {
                        "type": "string",
                        "description": "One short sentence explaining the change.",
                    },
                },
                "required": ["value", "reason"],
            },
        },
        {
            "type": "function",
            "name": "set_start_energy",
            "description": (
                "Raise or lower the gyro energy threshold that starts a motion segment. "
                "Lower if segments rarely start despite movement; raise if noise keeps opening segments."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "New start_energy threshold."},
                    "reason": {"type": "string", "description": "One short sentence."},
                },
                "required": ["value", "reason"],
            },
        },
        {
            "type": "function",
            "name": "set_end_energy",
            "description": (
                "Adjust the energy below which the segment counts as quiet. "
                "Lower if segments end too early; raise if they linger after motion stops."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "New end_energy threshold."},
                    "reason": {"type": "string", "description": "One short sentence."},
                },
                "required": ["value", "reason"],
            },
        },
        {
            "type": "function",
            "name": "noop",
            "description": "Make no tuning changes this cycle (min_confidence or energy).",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation.",
                    },
                },
                "required": ["reason"],
            },
        },
    ]


def validate_and_apply_min_confidence(
    raw: float,
    current: float,
    settings: AgentSettings,
) -> tuple[float, str]:
    """
    Clamp to [clamp_min, clamp_max], limit delta from current by max_delta.
    Returns (new_value, note).
    """
    lo, hi = settings.clamp_min, settings.clamp_max
    target = max(lo, min(hi, float(raw)))
    delta = target - current
    if abs(delta) > settings.max_delta:
        step = settings.max_delta if delta > 0 else -settings.max_delta
        target = current + step
        target = max(lo, min(hi, target))
        note = f"delta capped to ±{settings.max_delta}; applied {target:.4f}"
    else:
        note = f"applied {target:.4f}"
    return target, note


def validate_energy_scalar(
    raw: float,
    current: float,
    lo: float,
    hi: float,
    max_delta: float,
) -> tuple[float, str]:
    """Clamp to [lo, hi], limit step from current by max_delta."""
    target = max(lo, min(hi, float(raw)))
    delta = target - current
    if abs(delta) > max_delta:
        step = max_delta if delta > 0 else -max_delta
        target = current + step
        target = max(lo, min(hi, target))
        note = f"delta capped to ±{max_delta}; applied {target:.4f}"
    else:
        note = f"applied {target:.4f}"
    return target, note


def fix_energy_hysteresis(cfg: Dict[str, Any], settings: AgentSettings) -> None:
    """Ensure start_energy - end_energy >= energy_min_gap (mutates cfg). Call under cfg_lock."""
    gap = float(settings.energy_min_gap)
    el, eh = settings.energy_end_clamp_min, settings.energy_end_clamp_max
    sl, sh = settings.energy_start_clamp_min, settings.energy_start_clamp_max
    for _ in range(6):
        s = float(cfg["start_energy"])
        e = float(cfg["end_energy"])
        if s - e >= gap - 1e-12:
            return
        target_e = s - gap
        e = max(el, min(eh, min(e, target_e)))
        cfg["end_energy"] = e
        s = float(cfg["start_energy"])
        if s - e >= gap - 1e-12:
            return
        target_s = e + gap
        s = max(sl, min(sh, max(s, target_s)))
        cfg["start_energy"] = s


def apply_tool_call(
    name: str,
    arguments_json: str,
    cfg: Dict[str, Any],
    cfg_lock: threading.Lock,
    settings: AgentSettings,
    last_change_mono: List[float],
    last_energy_change_mono: List[float],
) -> str:
    """Execute one tool call; returns a short result string for the LLM."""
    now = time.monotonic()

    if name == "noop":
        try:
            args = json.loads(arguments_json) if arguments_json else {}
        except json.JSONDecodeError:
            args = {}
        reason = str(args.get("reason", ""))[:200]
        return f"noop acknowledged: {reason}"

    if name == "set_min_confidence":
        with cfg_lock:
            cur = float(cfg["min_confidence"])
        try:
            args = json.loads(arguments_json)
            val = float(args.get("value"))
            reason = str(args.get("reason", ""))[:300]
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            return f"invalid arguments: {e}"

        if last_change_mono and (now - last_change_mono[0]) < settings.cooldown_sec:
            return (
                f"cooldown active ({settings.cooldown_sec}s); no change. "
                f"current min_confidence={cur:.4f}"
            )

        new_val, note = validate_and_apply_min_confidence(val, cur, settings)
        with cfg_lock:
            old = float(cfg["min_confidence"])
            cfg["min_confidence"] = new_val
            final = float(cfg["min_confidence"])

        last_change_mono.clear()
        last_change_mono.append(now)
        summary = {
            "old_min_confidence": old,
            "new_min_confidence": final,
            "reason": reason,
            "validation": note,
        }
        print(f"[tuning_agent] {json.dumps(summary)}")
        return json.dumps(summary)

    if name in ("set_start_energy", "set_end_energy"):
        try:
            args = json.loads(arguments_json)
            val = float(args.get("value"))
            reason = str(args.get("reason", ""))[:300]
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            return f"invalid arguments: {e}"

        if last_energy_change_mono and (
            now - last_energy_change_mono[0] < settings.energy_cooldown_sec
        ):
            with cfg_lock:
                s0 = float(cfg["start_energy"])
                e0 = float(cfg["end_energy"])
            return (
                f"energy cooldown active ({settings.energy_cooldown_sec}s); no change. "
                f"start_energy={s0:.4f} end_energy={e0:.4f}"
            )

        with cfg_lock:
            cur_s = float(cfg["start_energy"])
            cur_e = float(cfg["end_energy"])

        if name == "set_start_energy":
            lo, hi = settings.energy_start_clamp_min, settings.energy_start_clamp_max
            new_s, note = validate_energy_scalar(val, cur_s, lo, hi, settings.energy_max_delta)
            with cfg_lock:
                cfg["start_energy"] = new_s
                fix_energy_hysteresis(cfg, settings)
                final_s = float(cfg["start_energy"])
                final_e = float(cfg["end_energy"])
        else:
            lo, hi = settings.energy_end_clamp_min, settings.energy_end_clamp_max
            new_e, note = validate_energy_scalar(val, cur_e, lo, hi, settings.energy_max_delta)
            with cfg_lock:
                cfg["end_energy"] = new_e
                fix_energy_hysteresis(cfg, settings)
                final_s = float(cfg["start_energy"])
                final_e = float(cfg["end_energy"])

        last_energy_change_mono.clear()
        last_energy_change_mono.append(now)
        summary = {
            "tool": name,
            "start_energy": final_s,
            "end_energy": final_e,
            "reason": reason,
            "validation": note,
        }
        print(f"[tuning_agent] {json.dumps(summary)}")
        return json.dumps(summary)

    return f"unknown tool {name}"


def _drain_queue(q: queue.Queue) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            break
    return out


def _output_item_type(item: Any) -> Optional[str]:
    if isinstance(item, dict):
        return item.get("type")
    return getattr(item, "type", None)


def _function_call_fields(item: Any) -> tuple[str, str, Optional[str]]:
    """Returns (name, arguments_json, call_id)."""
    if isinstance(item, dict):
        return (
            str(item.get("name") or ""),
            str(item.get("arguments") or "{}"),
            item.get("call_id") or item.get("id"),
        )
    name = getattr(item, "name", None) or ""
    args = getattr(item, "arguments", None)
    if args is not None and not isinstance(args, str):
        args = json.dumps(args)
    cid = getattr(item, "call_id", None) or getattr(item, "id", None)
    return (str(name), str(args or "{}"), cid)


def _safe_put(q: queue.Queue, item: Dict[str, Any]) -> None:
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass


def run_agent_loop(
    metric_queue: queue.Queue,
    cfg: Dict[str, Any],
    cfg_lock: threading.Lock,
    stop_event: threading.Event,
    settings: AgentSettings,
) -> None:
    """
    Background loop: on interval, drain recent segment metrics, call LLM, apply tools.
    """
    if OpenAI is None:
        print("tuning_agent: openai package not installed; agent thread exiting.")
        return

    api_key = settings.api_key or ""
    kwargs: Dict[str, Any] = {"api_key": api_key}
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    client = OpenAI(**kwargs)

    last_change: List[float] = []
    last_energy_change: List[float] = []
    tools = tool_definitions()
    system_prompt = (
        "You tune a wrist IMU gesture live pipeline: min_confidence, start_energy, end_energy. "
        "Each recent_segments item includes: segment_frames (motion segment length in frames); "
        "max_energy and max_energy_above_start (peak gyro energy minus start threshold — margin into segment); "
        "fired (any gesture latched) and fire_count (UDP fires in that segment); "
        "max_max_prob, max_prob_above_min (peak prob minus min_confidence — headroom vs threshold); "
        "best_non_idle; frames_high_conf_non_idle (pre-first-fire frames with trigger class above min_conf); "
        "idle_high_conf_frames_before_segment (spurious trigger-class frames while idle, energy at/below start — "
        "high values suggest raising min_confidence or start_energy); "
        "thresholds_at_segment_end (snapshot of key thresholds when the segment closed). "
        "current_config is the live cfg now (may differ slightly from thresholds_at_segment_end in old rows). "
        "Use set_min_confidence when max_prob_above_min / frames_high_conf_non_idle / idle_high_conf_* indicate "
        "threshold mismatch. Use set_start_energy when max_energy_above_start is tiny (weak peaks) or idle_high_conf "
        "is high. Use set_end_energy when segment_frames are extreme (very short vs very long vs intent). "
        "If ambiguous, noop. Call exactly one tool per turn."
    )

    while not stop_event.is_set():
        stop_event.wait(timeout=settings.interval_sec)
        if stop_event.is_set():
            break

        batch = _drain_queue(metric_queue)
        if not batch:
            continue

        with cfg_lock:
            snap = {
                "min_confidence": float(cfg["min_confidence"]),
                "start_energy": float(cfg["start_energy"]),
                "end_energy": float(cfg["end_energy"]),
            }
        user_payload = {
            "recent_segments": batch[-16:],
            "current_config": snap,
        }
        input_user: List[Dict[str, Any]] = [
            {"role": "user", "content": json.dumps(user_payload, indent=0)},
        ]

        try:
            resp = client.responses.create(
                model=settings.model,
                instructions=system_prompt,
                tools=tools,
                input=input_user,
                tool_choice="auto",
            )
        except Exception as e:
            print(f"[tuning_agent] API error: {e!r}")
            continue

        out = getattr(resp, "output", None) or []
        function_calls = [item for item in out if _output_item_type(item) == "function_call"]

        if not function_calls:
            text = getattr(resp, "output_text", None) or ""
            if text:
                print(f"[tuning_agent] model text: {text[:500]}")
            continue

        input_list: List[Any] = list(input_user) + list(out)
        n_outputs = 0
        for item in function_calls:
            name, args_json, call_id = _function_call_fields(item)
            if not call_id:
                print(f"[tuning_agent] skipping tool {name!r}: missing call_id")
                continue
            result = apply_tool_call(
                name,
                args_json,
                cfg,
                cfg_lock,
                settings,
                last_change,
                last_energy_change,
            )
            input_list.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": result,
                }
            )
            n_outputs += 1

        if n_outputs == 0:
            continue

        try:
            resp2 = client.responses.create(
                model=settings.model,
                instructions=system_prompt,
                tools=tools,
                input=input_list,
            )
            text2 = getattr(resp2, "output_text", None) or ""
            if text2.strip():
                print(f"[tuning_agent] {text2[:500]}")
        except Exception as e:
            print(f"[tuning_agent] follow-up API error: {e!r}")


def enqueue_segment_metric(q: queue.Queue, record: Dict[str, Any]) -> None:
    _safe_put(q, record)

