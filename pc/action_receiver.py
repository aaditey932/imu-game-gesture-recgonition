#!/usr/bin/env python3
"""
Run this script on your gaming PC (not on the Raspberry Pi).

It listens for UDP JSON packets from run_live.py and simulates keypresses locally
via pynput, so Steam/games see normal keyboard input without X11 on the Pi.

Default key map matches **Punch Kick Duck** keyboard controls:
  punch_left → W, kick_left → S, duck_left → X,
  punch_right → O, kick_right → K, duck_right → M.

Usage (from repo root):
  pip install -r pc/requirements.txt
  python3 pc/action_receiver.py
  python3 pc/action_receiver.py --port 5005 --kick-left a

Firewall: allow inbound UDP on the chosen port from your Pi's IP.

On the Pi, run_live.py defaults to THIS_PC_LAN_IP / DEFAULT_UDP_PORT below, or override:
  export ACTION_RECEIVER_HOST=...
  export ACTION_RECEIVER_PORT=5005
  python3 run_live.py

Payload format (one JSON object per datagram):
  {"action": "<gesture>", "confidence": 0.0-1.0}

  <gesture> is the lowercase class name from the Pi: punch_left, kick_left,
  duck_left, punch_right, kick_right, duck_right.
"""

import argparse
import json
import socket
import sys

# This PC's LAN IPv4 (set from machine when wiring the Pi). Update if you change Wi‑Fi/network.
THIS_PC_LAN_IP = "192.168.1.218"
DEFAULT_UDP_PORT = 5005

# Punch Kick Duck default keyboard (Controls screen).
DEFAULT_ACTION_KEYS = {
    "punch_left": "w",
    "kick_left": "s",
    "duck_left": "x",
    "punch_right": "o",
    "kick_right": "k",
    "duck_right": "m",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UDP receiver: gesture class names (from run_live.py) -> key taps"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Address to bind (default: all interfaces)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_UDP_PORT,
        help="UDP port (must match DEFAULT_ACTION_RECEIVER_PORT / ACTION_RECEIVER_PORT on the Pi)",
    )
    for name, default in DEFAULT_ACTION_KEYS.items():
        arg = name.replace("_", "-")
        parser.add_argument(
            f"--{arg}",
            default=default,
            metavar="KEY",
            help=f"Key for action '{name}' (default: {default!r})",
        )
    args = parser.parse_args()

    action_keys = {
        "punch_left": args.punch_left,
        "kick_left": args.kick_left,
        "duck_left": args.duck_left,
        "punch_right": args.punch_right,
        "kick_right": args.kick_right,
        "duck_right": args.duck_right,
    }

    try:
        from pynput.keyboard import Controller
    except ImportError as e:
        print("pynput is required on the PC: pip install pynput", file=sys.stderr)
        raise SystemExit(1) from e

    keyboard = Controller()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))
    print(
        f"Listening on UDP {args.host}:{args.port}; Pi default target {THIS_PC_LAN_IP}:{args.port} "
        f"(Ctrl+C to quit)"
    )
    print(f"Key map: {action_keys}")

    while True:
        data, addr = sock.recvfrom(4096)
        try:
            obj = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Bad JSON from {addr}: {e!r}")
            continue
        if not isinstance(obj, dict):
            print(f"Expected JSON object from {addr}, got {type(obj).__name__}")
            continue
        action = obj.get("action")
        if not isinstance(action, str):
            print(f"Missing or invalid 'action' from {addr}: {obj!r}")
            continue
        action = action.lower().strip()
        confidence = obj.get("confidence", 1.0)
        print(f"from {addr}: action={action!r} confidence={confidence!r}")

        key = action_keys.get(action)
        if key is None:
            print(f"Unknown action {action!r} (expected {list(action_keys)})")
            continue
        try:
            keyboard.tap(key)
        except Exception as e:
            print(f"Key tap failed for {action!r}: {e!r}")


if __name__ == "__main__":
    main()
