"""
Append precomputed v2 feature columns to gesture CSVs (one column per FEATURE_NAMES).

Training/inference still compute features from IMU+flex columns; extra columns are optional
for inspection, export, or external tools. Re-running overwrites feature columns if present.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

_repo_src = Path(__file__).resolve().parents[1] / "src"
if _repo_src.is_dir():
    sys.path.insert(0, str(_repo_src))

from imugesture.features import FEATURE_NAMES, window_to_features

WINDOW_SIZE = 15
N_FLEX_CHANNELS = 2
FEATURE_SET = set(FEATURE_NAMES)


def row_dict_to_window(row: dict) -> list:
    """Same parsing as scripts/train.load_csv."""
    window = []
    for i in range(WINDOW_SIZE):
        ax = float(row.get(f"ax_{i}", 0))
        ay = float(row.get(f"ay_{i}", 0))
        az = float(row.get(f"az_{i}", 0))
        gx = float(row.get(f"gx_{i}", 0))
        gy = float(row.get(f"gy_{i}", 0))
        gz = float(row.get(f"gz_{i}", 0))
        flex = []
        for j in range(N_FLEX_CHANNELS):
            key = f"f{j}_{i}"
            v = row.get(key, "")
            flex.append(float(v) if (v is not None and str(v).strip() != "") else 0.0)
        window.append((ax, ay, az, gx, gy, gz) + tuple(flex))
    return window


def strip_feature_columns(fieldnames: list[str] | None) -> list[str]:
    if not fieldnames:
        return []
    return [k for k in fieldnames if k not in FEATURE_SET]


def process_file(path: Path, dry_run: bool, backup: bool) -> int:
    path = path.resolve()
    if not path.is_file():
        print(f"Skip (missing): {path}", file=sys.stderr)
        return 0

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        orig = reader.fieldnames
        if not orig or orig[0] != "label":
            print(f"Skip (bad header): {path}", file=sys.stderr)
            return 0
        base_fields = strip_feature_columns(list(orig))
        rows_in = list(reader)

    n = 0
    out_rows = []
    for row in rows_in:
        if not row or not str(row.get("label", "")).strip():
            continue
        w = row_dict_to_window(row)
        feats = window_to_features(w)
        # Keep only base keys so we do not duplicate stale feature keys
        slim = {k: row.get(k, "") for k in base_fields}
        for name, val in zip(FEATURE_NAMES, feats):
            slim[name] = val
        out_rows.append(slim)
        n += 1

    out_fieldnames = base_fields + list(FEATURE_NAMES)

    print(f"{path.name}: {n} rows, +{len(FEATURE_NAMES)} feature columns (schema v2)")

    if dry_run:
        return n

    if backup:
        bak = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, bak)
        print(f"  backup: {bak}")

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)

    print(f"  wrote: {path}")
    return n


def main() -> None:
    p = argparse.ArgumentParser(
        description="Append v2 feature columns to gesture_data-style CSV files."
    )
    p.add_argument(
        "csv_files",
        nargs="*",
        type=Path,
        help="CSV paths (default: data/gesture_data.csv and data/old_gesture_data.csv)",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--backup", action="store_true", help="Copy original to .bak before write")
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    default_paths = [
        repo / "data" / "gesture_data.csv",
        repo / "data" / "old_gesture_data.csv",
    ]
    paths = args.csv_files if args.csv_files else default_paths

    total = 0
    for path in paths:
        total += process_file(path, args.dry_run, args.backup)
    print(f"Done. Total rows processed: {total}")


if __name__ == "__main__":
    main()
