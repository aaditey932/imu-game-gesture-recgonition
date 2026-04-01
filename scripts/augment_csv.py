"""
Append augmented rows to gesture_data.csv: sample existing rows per class and add
Gaussian noise to IMU and flex columns (same schema as collect_data / train).
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_repo_src = Path(__file__).resolve().parents[1] / "src"
if _repo_src.is_dir():
    sys.path.insert(0, str(_repo_src))

from imugesture.paths import DATA_CSV

SIGMA_ACC = 0.05
SIGMA_GYRO = 0.02
SIGMA_FLEX = 8.0


def augment_row(parent: dict, fieldnames: list, rng: random.Random) -> dict:
    row = dict(parent)
    for k in fieldnames:
        if k in ("label", "timestamp_iso", "take_id"):
            continue
        if not row.get(k, "").strip() and k != "label":
            continue
        try:
            if k.startswith(("ax_", "ay_", "az_")):
                row[k] = float(row[k]) + rng.gauss(0, SIGMA_ACC)
            elif k.startswith(("gx_", "gy_", "gz_")):
                row[k] = float(row[k]) + rng.gauss(0, SIGMA_GYRO)
            elif k.startswith("f0_") or k.startswith("f1_"):
                v = float(row[k]) + rng.gauss(0, SIGMA_FLEX)
                row[k] = int(round(v))
        except (TypeError, ValueError):
            pass
    row["timestamp_iso"] = datetime.now(timezone.utc).isoformat()
    return row


def parse_args():
    p = argparse.ArgumentParser(
        description="Append augmented gesture rows (noise) to a gesture_data CSV."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help=f"Input CSV (default: {DATA_CSV})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV to append to (default: same as --input)",
    )
    p.add_argument("--per-class", type=int, default=100, metavar="N")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    in_path = args.input or DATA_CSV
    out_path = args.output or in_path
    per = args.per_class

    if not in_path.is_file():
        print(f"Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    with in_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames or fieldnames[0] != "label":
            print("Invalid CSV: expected header starting with label.", file=sys.stderr)
            sys.exit(1)
        rows = list(reader)

    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        lab = (r.get("label") or "").strip()
        if lab:
            by_label[lab].append(r)

    rng = random.Random(args.seed)
    new_rows: list[dict] = []

    for label in sorted(by_label.keys()):
        pool = by_label[label]
        if len(pool) < per:
            print(
                f"Not enough rows for label {label!r}: have {len(pool)}, need {per}.",
                file=sys.stderr,
            )
            sys.exit(1)
        for _ in range(per):
            parent = rng.choice(pool)
            new_rows.append(augment_row(parent, fieldnames, rng))

    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        for r in new_rows:
            writer.writerow(r)

    print(f"Appended {len(new_rows)} rows to {out_path} ({per} per class, {len(by_label)} classes).")


if __name__ == "__main__":
    main()
