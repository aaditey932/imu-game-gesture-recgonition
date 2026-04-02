"""
Rewrite gesture CSV so each take_id labels exactly N consecutive rows (file order).

Use after older collect_data runs that reset take_id on each process start, which stacked
many rows under the same id. New collection resumes max(take_id) — see collect_data.py.

Example:
  python scripts/normalize_take_ids.py
  python scripts/normalize_take_ids.py --csv data/gesture_data.csv --dry-run
"""

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

_repo_src = Path(__file__).resolve().parents[1] / "src"
if _repo_src.is_dir():
    sys.path.insert(0, str(_repo_src))

from imugesture.paths import DATA_CSV

DEFAULT_CHUNK = 20


def main():
    p = argparse.ArgumentParser(
        description="Reassign take_id in file order: rows 1–N share id 1, next N share id 2, …"
    )
    p.add_argument("--csv", type=str, default=str(DATA_CSV), help="Path to CSV")
    p.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK,
        metavar="N",
        help=f"Rows per take_id (default {DEFAULT_CHUNK}, match TAKE_ID_EVERY_N_WINDOWS in collect_data.py).",
    )
    p.add_argument(
        "--keep-incomplete",
        action="store_true",
        help="Keep the final partial block (<N rows) as its own take_id instead of dropping it.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts only; do not write.",
    )
    args = p.parse_args()

    if args.chunk_size < 1:
        print("chunk-size must be >= 1.")
        return 1

    path = Path(args.csv)
    if not path.is_file():
        print(f"Not found: {path}")
        return 1

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            print("Empty CSV.")
            return 1
        if "take_id" not in fieldnames:
            print("CSV has no take_id column.")
            return 1
        rows = list(reader)

    n = len(rows)
    full_blocks = n // args.chunk_size
    remainder = n % args.chunk_size
    drop_tail = remainder > 0 and not args.keep_incomplete
    kept = n if not drop_tail else full_blocks * args.chunk_size

    print(f"Rows: {n}, chunk_size={args.chunk_size}")
    print(f"Full blocks: {full_blocks}, remainder: {remainder}")
    if drop_tail:
        print(f"Dropping last {remainder} row(s) so every take_id has exactly {args.chunk_size} rows.")
    elif remainder and args.keep_incomplete:
        print(f"Last take_id will have {remainder} row(s).")

    if args.dry_run:
        new_ids = (full_blocks + (1 if remainder and args.keep_incomplete else 0)) if kept else 0
        print(f"Would write {kept} rows with {new_ids} distinct take_id value(s).")
        return 0

    out_rows = []
    for i in range(kept):
        r = dict(rows[i])
        r["take_id"] = str(i // args.chunk_size + 1)
        out_rows.append(r)

    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    print(f"Backup: {backup}")

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows to {path} (take_id 1–{out_rows[-1]['take_id'] if out_rows else 0}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
