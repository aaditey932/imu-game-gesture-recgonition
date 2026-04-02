"""
Remove the first N occurrences of each left-gesture class from a gesture CSV (top-to-bottom order).

Left classes: duck_left, kick_left, punch_left. Other rows are kept as-is.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Repository root: scripts/ -> parent
REPO_ROOT = Path(__file__).resolve().parents[1]

LEFT_LABELS = frozenset({"duck_left", "kick_left", "punch_left"})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Drop the first N rows per left gesture class, reading from the top of the CSV."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data" / "gesture_data.csv",
        help="Input CSV path (default: data/gesture_data.csv).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: print summary only; use -o to write).",
    )
    p.add_argument(
        "-n",
        type=int,
        default=200,
        metavar="N",
        help="Number of leading rows to remove per left class (default: 200).",
    )
    p.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite --input (writes to a temp file then replaces). Implies --output is ignored.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n_drop = max(0, args.n)
    path_in = args.input

    if not path_in.is_file():
        print(f"Input not found: {path_in}", file=sys.stderr)
        sys.exit(1)

    if args.in_place:
        path_out = path_in.with_suffix(path_in.suffix + ".tmp")
    else:
        path_out = args.output

    skipped = {label: 0 for label in LEFT_LABELS}
    kept_left = {label: 0 for label in LEFT_LABELS}
    kept_other = 0

    with open(path_in, newline="") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames
        if not fieldnames or fieldnames[0] != "label":
            print("Expected CSV with 'label' as first column.", file=sys.stderr)
            sys.exit(1)

        rows_out: list[dict[str, str]] = []
        for row in reader:
            label = (row.get("label") or "").strip()
            if label in LEFT_LABELS:
                if skipped[label] < n_drop:
                    skipped[label] += 1
                    continue
                kept_left[label] += 1
            else:
                kept_other += 1
            rows_out.append(row)

    summary_lines = [
        f"Removed first {n_drop} per left class (when available):",
        f"  duck_left:  skipped {skipped['duck_left']}, kept {kept_left['duck_left']}",
        f"  kick_left:  skipped {skipped['kick_left']}, kept {kept_left['kick_left']}",
        f"  punch_left: skipped {skipped['punch_left']}, kept {kept_left['punch_left']}",
        f"Other rows (unchanged): {kept_other}",
        f"Total output rows: {len(rows_out)}",
    ]
    for line in summary_lines:
        print(line)

    if path_out is None:
        print("(No file written; pass --output PATH or --in-place.)")
        return

    path_out.parent.mkdir(parents=True, exist_ok=True)
    with open(path_out, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    if args.in_place:
        path_out.replace(path_in)
        print(f"Wrote in-place: {path_in}")
    else:
        print(f"Wrote: {path_out}")


if __name__ == "__main__":
    main()
