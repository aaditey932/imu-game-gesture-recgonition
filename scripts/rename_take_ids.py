"""
Rewrite take_id in gesture CSVs to match collect_data.py: one take_id per chunk of N
consecutive windows within each recording run.

A "run" is consecutive rows sharing the same (label, timestamp_iso, old take_id).
Within each run, row index i (0-based) gets take_id = next_global_id + floor(i / N).

Keep in sync with TAKE_ID_EVERY_N_WINDOWS in collect_data.py (default 20).
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

from imugesture.paths import DATA_CSV  # noqa: E402

# Default chunk size — must match collect_data.TAKE_ID_EVERY_N_WINDOWS
DEFAULT_CHUNK = 20


def iter_run_groups(rows: list[list[str]], header: list[str]):
    """Split data rows into consecutive runs with same (label, timestamp_iso, take_id)."""
    i_label = header.index("label")
    i_ts = header.index("timestamp_iso")
    i_take = header.index("take_id")

    current: list[list[str]] | None = None
    key: tuple[str, str, str] | None = None

    for row in rows:
        if not row:
            continue
        k = (row[i_label], row[i_ts], row[i_take])
        if key is None or k != key:
            if current is not None:
                yield current
            current = [row]
            key = k
        else:
            current.append(row)
    if current is not None:
        yield current


def remap_take_ids(
    rows: list[list[str]],
    header: list[str],
    chunk: int,
    start_id: int,
) -> tuple[list[list[str]], int, int]:
    """
    Returns (new_data_rows, next_id_after_file, n_runs_processed).
    """
    if "take_id" not in header or "timestamp_iso" not in header:
        raise ValueError("CSV must include timestamp_iso and take_id columns.")

    i_take = header.index("take_id")
    data_rows = [r for r in rows if r]
    groups = list(iter_run_groups(data_rows, header))

    next_id = start_id
    out: list[list[str]] = []
    for group in groups:
        for i, row in enumerate(group):
            row = list(row)
            row[i_take] = str(next_id + (i // chunk))
            out.append(row)
        next_id += (len(group) + chunk - 1) // chunk

    return out, next_id, len(groups)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Assign new take_id every N windows per recording run (matches collect_data.py)."
    )
    p.add_argument(
        "csv_files",
        nargs="*",
        type=Path,
        help=f"CSV files to rewrite (default: {DATA_CSV})",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK,
        help=f"Windows per take_id (default: {DEFAULT_CHUNK}, same as TAKE_ID_EVERY_N_WINDOWS).",
    )
    p.add_argument(
        "--start-id",
        type=int,
        default=1,
        help="First take_id to assign in the first file (default: 1).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary only; do not write files.",
    )
    p.add_argument(
        "--backup",
        action="store_true",
        help="Before overwriting, copy each input to <name>.bak",
    )
    args = p.parse_args()

    paths = args.csv_files if args.csv_files else [DATA_CSV]
    chunk = max(1, args.chunk_size)
    next_global = args.start_id

    for path in paths:
        path = path.resolve()
        if not path.is_file():
            print(f"Skip (not a file): {path}", file=sys.stderr)
            continue

        with open(path, newline="") as f:
            reader = csv.reader(f)
            all_rows = list(reader)
        if not all_rows:
            print(f"Skip (empty): {path}")
            continue

        header, body = all_rows[0], all_rows[1:]
        new_body, next_global, n_runs = remap_take_ids(body, header, chunk, next_global)

        i_take = header.index("take_id")
        if new_body:
            ids = [int(r[i_take]) for r in new_body]
            tid_lo, tid_hi = min(ids), max(ids)
        else:
            tid_lo = tid_hi = None

        print(
            f"{path.name}: {len(body)} rows, {n_runs} run(s), "
            f"take_id {tid_lo}..{tid_hi} (next file starts at {next_global})"
        )

        if args.dry_run:
            continue

        if args.backup:
            bak = path.with_suffix(path.suffix + ".bak")
            shutil.copy2(path, bak)
            print(f"  backup: {bak}")

        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(new_body)

        print(f"  wrote: {path}")


if __name__ == "__main__":
    main()
