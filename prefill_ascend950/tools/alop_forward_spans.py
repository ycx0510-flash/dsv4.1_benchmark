#!/usr/bin/env python3
"""Summarize ALOP/aclprof task_time CSV as per-forward device spans.

ALOP records every kernel with start/stop timestamps.  A forward with fixed
shapes enqueues the same number of tasks each time, so the task list can be
split into equal groups after dropping the one ``TensorMove`` preamble task
that aclprof emits before the first forward.  This is the same per-forward
span definition used for the torch traces: ``max(stop) - min(start)`` inside
one forward, excluding host gaps between forwards.

Usage:
    python benchmarks/prefill_ascend950/tools/alop_forward_spans.py \
        results/<run>/cases/<case> --warmup 1
    python benchmarks/prefill_ascend950/tools/alop_forward_spans.py \
        results/<run> --rank 0 --json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path


def pick_csv(path: Path, rank: int) -> Path:
    if path.is_file():
        return path
    pattern = f"**/rank_{rank}/profiler/**/task_time_*.csv"
    found = sorted(path.glob(pattern))
    if not found:
        found = sorted(path.glob("**/task_time_*.csv"))
    if not found:
        raise SystemExit(f"no task_time_*.csv under {path}")
    # Multiple cases/ranks: take the most recently modified one.
    return max(found, key=lambda p: p.stat().st_mtime)


def configured_forwards(csv_path: Path) -> int | None:
    for parent in csv_path.parents:
        config = parent / "resolved_config.json"
        if config.is_file():
            try:
                profile = json.loads(config.read_text())["profile"]
                return int(profile["wait_steps"]) + int(profile["warmup_steps"]) + int(profile["active_steps"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                return None
    return None


def read_tasks(csv_path: Path) -> list[tuple[float, float]]:
    tasks: list[tuple[float, float]] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            try:
                start = float(row["task_start(us)"])
                stop = float(row["task_stop(us)"])
            except (KeyError, TypeError, ValueError):
                continue
            tasks.append((start, stop))
    if not tasks:
        raise SystemExit(f"no task rows in {csv_path}")
    tasks.sort()
    return tasks


def summarize(tasks: list[tuple[float, float]], forwards: int) -> list[dict]:
    # aclprof emits one isolated TensorMove before the first forward.  Drop it
    # before dividing; otherwise a single-forward run absorbs the 17ms gap.
    preamble = 1 if len(tasks) > 1 and tasks[1][0] - tasks[0][1] > 1000 else 0
    body = tasks[preamble:]
    per_group = round(len(body) / forwards)
    if per_group < 10:
        raise SystemExit(f"{len(body)} tasks cannot be split into {forwards} forwards")
    remainder = len(body) - per_group * forwards
    if remainder < 0 or remainder > forwards:
        raise SystemExit(
            f"task count {len(body)} is not {forwards} equal groups; pass the correct --forwards"
        )
    body = body[remainder:]
    rows = []
    for index in range(forwards):
        group = body[index * per_group : (index + 1) * per_group]
        start = min(item[0] for item in group)
        stop = max(item[1] for item in group)
        rows.append(
            {
                "forward": index,
                "tasks": len(group),
                "span_ms": round((stop - start) / 1000.0, 3),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="run dir, case dir, rank dir, or task_time CSV")
    parser.add_argument("--rank", type=int, default=0, help="rank to read when path is a run/case dir")
    parser.add_argument("--forwards", type=int, default=None, help="forward groups (default: profile wait+warmup+active)")
    parser.add_argument("--warmup", type=int, default=0, help="leading forwards excluded from the mean")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args()

    csv_path = pick_csv(args.path.resolve(), args.rank)
    forwards = args.forwards or configured_forwards(csv_path)
    if not forwards:
        raise SystemExit("cannot infer forward count; pass --forwards")
    rows = summarize(read_tasks(csv_path), forwards)
    active = rows[args.warmup :]
    mean = sum(row["span_ms"] for row in active) / len(active) if active else 0.0
    result = {
        "task_time_csv": str(csv_path),
        "forwards": rows,
        "warmup": args.warmup,
        "active_mean_span_ms": round(mean, 3),
    }
    if args.json:
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return
    print(f"file: {csv_path}")
    print(f"{'forward':>7} {'tasks':>7} {'span_ms':>10}")
    for row in rows:
        marker = " (warmup)" if row["forward"] < args.warmup else ""
        print(f"{row['forward']:>7} {row['tasks']:>7} {row['span_ms']:>10.3f}{marker}")
    print(f"active mean span: {mean:.3f} ms over {len(active)} forward(s)")


if __name__ == "__main__":
    main()
