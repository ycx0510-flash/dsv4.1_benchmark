#!/usr/bin/env python3
"""Check whether an ALOP capture is complete enough for MindStudio Insight.

A task-time-only ALOP DB (``alop_aicore_metrics=false``) is valid for
``task_time``/forward-span analysis, but it has no ``TASK_PMU_INFO`` and its
``op_summary`` has only the base columns.  MindStudio's unified DB import expects
the 21-table schema / 48-column op summary produced with AICore metrics enabled.

Usage:
    python benchmarks/prefill_ascend950/tools/check_alop_profile.py \
        results/<run> --rank 0
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

REQUIRED_TABLES = ("TASK", "TASK_PMU_INFO", "COMPUTE_TASK_INFO", "AICORE_FREQ", "CANN_API", "META_DATA")


def find_db(path: Path, rank: int) -> Path:
    if path.is_file() and path.suffix == ".db":
        return path
    patterns = [
        f"**/rank_{rank}/profiler/PROF_*/msprof_*.db",
        f"**/rank_{rank}/profiler/**/msprof_*.db",
        "**/msprof_*.db",
    ]
    if path.is_dir() and path.name.startswith("PROF_"):
        patterns.insert(0, "msprof_*.db")
    for pattern in patterns:
        found = sorted(path.glob(pattern))
        if found:
            return max(found, key=lambda item: item.stat().st_mtime)
    raise SystemExit(f"no msprof_*.db under {path}")


def find_op_summary(db: Path) -> Path | None:
    output = db.parent / "mindstudio_profiler_output"
    found = sorted(output.glob("op_summary_*.csv")) if output.is_dir() else []
    return max(found, key=lambda item: item.stat().st_mtime) if found else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="run/case/rank/PROF dir or msprof DB")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    db = find_db(args.path.resolve(), args.rank)
    connection = sqlite3.connect(db)
    integrity = connection.execute("pragma integrity_check").fetchone()[0]
    tables = [row[0] for row in connection.execute("select name from sqlite_master where type='table' order by name")]
    counts = {}
    for table in REQUIRED_TABLES:
        if table in tables:
            counts[table] = connection.execute(f"select count(*) from {table}").fetchone()[0]
    missing = [table for table in REQUIRED_TABLES if table not in tables]
    op_summary = find_op_summary(db)
    columns = 0
    if op_summary is not None:
        with op_summary.open(newline="", encoding="utf-8-sig") as handle:
            header = next(csv.reader(handle), [])
        columns = len(header)
    mindstudio_ready = not missing and columns >= 48 and integrity == "ok"
    result = {
        "db": str(db),
        "db_bytes": db.stat().st_size,
        "integrity": integrity,
        "tables": len(tables),
        "missing_tables": missing,
        "table_counts": counts,
        "op_summary": str(op_summary) if op_summary else None,
        "op_summary_columns": columns,
        "mindstudio_db_ready": mindstudio_ready,
        "next": (
            "import the single .db file into MindStudio Insight"
            if mindstudio_ready
            else "re-collect with profile.alop_aicore_metrics=true; do not import this DB"
        ),
    }
    if args.json:
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(f"db: {db}")
        print(f"size: {db.stat().st_size} bytes, integrity={integrity}")
        print(f"tables: {len(tables)}, missing required: {missing or 'none'}")
        for table, count in counts.items():
            print(f"  {table}: {count}")
        print(f"op_summary: {op_summary}")
        print(f"op_summary columns: {columns}")
        print(f"MindStudio DB ready: {mindstudio_ready}")
        print(f"next: {result['next']}")
    raise SystemExit(0 if mindstudio_ready else 1)


if __name__ == "__main__":
    main()
