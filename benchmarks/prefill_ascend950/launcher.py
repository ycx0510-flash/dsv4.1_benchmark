"""Isolated eight-rank cases with fail-fast cleanup and resumable artifacts."""

from __future__ import annotations

import json
import multiprocessing
import os
import socket
import time
from multiprocessing.process import BaseProcess
from pathlib import Path

from .config import digest
from .doctor import check
from .io_utils import write_json, write_jsonl
from .planner import plan, rank_mapping
from .report import aggregate, summarize


def rendezvous() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"tcp://127.0.0.1:{sock.getsockname()[1]}"


def case_directory(output: Path, case: dict) -> Path:
    """Readable case directory: <strategy>_L<length>_h<hit>_<UTC timestamp>."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    name = f"{case['strategy']}_L{case['context_length']}_h{case['cache_hit_rate_requested']}_{stamp}"
    path = output / "cases" / name
    if path.exists():
        path = output / "cases" / f"{name}_{case['case_id'][:8]}"
    return path


def find_case_directory(output: Path, case_id: str) -> Path | None:
    """Resume lookup by case_id; directory names are no longer the case_id."""
    for path in sorted((output / "cases").glob("*/case.json")):
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("case_id") == case_id:
                return path.parent
        except (OSError, ValueError):
            continue
    return None


def terminate(processes: list[BaseProcess]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + 5
    for process in processes:
        process.join(timeout=max(0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.kill()
        process.join(timeout=5)


def execute_case(cfg: dict, case: dict, devices: list[int], model_dir: Path, directory: Path) -> dict:
    from .worker import entry

    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "case.json", case)
    if case["status"] != "PLANNED":
        result = dict(case, timing_ok=False)
        write_json(directory / "result.json", result)
        return result
    mapping = rank_mapping(case, devices)
    write_json(directory / "rank_mapping.json", mapping)
    init = rendezvous()
    processes = []
    started = time.monotonic()
    status, error = None, None
    try:
        context = multiprocessing.get_context("spawn")
        for rank in mapping:
            process = context.Process(
                target=entry, args=(cfg, case, rank, str(model_dir), init, str(directory)), name=f"prefill950-rank{rank['global_rank']}"
            )
            process.start()
            processes.append(process)
        while any(p.is_alive() for p in processes):
            failures = [p for p in processes if p.exitcode not in (None, 0)]
            if failures:
                status, error = "FAILED", f"Worker failure: {[(p.name, p.exitcode) for p in failures]}"
                break
            if time.monotonic() - started > cfg["runtime"]["case_timeout_s"]:
                status, error = "TIMEOUT", "Case exceeded case_timeout_s (including model load)"
                break
            time.sleep(0.25)
        if status is None and any(p.exitcode != 0 for p in processes):
            status, error = "FAILED", "One or more rank processes exited unsuccessfully"
    except BaseException as exc:
        status, error = "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED", str(exc)
        if isinstance(exc, KeyboardInterrupt):
            raise
    finally:
        terminate(processes)
        if status is not None:
            errors = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("rank_*/error.json"))]
            if any(e.get("status") == "OOM" for e in errors):
                status = "OOM"
            elif any(e.get("status") == "PATH_MISMATCH" for e in errors):
                status = "PATH_MISMATCH"
            write_json(directory / "result.json", dict(case, status=status, timing_ok=False, error=error, rank_errors=errors))
    if status is not None:
        return json.loads((directory / "result.json").read_text(encoding="utf-8"))
    try:
        ranks = [json.loads((directory / f"rank_{r}" / "outcome.json").read_text(encoding="utf-8")) for r in range(8)]
        result = aggregate(case, cfg, ranks)
    except (ValueError, OSError, KeyError) as exc:
        result = dict(case, status="INCOMPLETE", timing_ok=False, error=str(exc))
    result["case_wall_seconds"] = time.monotonic() - started
    write_json(directory / "result.json", result)
    return result


def run(cfg: dict, devices: list[int], model_dir: Path, output: Path, resume: bool = False) -> dict:
    output = output.resolve()
    if output.exists() and not resume:
        raise ValueError("Output directory already exists; choose a new directory or use --resume")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".running.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(fd, f"{os.getpid()}\n".encode())
        old_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8")) if (output / "manifest.json").exists() else None
        doctor = check(cfg, devices, model_dir, output / "doctor")
        if not doctor["ok"]:
            raise ValueError("950 preflight failed: " + "; ".join(doctor["errors"]))
        manifest = {k: doctor[k] for k in ("environment_id", "native_source_id", "model_id")}
        manifest.update(config_hash=digest(cfg), devices=devices, schema_version=1)
        if old_manifest is not None and old_manifest != manifest:
            raise ValueError("Resume identity mismatch: configuration, environment, native source or model changed")
        write_json(output / "manifest.json", manifest)
        write_json(output / "resolved_config.json", cfg)
        cases = plan(cfg)
        write_jsonl(output / "cases.jsonl", cases)
        failed = 0
        for number, case in enumerate(cases, 1):
            existing = find_case_directory(output, case["case_id"]) if resume else None
            if existing is not None:
                result_path = existing / "result.json"
                if result_path.exists() and json.loads(result_path.read_text(encoding="utf-8")).get("status") == "OK":
                    print(f"[{number}/{len(cases)}] {existing.name} already OK", flush=True)
                    continue
                # Preserve failed attempts; never mix stale rank outcomes with a new run.
                previous = output / "attempts" / f"{existing.name}-{time.time_ns()}"
                previous.parent.mkdir(parents=True, exist_ok=True)
                existing.rename(previous)
            directory = case_directory(output, case)
            print(f"[{number}/{len(cases)}] {directory.name}", flush=True)
            result = execute_case(cfg, case, devices, model_dir.resolve(), directory)
            failed += result["status"] != "OK"
            print(f"  {result['status']} {result.get('forward_p50_ms', '')}", flush=True)
        summary = summarize(output)
        return {"ok": failed == 0, "output": str(output), **summary}
    finally:
        os.close(fd)
        lock.unlink(missing_ok=True)
