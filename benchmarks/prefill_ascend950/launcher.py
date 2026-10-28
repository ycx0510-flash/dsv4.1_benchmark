"""Eight-rank cases with fail-fast cleanup, batch model reuse and resumable artifacts."""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import socket
import sys
import time
from multiprocessing.process import BaseProcess
from pathlib import Path

from .config import digest
from .doctor import check
from .io_utils import write_json, write_jsonl
from .planner import batch_groups, plan, rank_mapping
from .report import aggregate, summarize

POLL_SECONDS = 0.5
STOP_GRACE_SECONDS = 15.0


class Progress:
    """Case progress display: an in-place bar on a TTY, plain lines otherwise."""

    def __init__(self, total: int, stream=None) -> None:
        self.total = total
        self.completed = 0
        self.failed = 0
        self.current = ""
        self.stream = stream or sys.stdout
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.last_len = 0

    @property
    def ok(self) -> int:
        return self.completed - self.failed

    def _bar(self) -> str:
        width = 30
        filled = int(width * self.completed / self.total) if self.total else width
        return "█" * filled + "░" * (width - filled)

    def _render(self) -> None:
        if not self.tty:
            return
        line = f"\r[{self._bar()}] {self.completed}/{self.total} ok={self.ok} failed={self.failed} | {self.current}"
        pad = max(0, self.last_len - len(line))
        self.stream.write(line + " " * pad)
        self.stream.flush()
        self.last_len = len(line)

    def _clear(self) -> None:
        if self.tty and self.last_len:
            self.stream.write("\r" + " " * self.last_len + "\r")
            self.stream.flush()
            self.last_len = 0

    def note(self, text: str) -> None:
        """Print one log line above the bar without losing the bar state."""
        self._clear()
        print(text, file=self.stream, flush=True)
        self._render()

    def label(self, text: str) -> None:
        """Change what the bar reports without printing a line."""
        self.current = text
        self._render()

    def start_case(self, label: str) -> None:
        self.current = label
        if self.tty:
            self._render()
        else:
            print(f"  running: {label}", file=self.stream, flush=True)

    def advance(self, status: str) -> None:
        self.completed += 1
        if status != "OK":
            self.failed += 1
        self._render()

    def finish(self) -> None:
        self._clear()
        if self.tty:
            self.stream.write(f"[{self._bar()}] {self.completed}/{self.total} ok={self.ok} failed={self.failed}\n")
            self.stream.flush()


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


def reset_case_outputs(directory: Path) -> None:
    """Drop stale per-rank artifacts before a (re)start; keep case.json/rank_mapping.json."""
    (directory / "result.json").unlink(missing_ok=True)
    for rank_dir in directory.glob("rank_*"):
        shutil.rmtree(rank_dir, ignore_errors=True)


def rank_errors(directory: Path) -> list[dict]:
    errors = []
    for path in sorted(directory.glob("rank_*/error.json")):
        try:
            errors.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return errors


def case_finished(directory: Path) -> tuple[bool, bool]:
    """(every rank finished, every rank produced a measured outcome).

    A case skipped in-process after a recoverable preparation failure leaves a
    skipped.json on every rank instead of outcome.json; from the launcher's point
    of view the case is finished and must not trigger a model reload.
    """
    outcomes = [(directory / f"rank_{rank}" / "outcome.json").exists() for rank in range(8)]
    skips = [(directory / f"rank_{rank}" / "skipped.json").exists() for rank in range(8)]
    return all(done or skipped for done, skipped in zip(outcomes, skips)), all(outcomes)


def error_result(case: dict, directory: Path, status: str, error: str, wall_seconds: float | None = None) -> dict:
    errors = rank_errors(directory)
    if any(item.get("status") == "OOM" for item in errors):
        status = "OOM"
    elif any(item.get("status") == "PATH_MISMATCH" for item in errors):
        status = "PATH_MISMATCH"
    result = dict(case, status=status, timing_ok=False, error=error, rank_errors=errors)
    if wall_seconds is not None:
        result["case_wall_seconds"] = wall_seconds
    write_json(directory / "result.json", result)
    return result


def aggregate_case(cfg: dict, case: dict, directory: Path, wall_seconds: float | None = None) -> dict:
    try:
        ranks = [json.loads((directory / f"rank_{r}" / "outcome.json").read_text(encoding="utf-8")) for r in range(8)]
        result = aggregate(case, cfg, ranks)
    except (ValueError, OSError, KeyError) as exc:
        result = dict(case, status="INCOMPLETE", timing_ok=False, error=str(exc))
    if wall_seconds is not None:
        result["case_wall_seconds"] = wall_seconds
    write_json(directory / "result.json", result)
    return result


def spawn_ranks(cfg: dict, pending: list[tuple[dict, Path]], mapping: list[dict], model_dir: Path, init: str, group_dir: Path) -> list[BaseProcess]:
    from .worker import entry

    cases = [case for case, _ in pending]
    directories = [str(directory) for _, directory in pending]
    context = multiprocessing.get_context("spawn")
    processes: list[BaseProcess] = []
    try:
        for rank in mapping:
            process = context.Process(
                target=entry,
                args=(cfg, cases, rank, str(model_dir), init, str(group_dir), directories),
                name=f"prefill950-rank{rank['global_rank']}",
            )
            process.start()
            processes.append(process)
    except BaseException:
        terminate(processes)
        raise
    return processes


def execute_case(cfg: dict, case: dict, devices: list[int], model_dir: Path, directory: Path) -> dict:
    """Single-case mode: one model load per case (historical behavior)."""
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "case.json", case)
    if case["status"] != "PLANNED":
        result = dict(case, timing_ok=False)
        write_json(directory / "result.json", result)
        return result
    mapping = rank_mapping(case, devices)
    write_json(directory / "rank_mapping.json", mapping)
    init = rendezvous()
    processes: list[BaseProcess] = []
    started = time.monotonic()
    status, error = None, None
    try:
        processes = spawn_ranks(cfg, [(case, directory)], mapping, model_dir, init, directory)
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
            error_result(case, directory, status, error, time.monotonic() - started)
    if status is not None:
        return json.loads((directory / "result.json").read_text(encoding="utf-8"))
    result = aggregate_case(cfg, case, directory, time.monotonic() - started)
    return result


def watch_group(
    cfg: dict,
    processes: list[BaseProcess],
    pending: list[tuple[dict, Path]],
    group_dir: Path,
    progress: Progress,
) -> dict:
    """Wait for model load, then report completed cases and the first failure.

    Returns one of: done / load_failed / load_timeout / case_failed /
    incomplete / timeout. Only the launcher aggregates case results, so a
    worker can never publish a case that another rank did not finish.
    """
    timeout = cfg["runtime"]["case_timeout_s"]
    ready = [group_dir / f"ready_rank_{rank}.json" for rank in range(8)]
    load_started = time.monotonic()
    while True:
        failures = [p for p in processes if p.exitcode not in (None, 0)]
        if failures:
            return {"phase": "load_failed", "detail": [(p.name, p.exitcode) for p in failures]}
        if all(path.exists() for path in ready):
            break
        if time.monotonic() - load_started > timeout:
            return {"phase": "load_timeout", "detail": "model load exceeded case_timeout_s"}
        time.sleep(POLL_SECONDS)
    progress.note(f"  model loaded in {time.monotonic() - load_started:.1f}s")

    index = 0
    current_started = time.monotonic()
    progress.start_case(pending[index][1].name)
    while True:
        case, directory = pending[index]
        finished, measured = case_finished(directory)
        if finished:
            case_seconds = time.monotonic() - current_started
            if measured:
                result = aggregate_case(cfg, case, directory, case_seconds)
            else:
                # Every rank skipped this case in-process after agreeing that
                # preparation failed before the first collective. The model and
                # its workspaces are still valid, so continue in the same group.
                errors = rank_errors(directory)
                detail = "; ".join(
                    f"rank {item.get('rank')}: {item.get('type', 'Error')}: {str(item.get('error', ''))[:200]}"
                    for item in errors
                ) or "no rank reported a preparation error"
                result = error_result(
                    case,
                    directory,
                    "FAILED",
                    f"Skipped in-process after a recoverable preparation failure: {detail}",
                    case_seconds,
                )
            index += 1
            if index == len(pending):
                deadline = time.monotonic() + STOP_GRACE_SECONDS
                while time.monotonic() < deadline and any(p.is_alive() for p in processes):
                    time.sleep(POLL_SECONDS)
                return {"phase": "done", "aggregated": index, "last": result}
            current_started = time.monotonic()
            progress.start_case(pending[index][1].name)
            continue
        failures = [p for p in processes if p.exitcode is not None and p.exitcode != 0]
        if failures:
            return {"phase": "case_failed", "index": index, "detail": [(p.name, p.exitcode) for p in failures]}
        if all(p.exitcode is not None for p in processes):
            return {
                "phase": "incomplete",
                "index": index,
                "detail": [(p.name, p.exitcode) for p in processes],
            }
        if time.monotonic() - current_started > timeout:
            return {"phase": "timeout", "index": index, "detail": "case exceeded case_timeout_s"}
        time.sleep(POLL_SECONDS)


def execute_group(
    cfg: dict,
    group: dict,
    pending: list[tuple[dict, Path]],
    devices: list[int],
    model_dir: Path,
    output: Path,
    progress: Progress,
) -> None:
    """Run a group with one model load; restart the group after a case failure."""
    remaining = list(pending)
    mapping = rank_mapping(remaining[0][0], devices)
    reported: set[str] = set()

    def publish(cases_and_dirs: list[tuple[dict, Path]]) -> None:
        """Print and count every case whose result.json exists and is not yet reported."""
        for case, directory in cases_and_dirs:
            if case["case_id"] in reported:
                continue
            path = directory / "result.json"
            if not path.exists():
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            reported.add(case["case_id"])
            progress.note(
                f"[{progress.completed + 1}/{progress.total}] {directory.name} "
                f"{result['status']} {result.get('forward_p50_ms', '')}"
            )
            progress.advance(result["status"])

    while remaining:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        base = f"{group['strategy']}_cap{group['capacity']}_{stamp}"
        group_dir = output / "groups" / base
        suffix = 1
        while group_dir.exists():
            suffix += 1
            group_dir = output / "groups" / f"{base}_{suffix}"
        group_dir.mkdir(parents=True)
        write_json(
            group_dir / "group.json",
            {
                "strategy": group["strategy"],
                "capacity": group["capacity"],
                "cases": [case["case_id"] for case, _ in remaining],
            },
        )
        # A previous attempt may have left partial rank outcomes; never let the
        # next attempt aggregate stale files from a dead worker group.
        for _, directory in remaining:
            reset_case_outputs(directory)
        progress.note(
            f"  loading model: strategy={group['strategy']} capacity={group['capacity']} cases={len(remaining)}"
        )
        progress.label(f"loading model (capacity={group['capacity']})")
        processes = spawn_ranks(cfg, remaining, mapping, model_dir, rendezvous(), group_dir)
        try:
            outcome = watch_group(cfg, processes, remaining, group_dir, progress)
        finally:
            terminate(processes)

        phase = outcome["phase"]
        if phase == "done":
            publish(remaining)
            break
        if phase in ("load_failed", "load_timeout"):
            detail = json.dumps(outcome.get("detail"), ensure_ascii=False)
            for case, directory in remaining:
                error_result(case, directory, "FAILED", f"Model load failed ({phase}): {detail}")
            publish(remaining)
            break

        index = outcome["index"]
        case, directory = remaining[index]
        if phase == "timeout":
            error_result(case, directory, "TIMEOUT", "Case exceeded case_timeout_s")
        elif phase == "incomplete":
            error_result(
                case,
                directory,
                "INCOMPLETE",
                f"Workers exited before all rank outcomes were written: {outcome.get('detail')}",
            )
        else:
            error_result(case, directory, "FAILED", f"Worker failure: {outcome.get('detail')}")
        # Publish successful cases that ran before the failure plus the failed case.
        publish(remaining[: index + 1])
        # The failed group is torn down; resume the rest with a fresh model load.
        remaining = remaining[index + 1 :]


def run_batch(
    cfg: dict, cases: list[dict], devices: list[int], model_dir: Path, output: Path, resume: bool, skip_failed: bool
) -> int:
    progress = Progress(len(cases))
    for case in cases:
        if case["status"] == "PLANNED":
            continue
        directory = case_directory(output, case)
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "case.json", case)
        write_json(directory / "result.json", dict(case, timing_ok=False))
        progress.note(f"[{progress.completed + 1}/{progress.total}] {directory.name} {case['status']}")
        progress.advance(case["status"])

    groups = batch_groups(cases, cfg)
    planned = sum(len(group["cases"]) for group in groups)
    progress.note(f"batch execution: {planned} cases in {len(groups)} model load(s)")
    for number, group in enumerate(groups, 1):
        pending: list[tuple[dict, Path]] = []
        for case in group["cases"]:
            existing = find_case_directory(output, case["case_id"]) if resume else None
            if existing is not None:
                result_path = existing / "result.json"
                if result_path.exists():
                    previous_status = json.loads(result_path.read_text(encoding="utf-8")).get("status", "FAILED")
                    if previous_status == "OK" or skip_failed:
                        progress.note(
                            f"[{progress.completed + 1}/{progress.total}] {existing.name} already {previous_status}"
                        )
                        progress.advance(previous_status)
                        continue
                # Preserve failed attempts; never mix stale rank outcomes with a new run.
                previous = output / "attempts" / f"{existing.name}-{time.time_ns()}"
                previous.parent.mkdir(parents=True, exist_ok=True)
                existing.rename(previous)
            directory = case_directory(output, case)
            directory.mkdir(parents=True, exist_ok=True)
            write_json(directory / "case.json", case)
            write_json(directory / "rank_mapping.json", rank_mapping(case, devices))
            pending.append((case, directory))
        if not pending:
            continue
        progress.note(
            f"=== group {number}/{len(groups)} strategy={group['strategy']} "
            f"capacity={group['capacity']} cases={len(pending)} (one model load) ==="
        )
        execute_group(cfg, group, pending, devices, model_dir, output, progress)
    progress.finish()
    return progress.failed


def run(cfg: dict, devices: list[int], model_dir: Path, output: Path, resume: bool = False, skip_failed: bool = False) -> dict:
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
        if cfg["batch"]["enabled"]:
            failed = run_batch(cfg, cases, devices, model_dir.resolve(), output, resume, skip_failed)
        else:
            progress = Progress(len(cases))
            for case in cases:
                existing = find_case_directory(output, case["case_id"]) if resume else None
                if existing is not None:
                    result_path = existing / "result.json"
                    if result_path.exists():
                        previous_status = json.loads(result_path.read_text(encoding="utf-8")).get("status", "FAILED")
                        if previous_status == "OK" or skip_failed:
                            progress.note(
                                f"[{progress.completed + 1}/{progress.total}] {existing.name} already {previous_status}"
                            )
                            progress.advance(previous_status)
                            continue
                    # Preserve failed attempts; never mix stale rank outcomes with a new run.
                    previous = output / "attempts" / f"{existing.name}-{time.time_ns()}"
                    previous.parent.mkdir(parents=True, exist_ok=True)
                    existing.rename(previous)
                directory = case_directory(output, case)
                progress.start_case(directory.name)
                result = execute_case(cfg, case, devices, model_dir.resolve(), directory)
                failed += result["status"] != "OK"
                progress.note(
                    f"[{progress.completed + 1}/{progress.total}] {directory.name} "
                    f"{result['status']} {result.get('forward_p50_ms', '')}"
                )
                progress.advance(result["status"])
            progress.finish()
        summary = summarize(output)
        return {"ok": failed == 0, "output": str(output), **summary}
    finally:
        os.close(fd)
        lock.unlink(missing_ok=True)
