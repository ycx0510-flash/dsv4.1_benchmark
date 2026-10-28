"""NPU profiling is imported only inside an initialized worker."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .io_utils import write_json


def events(data: dict | list) -> list[dict]:
    return data if isinstance(data, list) else data.get("traceEvents", [])


def kernel_events(data: dict | list) -> list[dict]:
    return [
        e
        for e in events(data)
        if str(e.get("cat", "")).lower() in ("kernel", "npu_kernel", "ai_core", "aicpu") or "Task Type" in (e.get("args") or {})
    ]


def inspect_traces(directory: Path) -> dict:
    traces, counts, errors, forwards = [], Counter(), [], 0
    for path in sorted(directory.rglob("*.json")):
        if "trace" not in path.name.lower():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            kernels = kernel_events(data)
            forwards += sum(e.get("name") == "prefill950:native_model_forward" and e.get("ph") == "X" for e in events(data))
            if kernels:
                traces.append(str(path.relative_to(directory)))
                counts.update(str(e.get("name", "unknown")) for e in kernels)
        except (ValueError, OSError, AttributeError) as error:
            errors.append({"file": str(path), "error": str(error)})
    return {
        "valid": bool(traces),
        "traces": traces,
        "npu_kernel_events": sum(counts.values()),
        "forward_ranges": forwards,
        "kernel_counts": dict(counts),
        "parse_errors": errors,
        "files": [str(p.relative_to(directory)) for p in sorted(directory.rglob("*")) if p.is_file()],
    }


def create_profiler(directory: Path, cfg: dict, rank: int) -> Any:
    import torch_npu

    p = torch_npu.profiler
    for attr in ("profile", "schedule", "ProfilerActivity", "tensorboard_trace_handler", "_ExperimentalConfig", "ProfilerLevel"):
        if not hasattr(p, attr):
            raise RuntimeError(f"torch_npu.profiler missing required capability {attr}")
    directory.mkdir(parents=True, exist_ok=True)
    # record_shapes only covers torch/aten CPU ops. NPU-side operator shapes are
    # exported by the CANN text parser only when data simplification is off and
    # op_attr is enabled (which itself requires DB export).
    experimental = p._ExperimentalConfig(
        profiler_level=getattr(p.ProfilerLevel, cfg["level"]),
        data_simplification=False,
        op_attr=True,
        export_type=[p.ExportType.Db, p.ExportType.Text],
    )
    return p.profile(
        activities=[p.ProfilerActivity.CPU, p.ProfilerActivity.NPU],
        schedule=p.schedule(wait=cfg["wait_steps"], warmup=cfg["warmup_steps"], active=cfg["active_steps"], repeat=1),
        on_trace_ready=p.tensorboard_trace_handler(str(directory), worker_name=f"rank_{rank}"),
        record_shapes=cfg["record_shapes"],
        with_stack=cfg["with_stack"],
        profile_memory=cfg["profile_memory"],
        experimental_config=experimental,
    )


def save_profile_index(directory: Path, expected_forwards: int) -> dict:
    result = inspect_traces(directory)
    result["expected_active_forwards"] = expected_forwards
    result["valid"] = result["valid"] and result["forward_ranges"] >= expected_forwards
    write_json(directory.parent / "profiler_index.json", result)
    return result
