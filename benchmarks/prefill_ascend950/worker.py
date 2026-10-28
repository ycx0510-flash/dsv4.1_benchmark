"""One native Ascend rank. Imported by spawn without importing torch in the parent."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from .config import digest, native_environment
from .io_utils import write_json, write_jsonl
from .planner import steps


class NativePathMismatch(RuntimeError):
    """The pinned model or repeated native metadata/module path did not match."""


def entry(cfg: dict, case: dict, mapping: dict, model_dir: str, init_method: str, output: str) -> None:
    # deep-jit resolves bisheng from ASCEND_HOME_PATH. The shim keeps every CANN
    # path identical but routes device compiles through the hardened wrapper.
    jit_home = os.environ.get("PREFILL950_JIT_ASCEND_HOME")
    if jit_home:
        os.environ["ASCEND_HOME_PATH"] = jit_home
    directory = Path(output) / f"rank_{mapping['global_rank']}"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "worker.log").open("a", encoding="utf-8", buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        sys.stdout = log
        sys.stderr = log
        try:
            os.environ.update(native_environment(cfg))
            visible = ",".join(map(str, mapping["visible_devices"]))
            os.environ.update(CUDA_VISIBLE_DEVICES=visible, ASCEND_RT_VISIBLE_DEVICES=visible)
            _run(cfg, case, mapping, model_dir, init_method, directory)
        except BaseException as error:
            traceback.print_exc()
            write_json(
                directory / "error.json",
                {
                    "rank": mapping["global_rank"],
                    "type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "status": (
                        "PATH_MISMATCH"
                        if isinstance(error, NativePathMismatch)
                        else "OOM"
                        if "out of memory" in str(error).lower() or "OutOfMemory" in type(error).__name__
                        else "FAILED"
                    ),
                },
            )
            log.flush()
            raise SystemExit(1) from error


def _run(cfg: dict, case: dict, mapping: dict, model_dir: str, init_method: str, directory: Path) -> None:
    # This is intentionally the first accelerator/native import in the spawned rank.
    from deepinfer import PLATFORM

    if not PLATFORM.is_npu:
        raise RuntimeError("Ascend platform was not selected")
    import torch
    import torch.distributed as dist

    from deepinfer.engine.arg_utils import EngineArgs
    from deepinfer.engine.engine_type import EngineType
    from deepinfer.model_executor import eplb
    from deepinfer.worker.base import WorkerBase, set_pdeathsig
    from deepinfer.worker.cache_engine import CacheEngine

    from .alignment import NativeDiagnostic
    from .profiler import create_profiler
    from .workload import NativeSequences, StateSnapshot, seed_cache

    class BenchWorker(WorkerBase):
        def start_slave_worker_loop(self) -> None:
            # All eight ranks execute the same benchmark loop. Native follower
            # metadata broadcasts remain in ModelRunner._prepare_input_follower.
            set_pdeathsig()

    set_pdeathsig()
    batch = case["per_dp_batch_size"]
    capacity = max(batch * (cfg["runtime"]["mtp_draft_length"] + 1), case["cp"] * case["max_rows_per_rank"])
    args = EngineArgs(
        model=model_dir,
        tokenizer=model_dir,
        engine_type=EngineType.PROMPT,
        load_format="dummy" if cfg["runtime"]["weights"] == "dummy" else "auto",
        seed=cfg["runtime"]["seed"],
        max_model_len=cfg["runtime"]["max_model_len"],
        data_parallel_size=case["dp"],
        context_parallel_size=case["cp"],
        expert_parallel_size=8,
        engram_parallel_size=8,
        engine_rank=mapping["dp_rank"],
        engine_group_rank=0,
        max_num_seqs=batch,
        max_num_batched_tokens=capacity,
        chunked_prefill=True,
        gpu_memory_utilization=cfg["runtime"]["gpu_memory_utilization"],
        tokens_per_block=128,
    )
    model_cfg, cache_cfg, parallel_cfg, scheduler_cfg = args.create_engine_configs()
    # runtime.engram=false builds the native graph without Engram. The env var is
    # kept as an explicit override for ad-hoc diagnostic runs.
    disable_engram = not cfg["runtime"]["engram"] or os.environ.get("PREFILL950_DISABLE_ENGRAM") == "1"
    if disable_engram:
        model_cfg.model_cfg.gpt.engram = None
    worker = BenchWorker(
        EngineType.PROMPT,
        model_cfg,
        parallel_cfg,
        scheduler_cfg,
        cache_cfg,
        mapping["cp_rank"],
        mapping["cp_rank"],
        init_method,
        mapping["cp_rank"] == 0,
    )
    worker.load_model(skip_load=cfg["runtime"]["weights"] == "dummy")
    runner = worker.model_runner
    model = runner.model
    layers = model.model.layers
    if (
        type(model).__name__ != "DeepseekVLForCausalLM"
        or len(layers) != 40
        or not (model.engram_enabled or disable_engram)
        or not model.hc_enabled
    ):
        raise NativePathMismatch("Loaded model does not match the full native V4.1 Flash graph contract")
    branches = []
    for i, layer in enumerate(layers):
        mlp = layer.mlp
        if not mlp.use_mega_moe or mlp.mma_type != "fp8xfp4":
            raise NativePathMismatch(f"Layer {i}: required Ascend FP8xFP4 MegaMoE path unavailable")
        branches.append(
            {
                "layer": i,
                "use_mega_moe": mlp.use_mega_moe,
                "mma_type": mlp.mma_type,
                "physical_experts": mlp.num_physical_experts,
                "local_experts": mlp.num_local_physical_experts,
            }
        )
    cache = CacheEngine(cache_cfg, model.cfg, batch, num_gpu_blocks=case["logical_cache_blocks"])
    worker.cache_engine, worker.gpu_cache = cache, cache.gpu_cache
    with torch.inference_mode():
        seeded = seed_cache(cache, case, cfg["runtime"]["seed"], mapping["dp_rank"] * batch)
        snapshot = StateSnapshot(cache)
    write_json(
        directory / "native_contract.json",
        {
            "rank_mapping": mapping,
            "model_class": type(model).__name__,
            "branches": branches,
            "cache": seeded,
            "max_num_batched_tokens": capacity,
            "runtime_max_model_len": model_cfg.runtime_max_model_len,
            "model_cfg": json.loads(json.dumps(model_cfg.model_cfg, default=str)),
            "environment": native_environment(cfg),
        },
    )
    sequences = NativeSequences(case, cfg["runtime"]["seed"], mapping["dp_rank"] * batch) if mapping["cp_rank"] == 0 else None
    trajectory = list(steps(case))
    chunks = len(trajectory)
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    profiling_enabled = False
    forward_scope = None
    forward_invocations = 0

    def before(*_: Any) -> None:
        nonlocal forward_scope, forward_invocations
        forward_invocations += 1
        if profiling_enabled:
            forward_scope = torch.profiler.record_function("prefill950:native_model_forward")
            forward_scope.__enter__()

    def after(*_: Any) -> None:
        if profiling_enabled:
            forward_scope.__exit__(None, None, None)

    handles = [model.register_forward_pre_hook(before), model.register_forward_hook(after)]
    minimum_free = torch.npu.mem_get_info()[0]

    @torch.inference_mode()
    def run_trajectory(iteration: int = 0) -> None:
        """One unmeasured trajectory; used by alignment, warmup and profiling."""
        nonlocal minimum_free
        with torch.profiler.record_function("prefill950:cache_restore") if profiling_enabled else nullcontext():
            snapshot.restore()
            torch.npu.synchronize()
        for step in trajectory:
            # Construct a new native SequenceMetadata and InputMetadata per call.
            metadata = sequences.metadata(step) if sequences is not None else None
            dist.barrier()  # CPU/Gloo coordination, outside both NPU event boundaries.
            with (
                torch.profiler.record_function(f"prefill950:case={case['case_id']}:iteration={iteration}:chunk={step['chunk']}")
                if profiling_enabled
                else nullcontext()
            ):
                output = runner.execute_main_model(metadata, cache.gpu_cache)
            torch.npu.synchronize()
            del output, metadata
            minimum_free = min(minimum_free, torch.npu.mem_get_info()[0])

    @torch.inference_mode()
    def run_batch(count: int) -> dict:
        """Run `count` trajectories back-to-back with one NPU event pair.

        No synchronize, barrier or cache restore is inserted between forwards, so
        dispatch never drains; the event span covers the whole batch and the
        reported latency is ``total_ms / (count * chunks)`` per step.
        """
        nonlocal minimum_free, forward_invocations
        snapshot.restore()
        torch.npu.synchronize()  # Settle the restore before the timed span.
        calls_before = forward_invocations
        wall_start = time.perf_counter()

        def build(step: dict) -> Any:
            return sequences.metadata(step) if sequences is not None else None

        # Build the first metadata outside the span, then keep the device queue fed.
        first = build(trajectory[0])
        start.record()
        output = runner.execute_main_model(first, cache.gpu_cache)
        del output, first
        for iteration in range(count):
            for chunk_index, step in enumerate(trajectory):
                if iteration == 0 and chunk_index == 0:
                    continue
                metadata = build(step)
                output = runner.execute_main_model(metadata, cache.gpu_cache)
                del output, metadata
        end.record()
        end.synchronize()
        total_ms = start.elapsed_time(end)
        wall_ms = (time.perf_counter() - wall_start) * 1000
        observed = forward_invocations - calls_before
        expected = count * chunks
        if observed != expected:
            raise RuntimeError(f"Expected {expected} root forwards in batch, observed {observed}")
        minimum_free = min(minimum_free, torch.npu.mem_get_info()[0])
        return {"count": count, "chunks": chunks, "total_ms": total_ms, "wall_ms": wall_ms, "root_forwards": observed}

    graph = {"match": False, "kind": "NATIVE_METADATA_AND_MODULE_REPEATABILITY", "external_reference_validated": False}
    if cfg["alignment"]["enabled"]:
        signatures = []
        for i in range(2):
            with NativeDiagnostic(runner) as diagnostic:
                run_trajectory()
            sig = diagnostic.result()
            write_json(directory / f"graph_signature_{i}.json", sig)
            signatures.append(sig["sha256"])
        graph.update(match=signatures[0] == signatures[1], signatures=signatures, branch_sha256=digest(branches))
        if not graph["match"]:
            raise NativePathMismatch("Repeated fresh native metadata/module signatures differ; inspect graph_signature files")
    for _ in range(cfg["measurement"]["warmup"]):
        run_trajectory()
    preparation_memory = {
        "preparation_peak_torch_allocated_bytes": torch.npu.max_memory_allocated(),
        "preparation_peak_torch_reserved_bytes": torch.npu.max_memory_reserved(),
        "cache_block_bytes": seeded["block_bytes"],
        "cache_state_bytes": seeded["state_bytes"],
        "state_snapshot_bytes": snapshot.saved.numel(),
    }
    torch.npu.reset_peak_memory_stats()
    manager = eplb.current()
    if manager is None:
        raise RuntimeError("Native EPLB manager is missing")
    with torch.inference_mode():
        for i in range(manager.num_moe_layers):
            manager.layer(i).expert_count.zero_()
    samples = []
    batches = []
    for r in range(cfg["measurement"]["rounds"]):
        record = run_batch(cfg["measurement"]["iterations"])
        batches.append({"round": r, **record})
        per_step_ms = record["total_ms"] / record["count"] / chunks
        per_step_host_ms = record["wall_ms"] / record["count"] / chunks
        for k in range(record["count"]):
            for step in trajectory:
                samples.append(
                    {
                        "round": r,
                        "iteration": k,
                        "chunk": step["chunk"],
                        "forward_ms": per_step_ms,
                        "host_preparation_and_forward_ms": per_step_host_ms,
                        "batch_total_ms": record["total_ms"],
                    }
                )
        write_jsonl(directory / "samples.jsonl", samples)
    expert_assignments = sum(int(manager.layer(i).expert_count.sum().item()) for i in range(manager.num_moe_layers))
    memory = {
        "peak_torch_allocated_bytes": torch.npu.max_memory_allocated(),
        "peak_torch_reserved_bytes": torch.npu.max_memory_reserved(),
        "minimum_device_free_bytes": int(minimum_free),
        "memory_scope": "Torch peaks plus device-free samples after cache seeding and each forward; not continuous HBM peak",
        **preparation_memory,
    }
    # CANN cannot export shapes for custom AscendC kernels. Record module input
    # shapes during the profiler's active steps as a companion dataset.
    shape_records: list[dict] = []
    shape_gate = {"active": False, "pindex": 0}

    def tensor_shapes(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, (list, tuple)):
            items = [tensor_shapes(item) for item in value]
            items = [item for item in items if item is not None]
            return items or None
        if isinstance(value, dict):
            items = {str(key): tensor_shapes(item) for key, item in value.items()}
            items = {key: item for key, item in items.items() if item is not None}
            return items or None
        return None

    def shape_hook(name: str):
        def hook(module: Any, args: tuple) -> None:
            if not shape_gate["active"]:
                return
            inputs = tensor_shapes(args)
            if inputs is None:
                return
            shape_records.append(
                {"pindex": shape_gate["pindex"], "module": name, "type": type(module).__qualname__, "inputs": inputs}
            )

        return hook

    profile = {"valid": None, "status": "DISABLED"}
    if cfg["profile"]["enabled"]:
        selected = cfg["profile"]["ranks"] == "all" or mapping["global_rank"] == 0
        profiling_enabled = selected
        profile = {"valid": None, "status": "SKIPPED", "rank": mapping["global_rank"]}
        shape_handles = []
        profiler, profile_error = None, None
        if selected:
            shape_handles = [module.register_forward_pre_hook(shape_hook(name)) for name, module in model.named_modules()]
            try:
                profiler = create_profiler(directory / "profiler", cfg["profile"], mapping["global_rank"])
                profiler.start()
            except Exception as error:
                profile_error = f"{type(error).__name__}: {error}"
        # All ranks run the same profiling steps; only the selected rank records.
        for pindex in range(sum(cfg["profile"][k] for k in ("wait_steps", "warmup_steps", "active_steps"))):
            shape_gate.update(
                active=selected and pindex >= cfg["profile"]["wait_steps"] + cfg["profile"]["warmup_steps"],
                pindex=pindex,
            )
            run_trajectory(iteration=pindex)
            if profiler is not None and profile_error is None:
                try:
                    profiler.step()
                except Exception as error:
                    profile_error = f"{type(error).__name__}: {error}"
        if profiler is not None:
            try:
                profiler.stop()
            except Exception as error:
                profile_error = f"{type(error).__name__}: {error}"
        profiling_enabled = False
        shape_gate["active"] = False
        for shape_handle in shape_handles:
            shape_handle.remove()
        if selected:
            write_jsonl(directory / "module_shapes.jsonl", shape_records)
            expected = cfg["profile"]["active_steps"] * len(trajectory)
            if profile_error:
                profile = {"valid": False, "status": "FAILED", "error": profile_error, "expected_active_forwards": expected}
            else:
                # Raw traces stay on disk; parsing is deferred to offline analysis.
                profile = {
                    "valid": None,
                    "status": "COLLECTED",
                    "expected_active_forwards": expected,
                    "note": "offline parsing pending",
                }
            write_json(directory / "profiler_index.json", profile)
    for handle in handles:
        handle.remove()
    write_json(
        directory / "outcome.json",
        {
            "rank": mapping["global_rank"],
            "timing_mode": "batch_average",
            "batches": batches,
            "samples": samples,
            "profile": profile,
            "graph_alignment": graph,
            "expert_assignments": expert_assignments,
            **memory,
        },
    )
    sys.stdout.flush()
    worker.stop()
