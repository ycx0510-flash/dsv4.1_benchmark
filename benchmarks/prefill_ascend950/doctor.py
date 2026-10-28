"""Reproducibility manifest and isolated hardware/extension preflight."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

from .config import BASE_COMMIT, MODEL_CONFIG_SHA256, MODEL_ID, MODEL_REVISION, SUBMODULES, digest, native_environment
from .io_utils import file_hash, write_json

ROOT = Path(__file__).resolve().parents[2]


def unique_path(value: str) -> str:
    """Drop duplicate search-path entries while keeping the effective resolution order."""
    seen, items = set(), []
    for item in value.split(":"):
        if item not in seen:
            seen.add(item)
            items.append(item)
    return ":".join(items)


def command(args: list[str], timeout: int = 60) -> dict:
    try:
        result = subprocess.run(
            args,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
        return {"exit_code": result.returncode, "output": result.stdout.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"exit_code": -1, "output": str(error)}


def source_manifest() -> dict:
    native_paths = [
        "deepinfer",
        "deep_cluster",
        "csrc",
        "setup.py",
        "pyproject.toml",
        "requirements*.txt",
        "scripts",
        "CMakeLists.txt",
        "cmake",
    ]
    tracked = command(["git", "ls-files", *native_paths])
    if tracked["exit_code"]:
        raise ValueError(f"Cannot enumerate native source: {tracked['output']}")
    files = {p: file_hash(ROOT / p) for p in tracked["output"].splitlines() if (ROOT / p).is_file()}
    diff = command(["git", "diff", "--exit-code", BASE_COMMIT, "--", *native_paths])
    modules = {}
    for name, expected in SUBMODULES.items():
        directory = ROOT / "vendor" / name
        head = command(["git", "-C", str(directory), "rev-parse", "HEAD"])
        dirty = command(["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=no"])
        modules[name] = {"expected": expected, "actual": head["output"], "clean": dirty["exit_code"] == 0 and not dirty["output"]}
    gitlinks = command(["git", "ls-tree", BASE_COMMIT, "vendor/"])
    return {
        "base_commit": BASE_COMMIT,
        "head": command(["git", "rev-parse", "HEAD"])["output"],
        "native_source_unchanged": diff["exit_code"] == 0,
        "native_source_id": digest({"files": files, "submodules": modules}),
        "files": files,
        "submodules": modules,
        "base_gitlinks": gitlinks["output"],
    }


def model_manifest(directory: Path, weights: str) -> dict:
    if not directory.is_dir():
        raise ValueError(f"Model directory does not exist: {directory}")
    files = {
        str(p.relative_to(directory)): file_hash(p)
        for p in sorted(directory.rglob("*.json"))
        if ".cache" not in p.parts and p.stat().st_size < 256 * 1024 * 1024
    }
    for name in ("config.json", "tokenizer.json"):
        if name not in files:
            raise ValueError(f"Native VL loader requires {name} in {directory}")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if digest(config) != MODEL_CONFIG_SHA256:
        raise ValueError("Model config differs from pinned V4.1 Flash revision (canonical JSON SHA256)")
    text_cfg = config.get("text_config", config.get("language_config", config))
    if text_cfg.get("num_hidden_layers") != 40:
        raise ValueError("Expected the full 40-layer V4.1 Flash model configuration")
    shards = {}
    if weights == "real":
        indexes = list(directory.glob("*.safetensors.index.json"))
        if indexes:
            names = set(json.loads(indexes[0].read_text(encoding="utf-8"))["weight_map"].values())
            for name in sorted(names):
                path = directory / name
                if not path.is_file():
                    raise ValueError(f"Missing checkpoint shard: {name}")
                shards[name] = {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        else:
            shards = {p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in directory.glob("*.safetensors")}
            if not shards:
                raise ValueError("Real weights require safetensors shards")
    return {
        "model": MODEL_ID,
        "expected_revision": MODEL_REVISION,
        "directory": str(directory.resolve()),
        "small_file_sha256": files,
        "weights": weights,
        "shards": shards,
        "weight_identity_method": "size_and_mtime_plus_index_hash" if shards else "native_dummy_seed",
        "model_id": digest({"files": files, "weights": weights, "shards": shards}),
    }


def hardware_probe(output: Path) -> None:
    """Internal subprocess only: initializes the runtime, never used by planning."""
    try:
        import deep_ep
        import deep_gemm
        import torch
        import torch_npu

        import deepinfer._C.engine as engine
        from deepinfer import PLATFORM

        devices = []
        for i in range(torch.npu.device_count()):
            properties = torch.npu.get_device_properties(i)
            devices.append(
                {
                    "index": i,
                    "name": torch.npu.get_device_name(i),
                    "total_memory": int(properties.total_memory),
                    "properties": str(properties),
                }
            )
        attrs = ("profile", "schedule", "ProfilerActivity", "tensorboard_trace_handler", "_ExperimentalConfig", "ProfilerLevel")
        write_json(
            output,
            {
                "ok": PLATFORM.is_npu and torch.npu.is_available(),
                "devices": devices,
                "torch": torch.__version__,
                "torch_npu": torch_npu.__version__,
                "profiler_capabilities": {a: hasattr(torch_npu.profiler, a) for a in attrs},
                "extensions": {"deep_ep": deep_ep.__file__, "deep_gemm": deep_gemm.__file__, "engine": engine.__file__},
                "engram_buffer": hasattr(deep_ep, "EngramBuffer"),
            },
        )
    except Exception as error:
        import traceback

        write_json(output, {"ok": False, "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})


def check(cfg: dict, devices: list[int], model_dir: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    errors = []
    if platform.system() != "Linux":
        errors.append("Execution requires Linux with eight Ascend 950 devices")
    if sys.version_info[:2] != (3, 12):
        errors.append("Python 3.12 is required")
    allow_local = os.environ.get("PREFILL950_ALLOW_LOCAL_SOURCE") == "1"
    source = source_manifest()
    source["local_patch_accepted"] = allow_local
    if not source["native_source_unchanged"]:
        if allow_local:
            source["native_source_modified"] = True
        else:
            errors.append("Native runtime differs from pinned baseline; use an unchanged native source tree")
    for name, info in source["submodules"].items():
        if info["actual"] != info["expected"] or not info["clean"]:
            # Accept a dirty working tree for the pinned revision only; a different
            # revision always fails. source_manifest still records the local patch.
            if allow_local and info["actual"] == info["expected"]:
                info["local_patch_accepted"] = True
            else:
                errors.append(f"Uninitialized, changed or unpinned submodule: {name}")
    try:
        model = model_manifest(model_dir, cfg["runtime"]["weights"])
    except (ValueError, OSError, KeyError) as error:
        model = {"error": str(error)}
        errors.append(str(error))
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}
    cann_home = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    version_files = {}
    for name in ("version.cfg", "version.info", "aarch64-linux/ascend_toolkit_install.info", "x86_64-linux/ascend_toolkit_install.info"):
        path = cann_home / name
        if path.is_file():
            version_files[str(path)] = path.read_text(encoding="utf-8", errors="replace")
    if not any(re.search(r"(?<![\d.])9\.2\.0(?![\d.])", text) for text in version_files.values()):
        errors.append(f"Cannot establish CANN 9.2.0 from version files under {cann_home}")
    probe = {"ok": False, "error": "Linux required"}
    probe_log = {"exit_code": -1, "output": "Skipped on non-Linux host"}
    if platform.system() == "Linux":
        env = os.environ.copy()
        env.update(native_environment(cfg))
        env.update(CUDA_VISIBLE_DEVICES=",".join(map(str, devices)), ASCEND_RT_VISIBLE_DEVICES=",".join(map(str, devices)))
        probe_path = output / "hardware_probe.json"
        try:
            child = subprocess.run(
                [sys.executable, "-m", "benchmarks.prefill_ascend950.cli", "_probe", "--out", str(probe_path)],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=180,
                check=False,
            )
            probe_log = {"exit_code": child.returncode, "output": child.stdout}
            probe = json.loads(probe_path.read_text(encoding="utf-8")) if probe_path.exists() else {"ok": False, "error": child.stdout}
        except (OSError, subprocess.TimeoutExpired, ValueError) as error:
            probe = {"ok": False, "error": str(error)}
    if not probe.get("ok"):
        errors.append(f"Native Ascend runtime probe failed: {probe.get('error')}")
    else:
        if len(probe["devices"]) != 8 or any("950" not in d["name"] for d in probe["devices"]):
            errors.append("Eight visible Ascend 950 devices are required")
        # torch_npu.__version__ normalizes the wheel version to e.g. "2.13.0.rc1".
        torch_npu_version = probe["torch_npu"].replace(".rc", "rc")
        if probe["torch"] != "2.13.0+cpu" or not torch_npu_version.startswith("2.13.0rc1"):
            errors.append("Expected torch 2.13.0+cpu and torch_npu 2.13.0rc1")
        if not probe.get("engram_buffer"):
            errors.append("Ascend DeepEP EngramBuffer capability missing")
        if cfg["profile"]["enabled"] and not all(probe["profiler_capabilities"].values()):
            errors.append("Required torch_npu profiler capabilities missing")
    compiler = command([os.environ.get("CXX", "g++"), "--version"])
    major = re.search(r"\b(\d+)\.\d+", compiler["output"])
    if compiler["exit_code"] or not major or int(major[1]) < 13:
        errors.append("GCC >= 13 is required by the native C++20 build")
    info = command(["npu-smi", "info"])
    if info["exit_code"]:
        errors.append("npu-smi info unavailable; driver/firmware inventory cannot be recorded")
    environment = {
        "system": platform.platform(),
        "architecture": platform.machine(),
        "libc": platform.libc_ver(),
        "python": sys.version,
        "packages": packages,
        "cann_version_files": version_files,
        "compiler": compiler,
        "device_inventory": probe,
        "device_mapping": devices,
        "native_environment": native_environment(cfg),
        "inherited_deepinfer_environment": {k: v for k, v in os.environ.items() if k.startswith("DEEPINFER_")},
    }
    extension_dir = ROOT / "deepinfer" / "_so" / f"ascend-{platform.machine()}"
    environment["native_extension_sha256"] = {p.name: file_hash(p) for p in sorted(extension_dir.glob("*.so"))}
    runtime_environment = {
        k: v
        for k, v in os.environ.items()
        if k.startswith(("HCCL_", "HCOMM_", "TASK_QUEUE_", "PYTORCH_NPU_", "TILELANG_", "OMP_"))
        or k in ("LD_LIBRARY_PATH", "LD_PRELOAD", "ASCEND_HOME_PATH", "CXX", "CC")
    }
    # CANN's set_env.sh is not idempotent: sourcing it again appends duplicate
    # entries without changing library resolution. Dedupe so the fingerprint is
    # stable across shells that sourced the environment a different number of times.
    for key in ("LD_LIBRARY_PATH", "LD_PRELOAD"):
        if key in runtime_environment:
            runtime_environment[key] = unique_path(runtime_environment[key])
    environment["runtime_environment"] = runtime_environment
    version_paths = (Path("/usr/local/Ascend/driver/version.info"), Path("/usr/local/Ascend/firmware/version.info"))
    environment["driver_firmware_version_files"] = {
        str(p): p.read_text(encoding="utf-8", errors="replace") for p in version_paths if p.is_file()
    }
    result = {
        "ok": not errors,
        "errors": errors,
        "source": source,
        "model": model,
        "environment": environment,
        "environment_id": digest(environment),
        "native_source_id": source["native_source_id"],
        "model_id": model.get("model_id"),
        "npu_smi": info,
        "topology": command(["npu-smi", "info", "-t", "topo"]),
        "probe_log": probe_log,
        "pip_freeze": command([sys.executable, "-m", "pip", "freeze"]),
        "verification": "Preflight only; hardware performance parity requires measured comparison",
    }
    write_json(output / "doctor.json", result)
    return result
