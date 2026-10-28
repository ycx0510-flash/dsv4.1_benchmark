"""Strict, accelerator-independent configuration and reproducibility keys."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

BASE_COMMIT = "553741557a57df44a46fb3ab72b6aafeaf945337"
MODEL_ID = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
MODEL_REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
MODEL_CONFIG_SHA256 = "d7637228d27528f6bd259781b5a27258068f50bf637c9c83aab784d81579669d"
SUBMODULES = {
    "DeepEP-Ascend": "3b25377d04b24fc6154698ded78a2bcb2c59afff",
    "DeepGEMM-Ascend": "3a35326271b069b8b5bf1a754435acd6cc91575b",
    "FlashMLA": "2e5429fc5653bab6e081f09477126f731882a6a9",
    "DeepSelect": "bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18",
}
STRATEGIES = {"DP8CP1EP8": (8, 1, 8), "DP4CP2EP8": (4, 2, 8), "DP2CP4EP8": (2, 4, 8), "DP1CP8EP8": (1, 8, 8)}
DEFAULTS = {
    "schema_version": 1,
    "runtime": {
        "weights": "dummy",
        "engram": True,
        "max_model_len": 1048576,
        "mtp_draft_length": 5,
        "seed": 20261026,
        "case_timeout_s": 3600,
        "gpu_memory_utilization": 0.95,
    },
    "parallel": {"strategies": list(STRATEGIES), "batch_mode": "fixed_global", "global_batch_size": 8},
    "workload": {
        "context_lengths": [8192],
        "cache_hit_rates": [0.5],
        "alignment_policy": "floor",
        "cache_source": "synthetic_hbm",
        # Chunking is opt-in: set execution=chunked_suffix and a chunk size.
        "execution": "single_forward",
        "chunk_tokens_per_request": None,
    },
    "yoco": {
        "mode": "compact",
        "keep_tokens": 140,
        "store_extra_state": True,
        "max_backtracking_tokens": 6,
        "encoder_swa_state": "available",
    },
    "moe": {"force_load_balance": True, "force_imbalance_ratio": 0, "extra_experts": 32, "rearrangement_interval": 0},
    # One timing span covers `iterations` back-to-back forwards (no sync in between).
    "measurement": {"warmup": 0, "iterations": 10, "rounds": 1, "timing_scope": "native_model_forward"},
    "profile": {
        "enabled": True,
        "ranks": "all",
        "level": "Level1",
        "wait_steps": 1,
        "warmup_steps": 2,
        "active_steps": 3,
        "record_shapes": True,
        "with_stack": False,
        "profile_memory": False,
    },
    "alignment": {"enabled": True},
}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def integer(value: Any, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def rate(value: Any) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("cache hit rate must be a fraction in [0, 1]")
    try:
        result = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"Invalid cache hit rate: {value!r}") from error
    if not result.is_finite() or not 0 <= result <= 1:
        raise ValueError(f"cache hit rate must be in [0, 1], got {value!r}")
    return result


def resolve(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a JSON object")
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    cfg = copy.deepcopy(DEFAULTS)
    for section, values in raw.items():
        if section == "schema_version":
            cfg[section] = values
            continue
        if not isinstance(values, dict):
            raise ValueError(f"{section} must be an object")
        allowed = set(DEFAULTS[section]) | ({"per_dp_batch_size"} if section == "parallel" else set())
        if set(values) - allowed:
            raise ValueError(f"Unknown {section} keys: {sorted(set(values) - allowed)}")
        if section == "parallel" and values.get("batch_mode") == "fixed_per_dp":
            cfg[section].pop("global_batch_size", None)
        cfg[section].update(values)
    if type(cfg["schema_version"]) is not int or cfg["schema_version"] != 1:
        raise ValueError("Only schema_version=1 is supported")
    r, p, w, y, m, measure, prof = (cfg[k] for k in ("runtime", "parallel", "workload", "yoco", "moe", "measurement", "profile"))
    for section in ("yoco", "profile", "alignment", "moe"):
        for key, default in DEFAULTS[section].items():
            if type(default) is bool and type(cfg[section][key]) is not bool:
                raise ValueError(f"{section}.{key} must be a boolean")
    for name in ("max_model_len", "case_timeout_s"):
        integer(r[name], f"runtime.{name}")
    integer(r["seed"], "runtime.seed", 0)
    integer(r["mtp_draft_length"], "runtime.mtp_draft_length", 0)
    if type(r["engram"]) is not bool:
        raise ValueError("runtime.engram must be a boolean")
    if (
        r["weights"] not in ("dummy", "real")
        or type(r["gpu_memory_utilization"]) not in (int, float)
        or not 0 < r["gpu_memory_utilization"] < 1
    ):
        raise ValueError("weights must be dummy/real; gpu_memory_utilization must be in (0, 1)")
    if (
        not isinstance(p["strategies"], list)
        or not p["strategies"]
        or any(not isinstance(s, str) or s not in STRATEGIES for s in p["strategies"])
    ):
        raise ValueError(f"strategies must be a nonempty selection of {list(STRATEGIES)}")
    if len(set(p["strategies"])) != len(p["strategies"]):
        raise ValueError("Duplicate strategies")
    if p["batch_mode"] == "fixed_global":
        g = integer(p.get("global_batch_size"), "parallel.global_batch_size")
        if "per_dp_batch_size" in p or any(g % STRATEGIES[s][0] for s in p["strategies"]):
            raise ValueError("fixed_global requires divisible global_batch_size and no per_dp_batch_size")
    elif p["batch_mode"] == "fixed_per_dp":
        integer(p.get("per_dp_batch_size"), "parallel.per_dp_batch_size")
        if "global_batch_size" in p:
            raise ValueError("fixed_per_dp cannot specify global_batch_size")
    else:
        raise ValueError("batch_mode must be fixed_global/fixed_per_dp")
    if not isinstance(w["context_lengths"], list) or not w["context_lengths"]:
        raise ValueError("context_lengths must be a nonempty list")
    for length in w["context_lengths"]:
        if integer(length, "context_length") > r["max_model_len"]:
            raise ValueError("context length exceeds max_model_len")
    if len(set(w["context_lengths"])) != len(w["context_lengths"]):
        raise ValueError("Duplicate context lengths")
    if not isinstance(w["cache_hit_rates"], list) or not w["cache_hit_rates"]:
        raise ValueError("cache_hit_rates must be a nonempty list")
    w["cache_hit_rates"] = [str(rate(h)) for h in w["cache_hit_rates"]]
    if len({rate(h) for h in w["cache_hit_rates"]}) != len(w["cache_hit_rates"]):
        raise ValueError("Duplicate cache hit rates")
    if w["alignment_policy"] not in ("floor", "exact") or w["cache_source"] != "synthetic_hbm":
        raise ValueError("Only floor/exact alignment and synthetic_hbm are supported")
    if w["execution"] == "chunked_suffix":
        integer(w["chunk_tokens_per_request"], "chunk_tokens_per_request")
    elif w["execution"] != "single_forward" or w["chunk_tokens_per_request"] is not None:
        raise ValueError("single_forward requires null chunk_tokens_per_request")
    if y["mode"] not in ("off", "mask", "compact") or y["encoder_swa_state"] != "available":
        raise ValueError("Only decoder YOCO off/mask/compact with available encoder SWA is supported")
    integer(y["keep_tokens"], "keep_tokens")
    integer(y["max_backtracking_tokens"], "max_backtracking_tokens", 0)
    if y["mode"] != "off" and y["keep_tokens"] < 128 + max(y["max_backtracking_tokens"], r["mtp_draft_length"] + 1):
        raise ValueError("keep_tokens must cover SWA window + effective backtracking")
    integer(m["extra_experts"], "extra_experts", 0)
    if (
        m["force_load_balance"] is not True
        or type(m["force_imbalance_ratio"]) not in (int, float)
        or m["force_imbalance_ratio"] != 0
        or type(m["rearrangement_interval"]) is not int
        or m["rearrangement_interval"] != 0
    ):
        raise ValueError("Native forced load balancing is required; imbalance/rearrangement must be disabled")
    for key in ("warmup", "iterations", "rounds"):
        integer(measure[key], f"measurement.{key}", 0 if key == "warmup" else 1)
    if measure["timing_scope"] != "native_model_forward":
        raise ValueError("Only native_model_forward timing is supported")
    if prof["ranks"] != "all" or prof["level"] not in ("Level0", "Level1", "Level2"):
        raise ValueError("Profile requires all ranks and Level0/Level1/Level2")
    for key in ("wait_steps", "warmup_steps", "active_steps"):
        integer(prof[key], f"profile.{key}", 1 if key == "active_steps" else 0)
    return cfg


def load(path: str | Path) -> dict:
    # Preserve decimal spelling across JSON input, including block boundaries.
    raw = json.loads(Path(path).read_text(encoding="utf-8"), parse_float=str)
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a JSON object")
    # gpu_memory_utilization is an actual floating runtime parameter, not a hit rate.
    if isinstance(raw.get("runtime"), dict) and "gpu_memory_utilization" in raw["runtime"]:
        raw["runtime"]["gpu_memory_utilization"] = float(raw["runtime"]["gpu_memory_utilization"])
    if isinstance(raw.get("moe"), dict) and isinstance(raw["moe"].get("force_imbalance_ratio"), str):
        raw["moe"]["force_imbalance_ratio"] = float(raw["moe"]["force_imbalance_ratio"])
    return resolve(raw)


def native_environment(cfg: dict) -> dict[str, str]:
    y, r, m = cfg["yoco"], cfg["runtime"], cfg["moe"]
    environment = {
        "DEEPINFER_ACCELERATOR": "ascend",
        "DEEPINFER_FORCE_LOAD_BALANCE": "1",
        "DEEPINFER_FORCE_IMBALANCE_RATIO": "0",
        "DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS": str(m["extra_experts"]),
        "DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL": "0",
        "DEEPINFER_EXPERT_COUNT_WINDOW_SIZE": "3",
        "DEEPINFER_REARRANGE_AFTER_PREFILL_IN_NORMAL_ENGINE": "0",
        "DEEPINFER_USE_NEXTN_DECODE": str(r["mtp_draft_length"]),
        "DEEPINFER_MAX_BACKTRACKING_TOKENS": str(y["max_backtracking_tokens"]),
        "DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE": y["mode"],
        "DEEPINFER_YOCO_PREFILL_KEEP_TOKENS": str(y["keep_tokens"]),
        "DEEPINFER_PREFILL_STORE_EXTRA_STATE": str(int(y["store_extra_state"])),
        "DEEPINFER_PREFILL_STORE_EXTRA_SIZE": "128",
        "DEEPINFER_ZERO_OVERHEAD_SCHEDULING": "0",
        "DEEPINFER_BATCH_INVARIANT": "0",
        "DEEPINFER_ENABLE_MTP_VARLEN": "0",
        "DEEPINFER_ENFORCE_EAGER": "1",
        "DEEPINFER_PROFILE_HTTP_ENABLE": "0",
        "DEEPINFER_GLOO_TIMEOUT": str(r["case_timeout_s"]),
    }
    # Ascend cannot keep the load-time Engram tables and the device EngramBuffer
    # resident at once; stage the tables on the host unless explicitly overridden.
    if r["engram"] and os.environ.get("DEEPINFER_ENGRAM_CPU_LOAD") is None:
        environment["DEEPINFER_ENGRAM_CPU_LOAD"] = "1"
    return environment
