"""Pure workload planning. Never import torch or DeepInfer here."""

from __future__ import annotations

import json
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path

from .config import STRATEGIES, digest, rate


def normalize_prefix(length: int, requested: str, mode: str, keep: int, policy: str) -> dict:
    target = Decimal(length) * rate(requested)
    minimum = keep if mode != "off" else 1
    cap = max(0, length - minimum) // 128 * 128
    cached = min(int(target) // 128 * 128, cap)
    reasons = []
    if Decimal(int(target) // 128 * 128) != target:
        reasons.append("block_floor")
    if int(target) // 128 * 128 > cap:
        reasons.append("yoco_min_suffix" if mode != "off" else "nonzero_query")
    if policy == "exact" and Decimal(cached) != target:
        raise ValueError(f"Exact hit rate cannot be represented: L={length}, h={requested}, cap={cap}")
    return {
        "cached_tokens_effective": cached,
        "new_tokens_per_request": length - cached,
        "cache_hit_rate_requested": str(requested),
        "cache_hit_rate_effective": cached / length,
        "alignment_adjustments": reasons,
    }


def steps(case: dict) -> Iterator[dict]:
    length, cached, batch, cp = (case[k] for k in ("context_length", "cached_tokens_effective", "per_dp_batch_size", "cp"))
    remaining = length - cached
    chunk = case["chunk_tokens_per_request"] if case["execution"] == "chunked_suffix" else remaining
    for j, start in enumerate(range(cached, length, chunk)):
        q = min(chunk, length - start)
        clip = min(max(length - case["keep_tokens"] - start, 0), q - 1) if case["yoco_mode"] == "compact" else 0
        consumer = q - clip
        padded = lambda n: n if cp == 1 else 2 * ((n + 2 * cp - 1) // (2 * cp))
        yield {
            "chunk": j,
            "q_start": start,
            "seqlen_q": q,
            "seqlen_k": start + q,
            "full_len": length,
            "producer_rows_per_dp": batch * q,
            "producer_padded_rows_per_rank": batch * padded(q),
            "consumer_rows_per_dp": batch * consumer,
            "consumer_padded_rows_per_rank": batch * padded(consumer),
            "new_tokens_global": case["global_batch_size"] * q,
        }


def _case_specs_from_file(cfg: dict, path: str) -> list[tuple[str, int, str]]:
    """Read explicit (strategy, context_length, hit) rows from a JSONL ledger.

    Unknown extra keys are ignored so a source tag may be kept in the manifest
    without becoming part of the benchmark case or its config hash.
    """
    manifest = Path(path).expanduser()
    specs: list[tuple[str, int, str]] = []
    seen: set[tuple[str, int, str]] = set()
    enabled = set(cfg["parallel"]["strategies"])
    with manifest.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except ValueError as error:
                raise ValueError(f"{manifest}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(item, dict):
                raise ValueError(f"{manifest}:{line_number}: each line must be a JSON object")
            for key in ("strategy", "context_length", "hit"):
                if key not in item:
                    raise ValueError(f"{manifest}:{line_number}: missing {key!r}")
            strategy = item["strategy"]
            length = item["context_length"]
            if not isinstance(strategy, str) or strategy not in enabled:
                raise ValueError(f"{manifest}:{line_number}: strategy {strategy!r} is not enabled")
            if type(length) is not int or length < 1 or length > cfg["runtime"]["max_model_len"]:
                raise ValueError(f"{manifest}:{line_number}: invalid context_length {length!r}")
            try:
                requested = str(rate(item["hit"]))
            except ValueError as error:
                raise ValueError(f"{manifest}:{line_number}: invalid hit {item['hit']!r}: {error}") from error
            key = (strategy, length, requested)
            if key in seen:
                raise ValueError(f"{manifest}:{line_number}: duplicate case {key!r}")
            seen.add(key)
            specs.append(key)
    if not specs:
        raise ValueError(f"{manifest}: no cases found")
    return specs


def _build_case(cfg: dict, strategy: str, length: int, requested: str) -> dict:
    dp, cp, ep = STRATEGIES[strategy]
    batch = (
        cfg["parallel"]["global_batch_size"] // dp
        if cfg["parallel"]["batch_mode"] == "fixed_global"
        else cfg["parallel"]["per_dp_batch_size"]
    )
    case = {
        "strategy": strategy,
        "dp": dp,
        "cp": cp,
        "ep": ep,
        "batch_mode": cfg["parallel"]["batch_mode"],
        "global_batch_size": dp * batch,
        "per_dp_batch_size": batch,
        "context_length": length,
        "yoco_mode": cfg["yoco"]["mode"],
        "keep_tokens": cfg["yoco"]["keep_tokens"],
        "execution": cfg["workload"]["execution"],
        "chunk_tokens_per_request": cfg["workload"]["chunk_tokens_per_request"],
        "cache_source": "synthetic_hbm",
        "config_hash": digest(cfg),
    }
    try:
        case.update(normalize_prefix(length, requested, case["yoco_mode"], case["keep_tokens"], cfg["workload"]["alignment_policy"]))
        trajectory = list(steps(case))
        case.update(
            status="PLANNED",
            chunk_count=len(trajectory),
            max_rows_per_rank=max(s["producer_padded_rows_per_rank"] for s in trajectory),
            logical_cache_blocks=1 + batch * ((length + 127) // 128),
            block_kv_payload_lower_bound_bytes=batch * ((length + 127) // 128) * 114048,
            capacity_note="Block KV estimate only; native allocator padding, weights, Engram, state and workspaces excluded",
        )
    except ValueError as error:
        case.update(status="INVALID_CONFIG", cache_hit_rate_requested=requested, error=str(error))
    case["case_id"] = digest(case)[:20]
    return case


def plan(cfg: dict) -> list[dict]:
    """Plan the Cartesian grid, or an explicit JSONL case ledger when requested."""
    cases_file = cfg["workload"].get("cases_file")
    if cases_file:
        specs = _case_specs_from_file(cfg, cases_file)
    else:
        specs = (
            (strategy, length, requested)
            for strategy in cfg["parallel"]["strategies"]
            for length in cfg["workload"]["context_lengths"]
            for requested in cfg["workload"]["cache_hit_rates"]
        )
    return [_build_case(cfg, strategy, length, requested) for strategy, length, requested in specs]


def case_capacity(case: dict, mtp_draft_length: int) -> int:
    """Token bound the native model is built with for this case (per rank)."""
    return max(case["per_dp_batch_size"] * (mtp_draft_length + 1), case["cp"] * case["max_rows_per_rank"])


def batch_groups(cases: list[dict], cfg: dict) -> list[dict]:
    """Split planned cases into groups that share one native model load.

    Model construction sizes communication workspaces by max_num_batched_tokens,
    so a group is built for the largest capacity it contains. ``power2`` keeps a
    group inside a 2x capacity band (long prompts cannot OOM short ones in the
    same load); ``single`` uses one group per strategy at the largest capacity.
    """
    grouping = cfg["batch"]["capacity_grouping"]
    limit = cfg["batch"]["max_cases_per_group"]
    buckets: dict[tuple[str, int], list[dict]] = {}
    order: list[tuple[str, int]] = []
    for case in cases:
        if case.get("status") != "PLANNED":
            continue
        capacity = case_capacity(case, cfg["runtime"]["mtp_draft_length"])
        band = 0 if grouping == "single" else 1 << (capacity - 1).bit_length()
        key = (case["strategy"], band)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(case)
    groups = []
    for key in order:
        members = buckets[key]
        chunks = [members] if not limit else [members[i : i + limit] for i in range(0, len(members), limit)]
        for chunk in chunks:
            groups.append(
                {
                    "strategy": key[0],
                    "capacity": max(case_capacity(case, cfg["runtime"]["mtp_draft_length"]) for case in chunk),
                    "cases": chunk,
                }
            )
    strategy_order = {name: index for index, name in enumerate(cfg["parallel"]["strategies"])}
    groups.sort(key=lambda group: (strategy_order[group["strategy"]], group["capacity"]))
    return groups


def rank_mapping(case: dict, devices: list[int]) -> list[dict]:
    return [
        {
            "global_rank": r,
            "dp_rank": r // case["cp"],
            "cp_rank": r % case["cp"],
            "device": devices[r],
            "visible_devices": devices[(r // case["cp"]) * case["cp"] : (r // case["cp"] + 1) * case["cp"]],
        }
        for r in range(8)
    ]
