"""Pure workload planning. Never import torch or DeepInfer here."""

from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal

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


def plan(cfg: dict) -> list[dict]:
    result = []
    for strategy in cfg["parallel"]["strategies"]:
        dp, cp, ep = STRATEGIES[strategy]
        batch = (
            cfg["parallel"]["global_batch_size"] // dp
            if cfg["parallel"]["batch_mode"] == "fixed_global"
            else cfg["parallel"]["per_dp_batch_size"]
        )
        for length in cfg["workload"]["context_lengths"]:
            for requested in cfg["workload"]["cache_hit_rates"]:
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
                    case.update(
                        normalize_prefix(length, requested, case["yoco_mode"], case["keep_tokens"], cfg["workload"]["alignment_policy"])
                    )
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
                result.append(case)
    return result


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
