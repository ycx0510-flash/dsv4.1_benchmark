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
