"""Rank-aware aggregation, comparable workloads and machine-readable reports."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .config import digest
from .io_utils import write_json


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    index = (len(values) - 1) * q
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] * (hi - index) + values[hi] * (index - lo) if lo != hi else values[lo]


def distribution(values: list[float]) -> dict:
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("Samples must be finite, positive and nonempty")
    mean = statistics.mean(values)
    std = statistics.pstdev(values)
    return {
        "count": len(values),
        "mean_ms": mean,
        "p50_ms": percentile(values, 0.5),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "min_ms": min(values),
        "max_ms": max(values),
        "std_ms": std,
        "cv": std / mean,
        "quantile_method": "linear_(n-1)q",
        "p99_reliable": len(values) >= 100,
    }


def aggregate(case: dict, cfg: dict, ranks: list[dict]) -> dict:
    if len(ranks) != 8 or {x["rank"] for x in ranks} != set(range(8)):
        raise ValueError("Exactly eight distinct rank outcomes are required")
    chunks = case["chunk_count"]
    expected = {
        (r, k, j) for r in range(cfg["measurement"]["rounds"]) for k in range(cfg["measurement"]["iterations"]) for j in range(chunks)
    }
    by_step = defaultdict(list)
    for outcome in ranks:
        samples = outcome["samples"]
        counts = Counter((s["round"], s["iteration"], s["chunk"]) for s in samples)
        if set(counts) != expected or any(v != 1 for v in counts.values()):
            raise ValueError(f"Incomplete or duplicate samples on rank {outcome['rank']}")
        for sample in samples:
            ms = sample["forward_ms"]
            if not math.isfinite(ms) or ms <= 0:
                raise ValueError("Invalid event timing")
            by_step[(sample["round"], sample["iteration"], sample["chunk"])].append(ms)
    chains = defaultdict(float)
    step_ms = []
    for (r, k, _), spans in sorted(by_step.items()):
        span = max(spans)
        step_ms.append(span)
        chains[(r, k)] += span
    times = list(chains.values())
    stats = distribution(times)
    tokens = case["global_batch_size"] * case["new_tokens_per_request"] * len(times)
    seconds = sum(times) / 1000
    profile_ok = not cfg["profile"]["enabled"] or all(r["profile"]["valid"] for r in ranks)
    graph_ok = not cfg["alignment"]["enabled"] or all(r["graph_alignment"]["match"] for r in ranks)
    load = [r["expert_assignments"] for r in ranks]
    result = dict(case)
    result.update(
        status="OK" if profile_ok else "timing_ok_profile_failed",
        timing_ok=True,
        timing_mode=ranks[0].get("timing_mode", "per_forward"),
        timing_scope="native_model_forward",
        timing_statistic="batch_event_span_divided_by_forward_count",
        forward=stats,
        forward_step=distribution(step_ms),
        forward_p50_ms=stats["p50_ms"],
        forward_p95_ms=stats["p95_ms"],
        new_tokens_per_s=tokens / seconds,
        new_tokens_per_s_per_card=tokens / seconds / 8,
        requests_per_s=case["global_batch_size"] * len(times) / seconds,
        logical_input_tokens_per_s=case["global_batch_size"] * case["context_length"] * len(times) / seconds,
        observed_moe_lb=max(load) / statistics.mean(load) if sum(load) else None,
        graph_alignment_status=("NATIVE_PATH_REPEATABLE" if graph_ok else "PATH_MISMATCH") if cfg["alignment"]["enabled"] else "NOT_RUN",
        performance_alignment_status="NOT_MEASURED_AGAINST_EXTERNAL_REFERENCE",
        peak_torch_allocated_bytes=max(r["peak_torch_allocated_bytes"] for r in ranks),
        peak_torch_reserved_bytes=max(r["peak_torch_reserved_bytes"] for r in ranks),
        minimum_device_free_bytes=min(r["minimum_device_free_bytes"] for r in ranks),
        rank_outcomes=[f"rank_{r}/outcome.json" for r in range(8)],
    )
    if not graph_ok:
        result.update(status="PATH_MISMATCH", timing_ok=False)
    return result


def comparison_key(case: dict, cfg: dict) -> str:
    batch = case["global_batch_size"] if case["batch_mode"] == "fixed_global" else case["per_dp_batch_size"]
    return digest(
        {
            "L": case["context_length"],
            "C": case["cached_tokens_effective"],
            "batch_mode": case["batch_mode"],
            "batch": batch,
            "execution": case["execution"],
            "chunk": case["chunk_tokens_per_request"],
            "runtime": cfg["runtime"],
            "yoco": cfg["yoco"],
            "moe": cfg["moe"],
            "timing_scope": cfg["measurement"]["timing_scope"],
            "cache": case["cache_source"],
        }
    )


def csv_file(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(run: Path, plots: bool = True) -> dict:
    results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((run / "cases").glob("*/result.json"))]
    cfg = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))
    rows, groups = [], defaultdict(list)
    fields = [
        "case_id",
        "strategy",
        "status",
        "context_length",
        "cache_hit_rate_requested",
        "cache_hit_rate_effective",
        "cached_tokens_effective",
        "new_tokens_per_request",
        "global_batch_size",
        "per_dp_batch_size",
        "batch_mode",
        "execution",
        "chunk_count",
        "timing_mode",
        "forward_p50_ms",
        "forward_p95_ms",
        "new_tokens_per_s",
        "new_tokens_per_s_per_card",
        "observed_moe_lb",
        "peak_torch_allocated_bytes",
        "minimum_device_free_bytes",
        "graph_alignment_status",
        "performance_alignment_status",
        "error",
    ]
    for result in results:
        rows.append({k: result.get(k) for k in fields})
        if result.get("timing_ok"):
            groups[comparison_key(result, cfg)].append(result)
    best = []
    for key, cases in groups.items():
        latency = min(x["forward_p50_ms"] for x in cases)
        throughput = max(x["new_tokens_per_s"] for x in cases)
        # Do not force a winner when differences are inside observed noise (at least 1%).
        tolerance = max(0.01, max(x["forward"]["cv"] for x in cases))
        first = cases[0]
        best.append(
            {
                "comparison_key": key,
                "context_length": first["context_length"],
                "cached_tokens_effective": first["cached_tokens_effective"],
                "batch_mode": first["batch_mode"],
                "latency_best_or_tied": ";".join(x["strategy"] for x in cases if x["forward_p50_ms"] <= latency * (1 + tolerance)),
                "p95_best_or_tied": ";".join(
                    x["strategy"] for x in cases if x["forward_p95_ms"] <= min(c["forward_p95_ms"] for c in cases) * (1 + tolerance)
                ),
                "throughput_best_or_tied": ";".join(x["strategy"] for x in cases if x["new_tokens_per_s"] >= throughput / (1 + tolerance)),
                "pareto_strategies": ";".join(
                    x["strategy"]
                    for x in cases
                    if not any(
                        y["forward_p50_ms"] <= x["forward_p50_ms"]
                        and y["new_tokens_per_s"] >= x["new_tokens_per_s"]
                        and (
                            y["forward_p50_ms"] < x["forward_p50_ms"] / (1 + tolerance)
                            or y["new_tokens_per_s"] > x["new_tokens_per_s"] * (1 + tolerance)
                        )
                        for y in cases
                    )
                ),
                "relative_noise_tolerance": tolerance,
                "eligible_strategies": len(cases),
                "missing_strategies": ";".join(s for s in cfg["parallel"]["strategies"] if s not in {x["strategy"] for x in cases}),
            }
        )
    csv_file(run / "summary.csv", rows)
    csv_file(run / "best_strategy.csv", best)
    write_json(run / "summary.json", {"cases": results, "best": best, "statuses": dict(Counter(x["status"] for x in results))})
    if plots and groups:
        try:
            _plots(run, groups)
        except ImportError as error:
            write_json(run / "plot_status.json", {"status": "UNAVAILABLE", "reason": str(error)})
    return {"cases": len(results), "statuses": dict(Counter(x["status"] for x in results)), "comparison_groups": len(groups)}


def _plots(run: Path, groups: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strategies = list(__import__("benchmarks.prefill_ascend950.config", fromlist=["STRATEGIES"]).STRATEGIES)
    ordered = sorted(groups.values(), key=lambda g: (g[0]["context_length"], g[0]["cached_tokens_effective"]))
    for field, name in (("forward_p50_ms", "latency"), ("new_tokens_per_s", "throughput")):
        data = [[next((c[field] for c in group if c["strategy"] == s), float("nan")) for s in strategies] for group in ordered]
        labels = [f"L={g[0]['context_length']} C={g[0]['cached_tokens_effective']}" for g in ordered]
        fig, ax = plt.subplots(figsize=(10, max(3, len(labels) * 0.3)))
        plotted = ax.imshow(data, aspect="auto")
        ax.set_xticks(range(4), strategies)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title(field + " (missing points masked)")
        fig.colorbar(plotted, ax=ax)
        fig.tight_layout()
        fig.savefig(run / f"heatmap_{name}.png", dpi=140)
        plt.close(fig)


def compare(reference: Path, candidate: Path) -> dict:
    ref_manifest = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))
    candidate_manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
    for key in ("environment_id", "model_id", "native_source_id"):
        if key not in ref_manifest or ref_manifest[key] != candidate_manifest.get(key):
            raise ValueError(f"Incompatible manifests: {key}")

    def index(run: Path) -> dict:
        cfg = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))
        cases = [json.loads(p.read_text(encoding="utf-8")) for p in (run / "cases").glob("*/result.json")]
        return {(comparison_key(c, cfg), c["strategy"]): c for c in cases if c.get("timing_ok")}

    a, b = index(reference), index(candidate)
    output = []
    for key in sorted(a.keys() & b.keys()):
        x, y = a[key], b[key]
        delta = abs(y["forward_p50_ms"] / x["forward_p50_ms"] - 1)
        p95_delta = abs(y["forward_p95_ms"] / x["forward_p95_ms"] - 1)
        stable = max(x["forward"]["cv"], y["forward"]["cv"]) <= 0.03
        enough = min(x["forward"]["count"], y["forward"]["count"]) >= 60
        output.append(
            {
                "reference_case": x["case_id"],
                "candidate_case": y["case_id"],
                "strategy": y["strategy"],
                "median_relative_error": delta,
                "p95_relative_error": p95_delta,
                "status": ("PASS" if delta <= 0.05 and p95_delta <= 0.1 else "FAIL") if stable and enough else "INCONCLUSIVE",
            }
        )
    return {
        "comparisons": output,
        "reference_only": len(a.keys() - b.keys()),
        "candidate_only": len(b.keys() - a.keys()),
        "scope": "Timing thresholds only; independent native metadata/kernel/communication review is required for full performance parity",
        "note": "Explicit external run comparison; source reuse alone does not establish performance parity.",
    }
