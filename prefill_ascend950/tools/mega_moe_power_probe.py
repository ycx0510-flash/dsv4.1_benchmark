"""MegaMoE-only power probe.

Runs the fused MegaMoE kernel back-to-back with the model's geometry and prints
per-iteration device time. Use it with an external npu-smi sampler to tell whether
the kernel alone can drive the AICore DVFS / power cap.

Usage:
    source build/prefill950-tools/prefill950-env.sh
    export PREFILL950_JIT_ASCEND_HOME=$PWD/build/prefill950-tools/ascend-toolkit-shim
    export PYTHONPATH=$PWD
    export LD_PRELOAD=$PWD/build/prefill950-tools/compat/libmsprof_compat.so
    python benchmarks/prefill_ascend950/tools/mega_moe_power_probe.py --iters 3000
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import torch
import torch.multiprocessing

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "vendor" / "DeepGEMM-Ascend" / "tests"))


def rank_loop(local_rank: int, args: argparse.Namespace) -> None:
    import torch.distributed as dist

    import test_mega_moe as mega

    _, num_ranks, group = mega.init_dist(local_rank, args.num_processes)
    case = mega.MegaMoECase(
        group,
        args.hidden,
        args.intermediate,
        args.experts,
        args.topk,
        args.max_tokens,
        args.shared,
        activation_clamp=args.clamp,
        with_baseline=False,
    )
    try:
        case.create_inputs(args.tokens)
        for _ in range(args.warmup):
            case.run_fused()
        torch.npu.synchronize()

        starts = [torch.npu.Event(enable_timing=True) for _ in range(args.iters)]
        ends = [torch.npu.Event(enable_timing=True) for _ in range(args.iters)]
        if local_rank == 0:
            print(
                f"PROBE_START {datetime.datetime.now().isoformat()} "
                f"tokens={args.tokens} experts={args.experts} topk={args.topk} iters={args.iters}",
                flush=True,
            )
        for index in range(args.iters):
            starts[index].record()
            case.run_fused()
            ends[index].record()
        torch.npu.synchronize()
        if local_rank == 0:
            print(f"PROBE_END {datetime.datetime.now().isoformat()}", flush=True)
            for index, (start, end) in enumerate(zip(starts, ends)):
                print(f"iter {index:5d} {start.elapsed_time(end):.3f} ms", flush=True)
    finally:
        case.destroy()
        torch.npu.empty_cache()
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="MegaMoE-only power probe")
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--intermediate", type=int, default=2304)
    parser.add_argument("--experts", type=int, default=416)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--shared", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=13120)
    parser.add_argument("--max-tokens", type=int, default=13120)
    parser.add_argument("--clamp", type=float, default=10.0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=3000)
    args = parser.parse_args()
    torch.multiprocessing.spawn(rank_loop, args=(args,), nprocs=args.num_processes)


if __name__ == "__main__":
    main()
