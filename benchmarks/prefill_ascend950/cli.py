"""python -m benchmarks.prefill_ascend950.cli --help"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load, resolve
from .io_utils import write_json, write_jsonl


def device_list(value: str) -> list[int]:
    try:
        values = [int(x) for x in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("devices must be comma-separated integers") from error
    if len(values) != 8 or len(set(values)) != 8 or min(values) < 0:
        raise argparse.ArgumentTypeError("Exactly eight distinct nonnegative device IDs are required")
    return values


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="DeepInfer native V4.1 Flash Prefill benchmark on 8 x Ascend 950")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("plan", "doctor", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, help="JSON configuration; omitted sections use strict defaults")
        command.add_argument("--weights", choices=("dummy", "real"))
        command.add_argument(
            "--out", type=Path, required=name == "run", default=Path("results/prefill950_doctor") if name == "doctor" else None
        )
        if name != "plan":
            command.add_argument("--devices", type=device_list, default=device_list("0,1,2,3,4,5,6,7"))
            command.add_argument("--model-dir", type=Path, required=True)
        if name == "run":
            command.add_argument("--resume", action="store_true")
            command.add_argument(
                "--batch-cases",
                action=argparse.BooleanOptionalAction,
                default=None,
                help="Load the native model once per capacity group and run many cases (overrides batch.enabled)",
            )
    command = commands.add_parser("summarize")
    command.add_argument("--run", type=Path, required=True)
    command.add_argument("--no-plots", action="store_true")
    command = commands.add_parser("compare")
    command.add_argument("--reference", type=Path, required=True)
    command.add_argument("--candidate", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command = commands.add_parser("prepare-model")
    command.add_argument("--model-dir", type=Path, required=True)
    command.add_argument("--weights", choices=("dummy", "real"), default="dummy")
    command.add_argument("--offline", action="store_true")
    command = commands.add_parser("deployment-manifest")
    command.add_argument("--out", type=Path, required=True)
    command = commands.add_parser("_probe", help=argparse.SUPPRESS)
    command.add_argument("--out", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command in ("plan", "doctor", "run"):
            cfg = load(args.config) if args.config else resolve({})
            if args.weights:
                cfg["runtime"]["weights"] = args.weights
            if args.command == "run" and args.batch_cases is not None:
                cfg["batch"]["enabled"] = args.batch_cases
        if args.command == "plan":
            from .planner import plan, steps

            cases = plan(cfg)
            result = {
                "resolved_config": cfg,
                "case_count": len(cases),
                "cases": cases,
                "first_step_per_case": {c["case_id"]: next(steps(c)) for c in cases if c["status"] == "PLANNED"},
                "note": "No accelerator initialized. Capacity estimates exclude weights, Engram, states and workspaces.",
            }
            if args.out:
                args.out.mkdir(parents=True, exist_ok=True)
                write_json(args.out / "resolved_config.json", cfg)
                write_jsonl(args.out / "cases.jsonl", cases)
                write_json(args.out / "plan.json", result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return int(any(c["status"] != "PLANNED" for c in cases))
        if args.command == "doctor":
            from .doctor import check

            result = check(cfg, args.devices, args.model_dir.resolve(), args.out.resolve())
            print(
                json.dumps(
                    {"ok": result["ok"], "errors": result["errors"], "report": str(args.out / "doctor.json")}, ensure_ascii=False, indent=2
                )
            )
            return int(not result["ok"])
        if args.command == "run":
            from .launcher import run

            result = run(cfg, args.devices, args.model_dir, args.out, args.resume)
            print(json.dumps(result, indent=2))
            return int(not result["ok"])
        if args.command == "summarize":
            from .report import summarize

            print(json.dumps(summarize(args.run, plots=not args.no_plots), indent=2))
            return 0
        if args.command == "compare":
            from .report import compare

            result = compare(args.reference, args.candidate)
            write_json(args.out, result)
            print(json.dumps(result, indent=2))
            return int(
                not result["comparisons"]
                or result["reference_only"]
                or result["candidate_only"]
                or any(c["status"] != "PASS" for c in result["comparisons"])
            )
        if args.command == "prepare-model":
            from .deployment import prepare_model

            print(json.dumps(prepare_model(args.model_dir, args.weights, args.offline), indent=2))
            return 0
        if args.command == "deployment-manifest":
            from .deployment import deployment_manifest

            deployment_manifest(args.out)
            return 0
        if args.command == "_probe":
            from .doctor import hardware_probe

            hardware_probe(args.out)
            return 0
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
