"""Optional ALOP/aclprof C-level profiling backend.

This backend is intentionally a thin adapter around the ALOP implementation that
already exists in ds41_mini.  It collects the C-level timeline (task_time plus
AICore PipeUtilization metrics by default; no torch_npu Python profiler, no
op_attr, no module shape hooks), then exports the raw PROF_* tree with
``msprof --export`` so the usual MindStudio CSV/JSON files are available.

The adapter is selected with::

    "profile": {"backend": "alop", ...}

and requires either ``profile.alop_root`` or ``PREFILL950_ALOP_ROOT`` to point at
a checkout containing ``ds41mini/tracing/alop.py``.  Keeping ALOP optional means
the standalone benchmark still works without that checkout.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _find_msprof() -> str | None:
    for candidate in (
        os.environ.get("MSPROF", ""),
        shutil.which("msprof") or "",
        "/usr/local/Ascend/cann-9.2.0/bin/msprof",
        "/usr/local/Ascend/cann-9.2.0/tools/profiler/bin/msprof",
    ):
        if candidate and Path(candidate).exists():
            return candidate
    return None


class AlopProfiler:
    """Small start/step/stop adapter with the same interface as profiler.py."""

    def __init__(self, directory: Path, cfg: dict, rank: int) -> None:
        self.directory = Path(directory)
        # worker.py passes the profile section itself, not the whole config.
        self.profile_cfg = cfg.get("profile", cfg)
        self.rank = rank
        self._prof = None
        self._steps = 0
        self._valid = False
        self._exported = False
        self._error = ""

    def _load_profiler(self) -> Any:
        profile_cfg = self.profile_cfg
        root = (
            profile_cfg.get("alop_root")
            or os.environ.get("PREFILL950_ALOP_ROOT")
            or ""
        )
        root = str(root).strip()
        if not root:
            raise RuntimeError(
                "profile.backend=alop requires profile.alop_root or "
                "PREFILL950_ALOP_ROOT pointing at the ds41_mini checkout"
            )
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            from ds41mini.tracing.alop import ALOPConfig, ALOPProfiler
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(
                f"cannot import ALOP from {root!r}: {type(error).__name__}: {error}"
            ) from error

        cfg = ALOPConfig()
        cfg.aicore_metric = "PipeUtilization"
        # C-level task timeline only: no Python profiler, no shape/tensor
        # attributes.  AICore metrics (PMU) are enabled by default because
        # MindStudio's unified DB import expects TASK_PMU_INFO; ds41 uses the
        # same task_time + aicore_metrics + acl_api preset.
        cfg.data_types.update(
            {
                "acl_api": True,
                "task_time": True,
                "aicore_metrics": bool(profile_cfg.get("alop_aicore_metrics", True)),
                "aicpu": False,
                "l2cache": False,
                "hccl_trace": False,
                "training_trace": False,
                "msproftx": False,
                "runtime_api": False,
                "task_time_l0": False,
                "task_memory": False,
                "task_time_l2": False,
                "op_attr": False,
                "task_time_l3": False,
            }
        )
        cfg.python_profiling["enabled"] = False
        cfg.python_profiling["record_shapes"] = False
        self.directory.mkdir(parents=True, exist_ok=True)
        return ALOPProfiler(
            cfg,
            cli_overrides={"out_dir": str(self.directory)},
        )

    def start(self) -> None:
        self._prof = self._load_profiler()
        import torch

        device = int(torch.npu.current_device())
        self._prof.start(device=device)
        if not getattr(self._prof, "is_started", False):
            raise RuntimeError("ALOP profiler did not start")
        self._valid = True

    def step(self) -> None:
        if self._prof is not None:
            self._prof.step()
            self._steps += 1

    def _export(self) -> None:
        if not self.profile_cfg.get("alop_export", True):
            return
        # Non-exporting ranks record only to keep CP overhead symmetric and
        # have their raw directory removed by worker.py; don't spend time
        # exporting that disposable data.
        if self.directory.name == "profiler_discard":
            return
        msprof = _find_msprof()
        if msprof is None:
            self._error = "msprof not found; raw PROF_* retained"
            return
        exported = False
        for prof_dir in sorted(self.directory.glob("PROF_*")):
            out = prof_dir / "mindstudio_profiler_output"
            if out.is_dir():
                exported = True
                continue
            try:
                subprocess.run(
                    [
                        msprof,
                        "--export=on",
                        f"--output={prof_dir}",
                        "--type=text",
                    ],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=600,
                )
            except Exception as error:  # noqa: BLE001
                self._error = f"msprof export failed: {type(error).__name__}: {error}"
                continue
            if out.is_dir():
                exported = True
        if not exported:
            self._error = self._error or "ALOP produced no exported PROF_* output"
        self._exported = exported

    def stop(self) -> None:
        if self._prof is None:
            return
        try:
            self._prof.stop()
            self._prof.finalize()
        finally:
            self._export()
            self._prof = None

    @property
    def valid(self) -> bool:
        return self._valid

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def exported(self) -> bool:
        return self._exported

    @property
    def error(self) -> str:
        return self._error
