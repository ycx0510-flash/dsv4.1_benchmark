"""Diagnostic native metadata/module signatures, collected outside timing passes."""

from __future__ import annotations

import hashlib
from typing import Any, Self

from .config import digest

# This diagnostic installs a pre-hook on every module for each pass, so the
# registries hold pass-specific hook ids that are not part of the graph layout.
_HOOK_REGISTRIES = {
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_forward_hooks",
    "_forward_hooks_with_kwargs",
    "_backward_pre_hooks",
    "_backward_hooks",
    "_backward_hooks_with_kwargs",
}


def signature(value: Any, depth: int = 0, seen: set[int] | None = None) -> Any:
    import torch

    if seen is None:
        seen = set()
    if isinstance(value, torch.Tensor):
        result = {"shape": list(value.shape), "dtype": str(value.dtype), "stride": list(value.stride()), "device_type": value.device.type}
        if value.device.type == "cpu":
            raw = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            result["cpu_value_sha256"] = hashlib.sha256(raw).hexdigest()
        return result
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    if depth > 6 or id(value) in seen:
        return {"type": type(value).__qualname__}
    seen = seen | {id(value)}
    if isinstance(value, dict):
        return {
            str(k): signature(v, depth + 1, seen)
            for k, v in sorted(value.items(), key=lambda x: str(x[0]))
            if str(k) not in _HOOK_REGISTRIES
        }
    if isinstance(value, (tuple, list)):
        return [signature(v, depth + 1, seen) for v in value]
    if hasattr(value, "__dict__"):
        return {"type": type(value).__qualname__, "fields": signature(vars(value), depth + 1, seen)}
    return {"type": type(value).__qualname__}


class NativeDiagnostic:
    def __init__(self, runner: Any) -> None:
        self.runner = runner
        self.modules, self.metadata, self.handles = [], [], []

    def __enter__(self) -> Self:
        self.original = self.runner._prepare_input_tensors

        def prepare(*args: Any, **kwargs: Any) -> Any:
            result = self.original(*args, **kwargs)
            self.metadata.append(signature(result))
            return result

        self.runner._prepare_input_tensors = prepare
        for name, module in self.runner.model.named_modules():

            def record(mod: Any, args: tuple, kwargs: dict, name: str = name) -> None:
                # KVCacheInterface is huge and mutates. Record tensor inputs and native
                # metadata layouts without traversing HBM cache storage or synchronizing.
                self.modules.append(
                    {
                        "name": name,
                        "type": type(mod).__qualname__,
                        "args": signature(args),
                        "kwargs": signature({k: v for k, v in kwargs.items() if k != "kv_caches"}),
                    }
                )

            self.handles.append(module.register_forward_pre_hook(record, with_kwargs=True))
        return self

    def __exit__(self, *_: Any) -> None:
        self.runner._prepare_input_tensors = self.original
        for handle in self.handles:
            handle.remove()

    def result(self) -> dict:
        result = {"cpu_metadata": self.metadata, "module_calls": self.modules}
        return dict(result, sha256=digest(result))
