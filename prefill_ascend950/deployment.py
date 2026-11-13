"""Pinned model preparation and build provenance."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

from .config import MODEL_ID, MODEL_REVISION
from .doctor import ROOT, command, model_manifest, source_manifest
from .io_utils import file_hash, write_json


def prepare_model(directory: Path, weights: str, offline: bool) -> dict:
    directory = directory.resolve()
    if not offline:
        from huggingface_hub import snapshot_download

        patterns = ["*.json", "*.model", "tokenizer*", "special_tokens_map*", "*.tiktoken"]
        snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=str(directory), allow_patterns=patterns)
    manifest = model_manifest(directory, weights)
    # Store provenance beside (not inside) the model, keeping model identity stable.
    write_json(directory.parent / f"{directory.name}.prefill950-model.json", manifest)
    return manifest


def deployment_manifest(output: Path) -> None:
    wheels = ROOT / "build" / f"ascend-{platform.machine()}" / "vendor-wheels"
    binaries = ROOT / "deepinfer" / "_so" / f"ascend-{platform.machine()}"
    write_json(
        output,
        {
            "python": sys.version,
            "executable": sys.executable,
            "source": source_manifest(),
            "pip_freeze": command([sys.executable, "-m", "pip", "freeze"]),
            "pip_check": command([sys.executable, "-m", "pip", "check"]),
            "wheel_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(wheels.glob("*.whl"))},
            "native_extension_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(binaries.glob("*.so"))},
            "cann_home": os.environ.get("ASCEND_HOME_PATH"),
            "status": "BUILT_PREFLIGHT_REQUIRED",
        },
    )
