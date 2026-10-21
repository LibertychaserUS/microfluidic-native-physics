"""Load the audited workspace modules without copying or modifying them (K6).

Thread limits are pinned before any numerical import so that every W3 process
is single-threaded and deterministic (K2).
"""

from __future__ import annotations

import os

for _variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import hashlib  # noqa: E402
import importlib  # noqa: E402
import sys  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from types import ModuleType  # noqa: E402


class DatalessFileError(RuntimeError):
    """An iCloud placeholder was found where file content is required (D8)."""


def ensure_materialised(path: Path) -> None:
    stat = os.stat(path)
    if stat.st_size > 0 and getattr(stat, "st_blocks", 1) == 0:
        raise DatalessFileError(
            f"{path} is an iCloud placeholder; run `brctl download` first"
        )


def sha256_file(path: Path) -> str:
    ensure_materialised(path)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Workspace:
    root: Path
    data_dir: Path
    strict_json: ModuleType
    dafd: ModuleType
    core: ModuleType
    intervals: ModuleType
    v2: ModuleType


def load_workspace(root: Path) -> Workspace:
    """Import the workspace `src` modules by name so their internal imports resolve."""

    root = Path(root).resolve()
    src = root / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"workspace src directory not found: {src}")
    for required in (
        "strict_json.py",
        "dafd_reproduction.py",
        "study_core.py",
        "interval_experiments.py",
        "native_physics_v2.py",
    ):
        ensure_materialised(src / required)
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    strict_json = importlib.import_module("strict_json")
    dafd = importlib.import_module("dafd_reproduction")
    core = importlib.import_module("study_core")
    intervals = importlib.import_module("interval_experiments")
    v2 = importlib.import_module("native_physics_v2")
    data_dir = root.parent / "data" / "osf_938rs"
    return Workspace(
        root=root,
        data_dir=data_dir,
        strict_json=strict_json,
        dafd=dafd,
        core=core,
        intervals=intervals,
        v2=v2,
    )
