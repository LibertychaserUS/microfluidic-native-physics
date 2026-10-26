"""Standard-library helpers for public-release integrity tooling."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable


IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".idea",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".vscode",
    "__pycache__",
    "build",
    "dist",
    "downloads",
    "env",
    "runs",
    "scratch",
    "venv",
}

IGNORED_RELATIVE_PREFIXES = {
    "artifacts/fast",
    "artifacts/local",
    "artifacts/superseded",
    "artifacts/tmp",
    "data/cache",
    "data/raw",
    "work/data/osf_938rs",
    "work/data/zenodo_7055018",
    "work/data/zenodo_17867478",
    "work/data/h_A1",
    "work/data/h_A2",
    "work/data/h_B1",
    "work/data/h_B2",
    "work/data/h_R2A1",
    "work/data/h_R2A2",
    "work/data/upstream",
}


class IntegrityError(ValueError):
    """Raised when an artifact violates a release-integrity contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    # MD5 is used only to compare with an upstream record, never for security.
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(token: str) -> None:
    raise IntegrityError(f"non-standard JSON numeric constant: {token}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IntegrityError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _assert_finite(value: Any, location: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise IntegrityError(f"non-finite JSON number at {location}")
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite(child, f"{location}[{index}]")


def strict_json_load(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(
                stream,
                parse_constant=_reject_constant,
                object_pairs_hook=_unique_object,
            )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"invalid strict JSON in {path}: {exc}") from exc
    _assert_finite(value)
    return value


def strict_json_dumps(value: Any) -> str:
    _assert_finite(value)
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def strict_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(strict_json_dumps(value), encoding="utf-8")
    temporary.replace(path)


def safe_relative_path(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise IntegrityError(f"unsafe relative path: {value!r}")
    return candidate


def _ignored(relative: Path, exact_excludes: set[str]) -> bool:
    relative_text = relative.as_posix()
    if relative_text in exact_excludes:
        return True
    if any(part in IGNORED_DIRECTORY_NAMES for part in relative.parts):
        return True
    return any(
        relative_text == prefix or relative_text.startswith(prefix + "/")
        for prefix in IGNORED_RELATIVE_PREFIXES
    )


def _git_ignored(root: Path) -> set[str] | None:
    """Relative paths that ``.gitignore`` excludes, or None when ``root`` is not a git checkout.

    Locally regenerated third-party tables, downloaded datasets and run outputs live
    inside the tree but are never part of the publishable inventory."""

    if not (root / ".git").exists():
        return None
    try:
        output = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--others", "--ignored", "--exclude-standard"],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return {item.decode("utf-8") for item in output.split(b"\0") if item}


def repository_files(
    root: Path,
    *,
    exact_excludes: Iterable[str] = (),
) -> list[Path]:
    """Return deterministic, publishable regular files below ``root``."""

    root = root.resolve()
    excludes = {safe_relative_path(item).as_posix() for item in exact_excludes}
    git_ignored = _git_ignored(root)
    files: list[Path] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if _ignored(relative, excludes):
            continue
        if git_ignored is not None and relative.as_posix() in git_ignored:
            continue
        if path.is_symlink():
            raise IntegrityError(f"symbolic links are not permitted: {relative}")
        if path.is_file():
            files.append(path)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())

