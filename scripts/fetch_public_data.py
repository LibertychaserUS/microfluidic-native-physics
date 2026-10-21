#!/usr/bin/env python3
"""Download public source files locally and verify recorded checksums."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path
from typing import Any

from release_lib import (
    IntegrityError,
    md5_file,
    safe_relative_path,
    sha256_file,
    strict_json_dumps,
    strict_json_load,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPOSITORY_ROOT / "data" / "source_registry.json"


def _verify_file(path: Path, record: dict[str, Any]) -> dict[str, object]:
    if not path.is_file():
        raise IntegrityError(f"source file is missing: {path}")
    actual_size = path.stat().st_size
    actual_sha256 = sha256_file(path)
    actual_md5 = md5_file(path)
    expected_size = int(record["bytes"])
    expected_sha256 = str(record["sha256"])
    expected_md5 = str(record["md5"]) if "md5" in record else actual_md5
    if (
        actual_size != expected_size
        or actual_sha256 != expected_sha256
        or actual_md5 != expected_md5
    ):
        raise IntegrityError(
            f"checksum mismatch for {path}: "
            f"size={actual_size}, sha256={actual_sha256}, md5={actual_md5}"
        )
    return {
        "md5": actual_md5,
        "path": str(path),
        "sha256": actual_sha256,
        "size_bytes": actual_size,
        "verified": True,
    }


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "microfluidic-native-physics-release/0.3"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with temporary.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(destination)


def fetch_dataset(
    registry: dict[str, Any],
    dataset_name: str,
    destination_root: Path,
    *,
    verify_only: bool,
    force: bool,
) -> list[dict[str, object]]:
    datasets = registry.get("datasets")
    if not isinstance(datasets, dict) or dataset_name not in datasets:
        raise IntegrityError(f"unknown dataset: {dataset_name}")
    dataset = datasets[dataset_name]
    status = dataset.get("status")
    if status == "verify_only_manual_download" and not verify_only:
        raise IntegrityError(
            f"{dataset_name} has no recorded direct download URL; download the files "
            "manually into the destination and re-run with --verify-only"
        )
    if status not in {"download_enabled", "verify_only_manual_download"}:
        raise IntegrityError(f"download disabled for {dataset_name}: {status}")
    files = dataset.get("files")
    if not isinstance(files, list) or not files:
        raise IntegrityError(f"no downloadable files registered for {dataset_name}")
    results = []
    for record in files:
        if not isinstance(record, dict):
            raise IntegrityError("source file record must be an object")
        relative = safe_relative_path(str(record.get("relative_path", "")))
        destination = destination_root / relative
        if verify_only:
            results.append(_verify_file(destination, record))
            continue
        if destination.exists() and not force:
            results.append(_verify_file(destination, record))
            continue
        _download(str(record["url"]), destination)
        try:
            results.append(_verify_file(destination, record))
        except Exception:
            destination.unlink(missing_ok=True)
            raise
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        registry = strict_json_load(args.registry)
        destination = args.destination
        if destination is None:
            destination = REPOSITORY_ROOT / safe_relative_path(
                str(registry["default_destination"])
            )
        results = fetch_dataset(
            registry,
            args.dataset,
            destination.resolve(),
            verify_only=args.verify_only,
            force=args.force,
        )
    except IntegrityError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        strict_json_dumps(
            {
                "dataset": args.dataset,
                "files": results,
                "policy": "local_verified_download_not_committed",
                "valid": True,
            }
        ),
        end="",
    )


if __name__ == "__main__":
    main()

