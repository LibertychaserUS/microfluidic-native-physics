#!/usr/bin/env python3
"""Compare two result directories by exact relative-path and SHA-256 identity."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from release_lib import (
    IntegrityError,
    repository_files,
    safe_relative_path,
    sha256_file,
    strict_json_dumps,
    strict_json_write,
)


def _inventory(root: Path, excluded: set[str]) -> dict[str, dict[str, object]]:
    root = root.resolve()
    return {
        path.relative_to(root).as_posix(): {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in repository_files(root, exact_excludes=excluded)
    }


def compare_runs(left: Path, right: Path, excluded: set[str]) -> dict[str, Any]:
    left_inventory = _inventory(left, excluded)
    right_inventory = _inventory(right, excluded)
    left_paths = set(left_inventory)
    right_paths = set(right_inventory)
    changed = []
    for path in sorted(left_paths & right_paths):
        if left_inventory[path] != right_inventory[path]:
            changed.append(
                {
                    "left": left_inventory[path],
                    "path": path,
                    "right": right_inventory[path],
                }
            )
    only_left = sorted(left_paths - right_paths)
    only_right = sorted(right_paths - left_paths)
    return {
        "changed": changed,
        "excluded_exact_paths": sorted(excluded),
        "left": str(left.resolve()),
        "left_files": len(left_inventory),
        "only_left": only_left,
        "only_right": only_right,
        "right": str(right.resolve()),
        "right_files": len(right_inventory),
        "schema_version": "deterministic-run-comparison-v1",
        "valid": not changed and not only_left and not only_right,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exact relative path to exclude; glob patterns are intentionally unsupported.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        excluded = {safe_relative_path(value).as_posix() for value in args.exclude}
        result = compare_runs(args.left, args.right, excluded)
    except IntegrityError as exc:
        raise SystemExit(str(exc)) from exc
    if args.output:
        strict_json_write(args.output, result)
    print(strict_json_dumps(result), end="")
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

