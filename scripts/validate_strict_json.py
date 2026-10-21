#!/usr/bin/env python3
"""Validate JSON with standards-compliant constants and unique object keys."""

from __future__ import annotations

import argparse
from pathlib import Path

from release_lib import IntegrityError, repository_files, strict_json_dumps, strict_json_load


def validate_paths(paths: list[Path]) -> list[Path]:
    validated: list[Path] = []
    for supplied in paths:
        if supplied.is_dir():
            candidates = [path for path in repository_files(supplied) if path.suffix == ".json"]
        elif supplied.suffix == ".json":
            candidates = [supplied]
        else:
            candidates = []
        for candidate in candidates:
            strict_json_load(candidate)
            validated.append(candidate)
    return sorted(set(path.resolve() for path in validated))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        validated = validate_paths(args.paths)
    except IntegrityError as exc:
        raise SystemExit(str(exc)) from exc
    print(strict_json_dumps({"strict_json_files": len(validated), "valid": True}), end="")


if __name__ == "__main__":
    main()

