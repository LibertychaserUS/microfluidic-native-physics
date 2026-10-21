#!/usr/bin/env python3
"""Require exact versions and SHA-256 hashes in a pip requirements lock."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from release_lib import IntegrityError, strict_json_dumps


HASH = re.compile(r"--hash=sha256:[0-9a-f]{64}\b")
PIN = re.compile(r"^[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?==[^\s\\]+")


def requirement_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    current = ""
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        current = f"{current} {line}".strip()
        if current.endswith("\\"):
            current = current[:-1].strip()
            continue
        blocks.append(current)
        current = ""
    if current:
        raise IntegrityError("unterminated line continuation in requirements lock")
    return blocks


def validate_lock(path: Path) -> int:
    blocks = requirement_blocks(path.read_text(encoding="utf-8"))
    if not blocks:
        raise IntegrityError("requirements lock is empty")
    for block in blocks:
        if block.startswith(("--index-url", "--extra-index-url", "--find-links")):
            continue
        if block.startswith(("-e ", "--editable", "git+", "http://", "https://")):
            raise IntegrityError(f"unverifiable requirement source: {block}")
        if not PIN.match(block):
            raise IntegrityError(f"requirement is not exactly pinned: {block}")
        if not HASH.search(block):
            raise IntegrityError(f"requirement has no SHA-256 hash: {block}")
    return len(blocks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.path.exists() and args.allow_missing:
        result = {"path": str(args.path), "status": "not_present_pre_release", "valid": True}
    elif not args.path.is_file():
        raise SystemExit(f"requirements lock is missing: {args.path}")
    else:
        try:
            count = validate_lock(args.path)
        except IntegrityError as exc:
            raise SystemExit(str(exc)) from exc
        result = {"blocks": count, "path": str(args.path), "valid": True}
    print(strict_json_dumps(result), end="")


if __name__ == "__main__":
    main()

