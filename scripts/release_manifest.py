#!/usr/bin/env python3
"""Create or verify a deterministic SHA-256 release-file manifest."""

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
    strict_json_load,
    strict_json_write,
)


SCHEMA_VERSION = "release-file-manifest-v1"


def build_manifest(root: Path, excluded: set[str] | None = None) -> dict[str, Any]:
    root = root.resolve()
    excluded = excluded or set()
    records = []
    for path in repository_files(root, exact_excludes=excluded):
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "files": records,
        "generated_by": "scripts/release_manifest.py",
        "root": ".",
        "schema_version": SCHEMA_VERSION,
    }


def verify_manifest(
    root: Path,
    manifest_path: Path,
    *,
    allow_unlisted: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    payload = strict_json_load(manifest_path)
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise IntegrityError("unsupported or missing release manifest schema")
    records = payload.get("files")
    if not isinstance(records, list):
        raise IntegrityError("release manifest files must be a list")
    manifest_relative: str | None = None
    try:
        manifest_relative = manifest_path.resolve().relative_to(root).as_posix()
    except ValueError:
        pass
    expected_paths: set[str] = set()
    mismatches: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, dict):
            raise IntegrityError("release manifest file record must be an object")
        relative = safe_relative_path(str(record.get("path", "")))
        relative_text = relative.as_posix()
        if relative_text in expected_paths:
            raise IntegrityError(f"duplicate manifest path: {relative_text}")
        expected_paths.add(relative_text)
        target = root / relative
        if not target.is_file() or target.is_symlink():
            mismatches.append({"path": relative_text, "problem": "missing_or_not_regular"})
            continue
        actual_hash = sha256_file(target)
        actual_size = target.stat().st_size
        if actual_hash != record.get("sha256") or actual_size != record.get("size_bytes"):
            mismatches.append(
                {
                    "actual_sha256": actual_hash,
                    "actual_size_bytes": actual_size,
                    "expected_sha256": record.get("sha256"),
                    "expected_size_bytes": record.get("size_bytes"),
                    "path": relative_text,
                    "problem": "content_mismatch",
                }
            )
    unlisted: list[str] = []
    if not allow_unlisted:
        excludes = {manifest_relative} if manifest_relative else set()
        actual_paths = {
            path.relative_to(root).as_posix()
            for path in repository_files(root, exact_excludes=excludes)
        }
        unlisted = sorted(actual_paths - expected_paths)
        missing_from_tree = sorted(expected_paths - actual_paths)
        for relative_text in missing_from_tree:
            if not any(item["path"] == relative_text for item in mismatches):
                mismatches.append({"path": relative_text, "problem": "not_in_inventory"})
    return {
        "allow_unlisted": allow_unlisted,
        "files_checked": len(records),
        "mismatches": mismatches,
        "schema_version": SCHEMA_VERSION,
        "unlisted": unlisted,
        "valid": not mismatches and not unlisted,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--root", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--allow-unlisted", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "create":
            root = args.root.resolve()
            exclusions: set[str] = set()
            try:
                exclusions.add(args.output.resolve().relative_to(root).as_posix())
            except ValueError:
                pass
            manifest = build_manifest(root, exclusions)
            strict_json_write(args.output, manifest)
            result = {"files": len(manifest["files"]), "output": str(args.output), "valid": True}
        else:
            result = verify_manifest(
                args.root,
                args.manifest,
                allow_unlisted=args.allow_unlisted,
            )
    except IntegrityError as exc:
        raise SystemExit(str(exc)) from exc
    print(strict_json_dumps(result), end="")
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

