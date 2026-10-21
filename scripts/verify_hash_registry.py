#!/usr/bin/env python3
"""Verify hashes/appendix_d_registry.json against the released tree (standard library only).

Every entry whose status is 'released' must hash to its registered digest.
'released_superset' / 'released_redacted' entries must hash to their released_sha256.
'not_redistributed_*' entries are checked only when a local copy exists at release_path
(for example after the public datasets were downloaded into work/data/); otherwise they
are reported as absent, which is the expected state of a fresh clone.
With --markdown the script prints the registry as a table (used for docs/HASH_REGISTRY.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(registry: dict, root: Path) -> tuple[list[dict], list[str]]:
    rows, failures = [], []
    for entry in registry["entries"]:
        path = root / entry["release_path"]
        status = entry["status"]
        expected = entry.get("released_sha256", entry["sha256"]) if status.startswith("released") else entry["sha256"]
        if path.is_file():
            actual = sha256_file(path)
            ok = actual == expected
            result = "verified" if ok else "MISMATCH"
            if not ok:
                failures.append(f"{entry['release_path']}: sha256 {actual[:12]} != expected {expected[:12]}")
        else:
            if status.startswith("released"):
                result = "MISSING"
                failures.append(f"{entry['release_path']}: released file missing")
            else:
                result = "absent (not redistributed)"
        rows.append({"object": entry["object"], "sha256": entry["sha256"], "status": status, "release_path": entry["release_path"], "result": result})
    return rows, failures


def markdown(registry: dict, rows: list[dict]) -> str:
    out = ["# Hash registry", "", registry["description"], "",
           "Status values: `released` (bytes in this repository hash to the digest), `released_superset` / `released_redacted` (released file differs from the frozen bytes for the reason given in `hashes/appendix_d_registry.json`), `not_redistributed_*` (bytes are not in this repository).", ""]
    for section, title in (("appendix_d", "Appendix D of the manuscript (24 digests, in order)"), ("supplementary", "Supplementary frozen digests")):
        out += [f"## {title}", "", "| Object | SHA-256 | Status | Path |", "|---|---|---|---|"]
        for entry, row in zip(registry["entries"], rows):
            if entry["section"] != section:
                continue
            out.append(f"| {entry['object']} | `{entry['sha256']}` | {row['status']} | `{entry['release_path']}` |")
        out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=ROOT / "hashes" / "appendix_d_registry.json")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--markdown", action="store_true", help="print the registry as a Markdown table instead of verifying")
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    rows, failures = verify(registry, args.root.resolve())
    if args.markdown:
        print(markdown(registry, rows))
        return 0
    for row in rows:
        print(f"{row['result']:<28} {row['status']:<36} {row['release_path']}")
    print(f"\n{len(rows)} entries; {sum(r['result'] == 'verified' for r in rows)} verified; "
          f"{sum(r['result'].startswith('absent') for r in rows)} absent (not redistributed)")
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print(" -", f)
        return 1
    print("HASH REGISTRY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
