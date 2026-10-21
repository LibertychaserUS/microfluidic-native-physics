#!/usr/bin/env python3
"""Check the five canonical run pairs from their artifact_hashes.json manifests (standard library only).

For each pair (run1, run2) under work/microfluidic_reliability/ the two manifests must be
byte-identical, and every released file listed in a manifest must re-hash to its entry.
Files listed in a manifest but not present in the release are the per-row artifacts that
stay in the authors' workspace; they are counted and listed, never silently passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work" / "microfluidic_reliability"
PAIRS = ["v2", "expanded", "w3", "w4", "w5"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=WORK)
    args = parser.parse_args()
    failures: list[str] = []
    for suite in PAIRS:
        a = args.work / f"{suite}_artifacts_canonical_run1" / "artifact_hashes.json"
        b = args.work / f"{suite}_artifacts_canonical_run2" / "artifact_hashes.json"
        ta, tb = a.read_bytes(), b.read_bytes()
        identical = ta == tb
        if not identical:
            failures.append(f"{suite}: run1 and run2 manifests differ")
        manifest = json.loads(tb)
        present = absent = 0
        for name, expected in manifest.items():
            for run_dir in (a.parent, b.parent):
                target = run_dir / name
                if not target.is_file():
                    absent += 1
                    continue
                present += 1
                actual = sha256_file(target)
                if actual != expected:
                    failures.append(f"{run_dir.name}/{name}: sha256 {actual[:12]} != manifest {expected[:12]}")
        print(f"{suite:<9} manifests identical={identical} entries={len(manifest)} released files re-hashed={present} "
              f"not released (per-row, in workspace)={absent}")
    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures:
            print(" -", f)
        return 1
    print("RUN MANIFESTS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
