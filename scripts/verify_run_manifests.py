#!/usr/bin/env python3
"""Check every run pair and every single-run directory from its artifact_hashes.json (standard library only).

For a pair (run1, run2) under work/microfluidic_reliability/ the two manifests must be
byte-identical, except for the E4 pair, whose manifests differ only in files that the
stored identity record (e4_identity/e4_identity.json, register entry E-D10) lists as
identical once the wall-clock timing column is removed. Every released file listed in a
manifest must re-hash to its entry; files listed but not released (per-row tables that
stay in the workspace) are counted and never silently passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work" / "microfluidic_reliability"

PAIRS = [
    ("v2_artifacts_canonical_run1", "v2_artifacts_canonical_run2", None),
    ("expanded_artifacts_canonical_run1", "expanded_artifacts_canonical_run2", None),
    ("w3_artifacts_canonical_run1", "w3_artifacts_canonical_run2", None),
    ("w4_artifacts_canonical_run1", "w4_artifacts_canonical_run2", None),
    ("w5_artifacts_canonical_run1", "w5_artifacts_canonical_run2", None),
    ("e0_run1", "e0_run2", None),
    ("e1e2_run1", "e1e2_run2", None),
    ("e1b_run1", "e1b_run2", None),
    ("e1b_run1/cell_tau0.05_B10", "e1b_run2/cell_tau0.05_B10", None),
    ("e1b_run1/cell_tau0.05_B20", "e1b_run2/cell_tau0.05_B20", None),
    ("e1b_run1/cell_tau0.10_B10", "e1b_run2/cell_tau0.10_B10", None),
    ("e3_run1", "e3_run2", None),
    ("e4_mac_run1", "e4_mac_run2", "e4_identity/e4_identity.json"),
    ("e5_run1", "e5_run2", None),
    ("e6_run1", "e6_run2", None),
    ("e7_run1", "e7_run2", None),
    ("fidelity_run1", "fidelity_run2", None),
    ("h_fewshot_rule_run1", "h_fewshot_rule_run2", None),
    ("h_A1_run1", "h_A1_run2", None),
    ("h_R2A1_run1", "h_R2A1_run2", None),
    ("h_R2A1_run3", "h_R2A1_run4", None),
    ("h_R2A2_run1", "h_R2A2_run2", None),
    ("h_portfolio_run1", "h_portfolio_run2", None),
    ("h_portfolio2_run1", "h_portfolio2_run2", None),
    ("posthoc_review_round3_run1", "posthoc_review_round3_run2", None),
    ("posthoc_review_round4_hg_run1", "posthoc_review_round4_hg_run2", None),
    ("posthoc_review_round4_hg2_run1", "posthoc_review_round4_hg2_run2", None),
]
SINGLES = [
    "posthoc_review_round2", "e1_reanalysis", "e1_dense_sensitivity", "e6_posthoc", "e_tie_check",
    "h_A1_reveal", "h_R2A1_reveal", "h_R2A1_reveal_b1", "h_R2A2_reveal", "posthoc_review_round4",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_differs_only_in_timing(pa: Path, pb: Path, timing_key: str) -> bool:
    """True when both files exist and every differing leaf sits under a key containing timing_key."""
    if not (pa.is_file() and pb.is_file()):
        return False
    a, b = json.loads(pa.read_text(encoding="utf-8")), json.loads(pb.read_text(encoding="utf-8"))
    bad: list[str] = []

    def walk(x, y, path: str) -> None:
        if isinstance(x, dict) and isinstance(y, dict) and set(x) == set(y):
            for k in x:
                walk(x[k], y[k], f"{path}.{k}")
        elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f"{path}[{i}]")
        elif x != y and timing_key not in path:
            bad.append(path)

    walk(a, b, "")
    return not bad


def rehash(work: Path, run_dir: Path, manifest: dict, failures: list[str]) -> tuple[int, int]:
    present = absent = 0
    for name, expected in manifest.items():
        target = run_dir / name
        if not target.is_file():
            absent += 1
            continue
        present += 1
        actual = sha256_file(target)
        if actual != expected:
            failures.append(f"{run_dir.relative_to(work)}/{name}: sha256 {actual[:12]} != manifest {expected[:12]}")
    return present, absent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=WORK)
    args = parser.parse_args()
    work = args.work.resolve()
    failures: list[str] = []
    for a, b, identity in PAIRS:
        pa, pb = work / a / "artifact_hashes.json", work / b / "artifact_hashes.json"
        ma, mb = json.loads(pa.read_bytes()), json.loads(pb.read_bytes())
        identical = pa.read_bytes() == pb.read_bytes()
        note = "identical"
        if not identical:
            if identity is None:
                failures.append(f"{a} / {b}: manifests differ")
                note = "DIFFER"
            else:
                record = json.loads((work / identity).read_text(encoding="utf-8"))
                files = record.get("files", {})
                for name in set(ma) | set(mb):
                    if ma.get(name) == mb.get(name):
                        continue
                    entry = files.get(name)
                    covered = bool(entry and entry.get("identical_without_fit_seconds") is True
                                   and entry.get("sha256_run1") == ma.get(name) and entry.get("sha256_run2") == mb.get(name))
                    if not covered and name.endswith(".json"):
                        # results.json is not listed in the identity record; accept it only if the two
                        # JSON documents differ in nothing but leaves under a key named after the timing column
                        covered = _json_differs_only_in_timing(work / a / name, work / b / name, record.get("timing_column", "fit_seconds"))
                    if not covered:
                        failures.append(f"{a} / {b}: {name} differs and is not covered by {identity}")
                note = f"differ only in the timing column per {identity}" if record.get("identical_except_timing") else "DIFFER (record says not identical)"
        pr1, ab1 = rehash(work, work / a, ma, failures)
        pr2, ab2 = rehash(work, work / b, mb, failures)
        print(f"{a:<36} {note:<48} entries={len(ma):>3} released re-hashed={pr1 + pr2:>3} workspace-only={ab1 + ab2:>3}")
    for d in SINGLES:
        m = json.loads((work / d / "artifact_hashes.json").read_text(encoding="utf-8"))
        pr, ab = rehash(work, work / d, m, failures)
        print(f"{d:<36} {'single run':<48} entries={len(m):>3} released re-hashed={pr:>3} workspace-only={ab:>3}")
    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures:
            print(" -", f)
        return 1
    print("RUN MANIFESTS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
