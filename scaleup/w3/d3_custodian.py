"""D3 custodian — implements D3_BLIND_PROTOCOL.md §4 with the bound parameters of
D3_BINDING.json. It is the only program that reads laboratory-B targets before
reveal, and it prints nothing but counts and digests.

Usage:
  python -m w3.d3_custodian --download-dir <zenodo_17867478 dir> --binding <D3_BINDING.json> \
      --protocol <D3_BLIND_PROTOCOL.md> --output-dir <d3_blind_artifacts dir>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import uuid
from pathlib import Path

REQUIRED_INPUTS = ("ESM", "HSPC", "CHOL", "PEG", "TFR", "FRR", "AQUEOUS")
TARGETS = ("SIZE", "PDI", "OUTPUT")
INPUT_COLUMNS_OUT = ("ESM", "HSPC", "CHOL", "PEG", "TFR", "FRR", "AQUEOUS")
NAMESPACE = uuid.UUID("3f1a2b9c-4d5e-4f60-8a71-d3b1c0ffee00")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _read_rows(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def _opaque_id(file_sha: str, row_index: int) -> str:
    return str(uuid.uuid5(NAMESPACE, f"{file_sha}:{row_index}"))


def _partition(rows: list[dict], file_sha: str, *, keep_missing_targets: bool):
    """Return (kept rows with opaque ids, quarantine records). Never looks at target values
    beyond a finiteness test used only when keep_missing_targets is False."""

    kept: list[tuple[str, dict]] = []
    quarantine: list[dict] = []
    for index, row in enumerate(rows):
        chip = (row.get("CHIP") or "").strip()
        if chip != "Micromixer":
            quarantine.append({"row_index": index, "reason": f"CHIP={chip or 'missing'}"})
            continue
        missing_inputs = [c for c in REQUIRED_INPUTS if c == "AQUEOUS" and not (row.get(c) or "").strip()] + [
            c for c in REQUIRED_INPUTS if c != "AQUEOUS" and not _finite(row.get(c, ""))
        ]
        if missing_inputs:
            quarantine.append({"row_index": index, "reason": f"missing_inputs:{','.join(missing_inputs)}"})
            continue
        if not keep_missing_targets and not _finite(row.get("SIZE", "")):
            quarantine.append({"row_index": index, "reason": "development_row_missing_SIZE"})
            continue
        kept.append((_opaque_id(file_sha, index), row))
    return kept, quarantine


def _write_csv(path: Path, header: list[str], rows: list[list[str]], mode: int | None = None) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    if mode is not None:
        os.chmod(path, mode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    binding = json.loads(args.binding.read_text(encoding="utf-8"))
    protocol_sha = sha256_file(args.protocol)
    if protocol_sha != binding["protocol_sha256"]:
        raise ValueError("protocol digest does not match the binding file")
    binding_sha = sha256_file(args.binding)

    raw_dir = args.download_dir
    files = binding["data_files"]
    manifest: dict = {
        "schema_version": "d3-custodian-v1",
        "protocol": {"file": args.protocol.name, "sha256": protocol_sha},
        "binding": {"file": args.binding.name, "sha256": binding_sha},
        "raw_sources": {},
        "artifacts": {},
        "counts": {},
        "quarantine": {},
        "cluster_report": {},
        "stratum_report": {},
        "policy": "development rows carry targets; sealed rows carry inputs only; sealed targets live in 0600 files outside the repository; this program prints counts and digests only",
    }
    for name in ("microfluidics_dataset.zip", "README.pdf"):
        manifest["raw_sources"][name] = sha256_file(raw_dir / name)
    for key in ("development", "sealed_primary", "sealed_secondary"):
        rel = files[key]["path"]
        path = raw_dir / rel
        manifest["raw_sources"][rel] = sha256_file(path)
        xlsx = path.with_suffix(".xlsx")
        if xlsx.exists():
            manifest["raw_sources"][str(Path(rel).with_suffix(".xlsx"))] = sha256_file(xlsx)

    def emit(key: str, *, sealed: bool) -> None:
        rel = files[key]["path"]
        path = raw_dir / rel
        header, rows = _read_rows(path)
        if header != files[key]["header"]:
            raise ValueError(f"{rel}: header {header} differs from the bound header")
        file_sha = manifest["raw_sources"][rel]
        kept, quarantine = _partition(rows, file_sha, keep_missing_targets=sealed)
        ids = [row.get("ID", "") for _, row in kept]
        manifest["cluster_report"][key] = {"n_rows": len(kept), "n_unique_ids": len(set(ids))}
        strata: dict[str, int] = {}
        for _, row in kept:
            level = row["AQUEOUS"].strip()
            strata[level] = strata.get(level, 0) + 1
        manifest["stratum_report"][key] = strata
        manifest["quarantine"][key] = quarantine
        if sealed:
            features = [[oid, row["ID"]] + [row[c].strip() for c in INPUT_COLUMNS_OUT] for oid, row in kept]
            labels = [[oid] + [row.get(t, "").strip() for t in TARGETS] for oid, row in kept]
            fpath = output / f"{key}_features.csv"
            lpath = output / f"{key}_sealed_labels.csv"
            _write_csv(fpath, ["opaque_row_id", "ID", *INPUT_COLUMNS_OUT], features)
            _write_csv(lpath, ["opaque_row_id", *TARGETS], labels, mode=0o600)
            manifest["artifacts"][f"{key}_features"] = {"file": fpath.name, "rows": len(features), "sha256": sha256_file(fpath)}
            manifest["artifacts"][f"{key}_sealed_labels"] = {"file": lpath.name, "rows": len(labels), "sha256": sha256_file(lpath), "permissions": "0600", "kept_outside_repository": True}
        else:
            table = [[oid, row["ID"]] + [row[c].strip() for c in INPUT_COLUMNS_OUT] + [row.get(t, "").strip() for t in TARGETS] for oid, row in kept]
            dpath = output / "development_labeled.csv"
            _write_csv(dpath, ["opaque_row_id", "ID", *INPUT_COLUMNS_OUT, *TARGETS], table)
            manifest["artifacts"]["development_labeled"] = {"file": dpath.name, "rows": len(table), "sha256": sha256_file(dpath)}
        manifest["counts"][key] = {"raw_rows": len(rows), "kept": len(kept), "quarantined": len(quarantine)}

    emit("development", sealed=False)
    emit("sealed_primary", sealed=True)
    emit("sealed_secondary", sealed=True)

    manifest_path = output / "custodian_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("counts", "cluster_report", "stratum_report")}, indent=2))
    print("manifest sha256:", sha256_file(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
