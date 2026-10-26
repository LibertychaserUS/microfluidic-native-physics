#!/usr/bin/env python3
"""H_PROTOCOL_v1 §2-3: custodian for an admitted dataset.

Reads the frozen binding, builds the unified analysis table from the downloaded ESI
workbooks, applies the sealing rules of the binding, and writes

  development_labeled.csv      development rows with inputs and targets (open)
  sealed_features.csv          sealed rows, inputs only
  sealed_labels.csv            sealed rows, targets only (mode 0600, never committed)
  custodian_manifest.json      digests, counts per level, sealing decisions
  h_<code>_devdiag.json        development-set target-scale diagnosis and tolerance

The sealing decision uses row counts only; no target value of a sealed row is read
by any program before the reveal gate. The custodian is the only program that
touches the sealed labels, and it does so only to write them out.

Usage: python h_custodian.py --binding <H_BINDING_A1.json> --protocol <H_PROTOCOL_v1.md> --zip <path> --output <artifact dir>
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)


def parse_sheet(ws, spec: dict, columns: dict) -> tuple[pd.DataFrame, dict]:
    rows = [list(r) for r in ws.iter_rows(values_only=True)]
    header_i = None
    for i, r in enumerate(rows):
        if any(isinstance(c, str) and c.strip() == spec["header_marker"] for c in r):
            header_i = i
            break
    if header_i is None:
        raise ValueError(f"{ws.title}: header marker not found")
    header = {str(c).strip(): j for j, c in enumerate(rows[header_i]) if c is not None and str(c).strip()}
    want = {}
    for name in columns["native"]:
        want[name] = header[name]
    for name in ["Ca_d", "Ca_c", "q=Qd/Qc", "Re_d", "Re_c", "El_d"]:
        want[name] = header[name]
    want["Alpha"] = header[spec["alpha_col"]]
    want["Wi"] = header[spec["wi_col"]]
    want["L_um"] = header[spec["target_col"]]
    want["H_um"] = header[spec["secondary_target_col"]]
    body = rows[header_i + 1:]
    qd, qc = header["Qd [ul/min]"], header["Qc [ul/min]"]
    # per-sheet constants: input columns with exactly one numeric cell
    const = {}
    for name, j in want.items():
        if name in ("L_um", "H_um"):
            continue
        vals = [r[j] for r in body if j < len(r) and is_num(r[j])]
        if len(vals) == 1:
            const[name] = float(vals[0])
    records = []
    n_exp = n_missing_target = n_missing_input = n_nonpositive_target = 0
    for r in body:
        if not (qd < len(r) and qc < len(r) and is_num(r[qd]) and is_num(r[qc])):
            continue
        n_exp += 1
        rec = {}
        ok = True
        for name, j in want.items():
            v = r[j] if j < len(r) else None
            if is_num(v):
                rec[name] = float(v)
            elif name in const:
                rec[name] = const[name]
            else:
                rec[name] = np.nan
        if not np.isfinite(rec["L_um"]):
            n_missing_target += 1
            continue
        if rec["L_um"] <= 0:  # binding v2: no measurable droplet, not a size measurement
            n_nonpositive_target += 1
            continue
        if np.isfinite(rec["H_um"]) and rec["H_um"] <= 0:
            rec["H_um"] = np.nan
        inputs = [rec[n] for n in want if n not in ("L_um", "H_um")]
        if not all(np.isfinite(inputs)):
            n_missing_input += 1
            continue
        records.append(rec)
    frame = pd.DataFrame(records)
    info = {"sheet": ws.title, "header_row_index": header_i, "experiment_rows": n_exp,
            "dropped_missing_target": n_missing_target, "dropped_nonpositive_target": n_nonpositive_target, "dropped_missing_input": n_missing_input,
            "kept": int(len(frame)), "per_sheet_constants": sorted(const)}
    return frame, info


def choose_level(counts: dict[str, int], exclude_rows: set | None, rows_by_level: dict[str, set]) -> tuple[str | None, dict]:
    eligible = {k: v for k, v in counts.items() if v >= 10}
    if not eligible:
        return None, {"reason": "no level with >= 10 rows"}
    vmax = max(eligible.values())
    cand = {k: v for k, v in eligible.items() if v < vmax}
    if not cand:
        return None, {"reason": "every eligible level attains the maximum count"}
    med = float(np.median(list(cand.values())))
    order = sorted(cand, key=lambda k: (abs(cand[k] - med), k))
    trace = {"eligible_counts": eligible, "maximum_count_excluded": vmax, "median_of_candidates": med, "candidate_order": order}
    for k in order:
        if exclude_rows is None or not (rows_by_level[k] & exclude_rows):
            return k, {**trace, "chosen": k}
    return None, {**trace, "reason": "every candidate overlaps the already sealed rows"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binding", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    binding = json.loads(args.binding.read_text())
    if sha256_file(args.protocol) != binding["protocol_sha256"]:
        raise ValueError("protocol digest does not match the binding")
    zip_sha = sha256_file(args.zip)
    out = args.output
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing non-empty output directory {out}")
    out.mkdir(parents=True, exist_ok=True)

    z = zipfile.ZipFile(args.zip)
    frames, sheet_infos = [], []
    for wb_spec in binding["workbooks"]:
        wb = openpyxl.load_workbook(io.BytesIO(z.read(wb_spec["member"])), read_only=True, data_only=True)
        for ws in wb.worksheets:
            frame, info = parse_sheet(ws, wb_spec, binding["columns"])
            info["member"] = wb_spec["member"]
            info["family"] = wb_spec["family"]
            frame["workbook"] = wb_spec["member"]
            frame["family"] = wb_spec["family"]
            frame["sheet"] = ws.title
            frame["fluid_family"] = wb_spec["family"] + ":" + ws.title
            frames.append(frame)
            sheet_infos.append(info)
    table = pd.concat(frames, ignore_index=True)
    table["device"] = table["Etching Depth [um]"].round(0).astype(int).astype(str) + "x" + table["Width [um]"].round(0).astype(int).astype(str)
    table.insert(0, "row_id", [f"A1:{hashlib.sha256((zip_sha + ':' + str(i)).encode()).hexdigest()[:12]}" for i in range(len(table))])

    # sealing decisions (counts only)
    s2_counts = table.groupby("fluid_family").size().to_dict()
    rows_by_s2 = {k: set(v) for k, v in table.groupby("fluid_family").groups.items()}
    s2_level, s2_trace = choose_level(s2_counts, None, rows_by_s2)
    sealed_idx = set(rows_by_s2[s2_level]) if s2_level else set()
    s3_counts = table.groupby("device").size().to_dict()
    rows_by_s3 = {k: set(v) for k, v in table.groupby("device").groups.items()}
    s3_level, s3_trace = choose_level(s3_counts, sealed_idx, rows_by_s3)
    if s3_level:
        sealed_idx |= set(rows_by_s3[s3_level])
    table["sealed_tier"] = ""
    if s2_level:
        table.loc[table["fluid_family"] == s2_level, "sealed_tier"] = "S2"
    if s3_level:
        table.loc[table["device"] == s3_level, "sealed_tier"] = table.loc[table["device"] == s3_level, "sealed_tier"].replace("", "S3")
    sealed = table.loc[table["sealed_tier"] != ""].copy()
    dev = table.loc[table["sealed_tier"] == ""].copy()

    input_cols = binding["columns"]["native"] + ["Ca_d", "Ca_c", "q=Qd/Qc", "Re_d", "Re_c", "Alpha", "Wi", "El_d"]
    meta_cols = ["row_id", "workbook", "family", "sheet", "fluid_family", "device", "sealed_tier"]
    dev_path = out / "development_labeled.csv"
    dev[meta_cols + input_cols + ["L_um", "H_um"]].to_csv(dev_path, index=False, float_format="%.12g", lineterminator="\n")
    feat_path = out / "sealed_features.csv"
    sealed[meta_cols + input_cols].to_csv(feat_path, index=False, float_format="%.12g", lineterminator="\n")
    lab_path = out / "sealed_labels.csv"
    sealed[["row_id", "L_um", "H_um"]].to_csv(lab_path, index=False, float_format="%.12g", lineterminator="\n")
    os.chmod(lab_path, 0o600)

    # development-set diagnosis (development labels only)
    y = dev["L_um"].to_numpy(float)
    skew = float(pd.Series(y).skew())
    ratio = float(y.max() / y.min())
    log_target = bool(ratio > 10 or skew > 2)
    devdiag = {
        "schema_version": "h-devdiag-v1", "dataset_code": binding["dataset_code"],
        "n_development_rows": int(len(dev)), "target": "L_um",
        "max_over_min": ratio, "skewness": skew, "rule": binding["target_scale_rule"],
        "target_scale": "log" if log_target else "raw",
        "development_median_L_um": float(np.median(y)),
        "tau_relative": binding["tolerance"]["relative"],
        "tau_absolute_um": float(0.10 * np.median(y)),
        "development_fluid_family_counts": dev.groupby("fluid_family").size().to_dict(),
        "development_device_counts": dev.groupby("device").size().to_dict(),
    }
    (out / f"h_{binding['dataset_code']}_devdiag.json").write_text(json.dumps(devdiag, indent=2) + "\n")

    manifest = {
        "schema_version": "h-custodian-v2",
        "dataset_code": binding["dataset_code"],
        "protocol": {"file": args.protocol.name, "sha256": binding["protocol_sha256"]},
        "binding": {"file": args.binding.name, "sha256": sha256_file(args.binding)},
        "source_zip": {"file": args.zip.name, "sha256": zip_sha},
        "sheets": sheet_infos,
        "n_rows_total": int(len(table)),
        "fluid_family_counts": s2_counts,
        "device_counts": s3_counts,
        "sealing": {"S2": {"level": s2_level, **s2_trace}, "S3": {"level": s3_level, **s3_trace}},
        "n_sealed_rows": int(len(sealed)), "n_development_rows": int(len(dev)),
        "artifacts": {
            "development_labeled": {"file": dev_path.name, "rows": int(len(dev)), "sha256": sha256_file(dev_path)},
            "sealed_features": {"file": feat_path.name, "rows": int(len(sealed)), "sha256": sha256_file(feat_path)},
            "sealed_labels": {"file": lab_path.name, "rows": int(len(sealed)), "sha256": sha256_file(lab_path), "permissions": "0600", "kept_outside_repository": True},
            "devdiag": {"file": f"h_{binding['dataset_code']}_devdiag.json", "sha256": sha256_file(out / f"h_{binding['dataset_code']}_devdiag.json")},
        },
    }
    (out / "custodian_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("n_rows_total", "fluid_family_counts", "device_counts", "sealing", "n_sealed_rows", "n_development_rows")}, indent=1))
    print("devdiag:", {k: devdiag[k] for k in ("max_over_min", "skewness", "target_scale", "development_median_L_um", "tau_absolute_um")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
