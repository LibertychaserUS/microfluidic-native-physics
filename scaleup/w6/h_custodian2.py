#!/usr/bin/env python3
"""H_PROTOCOL_v2 §2-3: binding-driven custodian for the round-2 datasets.

Builds the unified analysis table
    row_id, s1_group, s2_group, s3_group, s4_group, <meta>, <native inputs>, <dimensionless inputs>, target
from one source file, applies the sealing rules of H_PROTOCOL_v2 §3 (identical to v1:
S2/S3 median-count rule, S3 overlap rule, S4 minority-class rule) and writes

  development_labeled.csv      development rows with inputs and target (open)
  sealed_features.csv          sealed rows (S2/S3/S4), inputs only
  sealed_labels.csv            sealed rows, target only (mode 0600, never committed)
  custodian_manifest.json      digests, counts per level, sealing decisions, parsing counts
  h_<code>_devdiag.json        development-set target-scale diagnosis and tolerance

Loaders (chosen by binding["loader"]):
  flat_xlsx      one worksheet with a header row; numeric input and target columns (R2-A1)
  lnp_atlas_csv  LNP Atlas CSV with the three fixed parsing/encoding rules of the protocol (R2-A2)

The sealing decision uses row counts only; no target value of a sealed row is read by
any program before the reveal gate.

Usage: python h_custodian2.py --binding <json> --protocol <H_PROTOCOL_v2.md> --source <file> --output <artifact dir>
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

META_COLS = ["row_id", "s1_group", "s2_group", "s3_group", "s4_group", "sealed_tier", "context"]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def md5_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def fmt(v: float) -> str:
    return format(float(v), ".6g")


# ----------------------------------------------------------------------------- loaders
def load_flat_xlsx(path: Path, spec: dict, columns: dict) -> tuple[pd.DataFrame, dict]:
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True)[spec["sheet"]]
    rows = list(ws.iter_rows(values_only=True))
    header = [str(c).strip() if c is not None else "" for c in rows[spec["header_row_index"]]]
    idx = {h: j for j, h in enumerate(header) if h}
    need = columns["native"] + [spec["target_col"]] + spec.get("meta_cols", []) + spec.get("key_cols", [])
    for name in need:
        if name not in idx:
            raise ValueError(f"column {name!r} not in header {header}")
    records, n_blank, n_missing_input, n_missing_target, n_nonpositive = [], 0, 0, 0, 0
    for i, r in enumerate(rows[spec["header_row_index"] + 1:]):
        if not any(c is not None for c in r):
            n_blank += 1
            continue
        rec = {"source_row": i}
        ok = True
        for name in columns["native"] + spec.get("key_cols", []):
            v = r[idx[name]] if idx[name] < len(r) else None
            if not is_num(v):
                ok = False
                break
            rec[name] = float(v)
        if not ok:
            n_missing_input += 1
            continue
        t = r[idx[spec["target_col"]]] if idx[spec["target_col"]] < len(r) else None
        if not is_num(t):
            n_missing_target += 1
            continue
        if float(t) <= 0:
            n_nonpositive += 1
            continue
        rec["target"] = float(t)
        for name in spec.get("meta_cols", []):
            v = r[idx[name]] if idx[name] < len(r) else None
            rec["meta_" + name] = "" if v is None else str(v)
        records.append(rec)
    frame = pd.DataFrame(records)
    # groups
    frame["s2_group"] = ["fluid:" + "|".join(fmt(row[c]) for c in spec["s2_key_cols"]) for _, row in frame.iterrows()]
    frame["s3_group"] = ["device:" + "|".join(fmt(row[c]) for c in spec["s3_key_cols"]) for _, row in frame.iterrows()]
    frame["s1_group"] = frame["s2_group"] + "@" + frame["s3_group"]
    frame["s4_group"] = ""
    frame["context"] = frame["s3_group"]
    # dimensionless columns (constant unit factors only; see binding)
    d = frame
    if columns.get("dimensionless"):
        area_in = math.pi / 4.0 * d["inner"] ** 2
        area_out = math.pi / 4.0 * (d["channel"] ** 2 - d["outer"] ** 2)
        u_d = d["disp_flow"] / area_in
        u_c = d["cont_flow"] / area_out
        d["Ca_c"] = d["cont_viscosity"] * u_c / d["interfacial_tension"]
        d["Ca_d"] = d["disp_viscosity"] * u_d / d["interfacial_tension"]
        d["q_flow"] = d["disp_flow"] / d["cont_flow"]
        d["lambda_mu"] = d["disp_viscosity"] / d["cont_viscosity"]
        d["Re_c"] = d["cont_density"] * u_c * (d["channel"] - d["outer"]) / d["cont_viscosity"]
        d["Re_d"] = d["disp_density"] * u_d * d["inner"] / d["disp_viscosity"]
        d["geo_inner_over_channel"] = d["inner"] / d["channel"]
        missing = [c for c in columns["dimensionless"] if c not in d.columns]
        if missing:
            raise ValueError(f"dimensionless columns not produced: {missing}")
    info = {"n_source_rows": len(rows) - spec["header_row_index"] - 1, "n_blank": n_blank, "dropped_missing_input": n_missing_input,
            "dropped_missing_target": n_missing_target, "dropped_nonpositive_target": n_nonpositive, "kept": int(len(frame))}
    return frame, info


_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def parse_size_cell(raw: str) -> tuple[float | None, str]:
    s = (raw or "").strip().replace(" ", " ").replace("\xa0", " ")
    if not s:
        return None, "empty"
    if re.fullmatch(_NUM, s):
        return float(s), "plain"
    m = re.fullmatch(rf"({_NUM})\s*(?:±|\+/-|\+-)\s*{_NUM}\s*(?:nm)?", s)
    if m:
        return float(m.group(1)), "plus_minus"
    if re.fullmatch(rf"{_NUM}\s*[-–~]\s*{_NUM}\s*(?:nm)?", s):
        return None, "range"
    return None, "other_text"


def parse_ratio(raw: str) -> list[float] | None:
    parts = [p.strip() for p in (raw or "").split(":")]
    nums = []
    for p in parts:
        m = re.match(_NUM, p)
        if not m:
            break
        nums.append(float(m.group(0)))
    if len(nums) < 4:
        return None
    nums = nums[:4]
    total = sum(nums)
    if total <= 0:
        return None
    return [v / total for v in nums]


METHOD_FLAGS = {"flag_herringbone": ("herringbone",), "flag_tjunction": ("t-junction",), "flag_nanoassemblr": ("nanoassemblr", "ignite"),
                "flag_pipette": ("pipett", "hand"), "flag_ethanol_injection": ("ethanol injection",)}
MF_KEYWORDS = ("microfluidic", "nanoassemblr", "herringbone", "t-junction", "ignite")
LIPID_COLS = ("ionizable_lipid", "helper_lipid", "peg_lipid", "sterol_lipid")


def load_lnp_atlas(path: Path, spec: dict, columns: dict) -> tuple[pd.DataFrame, dict]:
    # H-D10: the deposit is not valid UTF-8 (byte pairs A1 BE / A1 C6 = '±' / '°' in EUC-KR); decode UTF-8 strictly first, else cp949
    data = path.read_bytes()
    try:
        text, encoding = data.decode("utf8"), "utf8"
    except UnicodeDecodeError:
        text, encoding = data.decode("cp949"), "cp949"
    raw = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    counts = {"n_source_rows": int(len(raw)), "encoding_used": encoding, "size": {}, "ratio_dropped": 0, "kept_microfluidic": 0, "kept_non_microfluidic": 0}
    records = []
    for i, row in raw.iterrows():
        size, kind = parse_size_cell(row["particle_size_nm_std"])
        counts["size"][kind] = counts["size"].get(kind, 0) + 1
        if size is None or size <= 0:
            if size is not None:
                counts["size"]["nonpositive"] = counts["size"].get("nonpositive", 0) + 1
            continue
        fr = parse_ratio(row["lipid_molar_ratio"])
        if fr is None:
            counts["ratio_dropped"] += 1
            continue
        syn = (row["synthesis_info"] or "").lower()
        mf = any(k in syn for k in MF_KEYWORDS)
        doi = (row["paper_doi"] or "").strip().lower()
        doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
        rec = {"source_row": int(i), "target": size, "frac_ionizable": fr[0], "frac_helper": fr[1], "frac_sterol": fr[2], "frac_peg": fr[3],
               "s2_group": "ion:" + ((row["ionizable_lipid"] or "").strip() or "(blank)"),
               "s3_group": "paper:" + (doi or "(no doi)"), "s4_group": "microfluidic" if mf else "non_microfluidic",
               "meta_target_type": (row["target_type"] or "").strip() or "(blank)", "meta_paper_year": (row["paper_year"] or "").strip()}
        rec["s1_group"] = rec["s3_group"]
        rec["context"] = rec["s4_group"]
        for c in LIPID_COLS:
            rec["cat_" + c] = (row[c] or "").strip() or "(blank)"
        for flag, keys in METHOD_FLAGS.items():
            rec[flag] = 1.0 if any(k in syn for k in keys) else 0.0
        records.append(rec)
        counts["kept_microfluidic" if mf else "kept_non_microfluidic"] += 1
    frame = pd.DataFrame(records)
    # one-hot encodings over all parsed rows (inputs only; label-free)
    onehot_cols = []
    for c in LIPID_COLS:
        for lv in sorted(frame["cat_" + c].unique()):
            name = f"{c}={lv}"
            frame[name] = (frame["cat_" + c] == lv).astype(float)
            onehot_cols.append(name)
    for lv in sorted(frame["meta_target_type"].unique()):
        name = f"target_type={lv}"
        frame[name] = (frame["meta_target_type"] == lv).astype(float)
        onehot_cols.append(name)
    native = ["frac_ionizable", "frac_helper", "frac_sterol", "frac_peg"] + onehot_cols + list(METHOD_FLAGS)
    columns["native"] = native  # the binding declares the construction rule; the concrete list is written to the manifest
    counts["n_native_columns"] = len(native)
    return frame, counts


# ----------------------------------------------------------------------------- sealing
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
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    binding = json.loads(args.binding.read_text())
    if sha256_file(args.protocol) != binding["protocol_sha256"]:
        raise ValueError("protocol digest does not match the binding")
    if md5_file(args.source) != binding["source_md5_expected"]:
        raise ValueError("source file md5 does not match the binding")
    src_sha = sha256_file(args.source)
    out = args.output
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing non-empty output directory {out}")
    out.mkdir(parents=True, exist_ok=True)
    code = binding["dataset_code"]
    columns = dict(binding["columns"])
    if binding["loader"] == "flat_xlsx":
        table, info = load_flat_xlsx(args.source, binding["loader_spec"], columns)
    elif binding["loader"] == "lnp_atlas_csv":
        table, info = load_lnp_atlas(args.source, binding["loader_spec"], columns)
    else:
        raise ValueError(binding["loader"])
    table = table.reset_index(drop=True)
    table.insert(0, "row_id", [f"{code}:{hashlib.sha256((src_sha + ':' + str(r)).encode()).hexdigest()[:12]}" for r in table["source_row"]])
    has_s1 = binding["groups"]["S1"] is not None
    if not has_s1:
        table["s1_group"] = ""
    # S4: two classes -> minority sealed entirely; main table = majority class
    s4_defined = binding["groups"]["S4"] is not None
    table["sealed_tier"] = ""
    if s4_defined:
        cls = table["s4_group"].value_counts()
        if len(cls) != 2:
            raise ValueError(f"S4 expects exactly two classes, found {cls.to_dict()}")
        minority = cls.idxmin()
        table.loc[table["s4_group"] == minority, "sealed_tier"] = "S4"
        main = table[table["s4_group"] != minority]
    else:
        table["s4_group"] = ""
        main = table
    main_idx = main.index
    s2_counts = main.groupby("s2_group").size().to_dict()
    rows_by_s2 = {k: set(v) for k, v in main.groupby("s2_group").groups.items()}
    s2_level, s2_trace = choose_level(s2_counts, None, rows_by_s2)
    sealed_idx = set(rows_by_s2[s2_level]) if s2_level else set()
    s3_counts = main.groupby("s3_group").size().to_dict()
    rows_by_s3 = {k: set(v) for k, v in main.groupby("s3_group").groups.items()}
    s3_level, s3_trace = choose_level(s3_counts, sealed_idx, rows_by_s3)
    if s2_level:
        table.loc[list(rows_by_s2[s2_level]), "sealed_tier"] = "S2"
    if s3_level:
        table.loc[list(rows_by_s3[s3_level]), "sealed_tier"] = "S3"
    sealed = table.loc[table["sealed_tier"] != ""].copy()
    dev = table.loc[(table["sealed_tier"] == "") & table.index.isin(main_idx)].copy()
    input_cols = columns["native"] + list(columns.get("dimensionless") or [])
    meta_extra = [c for c in table.columns if c.startswith("meta_")]
    meta_cols = META_COLS + meta_extra
    dev_path = out / "development_labeled.csv"
    dev[meta_cols + input_cols + ["target"]].to_csv(dev_path, index=False, float_format="%.12g", lineterminator="\n")
    feat_path = out / "sealed_features.csv"
    sealed[meta_cols + input_cols].to_csv(feat_path, index=False, float_format="%.12g", lineterminator="\n")
    lab_path = out / "sealed_labels.csv"
    sealed[["row_id", "target"]].to_csv(lab_path, index=False, float_format="%.12g", lineterminator="\n")
    os.chmod(lab_path, 0o600)
    y = dev["target"].to_numpy(float)
    skew = float(pd.Series(y).skew())
    ratio = float(y.max() / y.min())
    log_target = bool(ratio > 10 or skew > 2)
    devdiag = {
        "schema_version": "h-devdiag-v2", "dataset_code": code, "n_development_rows": int(len(dev)), "target": "target",
        "target_unit": columns.get("target_unit"), "max_over_min": ratio, "skewness": skew, "rule": binding["target_scale_rule"],
        "target_scale": "log" if log_target else "raw", "development_median_target": float(np.median(y)),
        "tau_relative": binding["tolerance"]["relative"], "tau_absolute": float(0.10 * np.median(y)),
        "development_s2_counts": dev.groupby("s2_group").size().to_dict(), "development_s3_counts": dev.groupby("s3_group").size().to_dict(),
        "development_s1_levels": int(dev["s1_group"].nunique()) if has_s1 else None,
    }
    (out / f"h_{code}_devdiag.json").write_text(json.dumps(devdiag, indent=2) + "\n")
    manifest = {
        "schema_version": "h-custodian-v2", "dataset_code": code,
        "protocol": {"file": args.protocol.name, "sha256": binding["protocol_sha256"]},
        "binding": {"file": args.binding.name, "sha256": sha256_file(args.binding)},
        "source": {"file": args.source.name, "sha256": src_sha, "md5": md5_file(args.source)},
        "parsing": info, "native_columns": columns["native"], "dimensionless_columns": list(columns.get("dimensionless") or []),
        "n_rows_total": int(len(table)), "n_main_table_rows": int(len(main)),
        "s2_counts_main": s2_counts, "s3_counts_main": s3_counts,
        "s4_classes": table["s4_group"].value_counts().to_dict() if s4_defined else None,
        "sealing": {"S2": {"level": s2_level, **s2_trace}, "S3": {"level": s3_level, **s3_trace},
                    "S4": {"level": (table.loc[table["sealed_tier"] == "S4", "s4_group"].iloc[0] if (table["sealed_tier"] == "S4").any() else None),
                           "rule": "minority class sealed entirely"} if s4_defined else None},
        "n_sealed_rows": int(len(sealed)), "n_sealed_by_tier": sealed["sealed_tier"].value_counts().to_dict(), "n_development_rows": int(len(dev)),
        "artifacts": {
            "development_labeled": {"file": dev_path.name, "rows": int(len(dev)), "sha256": sha256_file(dev_path)},
            "sealed_features": {"file": feat_path.name, "rows": int(len(sealed)), "sha256": sha256_file(feat_path)},
            "sealed_labels": {"file": lab_path.name, "rows": int(len(sealed)), "sha256": sha256_file(lab_path), "permissions": "0600", "kept_outside_repository": True},
            "devdiag": {"file": f"h_{code}_devdiag.json", "sha256": sha256_file(out / f"h_{code}_devdiag.json")},
        },
    }
    (out / "custodian_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("parsing", "n_rows_total", "n_main_table_rows", "s2_counts_main", "s3_counts_main", "s4_classes", "sealing", "n_sealed_by_tier", "n_development_rows")}, indent=1, default=str)[:6000])
    print("devdiag:", {k: devdiag[k] for k in ("max_over_min", "skewness", "target_scale", "development_median_target", "tau_absolute")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
