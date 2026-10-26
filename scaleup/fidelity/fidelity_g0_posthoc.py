"""Post-hoc G0 diagnostic for FIDELITY_PROTOCOL_v1 (register entry F-D7).

Matches the upstream 932-row and 868-row training tables to the OSF tables row by row on the eight
features plus the measured diameter and the dispersed flow rate (six significant digits), then counts
per column how many matched rows differ.  Not part of the frozen protocol; written after the two
runs to read G0 honestly (the pre-registered first-match rule pairs duplicate input tuples with a
different replicate).

    python -I scaleup/fidelity/fidelity_g0_posthoc.py --upstream <clone> --data <OSF dir> --out <json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

COLUMN_MAP = {
    "orifice_width": "Orifice width (um)", "aspect_ratio": "Normalized channel depth", "flow_rate_ratio": "Flow rate ratio",
    "capillary_number": "Capillary number", "normalized_oil_inlet": "Normalized continuous inlet",
    "normalized_water_inlet": "Normalized dispersed inlet", "expansion_ratio": "Normalized outlet width",
    "viscosity_ratio": "viscosity ratio", "hyd_d": "Hydraulic diameter", "droplet_size": "Observed droplet diameter (um)",
    "normalized_hyd_size": "Normalized droplet diameter", "q_in": "Dispersed flow rate ul/h",
    "generation_rate": "Observed generation rate (Hz)",
}
KEY = ["orifice_width", "aspect_ratio", "flow_rate_ratio", "capillary_number", "normalized_oil_inlet", "normalized_water_inlet",
       "expansion_ratio", "viscosity_ratio", "droplet_size", "q_in"]
COMPARE = ["hyd_d", "normalized_hyd_size", "generation_rate"]
UP932 = "app/mod_dafd/helper_scripts/experimental_data/20231001_all_data_dafd3.csv"
UP868 = "app/mod_dafd/helper_scripts/experimental_data/DAFD3_data.csv"


def keys(frame: pd.DataFrame, cols: list[str]) -> list[tuple]:
    return [tuple(float(f"{v:.6g}") for v in row) for row in frame[cols].to_numpy(dtype=float)]


def compare(up: pd.DataFrame, ref: pd.DataFrame, labels: np.ndarray) -> dict:
    ku, kr = keys(up, KEY), keys(ref, [COLUMN_MAP[c] for c in KEY])
    index: dict[tuple, list[int]] = {}
    for i, k in enumerate(kr):
        index.setdefault(k, []).append(i)
    used: set[int] = set()
    pairs = []
    unmatched = 0
    for i, k in enumerate(ku):
        cands = [j for j in index.get(k, []) if j not in used]
        if not cands:
            unmatched += 1
            continue
        used.add(cands[0])
        pairs.append((i, cands[0]))
    out = {"n_upstream": int(len(up)), "n_reference": int(len(ref)), "matched": int(len(pairs)), "unmatched": int(unmatched), "columns": {}}
    lab = labels[[p[1] for p in pairs]]
    for c in COMPARE:
        a = up[c].to_numpy(float)[[p[0] for p in pairs]]
        b = ref[COLUMN_MAP[c]].to_numpy(float)[[p[1] for p in pairs]]
        out["columns"][c] = {}
        for label in sorted(set(lab.tolist())):
            m = lab == label
            both = np.isfinite(a[m]) & np.isfinite(b[m])
            rel = np.abs(a[m][both] - b[m][both]) / np.maximum(np.abs(b[m][both]), 1e-12)
            out["columns"][c][label] = {
                "n_rows": int(m.sum()), "n_both_finite": int(both.sum()),
                "n_rows_rel_diff_above_1e-6": int((rel > 1e-6).sum()), "n_rows_rel_diff_above_1e-3": int((rel > 1e-3).sum()),
                "max_rel_diff": float(rel.max()) if both.any() else None,
                "n_reference_missing": int((~np.isfinite(b[m])).sum()), "n_upstream_missing": int((~np.isfinite(a[m])).sum()),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    comp = pd.read_excel(args.data / "Comprehensive_normalized.xlsx")
    ext = pd.read_excel(args.data / "Generalizability_data_normalized.xlsx")
    up932 = pd.read_csv(args.upstream / UP932)
    up868 = pd.read_csv(args.upstream / UP868)
    union = pd.concat([comp, ext], ignore_index=True)
    labels = np.array(["comprehensive"] * len(comp) + ["external"] * len(ext))
    result = {
        "schema_version": "fidelity-g0-posthoc-v1",
        "inputs_sha256": {UP932: hashlib.sha256((args.upstream / UP932).read_bytes()).hexdigest(),
                          UP868: hashlib.sha256((args.upstream / UP868).read_bytes()).hexdigest()},
        "matching_key": KEY,
        "DAFD3_data_868_vs_OSF_comprehensive": compare(up868, comp, np.array(["comprehensive"] * len(comp))),
        "all_data_932_vs_OSF_comprehensive_union_external": compare(up932, union, labels),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result["all_data_932_vs_OSF_comprehensive_union_external"]["columns"].items()}, indent=1))


if __name__ == "__main__":
    main()
