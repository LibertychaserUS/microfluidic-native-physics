#!/usr/bin/env python3
"""H_PROTOCOL_v2 section 5: multi-dataset decisions from the per-dataset reveal results.

Reads the frozen reveal result files (v1 schema for A1, v2 schema for the round-2 datasets)
and applies the written rules of section 5 without any hand editing:

  H-A  success = every admitted dataset's S2/S3 r in [2.2, 3.0], S2/S3 dispersion <= S1 dispersion
       (where S1 exists) and S4 lower bound above the S2/S3 upper bounds (where S4 exists);
       falsified = any dataset with an S2/S3 r outside the band with a disjoint 95% interval, or
       dispersion > S1; not determinable = no dataset defines S2 or S3.
  H-B  B1 success = bounded MLP better and no invalid prediction on every tier in >= 2 datasets;
       falsified = any dataset fails.  B2 not evaluable (no anchor on any admitted dataset).
  H-C  C1 success = median paired difference < 0 and Wilcoxon p < 0.05 (BH over the evaluable
       datasets) in >= 2 datasets; falsified = median >= 0 in the majority of evaluable datasets.
       C2 evaluable only where >= 20 scale-different pairs exist.
  H-D  success = |A - W| <= 0.10 on every S2/S3 tier of every dataset and Spearman(A, MAPE) across
       tiers not significant; falsified = any tier with A - W > 0.10.
  H-E  E1 success = rho <= 0.5 on both tiers in >= 2 datasets; falsified = rho >= 0.7 in the majority.
       E2 per UQ model: width ratio > 1 and Spearman(|error|, width) > 0.3 with BH-corrected p < 0.05
       on a sealed tier in >= 2 datasets -> success, else falsified.

Usage: python h_combine.py --result A1=<path> --result R2A1=<path> [--result R2A2=<path>] --output <json>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

BAND = (2.2, 3.0)
UQ = ("qrf", "gp", "deep_ensemble")


def bh(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out.tolist()


def rho_p(rho: float | None, n: int) -> float | None:
    if rho is None or n < 4 or abs(rho) >= 1:
        return None if rho is None else (0.0 if abs(rho) >= 1 else None)
    t = rho * math.sqrt((n - 2) / (1 - rho * rho))
    return float(2 * stats.t.sf(abs(t), n - 2))


def tier_entry(res: dict, tier: str, key: str = "xgboost|native") -> dict | None:
    t = res.get("tiers", {}).get(tier)
    return None if t is None else t["models"].get(key)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", action="append", required=True, help="CODE=path to h_reveal_results.json")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    results = {}
    for item in args.result:
        code, path = item.split("=", 1)
        results[code] = json.loads(Path(path).read_text())
    out = {"schema_version": "h-combine-v1", "datasets": list(results), "per_dataset": {}, "decisions": {}}

    # ---- H-A
    ha_rows = {}
    for code, res in results.items():
        tiers = [t for t in ("S2", "S3") if tier_entry(res, t)]
        if not tiers:
            continue
        per = {}
        for t in tiers:
            e = tier_entry(res, t)
            r, ci = e["ratio_to_dev_ref"], e["ratio_ci95"]
            per[t] = {"r": r, "ci": ci, "in_band": BAND[0] <= r <= BAND[1], "ci_disjoint": ci[1] < BAND[0] or ci[0] > BAND[1]}
        disp = (max(per[t]["r"] for t in tiers) / min(per[t]["r"] for t in tiers)) if len(tiers) == 2 else None
        s1 = res.get("H_A", {}).get("S1_dispersion_max_over_min")
        s4 = tier_entry(res, "S4")
        s4_ok = None if s4 is None else bool(s4["ratio_ci95"][0] > max(per[t]["ci"][1] for t in tiers))
        ha_rows[code] = {"per_tier": per, "S2_S3_dispersion": disp, "S1_dispersion": s1, "dispersion_within_S1": None if (disp is None or s1 is None) else bool(disp <= s1),
                         "S4_r": None if s4 is None else s4["ratio_to_dev_ref"], "S4_lower_above_S2S3_upper": s4_ok}
    if not ha_rows:
        ha_dec = "not_determinable: no dataset defines S2 or S3"
    else:
        falsify = [c for c, v in ha_rows.items() if any(p["ci_disjoint"] for p in v["per_tier"].values()) or v["dispersion_within_S1"] is False]
        success = all(all(p["in_band"] for p in v["per_tier"].values()) and v["dispersion_within_S1"] in (True, None) and v["S4_lower_above_S2S3_upper"] in (True, None) for v in ha_rows.values())
        ha_dec = f"falsified (datasets: {', '.join(falsify)})" if falsify else ("supported" if success else "point estimates outside the band without disjoint intervals: neither supported nor falsified")
    out["decisions"]["H_A"] = {"rule": "section 5 H-A", "decision": ha_dec, "per_dataset": ha_rows}

    # ---- H-B B1
    hb = {}
    for code, res in results.items():
        b1 = res.get("H_B", {}).get("B1", {})
        hb[code] = {"tiers": b1, "all_tiers_pass": bool(b1) and all(v["bounded_lower_error_and_no_invalid"] for v in b1.values())}
    n_pass = sum(v["all_tiers_pass"] for v in hb.values())
    fails = [c for c, v in hb.items() if not v["all_tiers_pass"]]
    out["decisions"]["H_B"] = {"B1_decision": f"falsified (datasets failing: {', '.join(fails)})" if fails else ("supported" if n_pass >= 2 else "not determinable: fewer than 2 datasets"),
                               "B1_per_dataset": hb, "B2_decision": "not evaluable: no anchor on any admitted dataset"}

    # ---- H-C C1 / C2
    hc = {}
    evaluable = {c: r for c, r in results.items() if r.get("H_C", {}).get("evaluable", "C1_dev_paired" in r.get("H_C", {}))}
    pv = {}
    for code, res in evaluable.items():
        d = res["H_C"]["C1_dev_paired"]
        hc[code] = {"median_diff_pp": d["median_diff_native_minus_dimensionless_pp"], "wilcoxon_p": d["wilcoxon_p"], "n_pairs": d["n_pairs"],
                    "C1_sealed": res["H_C"].get("C1_sealed"), "C2": res["H_C"].get("C2")}
        pv[code] = d["wilcoxon_p"] if d["wilcoxon_p"] is not None else 1.0
    adj = dict(zip(pv, bh(list(pv.values()))))
    for code in hc:
        hc[code]["wilcoxon_p_bh"] = adj[code]
        hc[code]["C1_pass"] = bool(hc[code]["median_diff_pp"] < 0 and adj[code] < 0.05)
    n_c1 = sum(v["C1_pass"] for v in hc.values())
    n_nonneg = sum(v["median_diff_pp"] >= 0 for v in hc.values())
    if not hc:
        c1_dec = "not evaluable on any admitted dataset"
    elif n_c1 >= 2:
        c1_dec = "supported"
    elif n_nonneg > len(hc) / 2:
        c1_dec = "falsified (median difference >= 0 in the majority of evaluable datasets)"
    else:
        c1_dec = "not determinable (fewer than 2 datasets pass and no majority with median >= 0)"
    c2 = {c: v["C2"] for c, v in hc.items()}
    out["decisions"]["H_C"] = {"C1_decision": c1_dec, "C1_per_dataset": hc, "n_evaluable": len(hc),
                               "C2_decision": "evaluable on: " + ", ".join(c for c, v in c2.items() if v and v.get("evaluable")) if any(v and v.get("evaluable") for v in c2.values()) else "not determinable on every dataset (fewer than 20 scale-different equal-group pairs)"}

    # ---- H-D
    hd_tiers = []
    for code, res in results.items():
        for t, v in res.get("H_D", {}).get("per_tier", {}).items():
            hd_tiers.append({"dataset": code, "tier": t, "A": v["A_abs"], "W": v["W_abs"], "A_minus_W": v["A_minus_W_abs"], "mape": v["mape"]})
    any_gap = [f"{r['dataset']}:{r['tier']}" for r in hd_tiers if r["A_minus_W"] > 0.10]
    all_close = all(abs(r["A_minus_W"]) <= 0.10 for r in hd_tiers)
    A_vals = [r["A"] for r in hd_tiers]; M_vals = [r["mape"] for r in hd_tiers]
    if len(hd_tiers) >= 4 and np.std(A_vals) > 0:
        sp = stats.spearmanr(A_vals, M_vals); sp_rho, sp_p = float(sp.correlation), float(sp.pvalue)
    else:
        sp_rho, sp_p = None, None
    hd_dec = f"falsified (A - W > 0.10 on {', '.join(any_gap)})" if any_gap else ("supported" if all_close and (sp_p is None or sp_p >= 0.05) else ("supported except the Spearman condition" if all_close else "not supported"))
    out["decisions"]["H_D"] = {"decision": hd_dec, "tiers": hd_tiers, "spearman_A_vs_mape": {"rho": sp_rho, "p": sp_p, "note": "undefined when A is constant across tiers"}}

    # ---- H-E
    he = {}
    e2_rows = []
    for code, res in results.items():
        h = res.get("H_E", {})
        tiers = [t for t in ("S2", "S3") if f"spearman_S0_vs_{t}" in h]
        rhos = {t: h[f"spearman_S0_vs_{t}"] for t in tiers}
        he[code] = {"rho": rhos, "E1_pass": bool(tiers) and all(v <= 0.5 for v in rhos.values()), "E1_high": bool(tiers) and all(v >= 0.7 for v in rhos.values())}
        for t in tiers:
            n = tier_entry(res, t)["n"]
            for m in UQ:
                w_shift = (h.get(f"{t}_native_widths") or {}).get(m)
                w_in = (h.get("S0_native_widths") or {}).get(m)
                rho = (h.get(f"{t}_spearman_abs_error_vs_native_width") or {}).get(m)
                p = (h.get(f"{t}_spearman_abs_error_vs_native_width_p") or {}).get(m)
                if p is None:
                    p = rho_p(rho, n)
                e2_rows.append({"dataset": code, "tier": t, "model": m, "width_ratio": (w_shift / w_in) if (w_shift and w_in) else None, "rho": rho, "p": p, "n": n})
    n_e1 = sum(v["E1_pass"] for v in he.values()); n_high = sum(v["E1_high"] for v in he.values())
    e1_dec = "supported" if n_e1 >= 2 else ("falsified (rho >= 0.7 in the majority)" if n_high > len(he) / 2 else "not determinable")
    ps = [r["p"] for r in e2_rows if r["p"] is not None]
    adj_p = bh(ps); k = 0
    for r in e2_rows:
        if r["p"] is not None:
            r["p_bh"] = adj_p[k]; k += 1
        else:
            r["p_bh"] = None
        r["pass"] = bool(r["width_ratio"] is not None and r["width_ratio"] > 1 and (r["rho"] or 0) > 0.3 and r["p_bh"] is not None and r["p_bh"] < 0.05)
    e2 = {}
    for m in UQ:
        ds = sorted({r["dataset"] for r in e2_rows if r["model"] == m and r["pass"]})
        e2[m] = {"datasets_passing": ds, "decision": "supported" if len(ds) >= 2 else "falsified"}
    out["decisions"]["H_E"] = {"E1_decision": e1_dec, "E1_per_dataset": he, "E2_per_model": e2, "E2_rows": e2_rows}
    out["per_dataset"] = {c: {"reveal_utc": r.get("reveal_utc"), "dev_ref": r.get("dev_ref_mape"), "tiers": list(r.get("tiers", {}).keys())} for c, r in results.items()}
    args.output.write_text(json.dumps(out, indent=2, default=lambda o: None if isinstance(o, float) and not math.isfinite(o) else str(o)) + "\n")
    for k, v in out["decisions"].items():
        print(k, "->", v.get("decision") or v.get("B1_decision") or v.get("C1_decision") or v.get("E1_decision"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
