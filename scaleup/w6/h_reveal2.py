#!/usr/bin/env python3
"""H_PROTOCOL_v2 reveal program (binding-driven; same gate, scoring and single-dataset decision
blocks as the frozen v1 program h_reveal.py, generalised to the unified column names and to an
optional S4 tier).  The multi-dataset decisions of protocol section 5 are produced afterwards by
h_combine.py from the per-dataset result files.

Usage: python h_reveal2.py --binding <json> --protocol <md> --artifacts <custodian dir> --run <h_run2 dir> --output <reveal dir>
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BOOT_N, BOOT_SEED = 2000, 0
SIX = ["xgboost", "random_forest", "mlp_log", "qrf", "gp", "deep_ensemble"]
BAND = (2.2, 3.0)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sanitize(v):
    if isinstance(v, dict):
        return {str(k): sanitize(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [sanitize(x) for x in v]
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if not math.isfinite(f) else f
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(sanitize(payload), indent=2, allow_nan=False) + "\n")


def mape(y, p) -> float:
    return float(100.0 * np.mean(np.abs(p - y) / np.abs(y)))


def mdape(y, p) -> float:
    return float(100.0 * np.median(np.abs(p - y) / np.abs(y)))


def boot_mape(y, p, n=BOOT_N, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = [mape(y[s], p[s]) for s in (rng.choice(idx, size=len(idx), replace=True) for _ in range(n))]
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binding", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--runner", type=Path, default=Path(__file__).with_name("h_run2.py"))
    args = ap.parse_args(argv)
    binding = json.loads(args.binding.read_text())
    manifest = json.loads((args.artifacts / "custodian_manifest.json").read_text())
    run_hashes = json.loads((args.run / "artifact_hashes.json").read_text())
    env = json.loads((args.run / "environment.json").read_text())
    checks = {
        "protocol": sha256_file(args.protocol) == binding["protocol_sha256"],
        "binding": sha256_file(args.binding) == manifest["binding"]["sha256"],
        "runner_script": sha256_file(args.runner) == env["script_sha256"],
        "bundle": sha256_file(args.run / "sealed_prediction_bundle.csv") == run_hashes["sealed_prediction_bundle.csv"],
        "bundle_sidecar": (args.run / "sealed_prediction_bundle.sha256").read_text().split()[0] == run_hashes["sealed_prediction_bundle.csv"],
        "sealed_features": sha256_file(args.artifacts / "sealed_features.csv") == manifest["artifacts"]["sealed_features"]["sha256"],
        "sealed_labels": sha256_file(args.artifacts / "sealed_labels.csv") == manifest["artifacts"]["sealed_labels"]["sha256"],
        "development": sha256_file(args.artifacts / "development_labeled.csv") == manifest["artifacts"]["development_labeled"]["sha256"],
    }
    if not all(checks.values()):
        raise SystemExit(f"digest checks failed: {checks}")
    args.output.mkdir(parents=True, exist_ok=True)
    gate = args.output / "REVEAL_GATE_CONSUMED.json"
    fd = os.open(gate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)  # a second reveal is refused here
    reveal_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with os.fdopen(fd, "w") as fh:
        json.dump({"reveal_utc": reveal_utc, "bundle_sha256": run_hashes["sealed_prediction_bundle.csv"], "checks": checks}, fh, indent=2)

    labels = pd.read_csv(args.artifacts / "sealed_labels.csv")
    feats = pd.read_csv(args.artifacts / "sealed_features.csv", keep_default_na=False, na_values=[])
    bundle = pd.read_csv(args.run / "sealed_prediction_bundle.csv")
    summary_dev = json.loads((args.run / "h_run_summary.json").read_text())
    dev_ref = float(summary_dev["dev_ref"]["mape_mean"])
    tau_abs = float(summary_dev["tau_abs"])
    joined = bundle.merge(labels, on="row_id", how="inner").merge(feats[["row_id", "s2_group", "s3_group", "s4_group", "sealed_tier"]], on="row_id", how="inner")
    if len(joined) != len(bundle):
        raise SystemExit("bundle rows and sealed labels do not join one-to-one")
    tiers = [t for t in ("S2", "S3", "S4") if (joined["sealed_tier"].astype(str) == t).any()]
    has_dimless = bool(summary_dev.get("n_dimensionless_features"))

    results = {"schema_version": "h-reveal-v2", "dataset_code": binding["dataset_code"], "reveal_utc": reveal_utc, "checks": checks,
               "dev_ref_mape": dev_ref, "dev_ref_grouping": summary_dev["dev_ref"]["grouping"], "tau_abs": tau_abs, "log_target": summary_dev["log_target"],
               "tiers_present": tiers, "tiers": {}}
    scored_rows = []
    for tier in tiers:
        part = joined[joined["cell"] == f"sealed_{tier}"]
        seal = manifest["sealing"][tier]
        tier_block = {"n_rows": int(part["row_id"].nunique()), "sealed_level": seal.get("level") if isinstance(seal, dict) else seal, "models": {}}
        for (model, fset), g in part.groupby(["model", "feature_set"]):
            y = g["target"].to_numpy(float); p = g["pred_um"].to_numpy(float)
            lo, hi = g["lower_um__split_conformal"].to_numpy(float), g["upper_um__split_conformal"].to_numpy(float)
            ci = boot_mape(y, p)
            e = {"n": int(len(y)), "mape": mape(y, p), "mape_ci95": ci, "mdape": mdape(y, p), "ratio_to_dev_ref": mape(y, p) / dev_ref,
                 "ratio_ci95": [ci[0] / dev_ref, ci[1] / dev_ref], "n_nonpositive_pred": int((p <= 0).sum()),
                 "signed_bias": float(np.mean((p - y) / y)), "fraction_over_predicted": float(np.mean(p > y)),
                 "split_coverage": float(np.mean((y >= lo) & (y <= hi))), "split_mean_width": float(np.mean(hi - lo)),
                 "A_abs": float(np.mean((np.abs(p - y) <= tau_abs) & (lo >= y - tau_abs) & (hi <= y + tau_abs))), "W_abs": float(np.mean((hi - lo) < 2 * tau_abs)),
                 "A_rel": float(np.mean((np.abs(p - y) <= 0.1 * y) & (lo >= 0.9 * y) & (hi <= 1.1 * y))), "W_rel": float(np.mean((hi - lo) < 0.2 * y))}
            if "lower_um__cqr" in g and g["lower_um__cqr"].notna().any():
                c_lo, c_hi = g["lower_um__cqr"].to_numpy(float), g["upper_um__cqr"].to_numpy(float)
                e["cqr_coverage"] = float(np.mean((y >= c_lo) & (y <= c_hi))); e["cqr_mean_width"] = float(np.mean(c_hi - c_lo))
            if "lower_um__native" in g and g["lower_um__native"].notna().any():
                n_lo, n_hi = g["lower_um__native"].to_numpy(float), g["upper_um__native"].to_numpy(float)
                e["native_coverage"] = float(np.mean((y >= n_lo) & (y <= n_hi))); e["native_mean_width"] = float(np.mean(n_hi - n_lo))
                sp = stats.spearmanr(np.abs(p - y), n_hi - n_lo)
                e["spearman_abs_error_vs_native_width"] = float(sp.correlation); e["spearman_abs_error_vs_native_width_p"] = float(sp.pvalue)
                k_lo, k_hi = g["lower_um__conformalised_native"].to_numpy(float), g["upper_um__conformalised_native"].to_numpy(float)
                e["conformalised_native_coverage"] = float(np.mean((y >= k_lo) & (y <= k_hi)))
            if tier in ("S3", "S4"):
                e["per_s2_level_mape"] = {lv: mape(gg["target"].to_numpy(float), gg["pred_um"].to_numpy(float)) for lv, gg in g.groupby("s2_group")}
            if tier == "S4":
                e["per_s3_level_mape"] = {lv: mape(gg["target"].to_numpy(float), gg["pred_um"].to_numpy(float)) for lv, gg in g.groupby("s3_group")}
            tier_block["models"][f"{model}|{fset}"] = e
            for _, r in g.iterrows():
                scored_rows.append({"tier": tier, "model": model, "feature_set": fset, "row_id": r["row_id"], "s2_group": r["s2_group"], "s3_group": r["s3_group"],
                                    "actual": r["target"], "pred": r["pred_um"], "lower": r["lower_um__split_conformal"], "upper": r["upper_um__split_conformal"]})
        results["tiers"][tier] = tier_block

    main_tiers = [t for t in ("S2", "S3") if t in tiers]
    xs = {t: results["tiers"][t]["models"]["xgboost|native"] for t in main_tiers}
    def in_band(e): return BAND[0] <= e["ratio_to_dev_ref"] <= BAND[1]
    def ci_disjoint(e): return e["ratio_ci95"][1] < BAND[0] or e["ratio_ci95"][0] > BAND[1]
    ha = {"statement": "shift multiplier r = MAPE_sealed / dev-ref lies in [2.2, 3.0] for S2/S3 (xgboost, native features)",
          "per_tier": {t: {"r": xs[t]["ratio_to_dev_ref"], "r_ci95": xs[t]["ratio_ci95"], "in_band": in_band(xs[t]), "ci_disjoint_from_band": ci_disjoint(xs[t])} for t in main_tiers},
          "S2_dev_LOCO_r_retrospective": summary_dev["H_A_dev"]["ratio_to_dev_ref"],
          "S1_dispersion_max_over_min": summary_dev["H_A_dev"].get("S1_dispersion_max_over_min")}
    if len(main_tiers) == 2:
        rs = [xs[t]["ratio_to_dev_ref"] for t in main_tiers]
        ha["S2_S3_dispersion_max_over_min"] = float(max(rs) / min(rs))
        ha["dispersion_within_S1"] = (ha["S2_S3_dispersion_max_over_min"] <= ha["S1_dispersion_max_over_min"]) if ha["S1_dispersion_max_over_min"] else None
    if "S4" in tiers:
        x4 = results["tiers"]["S4"]["models"]["xgboost|native"]
        ha["S4"] = {"r": x4["ratio_to_dev_ref"], "r_ci95": x4["ratio_ci95"],
                    "S4_lower_above_S2S3_upper": bool(x4["ratio_ci95"][0] > max(xs[t]["ratio_ci95"][1] for t in main_tiers))}
    any_falsify = any(v["ci_disjoint_from_band"] for v in ha["per_tier"].values()) or (ha.get("dispersion_within_S1") is False)
    all_band = all(v["in_band"] for v in ha["per_tier"].values()) and (ha.get("dispersion_within_S1") in (True, None)) and (ha.get("S4", {}).get("S4_lower_above_S2S3_upper", True))
    ha["decision_single_dataset"] = "falsified_on_this_dataset" if any_falsify else ("consistent_on_this_dataset" if all_band else "point_estimate_outside_band_but_ci_overlaps")
    ha["decision_protocol"] = "deferred to h_combine.py (multi-dataset rule of section 5)"
    results["H_A"] = ha
    hb = {}
    for tier in main_tiers:
        mb, mu = results["tiers"][tier]["models"]["mlp_log|native"], results["tiers"][tier]["models"]["mlp_raw|native"]
        hb[tier] = {"bounded_mape": mb["mape"], "unbounded_mape": mu["mape"], "bounded_nonpositive": mb["n_nonpositive_pred"], "unbounded_nonpositive": mu["n_nonpositive_pred"],
                    "bounded_lower_error_and_no_invalid": bool(mb["mape"] < mu["mape"] and mb["n_nonpositive_pred"] == 0)}
    results["H_B"] = {"B1": hb, "B1_decision_single_dataset": "consistent_on_this_dataset" if all(v["bounded_lower_error_and_no_invalid"] for v in hb.values()) else "falsified_on_this_dataset",
                      "B1_decision_protocol": "deferred to h_combine.py", "B2": "not_evaluable: no anchor defined (binding)"}
    if has_dimless:
        c1 = {}
        for tier in main_tiers:
            n_, d_ = results["tiers"][tier]["models"]["xgboost|native"], results["tiers"][tier]["models"]["xgboost|dimensionless"]
            c1[tier] = {"native_mape": n_["mape"], "dimensionless_mape": d_["mape"], "diff_native_minus_dimensionless_pp": n_["mape"] - d_["mape"]}
        hc_dev = summary_dev["H_C_dev"]
        results["H_C"] = {"evaluable": True, "C1_sealed": c1, "C1_dev_paired": hc_dev, "C2": summary_dev["H_C_C2_pairs"],
                          "C1_decision_single_dataset": "consistent_on_this_dataset" if (hc_dev["median_diff_native_minus_dimensionless_pp"] < 0 and (hc_dev["wilcoxon_p"] or 1) < 0.05) else "not_supported_on_this_dataset",
                          "C1_decision_protocol": "deferred to h_combine.py",
                          "C2_decision": "evaluable (pairs listed in the run summary)" if summary_dev["H_C_C2_pairs"]["evaluable"] else "not_determinable: fewer than 20 equal-dimensionless-group pairs with different absolute scale"}
    else:
        results["H_C"] = {"evaluable": False, "reason": "no dimensionless inputs on this dataset"}
    hd = {}
    for tier in main_tiers:
        e = results["tiers"][tier]["models"]["xgboost|native"]
        hd[tier] = {"A_abs": e["A_abs"], "W_abs": e["W_abs"], "A_minus_W_abs": e["A_abs"] - e["W_abs"], "A_rel": e["A_rel"], "W_rel": e["W_rel"], "A_minus_W_rel": e["A_rel"] - e["W_rel"], "mape": e["mape"]}
    results["H_D"] = {"per_tier": hd, "dev_per_level": summary_dev["H_D_dev"],
                      "decision_single_dataset": "consistent_on_this_dataset" if all(abs(v["A_minus_W_abs"]) <= 0.10 for v in hd.values()) else ("falsified_on_this_dataset" if any(v["A_minus_W_abs"] > 0.10 for v in hd.values()) else "A_below_W_by_more_than_0.10"),
                      "decision_protocol": "deferred to h_combine.py"}
    s0 = pd.Series(summary_dev["H_E_dev"]["S0_mape"]).reindex(SIX)
    he = {"S0_mape": s0.to_dict()}
    for tier in main_tiers:
        m = pd.Series({k: results["tiers"][tier]["models"][f"{k}|native"]["mape"] for k in SIX})
        he[f"{tier}_mape"] = m.to_dict()
        he[f"spearman_S0_vs_{tier}"] = float(stats.spearmanr(s0.rank(), m.rank()).correlation)
        he[f"{tier}_native_widths"] = {k: results["tiers"][tier]["models"][f"{k}|native"].get("native_mean_width") for k in ("qrf", "gp", "deep_ensemble")}
        he[f"{tier}_spearman_abs_error_vs_native_width"] = {k: results["tiers"][tier]["models"][f"{k}|native"].get("spearman_abs_error_vs_native_width") for k in ("qrf", "gp", "deep_ensemble")}
        he[f"{tier}_spearman_abs_error_vs_native_width_p"] = {k: results["tiers"][tier]["models"][f"{k}|native"].get("spearman_abs_error_vs_native_width_p") for k in ("qrf", "gp", "deep_ensemble")}
    he["S0_native_widths"] = {k: v["S0_mean_native_width_um"] for k, v in summary_dev["H_E_dev"]["native_width_ratio"].items()}
    he["E1_decision_single_dataset"] = "consistent_on_this_dataset" if all(he[f"spearman_S0_vs_{t}"] <= 0.5 for t in main_tiers) else "not_supported_on_this_dataset"
    he["E1_decision_protocol"] = "deferred to h_combine.py"
    he["E2_decision_single_dataset"] = {t: {k: bool((he[f"{t}_native_widths"][k] or 0) > (he["S0_native_widths"][k] or np.inf) and (he[f"{t}_spearman_abs_error_vs_native_width"][k] or 0) > 0.3) for k in ("qrf", "gp", "deep_ensemble")} for t in main_tiers}
    results["H_E"] = he

    pd.DataFrame(scored_rows).to_csv(args.output / "sealed_scored_rows.csv", index=False, float_format="%.10g", lineterminator="\n")
    write_json(args.output / "h_reveal_results.json", results)
    hashes = {p.name: sha256_file(p) for p in sorted(args.output.iterdir()) if p.name != "artifact_hashes.json"}
    write_json(args.output / "artifact_hashes.json", hashes)
    print(json.dumps(sanitize({t: {k: xs[t][k] for k in ("n", "mape", "mape_ci95", "ratio_to_dev_ref", "split_coverage")} for t in main_tiers}), indent=1))
    if "S4" in tiers:
        print("S4 xgb:", json.dumps(sanitize({k: results["tiers"]["S4"]["models"]["xgboost|native"][k] for k in ("n", "mape", "mape_ci95", "ratio_to_dev_ref")})))
    print("H_A", ha["decision_single_dataset"], "| H_B", results["H_B"]["B1_decision_single_dataset"], "| H_C", results["H_C"].get("C1_decision_single_dataset", "n/a"),
          "| H_D", results["H_D"]["decision_single_dataset"], "| H_E", he["E1_decision_single_dataset"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
