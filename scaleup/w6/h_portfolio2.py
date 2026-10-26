#!/usr/bin/env python3
"""H-F v2: conformal bias-hedging portfolio with the pre-registered A2 screen on the sealed tiers
of R2-A1 and R2-A2 (frozen protocol validation_plan/H_F_PORTFOLIO_PROTOCOL_v2.md; post hoc, after
the two reveals).  Selection and scoring code is imported unchanged from the frozen v1 program
w6/h_portfolio.py; only unit construction, the A2 screen and the protocol-level decision are new.

Usage: python h_portfolio2.py --work <work/microfluidic_reliability> --repo-scaleup <scaleup dir> --output-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from w6.h_portfolio import (  # noqa: E402
    B_BOOT, BOOT_SEED, DECISION_KS, DELTA_THRESHOLD, K_VALUES, MIN_CLUSTERS, NOMINAL, STRATEGIES, TAU_REL, AD_K, AD_QUANTILE,
    bootstrap_indices, finite_sample_quantile, knn_ad_flags, score, select_all, sha256_file, write_json,
)

UNIT_SPECS = [
    # code, dataset, tier, cluster column (None -> rows), residual pool cell
    ("V1_R2A1_S2", "R2A1", "S2", "disp_flow", "S2_dev"),
    ("V2_R2A1_S3", "R2A1", "S3", "s2_group", "S2_dev"),
    ("V3_R2A2_S2", "R2A2", "S2", "s3_group", "S2_dev"),
    ("V4_R2A2_S3", "R2A2", "S3", None, "S1"),
    ("V5_R2A2_S4", "R2A2", "S4", "s3_group", "S1"),
]


def build_units(work: Path) -> dict[str, dict]:
    units: dict[str, dict] = {}
    cache: dict[str, dict] = {}
    for code, ds, tier, ccol, pool_cell in UNIT_SPECS:
        if ds not in cache:
            art = work / f"h_{ds}_artifacts"
            manifest = json.loads((art / "custodian_manifest.json").read_text())
            cache[ds] = {
                "native": list(manifest["native_columns"]),
                "dev": pd.read_csv(art / "development_labeled.csv", keep_default_na=False, na_values=[]),
                "sealed": pd.read_csv(art / "sealed_features.csv", keep_default_na=False, na_values=[]).set_index("row_id"),
                "pp": pd.read_csv(work / f"h_{ds}_run1" / "point_predictions.csv"),
                "rev": pd.read_csv(work / f"h_{ds}_reveal" / "sealed_scored_rows.csv"),
                "devdiag": json.loads((art / f"h_{ds}_devdiag.json").read_text()),
            }
        c = cache[ds]
        rev = c["rev"]
        part = rev[(rev.tier == tier) & (rev.model == "xgboost") & (rev.feature_set == "native")].sort_values("row_id", kind="stable").reset_index(drop=True)
        if part.empty:
            raise ValueError(f"{code}: no revealed rows")
        feats = c["sealed"].loc[part.row_id]
        pool_rows = c["pp"][(c["pp"].cell == pool_cell) & (c["pp"].model == "xgboost") & (c["pp"].feature_set == "native")]
        if pool_rows.empty:
            raise ValueError(f"{code}: residual pool cell {pool_cell} absent")
        r = np.log(pool_rows["actual_um"].to_numpy(float)) - np.log(pool_rows["pred_um"].to_numpy(float))
        dev_x = c["dev"][c["native"]].to_numpy(float)
        ad_ok, thr = knn_ad_flags(dev_x, feats[c["native"]].to_numpy(float))
        cluster = feats[ccol].astype(str).to_numpy() if ccol else part.row_id.astype(str).to_numpy()
        f, y = part["pred"].to_numpy(float), part["actual"].to_numpy(float)
        if not (np.all(f > 0) and np.all(y > 0)):
            raise ValueError(f"{code}: non-positive prediction or outcome")
        units[code] = {"code": code, "ids": part.row_id.astype(str).to_numpy(), "context": np.array([tier] * len(part)), "f": f, "y": y,
                       "cluster": cluster, "cluster_col": ccol or "row", "ad_ok": ad_ok,
                       "ad_source": f"kNN-{AD_K} q{AD_QUANTILE:g} on standardised native inputs of the {ds} development set (threshold {thr:.6g})",
                       "pool": r, "pool_note": f"h_{ds}_run1 point_predictions cell {pool_cell}, xgboost native ({len(r)} rows)",
                       "q": finite_sample_quantile(np.abs(r), NOMINAL), "s": float(np.std(r, ddof=1)), "n_pool": int(len(r)),
                       "tau_abs": float(c["devdiag"]["tau_absolute"]), "source": f"h_{ds}_run1 sealed_{tier} xgboost native (frozen bundle, revealed labels)",
                       "tier": f"{ds} {tier}" + (" (q from S2 tier as proxy)" if (tier != "S2" and pool_cell == "S2_dev") else ""), "dataset": ds}
    return units


def summarise(units: dict[str, dict], scored: pd.DataFrame, v1_summary: dict | None) -> dict:
    summary: dict = {"units": {}, "A2_screen": {}, "decision": {}, "cross_unit": {}}
    a2_units, delta5, mean_over_q, bias_b = [], {}, {}, {}
    for code, u in units.items():
        r = np.log(u["y"]) - np.log(u["f"])
        m = float(np.mean(r)); sd = float(np.std(r, ddof=1)) if len(r) > 1 else float("nan")
        b = abs(m) / sd if sd and np.isfinite(sd) and sd > 0 else None
        a2 = bool(abs(m) <= u["q"])
        rng = np.random.default_rng(BOOT_SEED)
        draws, boot_kind, n_clusters = bootstrap_indices(u["cluster"], rng)
        us = {"dataset": u["dataset"], "n_queries": int(len(u["y"])), "source": u["source"], "tier": u["tier"], "cluster_col": u["cluster_col"], "n_clusters": n_clusters,
              "bootstrap": boot_kind, "B": B_BOOT, "q_log": u["q"], "s_log": u["s"], "n_residual_pool": u["n_pool"], "pool_note": u["pool_note"], "ad_source": u["ad_source"],
              "ad_pass_fraction": float(np.mean(u["ad_ok"])), "tau_abs": u["tau_abs"],
              "bias": {"mean_log_residual": m, "sd_log_residual": sd, "abs_mean_over_q": abs(m) / u["q"] if u["q"] > 0 else None, "b_abs_mean_over_sd": b,
                       "s_majority_sign_fraction": float(max(np.mean(r > 0), np.mean(r < 0))), "A2_holds": a2},
              "results": {}}
        part_all = scored[scored.unit == code]
        holds_at = {}
        for include_self in (False, True):
            key_self = "self_excluded" if not include_self else "self_included"
            us["results"][key_self] = {}
            part_s = part_all[part_all.include_self == include_self]
            for tol_kind in ("relative", "absolute"):
                part_t = part_s[part_s.tol_kind == tol_kind]
                res_t = {}
                for k in K_VALUES:
                    part_k = part_t[part_t.K == k]
                    by = {st: part_k[part_k.strategy == st].sort_values("query_id", kind="stable") for st in STRATEGIES}
                    ids_order = by["greedy"].query_id.to_numpy()
                    if not np.array_equal(ids_order, u["ids"]):
                        raise RuntimeError(f"{code}: query order differs from the unit order")
                    succ = {st: by[st].success.to_numpy(bool) for st in STRATEGIES}
                    oracle = by["greedy"].oracle_achievable.to_numpy(bool)
                    diff = succ["portfolio_q"].astype(float) - succ["greedy"].astype(float)
                    boots = np.array([np.mean(diff[idx]) for idx in draws])
                    lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
                    delta = float(np.mean(diff)); greedy_rate = float(np.mean(succ["greedy"])); oracle_rate = float(np.mean(oracle))
                    answered = by["containment"].answered.to_numpy(bool)
                    entry = {"success_at_k": {st: float(np.mean(succ[st])) for st in STRATEGIES if st != "containment"},
                             "containment": {"answered_fraction": float(np.mean(answered)),
                                             "precision_among_answered": float(np.mean(succ["containment"][answered])) if answered.any() else None,
                                             "answered_and_success_fraction": float(np.mean(succ["containment"]))},
                             "oracle_upper_bound": oracle_rate, "delta_portfolio_minus_greedy": {"value": delta, "ci_low": lo, "ci_high": hi},
                             "delta_naive_minus_greedy": float(np.mean(succ["naive_sd"]) - greedy_rate), "oracle_limited": bool(oracle_rate < DELTA_THRESHOLD + greedy_rate)}
                    if not include_self and tol_kind == "relative":
                        holds = bool(delta >= DELTA_THRESHOLD and lo > 0)
                        entry["holds_at_k"] = holds; holds_at[k] = holds
                        if k == 5:
                            delta5[code] = delta; entry_k5 = entry
                    res_t[str(k)] = entry
                us["results"][key_self][tol_kind] = res_t
        us["unit_holds_relative_K3_and_K5"] = bool(all(holds_at.get(k, False) for k in DECISION_KS))
        us["K5_ci_high_below_zero"] = bool(entry_k5["delta_portfolio_minus_greedy"]["ci_high"] < 0)
        summary["units"][code] = us
        if a2:
            a2_units.append(code)
        mean_over_q[code] = us["bias"]["abs_mean_over_q"]; bias_b[code] = b
    # protocol-level decision on the A2 set
    holds = {c: summary["units"][c]["unit_holds_relative_K3_and_K5"] for c in a2_units}
    outside = {c: summary["units"][c]["unit_holds_relative_K3_and_K5"] for c in units if c not in a2_units}
    if not a2_units:
        verdict = "not determinable: no unit satisfies A2"
    elif all(holds.values()):
        verdict = "supported under the A2 screen"
    elif any(holds.values()) and not any(summary["units"][c]["K5_ci_high_below_zero"] for c in a2_units):
        verdict = "partially supported: " + ", ".join(f"{c}={'holds' if v else 'fails'}" for c, v in holds.items())
    else:
        verdict = "not supported under the A2 screen"
    summary["A2_screen"] = {"units_satisfying_A2": a2_units, "units_outside_A2": [c for c in units if c not in a2_units], "holds_within_A2": holds,
                            "holds_outside_A2_descriptive": outside}
    v1_holds_s2 = None
    if v1_summary:
        v1_holds_s2 = bool(v1_summary["units"]["U3_A1_S2"]["unit_holds_relative_K3_and_K5"])
    summary["decision"] = {"verdict": verdict, "sentence_recovers_usability_allowed": bool(verdict == "supported under the A2 screen" and v1_holds_s2),
                           "v1_A1_S2_holds": v1_holds_s2}
    # descriptive cross-unit correlation over v1 + v2 units
    d5, bb, mq, names = [], [], [], []
    if v1_summary:
        for c, u1 in v1_summary["units"].items():
            d5.append(u1["results"]["self_excluded"]["relative"]["5"]["delta_portfolio_minus_greedy"]["value"])
            bb.append(u1["bias_dominance"]["b_abs_mean_over_sd"]); mq.append(abs(u1["bias_dominance"]["mean_log_residual"]) / u1["q_log"]); names.append(c)
    for c in units:
        d5.append(delta5[c]); bb.append(bias_b[c]); mq.append(mean_over_q[c]); names.append(c)
    ok = [i for i in range(len(d5)) if bb[i] is not None and mq[i] is not None]
    rb = spearmanr([d5[i] for i in ok], [bb[i] for i in ok]).statistic if len(ok) >= 3 else float("nan")
    rq = spearmanr([d5[i] for i in ok], [mq[i] for i in ok]).statistic if len(ok) >= 3 else float("nan")
    summary["cross_unit"] = {"units": names, "delta_K5": dict(zip(names, d5)), "b": dict(zip(names, bb)), "abs_mean_over_q": dict(zip(names, mq)),
                             "spearman_delta5_vs_b": None if not math.isfinite(rb) else float(rb), "spearman_delta5_vs_abs_mean_over_q": None if not math.isfinite(rq) else float(rq),
                             "note": "descriptive only (v1 four units + v2 five units); not part of the decision"}
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--repo-scaleup", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing non-empty output directory {out}")
    out.mkdir(parents=True, exist_ok=True)
    units = build_units(args.work)
    rows = []
    for u in units.values():
        rows.extend(select_all(u, include_self=False))
        rows.extend(select_all(u, include_self=True))
    selections = pd.DataFrame(rows)
    selections.to_csv(out / "portfolio_selections.csv", index=False, float_format="%.10g", lineterminator="\n")
    scored = score(units, selections)
    scored.to_csv(out / "portfolio_scored.csv", index=False, float_format="%.10g", lineterminator="\n")
    v1_path = args.repo_scaleup / "h_portfolio" / "portfolio_summary.json"
    v1 = json.loads(v1_path.read_text()) if v1_path.exists() else None
    summary = {"schema_version": "h-portfolio-v2", "status": "retrospective; post hoc; after the R2-A1/R2-A2 reveals; protocol H_F_PORTFOLIO_PROTOCOL_v2.md frozen before the reveals",
               "constants": {"nominal": NOMINAL, "K": list(K_VALUES), "tau_rel": TAU_REL, "B": B_BOOT, "boot_seed": BOOT_SEED, "min_clusters": MIN_CLUSTERS,
                             "delta_threshold": DELTA_THRESHOLD, "decision_K": list(DECISION_KS), "ad_k": AD_K, "ad_quantile": AD_QUANTILE},
               **summarise(units, scored, v1)}
    write_json(out / "portfolio_summary.json", summary)
    inputs = {}
    for ds in ("R2A1", "R2A2"):
        for rel in (f"h_{ds}_artifacts/development_labeled.csv", f"h_{ds}_artifacts/sealed_features.csv", f"h_{ds}_artifacts/custodian_manifest.json",
                    f"h_{ds}_run1/point_predictions.csv", f"h_{ds}_reveal/sealed_scored_rows.csv"):
            inputs[rel] = sha256_file(args.work / rel)
    write_json(out / "environment.json", {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                                          "sklearn": sklearn.__version__, "platform": platform.platform(), "script_sha256": sha256_file(Path(__file__).resolve()),
                                          "v1_program_sha256": sha256_file(Path(__file__).with_name("h_portfolio.py")), "v1_summary_sha256": sha256_file(v1_path) if v1 else None,
                                          "input_sha256": inputs})
    manifest = {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.name != "artifact_hashes.json"}
    write_json(out / "artifact_hashes.json", manifest)
    for code, s in summary["units"].items():
        rel = s["results"]["self_excluded"]["relative"]
        print(code, "n=%d q=%.3f |m|/q=%.2f A2=%s" % (s["n_queries"], s["q_log"], s["bias"]["abs_mean_over_q"], s["bias"]["A2_holds"]),
              {k: (round(v["success_at_k"]["greedy"], 3), round(v["success_at_k"]["portfolio_q"], 3), round(v["delta_portfolio_minus_greedy"]["value"], 3)) for k, v in rel.items()}, "holds:", s["unit_holds_relative_K3_and_K5"])
    print("decision:", summary["decision"], "| cross:", {k: summary["cross_unit"][k] for k in ("spearman_delta5_vs_b", "spearman_delta5_vs_abs_mean_over_q")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
