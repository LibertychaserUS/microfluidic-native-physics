#!/usr/bin/env python3
"""H-F: conformal bias-hedging portfolio on the target-search replay (frozen protocol
validation_plan/H_F_PORTFOLIO_PROTOCOL_v1.md; retrospective; all labels already open).

Reads only frozen artifacts (no model is trained here):
  U1 D1 external   W3 cell D1_external, xgboost        (w3_artifacts_canonical_run2/point_predictions.csv)
  U2 D2 TX100      W3 cell D2_chemistry_loco repeat 2   (same file)
  U3 A1 S2         h_A1_run1/point_predictions.csv cell sealed_S2, xgboost native
  U4 A1 S3         h_A1_run1/point_predictions.csv cell sealed_S3, xgboost native
Group-calibrated conformal radius q (log scale) and naive SD s come from frozen out-of-group
predictions (D1_geometry, the other D2 LOCO folds, A1 S2_dev).  Selections are written
before any outcome is read; scoring follows.  Decision rules are those of the protocol and
are computed here, never edited by hand.

Usage: python h_portfolio.py --work <work/microfluidic_reliability> --output-dir <dir>
"""

from __future__ import annotations

import argparse
import hashlib
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
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from w3.datasets import load_d1, load_d2  # noqa: E402
from w3.workspace import load_workspace  # noqa: E402

NOMINAL = 0.90
K_VALUES = (1, 2, 3, 5)
TAU_REL = 0.10
B_BOOT = 2000
BOOT_SEED = 0
MIN_CLUSTERS = 8
DELTA_THRESHOLD = 0.10
DECISION_KS = (3, 5)
AD_K = 5
AD_QUANTILE = 0.95
A1_NATIVE = ["Qd [ul/min]", "Qc [ul/min]", "Eta_d [Pa*s]", "Eta_c [Pa*s]", "Interfacial Tension [mN*m]", "Etching Depth [um]", "Width [um]"]
STRATEGIES = ("greedy", "portfolio_q", "naive_sd", "ad_greedy", "containment")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sanitize(value):
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if not math.isfinite(f) else f
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(sanitize(payload), indent=2, allow_nan=False) + "\n")


def finite_sample_quantile(scores: np.ndarray, coverage: float) -> float:
    n = len(scores)
    level = min(1.0, math.ceil((n + 1) * coverage) / n)
    return float(np.quantile(scores, level, method="higher"))


# ----------------------------------------------------------------------------- units
def knn_ad_flags(reference: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, float]:
    """kNN-5 q95 applicability on standardised inputs (reference leave-one-out threshold)."""
    mu, sd = reference.mean(axis=0), reference.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    zr, zq = (reference - mu) / sd, (query - mu) / sd
    nn = NearestNeighbors(n_neighbors=AD_K + 1).fit(zr)
    ref_d = nn.kneighbors(zr)[0][:, AD_K]  # exclude self (distance 0 at position 0)
    threshold = float(np.quantile(ref_d, AD_QUANTILE))
    q_d = NearestNeighbors(n_neighbors=AD_K).fit(zr).kneighbors(zq)[0][:, AD_K - 1]
    return q_d <= threshold, threshold


def build_units(work: Path) -> tuple[dict[str, dict], dict]:
    w3pp = pd.read_csv(work / "w3_artifacts_canonical_run2" / "point_predictions.csv")
    w3ad = pd.read_csv(work / "w3_artifacts_canonical_run2" / "ad_scores.csv")
    a1pp = pd.read_csv(work / "h_A1_run1" / "point_predictions.csv")
    # the run-side file carries NaN for sealed outcomes; the revealed outcomes live in the reveal artifact (H-D9)
    a1rev = pd.read_csv(work / "h_A1_reveal" / "sealed_scored_rows.csv")
    a1_sealed = pd.read_csv(work / "h_A1_artifacts" / "sealed_features.csv")
    a1_dev = pd.read_csv(work / "h_A1_artifacts" / "development_labeled.csv")
    a1_diag = json.loads((work / "h_A1_artifacts" / "h_A1_devdiag.json").read_text())
    ws = load_workspace(work)
    d1_comp, _ = load_d1(ws)
    d2 = load_d2(ws)
    chem = d2.groups["chemistry"]
    d2_conc = dict(zip(d2.row_id, d2.frame[ws.v2.CONCENTRATION].to_numpy(float)))
    medians = {"D1_comprehensive": float(np.median(d1_comp.target_um)), "D2_non_tx100": float(np.median(d2.target_um[chem != "3.5"])),
               "A1_development": float(a1_diag["development_median_L_um"])}
    units: dict[str, dict] = {}

    def residual_pool(frame: pd.DataFrame) -> np.ndarray:
        return np.log(frame["actual_um"].to_numpy(float)) - np.log(frame["pred_um"].to_numpy(float))

    # U1 D1 external
    ext = w3pp[(w3pp.cell == "D1_external") & (w3pp.model == "xgboost")].sort_values("row_id", kind="stable").reset_index(drop=True)
    ad1 = w3ad[w3ad.cell == "D1_external"].set_index("row_id")["accepted__knn_5__q0_95"]
    if len(ext) != 64:
        raise ValueError("U1 expects 64 rows")
    units["U1_D1_external"] = {
        "ids": ext.row_id.astype(str).to_numpy(), "context": ext.group.astype(str).to_numpy(), "f": ext.pred_um.to_numpy(float),
        "y": ext.actual_um.to_numpy(float), "cluster": ext.group.astype(str).to_numpy(), "cluster_col": "literature source",
        "ad_ok": ad1.reindex(ext.row_id).to_numpy(bool), "ad_source": "W3 ad_scores accepted__knn_5__q0_95 (frozen)",
        "pool": residual_pool(w3pp[(w3pp.cell == "D1_geometry") & (w3pp.model == "xgboost")]),
        "pool_note": "W3 D1_geometry xgboost test rows, 30 group-split repeats pooled", "tau_abs": TAU_REL * medians["D1_comprehensive"],
        "source": "W3 cell D1_external xgboost (frozen point_predictions.csv)", "tier": "unseen literature source",
    }
    # U2 D2 TX100
    loco = w3pp[(w3pp.cell == "D2_chemistry_loco") & (w3pp.model == "xgboost")]
    tx = loco[loco.repeat == 2].sort_values("row_id", kind="stable").reset_index(drop=True)
    if len(tx) != 87 or set(tx.group.astype(str)) != {"3.5"}:
        raise ValueError("U2 expects the 87 TX100 rows of LOCO fold 2")
    ad2 = w3ad[(w3ad.cell == "D2_chemistry_loco") & (w3ad.repeat == 2)].set_index("row_id")["accepted__knn_5__q0_95"]
    units["U2_D2_tx100"] = {
        "ids": tx.row_id.astype(str).to_numpy(), "context": np.array(["tx100"] * len(tx)), "f": tx.pred_um.to_numpy(float),
        "y": tx.actual_um.to_numpy(float), "cluster": np.array([format(d2_conc[r], ".6g") for r in tx.row_id]), "cluster_col": "surfactant concentration",
        "ad_ok": ad2.reindex(tx.row_id).to_numpy(bool), "ad_source": "W3 ad_scores accepted__knn_5__q0_95 (frozen, LOCO fold 2)",
        "pool": residual_pool(loco[loco.repeat != 2]), "pool_note": "W3 D2_chemistry_loco xgboost test rows of the five non-TX100 folds pooled",
        "tau_abs": TAU_REL * medians["D2_non_tx100"], "source": "W3 cell D2_chemistry_loco fold 2 (held-out CMC 3.5) xgboost (frozen)",
        "tier": "unseen chemistry",
    }
    # U3 / U4 A1 sealed tiers
    dev_x = a1_dev[A1_NATIVE].to_numpy(float)
    sealed_idx = a1_sealed.set_index("row_id")
    s2dev = a1pp[(a1pp.cell == "S2_dev") & (a1pp.model == "xgboost") & (a1pp.feature_set == "native")]
    if len(s2dev) != 661:
        raise ValueError("A1 S2_dev pool expects 661 rows")
    pool_a1 = residual_pool(s2dev)
    for code, cell, ccol, cname, tier in (("U3_A1_S2", "sealed_S2", "Qc [ul/min]", "Qc level", "unseen fluid family"),
                                          ("U4_A1_S3", "sealed_S3", "fluid_family", "fluid family", "unseen device instance (q from the S2 tier as proxy)")):
        part = a1pp[(a1pp.cell == cell) & (a1pp.model == "xgboost") & (a1pp.feature_set == "native")].sort_values("row_id", kind="stable").reset_index(drop=True)
        rev = a1rev[(a1rev.tier == cell.replace("sealed_", "")) & (a1rev.model == "xgboost") & (a1rev.feature_set == "native")].set_index("row_id")
        rev = rev.reindex(part.row_id)
        if rev["actual_um"].isna().any() or not np.allclose(rev["pred_um"].to_numpy(float), part.pred_um.to_numpy(float), rtol=0, atol=1e-9):
            raise ValueError(f"{cell}: reveal rows do not match the run-side predictions")
        part = part.assign(actual_um=rev["actual_um"].to_numpy(float))
        feats = sealed_idx.loc[part.row_id]
        ad_ok, thr = knn_ad_flags(dev_x, feats[A1_NATIVE].to_numpy(float))
        units[code] = {
            "ids": part.row_id.astype(str).to_numpy(), "context": np.array([cell] * len(part)), "f": part.pred_um.to_numpy(float),
            "y": part.actual_um.to_numpy(float), "cluster": feats[ccol].astype(str).to_numpy(), "cluster_col": cname,
            "ad_ok": ad_ok, "ad_source": f"kNN-{AD_K} q{AD_QUANTILE:g} computed here on standardised native inputs (threshold {thr:.6g})",
            "pool": pool_a1, "pool_note": "h_A1_run1 S2_dev xgboost native leave-one-fluid-family-out rows (661)",
            "tau_abs": TAU_REL * medians["A1_development"], "source": f"h_A1_run1 cell {cell} xgboost native (frozen)", "tier": tier,
        }
    for u in units.values():
        r = u["pool"]
        u["q"] = finite_sample_quantile(np.abs(r), NOMINAL)
        u["s"] = float(np.std(r, ddof=1))
        u["n_pool"] = int(len(r))
        if not (np.all(u["f"] > 0) and np.all(u["y"] > 0)):
            raise ValueError("non-positive prediction or outcome")
    return units, medians


# ----------------------------------------------------------------------------- selection (label-free)
def order_by_gap(target_log: float, lf: np.ndarray, ids: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    cand = np.flatnonzero(allowed)
    gap = np.abs(lf[cand] - target_log)
    return cand[np.lexsort((ids[cand], gap))]


def spread_select(d_log: float, radius: float, k: int, lf: np.ndarray, ids: np.ndarray, allowed: np.ndarray) -> list[int]:
    chosen: list[int] = []
    taken = np.zeros(len(lf), dtype=bool)
    for m in range(1, k + 1):
        sub = d_log + ((2 * m - k - 1) / k) * radius
        order = order_by_gap(sub, lf, ids, allowed & ~taken)
        if len(order) == 0:
            break
        chosen.append(int(order[0]))
        taken[order[0]] = True
    return chosen


def select_all(u: dict, include_self: bool) -> list[dict]:
    lf = np.log(u["f"])
    ids, ctx = u["ids"], u["context"]
    n = len(ids)
    rows = []
    for i in range(n):
        d_log = math.log(u["y"][i])  # the query is the observed value; no outcome of any candidate is read here
        base_allowed = ctx == ctx[i]
        if not include_self:
            base_allowed = base_allowed & (np.arange(n) != i)
        greedy_order = order_by_gap(d_log, lf, ids, base_allowed)
        for k in K_VALUES:
            sel = {
                "greedy": [int(j) for j in greedy_order[:k]],
                "portfolio_q": spread_select(d_log, u["q"], k, lf, ids, base_allowed),
                "naive_sd": spread_select(d_log, u["s"], k, lf, ids, base_allowed),
                "ad_greedy": [int(j) for j in order_by_gap(d_log, lf, ids, base_allowed & u["ad_ok"])[:k]],
            }
            for tol_kind in ("relative", "absolute"):
                tau = TAU_REL * u["y"][i] if tol_kind == "relative" else u["tau_abs"]
                answered = False
                if len(greedy_order):
                    j = int(greedy_order[0])
                    fj = u["f"][j]
                    answered = bool(abs(fj - u["y"][i]) <= tau and fj * math.exp(-u["q"]) >= u["y"][i] - tau and fj * math.exp(u["q"]) <= u["y"][i] + tau)
                for strategy in STRATEGIES:
                    chosen = sel["greedy"] if strategy == "containment" else sel[strategy]
                    if strategy == "containment" and not answered:
                        chosen = []
                    rows.append({"unit": u["code"], "include_self": include_self, "query_id": ids[i], "cluster": u["cluster"][i],
                                 "tol_kind": tol_kind, "tau": tau, "K": k, "strategy": strategy, "answered": answered if strategy == "containment" else True,
                                 "n_pool": int(base_allowed.sum()), "selected_ids": ";".join(ids[j] for j in chosen)})
    return rows


# ----------------------------------------------------------------------------- scoring (outcomes read here)
def score(units: dict[str, dict], selections: pd.DataFrame) -> pd.DataFrame:
    out = []
    for u_code, part in selections.groupby("unit", sort=False):
        u = units[u_code]
        y_by_id = dict(zip(u["ids"], u["y"]))
        ctx_by_id = dict(zip(u["ids"], u["context"]))
        for rec in part.to_dict(orient="records"):
            d = y_by_id[rec["query_id"]]
            others = np.array([y_by_id[j] for j in u["ids"] if ctx_by_id[j] == ctx_by_id[rec["query_id"]] and (rec["include_self"] or j != rec["query_id"])])
            oracle = bool(np.min(np.abs(others - d)) <= rec["tau"]) if len(others) else False
            chosen = [j for j in str(rec["selected_ids"]).split(";") if j]
            err = float(np.min([abs(y_by_id[j] - d) for j in chosen])) if chosen else float("nan")
            success = bool(chosen) and err <= rec["tau"]
            out.append({**rec, "oracle_achievable": oracle, "achieved_error_um": err, "success": success})
    return pd.DataFrame(out)


def bootstrap_indices(cluster: np.ndarray, rng: np.random.Generator) -> tuple[list[np.ndarray], str, int]:
    levels = np.unique(cluster)
    n = len(cluster)
    if len(levels) >= MIN_CLUSTERS:
        members = {lv: np.flatnonzero(cluster == lv) for lv in levels}
        draws = []
        for _ in range(B_BOOT):
            picked = rng.choice(levels, size=len(levels), replace=True)
            draws.append(np.concatenate([members[lv] for lv in picked]))
        return draws, "cluster", int(len(levels))
    return [rng.choice(n, size=n, replace=True) for _ in range(B_BOOT)], "row", int(len(levels))


def summarise(units: dict[str, dict], scored: pd.DataFrame, medians: dict) -> dict:
    summary: dict = {"units": {}, "cross_unit": {}}
    delta5, bias_b, bias_s, holds_all = [], [], [], []
    for code, u in units.items():
        r = np.log(u["y"]) - np.log(u["f"])
        b = float(abs(np.mean(r)) / np.std(r, ddof=1))
        s_sign = float(max(np.mean(r > 0), np.mean(r < 0)))
        rng = np.random.default_rng(BOOT_SEED)
        draws, boot_kind, n_clusters = bootstrap_indices(u["cluster"], rng)
        unit_s = {"n_queries": int(len(u["y"])), "source": u["source"], "tier": u["tier"], "contexts": {str(k): int(v) for k, v in zip(*np.unique(u["context"], return_counts=True))},
                  "cluster_col": u["cluster_col"], "n_clusters": n_clusters, "bootstrap": boot_kind, "B": B_BOOT,
                  "q_log": u["q"], "s_log": u["s"], "n_residual_pool": u["n_pool"], "pool_note": u["pool_note"], "ad_source": u["ad_source"],
                  "ad_pass_fraction": float(np.mean(u["ad_ok"])), "tau_abs_um": u["tau_abs"],
                  "bias_dominance": {"mean_log_residual": float(np.mean(r)), "sd_log_residual": float(np.std(r, ddof=1)), "b_abs_mean_over_sd": b, "s_majority_sign_fraction": s_sign,
                                     "fraction_positive": float(np.mean(r > 0)), "n": int(len(r))},
                  "results": {}}
        part_all = scored[scored.unit == code]
        holds_at = {}
        for include_self in (False, True):
            key_self = "self_excluded" if not include_self else "self_included"
            unit_s["results"][key_self] = {}
            part_s = part_all[part_all.include_self == include_self]
            for tol_kind in ("relative", "absolute"):
                part_t = part_s[part_s.tol_kind == tol_kind]
                res_t = {}
                for k in K_VALUES:
                    part_k = part_t[part_t.K == k]
                    by = {st: part_k[part_k.strategy == st].sort_values("query_id", kind="stable") for st in STRATEGIES}
                    ids_order = by["greedy"].query_id.to_numpy()
                    succ = {st: by[st].success.to_numpy(bool) for st in STRATEGIES}
                    for st in STRATEGIES:
                        if not np.array_equal(by[st].query_id.to_numpy(), ids_order):
                            raise RuntimeError("query alignment")
                    oracle = by["greedy"].oracle_achievable.to_numpy(bool)
                    if not np.array_equal(ids_order, u["ids"]):
                        raise RuntimeError(f"{code}: query order differs from the unit order; bootstrap draws would be misaligned")
                    diff = succ["portfolio_q"].astype(float) - succ["greedy"].astype(float)
                    boots = np.array([np.mean(diff[idx]) for idx in draws])
                    lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
                    delta = float(np.mean(diff))
                    greedy_rate = float(np.mean(succ["greedy"]))
                    oracle_rate = float(np.mean(oracle))
                    answered = by["containment"].answered.to_numpy(bool)
                    entry = {"success_at_k": {st: float(np.mean(succ[st])) for st in STRATEGIES if st != "containment"},
                             "containment": {"answered_fraction": float(np.mean(answered)),
                                             "precision_among_answered": float(np.mean(succ["containment"][answered])) if answered.any() else None,
                                             "answered_and_success_fraction": float(np.mean(succ["containment"]))},
                             "oracle_upper_bound": oracle_rate,
                             "delta_portfolio_minus_greedy": {"value": delta, "ci_low": lo, "ci_high": hi},
                             "delta_naive_minus_greedy": float(np.mean(succ["naive_sd"]) - greedy_rate),
                             "oracle_limited": bool(oracle_rate < DELTA_THRESHOLD + greedy_rate)}
                    if not include_self and tol_kind == "relative":
                        holds = bool(delta >= DELTA_THRESHOLD and lo > 0)
                        entry["holds_at_k"] = holds
                        holds_at[k] = holds
                        if k == 5:
                            delta5.append(delta)
                    res_t[str(k)] = entry
                unit_s["results"][key_self][tol_kind] = res_t
        unit_holds = bool(all(holds_at.get(k, False) for k in DECISION_KS))
        unit_s["unit_holds_relative_K3_and_K5"] = unit_holds
        holds_all.append(unit_holds)
        bias_b.append(b)
        bias_s.append(s_sign)
        summary["units"][code] = unit_s
    rho_b = spearmanr(delta5, bias_b).statistic
    rho_s = spearmanr(delta5, bias_s).statistic
    any_holds = bool(any(holds_all))
    summary["cross_unit"] = {
        "delta_K5_relative_by_unit": dict(zip(units.keys(), delta5)),
        "b_by_unit": dict(zip(units.keys(), bias_b)), "s_sign_by_unit": dict(zip(units.keys(), bias_s)),
        "spearman_delta5_vs_b": None if not math.isfinite(rho_b) else float(rho_b),
        "spearman_delta5_vs_s_sign": None if not math.isfinite(rho_s) else float(rho_s),
        "n_units_holding": int(sum(holds_all)),
        "sentence_recovers_usability_under_bias_dominated_shift_allowed": bool(any_holds and math.isfinite(rho_b) and rho_b > 0 and math.isfinite(rho_s) and rho_s > 0),
        "wording_if_none_holds": None if any_holds else "no unit meets the pre-registered criterion",
    }
    summary["development_medians_um"] = medians
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing non-empty output directory {out}")
    out.mkdir(parents=True, exist_ok=True)
    units, medians = build_units(args.work)
    for code, u in units.items():
        u["code"] = code
    rows = []
    for u in units.values():
        rows.extend(select_all(u, include_self=False))
        rows.extend(select_all(u, include_self=True))
    selections = pd.DataFrame(rows)
    selections.to_csv(out / "portfolio_selections.csv", index=False, float_format="%.10g", lineterminator="\n")  # written before outcomes are read
    scored = score(units, selections)
    scored.to_csv(out / "portfolio_scored.csv", index=False, float_format="%.10g", lineterminator="\n")
    summary = {"schema_version": "h-portfolio-v1",
               "status": "retrospective; post hoc; labels already open; protocol H_F_PORTFOLIO_PROTOCOL_v1.md frozen before the run",
               "constants": {"nominal": NOMINAL, "K": list(K_VALUES), "tau_rel": TAU_REL, "B": B_BOOT, "boot_seed": BOOT_SEED, "min_clusters": MIN_CLUSTERS,
                             "delta_threshold": DELTA_THRESHOLD, "decision_K": list(DECISION_KS), "ad_k": AD_K, "ad_quantile": AD_QUANTILE},
               **summarise(units, scored, medians)}
    write_json(out / "portfolio_summary.json", summary)
    inputs = ["w3_artifacts_canonical_run2/point_predictions.csv", "w3_artifacts_canonical_run2/ad_scores.csv", "h_A1_run1/point_predictions.csv",
              "h_A1_reveal/sealed_scored_rows.csv",
              "h_A1_artifacts/sealed_features.csv", "h_A1_artifacts/development_labeled.csv", "h_A1_artifacts/h_A1_devdiag.json"]
    write_json(out / "environment.json", {
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__, "platform": platform.platform(), "script_sha256": sha256_file(Path(__file__).resolve()),
        "input_sha256": {name: sha256_file(args.work / name) for name in inputs},
    })
    manifest = {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.name != "artifact_hashes.json"}
    write_json(out / "artifact_hashes.json", manifest)
    for code, s in summary["units"].items():
        rel = s["results"]["self_excluded"]["relative"]
        print(code, "q=%.4f s=%.4f" % (s["q_log"], s["s_log"]), {k: (round(v["success_at_k"]["greedy"], 3), round(v["success_at_k"]["portfolio_q"], 3), round(v["oracle_upper_bound"], 3)) for k, v in rel.items()}, "holds:", s["unit_holds_relative_K3_and_K5"])
    print("cross-unit:", summary["cross_unit"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
