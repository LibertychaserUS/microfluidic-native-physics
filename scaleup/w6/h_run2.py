#!/usr/bin/env python3
"""H_PROTOCOL_v2 development-side runner (binding-driven; same cells, models, intervals and
development-side quantities as the frozen v1 runner h_run.py, with column names taken from the
binding / custodian manifest instead of the A1 names).

Cells:
  S0         row-random 80/20 inside the development set, seeds 0-14
  S1         GroupShuffleSplit 80/20 by s1_group, seeds 0-14 (only when the binding defines S1)
  S2_dev     leave-one-s2_group-out over the development levels (labels open; retrospective)
  sealed_S2 / sealed_S3 / sealed_S4   train on all development rows, predict the sealed rows (inputs only)

Models, intervals and the dev-ref are those of h_run.py (untuned W3/W4 configurations).
Feature sets: native (manifest["native_columns"]) and, when present, dimensionless
(manifest["dimensionless_columns"]).  H-C quantities are produced only with a dimensionless set.

Usage: python h_run2.py --binding <json> --protocol <H_PROTOCOL_v2.md> --artifacts <custodian dir> --output <run dir>
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import xgboost  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, ShuffleSplit  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from w3.uq_models import DeepEnsemble, GaussianProcessModel, QuantileForest  # noqa: E402

warnings.simplefilter("ignore", ConvergenceWarning)

NOMINAL = 0.90
RANDOM_SEEDS = tuple(range(15))
CAL_SEED = 42
XGB_SEED = 42
POINT_MODELS = ("xgboost", "random_forest", "mlp_log", "mlp_raw")
UQ_MODELS = ("qrf", "gp", "deep_ensemble")
ALL_MODELS = POINT_MODELS + UQ_MODELS
TARGET = "target"


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


def finite_sample_quantile(scores: np.ndarray, coverage: float) -> float:
    n = len(scores)
    rank = math.ceil((n + 1) * coverage)
    if rank > n:
        return float("inf")
    return float(np.sort(scores)[rank - 1])


# ----------------------------------------------------------------------------- models (identical to h_run.py)
def make_point(name: str, seed: int):
    if name == "xgboost":
        return XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.3, objective="reg:squarederror", random_state=seed, n_jobs=1)
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=seed, n_jobs=1)
    if name in ("mlp_log", "mlp_raw"):
        return Pipeline([("scale", StandardScaler()),
                         ("mlp", MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu", solver="adam", learning_rate_init=1e-3,
                                              max_iter=2000, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=seed))])
    raise ValueError(name)


class Fitted:
    def __init__(self, name: str, x_fit: np.ndarray, y_fit: np.ndarray, seed: int, log_target: bool):
        self.name = name
        # H-D13: the bounded model (mlp_log) is always log-target and the unbounded contrast (mlp_raw) always raw-scale,
        # whatever the dataset's devdiag scale; every other model follows the devdiag scale (protocol section 4)
        self.log = True if name == "mlp_log" else (False if name == "mlp_raw" else log_target)
        y = np.log(y_fit) if self.log else y_fit
        self.uq = None
        if name in POINT_MODELS:
            if name in ("mlp_log", "mlp_raw"):
                self.scaler = StandardScaler().fit(y.reshape(-1, 1))
                yt = self.scaler.transform(y.reshape(-1, 1)).ravel()
            else:
                self.scaler = None
                yt = y
            self.est = make_point(name, seed).fit(x_fit, yt)
        elif name == "qrf":
            self.uq = QuantileForest(seed).fit(x_fit, y)
        elif name == "gp":
            self.uq = GaussianProcessModel(seed, x_fit.shape[1]).fit(x_fit, y)
        elif name == "deep_ensemble":
            self.uq = DeepEnsemble(seed, standardise_target=True).fit(x_fit, y)
        else:
            raise ValueError(name)

    def _back(self, r: np.ndarray) -> np.ndarray:
        return np.exp(r) if self.log else r

    def response(self, x: np.ndarray) -> np.ndarray:
        if self.uq is not None:
            return np.asarray(self.uq.predict(x).point, dtype=float)
        p = np.asarray(self.est.predict(x), dtype=float).reshape(-1)
        if self.scaler is not None:
            p = self.scaler.inverse_transform(p.reshape(-1, 1)).ravel()
        return p

    def native(self, x: np.ndarray):
        out = self.uq.predict(x)
        return np.asarray(out.lower, float), np.asarray(out.upper, float), np.asarray(out.score, float)


# ----------------------------------------------------------------------------- one cell (identical logic to h_run.py)
def run_cell(table: pd.DataFrame, feature_cols: list[str], feature_set: str, train_idx: np.ndarray, test_idx: np.ndarray, cal_groups: np.ndarray | None,
             seed: int, log_target: bool, models: tuple[str, ...], tau_abs: float, cell: str, repeat, group_col: str,
             point_rows: list, interval_rows: list, sealed: bool = False) -> dict:
    x_all = table[feature_cols].to_numpy(float)
    y_all = table[TARGET].to_numpy(float) if not sealed else None
    # H-D12: the grouped 20% calibration partition needs >= 2 groups among the training rows; otherwise fall back to a row-random partition
    if cal_groups is None or len(set(cal_groups[train_idx])) < 2:
        fit_rel, cal_rel = next(ShuffleSplit(n_splits=1, test_size=0.2, random_state=CAL_SEED).split(train_idx))
        cal_rule = "row" if cal_groups is None else "row (single-group fallback, H-D12)"
    else:
        fit_rel, cal_rel = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=CAL_SEED).split(train_idx, groups=cal_groups[train_idx]))
        cal_rule = "group"
    fit_idx, cal_idx = train_idx[fit_rel], train_idx[cal_rel]
    yfit = table[TARGET].to_numpy(float)[fit_idx]
    ycal = table[TARGET].to_numpy(float)[cal_idx]
    summary = {"cell": cell, "repeat": repeat, "feature_set": feature_set, "n_fit": int(len(fit_idx)), "n_cal": int(len(cal_idx)), "n_test": int(len(test_idx)),
               "calibration_rule_used": cal_rule, "models": {}}
    ylog_fit = np.log(yfit) if log_target else yfit
    q_lo = GradientBoostingRegressor(loss="quantile", alpha=0.05, n_estimators=300, max_depth=3, random_state=seed).fit(x_all[fit_idx], ylog_fit)
    q_hi = GradientBoostingRegressor(loss="quantile", alpha=0.95, n_estimators=300, max_depth=3, random_state=seed).fit(x_all[fit_idx], ylog_fit)
    ycal_r = np.log(ycal) if log_target else ycal
    cqr_scores = np.maximum(q_lo.predict(x_all[cal_idx]) - ycal_r, ycal_r - q_hi.predict(x_all[cal_idx]))
    cqr_q = finite_sample_quantile(cqr_scores, NOMINAL)
    back = (lambda r: np.exp(r)) if log_target else (lambda r: r)
    for name in models:
        m = Fitted(name, x_all[fit_idx], yfit, seed, log_target)
        pred_r = m.response(x_all[test_idx])
        pred_um = m._back(pred_r)
        cal_pred_r = m.response(x_all[cal_idx])
        ycal_m = np.log(ycal) if m.log else ycal
        radius = finite_sample_quantile(np.abs(ycal_m - cal_pred_r), NOMINAL)
        lo_um, hi_um = m._back(pred_r - radius), m._back(pred_r + radius)
        entry = {"n_test": int(len(test_idx))}
        if not sealed:
            y = y_all[test_idx]
            entry.update({"mape": mape(y, pred_um), "mdape": mdape(y, pred_um), "n_nonpositive_pred": int((pred_um <= 0).sum()),
                          "split_coverage": float(np.mean((y >= lo_um) & (y <= hi_um))), "split_mean_width_um": float(np.mean(hi_um - lo_um)) if np.isfinite(radius) else None,
                          "within_tau_abs_contained": float(np.mean((np.abs(pred_um - y) <= tau_abs) & (lo_um >= y - tau_abs) & (hi_um <= y + tau_abs))),
                          "width_below_2tau_abs": float(np.mean((hi_um - lo_um) < 2 * tau_abs)),
                          "within_tau_rel_contained": float(np.mean((np.abs(pred_um - y) <= 0.10 * y) & (lo_um >= 0.9 * y) & (hi_um <= 1.1 * y))),
                          "width_below_2tau_rel": float(np.mean((hi_um - lo_um) < 0.2 * y))})
        for k, ridx in enumerate(test_idx):
            point_rows.append({"cell": cell, "repeat": repeat, "model": name, "feature_set": feature_set,
                               "row_id": table["row_id"].iat[ridx], "group": table[group_col].iat[ridx],
                               "actual_um": None if sealed else float(y_all[ridx]), "pred_um": float(pred_um[k])})
            interval_rows.append({"cell": cell, "repeat": repeat, "model": name, "method": "split_conformal", "row_id": table["row_id"].iat[ridx],
                                  "lower_um": float(lo_um[k]), "upper_um": float(hi_um[k]), "native_score": None})
        if name == "xgboost":
            lo_c, hi_c = back(q_lo.predict(x_all[test_idx]) - cqr_q), back(q_hi.predict(x_all[test_idx]) + cqr_q)
            if not sealed:
                y = y_all[test_idx]
                entry["cqr_coverage"] = float(np.mean((y >= lo_c) & (y <= hi_c)))
                entry["cqr_mean_width_um"] = float(np.mean(hi_c - lo_c)) if np.isfinite(cqr_q) else None
            for k, ridx in enumerate(test_idx):
                interval_rows.append({"cell": cell, "repeat": repeat, "model": name, "method": "cqr", "row_id": table["row_id"].iat[ridx],
                                      "lower_um": float(lo_c[k]), "upper_um": float(hi_c[k]), "native_score": None})
        if m.uq is not None:
            nlo, nhi, nsc = m.native(x_all[test_idx])
            clo, chi, _ = m.native(x_all[cal_idx])
            cq = finite_sample_quantile(np.maximum(clo - ycal_m, ycal_m - chi), NOMINAL)
            nlo_um, nhi_um = m._back(nlo), m._back(nhi)
            klo_um, khi_um = m._back(nlo - cq), m._back(nhi + cq)
            if not sealed:
                y = y_all[test_idx]
                entry["native_coverage"] = float(np.mean((y >= nlo_um) & (y <= nhi_um)))
                entry["native_mean_width_um"] = float(np.mean(nhi_um - nlo_um))
                entry["conformalised_native_coverage"] = float(np.mean((y >= klo_um) & (y <= khi_um)))
                entry["spearman_abs_error_vs_native_width"] = float(stats.spearmanr(np.abs(pred_um - y), nhi_um - nlo_um).correlation) if len(y) > 3 else None
            for k, ridx in enumerate(test_idx):
                interval_rows.append({"cell": cell, "repeat": repeat, "model": name, "method": "native", "row_id": table["row_id"].iat[ridx],
                                      "lower_um": float(nlo_um[k]), "upper_um": float(nhi_um[k]), "native_score": float(nsc[k])})
                interval_rows.append({"cell": cell, "repeat": repeat, "model": name, "method": "conformalised_native", "row_id": table["row_id"].iat[ridx],
                                      "lower_um": float(klo_um[k]), "upper_um": float(khi_um[k]), "native_score": None})
        summary["models"][name] = entry
    return summary


def group_kfold_ref(x: np.ndarray, y: np.ndarray, groups: np.ndarray, log_target: bool) -> tuple[list[dict], float]:
    folds = []
    n_splits = min(5, len(set(groups)))  # H-D11: fewer than five dependency groups -> leave-one-group-out
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(x, y, groups)):
        est = make_point("xgboost", XGB_SEED).fit(x[tr], np.log(y[tr]) if log_target else y[tr])
        p = est.predict(x[te]); p = np.exp(p) if log_target else p
        folds.append({"fold": k, "n_held": int(len(te)), "n_held_groups": int(len(set(groups[te]))), "mape": mape(y[te], p), "mdape": mdape(y[te], p)})
    return folds, float(np.mean([f["mape"] for f in folds]))


# ----------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binding", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    binding = json.loads(args.binding.read_text())
    if sha256_file(args.protocol) != binding["protocol_sha256"]:
        raise ValueError("protocol digest mismatch")
    manifest = json.loads((args.artifacts / "custodian_manifest.json").read_text())
    if manifest["binding"]["sha256"] != sha256_file(args.binding):
        raise ValueError("binding digest mismatch against custodian manifest")
    for key in ("development_labeled", "sealed_features", "devdiag"):
        entry = manifest["artifacts"][key]
        if sha256_file(args.artifacts / entry["file"]) != entry["sha256"]:
            raise ValueError(f"{entry['file']}: digest mismatch")
    out = args.output
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing non-empty output directory {out}")
    out.mkdir(parents=True, exist_ok=True)
    devdiag = json.loads((args.artifacts / manifest["artifacts"]["devdiag"]["file"]).read_text())
    log_target = devdiag["target_scale"] == "log"
    tau_abs = float(devdiag["tau_absolute"])
    dev = pd.read_csv(args.artifacts / "development_labeled.csv", keep_default_na=False, na_values=[])
    sealed_x = pd.read_csv(args.artifacts / "sealed_features.csv", keep_default_na=False, na_values=[])
    native = list(manifest["native_columns"])
    dimless = list(manifest.get("dimensionless_columns") or [])
    has_s1 = binding["groups"]["S1"] is not None
    c2_scale = binding.get("c2_scale_column")
    for frame in (dev, sealed_x):
        for c in native + dimless:
            frame[c] = frame[c].astype(float)
    dev[TARGET] = dev[TARGET].astype(float)
    point_rows: list[dict] = []
    interval_rows: list[dict] = []
    sealed_tiers = sorted(set(sealed_x["sealed_tier"].astype(str)))
    summary = {"schema_version": "h-run-v2", "dataset_code": binding["dataset_code"], "log_target": log_target, "tau_abs": tau_abs,
               "n_development": int(len(dev)), "n_sealed": int(len(sealed_x)), "sealed_tiers": sealed_tiers, "has_s1": has_s1,
               "n_native_features": len(native), "n_dimensionless_features": len(dimless), "cells": []}
    s2 = dev["s2_group"].astype(str).to_numpy()
    s1 = dev["s1_group"].astype(str).to_numpy() if has_s1 else s2
    x_dev = dev[native].to_numpy(float)
    y_dev = dev[TARGET].to_numpy(float)

    # dev-ref: 5-fold GroupKFold by the finest dependency group (S1 if defined, else S2)
    folds, dev_ref = group_kfold_ref(x_dev, y_dev, s1, log_target)
    summary["dev_ref"] = {"folds": folds, "mape_mean": dev_ref, "grouping": "s1_group" if has_s1 else "s2_group", "n_splits": len(folds), "n_groups": int(len(set(s1))),
                          "model": "xgboost native features, log target" if log_target else "xgboost native features"}
    if dimless:
        folds_d, ref_d = group_kfold_ref(dev[dimless].to_numpy(float), y_dev, s1, log_target)
        summary["dev_ref_dimensionless"] = {"folds": folds_d, "mape_mean": ref_d}
    else:
        summary["dev_ref_dimensionless"] = None

    idx = np.arange(len(dev))
    for seed in RANDOM_SEEDS:  # S0
        tr, te = next(ShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(idx))
        summary["cells"].append(run_cell(dev, native, "native", idx[tr], idx[te], None, seed, log_target, ALL_MODELS, tau_abs, "S0", seed, "s2_group", point_rows, interval_rows))
    if has_s1:  # S1 group-random
        for seed in RANDOM_SEEDS:
            tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(idx, groups=s1))
            summary["cells"].append(run_cell(dev, native, "native", idx[tr], idx[te], s1, seed, log_target, ALL_MODELS, tau_abs, "S1", seed, "s1_group", point_rows, interval_rows))
    for lev in sorted(set(s2)):  # S2_dev leave-one-level-out
        te = idx[s2 == lev]; tr = idx[s2 != lev]
        summary["cells"].append(run_cell(dev, native, "native", tr, te, s2, XGB_SEED, log_target, ALL_MODELS, tau_abs, "S2_dev", lev, "s2_group", point_rows, interval_rows))
        if dimless:
            summary["cells"].append(run_cell(dev, dimless, "dimensionless", tr, te, s2, XGB_SEED, log_target, ("xgboost",), tau_abs, "S2_dev", lev, "s2_group", point_rows, interval_rows))
    # sealed bundle
    both = pd.concat([dev, sealed_x.assign(**{TARGET: np.nan})], ignore_index=True)
    tr = np.arange(len(dev)); te = np.arange(len(dev), len(both))
    both_s2 = both["s2_group"].astype(str).to_numpy()
    for tier in sealed_tiers:
        te_t = te[(both["sealed_tier"].astype(str).to_numpy()[te] == tier)]
        summary["cells"].append(run_cell(both, native, "native", tr, te_t, both_s2, XGB_SEED, log_target, ALL_MODELS, tau_abs, f"sealed_{tier}", tier, "s2_group", point_rows, interval_rows, sealed=True))
        if dimless:
            summary["cells"].append(run_cell(both, dimless, "dimensionless", tr, te_t, both_s2, XGB_SEED, log_target, ("xgboost",), tau_abs, f"sealed_{tier}", tier, "s2_group", point_rows, interval_rows, sealed=True))

    # ---- development-side hypothesis quantities
    pts = pd.DataFrame(point_rows)
    ints = pd.DataFrame(interval_rows)
    dev_pts = pts[pts["cell"].isin(["S0", "S1", "S2_dev"])].copy()
    dev_pts["ape"] = np.abs(dev_pts["pred_um"] - dev_pts["actual_um"]) / dev_pts["actual_um"]
    s2n = dev_pts[(dev_pts.cell == "S2_dev") & (dev_pts.model == "xgboost") & (dev_pts.feature_set == "native")]
    lev_mape = s2n.groupby("repeat")["ape"].mean().mul(100.0)
    summary["H_A_dev"] = {"S2_dev_macro_mape_xgb_native": float(lev_mape.mean()), "per_level": lev_mape.round(4).to_dict(),
                          "ratio_to_dev_ref": float(lev_mape.mean() / dev_ref), "note": "development leave-one-level-out, labels open, retrospective"}
    if has_s1:
        s1c = [c for c in summary["cells"] if c["cell"] == "S1"]
        r_s1 = [c["models"]["xgboost"]["mape"] / dev_ref for c in s1c]
        summary["H_A_dev"]["S1_r_per_seed"] = r_s1
        summary["H_A_dev"]["S1_dispersion_max_over_min"] = float(max(r_s1) / min(r_s1)) if min(r_s1) > 0 else None
    else:
        summary["H_A_dev"]["S1_dispersion_max_over_min"] = None
    if dimless:
        s2d = dev_pts[(dev_pts.cell == "S2_dev") & (dev_pts.model == "xgboost") & (dev_pts.feature_set == "dimensionless")]
        lev_mape_d = s2d.groupby("repeat")["ape"].mean().mul(100.0)
        paired = pd.DataFrame({"native": lev_mape, "dimensionless": lev_mape_d}).dropna()
        diff = (paired["native"] - paired["dimensionless"]).to_numpy()
        wil = stats.wilcoxon(diff) if len(diff) >= 6 and np.any(diff != 0) else None
        summary["H_C_dev"] = {"per_level": paired.round(4).to_dict(orient="index"), "median_diff_native_minus_dimensionless_pp": float(np.median(diff)),
                              "wilcoxon_p": float(wil.pvalue) if wil else None, "n_pairs": int(len(diff))}
        dl = both[dimless].to_numpy(float)
        sc = both[c2_scale].to_numpy(float) if c2_scale else None
        pairs = []
        for i in range(len(both)):
            for j in range(i + 1, len(both)):
                if np.all(np.abs(dl[i] - dl[j]) <= 0.05 * np.maximum(np.maximum(np.abs(dl[i]), np.abs(dl[j])), 1e-12)):
                    k = (sc[j] / sc[i]) if (sc is not None and sc[i]) else np.nan
                    pairs.append({"i": int(i), "j": int(j), "scale_factor": float(k), "same_scale": bool(abs(k - 1) < 0.05) if np.isfinite(k) else True})
        n_scaled = sum(1 for p in pairs if not p["same_scale"])
        summary["H_C_C2_pairs"] = {"scale_column": c2_scale, "n_pairs_equal_dimensionless_groups": len(pairs), "n_pairs_with_different_absolute_scale": n_scaled,
                                   "evaluable_rule": "requires >= 20 pairs with different absolute scale", "evaluable": bool(n_scaled >= 20)}
    else:
        summary["H_C_dev"] = None
        summary["H_C_C2_pairs"] = {"evaluable": False, "reason": "no dimensionless inputs on this dataset"}
    hd = []
    for c in summary["cells"]:
        if c["cell"] == "S2_dev" and c["feature_set"] == "native" and "within_tau_abs_contained" in c["models"]["xgboost"]:
            e = c["models"]["xgboost"]
            hd.append({"level": c["repeat"], "A_abs": e["within_tau_abs_contained"], "W_abs": e["width_below_2tau_abs"], "A_rel": e["within_tau_rel_contained"], "W_rel": e["width_below_2tau_rel"], "mape": e["mape"]})
    summary["H_D_dev"] = {"per_level": hd, "max_abs_A_minus_W": float(max(h["A_abs"] - h["W_abs"] for h in hd)) if hd else None}
    six = ["xgboost", "random_forest", "mlp_log", "qrf", "gp", "deep_ensemble"]
    s0 = dev_pts[(dev_pts.cell == "S0") & (dev_pts.feature_set == "native") & dev_pts.model.isin(six)].groupby("model")["ape"].mean().mul(100.0)
    s2m = dev_pts[(dev_pts.cell == "S2_dev") & (dev_pts.feature_set == "native") & dev_pts.model.isin(six)].groupby(["model", "repeat"])["ape"].mean().groupby("model").mean().mul(100.0)
    rho = stats.spearmanr(s0.rank().reindex(six), s2m.rank().reindex(six)).correlation
    summary["H_E_dev"] = {"S0_mape": s0.round(3).to_dict(), "S2_dev_macro_mape": s2m.round(3).to_dict(), "spearman_rank_S0_vs_S2dev": float(rho)}
    nw = ints[ints.method == "native"].copy(); nw["w"] = nw["upper_um"] - nw["lower_um"]
    ratio = {}
    for mname in UQ_MODELS:
        a = nw[(nw.model == mname) & (nw.cell == "S0")]["w"].mean(); b = nw[(nw.model == mname) & (nw.cell == "S2_dev")]["w"].mean()
        ratio[mname] = {"S0_mean_native_width_um": float(a), "S2_dev_mean_native_width_um": float(b), "ratio": float(b / a) if a else None}
    summary["H_E_dev"]["native_width_ratio"] = ratio

    pts.to_csv(out / "point_predictions.csv", index=False, float_format="%.10g", lineterminator="\n")
    ints.to_csv(out / "interval_rows.csv", index=False, float_format="%.10g", lineterminator="\n")
    bundle = pts[pts.cell.str.startswith("sealed")].drop(columns=["actual_um"]).merge(
        ints[ints.cell.str.startswith("sealed")].pivot_table(index=["cell", "model", "row_id"], columns="method", values=["lower_um", "upper_um"], aggfunc="first").pipe(
            lambda d: d.set_axis([f"{a}__{b}" for a, b in d.columns], axis=1)).reset_index(), on=["cell", "model", "row_id"], how="left")
    bundle.to_csv(out / "sealed_prediction_bundle.csv", index=False, float_format="%.10g", lineterminator="\n")
    (out / "sealed_prediction_bundle.sha256").write_text(sha256_file(out / "sealed_prediction_bundle.csv") + "\n")
    write_json(out / "h_run_summary.json", summary)
    write_json(out / "environment.json", {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
                                          "xgboost": xgboost.__version__, "platform": platform.platform(), "script_sha256": sha256_file(Path(__file__).resolve()),
                                          "inputs": {k: v["sha256"] for k, v in manifest["artifacts"].items() if k != "sealed_labels"}})
    hashes = {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.name != "artifact_hashes.json"}
    write_json(out / "artifact_hashes.json", hashes)
    print("dev_ref", round(dev_ref, 3), "| S2_dev macro", round(summary["H_A_dev"]["S2_dev_macro_mape_xgb_native"], 3), "| ratio", round(summary["H_A_dev"]["ratio_to_dev_ref"], 3),
          "| S1 disp", summary["H_A_dev"]["S1_dispersion_max_over_min"])
    print("H-E rho", round(summary["H_E_dev"]["spearman_rank_S0_vs_S2dev"], 3), "| C2", summary["H_C_C2_pairs"], "| bundle", hashes["sealed_prediction_bundle.csv"][:16])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
