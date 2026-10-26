#!/usr/bin/env python3
"""H_PROTOCOL_v1 development-side runner for an admitted dataset (A1).

Everything here uses development rows (labels open) and the sealed rows' INPUTS only.
It writes per-row predictions for the development cells, the development reference,
the H-C/H-D/H-E development-side quantities, and a sealed prediction bundle whose
SHA-256 is recorded before the reveal. Run twice in separate directories; the two
`artifact_hashes.json` must agree byte for byte.

Cells (fixed by the protocol and the binding):
  S0        row-random 80/20 inside the development set, seeds 0-14
  S2_dev    leave-one-fluid-family-out over the development families (labels open; retrospective)
  sealed    train on all development rows, predict the sealed S2 and S3 rows

Models (untuned, configurations of W3/W4): xgboost (100/3/0.3), random_forest (400, min leaf 2),
mlp_log (64-64, log target = bounded/positivity model), mlp_raw (64-64, raw target = unbounded
model), qrf, gp, deep_ensemble. Target scale for all log models follows the devdiag file.
Feature sets: native (7 columns) and dimensionless (8 columns) from the binding.

Usage: python h_run.py --binding <H_BINDING_A1_v2.json> --protocol <H_PROTOCOL_v1.md> --artifacts <custodian dir> --output <run dir>
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
BOOT_N, BOOT_SEED = 2000, 0
POINT_MODELS = ("xgboost", "random_forest", "mlp_log", "mlp_raw")
UQ_MODELS = ("qrf", "gp", "deep_ensemble")
ALL_MODELS = POINT_MODELS + UQ_MODELS
DIMLESS = ["Ca_d", "Ca_c", "q=Qd/Qc", "Re_d", "Re_c", "Alpha", "Wi", "El_d"]


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


# ----------------------------------------------------------------------------- models
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
    """A fitted model with a uniform predict-in-micrometres interface."""

    def __init__(self, name: str, x_fit: np.ndarray, y_um_fit: np.ndarray, seed: int, log_target: bool):
        self.name = name
        self.log = log_target and name != "mlp_raw"
        y = np.log(y_um_fit) if self.log else y_um_fit
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
        """Point prediction in model-response space (log or raw)."""
        if self.uq is not None:
            return np.asarray(self.uq.predict(x).point, dtype=float)
        p = np.asarray(self.est.predict(x), dtype=float).reshape(-1)
        if self.scaler is not None:
            p = self.scaler.inverse_transform(p.reshape(-1, 1)).ravel()
        return p

    def native(self, x: np.ndarray):
        """Native interval and score in response space (UQ models only)."""
        out = self.uq.predict(x)
        return np.asarray(out.lower, float), np.asarray(out.upper, float), np.asarray(out.score, float)


# ----------------------------------------------------------------------------- one cell
def run_cell(table: pd.DataFrame, feature_cols: list[str], train_idx: np.ndarray, test_idx: np.ndarray, cal_groups: np.ndarray | None,
             seed: int, log_target: bool, models: tuple[str, ...], tau_abs: float, cell: str, repeat, group_col: str,
             point_rows: list, interval_rows: list, sealed: bool = False) -> dict:
    x_all = table[feature_cols].to_numpy(float)
    y_all = table["L_um"].to_numpy(float) if not sealed else None
    # fit / calibration partition inside the training rows
    if cal_groups is None:
        splitter = ShuffleSplit(n_splits=1, test_size=0.2, random_state=CAL_SEED)
        fit_rel, cal_rel = next(splitter.split(train_idx))
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=CAL_SEED)
        fit_rel, cal_rel = next(splitter.split(train_idx, groups=cal_groups[train_idx]))
    fit_idx, cal_idx = train_idx[fit_rel], train_idx[cal_rel]
    yfit = table["L_um"].to_numpy(float)[fit_idx]
    ycal = table["L_um"].to_numpy(float)[cal_idx]
    summary = {"cell": cell, "repeat": repeat, "n_fit": int(len(fit_idx)), "n_cal": int(len(cal_idx)), "n_test": int(len(test_idx)), "models": {}}
    # CQR quantile regressors (log response) fitted once per cell
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
        # split conformal in the model's response space
        cal_pred_r = m.response(x_all[cal_idx])
        ycal_m = np.log(ycal) if m.log else ycal
        radius = finite_sample_quantile(np.abs(ycal_m - cal_pred_r), NOMINAL)
        lo_r, hi_r = pred_r - radius, pred_r + radius
        lo_um, hi_um = m._back(lo_r), m._back(hi_r)
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
            point_rows.append({"cell": cell, "repeat": repeat, "model": name, "feature_set": "native" if feature_cols[0].startswith("Qd") else "dimensionless",
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
            conf_scores = np.maximum(clo - ycal_m, ycal_m - chi)
            cq = finite_sample_quantile(conf_scores, NOMINAL)
            nlo_um, nhi_um = m._back(nlo), m._back(nhi)
            klo_um, khi_um = m._back(nlo - cq), m._back(nhi + cq)
            if not sealed:
                y = y_all[test_idx]
                entry["native_coverage"] = float(np.mean((y >= nlo_um) & (y <= nhi_um)))
                entry["native_mean_width_um"] = float(np.mean(nhi_um - nlo_um))
                entry["conformalised_native_coverage"] = float(np.mean((y >= klo_um) & (y <= khi_um)))
                rel_err = np.abs(pred_um - y) / y
                entry["spearman_abs_error_vs_native_width"] = float(stats.spearmanr(rel_err * y, nhi_um - nlo_um).correlation) if len(y) > 3 else None
            for k, ridx in enumerate(test_idx):
                interval_rows.append({"cell": cell, "repeat": repeat, "model": name, "method": "native", "row_id": table["row_id"].iat[ridx],
                                      "lower_um": float(nlo_um[k]), "upper_um": float(nhi_um[k]), "native_score": float(nsc[k])})
                interval_rows.append({"cell": cell, "repeat": repeat, "model": name, "method": "conformalised_native", "row_id": table["row_id"].iat[ridx],
                                      "lower_um": float(klo_um[k]), "upper_um": float(khi_um[k]), "native_score": None})
        summary["models"][name] = entry
    return summary


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
    tau_abs = float(devdiag["tau_absolute_um"])
    dev = pd.read_csv(args.artifacts / "development_labeled.csv")
    sealed_x = pd.read_csv(args.artifacts / "sealed_features.csv")
    native = binding["columns"]["native"]
    point_rows: list[dict] = []
    interval_rows: list[dict] = []
    summary = {"schema_version": "h-run-v1", "dataset_code": binding["dataset_code"], "log_target": log_target, "tau_abs_um": tau_abs,
               "n_development": int(len(dev)), "n_sealed": int(len(sealed_x)), "cells": []}
    groups = dev["fluid_family"].to_numpy()
    x_dev_native = dev[native].to_numpy(float)
    y_dev = dev["L_um"].to_numpy(float)

    # development reference: 5-fold GroupKFold over fluid families, XGBoost locked, native features
    gkf = GroupKFold(n_splits=5)
    folds = []
    for k, (tr, te) in enumerate(gkf.split(x_dev_native, y_dev, groups)):
        est = make_point("xgboost", XGB_SEED).fit(x_dev_native[tr], np.log(y_dev[tr]) if log_target else y_dev[tr])
        p = est.predict(x_dev_native[te]); p = np.exp(p) if log_target else p
        folds.append({"fold": k, "n_held": int(len(te)), "held_families": sorted(set(groups[te])), "mape": mape(y_dev[te], p), "mdape": mdape(y_dev[te], p)})
    dev_ref = float(np.mean([f["mape"] for f in folds]))
    summary["dev_ref"] = {"folds": folds, "mape_mean": dev_ref, "model": "xgboost native features, log target" if log_target else "xgboost native features"}
    # dimensionless-feature reference (for H-C)
    x_dev_dim = dev[DIMLESS].to_numpy(float)
    folds_d = []
    for k, (tr, te) in enumerate(gkf.split(x_dev_dim, y_dev, groups)):
        est = make_point("xgboost", XGB_SEED).fit(x_dev_dim[tr], np.log(y_dev[tr]) if log_target else y_dev[tr])
        p = est.predict(x_dev_dim[te]); p = np.exp(p) if log_target else p
        folds_d.append({"fold": k, "mape": mape(y_dev[te], p)})
    summary["dev_ref_dimensionless"] = {"folds": folds_d, "mape_mean": float(np.mean([f["mape"] for f in folds_d]))}

    # S0 cells
    idx = np.arange(len(dev))
    for seed in RANDOM_SEEDS:
        tr, te = next(ShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(idx))
        summary["cells"].append(run_cell(dev, native, idx[tr], idx[te], None, seed, log_target, ALL_MODELS, tau_abs, "S0", seed, "fluid_family", point_rows, interval_rows))
    # S2 development leave-one-fluid-family-out (both feature sets for xgboost; native for all)
    for fam in sorted(set(groups)):
        te = idx[groups == fam]; tr = idx[groups != fam]
        summary["cells"].append(run_cell(dev, native, tr, te, groups, XGB_SEED, log_target, ALL_MODELS, tau_abs, "S2_dev", fam, "fluid_family", point_rows, interval_rows))
        summary["cells"].append(run_cell(dev, DIMLESS, tr, te, groups, XGB_SEED, log_target, ("xgboost",), tau_abs, "S2_dev", fam, "fluid_family", point_rows, interval_rows))
    # sealed bundle: all development rows train; predict sealed S2 and S3 rows (inputs only)
    both = pd.concat([dev, sealed_x.assign(L_um=np.nan, H_um=np.nan)], ignore_index=True)
    tr = np.arange(len(dev)); te = np.arange(len(dev), len(both))
    for tier in ("S2", "S3"):
        te_t = te[(both["sealed_tier"].to_numpy()[te] == tier)]
        summary["cells"].append(run_cell(both, native, tr, te_t, both["fluid_family"].to_numpy(), XGB_SEED, log_target, ALL_MODELS, tau_abs, f"sealed_{tier}", tier, "fluid_family", point_rows, interval_rows, sealed=True))
        summary["cells"].append(run_cell(both, DIMLESS, tr, te_t, both["fluid_family"].to_numpy(), XGB_SEED, log_target, ("xgboost",), tau_abs, f"sealed_{tier}", tier, "fluid_family", point_rows, interval_rows, sealed=True))

    # ---- development-side hypothesis quantities
    pts = pd.DataFrame(point_rows)
    ints = pd.DataFrame(interval_rows)
    dev_pts = pts[pts["cell"].isin(["S0", "S2_dev"])].copy()
    dev_pts["ape"] = np.abs(dev_pts["pred_um"] - dev_pts["actual_um"]) / dev_pts["actual_um"]
    # H-A dev side: S2_dev macro MAPE over families (xgboost native) / dev_ref
    s2 = dev_pts[(dev_pts.cell == "S2_dev") & (dev_pts.model == "xgboost") & (dev_pts.feature_set == "native")]
    fam_mape = s2.groupby("repeat")["ape"].mean().mul(100.0)
    summary["H_A_dev"] = {"S2_dev_macro_mape_xgb_native": float(fam_mape.mean()), "per_family": fam_mape.round(4).to_dict(),
                          "ratio_to_dev_ref": float(fam_mape.mean() / dev_ref), "note": "development leave-one-family-out, labels open, retrospective"}
    # H-C C1 dev side: paired native vs dimensionless per family
    s2d = dev_pts[(dev_pts.cell == "S2_dev") & (dev_pts.model == "xgboost") & (dev_pts.feature_set == "dimensionless")]
    fam_mape_d = s2d.groupby("repeat")["ape"].mean().mul(100.0)
    paired = pd.DataFrame({"native": fam_mape, "dimensionless": fam_mape_d}).dropna()
    diff = (paired["native"] - paired["dimensionless"]).to_numpy()
    wil = stats.wilcoxon(diff) if len(diff) >= 6 and np.any(diff != 0) else None
    summary["H_C_dev"] = {"per_family": paired.round(4).to_dict(orient="index"), "median_diff_native_minus_dimensionless_pp": float(np.median(diff)),
                          "wilcoxon_p": float(wil.pvalue) if wil else None, "n_pairs": int(len(diff))}
    # H-C C2: candidate row pairs with equal dimensionless groups (5%) but different absolute scale (dev + sealed inputs)
    dl = both[DIMLESS].to_numpy(float); nat = both[["Qd [ul/min]", "Qc [ul/min]"]].to_numpy(float)
    pairs = []
    for i in range(len(both)):
        for j in range(i + 1, len(both)):
            if np.all(np.abs(dl[i] - dl[j]) <= 0.05 * np.maximum(np.maximum(np.abs(dl[i]), np.abs(dl[j])), 1e-12)):
                k = nat[j, 0] / nat[i, 0] if nat[i, 0] else np.nan
                pairs.append({"i": int(i), "j": int(j), "scale_factor_Qd": float(k), "same_scale": bool(abs(k - 1) < 0.05)})
    n_scaled = sum(1 for p in pairs if not p["same_scale"])
    summary["H_C_C2_pairs"] = {"n_pairs_equal_dimensionless_groups": len(pairs), "n_pairs_with_different_absolute_scale": n_scaled,
                               "evaluable_rule": "requires >= 20 pairs with different absolute scale", "evaluable": bool(n_scaled >= 20)}
    # H-D dev side: A vs W per S2_dev family (xgboost split conformal)
    hd = []
    for c in summary["cells"]:
        if c["cell"] == "S2_dev" and "xgboost" in c["models"] and "within_tau_abs_contained" in c["models"]["xgboost"]:
            e = c["models"]["xgboost"]
            hd.append({"family": c["repeat"], "A_abs": e["within_tau_abs_contained"], "W_abs": e["width_below_2tau_abs"], "A_rel": e["within_tau_rel_contained"], "W_rel": e["width_below_2tau_rel"], "mape": e["mape"]})
    summary["H_D_dev"] = {"per_family": hd, "max_abs_A_minus_W": float(max(h["A_abs"] - h["W_abs"] for h in hd)) if hd else None}
    # H-E dev side: rankings S0 vs S2_dev (six families; mlp_raw excluded as the unbounded contrast)
    six = ["xgboost", "random_forest", "mlp_log", "qrf", "gp", "deep_ensemble"]
    s0 = dev_pts[(dev_pts.cell == "S0") & (dev_pts.feature_set == "native") & dev_pts.model.isin(six)].groupby("model")["ape"].mean().mul(100.0)
    s2m = dev_pts[(dev_pts.cell == "S2_dev") & (dev_pts.feature_set == "native") & dev_pts.model.isin(six)].groupby(["model", "repeat"])["ape"].mean().groupby("model").mean().mul(100.0)
    rank0, rank2 = s0.rank(), s2m.rank()
    rho = stats.spearmanr(rank0.reindex(six), rank2.reindex(six)).correlation
    summary["H_E_dev"] = {"S0_mape": s0.round(3).to_dict(), "S2_dev_macro_mape": s2m.round(3).to_dict(), "spearman_rank_S0_vs_S2dev": float(rho)}
    # native width ratio (S2_dev vs S0) per UQ model
    nw = ints[ints.method == "native"].copy(); nw["w"] = nw["upper_um"] - nw["lower_um"]
    ratio = {}
    for mname in UQ_MODELS:
        a = nw[(nw.model == mname) & (nw.cell == "S0")]["w"].mean(); b = nw[(nw.model == mname) & (nw.cell == "S2_dev")]["w"].mean()
        ratio[mname] = {"S0_mean_native_width_um": float(a), "S2_dev_mean_native_width_um": float(b), "ratio": float(b / a) if a else None}
    summary["H_E_dev"]["native_width_ratio"] = ratio

    # ---- write outputs
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
    print("dev_ref", round(dev_ref, 3), "| S2_dev macro", round(summary["H_A_dev"]["S2_dev_macro_mape_xgb_native"], 3), "| ratio", round(summary["H_A_dev"]["ratio_to_dev_ref"], 3))
    print("H-C dev median diff pp", round(summary["H_C_dev"]["median_diff_native_minus_dimensionless_pp"], 3), "p", summary["H_C_dev"]["wilcoxon_p"], "| C2 pairs", summary["H_C_C2_pairs"])
    print("H-E rho", round(summary["H_E_dev"]["spearman_rank_S0_vs_S2dev"], 3), "| bundle", hashes["sealed_prediction_bundle.csv"][:16])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
