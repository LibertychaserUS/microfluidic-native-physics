"""Post-hoc, non-pre-registered analyses answering reviewer-style concerns (round 2).

Everything here is retrospective: every label it touches has already been revealed
(D2 TX100 by the V1 reveal, D3 laboratory B by the D3 reveal) or was never sealed
(D1). Nothing is written into a frozen directory; outputs go to a new directory and
are hashed. The analyses are:

  1. V1 development reference: leave-one-chemistry-out MAPE of the locked V1 model on
     the V1 development table, as a retrospective counterpart of the D3-style
     "development reference x 1.25" criterion.
  2. D3 alternative point metrics (sMAPE, median APE, log-ratio error, size-quartile
     bins) from the already-revealed scored rows.
  3. D3 log-target baseline: the locked XGBoost configuration fitted to log(size) on the
     246 development rows, predicting the 57 laboratory-B rows, with the same grouped
     5-fold development reference as the frozen raw-scale model.
  4. W3 metric robustness: MAPE, median APE, sMAPE and log-ratio error for every cell
     (XGBoost), ratios to the same-dataset reference cell under each metric, row-bootstrap
     intervals for the single-repeat cells, and size-quartile bins plus train/test size
     distributions for the two device-type cells.

Usage:
  python -m w3.posthoc_round2 --work <work/microfluidic_reliability> --output-dir <dir>
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
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import xgboost  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

SEED = 42
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0

V1_FEATURES = [
    "continuous_flow_ml_min",
    "dispersed_flow_ml_min",
    "equilibrium_interfacial_tension_mn_m",
    "concentration_over_cmc",
]
V1_TARGET = "target_diameter_mm"
V1_DEVELOPMENT_GROUPS = (0.0, 2.0, 4.33, 11.0, 20.0)  # CMC (mM); TX100 = 3.5 was sealed

D3_NUMERIC = ["ESM", "HSPC", "CHOL", "PEG", "TFR", "FRR"]
D3_AQUEOUS_LEVELS = ("MQ", "PBS")
D3_MULTIPLIERS = {"reproduction_success": 1.25, "strong_replication": 1.00}

W3_REFERENCE = {"D1": "D1_random", "D2": "D2_random"}
W3_SINGLE_REPEAT = ("D1_external", "D1_se_to_de", "D1_de_to_se")
W3_DEVICE_TYPE = {"D1_se_to_de": ("SE", "DE"), "D1_de_to_se": ("DE", "SE")}


# ----------------------------------------------------------------------------- utilities

def sanitize(value):
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return [sanitize(v) for v in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if math.isnan(f):
            return None
        if math.isinf(f):
            raise ValueError("infinite value cannot enter a JSON artifact")
        return f
    return value


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(sanitize(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locked_regressor(seed: int = SEED) -> XGBRegressor:
    return XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.3,
                        objective="reg:squarederror", random_state=seed, n_jobs=1)


def point_metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    """MAPE, median APE, sMAPE (all in percent), log-ratio error on positive predictions."""
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ape = np.abs(pred - actual) / np.abs(actual)
    smape = 2.0 * np.abs(pred - actual) / (np.abs(pred) + np.abs(actual))
    positive = pred > 0
    log_ratio = np.abs(np.log(pred[positive] / actual[positive])) if positive.any() else np.array([])
    return {
        "n": int(len(actual)),
        "mape_percent": float(100.0 * ape.mean()),
        "mdape_percent": float(100.0 * np.median(ape)),
        "smape_percent": float(100.0 * smape.mean()),
        "n_positive_predictions": int(positive.sum()),
        "n_negative_or_zero_predictions": int((~positive).sum()),
        "mean_abs_log_ratio_positive_only": float(log_ratio.mean()) if len(log_ratio) else None,
        "median_abs_log_ratio_positive_only": float(np.median(log_ratio)) if len(log_ratio) else None,
        "fraction_within_10pct_relative": float((ape <= 0.10).mean()),
    }


def row_bootstrap_mape(actual: np.ndarray, pred: np.ndarray, resamples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    ape = np.abs(np.asarray(pred) - np.asarray(actual)) / np.abs(np.asarray(actual))
    n = len(ape)
    draws = np.empty(resamples)
    for b in range(resamples):
        idx = rng.integers(0, n, n)
        draws[b] = ape[idx].mean()
    return {
        "resamples": int(resamples),
        "seed": int(seed),
        "mape_percent_ci95": [float(100.0 * np.quantile(draws, 0.025)), float(100.0 * np.quantile(draws, 0.975))],
    }


def quartile_bins(actual: np.ndarray, pred: np.ndarray) -> list[dict]:
    """Bins by quartiles of the observed size; bins chosen by rule, not by result."""
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    edges = np.quantile(actual, [0.0, 0.25, 0.5, 0.75, 1.0])
    out = []
    for i in range(4):
        lo, hi = edges[i], edges[i + 1]
        mask = (actual >= lo) & (actual <= hi) if i == 3 else (actual >= lo) & (actual < hi)
        if mask.sum() == 0:
            continue
        m = point_metrics(actual[mask], pred[mask])
        out.append({
            "bin": i + 1,
            "observed_range": [float(lo), float(hi)],
            "n": int(mask.sum()),
            "median_observed": float(np.median(actual[mask])),
            "median_predicted": float(np.median(pred[mask])),
            "mape_percent": m["mape_percent"],
            "mdape_percent": m["mdape_percent"],
            "fraction_negative_predictions": float((pred[mask] <= 0).mean()),
        })
    return out


def distribution_summary(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=float)
    q = np.quantile(v, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {"n": int(len(v)), "min": float(q[0]), "q25": float(q[1]), "median": float(q[2]),
            "q75": float(q[3]), "max": float(q[4])}


# ----------------------------------------------------------------------------- 1. V1 reference

def v1_development_reference(work: Path) -> dict:
    dev = pd.read_csv(work / "blind_artifacts" / "development_labeled.csv")
    recovered = json.loads((work / "blind_artifacts" / "blind_reveal_results_recovered.json").read_text())
    cmc = dev["cmc_mm"].to_numpy(dtype=float)
    folds = []
    pooled_ape = []
    for group in V1_DEVELOPMENT_GROUPS:
        held = np.isclose(cmc, group)
        if held.sum() == 0:
            raise ValueError(f"no development rows with CMC {group}")
        model = locked_regressor().fit(dev.loc[~held, V1_FEATURES].to_numpy(float), dev.loc[~held, V1_TARGET].to_numpy(float))
        pred = model.predict(dev.loc[held, V1_FEATURES].to_numpy(float))
        actual = dev.loc[held, V1_TARGET].to_numpy(float)
        m = point_metrics(actual, pred)
        folds.append({"held_out_cmc_mm": group, **m})
        pooled_ape.append(np.abs(pred - actual) / np.abs(actual))
    macro = float(np.mean([f["mape_percent"] for f in folds]))
    pooled = float(100.0 * np.concatenate(pooled_ape).mean())
    sealed_mape = None
    for key in ("mape_percent", "blind_mape_percent"):
        if isinstance(recovered, dict):
            # search a few plausible nests without assuming structure
            stack = [recovered]
            while stack and sealed_mape is None:
                node = stack.pop()
                if isinstance(node, dict):
                    if key in node and isinstance(node[key], (int, float)):
                        sealed_mape = float(node[key])
                        break
                    stack.extend(node.values())
    n_other = int((~np.isin(np.round(cmc, 2), np.round(V1_DEVELOPMENT_GROUPS, 2))).sum())
    out = {
        "schema_version": "posthoc-round2-v1-reference-v1",
        "status": "retrospective; NOT pre-registered; computed 2026-10-17 after both reveals",
        "estimand": "leave-one-chemistry-out MAPE of the locked V1 model over the five development "
                    "CMC groups of the V1 development table; the training side of every fold is the "
                    "rest of the development table (the TX100 group is never used)",
        "model": "XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.3, random_state=42, reg:squarederror)",
        "features": V1_FEATURES,
        "n_development_rows": int(len(dev)),
        "n_rows_outside_listed_groups_kept_in_training_only": n_other,
        "folds": folds,
        "macro_mape_percent": macro,
        "pooled_mape_percent": pooled,
        "retrospective_thresholds_mape_percent": {k: macro * v for k, v in D3_MULTIPLIERS.items()},
        "sealed_tx100_mape_percent_from_recovered_result": sealed_mape,
        "sealed_over_macro_reference_ratio": (sealed_mape / macro) if sealed_mape else None,
        "pre_registered_thresholds_for_comparison": {"reproduction_success_percent": 11.4, "strong_replication_percent": 6.0},
    }
    return out


# ----------------------------------------------------------------------------- 2+3. D3

def d3_design(frame: pd.DataFrame) -> np.ndarray:
    x = frame[D3_NUMERIC].to_numpy(dtype=float)
    levels = frame["AQUEOUS"].astype(str).str.strip().to_numpy()
    onehot = np.column_stack([(levels == level).astype(float) for level in D3_AQUEOUS_LEVELS])
    return np.hstack([x, onehot])


def d3_analyses(work: Path) -> tuple[dict, dict]:
    art = work / "d3_blind_artifacts"
    scored = pd.read_csv(art / "reveal" / "sealed_primary_scored_rows.csv")
    features = pd.read_csv(art / "sealed_primary_features.csv")
    dev = pd.read_csv(art / "development_labeled.csv")
    reference = json.loads((art / "prereveal" / "D3_DEVELOPMENT_REFERENCE.json").read_text())

    actual = scored["actual_size_nm"].to_numpy(float)
    pred_raw = scored["predicted_size_nm"].to_numpy(float)
    alt = {
        "schema_version": "posthoc-round2-d3-metrics-v1",
        "status": "retrospective; NOT pre-registered; from already-revealed scored rows",
        "frozen_model_laboratory_B": point_metrics(actual, pred_raw),
        "frozen_model_size_quartile_bins": quartile_bins(actual, pred_raw),
        "observed_size_distribution_nm": distribution_summary(actual),
        "predicted_size_distribution_nm": distribution_summary(pred_raw),
    }

    # log-target baseline: identical configuration and inputs, target log(SIZE)
    dev_ok = dev[np.isfinite(dev["SIZE"].to_numpy(float)) & (dev["SIZE"].to_numpy(float) > 0)]
    x_dev = d3_design(dev_ok)
    y_dev = dev_ok["SIZE"].to_numpy(float)
    groups = dev_ok["ID"].astype(str).to_numpy()

    def grouped_cv(target_transform, inverse):
        gkf = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        fold_rows = []
        for k, (tr, te) in enumerate(gkf.split(x_dev, y_dev, groups)):
            model = locked_regressor().fit(x_dev[tr], target_transform(y_dev[tr]))
            p = inverse(model.predict(x_dev[te]))
            fold_rows.append({"fold": k, "n_held": int(len(te)), **point_metrics(y_dev[te], p)})
        return fold_rows

    raw_folds = grouped_cv(lambda y: y, lambda p: p)
    log_folds = grouped_cv(np.log, np.exp)
    raw_ref = float(np.mean([f["mape_percent"] for f in raw_folds]))
    log_ref = float(np.mean([f["mape_percent"] for f in log_folds]))

    joined = features.merge(scored[["opaque_row_id", "actual_size_nm"]], on="opaque_row_id", how="inner")
    if len(joined) != len(scored):
        raise ValueError("sealed features and scored rows do not join one-to-one")
    x_b = d3_design(joined)
    y_b = joined["actual_size_nm"].to_numpy(float)
    log_model = locked_regressor().fit(x_dev, np.log(y_dev))
    pred_log = np.exp(log_model.predict(x_b))
    raw_model = locked_regressor().fit(x_dev, y_dev)
    pred_raw_refit = raw_model.predict(x_b)
    refit_matches_frozen = bool(np.allclose(np.sort(pred_raw_refit), np.sort(pred_raw), rtol=1e-6, atol=1e-6))

    log_metrics = point_metrics(y_b, pred_log)
    baseline = {
        "schema_version": "posthoc-round2-d3-log-target-v1",
        "status": "retrospective; NOT pre-registered; laboratory-B labels were already revealed; "
                  "the model was never part of the frozen protocol",
        "model": "XGBRegressor(100 trees, depth 3, learning rate 0.3, seed 42) on log(SIZE); "
                 "inputs identical to the frozen model (six numeric + one-hot aqueous medium)",
        "development_reference_check": {
            "frozen_reference_mape_percent": reference["reference_mape_percent"],
            "recomputed_raw_scale_reference_mape_percent": raw_ref,
            "recomputation_matches_frozen_within_0_01": bool(abs(raw_ref - reference["reference_mape_percent"]) < 0.01),
            "raw_scale_refit_predictions_match_frozen_bundle": refit_matches_frozen,
        },
        "log_target_development_reference": {
            "estimand": reference["estimand"].replace("SIZE MAPE", "SIZE MAPE of the log-target model"),
            "folds": log_folds,
            "reference_mape_percent": log_ref,
            "thresholds_mape_percent": {k: log_ref * v for k, v in D3_MULTIPLIERS.items()},
        },
        "laboratory_B": {
            "log_target_model": log_metrics,
            "frozen_raw_scale_model": point_metrics(y_b, pred_raw),
            "log_target_size_quartile_bins": quartile_bins(y_b, pred_log),
            "log_target_meets_reproduction_threshold_of_its_own_reference":
                bool(log_metrics["mape_percent"] <= log_ref * D3_MULTIPLIERS["reproduction_success"]),
            "log_target_meets_frozen_raw_scale_reproduction_threshold":
                bool(log_metrics["mape_percent"] <= reference["criteria_thresholds_mape_percent"]["reproduction_success"]),
            "predicted_size_distribution_nm_log_target": distribution_summary(pred_log),
        },
    }
    return alt, baseline


# ----------------------------------------------------------------------------- 4. W3 robustness

def w3_metric_robustness(work: Path) -> dict:
    pp = pd.read_csv(work / "w3_artifacts_canonical_run2" / "point_predictions.csv")
    pp = pp[pp["model"] == "xgboost"]
    metric_keys = ("mape_percent", "mdape_percent", "smape_percent", "mean_abs_log_ratio_positive_only")
    cells = {}
    for family, frame in pp.groupby("family"):
        per_repeat = []
        for repeat, sub in frame.groupby("repeat"):
            per_repeat.append(point_metrics(sub["actual_um"].to_numpy(float), sub["pred_um"].to_numpy(float)))
        agg = {k: float(np.mean([r[k] for r in per_repeat])) for k in metric_keys}
        entry = {"n_repeats": int(len(per_repeat)), "n_rows_per_repeat_mean": float(np.mean([r["n"] for r in per_repeat])),
                 "mean_over_repeats": agg,
                 "n_negative_predictions_total": int(sum(r["n_negative_or_zero_predictions"] for r in per_repeat))}
        if family in W3_SINGLE_REPEAT:
            sub = frame
            entry["row_bootstrap"] = row_bootstrap_mape(sub["actual_um"].to_numpy(float), sub["pred_um"].to_numpy(float),
                                                        BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
        if family in W3_DEVICE_TYPE:
            entry["size_quartile_bins_of_test_rows"] = quartile_bins(frame["actual_um"].to_numpy(float), frame["pred_um"].to_numpy(float))
        cells[family] = entry
    for family, entry in cells.items():
        ref = cells[W3_REFERENCE[family[:2]]]["mean_over_repeats"]
        entry["ratio_to_reference_cell"] = {k: (entry["mean_over_repeats"][k] / ref[k]) if ref[k] else None for k in metric_keys}
    # train/test size distributions for the device-type cells: the test rows of one direction are
    # the training rows of the other, so both device types' observed sizes are available label-side
    se_rows = pp[pp["family"] == "D1_de_to_se"]["actual_um"].to_numpy(float)
    de_rows = pp[pp["family"] == "D1_se_to_de"]["actual_um"].to_numpy(float)
    device = {
        "single_emulsion_observed_um": distribution_summary(se_rows),
        "double_emulsion_observed_um": distribution_summary(de_rows),
        "fraction_of_DE_rows_inside_SE_observed_range": float(((de_rows >= se_rows.min()) & (de_rows <= se_rows.max())).mean()),
        "fraction_of_SE_rows_inside_DE_observed_range": float(((se_rows >= de_rows.min()) & (se_rows <= de_rows.max())).mean()),
        "note": "label-side description of the confound between device type and target size; "
                "not usable as an applicability screen",
    }
    ranking = {}
    for k in metric_keys:
        ranking[k] = sorted(((fam, cells[fam]["ratio_to_reference_cell"][k]) for fam in cells
                             if fam not in W3_REFERENCE.values()), key=lambda t: -(t[1] or 0))
    return {
        "schema_version": "posthoc-round2-w3-robustness-v1",
        "status": "retrospective; NOT part of the frozen W3 protocol; computed from canonical run2 per-row predictions",
        "model": "xgboost",
        "cells": cells,
        "device_type_size_distributions": device,
        "cell_ranking_by_ratio_to_reference": ranking,
    }


# ----------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    out: Path = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    write_json(out / "v1_development_loco_reference.json", v1_development_reference(args.work))
    alt, baseline = d3_analyses(args.work)
    write_json(out / "d3_alternative_metrics.json", alt)
    write_json(out / "d3_log_target_baseline.json", baseline)
    write_json(out / "w3_metric_robustness.json", w3_metric_robustness(args.work))
    write_json(out / "environment.json", {
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "inputs": {
            "blind_artifacts/development_labeled.csv": sha256_file(args.work / "blind_artifacts" / "development_labeled.csv"),
            "d3_blind_artifacts/reveal/sealed_primary_scored_rows.csv": sha256_file(args.work / "d3_blind_artifacts" / "reveal" / "sealed_primary_scored_rows.csv"),
            "d3_blind_artifacts/sealed_primary_features.csv": sha256_file(args.work / "d3_blind_artifacts" / "sealed_primary_features.csv"),
            "d3_blind_artifacts/development_labeled.csv": sha256_file(args.work / "d3_blind_artifacts" / "development_labeled.csv"),
            "w3_artifacts_canonical_run2/point_predictions.csv": sha256_file(args.work / "w3_artifacts_canonical_run2" / "point_predictions.csv"),
        },
    })
    manifest = {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.name != "artifact_hashes.json"}
    write_json(out / "artifact_hashes.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
