#!/usr/bin/env python3
"""Post-hoc round 3: small-sample target-domain recalibration (frozen protocol
POSTHOC_ROUND3_RECALIBRATION_PROTOCOL_v1.md; labels already revealed; retrospective).

Two target domains: T1 = D2 TX100 group (87 rows, frozen V1 point predictions read
from the pre-reveal bundle), T2 = D3 laboratory B (57 rows, log-target XGBoost refitted
exactly as in posthoc_round2 and asserted to reproduce 43.05% MAPE).

Factors (frozen): k in {1,2,3,5,8,10,15,20}; sampling = random (200 repeats, seeds 0-199),
k-means space filling (random_state 0), farthest-point; calibration = 1-parameter
log-scale shift, 2-parameter log-affine (k>=2), scratch XGBoost on the k rows (k>=3),
uncalibrated baseline; evaluation on the remaining N-k rows (MAPE, MdAPE, signed bias),
split-conformal recalibration for k>=10. Decision rules R1-R5 are computed here and
never edited by hand. No expected numbers were written before the run.

Usage: python posthoc_round3.py --work <work/microfluidic_reliability> --output-dir <work/posthoc_review_round3>
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
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import xgboost  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from w3.posthoc_round2 import D3_AQUEOUS_LEVELS, D3_NUMERIC, d3_design, locked_regressor  # noqa: E402

K_GRID = (1, 2, 3, 5, 8, 10, 15, 20)
N_RANDOM = 200
RANDOM_SEEDS = tuple(range(N_RANDOM))
KMEANS_SEED = 0
CONFORMAL_MIN_K = 10
NOMINAL = 0.90
REFERENCE_LINE = {"T1_tx100": 11.4, "T2_labB_log": 21.11}
V1_FEATURES = ["continuous_flow_ml_min", "dispersed_flow_ml_min", "equilibrium_interfacial_tension_mn_m", "concentration_over_cmc"]
EXPECTED_SOURCE_MAPE = {"T1_tx100": (19.8436, 4), "T2_labB_log": (43.05, 2)}
FLOOR = 1e-6


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


def mape(y: np.ndarray, p: np.ndarray) -> float:
    return float(100.0 * np.mean(np.abs(p - y) / np.abs(y)))


def mdape(y: np.ndarray, p: np.ndarray) -> float:
    return float(100.0 * np.median(np.abs(p - y) / np.abs(y)))


def signed_bias(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) / np.abs(y)))


# ----------------------------------------------------------------------------- domains
def load_t1(work: Path) -> dict:
    art = work / "blind_artifacts"
    feats = pd.read_csv(art / "blind_features.csv")
    labels = pd.read_csv(art / "sealed_blind_labels.csv")
    preds = pd.read_csv(art / "prereveal" / "blind_predictions.csv")
    dev = pd.read_csv(art / "development_labeled.csv")
    m = feats.merge(labels, on="opaque_row_id", how="inner").merge(
        preds[["opaque_row_id", "predicted_diameter_mm"]], on="opaque_row_id", how="inner")
    if len(m) != 87:
        raise ValueError(f"T1 join produced {len(m)} rows, expected 87")
    y = m["target_diameter_mm"].to_numpy(float)
    f = m["predicted_diameter_mm"].to_numpy(float)
    observed = round(mape(y, f), EXPECTED_SOURCE_MAPE["T1_tx100"][1])
    if observed != EXPECTED_SOURCE_MAPE["T1_tx100"][0]:
        raise ValueError(f"T1 source MAPE {observed} != frozen 19.8436")
    x = m[V1_FEATURES].to_numpy(float)
    train_x = dev[V1_FEATURES].to_numpy(float)
    return {
        "name": "T1_tx100", "n": int(len(m)), "y": y, "f": f, "x": x, "train_x": train_x,
        "log_target_for_scratch": False, "unit": "mm",
        "source_model": "frozen V1 XGBoost point predictions read from blind_artifacts/prereveal/blind_predictions.csv",
        "source_mape_percent": observed,
    }


def load_t2(work: Path) -> dict:
    art = work / "d3_blind_artifacts"
    dev = pd.read_csv(art / "development_labeled.csv")
    feats = pd.read_csv(art / "sealed_primary_features.csv")
    scored = pd.read_csv(art / "reveal" / "sealed_primary_scored_rows.csv")
    dev_ok = dev[np.isfinite(dev["SIZE"].to_numpy(float)) & (dev["SIZE"].to_numpy(float) > 0)]
    x_dev = d3_design(dev_ok)
    y_dev = dev_ok["SIZE"].to_numpy(float)
    joined = feats.merge(scored[["opaque_row_id", "actual_size_nm"]], on="opaque_row_id", how="inner")
    if len(joined) != 57:
        raise ValueError(f"T2 join produced {len(joined)} rows, expected 57")
    x = d3_design(joined)
    y = joined["actual_size_nm"].to_numpy(float)
    model = locked_regressor().fit(x_dev, np.log(y_dev))
    f = np.exp(model.predict(x))
    observed = round(mape(y, f), EXPECTED_SOURCE_MAPE["T2_labB_log"][1])
    if observed != EXPECTED_SOURCE_MAPE["T2_labB_log"][0]:
        raise ValueError(f"T2 log-target source MAPE {observed} != 43.05")
    return {
        "name": "T2_labB_log", "n": int(len(joined)), "y": y, "f": f, "x": x, "train_x": x_dev,
        "log_target_for_scratch": True, "unit": "nm",
        "source_model": "log-target XGBoost (100/3/0.3, seed 42) refitted on the 246 laboratory-A rows exactly as in posthoc_round2",
        "source_mape_percent": observed,
    }


# ----------------------------------------------------------------------------- sampling
def standardise(train_x: np.ndarray, x: np.ndarray) -> np.ndarray:
    mu = train_x.mean(axis=0)
    sd = train_x.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (x - mu) / sd


def kmeans_rows(z: np.ndarray, k: int) -> np.ndarray:
    km = KMeans(n_clusters=k, random_state=KMEANS_SEED, n_init=10).fit(z)
    chosen: list[int] = []
    for c in km.cluster_centers_:
        order = np.argsort(np.linalg.norm(z - c, axis=1))
        for idx in order:
            if int(idx) not in chosen:
                chosen.append(int(idx)); break
    return np.array(chosen, dtype=int)


def farthest_point_rows(z: np.ndarray, train_mean_z: np.ndarray, k: int) -> np.ndarray:
    start = int(np.argmax(np.linalg.norm(z - train_mean_z, axis=1)))
    chosen = [start]
    dmin = np.linalg.norm(z - z[start], axis=1)
    while len(chosen) < k:
        dmin[chosen] = -np.inf
        nxt = int(np.argmax(dmin))
        chosen.append(nxt)
        dmin = np.minimum(dmin, np.linalg.norm(z - z[nxt], axis=1))
    return np.array(chosen, dtype=int)


# ----------------------------------------------------------------------------- calibration
def calibrate(domain: dict, idx: np.ndarray) -> dict:
    y, f, x = domain["y"], domain["f"], domain["x"]
    n = len(y)
    mask = np.ones(n, dtype=bool); mask[idx] = False
    f_floor = np.maximum(f, FLOOR)
    n_clipped = int((f < FLOOR).sum())
    lf, ly = np.log(f_floor), np.log(y)
    k = len(idx)
    out = {"k": k, "n_eval": int(mask.sum()), "n_clipped_predictions": n_clipped}
    # baseline
    out["baseline"] = {"mape": mape(y[mask], f[mask]), "mdape": mdape(y[mask], f[mask]), "bias": signed_bias(y[mask], f[mask])}
    # (i) 1-parameter shift in log space
    a = float(np.mean(ly[idx] - lf[idx]))
    p1 = np.exp(lf + a)
    out["scale"] = {"mape": mape(y[mask], p1[mask]), "mdape": mdape(y[mask], p1[mask]), "bias": signed_bias(y[mask], p1[mask]), "a": a}
    # (ii) 2-parameter affine in log space
    if k >= 2 and np.ptp(lf[idx]) > 0:
        A = np.column_stack([lf[idx], np.ones(k)])
        coef, *_ = np.linalg.lstsq(A, ly[idx], rcond=None)
        p2 = np.exp(coef[0] * lf + coef[1])
        out["affine"] = {"mape": mape(y[mask], p2[mask]), "mdape": mdape(y[mask], p2[mask]), "bias": signed_bias(y[mask], p2[mask]),
                         "alpha": float(coef[0]), "beta": float(coef[1])}
        cal_pred = p2
    else:
        out["affine"] = None
        cal_pred = p1
    # (iii) scratch model on the k rows
    if k >= 3:
        reg = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.3, objective="reg:squarederror", random_state=42, n_jobs=1)
        if domain["log_target_for_scratch"]:
            reg.fit(x[idx], ly[idx]); ps = np.exp(reg.predict(x))
        else:
            reg.fit(x[idx], y[idx]); ps = reg.predict(x)
        out["scratch"] = {"mape": mape(y[mask], ps[mask]), "mdape": mdape(y[mask], ps[mask]), "bias": signed_bias(y[mask], ps[mask]),
                          "n_nonpositive_predictions": int((ps[mask] <= 0).sum())}
    else:
        out["scratch"] = None
    # conformal recalibration (k >= 10) around the calibrated prediction
    if k >= CONFORMAL_MIN_K:
        res = np.abs(ly[idx] - np.log(np.maximum(cal_pred[idx], FLOOR)))
        rank = math.ceil((k + 1) * NOMINAL)
        q = float(np.sort(res)[min(rank, k) - 1]) if rank <= k else float("inf")
        covered = np.abs(ly[mask] - np.log(np.maximum(cal_pred[mask], FLOOR))) <= q
        out["conformal"] = {"radius_log": q, "coverage": float(covered.mean()),
                            "mean_relative_width": float(math.exp(q) - math.exp(-q)) if math.isfinite(q) else None}
    else:
        out["conformal"] = None
    return out


def summarise_domain(domain: dict) -> tuple[dict, list[dict]]:
    z = standardise(domain["train_x"], domain["x"])
    train_mean_z = np.zeros(z.shape[1])
    curves: list[dict] = []
    per_k: dict = {}
    for k in K_GRID:
        rows = []
        for seed in RANDOM_SEEDS:
            rng = np.random.default_rng(seed)
            idx = rng.choice(domain["n"], size=k, replace=False)
            r = calibrate(domain, idx); r.update({"strategy": "random", "seed": seed}); rows.append(r)
        km = calibrate(domain, kmeans_rows(z, k)); km.update({"strategy": "kmeans", "seed": KMEANS_SEED})
        fp = calibrate(domain, farthest_point_rows(z, train_mean_z, k)); fp.update({"strategy": "farthest_point", "seed": None})
        for r in rows + [km, fp]:
            for form in ("baseline", "scale", "affine", "scratch"):
                if r[form] is None:
                    continue
                curves.append({"domain": domain["name"], "k": k, "strategy": r["strategy"], "seed": r["seed"], "form": form,
                               "mape": r[form]["mape"], "mdape": r[form]["mdape"], "bias": r[form]["bias"],
                               "coverage": (r["conformal"] or {}).get("coverage") if form == ("affine" if r["affine"] else "scale") else None,
                               "mean_relative_width": (r["conformal"] or {}).get("mean_relative_width") if form == ("affine" if r["affine"] else "scale") else None,
                               "n_clipped_predictions": r["n_clipped_predictions"]})
        base = float(np.median([r["baseline"]["mape"] for r in rows]))
        summ_k = {"n_eval": rows[0]["n_eval"], "baseline_mape_median_over_draws": base, "random": {}, "kmeans": {}, "farthest_point": {}}
        for form in ("scale", "affine", "scratch"):
            vals = np.array([r[form]["mape"] for r in rows if r[form] is not None], dtype=float)
            if len(vals) == 0:
                summ_k["random"][form] = None
            else:
                summ_k["random"][form] = {
                    "median": float(np.median(vals)), "p05": float(np.quantile(vals, 0.05)), "p95": float(np.quantile(vals, 0.95)),
                    "mdape_median": float(np.median([r[form]["mdape"] for r in rows if r[form] is not None])),
                    "R1_effective": bool(np.median(vals) < base and np.quantile(vals, 0.95) < base),
                }
            for strat, r in (("kmeans", km), ("farthest_point", fp)):
                if r[form] is None:
                    summ_k[strat][form] = None
                else:
                    pct = float(np.mean(vals <= r[form]["mape"])) if len(vals) else None
                    summ_k[strat][form] = {"mape": r[form]["mape"], "percentile_in_random": pct}
        if k >= 3:
            diffs = np.array([r["affine"]["mape"] - r["scratch"]["mape"] for r in rows], dtype=float)
            summ_k["R3_affine_vs_scratch"] = {
                "median_affine": float(np.median([r["affine"]["mape"] for r in rows])),
                "median_scratch": float(np.median([r["scratch"]["mape"] for r in rows])),
                "paired_diff_p05": float(np.quantile(diffs, 0.05)), "paired_diff_p95": float(np.quantile(diffs, 0.95)),
                "calibration_beats_scratch": bool(np.median(diffs) < 0 and np.quantile(diffs, 0.95) < 0),
            }
        if k >= CONFORMAL_MIN_K:
            cov = np.array([r["conformal"]["coverage"] for r in rows]); wid = np.array([r["conformal"]["mean_relative_width"] for r in rows], dtype=float)
            summ_k["R5_conformal"] = {"coverage_median": float(np.median(cov)), "coverage_p05": float(np.quantile(cov, 0.05)),
                                      "coverage_p95": float(np.quantile(cov, 0.95)), "mean_relative_width_median": float(np.nanmedian(wid)),
                                      "coverage_median_in_0_85_0_95": bool(0.85 <= np.median(cov) <= 0.95)}
        per_k[str(k)] = summ_k
    ref = REFERENCE_LINE[domain["name"]]
    r2 = {}
    for form in ("scale", "affine", "scratch"):
        reached = [k for k in K_GRID if per_k[str(k)]["random"].get(form) and per_k[str(k)]["random"][form]["median"] <= ref]
        r2[form] = min(reached) if reached else "not reached within k <= 20"
    r4 = {}
    for strat in ("kmeans", "farthest_point"):
        for form in ("scale", "affine"):
            hits = [k for k in K_GRID if per_k[str(k)][strat].get(form) and per_k[str(k)][strat][form]["percentile_in_random"] is not None
                    and per_k[str(k)][strat][form]["percentile_in_random"] <= 0.10]
            r4[f"{strat}_{form}"] = {"k_at_or_below_p10": hits, "count": len(hits)}
    both_ok = all(r4[f"{s}_affine"]["count"] >= 4 for s in ("kmeans", "farthest_point"))
    summary = {
        "n_rows": domain["n"], "unit": domain["unit"], "source_model": domain["source_model"],
        "source_mape_percent_asserted": domain["source_mape_percent"],
        "reference_line_mape_percent": ref, "per_k": per_k,
        "R2_min_k_reaching_reference_line_random_median": r2,
        "R4_space_filling": {"detail": r4, "verdict_label_free_distance_can_guide_selection": both_ok},
    }
    return summary, curves


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing non-empty output directory {out}")
    out.mkdir(parents=True, exist_ok=True)
    domains = [load_t1(args.work), load_t2(args.work)]
    summary = {"schema_version": "posthoc-round3-recalibration-v1",
               "status": "retrospective; post hoc; labels already open; protocol POSTHOC_ROUND3_RECALIBRATION_PROTOCOL_v1.md frozen before the run",
               "k_grid": list(K_GRID), "n_random_repeats": N_RANDOM, "domains": {}}
    all_curves: list[dict] = []
    for d in domains:
        s, c = summarise_domain(d)
        summary["domains"][d["name"]] = s
        all_curves.extend(c)
    write_json(out / "recalibration_summary.json", summary)
    pd.DataFrame(all_curves).to_csv(out / "recalibration_curves.csv", index=False, float_format="%.10g", lineterminator="\n")
    inputs = ["blind_artifacts/blind_features.csv", "blind_artifacts/sealed_blind_labels.csv", "blind_artifacts/prereveal/blind_predictions.csv",
              "blind_artifacts/development_labeled.csv", "d3_blind_artifacts/development_labeled.csv",
              "d3_blind_artifacts/sealed_primary_features.csv", "d3_blind_artifacts/reveal/sealed_primary_scored_rows.csv"]
    write_json(out / "environment.json", {
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "sklearn": sklearn.__version__, "xgboost": xgboost.__version__, "platform": platform.platform(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "input_sha256": {name: sha256_file(args.work / name) for name in inputs},
    })
    manifest = {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.name != "artifact_hashes.json"}
    write_json(out / "artifact_hashes.json", manifest)
    for name, s in summary["domains"].items():
        print(name, "R2:", s["R2_min_k_reaching_reference_line_random_median"], "R4:", s["R4_space_filling"]["verdict_label_free_distance_can_guide_selection"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
