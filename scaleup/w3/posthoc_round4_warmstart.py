#!/usr/bin/env python3
"""Post-hoc round 4, arm (iv): warm-start residual boosting on the round-3 recalibration
cells (frozen protocol validation_plan/H_G_WARMSTART_PROTOCOL_v1.md; labels already revealed;
retrospective).

Everything of round 3 is imported from w3.posthoc_round3 and recomputed unchanged (same
domains T1 = D2 TX100 and T2 = D3 laboratory B, same k grid, same 200 random seeds, same
k-means / farthest-point draws, same three calibration arms, same evaluation rows).  The
recomputed per-k random medians of the three existing arms must reproduce the frozen
round-3 summary to 1e-9, otherwise the script aborts.

New arm (iv): XGBoost residual trees fitted on log y with base_margin = log f_source(x),
hyper-parameters fixed by the protocol, lambda in {0, 0.1} for down-weighted source rows
(in-sample source predictions as their base margin).  Rules R1/R2/R4 are computed for the
new arm; R5 is declared not applicable; R6 is the paired comparison against the affine and
scratch arms.  No expected number was written before the run.

Usage: python posthoc_round4_warmstart.py --work <work/microfluidic_reliability> --output-dir <dir>
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import xgboost  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from w3.posthoc_round2 import d3_design, locked_regressor  # noqa: E402
from w3.posthoc_round3 import (  # noqa: E402
    FLOOR, K_GRID, KMEANS_SEED, RANDOM_SEEDS, REFERENCE_LINE, V1_FEATURES,
    calibrate, farthest_point_rows, kmeans_rows, load_t1, load_t2, mape, mdape, sha256_file, signed_bias, standardise, write_json,
)

LAMBDAS = (0.0, 0.1)
WARM_PARAMS = dict(n_estimators=100, max_depth=2, learning_rate=0.1, min_child_weight=1, subsample=1.0, colsample_bytree=1.0,
                   reg_lambda=1.0, objective="reg:squarederror", random_state=42, n_jobs=1)
REPRO_TOL = 1e-9
SOURCE_REPRO_TOL = 1e-6
FROZEN_ROUND3 = Path(__file__).resolve().parents[1] / "posthoc_round3" / "recalibration_summary.json"
EXISTING_FORMS = ("baseline", "scale", "affine", "scratch")


def warm_name(lam: float) -> str:
    return f"warm_l{lam:g}"


# ----------------------------------------------------------------------------- source rows
def source_rows_t1(work: Path, domain: dict) -> tuple[dict | None, dict]:
    """Refit the locked V1 declaration on the 381 development rows; accept only if it reproduces the frozen TX100 predictions."""
    dev = pd.read_csv(work / "blind_artifacts" / "development_labeled.csv")
    x_dev = dev[V1_FEATURES].to_numpy(float)
    y_dev = dev["target_diameter_mm"].to_numpy(float)
    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.3, random_state=42, objective="reg:squarederror", n_jobs=1)
    model.fit(x_dev, y_dev)
    f_sealed = model.predict(domain["x"]).astype(float)
    max_abs = float(np.max(np.abs(f_sealed - domain["f"])))
    ok = bool(np.allclose(f_sealed, domain["f"], atol=SOURCE_REPRO_TOL, rtol=0.0))
    info = {"n_source_rows": int(len(dev)), "max_abs_diff_vs_frozen_mm": max_abs, "reproduces_frozen": ok,
            "note": "locked V1 declaration (XGBoost 100/3/0.3, seed 42, four V1 inputs, raw mm) refitted on development_labeled.csv"}
    if not ok:
        return None, info
    f_dev = model.predict(x_dev).astype(float)
    return {"x": x_dev, "ly": np.log(y_dev), "lf": np.log(np.maximum(f_dev, FLOOR))}, info


def source_rows_t2(work: Path, domain: dict) -> tuple[dict | None, dict]:
    art = work / "d3_blind_artifacts"
    dev = pd.read_csv(art / "development_labeled.csv")
    dev_ok = dev[np.isfinite(dev["SIZE"].to_numpy(float)) & (dev["SIZE"].to_numpy(float) > 0)]
    x_dev = d3_design(dev_ok)
    y_dev = dev_ok["SIZE"].to_numpy(float)
    model = locked_regressor().fit(x_dev, np.log(y_dev))
    f_sealed = np.exp(model.predict(domain["x"]))
    max_abs = float(np.max(np.abs(f_sealed - domain["f"])))
    ok = bool(np.allclose(f_sealed, domain["f"], atol=SOURCE_REPRO_TOL, rtol=0.0))
    info = {"n_source_rows": int(len(dev_ok)), "max_abs_diff_vs_frozen_nm": max_abs, "reproduces_frozen": ok,
            "note": "log-target locked regressor refitted on laboratory-A rows exactly as posthoc_round3.load_t2"}
    if not ok:
        return None, info
    f_dev = np.exp(model.predict(x_dev))
    return {"x": x_dev, "ly": np.log(y_dev), "lf": np.log(np.maximum(f_dev, FLOOR))}, info


# ----------------------------------------------------------------------------- arm (iv)
def warm_start(domain: dict, idx: np.ndarray, lam: float, source: dict | None) -> dict | None:
    if lam > 0 and source is None:
        return None
    y, f, x = domain["y"], domain["f"], domain["x"]
    n = len(y)
    mask = np.ones(n, dtype=bool)
    mask[idx] = False
    lf = np.log(np.maximum(f, FLOOR))
    ly = np.log(y)
    X, Y, BM, W = x[idx], ly[idx], lf[idx], np.ones(len(idx))
    if lam > 0:
        X = np.vstack([X, source["x"]])
        Y = np.concatenate([Y, source["ly"]])
        BM = np.concatenate([BM, source["lf"]])
        W = np.concatenate([W, np.full(len(source["ly"]), lam)])
    reg = XGBRegressor(**WARM_PARAMS)
    reg.fit(X, Y, base_margin=BM, sample_weight=W)
    p = np.exp(reg.predict(x, base_margin=lf).astype(float))
    return {"mape": mape(y[mask], p[mask]), "mdape": mdape(y[mask], p[mask]), "bias": signed_bias(y[mask], p[mask])}


# ----------------------------------------------------------------------------- per domain
def run_domain(domain: dict, source: dict | None, frozen: dict) -> tuple[dict, list[dict]]:
    z = standardise(domain["train_x"], domain["x"])
    train_mean_z = np.zeros(z.shape[1])
    forms = list(EXISTING_FORMS) + [warm_name(l) for l in LAMBDAS]
    curves: list[dict] = []
    per_k: dict = {}
    repro: dict = {}
    for k in K_GRID:
        draws = []
        for seed in RANDOM_SEEDS:
            rng = np.random.default_rng(seed)
            idx = rng.choice(domain["n"], size=k, replace=False)  # identical to round 3
            r = calibrate(domain, idx)
            for lam in LAMBDAS:
                r[warm_name(lam)] = warm_start(domain, idx, lam, source)
            r.update({"strategy": "random", "seed": seed})
            draws.append(r)
        idx_km = kmeans_rows(z, k)
        km = calibrate(domain, idx_km)
        for lam in LAMBDAS:
            km[warm_name(lam)] = warm_start(domain, idx_km, lam, source)
        km.update({"strategy": "kmeans", "seed": KMEANS_SEED})
        idx_fp = farthest_point_rows(z, train_mean_z, k)
        fp = calibrate(domain, idx_fp)
        for lam in LAMBDAS:
            fp[warm_name(lam)] = warm_start(domain, idx_fp, lam, source)
        fp.update({"strategy": "farthest_point", "seed": None})
        for r in draws + [km, fp]:
            for form in forms:
                if r.get(form) is None:
                    continue
                curves.append({"domain": domain["name"], "k": k, "strategy": r["strategy"], "seed": r["seed"], "form": form,
                               "mape": r[form]["mape"], "mdape": r[form]["mdape"], "bias": r[form]["bias"]})
        base = float(np.median([r["baseline"]["mape"] for r in draws]))
        # reproduction of the frozen round-3 random medians for the existing arms
        frozen_k = frozen["domains"][domain["name"]]["per_k"][str(k)]
        repro_k = {"baseline": {"recomputed": base, "frozen": frozen_k["baseline_mape_median_over_draws"]}}
        for form in ("scale", "affine", "scratch"):
            vals = [r[form]["mape"] for r in draws if r[form] is not None]
            recomputed = float(np.median(vals)) if vals else None
            frozen_val = (frozen_k["random"].get(form) or {}).get("median")
            repro_k[form] = {"recomputed": recomputed, "frozen": frozen_val}
        for form, pair in repro_k.items():
            a, b = pair["recomputed"], pair["frozen"]
            if (a is None) != (b is None) or (a is not None and abs(a - b) > REPRO_TOL):
                raise SystemExit(f"round-3 reproduction failed: {domain['name']} k={k} {form}: recomputed {a} vs frozen {b}")
        repro[str(k)] = repro_k
        summ_k = {"n_eval": draws[0]["n_eval"], "baseline_mape_median_over_draws": base, "random": {}, "kmeans": {}, "farthest_point": {}, "R6_paired": {}}
        for lam in LAMBDAS:
            form = warm_name(lam)
            vals = np.array([r[form]["mape"] for r in draws if r[form] is not None], dtype=float)
            if len(vals) == 0:
                summ_k["random"][form] = None
                summ_k["kmeans"][form] = None
                summ_k["farthest_point"][form] = None
                continue
            summ_k["random"][form] = {
                "median": float(np.median(vals)), "p05": float(np.quantile(vals, 0.05)), "p95": float(np.quantile(vals, 0.95)),
                "mdape_median": float(np.median([r[form]["mdape"] for r in draws if r[form] is not None])),
                "bias_median": float(np.median([r[form]["bias"] for r in draws if r[form] is not None])),
                "R1_effective": bool(np.median(vals) < base and np.quantile(vals, 0.95) < base),
            }
            for strat, r in (("kmeans", km), ("farthest_point", fp)):
                summ_k[strat][form] = None if r[form] is None else {"mape": r[form]["mape"], "percentile_in_random": float(np.mean(vals <= r[form]["mape"]))}
            for comp in ("affine", "scratch"):
                pairs = [(r[form]["mape"], r[comp]["mape"]) for r in draws if r[form] is not None and r[comp] is not None]
                if not pairs:
                    summ_k["R6_paired"][f"{form}_minus_{comp}"] = None
                    continue
                diffs = np.array([a - b for a, b in pairs], dtype=float)
                summ_k["R6_paired"][f"{form}_minus_{comp}"] = {
                    "n_pairs": int(len(diffs)), "median": float(np.median(diffs)),
                    "p05": float(np.quantile(diffs, 0.05)), "p95": float(np.quantile(diffs, 0.95)),
                    "warm_better": bool(np.median(diffs) < 0 and np.quantile(diffs, 0.95) < 0),
                }
        per_k[str(k)] = summ_k
    ref = REFERENCE_LINE[domain["name"]]
    r2, r4, r6 = {}, {}, {}
    for lam in LAMBDAS:
        form = warm_name(lam)
        reached = [k for k in K_GRID if per_k[str(k)]["random"].get(form) and per_k[str(k)]["random"][form]["median"] <= ref]
        r2[form] = min(reached) if reached else "not reached within k <= 20"
        for strat in ("kmeans", "farthest_point"):
            hits = [k for k in K_GRID if per_k[str(k)][strat].get(form) and per_k[str(k)][strat][form]["percentile_in_random"] <= 0.10]
            r4[f"{strat}_{form}"] = {"k_at_or_below_p10": hits, "count": len(hits)}
        for comp in ("affine", "scratch"):
            key = f"{form}_minus_{comp}"
            comparable = [k for k in K_GRID if per_k[str(k)]["R6_paired"].get(key)]
            better = [k for k in comparable if per_k[str(k)]["R6_paired"][key]["warm_better"]]
            threshold = max(len(comparable) - 2, 0)
            r6[key] = {"comparable_k": comparable, "better_k": better, "n_comparable": len(comparable), "n_better": len(better),
                       "threshold_n_better": threshold,
                       "claim_better_at_all_comparable_k_allowed": bool(len(comparable) > 0 and len(better) >= threshold)}
    summary = {
        "n_rows": domain["n"], "unit": domain["unit"], "source_model": domain["source_model"],
        "source_mape_percent_asserted": domain["source_mape_percent"], "reference_line_mape_percent": ref,
        "round3_reproduction_check": {"tolerance": REPRO_TOL, "passed": True, "per_k": repro},
        "per_k": per_k, "R2_min_k_reaching_reference_line_random_median": r2, "R4_space_filling": r4,
        "R5_conformal": "not evaluated for the warm-start arm (the k rows are all used for fitting; no held-out calibration rows)",
        "R6_decision": r6,
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
    frozen = json.loads(FROZEN_ROUND3.read_text())
    t1 = load_t1(args.work)
    t2 = load_t2(args.work)
    src1, info1 = source_rows_t1(args.work, t1)
    src2, info2 = source_rows_t2(args.work, t2)
    summary = {"schema_version": "posthoc-round4-warmstart-v1",
               "status": "retrospective; post hoc; labels already open; protocol H_G_WARMSTART_PROTOCOL_v1.md frozen before the run",
               "protocol_sha256_file": "validation_plan/H_G_WARMSTART_PROTOCOL_v1.md.sha256",
               "warm_params": WARM_PARAMS, "lambdas": list(LAMBDAS), "k_grid": list(K_GRID), "n_random_repeats": len(RANDOM_SEEDS),
               "source_rows": {"T1_tx100": info1, "T2_labB_log": info2},
               "lambda_positive_arm_evaluable": {"T1_tx100": src1 is not None, "T2_labB_log": src2 is not None},
               "frozen_round3_summary_sha256": sha256_file(FROZEN_ROUND3), "domains": {}}
    curves: list[dict] = []
    for d, src in ((t1, src1), (t2, src2)):
        s, c = run_domain(d, src, frozen)
        summary["domains"][d["name"]] = s
        curves.extend(c)
    write_json(out / "warm_start_summary.json", summary)
    pd.DataFrame(curves).to_csv(out / "warm_start_curves.csv", index=False, float_format="%.10g", lineterminator="\n")
    inputs = ["blind_artifacts/blind_features.csv", "blind_artifacts/sealed_blind_labels.csv", "blind_artifacts/prereveal/blind_predictions.csv",
              "blind_artifacts/development_labeled.csv", "d3_blind_artifacts/development_labeled.csv",
              "d3_blind_artifacts/sealed_primary_features.csv", "d3_blind_artifacts/reveal/sealed_primary_scored_rows.csv"]
    write_json(out / "environment.json", {
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "sklearn": sklearn.__version__, "xgboost": xgboost.__version__, "platform": platform.platform(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "round3_script_sha256": sha256_file(Path(__file__).resolve().parent / "posthoc_round3.py"),
        "input_sha256": {name: sha256_file(args.work / name) for name in inputs},
    })
    manifest = {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.name != "artifact_hashes.json"}
    write_json(out / "artifact_hashes.json", manifest)
    for name, s in summary["domains"].items():
        print(name, "R2:", s["R2_min_k_reaching_reference_line_random_median"], "R6:", {k: v["claim_better_at_all_comparable_k_allowed"] for k, v in s["R6_decision"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
