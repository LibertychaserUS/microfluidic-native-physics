#!/usr/bin/env python3
"""H-G v2: warm-start residual boosting on the sealed tiers of R2-A1 and R2-A2 (frozen protocol
validation_plan/H_G_WARMSTART_PROTOCOL_v2.md; post hoc, after the reveals).

Domains T3 = R2-A1 S2, T4 = R2-A1 S3, T5 = R2-A2 S2, T6 = R2-A2 S3, T7 = R2-A2 S4.  The source model
is the frozen xgboost|native point prediction of the sealed bundle; the locked XGBoost is refitted on
the development rows only to obtain in-sample base margins for the lambda = 0.1 arm and must reproduce
the bundle predictions (1e-6).  Sampling, the three round-3 arms and the warm arm are imported from the
frozen round-3 / H-G v1 programs; k values with fewer than 10 evaluation rows are not applicable.
Reference line for R2 = the dataset's dev-ref MAPE.

Usage: python posthoc_round4_warmstart2.py --work <work/microfluidic_reliability> --output-dir <dir>
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
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from w3.posthoc_round3 import FLOOR, K_GRID, KMEANS_SEED, RANDOM_SEEDS, calibrate, farthest_point_rows, kmeans_rows, sha256_file, standardise, write_json  # noqa: E402
from w3.posthoc_round4_warmstart import LAMBDAS, WARM_PARAMS, SOURCE_REPRO_TOL, warm_name, warm_start  # noqa: E402

MIN_EVAL_ROWS = 10
DOMAINS = [("T3_R2A1_S2", "R2A1", "S2"), ("T4_R2A1_S3", "R2A1", "S3"), ("T5_R2A2_S2", "R2A2", "S2"), ("T6_R2A2_S3", "R2A2", "S3"), ("T7_R2A2_S4", "R2A2", "S4")]
EXISTING_FORMS = ("baseline", "scale", "affine", "scratch")


def load_domains(work: Path) -> list[tuple[dict, dict | None, dict]]:
    out = []
    cache: dict[str, dict] = {}
    for name, ds, tier in DOMAINS:
        if ds not in cache:
            art = work / f"h_{ds}_artifacts"
            manifest = json.loads((art / "custodian_manifest.json").read_text())
            devdiag = json.loads((art / f"h_{ds}_devdiag.json").read_text())
            run_summary = json.loads((work / f"h_{ds}_run1" / "h_run_summary.json").read_text())
            dev = pd.read_csv(art / "development_labeled.csv", keep_default_na=False, na_values=[])
            native = list(manifest["native_columns"])
            x_dev = dev[native].to_numpy(float); y_dev = dev["target"].to_numpy(float)
            log_t = devdiag["target_scale"] == "log"
            # H-D14: the frozen sealed predictions come from the runner's 80% fit partition (GroupShuffleSplit by s2_group,
            # seed 42, test_size 0.2 over all development rows; the other 20% is the conformal calibration partition)
            train_idx = np.arange(len(dev))
            s2 = dev["s2_group"].astype(str).to_numpy()
            if len(set(s2)) < 2:
                fit_rel, _ = next(ShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(train_idx))
            else:
                fit_rel, _ = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(train_idx, groups=s2))
            fit_idx = train_idx[fit_rel]
            model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.3, objective="reg:squarederror", random_state=42, n_jobs=1)
            model.fit(x_dev[fit_idx], np.log(y_dev[fit_idx]) if log_t else y_dev[fit_idx])
            f_dev = model.predict(x_dev).astype(float); f_dev = np.exp(f_dev) if log_t else f_dev  # source rows = all development rows, base margin from the fit-partition model
            cache[ds] = {"native": native, "x_dev": x_dev, "y_dev": y_dev, "log": log_t, "model": model, "f_dev": f_dev, "dev_ref": float(run_summary["dev_ref"]["mape_mean"]),
                         "sealed": pd.read_csv(art / "sealed_features.csv", keep_default_na=False, na_values=[]).set_index("row_id"),
                         "rev": pd.read_csv(work / f"h_{ds}_reveal" / "sealed_scored_rows.csv"), "unit": devdiag.get("target_unit")}
        c = cache[ds]
        part = c["rev"][(c["rev"].tier == tier) & (c["rev"].model == "xgboost") & (c["rev"].feature_set == "native")].sort_values("row_id", kind="stable").reset_index(drop=True)
        x = c["sealed"].loc[part.row_id][c["native"]].to_numpy(float)
        y, f = part["actual"].to_numpy(float), part["pred"].to_numpy(float)
        f_refit = c["model"].predict(x).astype(float); f_refit = np.exp(f_refit) if c["log"] else f_refit
        max_abs = float(np.max(np.abs(f_refit - f)))
        ok = bool(np.allclose(f_refit, f, atol=SOURCE_REPRO_TOL, rtol=0.0))
        domain = {"name": name, "n": int(len(y)), "y": y, "f": f, "x": x, "train_x": c["x_dev"], "log_target_for_scratch": c["log"], "unit": c["unit"],
                  "source_model": f"h_{ds}_run1 sealed_{tier} xgboost native (frozen bundle)", "source_mape_percent": float(100 * np.mean(np.abs(f - y) / y)),
                  "reference_line": c["dev_ref"]}
        source = {"x": c["x_dev"], "ly": np.log(c["y_dev"]), "lf": np.log(np.maximum(c["f_dev"], FLOOR))} if ok else None
        info = {"n_source_rows": int(len(c["y_dev"])), "max_abs_diff_vs_frozen": max_abs, "reproduces_frozen": ok, "log_target": c["log"]}
        out.append((domain, source, info))
    return out


def run_domain(domain: dict, source: dict | None) -> tuple[dict, list[dict]]:
    z = standardise(domain["train_x"], domain["x"])
    train_mean_z = np.zeros(z.shape[1])
    forms = list(EXISTING_FORMS) + [warm_name(l) for l in LAMBDAS]
    ks = [k for k in K_GRID if domain["n"] - k >= MIN_EVAL_ROWS]
    curves: list[dict] = []
    per_k: dict = {}
    for k in ks:
        draws = []
        for seed in RANDOM_SEEDS:
            rng = np.random.default_rng(seed)
            idx = rng.choice(domain["n"], size=k, replace=False)
            r = calibrate(domain, idx)
            for lam in LAMBDAS:
                r[warm_name(lam)] = warm_start(domain, idx, lam, source)
            r.update({"strategy": "random", "seed": seed}); draws.append(r)
        idx_km = kmeans_rows(z, k); km = calibrate(domain, idx_km)
        for lam in LAMBDAS:
            km[warm_name(lam)] = warm_start(domain, idx_km, lam, source)
        km.update({"strategy": "kmeans", "seed": KMEANS_SEED})
        idx_fp = farthest_point_rows(z, train_mean_z, k); fp = calibrate(domain, idx_fp)
        for lam in LAMBDAS:
            fp[warm_name(lam)] = warm_start(domain, idx_fp, lam, source)
        fp.update({"strategy": "farthest_point", "seed": None})
        for r in draws + [km, fp]:
            for form in forms:
                if r.get(form) is None:
                    continue
                curves.append({"domain": domain["name"], "k": k, "strategy": r["strategy"], "seed": r["seed"], "form": form, "mape": r[form]["mape"], "mdape": r[form]["mdape"], "bias": r[form]["bias"]})
        base = float(np.median([r["baseline"]["mape"] for r in draws]))
        summ_k = {"n_eval": draws[0]["n_eval"], "baseline_mape_median_over_draws": base, "random": {}, "kmeans": {}, "farthest_point": {}, "R6_paired": {}}
        for form in ("scale", "affine", "scratch") + tuple(warm_name(l) for l in LAMBDAS):
            vals = np.array([r[form]["mape"] for r in draws if r.get(form) is not None], dtype=float)
            if len(vals) == 0:
                summ_k["random"][form] = None; summ_k["kmeans"][form] = None; summ_k["farthest_point"][form] = None
                continue
            summ_k["random"][form] = {"median": float(np.median(vals)), "p05": float(np.quantile(vals, 0.05)), "p95": float(np.quantile(vals, 0.95)),
                                      "mdape_median": float(np.median([r[form]["mdape"] for r in draws if r.get(form) is not None])),
                                      "bias_median": float(np.median([r[form]["bias"] for r in draws if r.get(form) is not None])),
                                      "R1_effective": bool(np.median(vals) < base and np.quantile(vals, 0.95) < base)}
            for strat, r in (("kmeans", km), ("farthest_point", fp)):
                summ_k[strat][form] = None if r.get(form) is None else {"mape": r[form]["mape"], "percentile_in_random": float(np.mean(vals <= r[form]["mape"]))}
        for lam in LAMBDAS:
            form = warm_name(lam)
            for comp in ("affine", "scratch"):
                pairs = [(r[form]["mape"], r[comp]["mape"]) for r in draws if r.get(form) is not None and r.get(comp) is not None]
                if not pairs:
                    summ_k["R6_paired"][f"{form}_minus_{comp}"] = None; continue
                diffs = np.array([a - b for a, b in pairs], dtype=float)
                summ_k["R6_paired"][f"{form}_minus_{comp}"] = {"n_pairs": int(len(diffs)), "median": float(np.median(diffs)), "p05": float(np.quantile(diffs, 0.05)),
                                                               "p95": float(np.quantile(diffs, 0.95)), "warm_better": bool(np.median(diffs) < 0 and np.quantile(diffs, 0.95) < 0)}
        per_k[str(k)] = summ_k
    ref = domain["reference_line"]
    r2, r4, r6 = {}, {}, {}
    for form in ("scale", "affine", "scratch") + tuple(warm_name(l) for l in LAMBDAS):
        reached = [k for k in ks if per_k[str(k)]["random"].get(form) and per_k[str(k)]["random"][form]["median"] <= ref]
        r2[form] = min(reached) if reached else "not reached"
    for lam in LAMBDAS:
        form = warm_name(lam)
        for strat in ("kmeans", "farthest_point"):
            hits = [k for k in ks if per_k[str(k)][strat].get(form) and per_k[str(k)][strat][form]["percentile_in_random"] <= 0.10]
            r4[f"{strat}_{form}"] = {"k_at_or_below_p10": hits, "count": len(hits)}
        for comp in ("affine", "scratch"):
            key = f"{form}_minus_{comp}"
            comparable = [k for k in ks if per_k[str(k)]["R6_paired"].get(key)]
            better = [k for k in comparable if per_k[str(k)]["R6_paired"][key]["warm_better"]]
            threshold = max(len(comparable) - 2, 1)  # H-D15: at least one comparable k must be better; the claim needs >= 3 comparable k
            r6[key] = {"comparable_k": comparable, "better_k": better, "n_comparable": len(comparable), "n_better": len(better), "threshold_n_better": threshold,
                       "claim_better_at_all_comparable_k_allowed": bool(len(comparable) >= 3 and len(better) >= threshold)}
    summary = {"n_rows": domain["n"], "unit": domain["unit"], "k_applicable": ks, "k_not_applicable": [k for k in K_GRID if k not in ks], "source_model": domain["source_model"],
               "source_mape_percent": domain["source_mape_percent"], "reference_line_mape_percent_dev_ref": ref, "per_k": per_k,
               "R2_min_k_reaching_dev_ref_random_median": r2, "R4_space_filling": r4, "R5_conformal": "not evaluated for the warm-start arm", "R6_decision": r6}
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
    domains = load_domains(args.work)
    summary = {"schema_version": "posthoc-round4-warmstart-v2", "status": "retrospective; post hoc; after the R2-A1/R2-A2 reveals; protocol H_G_WARMSTART_PROTOCOL_v2.md frozen before the reveals",
               "warm_params": WARM_PARAMS, "lambdas": list(LAMBDAS), "k_grid": list(K_GRID), "min_eval_rows": MIN_EVAL_ROWS, "n_random_repeats": len(RANDOM_SEEDS),
               "source_rows": {d["name"]: info for d, _, info in domains}, "lambda_positive_arm_evaluable": {d["name"]: s is not None for d, s, _ in domains}, "domains": {}}
    curves: list[dict] = []
    for d, src, _ in domains:
        s, c = run_domain(d, src)
        summary["domains"][d["name"]] = s; curves.extend(c)
    write_json(out / "warm_start_summary.json", summary)
    pd.DataFrame(curves).to_csv(out / "warm_start_curves.csv", index=False, float_format="%.10g", lineterminator="\n")
    inputs = {}
    for ds in ("R2A1", "R2A2"):
        for rel in (f"h_{ds}_artifacts/development_labeled.csv", f"h_{ds}_artifacts/sealed_features.csv", f"h_{ds}_reveal/sealed_scored_rows.csv", f"h_{ds}_run1/h_run_summary.json"):
            inputs[rel] = sha256_file(args.work / rel)
    write_json(out / "environment.json", {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
                                          "xgboost": xgboost.__version__, "platform": platform.platform(), "script_sha256": sha256_file(Path(__file__).resolve()),
                                          "v1_program_sha256": sha256_file(Path(__file__).with_name("posthoc_round4_warmstart.py")), "round3_script_sha256": sha256_file(Path(__file__).with_name("posthoc_round3.py")),
                                          "input_sha256": inputs})
    manifest = {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.name != "artifact_hashes.json"}
    write_json(out / "artifact_hashes.json", manifest)
    for name, s in summary["domains"].items():
        print(name, "n", s["n_rows"], "ks", s["k_applicable"], "R2:", s["R2_min_k_reaching_dev_ref_random_median"], "R6:", {k: v["claim_better_at_all_comparable_k_allowed"] for k, v in s["R6_decision"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
