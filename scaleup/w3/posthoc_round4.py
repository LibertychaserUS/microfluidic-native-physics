#!/usr/bin/env python3
"""Round-4 mechanism checks (retrospective; not part of any frozen protocol).

Blocks 5 and 6 (V2 held-out interval widths; W4 native widths internal vs external) were added for the
fifth and sixth review rounds.

Written after the fourth external review of manuscript v0.3.3. Every quantity
here is recomputed from canonical artifacts or from the frozen V2 code; nothing
is refitted and no frozen file is modified. Four blocks:

1. anchor_map        -- the V2 anchor-to-logit map as implemented
                        (``normalized_to_logit`` in ``src/native_physics_v2.py``):
                        clip order, saturation value, and the distribution of
                        anchor values y_phys on the 466-row table.
2. device_type_quartiles -- signed error direction per size quartile of the
                        device-type cells (same quartile rule as round 2).
3. prediction_ranges -- prediction ranges by model family in the device-type
                        cells, in micrometres and in model-response space,
                        against the training response range.
4. within_context_retention -- per source--geometry context retention of the
                        binary rules in the expanded suite, and the
                        within-source-context permutation row.

Usage: python posthoc_round4.py --work <work/microfluidic_reliability> --output-dir <work/posthoc_review_round4>
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from w3.datasets import load_d1  # noqa: E402
from w3.workspace import load_workspace  # noqa: E402

RULES = {
    "knn_5_q95": "accepted__knn_5__q0_95",
    "mahalanobis_oas_q95": "accepted__mahalanobis_oas__q0_95",
    "isolation_forest_q95": "accepted__isolation_forest__q0_95",
    "strict_minmax": "accepted__strict_minmax",
}
DEVICE_CELLS = {"D1_de_to_se": "SE", "D1_se_to_de": "DE"}  # cell -> test device type
MODELS = ("xgboost", "random_forest", "mlp")


def sanitize(value):
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, (np.floating, float)):
        f = float(value)
        if not np.isfinite(f):
            raise ValueError("non-finite value in output")
        return f
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(sanitize(payload), indent=2, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- 1. anchor map
def anchor_map(work: Path, v2) -> dict:
    sens = pd.read_csv(work / "v2_artifacts_canonical_run2" / "locked_physics_sensitivities.csv")
    d_mm = float(v2.D_PRIMARY_MM)
    eps = float(v2.EPSILON)
    y = sens["B0_primary_D195_oval_mm"].to_numpy(float) / d_mm
    valid = np.isfinite(y) & (y > 0.0) & (y < 1.0)
    if not np.array_equal(valid, sens["primary_anchor_valid"].to_numpy(bool)):
        raise ValueError("recomputed validity mask differs from the frozen artifact")
    # the frozen code path: outer clip -> normalise -> inner clip at machine epsilon -> logit
    logit_valid = v2.normalized_to_logit(np.clip(y[valid], eps, 1.0 - eps))
    delta = float(np.finfo(float).eps)
    saturated = float(v2.normalized_to_logit(np.array([1.0 - eps]))[0])
    probe = v2.normalized_to_logit(np.array([1.0 - 2.0 * eps, 1.0 - eps, 1.0 - 0.5 * eps, 0.5 * eps, eps, 2.0 * eps]))
    return {
        "code_path": "src/native_physics_v2.py: semi_empirical_anchor -> _torch_anchor -> normalized_to_logit; "
                     "validity mask re-applied in _network_output",
        "order_of_operations": "clip(y, eps, 1-eps) -> (y - eps)/(1 - 2 eps) -> clip(., machine_eps, 1 - machine_eps) -> log(u/(1-u))",
        "epsilon": eps,
        "inner_clip_machine_epsilon": delta,
        "saturated_logit_magnitude": abs(saturated),
        "probe_logits": {
            "y=1-2eps": float(probe[0]), "y=1-eps": float(probe[1]), "y=1-eps/2": float(probe[2]),
            "y=eps/2": float(probe[3]), "y=eps": float(probe[4]), "y=2eps": float(probe[5]),
        },
        "fallback_logit_for_invalid_anchor": 0.0,
        "fallback_diameter_mm": d_mm * (eps + (1.0 - 2.0 * eps) * 0.5),
        "n_rows": int(len(y)),
        "n_anchor_invalid": int((~valid).sum()),
        "fraction_anchor_invalid": float((~valid).mean()),
        "invalid_y_values_sorted": sorted(float(v) for v in y[~valid]),
        "invalid_y_min": float(y[~valid].min()),
        "invalid_y_max": float(y[~valid].max()),
        "valid_y_min": float(y[valid].min()),
        "valid_y_max": float(y[valid].max()),
        "n_valid_within_eps_of_lower_bound": int(((y > 0.0) & (y <= eps)).sum()),
        "n_valid_within_eps_of_upper_bound": int(((y >= 1.0 - eps) & (y < 1.0)).sum()),
        "anchor_logit_min": float(logit_valid.min()),
        "anchor_logit_max": float(logit_valid.max()),
        "note": "saturation (|logit| = 36.04) is reachable only for anchor-valid rows within eps of a bound; "
                "no such row exists in the 466-row table, so the two-stage clip is a recorded limitation of the "
                "frozen code, not an exercised path. The validity fallback to logit 0 (d = D/2) is a discontinuous "
                "guard: as y_phys crosses 1 the anchor term jumps from the saturated value to 0.",
    }


# ----------------------------------------------------------------------------- 2./3. device-type cells
def quartile_direction(actual: np.ndarray, pred: np.ndarray) -> list[dict]:
    edges = np.quantile(actual, [0.0, 0.25, 0.5, 0.75, 1.0])
    out = []
    for i in range(4):
        lo, hi = edges[i], edges[i + 1]
        mask = (actual >= lo) & (actual <= hi) if i == 3 else (actual >= lo) & (actual < hi)
        signed = (pred[mask] - actual[mask]) / actual[mask]
        out.append({
            "bin": i + 1,
            "observed_range_um": [float(lo), float(hi)],
            "n": int(mask.sum()),
            "mape_percent": float(100.0 * np.abs(signed).mean()),
            "mean_signed_relative_error": float(signed.mean()),
            "fraction_over_predicted": float((pred[mask] > actual[mask]).mean()),
            "median_predicted_over_observed": float(np.median(pred[mask] / actual[mask])),
        })
    return out


def device_type_blocks(work: Path) -> tuple[dict, dict]:
    pp = pd.read_csv(work / "w3_artifacts_canonical_run2" / "point_predictions.csv")
    ws = load_workspace(work)
    comp, _ = load_d1(ws)
    hydraulic = pd.Series(comp.size_proxy, index=comp.row_id)
    response = pd.Series(comp.response, index=comp.row_id)
    event = pd.Series(comp.groups["event_type"], index=comp.row_id)
    quartiles: dict = {}
    ranges: dict = {}
    for cell, test_type in DEVICE_CELLS.items():
        train_type = "DE" if test_type == "SE" else "SE"
        train_resp = response[event == train_type].to_numpy(float)
        test_resp = response[event == test_type].to_numpy(float)
        quartiles[cell] = {}
        ranges[cell] = {
            "test_device_type": test_type,
            "training_device_type": train_type,
            "training_response_range_all_rows": [float(train_resp.min()), float(train_resp.max())],
            "test_response_range_all_rows": [float(test_resp.min()), float(test_resp.max())],
            "fraction_test_rows_with_response_below_training_floor": float((test_resp < train_resp.min()).mean()),
            "fraction_test_rows_with_response_above_training_ceiling": float((test_resp > train_resp.max()).mean()),
            "note": "training rows of one direction are the test rows of the other; the fit partition is a "
                    "subset of these rows, so this range is an outer bound on the fit-partition response range",
            "models": {},
        }
        for model in MODELS:
            d = pp[(pp["cell"] == cell) & (pp["model"] == model)]
            if d.empty:
                continue
            if d["repeat"].nunique() != 1:
                raise ValueError(f"{cell}/{model}: expected a single repeat")
            a = d["actual_um"].to_numpy(float)
            p = d["pred_um"].to_numpy(float)
            h = hydraulic.loc[d["row_id"].to_numpy()].to_numpy(float)
            resp_pred = p / h
            quartiles[cell][model] = quartile_direction(a, p)
            ranges[cell]["models"][model] = {
                "n_test_rows": int(len(d)),
                "pred_um_min": float(p.min()), "pred_um_max": float(p.max()),
                "actual_um_min": float(a.min()), "actual_um_max": float(a.max()),
                "response_pred_min": float(resp_pred.min()), "response_pred_max": float(resp_pred.max()),
                "fraction_response_pred_within_training_response_range": float(
                    ((resp_pred >= train_resp.min()) & (resp_pred <= train_resp.max())).mean()),
                "fraction_over_predicted": float((p > a).mean()),
            }
    return quartiles, ranges


# ----------------------------------------------------------------------------- 4. within-context retention
def within_context_retention(work: Path) -> dict:
    scored = pd.read_csv(work / "expanded_artifacts_canonical_run2" / "applicability_scored.csv")
    if scored["model"].nunique() != 1:
        raise ValueError("expected a single model in applicability_scored.csv")
    rules = {}
    for name, col in RULES.items():
        g = scored.groupby(["source", "source_context"])[col].agg(["mean", "size"]).reset_index()
        rules[name] = {
            "n_contexts": int(len(g)),
            "n_contexts_all_or_none": int(((g["mean"] == 0.0) | (g["mean"] == 1.0)).sum()),
            "n_rows_retained": int(scored[col].sum()),
            "n_rows": int(len(scored)),
            "per_context": [
                {"source": str(r.source), "source_context": str(r.source_context), "n": int(r["size"]),
                 "retention_rate": float(r["mean"])}
                for _, r in g.iterrows()
            ],
        }
    rc = pd.read_csv(work / "expanded_artifacts_canonical_run2" / "applicability_random_controls.csv")
    sel = rc[(rc["method"] == "knn_5") & (rc["training_quantile"].astype(str) == "0.95")]
    perm = {}
    for _, r in sel.iterrows():
        perm[str(r["control_kind"])] = {
            "statistic": str(r["statistic"]),
            "observed_value": float(r["observed_value"]),
            "random_mean_risk": float(r["random_mean_risk"]),
            "one_sided_fraction_as_good_or_better": float(r["one_sided_fraction_as_good_or_better"]),
            "permutations": int(r["permutations"]),
            "bh_adjusted_p_within_family": float(r["bh_adjusted_p_within_family"]),
        }
    key = "within_source_context_physical_cluster_mask_permutation"
    invariant = abs(perm[key]["random_mean_risk"] - perm[key]["observed_value"]) < 1e-12
    return {
        "rules": rules,
        "knn_5_q95_permutation_controls": perm,
        "within_source_context_statistic_invariant": bool(invariant),
        "note": "when every source--geometry context is retained all-or-none, permuting scores within contexts "
                "leaves the retained row set unchanged, so the permutation statistic equals its observed value "
                "in every draw and the one-sided fraction is 1.0 by construction",
    }


# ----------------------------------------------------------------------------- 5. V2 held-out interval widths
def v2_interval_widths(work: Path) -> dict:
    pr = pd.read_csv(work / "v2_artifacts_canonical_run2" / "per_row_predictions_all_seeds.csv")
    out = {}
    for model, part in pr.groupby("model"):
        width_um = (part["interval_upper_mm"] - part["interval_lower_mm"]).to_numpy(float) * 1000.0
        if np.isfinite(width_um).sum() == 0:
            continue
        out[str(model)] = {
            "n_rows_all_seeds": int(np.isfinite(width_um).sum()),
            "mean_width_um": float(np.nanmean(width_um)),
            "median_width_um": float(np.nanmedian(width_um)),
        }
    return {
        "note": "LOCO out-of-fold heuristic intervals of the frozen V2 suite on held-out rows, all seeds pooled; "
                "models without an interval are omitted. Context: the locked retrieval rule answers a query only if "
                "the point prediction lies within the tolerance and the interval lies entirely inside the tolerance band.",
        "tolerance_band_full_width_um_absolute_rule": 20.0,
        "models": out,
    }


# ----------------------------------------------------------------------------- 6. W4 native interval widths by cell
def w4_native_widths(work: Path) -> dict:
    pp = pd.read_csv(work / "w4_artifacts_canonical_run2" / "point_predictions.csv")
    pp = pp[pp["cell"].isin(["D1_random", "D1_external"])].copy()
    pp["native_width_um"] = pp["native_upper_um"] - pp["native_lower_um"]
    out: dict = {}
    for (model, cell), part in pp.groupby(["model", "cell"]):
        out.setdefault(str(model), {})[str(cell)] = {
            "n_rows_all_repeats": int(len(part)),
            "mean_native_width_um": float(part["native_width_um"].mean()),
            "median_native_width_um": float(part["native_width_um"].median()),
            "mean_uq_abs": float(part["uq_abs"].mean()),
        }
    return {"note": "native (uncalibrated) interval widths of the W4 families on the D1 random reference test rows "
                    "(15 repeats pooled) and on the 64 external rows", "models": out}


# ----------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    ws = load_workspace(args.work)
    quartiles, ranges = device_type_blocks(args.work)
    payload = {
        "schema_version": "posthoc-round4-mechanism-checks-v1",
        "status": "retrospective; NOT part of any frozen protocol; recomputed from canonical artifacts and the frozen V2 code; "
                  "no model refitted, no frozen file modified",
        "anchor_map": anchor_map(args.work, ws.v2),
        "device_type_quartiles": quartiles,
        "prediction_ranges": ranges,
        "within_context_retention": within_context_retention(args.work),
        "v2_interval_widths": v2_interval_widths(args.work),
        "w4_native_widths": w4_native_widths(args.work),
    }
    write_json(out / "mechanism_checks.json", payload)
    inputs = [
        "v2_artifacts_canonical_run2/locked_physics_sensitivities.csv",
        "w3_artifacts_canonical_run2/point_predictions.csv",
        "expanded_artifacts_canonical_run2/applicability_scored.csv",
        "expanded_artifacts_canonical_run2/applicability_random_controls.csv",
        "v2_artifacts_canonical_run2/per_row_predictions_all_seeds.csv",
        "w4_artifacts_canonical_run2/point_predictions.csv",
        "src/native_physics_v2.py",
    ]
    write_json(out / "environment.json", {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "input_sha256": {name: sha256_file(args.work / name) for name in inputs},
    })
    manifest = {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.name != "artifact_hashes.json"}
    write_json(out / "artifact_hashes.json", manifest)
    print(json.dumps({k: v for k, v in payload["anchor_map"].items() if k.startswith(("n_", "valid", "invalid", "anchor", "saturated"))}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
