"""W4 orchestration — uncertainty-aware comparison models on the W3 cells.

  python -m w3.run_w4 run --work <work/microfluidic_reliability> --output-dir <dir> [--fast]
  python -m w3.run_w4 compare --a <run1> --b <run2>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .applicability import ALL_SCORES, CONTROL_METHODS, score_applicability
from .cells import Settings, make_cells, split_fit_calibration
from .datasets import load_d1, load_d2
from .evaluate import _spearman, aurc_fast, interval_row, point_rows
from .run_w3 import _environment, _hash_artifacts, _write_csv, compare_runs
from .uq_models import UQ_MODELS, fit_uq, predict_uq
from .workspace import load_workspace, sha256_file

PROTOCOL_PATH = Path(__file__).resolve().parent.parent / "W4_PROTOCOL.md"
INTERVAL_METHODS = ("native", "split_conformal", "conformalised_native")
UQ_SCORES = ("uq_rel", "uq_abs")
SURVIVAL_COVERAGE = 0.80
REFERENCE_FAMILY = {"D1": "D1_random", "D2": "D2_random"}


def _conformal_adjustments(finite_quantile, cal_y: np.ndarray, cal_pred, coverage: float) -> dict[str, float]:
    residual = np.abs(cal_y - cal_pred.point)
    conformity = np.maximum(cal_pred.lower - cal_y, cal_y - cal_pred.upper)
    return {
        "split_conformal": float(finite_quantile(residual, coverage)),
        "conformalised_native": float(finite_quantile(conformity, coverage)),
    }


def _interval_bounds(method: str, pred, adjustments: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    if method == "native":
        return pred.lower, pred.upper
    if method == "split_conformal":
        radius = adjustments["split_conformal"]
        return pred.point - radius, pred.point + radius
    if method == "conformalised_native":
        delta = adjustments["conformalised_native"]
        lower, upper = pred.lower - delta, pred.upper + delta
        return np.minimum(lower, upper), np.maximum(lower, upper)
    raise ValueError(f"unknown interval method {method}")


def run(work: Path, output_dir: Path, fast: bool) -> None:
    started = time.time()
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace = load_workspace(work)
    write_json = workspace.strict_json.write_strict_json
    finite_quantile = workspace.intervals.finite_sample_quantile
    settings = Settings.for_mode(fast)
    protocol_sha = sha256_file(PROTOCOL_PATH)

    d1_comp, d1_ext = load_d1(workspace)
    d2 = load_d2(workspace)
    cells = make_cells(d1_comp, d1_ext, d2, settings)

    write_json(output_dir / "configuration.json", {
        "schema_version": "w4-uncertainty-v0",
        "protocol_file": PROTOCOL_PATH.name,
        "protocol_sha256": protocol_sha,
        "run_mode": "fast_non_inferential" if fast else "locked_full_study",
        "settings": settings.__dict__,
        "uq_models": list(UQ_MODELS),
        "interval_methods": list(INTERVAL_METHODS),
        "uncertainty_scores": list(UQ_SCORES),
        "comparator_scores": list(ALL_SCORES),
        "survival_coverage_rule": SURVIVAL_COVERAGE,
        "cell_families": sorted({c.family for c in cells}),
        "n_cells": len(cells),
    })
    write_json(output_dir / "environment.json", _environment())
    write_json(output_dir / "data_audit.json", {"D1": d1_comp.audit, "D2": d2.audit})

    cell_records, point_pred, interval_rows_out = [], [], []
    summary_point, summary_interval, summary_scores = [], [], []
    deviations: list[dict] = []

    for cell in cells:
        seed = cell.repeat
        integrity = cell.assert_integrity()
        fit_idx, cal_idx, rule_used = split_fit_calibration(cell, seed)
        if rule_used != cell.cal_rule:
            deviations.append({"id": "W4-D1", "cell": cell.cell_id, "repeat": cell.repeat,
                               "deviation": f"calibration rule fell back from {cell.cal_rule} to {rule_used}",
                               "reason": "fewer than five dependency groups among the training rows (as W3-D1)"})
        test_groups = cell.test_groups
        actual_um = cell.spec_test.target_um[cell.test_idx]
        row_ids = cell.spec_test.row_id[cell.test_idx]
        base = {"cell": cell.cell_id, "family": cell.family, "shift_type": cell.shift_type,
                "repeat": cell.repeat, "status_note": cell.status_note}
        cell_records.append({**base, **integrity, "n_fit": int(len(fit_idx)), "n_calibration": int(len(cal_idx)),
                             "calibration_rule_protocol": cell.cal_rule, "calibration_rule_used": rule_used,
                             "test_group_key": cell.test_group_key})
        meta = {"calibration_unit": "row", "n_calibration_units": int(len(cal_idx)),
                "nominal_finite_threshold_attainable": bool(int(np.ceil((len(cal_idx) + 1) * settings.nominal_coverage)) <= len(cal_idx)),
                "finite_threshold_coverage_ceiling": float(len(cal_idx) / (len(cal_idx) + 1))}

        bundle = score_applicability(cell.spec_train, fit_idx, cal_idx, cell.spec_test,
                                     cell.test_idx, seed, settings.threshold_quantile, workspace)
        cal_y = cell.spec_train.response[cal_idx]

        for model_name in UQ_MODELS:
            fit = fit_uq(model_name, cell.spec_train, fit_idx, seed)
            if model_name == "gp" and fit.extra.get("convergence_warnings"):
                deviations.append({"id": "W4-C1", "cell": cell.cell_id, "repeat": cell.repeat,
                                   "note": f"GP optimiser emitted {fit.extra['convergence_warnings']} convergence warning(s); fit kept"})
            cal_pred = predict_uq(fit, cell.spec_train, cal_idx)
            adjustments = _conformal_adjustments(finite_quantile, cal_y, cal_pred, settings.nominal_coverage)
            pred = predict_uq(fit, cell.spec_test, cell.test_idx)
            pred_um = cell.spec_test.to_um(pred.point, cell.test_idx)
            mbase = {**base, "model": model_name}
            uq_abs = pred.score
            uq_rel = pred.score / np.maximum(np.abs(pred.point), 1e-12)
            point_pred.append(pd.DataFrame({**mbase, "row_id": row_ids, "group": test_groups,
                                            "actual_um": actual_um, "pred_um": pred_um,
                                            "native_lower_um": cell.spec_test.to_um(pred.lower, cell.test_idx),
                                            "native_upper_um": cell.spec_test.to_um(pred.upper, cell.test_idx),
                                            "uq_abs": uq_abs, "uq_rel": uq_rel}))
            summary_point.extend(point_rows({**mbase, **{f"extra_{k}": json.dumps(v) for k, v in fit.extra.items()}},
                                            actual_um, pred_um, test_groups))
            for method in INTERVAL_METHODS:
                lower, upper = _interval_bounds(method, pred, adjustments)
                lower_um, upper_um = cell.spec_test.to_um(lower, cell.test_idx), cell.spec_test.to_um(upper, cell.test_idx)
                ibase = {**mbase, "interval_method": method}
                interval_rows_out.append(pd.DataFrame({**ibase, "row_id": row_ids, "actual_um": actual_um,
                                                       "lower_um": lower_um, "upper_um": upper_um,
                                                       "covered": (actual_um >= lower_um) & (actual_um <= upper_um)}))
                row = interval_row(ibase, actual_um, lower_um, upper_um, test_groups, meta, settings.nominal_coverage)
                row["adjustment_response_units"] = adjustments.get(method, 0.0)
                row["survives_0_80"] = bool(row["empirical_coverage"] >= SURVIVAL_COVERAGE)
                summary_interval.append(row)

            rel = np.abs(pred_um - actual_um) / actual_um
            scores = {"uq_rel": uq_rel, "uq_abs": uq_abs, **{k: bundle.test[k] for k in ALL_SCORES}}
            aurcs = {k: aurc_fast(rel, v) for k, v in scores.items()}
            best_ad = min((k for k in ALL_SCORES if k not in CONTROL_METHODS), key=lambda k: aurcs[k])
            for score_name, vector in scores.items():
                summary_scores.append({**mbase, "score": score_name,
                                       "is_control": score_name in CONTROL_METHODS,
                                       "is_model_uncertainty": score_name in UQ_SCORES,
                                       "n": int(len(rel)), "aurc_relative": aurcs[score_name],
                                       "spearman_rho_relative": _spearman(rel, vector),
                                       "full_pool_relative_risk": float(rel.mean()),
                                       "uq_rel_beats_random": bool(aurcs["uq_rel"] < aurcs["random_score"]),
                                       "uq_rel_beats_size_only": bool(aurcs["uq_rel"] < aurcs["size_only"]),
                                       "uq_rel_beats_best_ad": bool(aurcs["uq_rel"] < aurcs[best_ad]),
                                       "best_ad_score": best_ad})
        print(f"[w4] {cell.cell_id} repeat={cell.repeat} done ({time.time() - started:.0f}s)", flush=True)

    _write_csv(output_dir / "cells.csv", pd.DataFrame(cell_records))
    _write_csv(output_dir / "point_predictions.csv", pd.concat(point_pred, ignore_index=True))
    _write_csv(output_dir / "interval_rows.csv", pd.concat(interval_rows_out, ignore_index=True))
    _write_csv(output_dir / "summary_point.csv", pd.DataFrame(summary_point))
    _write_csv(output_dir / "summary_interval.csv", pd.DataFrame(summary_interval))
    _write_csv(output_dir / "summary_uncertainty.csv", pd.DataFrame(summary_scores))
    write_json(output_dir / "deviations.json", deviations)
    write_json(output_dir / "w4_summary.json",
               w4_summary(pd.DataFrame(summary_point), pd.DataFrame(summary_interval), pd.DataFrame(summary_scores), fast))
    write_json(output_dir / "artifact_hashes.json", _hash_artifacts(output_dir))
    print(f"[w4] finished in {time.time() - started:.0f}s -> {output_dir}", flush=True)


def w4_summary(points: pd.DataFrame, intervals: pd.DataFrame, scores: pd.DataFrame, fast: bool) -> dict:
    """Protocol §8 — family × model summary with the §7 rule verdicts."""

    def mean_of(frame: pd.DataFrame, mask, column: str) -> float | None:
        values = frame.loc[mask, column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        return float(values.mean()) if len(values) else None

    overall = points["scope"] == "overall"
    families = sorted(points["family"].unique())
    out: dict = {"run_mode": "fast_non_inferential" if fast else "locked_full_study", "models": list(UQ_MODELS),
                 "survival_coverage_rule": SURVIVAL_COVERAGE, "cells": {}}
    for family in families:
        dataset = family.split("_")[0]
        cell: dict = {"dataset": dataset, "shift_type": str(points.loc[points["family"] == family, "shift_type"].iloc[0]),
                      "n_repeats": int(points.loc[overall & (points["family"] == family) & (points["model"] == UQ_MODELS[0])].shape[0]),
                      "models": {}}
        for model in UQ_MODELS:
            pm = overall & (points["family"] == family) & (points["model"] == model)
            rm = overall & (points["family"] == REFERENCE_FAMILY[dataset]) & (points["model"] == model)
            mape = mean_of(points, pm, "mape_percent")
            ref = mean_of(points, rm, "mape_percent")
            entry: dict = {"mape_percent": mape, "mape_ratio_vs_reference": (mape / ref) if (mape is not None and ref) else None,
                           "intervals": {}, "scores": {}}
            for method in INTERVAL_METHODS:
                im = (intervals["family"] == family) & (intervals["model"] == model) & (intervals["interval_method"] == method)
                cov = mean_of(intervals, im, "empirical_coverage")
                entry["intervals"][method] = {
                    "coverage": cov, "coverage_gap": mean_of(intervals, im, "coverage_gap"),
                    "mean_width_um": mean_of(intervals, im, "mean_width_um"),
                    "fraction_of_repeats_surviving_0_80": mean_of(intervals.assign(s=intervals["survives_0_80"].astype(float)), im, "s"),
                    "survives_0_80_on_mean_coverage": (cov is not None and cov >= SURVIVAL_COVERAGE),
                }
            sm = (scores["family"] == family) & (scores["model"] == model)
            for score in ("uq_rel", "uq_abs", "random_score", "size_only"):
                entry["scores"][score] = mean_of(scores, sm & (scores["score"] == score), "aurc_relative")
            ad_only = sm & ~scores["is_control"] & ~scores["is_model_uncertainty"]
            best = None
            for name in sorted(scores.loc[ad_only, "score"].unique()):
                value = mean_of(scores, sm & (scores["score"] == name), "aurc_relative")
                if value is not None and (best is None or value < best[1]):
                    best = (str(name), value)
            entry["scores"]["best_applicability_score"] = {"name": best[0], "aurc_relative": best[1]} if best else None
            uq = sm & (scores["score"] == "uq_rel")
            entry["rule3_uq_rel_useful_fraction_of_repeats"] = mean_of(
                scores.assign(u=(scores["uq_rel_beats_random"] & scores["uq_rel_beats_size_only"]).astype(float)), uq, "u")
            entry["uq_rel_beats_best_ad_fraction_of_repeats"] = mean_of(
                scores.assign(u=scores["uq_rel_beats_best_ad"].astype(float)), uq, "u")
            cell["models"][model] = entry
        out["cells"][family] = cell
    out["claim_boundary"] = ("Retrospective, exploratory; same cells as W3; no sealed holdout; TX100 fold post-reveal "
                             "retrospective; rule verdicts are descriptive; no candidate-level permutation test.")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--work", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--fast", action="store_true")
    cmp_parser = sub.add_parser("compare")
    cmp_parser.add_argument("--a", type=Path, required=True)
    cmp_parser.add_argument("--b", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args.work, args.output_dir, args.fast)
        return 0
    result = compare_runs(args.a, args.b)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
