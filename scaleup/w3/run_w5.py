"""W5 orchestration — covariate-shift-weighted split conformal on the W3 cells.

  python -m w3.run_w5 run --work <work/microfluidic_reliability> --output-dir <dir> [--fast]
  python -m w3.run_w5 compare --a <run1> --b <run2>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .applicability import score_applicability
from .cells import Settings, make_cells, split_fit_calibration
from .datasets import load_d1, load_d2
from .models import fit_point, predict_response
from .run_w3 import _environment, _hash_artifacts, _write_csv, compare_runs
from .weighted_conformal import domain_odds, effective_sample_size, knn_density_ratio, weighted_radii
from .workspace import load_workspace, sha256_file

PROTOCOL_PATH = Path(__file__).resolve().parent.parent / "W5_PROTOCOL.md"
WEIGHTINGS = ("unweighted", "w_domain", "w_knn")
POINT_MODEL = "xgboost"
SURVIVAL_COVERAGE = 0.80


def run(work: Path, output_dir: Path, fast: bool) -> None:
    started = time.time()
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace = load_workspace(work)
    write_json = workspace.strict_json.write_strict_json
    settings = Settings.for_mode(fast)
    protocol_sha = sha256_file(PROTOCOL_PATH)
    coverage = settings.nominal_coverage

    d1_comp, d1_ext = load_d1(workspace)
    d2 = load_d2(workspace)
    cells = make_cells(d1_comp, d1_ext, d2, settings)

    write_json(output_dir / "configuration.json", {
        "schema_version": "w5-weighted-conformal-v0",
        "protocol_file": PROTOCOL_PATH.name,
        "protocol_sha256": protocol_sha,
        "run_mode": "fast_non_inferential" if fast else "locked_full_study",
        "settings": settings.__dict__,
        "point_model": POINT_MODEL,
        "weightings": list(WEIGHTINGS),
        "survival_coverage_rule": SURVIVAL_COVERAGE,
        "cell_families": sorted({c.family for c in cells}),
        "n_cells": len(cells),
    })
    write_json(output_dir / "environment.json", _environment())

    rows_out, summary, cell_records, deviations = [], [], [], []
    for cell in cells:
        seed = cell.repeat
        integrity = cell.assert_integrity()
        fit_idx, cal_idx, rule_used = split_fit_calibration(cell, seed)
        if rule_used != cell.cal_rule:
            deviations.append({"id": "W5-D1", "cell": cell.cell_id, "repeat": cell.repeat,
                               "deviation": f"calibration rule fell back from {cell.cal_rule} to {rule_used}",
                               "reason": "fewer than five dependency groups among the training rows (as W3-D1)"})
        base = {"cell": cell.cell_id, "family": cell.family, "shift_type": cell.shift_type,
                "repeat": cell.repeat, "status_note": cell.status_note}
        cell_records.append({**base, **integrity, "n_fit": int(len(fit_idx)), "n_calibration": int(len(cal_idx)),
                             "calibration_rule_used": rule_used})

        point = fit_point(POINT_MODEL, cell.spec_train, fit_idx, seed, workspace)
        cal_y = cell.spec_train.response[cal_idx]
        residual = np.abs(cal_y - predict_response(point, cell.spec_train, cal_idx))
        center = predict_response(point, cell.spec_test, cell.test_idx)
        actual_um = cell.spec_test.target_um[cell.test_idx]
        row_ids = cell.spec_test.row_id[cell.test_idx]
        n_features = len(cell.spec_train.feature_cols)

        bundle = score_applicability(cell.spec_train, fit_idx, cal_idx, cell.spec_test,
                                     cell.test_idx, seed, settings.threshold_quantile, workspace)
        weights = {
            "unweighted": (np.ones(len(cal_idx)), np.ones(len(cell.test_idx))),
            "w_domain": (domain_odds(bundle.calibration["domain_classifier"]), domain_odds(bundle.test["domain_classifier"])),
            "w_knn": (knn_density_ratio(bundle.calibration["density_ratio_knn"], n_features, len(fit_idx), len(cell.test_idx)),
                      knn_density_ratio(bundle.test["density_ratio_knn"], n_features, len(fit_idx), len(cell.test_idx))),
        }
        for name, (w_cal, w_test) in weights.items():
            radii = weighted_radii(residual, w_cal, w_test, coverage)
            finite = np.isfinite(radii)
            lower = center - np.where(finite, radii, 0.0)
            upper = center + np.where(finite, radii, 0.0)
            lower_um = cell.spec_test.to_um(lower, cell.test_idx)
            upper_um = cell.spec_test.to_um(upper, cell.test_idx)
            covered = np.where(finite, (actual_um >= lower_um) & (actual_um <= upper_um), True)
            width_um = upper_um - lower_um
            ess_ratio = effective_sample_size(w_cal) / len(w_cal)
            wbase = {**base, "model": POINT_MODEL, "weighting": name}
            rows_out.append(pd.DataFrame({**wbase, "row_id": row_ids, "actual_um": actual_um, "pred_um": cell.spec_test.to_um(center, cell.test_idx),
                                          "radius_response": np.where(finite, radii, np.inf),
                                          "lower_um": np.where(finite, lower_um, -np.inf), "upper_um": np.where(finite, upper_um, np.inf),
                                          "covered": covered, "is_infinite": ~finite, "test_weight": w_test}))
            summary.append({**wbase, "nominal_coverage": coverage, "n": int(len(actual_um)),
                            "empirical_coverage": float(covered.mean()),
                            "coverage_gap": float(covered.mean() - coverage),
                            "fraction_infinite": float((~finite).mean()),
                            "finite_only_coverage": float(covered[finite].mean()) if finite.any() else None,
                            "finite_mean_width_um": float(width_um[finite].mean()) if finite.any() else None,
                            "ess_ratio_calibration": float(ess_ratio),
                            "mean_log10_test_weight": float(np.mean(np.log10(np.maximum(w_test, 1e-300)))),
                            "survives_0_80": bool(covered.mean() >= SURVIVAL_COVERAGE)})
        print(f"[w5] {cell.cell_id} repeat={cell.repeat} done ({time.time() - started:.0f}s)", flush=True)

    _write_csv(output_dir / "cells.csv", pd.DataFrame(cell_records))
    _write_csv(output_dir / "interval_rows.csv", pd.concat(rows_out, ignore_index=True))
    _write_csv(output_dir / "summary_weighted.csv", pd.DataFrame(summary))
    write_json(output_dir / "deviations.json", deviations)
    write_json(output_dir / "w5_summary.json", w5_summary(pd.DataFrame(summary), fast))
    write_json(output_dir / "artifact_hashes.json", _hash_artifacts(output_dir))
    print(f"[w5] finished in {time.time() - started:.0f}s -> {output_dir}", flush=True)


def w5_summary(frame: pd.DataFrame, fast: bool) -> dict:
    def mean_of(mask, column):
        values = pd.to_numeric(frame.loc[mask, column], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        return float(values.mean()) if len(values) else None

    out = {"run_mode": "fast_non_inferential" if fast else "locked_full_study", "point_model": POINT_MODEL,
           "weightings": list(WEIGHTINGS), "survival_coverage_rule": SURVIVAL_COVERAGE, "cells": {}}
    for family in sorted(frame["family"].unique()):
        fm = frame["family"] == family
        cell = {"shift_type": str(frame.loc[fm, "shift_type"].iloc[0]),
                "n_repeats": int(frame.loc[fm & (frame["weighting"] == "unweighted")].shape[0]), "weightings": {}}
        for name in WEIGHTINGS:
            m = fm & (frame["weighting"] == name)
            cov = mean_of(m, "empirical_coverage")
            cell["weightings"][name] = {
                "coverage": cov, "coverage_gap": mean_of(m, "coverage_gap"),
                "fraction_infinite": mean_of(m, "fraction_infinite"),
                "finite_only_coverage": mean_of(m, "finite_only_coverage"),
                "finite_mean_width_um": mean_of(m, "finite_mean_width_um"),
                "ess_ratio_calibration": mean_of(m, "ess_ratio_calibration"),
                "mean_log10_test_weight": mean_of(m, "mean_log10_test_weight"),
                "fraction_of_repeats_surviving_0_80": mean_of(m, "survives_0_80") if frame.loc[m, "survives_0_80"].dtype != object else float(frame.loc[m, "survives_0_80"].astype(bool).mean()),
                "survives_0_80_on_mean_coverage": (cov is not None and cov >= SURVIVAL_COVERAGE),
            }
        base_cov = cell["weightings"]["unweighted"]["coverage"]
        for name in ("w_domain", "w_knn"):
            c = cell["weightings"][name]["coverage"]
            cell["weightings"][name]["coverage_change_vs_unweighted"] = (c - base_cov) if (c is not None and base_cov is not None) else None
        out["cells"][family] = cell
    out["claim_boundary"] = ("Retrospective; same cells as W3; weights from inputs only; infinite intervals counted as covering "
                             "and reported separately; no sealed holdout; TX100 fold post-reveal retrospective.")
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
