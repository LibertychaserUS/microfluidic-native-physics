"""W3 orchestration — design D6/D7. Usage:

  python -m w3.run_w3 run --work <work/microfluidic_reliability> --output-dir <dir> [--fast]
  python -m w3.run_w3 compare --a <run1> --b <run2>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .applicability import ALL_SCORES, global_threshold_mask, score_applicability, within_group_median_mask
from .cells import Cell, Settings, make_cells, split_fit_calibration
from .datasets import load_d1, load_d2
from .evaluate import ad_rows, candidate_level_rows, interval_row, point_rows
from .intervals import INTERVAL_METHODS, fit_interval
from .models import POINT_MODELS, fit_point, predict_um
from .workspace import Workspace, load_workspace, sha256_file

PROTOCOL_PATH = Path(__file__).resolve().parent.parent / "W3_PROTOCOL.md"
CANDIDATE_LEVEL_REFERENCE_MODEL = "xgboost"


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")


def _environment() -> dict:
    import scipy
    import sklearn
    import xgboost

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    }


def _hash_artifacts(output_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(output_dir)): sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }


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

    d1_comp, d1_ext = load_d1(workspace)
    d2 = load_d2(workspace)
    cells = make_cells(d1_comp, d1_ext, d2, settings)

    write_json(output_dir / "configuration.json", {
        "schema_version": "w3-benchmark-v0",
        "protocol_file": PROTOCOL_PATH.name,
        "protocol_sha256": protocol_sha,
        "run_mode": "fast_non_inferential" if fast else "locked_full_study",
        "settings": settings.__dict__,
        "point_models": list(POINT_MODELS),
        "interval_methods": list(INTERVAL_METHODS),
        "applicability_scores": list(ALL_SCORES),
        "candidate_level_reference_model": CANDIDATE_LEVEL_REFERENCE_MODEL,
        "cell_families": sorted({c.family for c in cells}),
        "n_cells": len(cells),
    })
    write_json(output_dir / "environment.json", _environment())
    write_json(output_dir / "data_audit.json", {"D1": d1_comp.audit, "D2": d2.audit})

    cell_records, point_pred, interval_rows_out, ad_score_rows = [], [], [], []
    summary_point, summary_interval, summary_ad, summary_candidate = [], [], [], []
    deviations: list[dict] = [{
        "id": "W3-C1",
        "clarification": "Mondrian calibration groups use the training table's own key "
                         "(D1 geometry dependency group, D2 chemistry); the external table's "
                         "`source` key is undefined on training rows.",
    }]

    for cell in cells:
        seed = cell.repeat
        integrity = cell.assert_integrity()
        fit_idx, cal_idx, rule_used = split_fit_calibration(cell, seed)
        if rule_used != cell.cal_rule:
            deviations.append({
                "id": "W3-D1", "cell": cell.cell_id, "repeat": cell.repeat,
                "deviation": f"calibration rule fell back from {cell.cal_rule} to {rule_used}",
                "reason": "fewer than five dependency groups among the training rows",
            })
        cal_groups = cell.spec_train.groups[cell.cal_group_key][cal_idx]
        test_groups = cell.test_groups
        actual_um = cell.spec_test.target_um[cell.test_idx]
        row_ids = cell.spec_test.row_id[cell.test_idx]
        base = {"cell": cell.cell_id, "family": cell.family, "shift_type": cell.shift_type,
                "repeat": cell.repeat, "status_note": cell.status_note}
        cell_records.append({**base, **integrity, "n_fit": int(len(fit_idx)),
                             "n_calibration": int(len(cal_idx)),
                             "calibration_rule_protocol": cell.cal_rule,
                             "calibration_rule_used": rule_used,
                             "calibration_group_key": cell.cal_group_key,
                             "test_group_key": cell.test_group_key})

        bundle = score_applicability(cell.spec_train, fit_idx, cal_idx, cell.spec_test,
                                     cell.test_idx, seed, settings.threshold_quantile, workspace)
        global_masks = {m: global_threshold_mask(bundle.test[m], bundle.thresholds[m]) for m in ALL_SCORES}
        within_masks = {m: within_group_median_mask(bundle.test[m], test_groups) for m in ALL_SCORES}
        score_frame = pd.DataFrame({**base, "row_id": row_ids, "group": test_groups})
        for m in ALL_SCORES:
            score_frame[f"score__{m}"] = bundle.test[m]
            score_frame[f"threshold__{m}"] = bundle.thresholds[m]
            score_frame[f"accepted__{m}__q0_95"] = global_masks[m]
            score_frame[f"accepted__{m}__within_group_median"] = within_masks[m]
        ad_score_rows.append(score_frame)

        for model_name in POINT_MODELS:
            point = fit_point(model_name, cell.spec_train, fit_idx, seed, workspace)
            pred = predict_um(point, cell.spec_test, cell.test_idx)
            mbase = {**base, "model": model_name}
            point_pred.append(pd.DataFrame({**mbase, "row_id": row_ids, "group": test_groups,
                                            "actual_um": actual_um, "pred_um": pred,
                                            **({"mlp_n_iter": point.extra.get("n_iter")} if model_name == "mlp" else {})}))
            summary_point.extend(point_rows(mbase, actual_um, pred, test_groups))
            summary_ad.extend(ad_rows(mbase, actual_um, pred, test_groups, bundle.test, global_masks, within_masks))
            if model_name == CANDIDATE_LEVEL_REFERENCE_MODEL:
                summary_candidate.extend(candidate_level_rows(mbase, actual_um, pred, test_groups,
                                                              bundle.test, settings.permutations, seed))
            for method in INTERVAL_METHODS:
                fitted = fit_interval(method, point, cell.spec_train, fit_idx, cal_idx, cal_groups,
                                      seed, settings.nominal_coverage, settings.ensemble_members, workspace)
                lower, upper = fitted.predict_um(cell.spec_test, cell.test_idx)
                ibase = {**mbase, "interval_method": method}
                interval_rows_out.append(pd.DataFrame({**ibase, "row_id": row_ids, "group": test_groups,
                                                       "actual_um": actual_um, "lower_um": lower, "upper_um": upper,
                                                       "covered": (actual_um >= lower) & (actual_um <= upper)}))
                summary_interval.append(interval_row(ibase, actual_um, lower, upper, test_groups,
                                                     fitted.meta, settings.nominal_coverage))
        print(f"[w3] {cell.cell_id} repeat={cell.repeat} done ({time.time() - started:.0f}s)", flush=True)

    _write_csv(output_dir / "cells.csv", pd.DataFrame(cell_records))
    _write_csv(output_dir / "point_predictions.csv", pd.concat(point_pred, ignore_index=True))
    _write_csv(output_dir / "interval_rows.csv", pd.concat(interval_rows_out, ignore_index=True))
    _write_csv(output_dir / "ad_scores.csv", pd.concat(ad_score_rows, ignore_index=True))
    _write_csv(output_dir / "summary_point.csv", pd.DataFrame(summary_point))
    _write_csv(output_dir / "summary_interval.csv", pd.DataFrame(summary_interval))
    _write_csv(output_dir / "summary_ad.csv", pd.DataFrame(summary_ad))
    _write_csv(output_dir / "summary_candidate_level.csv", pd.DataFrame(summary_candidate))
    write_json(output_dir / "deviations.json", deviations)
    write_json(output_dir / "matrix_summary.json",
               matrix_summary(pd.DataFrame(summary_point), pd.DataFrame(summary_interval),
                              pd.DataFrame(summary_ad), pd.DataFrame(summary_candidate), fast))
    write_json(output_dir / "artifact_hashes.json", _hash_artifacts(output_dir))
    print(f"[w3] finished in {time.time() - started:.0f}s → {output_dir}", flush=True)


def matrix_summary(points: pd.DataFrame, intervals: pd.DataFrame, ad: pd.DataFrame,
                   candidate: pd.DataFrame, fast: bool) -> dict:
    """D5 §2.5 / protocol §8 — shift type × dataset matrix with explicit empty cells."""

    def mean_over_repeats(frame: pd.DataFrame, mask, column: str) -> float | None:
        values = frame.loc[mask, column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        return float(values.mean()) if len(values) else None

    ref_model = CANDIDATE_LEVEL_REFERENCE_MODEL
    overall = points["scope"] == "overall"
    reference = {
        "D1": mean_over_repeats(points, overall & (points["family"] == "D1_random") & (points["model"] == ref_model), "mape_percent"),
        "D2": mean_over_repeats(points, overall & (points["family"] == "D2_random") & (points["model"] == ref_model), "mape_percent"),
    }
    matrix: dict[str, dict] = {}
    for family in sorted(points["family"].unique()):
        dataset = family.split("_")[0]
        fam_point = overall & (points["family"] == family)
        cell: dict = {"dataset": dataset, "shift_type": str(points.loc[fam_point, "shift_type"].iloc[0]),
                      "n_repeats": int(points.loc[fam_point & (points["model"] == ref_model)].shape[0])}
        for model in POINT_MODELS:
            mape = mean_over_repeats(points, fam_point & (points["model"] == model), "mape_percent")
            cell[f"mape_percent__{model}"] = mape
        ref_mape = cell[f"mape_percent__{ref_model}"]
        cell["mape_doubling_factor_vs_reference"] = (
            float(ref_mape / reference[dataset]) if ref_mape is not None and reference[dataset] else None
        )
        for method in ("split_conformal", "cqr", "mondrian_group_conformal", "ensemble_variance"):
            mask = (intervals["family"] == family) & (intervals["model"] == ref_model) & (intervals["interval_method"] == method)
            cell[f"coverage_gap_90__{method}"] = mean_over_repeats(intervals, mask, "coverage_gap")
            cell[f"mean_width_um__{method}"] = mean_over_repeats(intervals, mask, "mean_width_um")
        ad_mask = (ad["family"] == family) & (ad["model"] == ref_model)
        random_wg = mean_over_repeats(ad, ad_mask & (ad["ad_method"] == "random_score"), "within_group_aurc_mean")
        best = None
        for method in ad["ad_method"].unique():
            if method in ("random_score", "size_only"):
                continue
            value = mean_over_repeats(ad, ad_mask & (ad["ad_method"] == method), "within_group_aurc_mean")
            if value is not None and (best is None or value < best[1]):
                best = (str(method), value)
        cell["best_within_group_aurc_method"] = best[0] if best else None
        cell["best_within_group_aurc_reduction_vs_random"] = (
            float(random_wg - best[1]) if best and random_wg is not None else None
        )
        knn = ad_mask & (ad["ad_method"] == "knn_5")
        cell["knn5_q95_abs10um_precision"] = mean_over_repeats(ad, knn, "global_q95_abs10um_precision")
        cell["knn5_q95_abs10um_recall"] = mean_over_repeats(ad, knn, "global_q95_abs10um_recall")
        cell["knn5_q95_rel10_precision"] = mean_over_repeats(ad, knn, "global_q95_rel10_precision")
        cell["knn5_q95_rel10_recall"] = mean_over_repeats(ad, knn, "global_q95_rel10_recall")
        verdicts = {}
        if len(candidate):
            cmask = candidate["family"] == family
            for method in sorted(candidate.loc[cmask, "ad_method"].unique()):
                sub = candidate.loc[cmask & (candidate["ad_method"] == method), "verdict"]
                verdicts[str(method)] = {str(k): int(v) for k, v in sub.value_counts().items()}
        cell["candidate_level_verdicts"] = verdicts
        matrix[family] = cell
    return {
        "run_mode": "fast_non_inferential" if fast else "locked_full_study",
        "reference_model": ref_model,
        "reference_mape_percent": reference,
        "cells": matrix,
        "not_in_hand": {
            "cross_laboratory": "D3 (Zenodo 17867478) not downloaded; awaiting user approval",
            "real_vs_simulation": "D5 (DLDNN CFD) not examined",
        },
        "claim_boundary": "Retrospective, exploratory; no sealed holdout; TX100 fold is post-reveal retrospective; no wet-lab or prospective claim.",
    }


def compare_runs(a: Path, b: Path) -> dict:
    """D6 — byte-identical manifests and every file re-hashes to its entry."""

    result = {"a": str(a), "b": str(b)}
    ma, mb = (Path(a) / "artifact_hashes.json"), (Path(b) / "artifact_hashes.json")
    result["manifests_byte_identical"] = ma.read_bytes() == mb.read_bytes()
    mismatches = []
    for root, manifest_path in ((Path(a), ma), (Path(b), mb)):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, expected in manifest.items():
            observed = hashlib.sha256((root / name).read_bytes()).hexdigest()
            if observed != expected:
                mismatches.append({"run": str(root), "file": name, "expected": expected, "observed": observed})
    result["rehash_mismatches"] = mismatches
    result["passed"] = bool(result["manifests_byte_identical"] and not mismatches)
    return result


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
