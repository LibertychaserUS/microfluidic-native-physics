"""Read-only post-reveal diagnostics for the prospective TX100 validation.

This module is intentionally separate from the frozen custodian, prediction,
reveal, and recovery programs.  It never writes beside or over any frozen
artifact.  All outputs are created exclusively in a caller-selected directory.

The uncertainty intervals emitted here are descriptive and conditional on the
87 revealed TX100 rows.  The chemistry-group calibration analysis is explicitly
post hoc and carries no conformal coverage guarantee.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor


ROW_ID = "opaque_row_id"
TARGET = "target_diameter_mm"
MODEL_FEATURES = [
    "continuous_flow_ml_min",
    "dispersed_flow_ml_min",
    "equilibrium_interfacial_tension_mn_m",
    "concentration_over_cmc",
]
SCORE_COLUMNS = {
    "strict_minmax": "minmax_score",
    "knn_q95": "knn5_distance",
    "mahalanobis_q95": "mahalanobis_oas_distance",
    "isolation_forest_q95": "isolation_forest_outlier_score",
}
MASK_COLUMNS = {
    "strict_minmax": "minmax_accept",
    "knn_q95": "knn5_q95_accept",
    "mahalanobis_q95": "mahalanobis_q95_accept",
    "isolation_forest_q95": "isolation_forest_q95_accept",
}
LOCKED_METHOD_ORDER = tuple(SCORE_COLUMNS)
TOP_K_VALUES = (1, 3, 5)
RELIABLE_RELATIVE_ERROR = 0.10
RANDOM_REPETITIONS = 10_000
RANDOM_STATE = 42
BOOTSTRAP_SEED = 20260716


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(child) for child in value]
    if isinstance(value, np.ndarray):
        return [_sanitize(child) for child in value.tolist()]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return None
        if not math.isfinite(number):
            raise ValueError("infinite value cannot enter a post-hoc artifact")
        return number
    if value is pd.NA:
        return None
    return value


def _exclusive_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        json.dump(
            _sanitize(payload),
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")


def _exclusive_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)


def _save_figure_exclusive(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def verify_bundle_hash(bundle_path: Path, hash_path: Path) -> str:
    parts = hash_path.read_text(encoding="utf-8").strip().split()
    if len(parts) != 2 or parts[1] != bundle_path.name:
        raise ValueError("invalid frozen bundle hash sidecar")
    actual = sha256(bundle_path)
    if parts[0].lower() != actual:
        raise ValueError("frozen bundle SHA-256 mismatch")
    return actual


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for one binomial proportion."""

    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Wilson interval requires 0 <= successes <= total and total > 0")
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z**2 / (4.0 * total**2)
        )
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def benjamini_hochberg(p_values: dict[str, float]) -> dict[str, float]:
    """Benjamini-Hochberg adjusted p-values with monotonicity enforcement."""

    if not p_values:
        return {}
    for value in p_values.values():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("p-values must lie in [0, 1]")
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    count = len(ordered)
    adjusted_sorted = [0.0] * count
    running = 1.0
    for reverse_index in range(count - 1, -1, -1):
        rank = reverse_index + 1
        key = ordered[reverse_index]
        candidate = min(1.0, float(p_values[key]) * count / rank)
        running = min(running, candidate)
        adjusted_sorted[reverse_index] = running
    return {key: adjusted_sorted[index] for index, key in enumerate(ordered)}


def correct_matched_budget_p(
    monte_carlo_p: float, retained_count: int, total_count: int
) -> tuple[float, str]:
    """Correct the matched-budget p-value for a degenerate retention budget.

    Retaining the complete pool leaves exactly one possible subset.  Its null
    risk equals the observed full-pool risk with probability one, so the valid
    one-sided p-value is exactly 1.0.  Comparing floating-point means from many
    permutations of the same complete set can spuriously produce a value below
    one; that historical value is retained separately for audit only.
    """

    if total_count <= 0 or not 0 <= retained_count <= total_count:
        raise ValueError("retained count must lie between zero and total count")
    if retained_count in (0, total_count):
        return 1.0, "degenerate null: only one subset exists at this budget"
    return float(monte_carlo_p), "nondegenerate matched-budget Monte Carlo null"


def tie_aware_risk_curve(
    relative_errors: np.ndarray, scores: np.ndarray
) -> tuple[pd.DataFrame, float]:
    """Build a risk-coverage curve only at complete score-tie boundaries.

    The AURC uses a right-step integral over [0, 1].  Every tied score block is
    admitted atomically, so arbitrary row order cannot improve or worsen AURC.
    """

    errors = np.asarray(relative_errors, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if len(errors) == 0 or len(errors) != len(scores):
        raise ValueError("risk and score arrays must have equal non-zero length")
    if not np.isfinite(errors).all() or not np.isfinite(scores).all():
        raise ValueError("risk and score arrays must be finite")
    order = np.argsort(scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered_errors = errors[order]
    boundaries = np.flatnonzero(
        np.r_[ordered_scores[1:] != ordered_scores[:-1], True]
    )
    cumulative = np.cumsum(ordered_errors)
    retained = boundaries + 1
    coverage = retained / len(errors)
    risk = cumulative[boundaries] / retained
    previous = np.r_[0.0, coverage[:-1]]
    aurc = float(np.sum((coverage - previous) * risk))
    curve = pd.DataFrame(
        {
            "score_threshold": ordered_scores[boundaries],
            "retained": retained,
            "coverage": coverage,
            "relative_risk": risk,
            "tie_block_size": np.diff(np.r_[-1, boundaries]),
        }
    )
    return curve, aurc


def point_diagnostics(predictions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    if labels.columns.tolist() != [ROW_ID, TARGET]:
        raise ValueError("sealed-label table does not match the frozen schema")
    for name, frame in (("prediction", predictions), ("label", labels)):
        if frame[ROW_ID].isna().any() or frame[ROW_ID].duplicated().any():
            raise ValueError(f"{name} opaque row IDs are missing or duplicated")
    if set(predictions[ROW_ID].astype(str)) != set(labels[ROW_ID].astype(str)):
        raise ValueError("prediction and label opaque row-ID sets differ")
    merged = predictions.merge(labels, on=ROW_ID, how="left", validate="one_to_one")
    actual = merged[TARGET].to_numpy(dtype=float)
    predicted = merged["predicted_diameter_mm"].to_numpy(dtype=float)
    if not np.isfinite(actual).all() or (actual <= 0).any():
        raise ValueError("revealed targets must be finite and positive")
    merged["signed_error_mm"] = predicted - actual
    merged["absolute_error_mm"] = np.abs(merged["signed_error_mm"])
    merged["signed_relative_error"] = merged["signed_error_mm"] / actual
    merged["absolute_relative_error"] = np.abs(merged["signed_relative_error"])
    merged["squared_error_mm2"] = merged["signed_error_mm"] ** 2
    merged["conformal_covered"] = (
        (actual >= merged["conformal_lower_mm"].to_numpy(dtype=float))
        & (actual <= merged["conformal_upper_mm"].to_numpy(dtype=float))
    )
    merged["reliable_within_10pct"] = (
        merged["absolute_relative_error"] <= RELIABLE_RELATIVE_ERROR
    )
    return merged


def summarize_point(merged: pd.DataFrame) -> dict[str, Any]:
    actual = merged[TARGET].to_numpy(dtype=float)
    predicted = merged["predicted_diameter_mm"].to_numpy(dtype=float)
    signed = predicted - actual
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    return {
        "n": int(len(merged)),
        "mape_percent": float(100.0 * merged["absolute_relative_error"].mean()),
        "mae_mm": float(merged["absolute_error_mm"].mean()),
        "rmse_mm": float(np.sqrt(merged["squared_error_mm2"].mean())),
        "r2": (
            float(1.0 - np.sum(signed**2) / denominator)
            if denominator > 0
            else None
        ),
        "signed_bias_mm_prediction_minus_observation": float(np.mean(signed)),
        "median_signed_error_mm": float(np.median(signed)),
        "signed_relative_bias": float(merged["signed_relative_error"].mean()),
        "overprediction_fraction": float(np.mean(signed > 0)),
        "underprediction_fraction": float(np.mean(signed < 0)),
    }


def conditional_row_uncertainty(
    merged: pd.DataFrame, repetitions: int
) -> dict[str, Any]:
    if repetitions < 100:
        raise ValueError("at least 100 row-bootstrap repetitions are required")
    actual = merged[TARGET].to_numpy(dtype=float)
    predicted = merged["predicted_diameter_mm"].to_numpy(dtype=float)
    covered = merged["conformal_covered"].to_numpy(dtype=bool)
    reliable = merged["reliable_within_10pct"].to_numpy(dtype=bool)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(merged), size=(repetitions, len(merged)))
    actual_sample = actual[indices]
    predicted_sample = predicted[indices]
    signed = predicted_sample - actual_sample
    bootstrap_values = {
        "mape_percent": 100.0 * np.mean(np.abs(signed) / actual_sample, axis=1),
        "mae_mm": np.mean(np.abs(signed), axis=1),
        "signed_bias_mm": np.mean(signed, axis=1),
        "signed_relative_bias": np.mean(signed / actual_sample, axis=1),
        "conformal_coverage": np.mean(covered[indices], axis=1),
        "reliable_fraction": np.mean(reliable[indices], axis=1),
    }
    percentile_intervals = {
        name: {
            "lower_2_5_percent": float(np.quantile(values, 0.025)),
            "upper_97_5_percent": float(np.quantile(values, 0.975)),
        }
        for name, values in bootstrap_values.items()
    }
    coverage_wilson = wilson_interval(int(covered.sum()), len(covered))
    reliable_wilson = wilson_interval(int(reliable.sum()), len(reliable))
    return {
        "scope_label": (
            "Conditional on these 87 revealed TX100 rows; row bootstrap treats rows "
            "as exchangeable and is not a population-level or cross-chemistry interval."
        ),
        "confidence_level": 0.95,
        "row_bootstrap": {
            "repetitions": repetitions,
            "seed": BOOTSTRAP_SEED,
            "percentile_intervals": percentile_intervals,
        },
        "wilson_intervals": {
            "conformal_coverage": {
                "successes": int(covered.sum()),
                "total": int(len(covered)),
                "lower": coverage_wilson[0],
                "upper": coverage_wilson[1],
            },
            "reliable_within_10pct_fraction": {
                "successes": int(reliable.sum()),
                "total": int(len(reliable)),
                "lower": reliable_wilson[0],
                "upper": reliable_wilson[1],
            },
        },
    }


def risk_coverage_analysis(merged: pd.DataFrame) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    errors = merged["absolute_relative_error"].to_numpy(dtype=float)
    summaries: dict[str, Any] = {}
    curves: dict[str, pd.DataFrame] = {}
    for method, score_column in SCORE_COLUMNS.items():
        curve, aurc = tie_aware_risk_curve(
            errors, merged[score_column].to_numpy(dtype=float)
        )
        curves[method] = curve
        summaries[method] = {
            "score_column": score_column,
            "tie_aware_aurc": aurc,
            "unique_score_thresholds": int(len(curve)),
            "largest_tie_block": int(curve["tie_block_size"].max()),
            "curve": curve.to_dict(orient="records"),
        }
    return {
        "risk": "mean absolute relative error",
        "selection_direction": "lower applicability score retained first",
        "tie_rule": "all exactly tied rows enter atomically",
        "aurc_definition": "right-step integral over coverage from zero to one",
        "methods": summaries,
    }, curves


def matched_budget_p_values(
    merged: pd.DataFrame, recovered_result: dict[str, Any]
) -> dict[str, Any]:
    errors = merged["absolute_relative_error"].to_numpy(dtype=float)
    rng = np.random.default_rng(RANDOM_STATE)
    p_values: dict[str, float] = {}
    rows: dict[str, Any] = {}
    for method in LOCKED_METHOD_ORDER:
        mask = merged[MASK_COLUMNS[method]].to_numpy(dtype=bool)
        count = int(mask.sum())
        observed = float(np.mean(errors[mask])) if count else None
        if count:
            random_risks = np.empty(RANDOM_REPETITIONS, dtype=float)
            for repeat in range(RANDOM_REPETITIONS):
                indices = rng.choice(len(errors), size=count, replace=False)
                random_risks[repeat] = float(np.mean(errors[indices]))
            p_value = float(
                (1 + np.sum(random_risks <= observed)) / (RANDOM_REPETITIONS + 1)
            )
        else:
            random_risks = np.array([], dtype=float)
            p_value = 1.0
        recovered_p = recovered_result["applicability"]["rules"][method][
            "matched_budget_random_retention"
        ]["one_sided_p_random_risk_le_observed"]
        corrected_p, correction_reason = correct_matched_budget_p(
            p_value, count, len(errors)
        )
        p_values[method] = corrected_p
        rows[method] = {
            "retained_count": count,
            "observed_relative_risk": observed,
            "independently_reproduced_locked_monte_carlo_p": p_value,
            "locked_recovered_p": recovered_p,
            "scientifically_corrected_unadjusted_p": corrected_p,
            "p_value_correction_reason": correction_reason,
            "absolute_recomputation_difference": (
                abs(p_value - float(recovered_p)) if recovered_p is not None else None
            ),
            "random_risk_mean": (
                float(np.mean(random_risks)) if len(random_risks) else None
            ),
        }
        if count == len(errors):
            rows[method]["locked_buggy_p"] = recovered_p
            rows[method]["corrected_degenerate_null_p"] = corrected_p
    adjusted = benjamini_hochberg(p_values)
    for method in rows:
        rows[method]["bh_adjusted_p_across_four_locked_tests"] = adjusted[method]
        rows[method]["bh_significant_at_0_05"] = bool(adjusted[method] <= 0.05)
    return {
        "family": "four locked applicability matched-budget random-retention tests",
        "direction": "probability random retained risk is no larger than observed",
        "random_repetitions": RANDOM_REPETITIONS,
        "random_state": RANDOM_STATE,
        "correction": "Benjamini-Hochberg applied to scientifically corrected p-values",
        "degenerate_null_policy": (
            "A zero- or full-retention budget has one possible subset and p=1.0. "
            "The frozen strict-minmax p=0.444155... is retained only as an audited "
            "floating-point/permutation artifact."
        ),
        "methods": rows,
    }


def build_locked_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.3,
        random_state=42,
        objective="reg:squarederror",
        n_jobs=1,
    )


def calibration_sensitivity(
    development: pd.DataFrame, merged: pd.DataFrame
) -> dict[str, Any]:
    groups = development["cmc_mm"].to_numpy(dtype=float)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("chemistry sensitivity requires at least two development CMC groups")
    actual_blind = merged[TARGET].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    group_maxima: list[float] = []
    for group in unique_groups:
        held = groups == group
        fit = ~held
        model = build_locked_model()
        model.fit(development.loc[fit, MODEL_FEATURES], development.loc[fit, TARGET])
        calibration_prediction = model.predict(development.loc[held, MODEL_FEATURES])
        residual = np.abs(
            calibration_prediction - development.loc[held, TARGET].to_numpy(dtype=float)
        )
        group_max = float(np.max(residual))
        group_maxima.append(group_max)
        blind_prediction = model.predict(merged[MODEL_FEATURES])
        blind_coverage = float(
            np.mean(np.abs(actual_blind - blind_prediction) <= group_max)
        )
        rows.append(
            {
                "held_out_cmc_mm": float(group),
                "held_out_rows": int(held.sum()),
                "maximum_absolute_residual_mm": group_max,
                "blind_tx100_coverage_using_that_group_max": blind_coverage,
                "interval_width_mm": 2.0 * group_max,
                "blind_point_mape_percent_for_that_loco_model": float(
                    100.0 * np.mean(np.abs(blind_prediction - actual_blind) / actual_blind)
                ),
            }
        )
    full_model = build_locked_model()
    full_model.fit(development[MODEL_FEATURES], development[TARGET])
    full_prediction = full_model.predict(merged[MODEL_FEATURES])
    frozen_prediction = merged["predicted_diameter_mm"].to_numpy(dtype=float)
    aggregate_radius = float(np.max(group_maxima))
    aggregate_coverage = float(
        np.mean(np.abs(actual_blind - full_prediction) <= aggregate_radius)
    )
    return {
        "status": "exploratory post hoc sensitivity analysis",
        "guarantee": (
            "None. TX100 labels were already revealed; six group maxima are too few "
            "for a useful 90% group-level finite-sample conformal quantile."
        ),
        "method": (
            "Leave one development CMC group out, fit the locked XGBoost declaration "
            "on all other groups, and use the maximum absolute residual in the held-out "
            "group as a radius."
        ),
        "development_cmc_groups": int(len(unique_groups)),
        "held_out_group_sensitivity": rows,
        "aggregate_max_of_group_maxima_radius_mm": aggregate_radius,
        "aggregate_interval_width_mm": 2.0 * aggregate_radius,
        "aggregate_tx100_coverage_around_full_development_model": aggregate_coverage,
        "full_model_max_absolute_difference_from_frozen_prediction_mm": float(
            np.max(np.abs(full_prediction - frozen_prediction))
        ),
    }


def target_search_comparison(
    bundle: dict[str, Any], merged: pd.DataFrame
) -> dict[str, Any]:
    queries = pd.DataFrame(bundle["queries"])
    rankings = pd.DataFrame(bundle["rankings"])
    label_by_id = merged.set_index(ROW_ID)[TARGET].to_dict()
    records: list[dict[str, Any]] = []
    for query in queries.to_dict(orient="records"):
        strategy = str(query["strategy"])
        target = float(query["target_mm"])
        selected = rankings[
            (rankings["strategy"] == strategy)
            & np.isclose(rankings["target_mm"].to_numpy(dtype=float), target)
        ].sort_values("rank")
        selected_ids = selected[ROW_ID].astype(str).tolist() if len(selected) else []
        selected_errors = np.array(
            [abs(float(label_by_id[row_id]) - target) / target for row_id in selected_ids],
            dtype=float,
        )
        row: dict[str, Any] = {
            "strategy": strategy,
            "target_mm": target,
            "answered": bool(query["answered"]),
            "top1_opaque_row_id": selected_ids[0] if selected_ids else None,
            "top1_actual_relative_target_error": (
                float(selected_errors[0]) if len(selected_errors) else None
            ),
        }
        for top_k in TOP_K_VALUES:
            row[f"success_at_{top_k}"] = bool(
                len(selected_errors)
                and np.any(selected_errors[:top_k] <= RELIABLE_RELATIVE_ERROR)
            )
        records.append(row)
    per_query = pd.DataFrame(records)
    strategies = sorted(per_query["strategy"].unique())
    aggregate: dict[str, Any] = {}
    for strategy, group in per_query.groupby("strategy", sort=True):
        aggregate[str(strategy)] = {
            "queries": int(len(group)),
            "answered": int(group["answered"].sum()),
            "answered_coverage": float(group["answered"].mean()),
            **{
                f"success_at_{top_k}_all_queries": float(
                    group[f"success_at_{top_k}"].mean()
                )
                for top_k in TOP_K_VALUES
            },
        }

    pairwise: list[dict[str, Any]] = []
    for left, right in itertools.combinations_with_replacement(strategies, 2):
        left_frame = per_query[per_query["strategy"] == left].set_index("target_mm")
        right_frame = per_query[per_query["strategy"] == right].set_index("target_mm")
        common = left_frame.index.intersection(right_frame.index)
        left_frame = left_frame.loc[common]
        right_frame = right_frame.loc[common]
        both_answered = left_frame["answered"].astype(bool) & right_frame[
            "answered"
        ].astype(bool)
        joint_unanswered = (~left_frame["answered"].astype(bool)) & (
            ~right_frame["answered"].astype(bool)
        )
        same_selection = (
            left_frame["top1_opaque_row_id"].fillna("__UNANSWERED__").to_numpy()
            == right_frame["top1_opaque_row_id"].fillna("__UNANSWERED__").to_numpy()
        )
        same_answered_selection = same_selection & both_answered.to_numpy()
        outcome_same = (
            left_frame["success_at_1"].to_numpy(dtype=bool)
            == right_frame["success_at_1"].to_numpy(dtype=bool)
        )
        pairwise.append(
            {
                "left": left,
                "right": right,
                "targets": int(len(common)),
                "joint_unanswered_targets": int(joint_unanswered.sum()),
                "both_answered_targets": int(both_answered.sum()),
                "identical_top1_including_joint_unanswered": int(same_selection.sum()),
                "nonidentical_top1_including_answer_disagreement": int(
                    len(common) - same_selection.sum()
                ),
                "identical_top1_among_both_answered": int(
                    same_answered_selection.sum()
                ),
                "identical_top1_fraction_among_both_answered": (
                    float(same_answered_selection.sum() / both_answered.sum())
                    if both_answered.any()
                    else None
                ),
                "identical_success_at_1_outcome": int(outcome_same.sum()),
                "nonidentical_success_at_1_outcome": int(
                    len(common) - outcome_same.sum()
                ),
            }
        )
    return {
        "selection_source": "frozen pre-reveal rankings; no reranking",
        "outcome_definition": "actual selected diameter within 10% of fixed target",
        "aggregate": aggregate,
        "pairwise_selection_and_outcome_comparison": pairwise,
        "per_query": records,
    }


def recovered_agreement(
    point: dict[str, Any], merged: pd.DataFrame, recovered: dict[str, Any]
) -> dict[str, Any]:
    comparisons = {
        "mape_percent": (
            point["mape_percent"], recovered["point_prediction"]["mape_percent"]
        ),
        "mae_mm": (point["mae_mm"], recovered["point_prediction"]["mae_mm"]),
        "rmse_mm": (point["rmse_mm"], recovered["point_prediction"]["rmse_mm"]),
        "r2": (point["r2"], recovered["point_prediction"]["r2"]),
        "conformal_coverage": (
            float(merged["conformal_covered"].mean()),
            recovered["conformal"]["observed_coverage"],
        ),
        "mean_interval_width_mm": (
            float(
                np.mean(
                    merged["conformal_upper_mm"] - merged["conformal_lower_mm"]
                )
            ),
            recovered["conformal"]["mean_interval_width_mm"],
        ),
    }
    details = {
        key: {
            "independent": left,
            "recovered": right,
            "absolute_difference": abs(float(left) - float(right)),
        }
        for key, (left, right) in comparisons.items()
    }
    return {
        "metrics": details,
        "all_absolute_differences_below_1e_12": bool(
            all(row["absolute_difference"] < 1e-12 for row in details.values())
        ),
    }


def plot_prediction_diagnostics(merged: pd.DataFrame, path: Path) -> None:
    actual = merged[TARGET].to_numpy(dtype=float)
    predicted = merged["predicted_diameter_mm"].to_numpy(dtype=float)
    signed = merged["signed_error_mm"].to_numpy(dtype=float)
    lower = min(float(actual.min()), float(predicted.min()))
    upper = max(float(actual.max()), float(predicted.max()))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    scatter = axes[0].scatter(
        actual,
        predicted,
        c=merged["absolute_relative_error"],
        cmap="viridis",
        s=34,
        edgecolor="white",
        linewidth=0.35,
    )
    axes[0].plot([lower, upper], [lower, upper], "--", color="#b91c1c", lw=1.3)
    axes[0].set_xlabel("Observed TX100 diameter (mm)")
    axes[0].set_ylabel("Frozen prediction (mm)")
    axes[0].set_title("Blind point predictions")
    colorbar = fig.colorbar(scatter, ax=axes[0], fraction=0.046, pad=0.04)
    colorbar.set_label("Absolute relative error")

    axes[1].scatter(predicted, signed, color="#2563eb", alpha=0.78, s=32)
    axes[1].axhline(0.0, color="#111827", lw=1.1)
    axes[1].axhline(float(np.mean(signed)), color="#b91c1c", ls="--", lw=1.3)
    axes[1].set_xlabel("Frozen prediction (mm)")
    axes[1].set_ylabel("Prediction − observation (mm)")
    axes[1].set_title("Signed residuals")
    fig.tight_layout()
    _save_figure_exclusive(fig, path)


def plot_risk_coverage(curves: dict[str, pd.DataFrame], path: Path) -> None:
    labels = {
        "strict_minmax": "Strict min–max",
        "knn_q95": "5-NN distance",
        "mahalanobis_q95": "OAS Mahalanobis",
        "isolation_forest_q95": "Isolation Forest",
    }
    fig, axis = plt.subplots(figsize=(7.0, 4.8))
    for method in LOCKED_METHOD_ORDER:
        curve = curves[method]
        axis.step(
            np.r_[0.0, curve["coverage"]],
            np.r_[curve["relative_risk"].iloc[0], curve["relative_risk"]],
            where="post",
            lw=1.8,
            label=labels[method],
        )
    axis.set_xlim(0, 1.01)
    axis.set_xlabel("Retained TX100 fraction")
    axis.set_ylabel("Mean absolute relative error")
    axis.set_title("Tie-aware selective risk")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    _save_figure_exclusive(fig, path)


def plot_calibration_sensitivity(calibration: dict[str, Any], path: Path) -> None:
    frame = pd.DataFrame(calibration["held_out_group_sensitivity"])
    labels = [f"{value:g}" for value in frame["held_out_cmc_mm"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].bar(labels, frame["maximum_absolute_residual_mm"], color="#7c3aed")
    axes[0].set_xlabel("Held-out development CMC (mM)")
    axes[0].set_ylabel("Maximum residual (mm)")
    axes[0].set_title("Post-hoc group-max radius")
    axes[1].bar(
        labels,
        frame["blind_tx100_coverage_using_that_group_max"],
        color="#0891b2",
    )
    axes[1].axhline(0.90, color="#b91c1c", ls="--", lw=1.2, label="0.90 reference")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("Held-out development CMC (mM)")
    axes[1].set_ylabel("Observed TX100 coverage")
    axes[1].set_title("Sensitivity, not a guarantee")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    _save_figure_exclusive(fig, path)


def plot_target_search(target_analysis: dict[str, Any], path: Path) -> None:
    aggregate = pd.DataFrame(target_analysis["aggregate"]).T
    strategies = aggregate.index.tolist()
    short = {
        "point_only": "Point",
        "strict_minmax": "Min–max",
        "knn_q95": "5-NN",
        "mahalanobis_q95": "Mahalanobis",
        "isolation_forest_q95": "Isolation F.",
        "conformal_target_inclusion": "Interval",
    }
    pairwise = pd.DataFrame(target_analysis["pairwise_selection_and_outcome_comparison"])
    matrix = np.full((len(strategies), len(strategies)), np.nan)
    for _, row in pairwise.iterrows():
        i = strategies.index(row["left"])
        j = strategies.index(row["right"])
        value = row["identical_top1_fraction_among_both_answered"]
        matrix[i, j] = value
        matrix[j, i] = value

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7))
    x = np.arange(len(strategies))
    width = 0.38
    axes[0].bar(
        x - width / 2,
        aggregate["answered_coverage"].to_numpy(dtype=float),
        width,
        label="Answered coverage",
        color="#2563eb",
    )
    axes[0].bar(
        x + width / 2,
        aggregate["success_at_1_all_queries"].to_numpy(dtype=float),
        width,
        label="Success@1",
        color="#16a34a",
    )
    axes[0].set_xticks(x, [short.get(value, value) for value in strategies], rotation=30, ha="right")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("Fraction of 13 fixed targets")
    axes[0].set_title("Frozen search outcomes")
    axes[0].legend(frameon=False)

    image = axes[1].imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    axes[1].set_xticks(x, [short.get(value, value) for value in strategies], rotation=45, ha="right")
    axes[1].set_yticks(x, [short.get(value, value) for value in strategies])
    axes[1].set_title("Identical top-1 among jointly answered")
    for i in range(len(strategies)):
        for j in range(len(strategies)):
            if np.isfinite(matrix[i, j]):
                axes[1].text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if matrix[i, j] > 0.55 else "black",
                )
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save_figure_exclusive(fig, path)


def run_posthoc_analysis(
    bundle_path: Path,
    bundle_hash_path: Path,
    recovered_result_path: Path,
    development_path: Path,
    sealed_labels_path: Path,
    incident_path: Path,
    gate_path: Path,
    output_dir: Path,
    bootstrap_repetitions: int = 10_000,
) -> dict[str, Any]:
    source_paths = [
        bundle_path,
        bundle_hash_path,
        recovered_result_path,
        development_path,
        sealed_labels_path,
        incident_path,
        gate_path,
    ]
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    source_hashes_before = {str(path.resolve()): sha256(path) for path in source_paths}
    bundle_hash = verify_bundle_hash(bundle_path, bundle_hash_path)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    recovered = json.loads(recovered_result_path.read_text(encoding="utf-8"))
    development = pd.read_csv(development_path)
    labels = pd.read_csv(sealed_labels_path)
    predictions = pd.DataFrame(bundle["predictions"])
    if sha256(development_path) != bundle["input_hashes"]["development_labeled"]:
        raise ValueError("development table differs from frozen bundle input")
    if recovered["prediction_bundle_sha256"] != bundle_hash:
        raise ValueError("recovered result does not belong to frozen bundle")
    if recovered["sealed_labels_sha256"] != sha256(sealed_labels_path):
        raise ValueError("sealed labels differ from the recovered reveal")

    merged = point_diagnostics(predictions, labels)
    point = summarize_point(merged)
    uncertainty = conditional_row_uncertainty(merged, bootstrap_repetitions)
    risk_summary, risk_curves = risk_coverage_analysis(merged)
    multiple_testing = matched_budget_p_values(merged, recovered)
    calibration = calibration_sensitivity(development, merged)
    target_analysis = target_search_comparison(bundle, merged)
    agreement = recovered_agreement(point, merged, recovered)

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = output_dir / "blind_tx100_merged_diagnostics.csv"
    figure_prediction = output_dir / "figure_1_blind_prediction_diagnostics.png"
    figure_risk = output_dir / "figure_2_blind_risk_coverage.png"
    figure_calibration = output_dir / "figure_3_posthoc_calibration_sensitivity.png"
    figure_search = output_dir / "figure_4_blind_target_search.png"
    summary_path = output_dir / "blind_posthoc_summary.json"
    _exclusive_csv(diagnostics_path, merged)
    plot_prediction_diagnostics(merged, figure_prediction)
    plot_risk_coverage(risk_curves, figure_risk)
    plot_calibration_sensitivity(calibration, figure_calibration)
    plot_target_search(target_analysis, figure_search)

    source_hashes_after = {str(path.resolve()): sha256(path) for path in source_paths}
    output_paths = [
        diagnostics_path,
        figure_prediction,
        figure_risk,
        figure_calibration,
        figure_search,
    ]
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "schema_version": "blind-posthoc-analysis-v2-degenerate-null-correction",
        "status": "post-reveal descriptive and sensitivity analysis",
        "read_only_audit": {
            "source_hashes_before": source_hashes_before,
            "source_hashes_after": source_hashes_after,
            "all_source_hashes_unchanged": source_hashes_before == source_hashes_after,
            "frozen_bundle_sha256": bundle_hash,
            "output_policy": "exclusive creation in a separate caller-selected directory",
        },
        "recovery_incident": {
            "incident_record_file": incident_path.name,
            "incident_record_sha256": sha256(incident_path),
            "consumed_gate": gate,
            "consumed_gate_sha256": sha256(gate_path),
            "recovery_audit_from_result": recovered.get("recovery_audit"),
            "disclosure_required": True,
        },
        "independent_recomputation": {
            "point_prediction": point,
            "conformal": {
                "observed_coverage": float(merged["conformal_covered"].mean()),
                "mean_interval_width_mm": float(
                    np.mean(merged["conformal_upper_mm"] - merged["conformal_lower_mm"])
                ),
                "nominal_coverage_reference": 0.90,
                "unseen_chemistry_guarantee_asserted": False,
            },
            "agreement_with_recovered_locked_result": agreement,
        },
        "conditional_within_tx100_uncertainty": uncertainty,
        "tie_aware_risk_coverage": risk_summary,
        "multiple_testing": multiple_testing,
        "exploratory_chemistry_group_calibration_sensitivity": calibration,
        "target_search_comparison": target_analysis,
        "publication_figures": [path.name for path in output_paths[1:]],
        "output_artifacts": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in output_paths
        },
        "interpretation_boundaries": [
            "All confidence intervals are conditional within the observed TX100 rows.",
            "The row bootstrap does not account for broader chemistry or laboratory sampling.",
            "The group-max calibration sensitivity was designed after label reveal and has no nominal guarantee.",
            "BH correction covers only the four locked matched-budget applicability tests.",
            "Target selections come unchanged from the frozen pre-reveal ranking artifact.",
        ],
    }
    _exclusive_json(summary_path, result)
    return result


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    project = root / "work" / "microfluidic_reliability"
    blind = project / "blind_artifacts"
    prereveal = blind / "prereveal"
    parser = argparse.ArgumentParser(description="Run read-only post-reveal TX100 analysis")
    parser.add_argument("--bundle", type=Path, default=prereveal / "blind_prediction_bundle.json")
    parser.add_argument("--bundle-hash", type=Path, default=prereveal / "blind_prediction_bundle.sha256")
    parser.add_argument("--recovered-result", type=Path, default=blind / "blind_reveal_results_recovered.json")
    parser.add_argument("--development", type=Path, default=blind / "development_labeled.csv")
    parser.add_argument("--sealed-labels", type=Path, default=blind / "sealed_blind_labels.csv")
    parser.add_argument("--incident", type=Path, default=project / "BLIND_REVEAL_INCIDENT.md")
    parser.add_argument("--gate", type=Path, default=blind / "BLIND_REVEAL_CONSUMED.json")
    parser.add_argument("--output-dir", type=Path, default=blind / "posthoc_analysis")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_posthoc_analysis(
        args.bundle,
        args.bundle_hash,
        args.recovered_result,
        args.development,
        args.sealed_labels,
        args.incident,
        args.gate,
        args.output_dir,
        args.bootstrap_repetitions,
    )
    print(
        json.dumps(
            _sanitize(
                {
                    "status": result["status"],
                    "read_only_verified": result["read_only_audit"][
                        "all_source_hashes_unchanged"
                    ],
                    "point_prediction": result["independent_recomputation"][
                        "point_prediction"
                    ],
                    "output_artifacts": result["output_artifacts"],
                }
            ),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
