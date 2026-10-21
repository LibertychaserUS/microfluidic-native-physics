"""Reliability and search-pruning analysis for the public DAFD 3.0 data.

This analysis deliberately separates three questions:

1. Does the published point predictor reproduce under random splitting?
2. Do nominal 90% conformal intervals retain coverage under a literature-source
   shift?
3. Can a label-free applicability-domain rule prune unreliable candidates while
   enriching the retained set for designs with small observed error?

The external labels are used only for retrospective evaluation. The trust-region
threshold is selected solely from the comprehensive training covariates.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from strict_json import dumps_strict, write_strict_json


MODULE_PATH = Path(__file__).with_name("dafd_reproduction.py")
SPEC = importlib.util.spec_from_file_location("dafd_reproduction", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
DAFD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DAFD)


def finite_sample_quantile(scores: np.ndarray, coverage: float) -> float:
    n = len(scores)
    level = min(np.ceil((n + 1) * coverage) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def build_quantile_model(quantile: float, seed: int) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        loss="quantile",
        alpha=quantile,
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        min_samples_leaf=5,
        random_state=seed,
    )


def ordered_interval(
    lower_model: GradientBoostingRegressor,
    upper_model: GradientBoostingRegressor,
    x: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    first = lower_model.predict(x)
    second = upper_model.predict(x)
    return np.minimum(first, second), np.maximum(first, second)


def interval_summary(
    lower: np.ndarray,
    upper: np.ndarray,
    actual: np.ndarray,
    hydraulic_diameter: np.ndarray,
) -> dict[str, float]:
    covered = (actual >= lower) & (actual <= upper)
    width_um = (upper - lower) * hydraulic_diameter
    return {
        "coverage": float(np.mean(covered)),
        "mean_width_um": float(np.mean(width_um)),
        "median_width_um": float(np.median(width_um)),
    }


def cqr_shift_experiment(
    comprehensive: pd.DataFrame,
    external: pd.DataFrame,
    seeds: range = range(15),
    nominal_coverage: float = 0.9,
) -> dict[str, object]:
    internal_runs: list[dict[str, float]] = []
    external_runs: list[dict[str, float]] = []

    for seed in seeds:
        indices = np.arange(len(comprehensive))
        proper_train, remainder = train_test_split(
            indices, test_size=0.4, random_state=seed, shuffle=True
        )
        calibration, internal_test = train_test_split(
            remainder, test_size=0.5, random_state=seed, shuffle=True
        )

        lower_model = build_quantile_model(0.05, seed)
        upper_model = build_quantile_model(0.95, seed)
        lower_model.fit(
            comprehensive.iloc[proper_train][DAFD.FEATURES],
            comprehensive.iloc[proper_train][DAFD.TARGET_NORMALIZED],
        )
        upper_model.fit(
            comprehensive.iloc[proper_train][DAFD.FEATURES],
            comprehensive.iloc[proper_train][DAFD.TARGET_NORMALIZED],
        )

        calibration_lower, calibration_upper = ordered_interval(
            lower_model,
            upper_model,
            comprehensive.iloc[calibration][DAFD.FEATURES],
        )
        calibration_actual = comprehensive.iloc[calibration][
            DAFD.TARGET_NORMALIZED
        ].to_numpy()
        conformity_scores = np.maximum(
            calibration_lower - calibration_actual,
            calibration_actual - calibration_upper,
        )
        adjustment = finite_sample_quantile(conformity_scores, nominal_coverage)

        for frame, collector in (
            (comprehensive.iloc[internal_test], internal_runs),
            (external, external_runs),
        ):
            lower, upper = ordered_interval(
                lower_model, upper_model, frame[DAFD.FEATURES]
            )
            collector.append(
                interval_summary(
                    lower - adjustment,
                    upper + adjustment,
                    frame[DAFD.TARGET_NORMALIZED].to_numpy(),
                    frame[DAFD.HYDRAULIC_DIAMETER].to_numpy(),
                )
            )

    return {
        "method": "90% conformalized quantile regression",
        "nominal_coverage": nominal_coverage,
        "n_runs": len(list(seeds)),
        "internal_random_split": DAFD.summarize(internal_runs),
        "external_literature_shift": DAFD.summarize(external_runs),
    }


def applicability_distance(
    comprehensive: pd.DataFrame,
    external: pd.DataFrame,
    neighbors: int = 5,
    threshold_quantile: float = 0.95,
) -> tuple[np.ndarray, float, np.ndarray]:
    scaler = StandardScaler().fit(comprehensive[DAFD.FEATURES])
    train_scaled = scaler.transform(comprehensive[DAFD.FEATURES])
    external_scaled = scaler.transform(external[DAFD.FEATURES])

    # Include the query point itself for the training-set calculation, hence k+1.
    training_neighbors = NearestNeighbors(n_neighbors=neighbors + 1).fit(train_scaled)
    training_distance = training_neighbors.kneighbors(train_scaled)[0][:, -1]
    threshold = float(np.quantile(training_distance, threshold_quantile))

    external_neighbors = NearestNeighbors(n_neighbors=neighbors).fit(train_scaled)
    external_distance = external_neighbors.kneighbors(external_scaled)[0][:, -1]
    accepted = external_distance <= threshold
    return external_distance, threshold, accepted


def point_metrics(actual_um: np.ndarray, predicted_um: np.ndarray) -> dict[str, float]:
    absolute_error = np.abs(actual_um - predicted_um)
    return {
        "n": int(len(actual_um)),
        "mape_percent": float(
            100.0 * mean_absolute_percentage_error(actual_um, predicted_um)
        ),
        "mae_um": float(np.mean(absolute_error)),
        "median_absolute_error_um": float(np.median(absolute_error)),
        "max_absolute_error_um": float(np.max(absolute_error)),
    }


def pruning_experiment(
    comprehensive: pd.DataFrame,
    external: pd.DataFrame,
    threshold_quantile: float = 0.95,
) -> tuple[dict[str, object], pd.DataFrame]:
    model = DAFD.build_model(seed=2024)
    model.fit(
        comprehensive[DAFD.FEATURES], comprehensive[DAFD.TARGET_NORMALIZED]
    )
    predicted_normalized = model.predict(external[DAFD.FEATURES])
    predicted_um = (
        predicted_normalized * external[DAFD.HYDRAULIC_DIAMETER].to_numpy()
    )
    actual_um = external[DAFD.TARGET_UM].to_numpy()
    absolute_error = np.abs(actual_um - predicted_um)

    distance, threshold, accepted = applicability_distance(
        comprehensive,
        external,
        neighbors=5,
        threshold_quantile=threshold_quantile,
    )
    correlation = spearmanr(distance, absolute_error)

    tolerance_rows: list[dict[str, float | int]] = []
    for tolerance_um in (5.0, 10.0, 15.0, 20.0):
        successful = absolute_error <= tolerance_um
        retained_successes = successful & accepted
        tolerance_rows.append(
            {
                "tolerance_um": tolerance_um,
                "unfiltered_success_rate": float(np.mean(successful)),
                "retained_success_rate_precision": float(
                    np.mean(successful[accepted])
                ),
                "successful_candidate_recall": float(
                    np.sum(retained_successes) / np.sum(successful)
                ),
                "n_successful_unfiltered": int(np.sum(successful)),
                "n_successful_retained": int(np.sum(retained_successes)),
            }
        )

    predictions = external[
        DAFD.FEATURES
        + [
            DAFD.HYDRAULIC_DIAMETER,
            DAFD.TARGET_NORMALIZED,
            DAFD.TARGET_UM,
            DAFD.REFERENCE,
        ]
    ].copy()
    predictions["Predicted normalized droplet diameter"] = predicted_normalized
    predictions["Predicted droplet diameter (um)"] = predicted_um
    predictions["Absolute error (um)"] = absolute_error
    predictions["Applicability distance"] = distance
    predictions["Retained by 95% training trust region"] = accepted

    by_reference: dict[str, object] = {}
    for reference, group in predictions.groupby(DAFD.REFERENCE):
        mask = group.index.to_numpy()
        group_actual = group[DAFD.TARGET_UM].to_numpy()
        group_predicted = group["Predicted droplet diameter (um)"].to_numpy()
        by_reference[str(reference)] = {
            **point_metrics(group_actual, group_predicted),
            "n_retained": int(np.sum(accepted[mask])),
            "retention_rate": float(np.mean(accepted[mask])),
        }

    result: dict[str, object] = {
        "threshold_selection": (
            "95th percentile of leave-one-out 5-nearest-neighbor distances in "
            "standardized comprehensive-training feature space"
        ),
        "threshold": threshold,
        "n_external_candidates": int(len(external)),
        "n_retained": int(np.sum(accepted)),
        "n_pruned": int(np.sum(~accepted)),
        "pruning_rate": float(np.mean(~accepted)),
        "candidate_reduction_factor": float(len(external) / np.sum(accepted)),
        "all_external": point_metrics(actual_um, predicted_um),
        "retained": point_metrics(actual_um[accepted], predicted_um[accepted]),
        "pruned": point_metrics(actual_um[~accepted], predicted_um[~accepted]),
        "distance_error_spearman": {
            "rho": float(correlation.statistic),
            "p_value": float(correlation.pvalue),
        },
        "tolerance_analysis": tolerance_rows,
        "by_reference": by_reference,
    }
    return result, predictions


def make_figures(
    cqr: dict[str, object],
    pruning: dict[str, object],
    predictions: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    internal = cqr["internal_random_split"]
    external = cqr["external_literature_shift"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    labels = ["Random split", "External literature"]
    means = [internal["coverage"]["mean"], external["coverage"]["mean"]]
    errors = [internal["coverage"]["std"], external["coverage"]["std"]]
    bars = ax.bar(labels, means, yerr=errors, capsize=5, color=["#3977B9", "#D45A45"])
    ax.axhline(0.9, color="#222222", linestyle="--", linewidth=1.2, label="Nominal 90%")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Empirical interval coverage")
    ax.legend(frameon=False)
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.035, f"{value:.1%}", ha="center")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_cqr_coverage_shift.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for reference, group in predictions.groupby(DAFD.REFERENCE):
        ax.scatter(
            group["Applicability distance"],
            group["Absolute error (um)"],
            label=reference,
            s=42,
            alpha=0.85,
        )
    ax.axvline(
        float(pruning["threshold"]),
        color="#222222",
        linestyle="--",
        linewidth=1.2,
        label="Training trust-region threshold",
    )
    ax.set_xlabel("5-nearest-neighbor applicability distance")
    ax.set_ylabel("Absolute diameter error (µm)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_distance_vs_error.png", dpi=220)
    plt.close(fig)

    tolerance = pd.DataFrame(pruning["tolerance_analysis"])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(
        tolerance["tolerance_um"],
        tolerance["unfiltered_success_rate"],
        marker="o",
        linewidth=2,
        label="No pruning",
    )
    ax.plot(
        tolerance["tolerance_um"],
        tolerance["retained_success_rate_precision"],
        marker="o",
        linewidth=2,
        label="Trust-region retained",
    )
    ax.set_xlabel("Accepted absolute-error tolerance (µm)")
    ax.set_ylabel("Fraction meeting tolerance")
    ax.set_ylim(0, 1.04)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_pruning_precision.png", dpi=220)
    plt.close(fig)


def main() -> None:
    workspace = Path(__file__).resolve().parents[3]
    data_dir = workspace / "work" / "data" / "osf_938rs"
    output_dir = workspace / "work" / "microfluidic_reliability" / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    comprehensive = DAFD.load_dataset(
        data_dir / "Comprehensive_normalized.xlsx", require_rate=True
    )
    external = DAFD.load_dataset(
        data_dir / "Generalizability_data_normalized.xlsx", require_rate=False
    )
    cqr = cqr_shift_experiment(comprehensive, external)
    pruning, predictions = pruning_experiment(comprehensive, external)
    result = {
        "cqr_distribution_shift": cqr,
        "label_free_applicability_pruning": pruning,
        "claim_boundary": (
            "The external labels are retrospective evaluation only. The results "
            "do not establish wet-lab feasibility or conformal coverage under "
            "arbitrary future distribution shifts."
        ),
    }

    write_strict_json(output_dir / "reliability_metrics.json", result)
    predictions.to_csv(
        output_dir / "external_reliability_predictions.csv",
        index=False,
        encoding="utf-8",
    )
    make_figures(cqr, pruning, predictions, output_dir)
    print(dumps_strict(result))


if __name__ == "__main__":
    main()
