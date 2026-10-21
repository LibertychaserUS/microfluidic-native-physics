"""Core utilities for the expanded microfluidic reliability study.

The module keeps candidate scoring separate from retrospective outcome
evaluation.  Applicability scorers receive training outcomes only when the
method is explicitly supervised (the OOF residual-risk comparator); external
outcomes are never accepted by a fitting function.
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.covariance import OAS
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    IsolationForest,
    RandomForestRegressor,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


MODULE_PATH = Path(__file__).with_name("dafd_reproduction.py")
SPEC = importlib.util.spec_from_file_location("dafd_reproduction", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
DAFD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DAFD)


GEOMETRY_FEATURES = [
    "Orifice width (um)",
    "Normalized channel depth",
    "Normalized dispersed inlet",
    "Normalized continuous inlet",
    "Normalized outlet width",
]
DEVICE_FLUID_FEATURES = GEOMETRY_FEATURES + ["viscosity ratio"]
RISK_QUANTILES = (0.90, 0.95, 0.975, 0.99)
KNN_VALUES = (3, 5, 10, 20)
RELATIVE_TOLERANCES = (0.05, 0.10, 0.15, 0.20)
ABSOLUTE_TOLERANCES_UM = (5.0, 10.0, 15.0, 20.0)


def group_ids(frame: pd.DataFrame, columns: Iterable[str]) -> np.ndarray:
    """Return stable row-group identifiers for exact values in ``columns``."""

    return pd.util.hash_pandas_object(
        frame[list(columns)], index=False, categorize=True
    ).to_numpy(dtype=np.uint64)


def comprehensive_experiment_clusters(frame: pd.DataFrame) -> np.ndarray:
    """Return physical-experiment clusters for the 868-row DAFD table.

    The source paper models each of 197 double-emulsion experiments as two
    single-emulsion events. Rows 0--196 contain inner diameters and rows
    671--867 contain the paired outer diameters. The 474 middle rows are
    independent single-emulsion records for the purpose of this audit.
    """

    if len(frame) != 868:
        raise ValueError("DAFD comprehensive clustering expects exactly 868 rows")
    clusters = np.arange(len(frame), dtype=int)
    clusters[671:868] = np.arange(197, dtype=int)
    return clusters


def external_experiment_clusters(frame: pd.DataFrame) -> np.ndarray:
    """Return clusters for the 64-row mixed-source stress set.

    The final 20 rows are the source authors' ten blind DE flow conditions.
    Within each five-condition block, FF1 inner-diameter rows precede the five
    paired FF2 outer-diameter rows.
    """

    if len(frame) != 64:
        raise ValueError("DAFD external clustering expects exactly 64 rows")
    clusters = np.arange(len(frame), dtype=int)
    for inner, outer in zip(range(44, 49), range(49, 54)):
        clusters[outer] = clusters[inner]
    for inner, outer in zip(range(54, 59), range(59, 64)):
        clusters[outer] = clusters[inner]
    return clusters


def dependency_group_ids(
    frame: pd.DataFrame,
    columns: Iterable[str],
    experiment_clusters: np.ndarray,
) -> np.ndarray:
    """Join grouping and paired-experiment relations as connected components."""

    if len(frame) != len(experiment_clusters):
        raise ValueError("frame and experiment_clusters must have equal length")
    parent = np.arange(len(frame), dtype=int)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    relations = [
        group_ids(frame, columns),
        np.asarray(experiment_clusters),
    ]
    for relation in relations:
        first: dict[object, int] = {}
        for index, value in enumerate(relation.tolist()):
            if value in first:
                union(first[value], index)
            else:
                first[value] = index
    roots = np.array([find(index) for index in range(len(frame))], dtype=int)
    _, compact = np.unique(roots, return_inverse=True)
    return compact


def build_regressor(name: str, seed: int):
    """Build fixed, untuned regressors used in model-robustness checks."""

    if name == "xgboost":
        return DAFD.build_model(seed)
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=seed,
            n_jobs=1,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            min_samples_leaf=10,
            l2_regularization=1.0,
            random_state=seed,
        )
    raise ValueError(f"Unknown regressor: {name}")


def prediction_frame(
    frame: pd.DataFrame,
    predicted_normalized: np.ndarray,
) -> pd.DataFrame:
    """Create per-row predictions and scale-aware error outcomes."""

    predicted_normalized = np.asarray(predicted_normalized, dtype=float)
    actual_normalized = frame[DAFD.TARGET_NORMALIZED].to_numpy(dtype=float)
    hydraulic = frame[DAFD.HYDRAULIC_DIAMETER].to_numpy(dtype=float)
    actual_um = frame[DAFD.TARGET_UM].to_numpy(dtype=float)
    predicted_um = predicted_normalized * hydraulic
    absolute_um = np.abs(predicted_um - actual_um)
    result = pd.DataFrame(
        {
            "actual_normalized": actual_normalized,
            "predicted_normalized": predicted_normalized,
            "actual_um": actual_um,
            "predicted_um": predicted_um,
            "absolute_normalized_error": np.abs(
                predicted_normalized - actual_normalized
            ),
            "absolute_relative_error": absolute_um / actual_um,
            "absolute_error_um": absolute_um,
            "squared_error_um": (predicted_um - actual_um) ** 2,
        },
        index=frame.index,
    )
    if DAFD.REFERENCE in frame.columns:
        result["source"] = frame[DAFD.REFERENCE].astype(str)
    return result.reset_index(names="row_index")


def summarize_prediction_rows(rows: pd.DataFrame) -> dict[str, float | int]:
    """Summarize a per-row prediction frame without assuming independence."""

    actual = rows["actual_um"].to_numpy(dtype=float)
    predicted = rows["predicted_um"].to_numpy(dtype=float)
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = (
        float(1.0 - np.sum((actual - predicted) ** 2) / denominator)
        if denominator > 0
        else float("nan")
    )
    return {
        "n_rows": int(len(rows)),
        "mean_absolute_normalized_error": float(
            rows["absolute_normalized_error"].mean()
        ),
        "mean_absolute_relative_error": float(
            rows["absolute_relative_error"].mean()
        ),
        "mape_percent": float(100.0 * rows["absolute_relative_error"].mean()),
        "mae_um": float(rows["absolute_error_um"].mean()),
        "median_absolute_error_um": float(rows["absolute_error_um"].median()),
        "rmse_um": float(np.sqrt(rows["squared_error_um"].mean())),
        "r2": r2,
    }


def risk_curve(
    errors: np.ndarray,
    scores: np.ndarray,
    minimum_count: int = 5,
) -> tuple[pd.DataFrame, float]:
    """Return an attainable-threshold curve and tie-invariant discrete AURC.

    Low scores are treated as more trustworthy. All observations sharing a
    score threshold enter together, so row order cannot change a tied-score
    curve. For scalar AURC, every retained count from one through ``n`` is used.
    Within a tied block, cumulative risk is averaged over all random tie orders
    analytically: after ``j`` of ``b`` tied observations enter, their expected
    added error is ``j * block_error_sum / b``. ``minimum_count`` is retained
    for API compatibility but intentionally does not truncate the full curve.
    """

    errors = np.asarray(errors, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if len(errors) != len(scores):
        raise ValueError("errors and scores must have the same length")
    if len(errors) == 0 or not np.isfinite(errors).all() or not np.isfinite(scores).all():
        raise ValueError("risk inputs must be finite and non-empty")
    del minimum_count
    unique_scores = np.unique(scores)
    retained_values: list[int] = []
    risk_values: list[float] = []
    expected_cumulative_risks: list[float] = []
    retained_before = 0
    error_sum_before = 0.0
    for threshold in unique_scores:
        block_errors = errors[scores == threshold]
        block_size = int(len(block_errors))
        block_error_sum = float(math.fsum(block_errors.tolist()))
        for within_block in range(1, block_size + 1):
            expected_error_sum = (
                error_sum_before
                + within_block * block_error_sum / block_size
            )
            expected_cumulative_risks.append(
                expected_error_sum / (retained_before + within_block)
            )
        retained_before += block_size
        error_sum_before = float(
            math.fsum([error_sum_before, block_error_sum])
        )
        retained_values.append(retained_before)
        risk_values.append(error_sum_before / retained_before)
    retained = np.asarray(retained_values, dtype=int)
    coverage = retained / len(errors)
    risk = np.asarray(risk_values, dtype=float)
    curve = pd.DataFrame(
        {
            "retained": retained,
            "coverage": coverage,
            "risk": risk,
            "score_threshold": unique_scores,
        }
    )
    aurc = float(math.fsum(expected_cumulative_risks) / len(errors))
    return curve, aurc


def strict_minmax_accept(
    train_x: pd.DataFrame, candidate_x: pd.DataFrame
) -> np.ndarray:
    lower = train_x.min(axis=0)
    upper = train_x.max(axis=0)
    return (
        candidate_x.ge(lower, axis="columns")
        & candidate_x.le(upper, axis="columns")
    ).all(axis=1).to_numpy(dtype=bool)


def _axis_score(train_scaled: np.ndarray, candidate_scaled: np.ndarray):
    lower = np.min(train_scaled, axis=0)
    upper = np.max(train_scaled, axis=0)
    center = (lower + upper) / 2.0
    half_range = np.maximum((upper - lower) / 2.0, 1e-9)
    return (
        np.max(np.abs((train_scaled - center) / half_range), axis=1),
        np.max(np.abs((candidate_scaled - center) / half_range), axis=1),
    )


def _knn_evaluation_scores(
    reference_scaled: np.ndarray,
    evaluation_scaled: np.ndarray,
    neighbors: int,
) -> np.ndarray:
    """Score held-out observations against a proper reference set."""

    effective_neighbors = min(int(neighbors), len(reference_scaled))
    if effective_neighbors < 1:
        raise ValueError("kNN applicability scoring requires a non-empty fit set")
    nearest = NearestNeighbors(n_neighbors=effective_neighbors).fit(
        reference_scaled
    )
    return nearest.kneighbors(evaluation_scaled)[0][:, -1]


def _mahalanobis_evaluation_scores(
    reference_scaled: np.ndarray,
    *evaluation_sets: np.ndarray,
) -> tuple[np.ndarray, ...]:
    covariance = OAS().fit(reference_scaled)
    center = covariance.location_
    precision = covariance.precision_

    def calculate(values: np.ndarray) -> np.ndarray:
        delta = values - center
        return np.sqrt(np.einsum("ij,jk,ik->i", delta, precision, delta))

    return tuple(calculate(values) for values in evaluation_sets)


def _bootstrap_disagreement(
    fit_frame: pd.DataFrame,
    calibration_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    fit_dependency_groups: np.ndarray,
    seed: int,
    members: int = 20,
):
    """Cluster-bootstrap predictive disagreement on held-out observations."""

    fit_dependency_groups = np.asarray(fit_dependency_groups)
    if len(fit_dependency_groups) != len(fit_frame):
        raise ValueError("fit_dependency_groups must contain one value per fit row")
    unique_groups = np.unique(fit_dependency_groups)
    if len(unique_groups) < 2:
        raise ValueError("cluster bootstrap requires at least two fit groups")
    rng = np.random.default_rng(seed)
    calibration_predictions: list[np.ndarray] = []
    candidate_predictions: list[np.ndarray] = []
    for member in range(members):
        sampled_groups = rng.choice(
            unique_groups, size=len(unique_groups), replace=True
        )
        indices = np.concatenate(
            [
                np.flatnonzero(fit_dependency_groups == group)
                for group in sampled_groups
            ]
        )
        model = DAFD.build_model(seed + member + 1)
        model.fit(
            fit_frame.iloc[indices][DAFD.FEATURES],
            fit_frame.iloc[indices][DAFD.TARGET_NORMALIZED],
        )
        calibration_predictions.append(
            model.predict(calibration_frame[DAFD.FEATURES])
            * calibration_frame[DAFD.HYDRAULIC_DIAMETER].to_numpy(dtype=float)
        )
        candidate_predictions.append(
            model.predict(candidate_frame[DAFD.FEATURES])
            * candidate_frame[DAFD.HYDRAULIC_DIAMETER].to_numpy(dtype=float)
        )
    return (
        np.std(np.vstack(calibration_predictions), axis=0, ddof=1),
        np.std(np.vstack(candidate_predictions), axis=0, ddof=1),
    )


def _oof_residual_risk(
    fit_frame: pd.DataFrame,
    calibration_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    fit_dependency_groups: np.ndarray,
    seed: int,
):
    """Fit a residual-risk model on dependency-aware OOF errors.

    The returned threshold scores are predictions for the held-out feature-only
    calibration partition, never in-sample risk-model predictions.
    """

    groups = np.asarray(fit_dependency_groups)
    if len(groups) != len(fit_frame):
        raise ValueError("fit_dependency_groups must contain one value per fit row")
    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))
    if n_splits < 2:
        raise ValueError("At least two groups are required for OOF residual risk")
    oof_prediction = np.full(len(fit_frame), np.nan, dtype=float)
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (fit_idx, held_idx) in enumerate(
        splitter.split(fit_frame, groups=groups)
    ):
        if set(groups[fit_idx]).intersection(groups[held_idx]):
            raise RuntimeError("dependency group crossed an OOF risk fold")
        model = DAFD.build_model(seed + fold)
        model.fit(
            fit_frame.iloc[fit_idx][DAFD.FEATURES],
            fit_frame.iloc[fit_idx][DAFD.TARGET_NORMALIZED],
        )
        oof_prediction[held_idx] = (
            model.predict(fit_frame.iloc[held_idx][DAFD.FEATURES])
            * fit_frame.iloc[held_idx][DAFD.HYDRAULIC_DIAMETER].to_numpy(
                dtype=float
            )
        )
    if not np.isfinite(oof_prediction).all():
        raise RuntimeError("OOF prediction is incomplete")
    actual_um = fit_frame[DAFD.TARGET_UM].to_numpy(dtype=float)
    oof_relative_error = np.abs(oof_prediction - actual_um) / actual_um
    risk_model = GradientBoostingRegressor(
        loss="huber",
        n_estimators=250,
        learning_rate=0.03,
        max_depth=2,
        min_samples_leaf=10,
        random_state=seed,
    )
    risk_model.fit(
        fit_frame[DAFD.FEATURES], np.log(oof_relative_error + 1e-4)
    )
    return (
        np.exp(risk_model.predict(calibration_frame[DAFD.FEATURES])),
        np.exp(risk_model.predict(candidate_frame[DAFD.FEATURES])),
    )


@dataclass(frozen=True)
class RiskScoreBundle:
    training: dict[str, np.ndarray]
    candidate: dict[str, np.ndarray]
    strict_minmax_candidate: np.ndarray
    metadata: dict[str, object] | None = None


def fit_risk_scores(
    train_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    seed: int = 2024,
    group_columns: Iterable[str] = GEOMETRY_FEATURES,
    experiment_clusters: np.ndarray | None = None,
    include_expensive: bool = True,
    calibration_fraction: float = 0.20,
) -> RiskScoreBundle:
    """Fit locked risk scores with dependency-grouped feature calibration.

    The supplied development frame is divided into proper fit and feature-only
    calibration partitions. Scalers and detectors see only the fit partition;
    ``training`` contains held-out calibration scores used for thresholds.
    Candidate outcomes are never accessed. The supervised disagreement and OOF
    residual-risk comparators use labels only from the proper fit partition.
    """

    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be strictly between zero and one")
    if len(train_frame) < 2:
        raise ValueError("risk scoring requires at least two development rows")
    group_columns = tuple(group_columns)
    supplied_experiment_clusters = experiment_clusters is not None
    if experiment_clusters is None:
        experiment_clusters = (
            comprehensive_experiment_clusters(train_frame)
            if len(train_frame) == 868
            else np.arange(len(train_frame), dtype=int)
        )
    experiment_clusters = np.asarray(experiment_clusters)
    if len(experiment_clusters) != len(train_frame):
        raise ValueError("experiment_clusters must contain one value per development row")
    dependency_groups = dependency_group_ids(
        train_frame,
        group_columns,
        experiment_clusters,
    )
    unique_dependency_groups = np.unique(dependency_groups)
    if len(unique_dependency_groups) < 2:
        raise ValueError("at least two dependency groups are required for calibration")
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=calibration_fraction,
        random_state=seed,
    )
    fit_indices, calibration_indices = next(
        splitter.split(train_frame, groups=dependency_groups)
    )
    fit_groups = dependency_groups[fit_indices]
    calibration_groups = dependency_groups[calibration_indices]
    overlap = set(fit_groups.tolist()).intersection(calibration_groups.tolist())
    if overlap:
        raise RuntimeError("dependency group crossed applicability fit/calibration split")

    fit_frame = train_frame.iloc[fit_indices]
    calibration_frame = train_frame.iloc[calibration_indices]
    scaler = StandardScaler().fit(fit_frame[DAFD.FEATURES])
    fit_scaled = scaler.transform(fit_frame[DAFD.FEATURES])
    calibration_scaled = scaler.transform(calibration_frame[DAFD.FEATURES])
    candidate_scaled = scaler.transform(candidate_frame[DAFD.FEATURES])
    training: dict[str, np.ndarray] = {}
    candidate: dict[str, np.ndarray] = {}

    random = np.random.default_rng(seed + 9173)
    training["random_score"] = random.uniform(size=len(calibration_frame))
    candidate["random_score"] = random.uniform(size=len(candidate_frame))

    _, training["axis_aligned"] = _axis_score(
        fit_scaled, calibration_scaled
    )
    _, candidate["axis_aligned"] = _axis_score(fit_scaled, candidate_scaled)
    for neighbors in KNN_VALUES:
        key = f"knn_{neighbors}"
        training[key] = _knn_evaluation_scores(
            fit_scaled, calibration_scaled, neighbors
        )
        candidate[key] = _knn_evaluation_scores(
            fit_scaled, candidate_scaled, neighbors
        )
    training["mahalanobis_oas"], candidate["mahalanobis_oas"] = (
        _mahalanobis_evaluation_scores(
            fit_scaled,
            calibration_scaled,
            candidate_scaled,
        )
    )

    isolation = IsolationForest(
        n_estimators=400,
        contamination="auto",
        random_state=seed,
        n_jobs=1,
    ).fit(fit_scaled)
    training["isolation_forest"] = -isolation.score_samples(calibration_scaled)
    candidate["isolation_forest"] = -isolation.score_samples(candidate_scaled)

    lof_neighbors = min(20, len(fit_scaled) - 1)
    if lof_neighbors < 1:
        raise ValueError("LOF applicability scoring requires at least two fit rows")
    lof = LocalOutlierFactor(
        n_neighbors=lof_neighbors,
        novelty=True,
    ).fit(fit_scaled)
    training["local_outlier_factor"] = -lof.score_samples(calibration_scaled)
    candidate["local_outlier_factor"] = -lof.score_samples(candidate_scaled)

    one_class = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale").fit(
        fit_scaled
    )
    training["one_class_svm"] = -one_class.decision_function(
        calibration_scaled
    )
    candidate["one_class_svm"] = -one_class.decision_function(candidate_scaled)

    hydraulic_center = float(fit_frame[DAFD.HYDRAULIC_DIAMETER].mean())
    hydraulic_scale = max(
        float(fit_frame[DAFD.HYDRAULIC_DIAMETER].std(ddof=0)),
        1e-9,
    )
    training["hydraulic_diameter_only"] = np.abs(
        (
            calibration_frame[DAFD.HYDRAULIC_DIAMETER].to_numpy(dtype=float)
            - hydraulic_center
        )
        / hydraulic_scale
    )
    candidate["hydraulic_diameter_only"] = np.abs(
        (
            candidate_frame[DAFD.HYDRAULIC_DIAMETER].to_numpy(dtype=float)
            - hydraulic_center
        )
        / hydraulic_scale
    )

    point_model = DAFD.build_model(seed)
    point_model.fit(
        fit_frame[DAFD.FEATURES], fit_frame[DAFD.TARGET_NORMALIZED]
    )
    fit_predicted_um = (
        point_model.predict(fit_frame[DAFD.FEATURES])
        * fit_frame[DAFD.HYDRAULIC_DIAMETER].to_numpy(dtype=float)
    )
    calibration_predicted_um = (
        point_model.predict(calibration_frame[DAFD.FEATURES])
        * calibration_frame[DAFD.HYDRAULIC_DIAMETER].to_numpy(dtype=float)
    )
    candidate_predicted_um = (
        point_model.predict(candidate_frame[DAFD.FEATURES])
        * candidate_frame[DAFD.HYDRAULIC_DIAMETER].to_numpy(dtype=float)
    )
    size_center = float(np.median(fit_predicted_um))
    size_scale = max(
        float(
            np.quantile(fit_predicted_um, 0.75)
            - np.quantile(fit_predicted_um, 0.25)
        ),
        1e-9,
    )
    training["predicted_size_only"] = np.abs(
        calibration_predicted_um - size_center
    ) / size_scale
    candidate["predicted_size_only"] = np.abs(
        candidate_predicted_um - size_center
    ) / size_scale

    if include_expensive:
        training["bootstrap_disagreement"], candidate[
            "bootstrap_disagreement"
        ] = _bootstrap_disagreement(
            fit_frame,
            calibration_frame,
            candidate_frame,
            fit_groups,
            seed,
        )
        training["oof_residual_risk"], candidate["oof_residual_risk"] = (
            _oof_residual_risk(
                fit_frame,
                calibration_frame,
                candidate_frame,
                fit_groups,
                seed,
            )
        )

    for key in training:
        if len(training[key]) != len(calibration_frame):
            raise RuntimeError(f"Calibration-score length mismatch: {key}")
        if len(candidate[key]) != len(candidate_frame):
            raise RuntimeError(f"Candidate-score length mismatch: {key}")
        if not np.isfinite(training[key]).all() or not np.isfinite(
            candidate[key]
        ).all():
            raise RuntimeError(f"Non-finite applicability score: {key}")

    metadata: dict[str, object] = {
        "schema_version": "risk-score-feature-calibration-v1",
        "seed": int(seed),
        "calibration_fraction": float(calibration_fraction),
        "group_columns": list(group_columns),
        "experiment_clusters_supplied": bool(supplied_experiment_clusters),
        "n_development_rows": int(len(train_frame)),
        "n_dependency_groups": int(len(unique_dependency_groups)),
        "n_fit_rows": int(len(fit_indices)),
        "n_calibration_rows": int(len(calibration_indices)),
        "n_fit_groups": int(len(np.unique(fit_groups))),
        "n_calibration_groups": int(len(np.unique(calibration_groups))),
        "dependency_group_overlap_count": int(len(overlap)),
        "fit_row_positions": fit_indices.astype(int).tolist(),
        "calibration_row_positions": calibration_indices.astype(int).tolist(),
        "fit_dependency_group_ids": np.unique(fit_groups).astype(int).tolist(),
        "calibration_dependency_group_ids": (
            np.unique(calibration_groups).astype(int).tolist()
        ),
        "threshold_score_source": "held_out_feature_calibration_partition",
        "lof_training_score_source": "novelty_scores_on_held_out_calibration",
        "bootstrap_resampling_unit": "fit_dependency_group",
        "oof_residual_risk_dependency_aware": bool(include_expensive),
    }
    return RiskScoreBundle(
        training=training,
        candidate=candidate,
        strict_minmax_candidate=strict_minmax_accept(
            fit_frame[DAFD.FEATURES], candidate_frame[DAFD.FEATURES]
        ),
        metadata=metadata,
    )


def threshold_from_training(scores: np.ndarray, quantile: float) -> float:
    return float(np.quantile(np.asarray(scores, dtype=float), quantile, method="higher"))


def rank_candidates(
    predicted_um: np.ndarray,
    target_um: float,
    allowed: np.ndarray,
    tolerance_um: float,
    top_k: int,
) -> np.ndarray:
    """Select candidates without access to any observed candidate outcomes."""

    predicted_um = np.asarray(predicted_um, dtype=float)
    allowed = np.asarray(allowed, dtype=bool)
    if len(predicted_um) != len(allowed):
        raise ValueError("predicted_um and allowed must have equal length")
    eligible = np.flatnonzero(
        allowed & (np.abs(predicted_um - float(target_um)) <= tolerance_um)
    )
    if len(eligible) == 0:
        return np.array([], dtype=int)
    order = np.argsort(
        np.abs(predicted_um[eligible] - float(target_um)), kind="mergesort"
    )
    return eligible[order[:top_k]]
