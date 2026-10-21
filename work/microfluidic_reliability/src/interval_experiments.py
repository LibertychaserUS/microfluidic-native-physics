"""Calibrated interval experiments for the DAFD reliability study.

The module implements three locked interval estimators:

* split conformal around the published XGBoost point predictor;
* conformalized quantile regression (CQR); and
* normalized split conformal using a training-only residual-scale model.

External outcomes are deliberately absent from every fitting and prediction
interface.  They enter only through :func:`summarize_interval_rows` (and the
thin orchestration wrapper that calls it after endpoints have been produced).
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold


CORE_PATH = Path(__file__).with_name("study_core.py")
CORE_SPEC = importlib.util.spec_from_file_location("study_core", CORE_PATH)
assert CORE_SPEC is not None and CORE_SPEC.loader is not None
if CORE_SPEC.name in sys.modules:
    CORE = sys.modules[CORE_SPEC.name]
else:
    CORE = importlib.util.module_from_spec(CORE_SPEC)
    sys.modules[CORE_SPEC.name] = CORE
    CORE_SPEC.loader.exec_module(CORE)
DAFD = CORE.DAFD


IntervalMethod = Literal[
    "split_conformal",
    "cqr",
    "normalized_split_conformal",
]
SplitMode = Literal["row_random", "geometry_group"]

INTERVAL_METHODS: tuple[IntervalMethod, ...] = (
    "split_conformal",
    "cqr",
    "normalized_split_conformal",
)
SPLIT_MODES: tuple[SplitMode, ...] = ("row_random", "geometry_group")
SUPPORTED_COVERAGES = (0.90, 0.95)
ENDPOINT_COLUMNS = [
    "center_normalized",
    "lower_normalized",
    "upper_normalized",
    "center_um",
    "lower_um",
    "upper_um",
    "interval_width_normalized",
    "interval_width_um",
]


def _event_types(n_rows: int, dataset: Literal["comprehensive", "external"]) -> np.ndarray:
    """Return the paper-defined SE/DE event labels for canonical DAFD tables."""

    event = np.full(n_rows, "SE", dtype=object)
    if dataset == "comprehensive" and n_rows == 868:
        event[:197] = "DE"
        event[671:868] = "DE"
    elif dataset == "external" and n_rows == 64:
        event[44:64] = "DE"
    return event


def _physical_clusters(
    frame: pd.DataFrame,
    dataset: Literal["comprehensive", "external"],
) -> np.ndarray:
    if dataset == "comprehensive" and len(frame) == 868:
        return CORE.comprehensive_experiment_clusters(frame)
    if dataset == "external" and len(frame) == 64:
        return CORE.external_experiment_clusters(frame)
    return np.arange(len(frame), dtype=int)


def finite_sample_quantile(scores: np.ndarray, coverage: float) -> float:
    """Return the conformal finite-sample order statistic.

    For ``n`` calibration scores, the selected rank is
    ``ceil((n + 1) * coverage)``, capped at ``n``.  Selecting the order
    statistic directly avoids the off-by-one ambiguity of NumPy's percentile
    interpolation conventions.
    """

    values = np.asarray(scores, dtype=float).reshape(-1)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("conformal scores must be finite and non-empty")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be strictly between zero and one")
    rank = min(int(np.ceil((len(values) + 1) * coverage)), len(values))
    return float(np.partition(values, rank - 1)[rank - 1])


@dataclass(frozen=True)
class CalibrationState:
    """Frozen conformal threshold and the statistical unit behind it."""

    threshold: float
    calibration_unit: Literal["row", "dependency_group"]
    n_calibration_units: int
    score_aggregation: Literal["none", "group_max"]
    requested_rank: int
    used_rank: int
    nominal_finite_threshold_attainable: bool
    finite_threshold_coverage_ceiling: float


def calibrate_conformal_scores(
    scores: np.ndarray,
    coverage: float,
    dependency_groups: np.ndarray | None = None,
) -> CalibrationState:
    """Calibrate row-marginal or simultaneous group-marginal intervals.

    When groups are supplied, each dependency group contributes one score: the
    maximum row nonconformity in that group.  The conformal order statistic is
    then taken over group maxima.  Under exchangeable groups, the resulting
    threshold targets simultaneous coverage of all rows in a new group rather
    than treating dependent rows as independent calibration units.
    """

    values = np.asarray(scores, dtype=float).reshape(-1)
    if len(values) == 0:
        raise ValueError("conformal scores must be non-empty")

    def state(
        unit_scores: np.ndarray,
        unit: Literal["row", "dependency_group"],
        aggregation: Literal["none", "group_max"],
    ) -> CalibrationState:
        n_units = len(unit_scores)
        requested_rank = int(np.ceil((n_units + 1) * coverage))
        used_rank = min(requested_rank, n_units)
        return CalibrationState(
            threshold=finite_sample_quantile(unit_scores, coverage),
            calibration_unit=unit,
            n_calibration_units=int(n_units),
            score_aggregation=aggregation,
            requested_rank=requested_rank,
            used_rank=used_rank,
            nominal_finite_threshold_attainable=requested_rank <= n_units,
            finite_threshold_coverage_ceiling=float(n_units / (n_units + 1)),
        )

    if dependency_groups is None:
        return state(values, "row", "none")
    groups = np.asarray(dependency_groups).reshape(-1)
    if len(groups) != len(values):
        raise ValueError("dependency_groups and scores must have equal length")
    group_maxima = np.array(
        [np.max(values[groups == group]) for group in np.unique(groups)],
        dtype=float,
    )
    return state(group_maxima, "dependency_group", "group_max")


@dataclass(frozen=True)
class IntervalSplit:
    """Mutually exclusive train/calibration/test row partitions."""

    train_indices: np.ndarray
    calibration_indices: np.ndarray
    test_indices: np.ndarray
    split_mode: SplitMode
    dependency_groups: np.ndarray

    def validate(self, n_rows: int) -> None:
        partitions = [
            np.asarray(self.train_indices, dtype=int),
            np.asarray(self.calibration_indices, dtype=int),
            np.asarray(self.test_indices, dtype=int),
        ]
        if any(len(partition) == 0 for partition in partitions):
            raise ValueError("train, calibration, and test must all be non-empty")
        combined = np.concatenate(partitions)
        if len(combined) != n_rows or len(np.unique(combined)) != n_rows:
            raise ValueError("train, calibration, and test must be mutually exclusive")
        if set(combined.tolist()) != set(range(n_rows)):
            raise ValueError("split partitions must cover every input row exactly once")
        if len(self.dependency_groups) != n_rows:
            raise ValueError("dependency_groups must contain one value per row")
        if self.split_mode == "geometry_group":
            group_sets = [
                set(np.asarray(self.dependency_groups)[partition].tolist())
                for partition in partitions
            ]
            if (
                group_sets[0] & group_sets[1]
                or group_sets[0] & group_sets[2]
                or group_sets[1] & group_sets[2]
            ):
                raise ValueError("dependency-aware groups cross split partitions")


def _row_random_split(n_rows: int, seed: int) -> tuple[np.ndarray, ...]:
    if n_rows < 5:
        raise ValueError("at least five rows are required for a three-way split")
    permutation = np.random.default_rng(seed).permutation(n_rows)
    train_stop = max(1, int(np.floor(0.60 * n_rows)))
    calibration_stop = max(train_stop + 1, int(np.floor(0.80 * n_rows)))
    calibration_stop = min(calibration_stop, n_rows - 1)
    return (
        np.sort(permutation[:train_stop]),
        np.sort(permutation[train_stop:calibration_stop]),
        np.sort(permutation[calibration_stop:]),
    )


def _group_split(groups: np.ndarray, seed: int) -> tuple[np.ndarray, ...]:
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 5:
        raise ValueError("at least five dependency groups are required")

    # A finite split-conformal threshold can target coverage c only when
    # n_cal / (n_cal + 1) >= c.  Reserve the 19 groups required for 95%
    # whenever the dataset permits it, then retain roughly 20% for testing.
    required_calibration = int(
        np.ceil(max(SUPPORTED_COVERAGES) / (1.0 - max(SUPPORTED_COVERAGES)))
    )
    n_test = max(2, int(np.rint(0.20 * len(unique_groups))))
    n_calibration = min(
        required_calibration,
        len(unique_groups) - n_test - 2,
    )
    n_train = len(unique_groups) - n_calibration - n_test
    if n_train < 2 or n_calibration < 1:
        raise ValueError("insufficient dependency groups for train/cal/test")

    shuffled = np.random.default_rng(seed).permutation(unique_groups)
    train_groups = shuffled[:n_train]
    calibration_groups = shuffled[n_train : n_train + n_calibration]
    test_groups = shuffled[n_train + n_calibration :]
    train_indices = np.flatnonzero(np.isin(groups, train_groups))
    calibration_indices = np.flatnonzero(np.isin(groups, calibration_groups))
    test_indices = np.flatnonzero(np.isin(groups, test_groups))
    return (
        np.sort(train_indices),
        np.sort(calibration_indices),
        np.sort(test_indices),
    )


def make_interval_split(
    frame: pd.DataFrame,
    split_mode: SplitMode,
    seed: int,
    geometry_columns: Iterable[str] = CORE.GEOMETRY_FEATURES,
    experiment_clusters: np.ndarray | None = None,
) -> IntervalSplit:
    """Build a locked row or dependency-aware three-way split.

    ``geometry_group`` joins exact geometry relations with physical-experiment
    relations as connected components.  On the canonical 868-row table the DE
    pairing is loaded automatically; custom tables must pass their own
    ``experiment_clusters`` when paired observations exist. Row-random uses
    60/20/20 rows. Group mode reserves enough calibration groups for a finite
    95% group-conformal order statistic when the group count permits it.
    """

    if split_mode not in SPLIT_MODES:
        raise ValueError(f"Unknown split mode: {split_mode}")
    if experiment_clusters is None:
        if len(frame) == 868:
            experiment_clusters = CORE.comprehensive_experiment_clusters(frame)
        else:
            experiment_clusters = np.arange(len(frame), dtype=int)
    experiment_clusters = np.asarray(experiment_clusters)
    if len(experiment_clusters) != len(frame):
        raise ValueError("experiment_clusters must contain one value per row")

    if split_mode == "geometry_group":
        dependency_groups = CORE.dependency_group_ids(
            frame,
            geometry_columns,
            experiment_clusters,
        )
        train_indices, calibration_indices, test_indices = _group_split(
            dependency_groups, seed
        )
    else:
        # Group identifiers are retained for leakage diagnostics even though
        # the paper-reproduction split intentionally operates at row level.
        dependency_groups = CORE.dependency_group_ids(
            frame,
            geometry_columns,
            experiment_clusters,
        )
        train_indices, calibration_indices, test_indices = _row_random_split(
            len(frame), seed
        )

    result = IntervalSplit(
        train_indices=train_indices,
        calibration_indices=calibration_indices,
        test_indices=test_indices,
        split_mode=split_mode,
        dependency_groups=dependency_groups,
    )
    result.validate(len(frame))
    return result


def _quantile_regressor(quantile: float, seed: int) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        loss="quantile",
        alpha=quantile,
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        min_samples_leaf=5,
        random_state=seed,
    )


def _residual_scale_model(seed: int) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        loss="quantile",
        alpha=0.90,
        n_estimators=250,
        max_depth=2,
        learning_rate=0.03,
        min_samples_leaf=10,
        random_state=seed,
    )


def _group_oof_absolute_residuals(
    train_frame: pd.DataFrame,
    training_groups: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Create leakage-free proper-training residual targets for scale fitting."""

    groups = np.asarray(training_groups).reshape(-1)
    if len(groups) != len(train_frame):
        raise ValueError("training_groups must contain one value per training row")
    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))
    if n_splits < 2:
        raise ValueError(
            "normalized split conformal requires at least two training groups"
        )
    predictions = np.full(len(train_frame), np.nan, dtype=float)
    fold_assignments = np.full(len(train_frame), -1, dtype=int)
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (fit_indices, held_indices) in enumerate(
        splitter.split(train_frame, groups=groups)
    ):
        fit_groups = set(groups[fit_indices].tolist())
        held_groups = set(groups[held_indices].tolist())
        if fit_groups & held_groups:
            raise RuntimeError("OOF dependency group leaked across a fold")
        model = DAFD.build_model(seed + fold + 1)
        model.fit(
            train_frame.iloc[fit_indices][DAFD.FEATURES],
            train_frame.iloc[fit_indices][DAFD.TARGET_NORMALIZED],
        )
        predictions[held_indices] = model.predict(
            train_frame.iloc[held_indices][DAFD.FEATURES]
        )
        fold_assignments[held_indices] = fold
    if not np.isfinite(predictions).all() or np.any(fold_assignments < 0):
        raise RuntimeError("group OOF residual construction is incomplete")
    actual = train_frame[DAFD.TARGET_NORMALIZED].to_numpy(dtype=float)
    return np.abs(actual - predictions), fold_assignments, n_splits


@dataclass
class FittedIntervalPredictor:
    """A fitted interval estimator whose prediction path is outcome-free."""

    method: IntervalMethod
    nominal_coverage: float
    seed: int
    adjustment: float
    point_model: object | None = None
    lower_model: object | None = None
    upper_model: object | None = None
    scale_model: object | None = None
    scale_floor: float = 1e-6
    calibration_unit: Literal["row", "dependency_group"] = "row"
    n_calibration_units: int = 0
    calibration_score_aggregation: Literal["none", "group_max"] = "none"
    calibration_requested_rank: int = 0
    calibration_used_rank: int = 0
    nominal_finite_threshold_attainable: bool = False
    finite_threshold_coverage_ceiling: float = 0.0
    scale_residual_protocol: Literal["not_applicable", "group_oof"] = (
        "not_applicable"
    )
    scale_oof_folds: int = 0
    n_scale_training_groups: int = 0
    scale_oof_fold_assignments: np.ndarray | None = None
    calibration_row_scores: np.ndarray | None = None
    calibration_unit_labels: np.ndarray | None = None
    calibration_unit_scores: np.ndarray | None = None

    def _normalized_endpoints(
        self, design_frame: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = design_frame[DAFD.FEATURES]
        if self.method == "cqr":
            if self.lower_model is None or self.upper_model is None:
                raise RuntimeError("CQR models are missing")
            first = np.asarray(self.lower_model.predict(x), dtype=float)
            second = np.asarray(self.upper_model.predict(x), dtype=float)
            lower = np.minimum(first, second) - self.adjustment
            upper = np.maximum(first, second) + self.adjustment
            center = (lower + upper) / 2.0
            return center, lower, upper

        if self.point_model is None:
            raise RuntimeError("point model is missing")
        center = np.asarray(self.point_model.predict(x), dtype=float)
        if self.method == "split_conformal":
            radius = np.full(len(design_frame), self.adjustment, dtype=float)
        elif self.method == "normalized_split_conformal":
            if self.scale_model is None:
                raise RuntimeError("normalized conformal scale model is missing")
            scale = np.maximum(
                np.asarray(self.scale_model.predict(x), dtype=float),
                self.scale_floor,
            )
            radius = self.adjustment * scale
        else:
            raise ValueError(f"Unknown interval method: {self.method}")
        return center, center - radius, center + radius

    def predict(self, design_frame: pd.DataFrame) -> pd.DataFrame:
        """Produce endpoints using features and hydraulic diameter only."""

        required = DAFD.FEATURES + [DAFD.HYDRAULIC_DIAMETER]
        missing = [column for column in required if column not in design_frame.columns]
        if missing:
            raise ValueError(f"Missing interval-design columns: {missing}")
        center, lower, upper = self._normalized_endpoints(design_frame)
        if not (
            np.isfinite(center).all()
            and np.isfinite(lower).all()
            and np.isfinite(upper).all()
        ):
            raise RuntimeError("interval prediction produced non-finite endpoints")
        if np.any(lower > upper):
            raise RuntimeError("interval lower endpoint exceeds upper endpoint")
        hydraulic = design_frame[DAFD.HYDRAULIC_DIAMETER].to_numpy(dtype=float)
        result = pd.DataFrame(
            {
                "row_index": design_frame.index.to_numpy(),
                "method": self.method,
                "nominal_coverage": self.nominal_coverage,
                "calibration_unit": self.calibration_unit,
                "n_calibration_units": self.n_calibration_units,
                "calibration_score_aggregation": self.calibration_score_aggregation,
                "calibration_requested_rank": self.calibration_requested_rank,
                "calibration_used_rank": self.calibration_used_rank,
                "nominal_finite_threshold_attainable": (
                    self.nominal_finite_threshold_attainable
                ),
                "finite_threshold_coverage_ceiling": (
                    self.finite_threshold_coverage_ceiling
                ),
                "scale_residual_protocol": self.scale_residual_protocol,
                "scale_oof_folds": self.scale_oof_folds,
                "n_scale_training_groups": self.n_scale_training_groups,
                "center_normalized": center,
                "lower_normalized": lower,
                "upper_normalized": upper,
                "center_um": center * hydraulic,
                "lower_um": lower * hydraulic,
                "upper_um": upper * hydraulic,
                "interval_width_normalized": upper - lower,
                "interval_width_um": (upper - lower) * hydraulic,
            }
        )
        if DAFD.REFERENCE in design_frame.columns:
            result["source"] = design_frame[DAFD.REFERENCE].astype(str).to_numpy()
        return result


def _calibration_score_audit(
    scores: np.ndarray,
    dependency_groups: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact calibration-unit labels and scores used by the quantile."""

    values = np.asarray(scores, dtype=float).reshape(-1)
    if dependency_groups is None:
        return np.arange(len(values), dtype=int), values.copy()
    groups = np.asarray(dependency_groups).reshape(-1)
    if len(groups) != len(values):
        raise ValueError("dependency_groups and scores must have equal length")
    labels = np.unique(groups)
    unit_scores = np.asarray(
        [np.max(values[groups == label]) for label in labels], dtype=float
    )
    return labels, unit_scores


def fit_interval_predictor(
    train_frame: pd.DataFrame,
    calibration_frame: pd.DataFrame,
    method: IntervalMethod,
    coverage: float,
    seed: int,
    training_groups: np.ndarray | None = None,
    calibration_groups: np.ndarray | None = None,
) -> FittedIntervalPredictor:
    """Fit and calibrate an interval estimator without candidate outcomes.

    Supplying ``calibration_groups`` activates maximum-score group conformal
    calibration. ``training_groups`` freezes the dependency units used to
    construct OOF residual targets for normalized split conformal.
    """

    if method not in INTERVAL_METHODS:
        raise ValueError(f"Unknown interval method: {method}")
    if coverage not in SUPPORTED_COVERAGES:
        raise ValueError(f"coverage must be one of {SUPPORTED_COVERAGES}")
    if len(train_frame) == 0 or len(calibration_frame) == 0:
        raise ValueError("training and calibration frames must be non-empty")

    train_x = train_frame[DAFD.FEATURES]
    train_y = train_frame[DAFD.TARGET_NORMALIZED].to_numpy(dtype=float)
    calibration_x = calibration_frame[DAFD.FEATURES]
    calibration_y = calibration_frame[DAFD.TARGET_NORMALIZED].to_numpy(
        dtype=float
    )
    if training_groups is None:
        training_groups = CORE.group_ids(train_frame, CORE.GEOMETRY_FEATURES)
    training_groups = np.asarray(training_groups).reshape(-1)
    if len(training_groups) != len(train_frame):
        raise ValueError("training_groups must contain one value per training row")
    if calibration_groups is not None:
        calibration_groups = np.asarray(calibration_groups).reshape(-1)
        if len(calibration_groups) != len(calibration_frame):
            raise ValueError(
                "calibration_groups must contain one value per calibration row"
            )
        if set(training_groups.tolist()) & set(calibration_groups.tolist()):
            raise ValueError("dependency groups cross training and calibration")

    if method == "cqr":
        tail = (1.0 - coverage) / 2.0
        lower_model = _quantile_regressor(tail, seed)
        upper_model = _quantile_regressor(1.0 - tail, seed)
        lower_model.fit(train_x, train_y)
        upper_model.fit(train_x, train_y)
        first = np.asarray(lower_model.predict(calibration_x), dtype=float)
        second = np.asarray(upper_model.predict(calibration_x), dtype=float)
        lower = np.minimum(first, second)
        upper = np.maximum(first, second)
        scores = np.maximum(lower - calibration_y, calibration_y - upper)
        calibration_state = calibrate_conformal_scores(
            scores, coverage, calibration_groups
        )
        unit_labels, unit_scores = _calibration_score_audit(
            scores, calibration_groups
        )
        return FittedIntervalPredictor(
            method=method,
            nominal_coverage=coverage,
            seed=seed,
            adjustment=calibration_state.threshold,
            lower_model=lower_model,
            upper_model=upper_model,
            calibration_unit=calibration_state.calibration_unit,
            n_calibration_units=calibration_state.n_calibration_units,
            calibration_score_aggregation=calibration_state.score_aggregation,
            calibration_requested_rank=calibration_state.requested_rank,
            calibration_used_rank=calibration_state.used_rank,
            nominal_finite_threshold_attainable=(
                calibration_state.nominal_finite_threshold_attainable
            ),
            finite_threshold_coverage_ceiling=(
                calibration_state.finite_threshold_coverage_ceiling
            ),
            calibration_row_scores=np.asarray(scores, dtype=float),
            calibration_unit_labels=unit_labels,
            calibration_unit_scores=unit_scores,
        )

    point_model = DAFD.build_model(seed)
    point_model.fit(train_x, train_y)
    calibration_prediction = np.asarray(
        point_model.predict(calibration_x), dtype=float
    )
    calibration_residual = np.abs(calibration_y - calibration_prediction)

    if method == "split_conformal":
        calibration_state = calibrate_conformal_scores(
            calibration_residual, coverage, calibration_groups
        )
        unit_labels, unit_scores = _calibration_score_audit(
            calibration_residual, calibration_groups
        )
        return FittedIntervalPredictor(
            method=method,
            nominal_coverage=coverage,
            seed=seed,
            adjustment=calibration_state.threshold,
            point_model=point_model,
            calibration_unit=calibration_state.calibration_unit,
            n_calibration_units=calibration_state.n_calibration_units,
            calibration_score_aggregation=calibration_state.score_aggregation,
            calibration_requested_rank=calibration_state.requested_rank,
            calibration_used_rank=calibration_state.used_rank,
            nominal_finite_threshold_attainable=(
                calibration_state.nominal_finite_threshold_attainable
            ),
            finite_threshold_coverage_ceiling=(
                calibration_state.finite_threshold_coverage_ceiling
            ),
            calibration_row_scores=np.asarray(calibration_residual, dtype=float),
            calibration_unit_labels=unit_labels,
            calibration_unit_scores=unit_scores,
        )

    train_residual, fold_assignments, n_oof_folds = (
        _group_oof_absolute_residuals(
            train_frame,
            training_groups=training_groups,
            seed=seed,
        )
    )
    positive_residual = train_residual[train_residual > 0.0]
    scale_floor = (
        max(float(np.quantile(positive_residual, 0.10)) * 0.10, 1e-6)
        if len(positive_residual)
        else 1e-6
    )
    scale_model = _residual_scale_model(seed)
    scale_model.fit(train_x, np.maximum(train_residual, scale_floor))
    calibration_scale = np.maximum(
        np.asarray(scale_model.predict(calibration_x), dtype=float),
        scale_floor,
    )
    normalized_scores = calibration_residual / calibration_scale
    calibration_state = calibrate_conformal_scores(
        normalized_scores, coverage, calibration_groups
    )
    unit_labels, unit_scores = _calibration_score_audit(
        normalized_scores, calibration_groups
    )
    return FittedIntervalPredictor(
        method=method,
        nominal_coverage=coverage,
        seed=seed,
        adjustment=calibration_state.threshold,
        point_model=point_model,
        scale_model=scale_model,
        scale_floor=scale_floor,
        calibration_unit=calibration_state.calibration_unit,
        n_calibration_units=calibration_state.n_calibration_units,
        calibration_score_aggregation=calibration_state.score_aggregation,
        calibration_requested_rank=calibration_state.requested_rank,
        calibration_used_rank=calibration_state.used_rank,
        nominal_finite_threshold_attainable=(
            calibration_state.nominal_finite_threshold_attainable
        ),
        finite_threshold_coverage_ceiling=(
            calibration_state.finite_threshold_coverage_ceiling
        ),
        scale_residual_protocol="group_oof",
        scale_oof_folds=n_oof_folds,
        n_scale_training_groups=int(len(np.unique(training_groups))),
        scale_oof_fold_assignments=fold_assignments,
        calibration_row_scores=np.asarray(normalized_scores, dtype=float),
        calibration_unit_labels=unit_labels,
        calibration_unit_scores=unit_scores,
    )


def score_interval_rows(
    intervals: pd.DataFrame,
    outcomes: pd.DataFrame,
    dependency_groups: np.ndarray | None = None,
) -> pd.DataFrame:
    """Attach retrospective coverage outcomes to precomputed endpoints."""

    if len(intervals) != len(outcomes):
        raise ValueError("intervals and outcomes must contain equal row counts")
    required = [DAFD.TARGET_NORMALIZED, DAFD.TARGET_UM]
    missing = [column for column in required if column not in outcomes.columns]
    if missing:
        raise ValueError(f"Missing interval outcome columns: {missing}")
    actual_normalized = outcomes[DAFD.TARGET_NORMALIZED].to_numpy(dtype=float)
    actual_um = outcomes[DAFD.TARGET_UM].to_numpy(dtype=float)
    result = intervals.copy()
    result["actual_normalized"] = actual_normalized
    result["actual_um"] = actual_um
    result["covered"] = (
        (actual_normalized >= result["lower_normalized"].to_numpy(dtype=float))
        & (actual_normalized <= result["upper_normalized"].to_numpy(dtype=float))
    )
    result["relative_interval_width"] = (
        result["interval_width_um"].to_numpy(dtype=float) / actual_um
    )
    if "source" not in result.columns and DAFD.REFERENCE in outcomes.columns:
        result["source"] = outcomes[DAFD.REFERENCE].astype(str).to_numpy()
    if dependency_groups is not None:
        dependency_groups = np.asarray(dependency_groups).reshape(-1)
        if len(dependency_groups) != len(result):
            raise ValueError(
                "dependency_groups must contain one value per interval row"
            )
        result["dependency_group"] = dependency_groups
    return result


def _summary_row(rows: pd.DataFrame) -> dict[str, float | int]:
    lower = rows["lower_normalized"].to_numpy(dtype=float)
    upper = rows["upper_normalized"].to_numpy(dtype=float)
    negative_lower = lower < 0.0
    above_one_upper = upper > 1.0
    summary: dict[str, float | int] = {
        "n_rows": int(len(rows)),
        "empirical_coverage": float(rows["covered"].mean()),
        "coverage_gap": float(
            rows["covered"].mean() - rows["nominal_coverage"].iloc[0]
        ),
        "mean_width_normalized": float(
            rows["interval_width_normalized"].mean()
        ),
        "median_width_normalized": float(
            rows["interval_width_normalized"].median()
        ),
        "mean_width_um": float(rows["interval_width_um"].mean()),
        "median_width_um": float(rows["interval_width_um"].median()),
        "mean_relative_width": float(rows["relative_interval_width"].mean()),
        "median_relative_width": float(
            rows["relative_interval_width"].median()
        ),
        "max_width_normalized": float(
            rows["interval_width_normalized"].max()
        ),
        "max_width_um": float(rows["interval_width_um"].max()),
        "minimum_lower_normalized": float(np.min(lower)),
        "maximum_upper_normalized": float(np.max(upper)),
        "n_negative_lower_endpoints": int(negative_lower.sum()),
        "fraction_negative_lower_endpoints": float(negative_lower.mean()),
        # An equivalent droplet diameter above the hydraulic diameter is not
        # asserted to be impossible; this is a descriptive geometry-scale flag.
        "n_upper_endpoints_above_hydraulic_diameter": int(
            above_one_upper.sum()
        ),
        "fraction_upper_endpoints_above_hydraulic_diameter": float(
            above_one_upper.mean()
        ),
    }
    if "dependency_group" in rows.columns:
        group_covered = rows.groupby("dependency_group", sort=False)[
            "covered"
        ].all()
        summary.update(
            {
                "n_dependency_groups": int(len(group_covered)),
                "empirical_simultaneous_group_coverage": float(
                    group_covered.mean()
                ),
                "simultaneous_group_coverage_gap": float(
                    group_covered.mean()
                    - rows["nominal_coverage"].iloc[0]
                ),
            }
        )
    return summary


def summarize_interval_rows(
    intervals: pd.DataFrame,
    outcomes: pd.DataFrame,
    dependency_groups: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Score endpoints and return overall plus optional source summaries."""

    scored = score_interval_rows(
        intervals, outcomes, dependency_groups=dependency_groups
    )
    summary: dict[str, object] = {"overall": _summary_row(scored)}
    if "source" in scored.columns:
        summary["by_source"] = {
            str(source): _summary_row(group.reset_index(drop=True))
            for source, group in scored.groupby("source", sort=True)
        }
    return scored, summary


@dataclass(frozen=True)
class IntervalExperimentResult:
    split: IntervalSplit
    predictor: FittedIntervalPredictor
    internal_rows: pd.DataFrame
    external_rows: pd.DataFrame
    summary: dict[str, object]


def _partition_audit(
    indices: np.ndarray,
    dependency_groups: np.ndarray,
    physical_clusters: np.ndarray,
    event_types: np.ndarray,
) -> dict[str, int]:
    indices = np.asarray(indices, dtype=int)
    de = event_types[indices] == "DE"
    return {
        "n_rows": int(len(indices)),
        "n_dependency_groups": int(
            len(np.unique(dependency_groups[indices]))
        ),
        "n_se_rows": int((~de).sum()),
        "n_de_rows": int(de.sum()),
        "n_de_physical_experiments": int(
            len(np.unique(physical_clusters[indices][de])) if de.any() else 0
        ),
    }


def _split_audit_metadata(
    split: IntervalSplit,
    frame: pd.DataFrame,
) -> dict[str, int]:
    physical = _physical_clusters(frame, "comprehensive")
    event = _event_types(len(frame), "comprehensive")
    result: dict[str, int] = {}
    for label, indices in (
        ("train", split.train_indices),
        ("calibration", split.calibration_indices),
        ("internal_test", split.test_indices),
    ):
        for name, value in _partition_audit(
            indices,
            split.dependency_groups,
            physical,
            event,
        ).items():
            result[f"{label}_{name}"] = value
    result["n_train_rows"] = result["train_n_rows"]
    result["n_calibration_rows"] = result["calibration_n_rows"]
    result["n_internal_test_rows"] = result["internal_test_n_rows"]
    return result


def _de_pair_allocation(
    membership: np.ndarray,
    n_rows: int,
) -> dict[str, int | float]:
    """Audit the 197 comprehensive-table DE pairs across split partitions."""

    if n_rows != 868:
        return {
            "n_de_pairs": 0,
            "n_de_pairs_train": 0,
            "n_de_pairs_calibration": 0,
            "n_de_pairs_test": 0,
            "n_de_pairs_crossing_partitions": 0,
            "de_pair_crossing_fraction": 0.0,
        }
    counts = {"train": 0, "calibration": 0, "test": 0, "crossing": 0}
    for inner in range(197):
        left = str(membership[inner])
        right = str(membership[inner + 671])
        if left == right:
            counts[left] += 1
        else:
            counts["crossing"] += 1
    return {
        "n_de_pairs": 197,
        "n_de_pairs_train": counts["train"],
        "n_de_pairs_calibration": counts["calibration"],
        "n_de_pairs_test": counts["test"],
        "n_de_pairs_crossing_partitions": counts["crossing"],
        "de_pair_crossing_fraction": float(counts["crossing"] / 197),
    }


def run_interval_experiment(
    comprehensive: pd.DataFrame,
    external: pd.DataFrame,
    method: IntervalMethod,
    coverage: float,
    split_mode: SplitMode,
    seed: int,
    geometry_columns: Iterable[str] = CORE.GEOMETRY_FEATURES,
    experiment_clusters: np.ndarray | None = None,
) -> IntervalExperimentResult:
    """Run one locked internal/external interval experiment."""

    split = make_interval_split(
        comprehensive,
        split_mode=split_mode,
        seed=seed,
        geometry_columns=geometry_columns,
        experiment_clusters=experiment_clusters,
    )
    train = comprehensive.iloc[split.train_indices]
    calibration = comprehensive.iloc[split.calibration_indices]
    internal = comprehensive.iloc[split.test_indices]
    predictor = fit_interval_predictor(
        train,
        calibration,
        method=method,
        coverage=coverage,
        seed=seed,
        training_groups=split.dependency_groups[split.train_indices],
        calibration_groups=(
            split.dependency_groups[split.calibration_indices]
            if split_mode == "geometry_group"
            else None
        ),
    )

    # Endpoint generation consumes only design columns; outcomes are attached
    # afterwards by the summary boundary below.
    internal_intervals = predictor.predict(internal)
    external_intervals = predictor.predict(external)
    external_clusters = (
        CORE.external_experiment_clusters(external)
        if len(external) == 64
        else np.arange(len(external), dtype=int)
    )
    external_groups = CORE.dependency_group_ids(
        external,
        geometry_columns,
        external_clusters,
    )
    internal_rows, internal_summary = summarize_interval_rows(
        internal_intervals,
        internal,
        dependency_groups=split.dependency_groups[split.test_indices],
    )
    external_rows, external_summary = summarize_interval_rows(
        external_intervals,
        external,
        dependency_groups=external_groups,
    )
    summary: dict[str, object] = {
        "method": method,
        "nominal_coverage": coverage,
        "split_mode": split_mode,
        "seed": seed,
        "calibration_unit": predictor.calibration_unit,
        "n_calibration_units": predictor.n_calibration_units,
        "calibration_score_aggregation": (
            predictor.calibration_score_aggregation
        ),
        "calibration_requested_rank": predictor.calibration_requested_rank,
        "calibration_used_rank": predictor.calibration_used_rank,
        "nominal_finite_threshold_attainable": (
            predictor.nominal_finite_threshold_attainable
        ),
        "finite_threshold_coverage_ceiling": (
            predictor.finite_threshold_coverage_ceiling
        ),
        "scale_residual_protocol": predictor.scale_residual_protocol,
        "scale_oof_folds": predictor.scale_oof_folds,
        "n_scale_training_groups": predictor.n_scale_training_groups,
        "n_train": int(len(train)),
        "n_calibration": int(len(calibration)),
        "n_internal_test": int(len(internal)),
        "n_external": int(len(external)),
        **_split_audit_metadata(split, comprehensive),
        "internal": internal_summary,
        "external": external_summary,
    }
    return IntervalExperimentResult(
        split=split,
        predictor=predictor,
        internal_rows=internal_rows,
        external_rows=external_rows,
        summary=summary,
    )


def _summary_records(
    result: IntervalExperimentResult,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    metadata = {
        "method": result.summary["method"],
        "protocol": result.summary["split_mode"],
        "nominal": result.summary["nominal_coverage"],
        "seed": result.summary["seed"],
        "calibration_unit": result.summary["calibration_unit"],
        "n_calibration_units": result.summary["n_calibration_units"],
        "calibration_score_aggregation": result.summary[
            "calibration_score_aggregation"
        ],
        "calibration_requested_rank": result.summary[
            "calibration_requested_rank"
        ],
        "calibration_used_rank": result.summary["calibration_used_rank"],
        "nominal_finite_threshold_attainable": result.summary[
            "nominal_finite_threshold_attainable"
        ],
        "finite_threshold_coverage_ceiling": result.summary[
            "finite_threshold_coverage_ceiling"
        ],
        "scale_residual_protocol": result.summary["scale_residual_protocol"],
        "scale_oof_folds": result.summary["scale_oof_folds"],
        "n_scale_training_groups": result.summary[
            "n_scale_training_groups"
        ],
        **{
            key: value
            for key, value in result.summary.items()
            if key.startswith(("train_n_", "calibration_n_", "internal_test_n_"))
            or key in {
                "n_train_rows",
                "n_calibration_rows",
                "n_internal_test_rows",
            }
        },
    }
    for dataset in ("internal", "external"):
        dataset_summary = result.summary[dataset]
        records.append(
            {
                **metadata,
                "dataset": dataset,
                "source": "__overall__",
                **dataset_summary["overall"],
            }
        )
        for source, source_summary in dataset_summary.get(
            "by_source", {}
        ).items():
            records.append(
                {
                    **metadata,
                    "dataset": dataset,
                    "source": source,
                    **source_summary,
                }
            )
    return records


def _suite_rows(
    rows: pd.DataFrame,
    protocol: SplitMode,
    nominal: float,
    seed: int,
) -> pd.DataFrame:
    result = rows.copy()
    result["protocol"] = protocol
    result["nominal"] = nominal
    result["seed"] = seed
    leading = [
        "row_index",
        "method",
        "protocol",
        "nominal",
        "seed",
        "lower_um",
        "upper_um",
    ]
    return result[leading + [column for column in result if column not in leading]]


def run_interval_suite(
    comprehensive: pd.DataFrame,
    external: pd.DataFrame,
    fast: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run the locked interval matrix for the main analysis.

    ``fast=True`` retains all methods, both protocols, and both nominal levels,
    but executes seed 0 only.  The full suite executes the 15 seeds used in the
    source study.  Returned tables are analysis-ready and intentionally keep
    per-row endpoints separate from aggregated summaries.
    """

    seeds = (0,) if fast else tuple(range(15))
    internal_tables: list[pd.DataFrame] = []
    external_tables: list[pd.DataFrame] = []
    summary_records: list[dict[str, object]] = []
    split_records: list[dict[str, object]] = []
    split_composition_records: list[dict[str, object]] = []
    split_pair_records: list[dict[str, object]] = []
    calibration_score_tables: list[pd.DataFrame] = []
    calibration_unit_score_tables: list[pd.DataFrame] = []
    physical_bound_tables: list[pd.DataFrame] = []
    comprehensive_physical = _physical_clusters(comprehensive, "comprehensive")
    comprehensive_event = _event_types(len(comprehensive), "comprehensive")
    external_clusters = (
        CORE.external_experiment_clusters(external)
        if len(external) == 64
        else np.arange(len(external), dtype=int)
    )
    external_groups = CORE.dependency_group_ids(
        external,
        CORE.GEOMETRY_FEATURES,
        external_clusters,
    )

    # Splits are deterministic for a protocol/seed and intentionally reused
    # across interval methods and nominal levels.
    split_cache: dict[tuple[SplitMode, int], IntervalSplit] = {}
    for protocol in SPLIT_MODES:
        for seed in seeds:
            split = make_interval_split(
                comprehensive,
                split_mode=protocol,
                seed=seed,
            )
            split_cache[(protocol, seed)] = split
            membership = np.empty(len(comprehensive), dtype=object)
            membership[split.train_indices] = "train"
            membership[split.calibration_indices] = "calibration"
            membership[split.test_indices] = "test"
            split_records.extend(
                {
                    "protocol": protocol,
                    "seed": seed,
                    "row_index": int(row_index),
                    "partition": str(membership[row_index]),
                    "dependency_group": int(split.dependency_groups[row_index]),
                    "physical_experiment_cluster": int(
                        comprehensive_physical[row_index]
                    ),
                    "event_type": str(comprehensive_event[row_index]),
                }
                for row_index in range(len(comprehensive))
            )
            for partition, indices in (
                ("train", split.train_indices),
                ("calibration", split.calibration_indices),
                ("test", split.test_indices),
            ):
                split_composition_records.append(
                    {
                        "protocol": protocol,
                        "seed": seed,
                        "partition": partition,
                        **_partition_audit(
                            indices,
                            split.dependency_groups,
                            comprehensive_physical,
                            comprehensive_event,
                        ),
                    }
                )
            split_pair_records.append(
                {
                    "protocol": protocol,
                    "seed": seed,
                    **_de_pair_allocation(membership, len(comprehensive)),
                }
            )

    for method in INTERVAL_METHODS:
        for protocol in SPLIT_MODES:
            for nominal in SUPPORTED_COVERAGES:
                for seed in seeds:
                    split = split_cache[(protocol, seed)]
                    train = comprehensive.iloc[split.train_indices]
                    calibration = comprehensive.iloc[split.calibration_indices]
                    internal = comprehensive.iloc[split.test_indices]
                    predictor = fit_interval_predictor(
                        train,
                        calibration,
                        method=method,
                        coverage=nominal,
                        seed=seed,
                        training_groups=split.dependency_groups[
                            split.train_indices
                        ],
                        calibration_groups=(
                            split.dependency_groups[split.calibration_indices]
                            if protocol == "geometry_group"
                            else None
                        ),
                    )
                    if (
                        predictor.calibration_row_scores is None
                        or predictor.calibration_unit_labels is None
                        or predictor.calibration_unit_scores is None
                    ):
                        raise RuntimeError("interval calibration audit state is missing")
                    calibration_positions = split.calibration_indices
                    calibration_score_tables.append(
                        pd.DataFrame(
                            {
                                "method": method,
                                "protocol": protocol,
                                "nominal": nominal,
                                "seed": seed,
                                "calibration_row_order": np.arange(
                                    len(calibration_positions), dtype=int
                                ),
                                "development_row_position": calibration_positions,
                                "development_row_index": comprehensive.index.to_numpy()[
                                    calibration_positions
                                ],
                                "dependency_group": split.dependency_groups[
                                    calibration_positions
                                ],
                                "physical_experiment_cluster": comprehensive_physical[
                                    calibration_positions
                                ],
                                "event_type": comprehensive_event[
                                    calibration_positions
                                ],
                                "nonconformity_score": (
                                    predictor.calibration_row_scores
                                ),
                            }
                        )
                    )
                    calibration_unit_score_tables.append(
                        pd.DataFrame(
                            {
                                "method": method,
                                "protocol": protocol,
                                "nominal": nominal,
                                "seed": seed,
                                "calibration_unit": predictor.calibration_unit,
                                "calibration_unit_label": (
                                    predictor.calibration_unit_labels
                                ),
                                "unit_nonconformity_score": (
                                    predictor.calibration_unit_scores
                                ),
                                "selected_threshold": predictor.adjustment,
                            }
                        )
                    )
                    internal_intervals = predictor.predict(internal)
                    external_intervals = predictor.predict(external)
                    internal_rows, internal_summary = summarize_interval_rows(
                        internal_intervals,
                        internal,
                        dependency_groups=split.dependency_groups[
                            split.test_indices
                        ],
                    )
                    external_rows, external_summary = summarize_interval_rows(
                        external_intervals,
                        external,
                        dependency_groups=external_groups,
                    )
                    result = IntervalExperimentResult(
                        split=split,
                        predictor=predictor,
                        internal_rows=internal_rows,
                        external_rows=external_rows,
                        summary={
                            "method": method,
                            "nominal_coverage": nominal,
                            "split_mode": protocol,
                            "seed": seed,
                            "calibration_unit": predictor.calibration_unit,
                            "n_calibration_units": (
                                predictor.n_calibration_units
                            ),
                            "calibration_score_aggregation": (
                                predictor.calibration_score_aggregation
                            ),
                            "calibration_requested_rank": (
                                predictor.calibration_requested_rank
                            ),
                            "calibration_used_rank": (
                                predictor.calibration_used_rank
                            ),
                            "nominal_finite_threshold_attainable": (
                                predictor.nominal_finite_threshold_attainable
                            ),
                            "finite_threshold_coverage_ceiling": (
                                predictor.finite_threshold_coverage_ceiling
                            ),
                            "scale_residual_protocol": (
                                predictor.scale_residual_protocol
                            ),
                            "scale_oof_folds": predictor.scale_oof_folds,
                            "n_scale_training_groups": (
                                predictor.n_scale_training_groups
                            ),
                            **_split_audit_metadata(split, comprehensive),
                            "internal": internal_summary,
                            "external": external_summary,
                        },
                    )
                    internal_tables.append(
                        _suite_rows(internal_rows, protocol, nominal, seed)
                    )
                    external_tables.append(
                        _suite_rows(external_rows, protocol, nominal, seed)
                    )
                    summary_records.extend(_summary_records(result))

                    for dataset, rows in (
                        ("internal", internal_rows),
                        ("external", external_rows),
                    ):
                        physical_bound_tables.append(
                            pd.DataFrame(
                                [
                                    {
                                        "method": method,
                                        "protocol": protocol,
                                        "nominal": nominal,
                                        "seed": seed,
                                        "dataset": dataset,
                                        "source": "__overall__",
                                        **_summary_row(rows),
                                        "upper_above_one_interpretation": (
                                            "descriptive_geometry_scale_flag_not_asserted_invalid"
                                        ),
                                    }
                                ]
                            )
                        )

    return {
        "internal_predictions": pd.concat(internal_tables, ignore_index=True),
        "external_predictions": pd.concat(external_tables, ignore_index=True),
        "summary": pd.DataFrame.from_records(summary_records),
        "splits": pd.DataFrame.from_records(split_records),
        "split_composition": pd.DataFrame.from_records(
            split_composition_records
        ),
        "split_de_pair_audit": pd.DataFrame.from_records(split_pair_records),
        "calibration_scores": pd.concat(
            calibration_score_tables, ignore_index=True
        ),
        "calibration_unit_scores": pd.concat(
            calibration_unit_score_tables, ignore_index=True
        ),
        "physical_bound_diagnostics": pd.concat(
            physical_bound_tables, ignore_index=True
        ),
    }
