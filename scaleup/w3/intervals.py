"""Interval estimators at nominal 0.90 — design D4 (intervals).

All calibration happens in model-response space on the calibration partition;
endpoints are converted to micrometres through the dataset's monotone positive
map, which preserves ordering and coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from .datasets import DatasetSpec
from .models import PointFit, fit_point, predict_response
from .workspace import Workspace

INTERVAL_METHODS = (
    "split_conformal",
    "cqr",
    "mondrian_group_conformal",
    "ensemble_variance",
)


@dataclass
class IntervalFit:
    method: str
    point: PointFit
    adjustment: float
    meta: dict = field(default_factory=dict)
    lower_model: object | None = None
    upper_model: object | None = None
    ensemble: list[PointFit] | None = None
    sd_floor: float = 1e-6

    def predict_um(self, spec: DatasetSpec, index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        index = np.asarray(index, dtype=int)
        if self.method == "cqr":
            x = spec.features(index)
            first = np.asarray(self.lower_model.predict(x), dtype=float)
            second = np.asarray(self.upper_model.predict(x), dtype=float)
            lower = np.minimum(first, second) - self.adjustment
            upper = np.maximum(first, second) + self.adjustment
        else:
            center = predict_response(self.point, spec, index)
            if self.method in ("split_conformal", "mondrian_group_conformal"):
                radius = np.full(len(index), self.adjustment, dtype=float)
            elif self.method == "ensemble_variance":
                sd = _ensemble_sd(self.ensemble, spec, index)
                radius = self.adjustment * np.maximum(sd, self.sd_floor)
            else:
                raise ValueError(f"unknown interval method {self.method}")
            lower, upper = center - radius, center + radius
        lower_um, upper_um = spec.to_um(lower, index), spec.to_um(upper, index)
        if not (np.isfinite(lower_um).all() and np.isfinite(upper_um).all()):
            raise RuntimeError(f"{self.method}: non-finite interval endpoint")
        if np.any(lower_um > upper_um):
            raise RuntimeError(f"{self.method}: lower endpoint exceeds upper endpoint")
        return lower_um, upper_um


def _quantile_regressor(quantile: float, seed: int) -> GradientBoostingRegressor:
    # Same hyperparameters as the expanded-suite CQR (interval_experiments._quantile_regressor).
    return GradientBoostingRegressor(
        loss="quantile", alpha=quantile, n_estimators=300, max_depth=3,
        learning_rate=0.03, min_samples_leaf=5, random_state=seed,
    )


def _ensemble_sd(ensemble: list[PointFit], spec: DatasetSpec, index: np.ndarray) -> np.ndarray:
    stack = np.vstack([predict_response(member, spec, index) for member in ensemble])
    return np.std(stack, axis=0, ddof=1)


def _cluster_bootstrap_ensemble(
    point: PointFit, spec: DatasetSpec, fit_idx: np.ndarray, seed: int, members: int,
    workspace: Workspace,
) -> list[PointFit]:
    clusters = spec.cluster_id[np.asarray(fit_idx, dtype=int)]
    unique = np.unique(clusters)
    if len(unique) < 2:
        raise ValueError("cluster bootstrap requires at least two fit clusters")
    rng = np.random.default_rng(seed + 7_001)
    ensemble: list[PointFit] = []
    for member in range(members):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([fit_idx[clusters == cluster] for cluster in sampled])
        ensemble.append(fit_point(point.name, spec, rows, seed + member + 1, workspace))
    return ensemble


def fit_interval(
    method: str,
    point: PointFit,
    spec: DatasetSpec,
    fit_idx: np.ndarray,
    cal_idx: np.ndarray,
    cal_groups: np.ndarray,
    seed: int,
    coverage: float,
    members: int,
    workspace: Workspace,
) -> IntervalFit:
    """D4 — fit one interval method; the test rows are never an argument (K1)."""

    if method not in INTERVAL_METHODS:
        raise ValueError(f"unknown interval method {method}")
    fit_idx = np.asarray(fit_idx, dtype=int)
    cal_idx = np.asarray(cal_idx, dtype=int)
    finite_quantile = workspace.intervals.finite_sample_quantile
    cal_y = spec.response[cal_idx]
    meta: dict = {
        "calibration_unit": "row",
        "n_calibration_units": int(len(cal_idx)),
        "nominal_finite_threshold_attainable": bool(
            int(np.ceil((len(cal_idx) + 1) * coverage)) <= len(cal_idx)
        ),
        "finite_threshold_coverage_ceiling": float(len(cal_idx) / (len(cal_idx) + 1)),
    }

    if method == "cqr":
        tail = (1.0 - coverage) / 2.0
        x_fit = spec.features(fit_idx)
        y_fit = spec.response[fit_idx]
        lower_model = _quantile_regressor(tail, seed).fit(x_fit, y_fit)
        upper_model = _quantile_regressor(1.0 - tail, seed).fit(x_fit, y_fit)
        x_cal = spec.features(cal_idx)
        first = np.asarray(lower_model.predict(x_cal), dtype=float)
        second = np.asarray(upper_model.predict(x_cal), dtype=float)
        scores = np.maximum(np.minimum(first, second) - cal_y, cal_y - np.maximum(first, second))
        return IntervalFit(
            method=method, point=point, adjustment=finite_quantile(scores, coverage),
            meta=meta, lower_model=lower_model, upper_model=upper_model,
        )

    residual = np.abs(cal_y - predict_response(point, spec, cal_idx))
    if method == "split_conformal":
        return IntervalFit(method=method, point=point,
                           adjustment=finite_quantile(residual, coverage), meta=meta)

    if method == "mondrian_group_conformal":
        state = workspace.intervals.calibrate_conformal_scores(
            residual, coverage, np.asarray(cal_groups)
        )
        meta = {
            "calibration_unit": state.calibration_unit,
            "n_calibration_units": int(state.n_calibration_units),
            "nominal_finite_threshold_attainable": bool(
                state.nominal_finite_threshold_attainable
            ),
            "finite_threshold_coverage_ceiling": float(
                state.finite_threshold_coverage_ceiling
            ),
        }
        return IntervalFit(method=method, point=point, adjustment=state.threshold, meta=meta)

    # ensemble_variance
    ensemble = _cluster_bootstrap_ensemble(point, spec, fit_idx, seed, members, workspace)
    sd_cal = _ensemble_sd(ensemble, spec, cal_idx)
    positive = sd_cal[sd_cal > 0]
    sd_floor = max(float(np.quantile(positive, 0.10)) * 0.10, 1e-9) if len(positive) else 1e-9
    normalised = residual / np.maximum(sd_cal, sd_floor)
    meta["ensemble_members"] = int(members)
    meta["sd_floor"] = float(sd_floor)
    return IntervalFit(
        method=method, point=point, adjustment=finite_quantile(normalised, coverage),
        meta=meta, ensemble=ensemble, sd_floor=sd_floor,
    )
