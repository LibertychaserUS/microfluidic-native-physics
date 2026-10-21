"""Covariate-shift-weighted split conformal (Tibshirani, Barber, Candès & Ramdas 2019).

Pure functions on arrays; no labels other than calibration residuals enter (K1).
"""

from __future__ import annotations

import numpy as np

P_CLIP = 1e-6


def weighted_radius(residuals: np.ndarray, cal_weights: np.ndarray, test_weight: float, coverage: float) -> float:
    """Smallest residual whose cumulative weighted mass reaches `coverage`; +inf when the
    calibration mass alone cannot (the test point's atom sits at +inf)."""

    residuals = np.asarray(residuals, dtype=float)
    cal_weights = np.asarray(cal_weights, dtype=float)
    if len(residuals) != len(cal_weights) or len(residuals) == 0:
        raise ValueError("residuals and weights must be non-empty and equal in length")
    if np.any(cal_weights < 0) or test_weight < 0:
        raise ValueError("weights must be non-negative")
    total = cal_weights.sum() + test_weight
    if total <= 0:
        raise ValueError("weights must have positive total")
    order = np.argsort(residuals, kind="mergesort")
    cumulative = np.cumsum(cal_weights[order]) / total
    hit = np.flatnonzero(cumulative >= coverage - 1e-12)
    if len(hit) == 0:
        return float("inf")
    return float(residuals[order][hit[0]])


def weighted_radii(residuals: np.ndarray, cal_weights: np.ndarray, test_weights: np.ndarray, coverage: float) -> np.ndarray:
    return np.array([weighted_radius(residuals, cal_weights, float(w), coverage) for w in np.asarray(test_weights, dtype=float)])


def domain_odds(prob_test: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(prob_test, dtype=float), P_CLIP, 1.0 - P_CLIP)
    return p / (1.0 - p)


def knn_density_ratio(distance_ratio: np.ndarray, n_features: int, n_fit: int, n_test: int) -> np.ndarray:
    """(r_fit / r_test)^d * n_fit / n_test from the W3 `density_ratio_knn` score."""

    ratio = np.asarray(distance_ratio, dtype=float)
    if np.any(ratio < 0) or not np.isfinite(ratio).all():
        raise ValueError("distance ratios must be finite and non-negative")
    return np.power(ratio, n_features) * (n_fit / n_test)


def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    denominator = float(np.sum(w ** 2))
    return float(np.sum(w) ** 2 / denominator) if denominator > 0 else 0.0
