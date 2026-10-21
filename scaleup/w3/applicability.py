"""Label-free applicability scores, controls and binary rules — design D4 (AD).

Scores are fitted on the fit partition's inputs; thresholds come from the
calibration partition's scores; test inputs are visible to the transductive
scores (`domain_classifier`, `density_ratio_knn`) and that is recorded. No
function here accepts a target column (K1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .datasets import DatasetSpec
from .workspace import Workspace

AD_METHODS = (
    "knn_5",
    "mahalanobis_oas",
    "isolation_forest",
    "domain_classifier",
    "density_ratio_knn",
)
CONTROL_METHODS = ("random_score", "size_only")
ALL_SCORES = AD_METHODS + CONTROL_METHODS


@dataclass
class ADBundle:
    calibration: dict[str, np.ndarray]
    test: dict[str, np.ndarray]
    thresholds: dict[str, float]
    metadata: dict


def _knn_distance(reference: np.ndarray, query: np.ndarray, k: int, exclude_self: bool) -> np.ndarray:
    if exclude_self:
        k_eff = min(k + 1, len(reference))
        if k_eff < 2:
            raise ValueError("density ratio needs at least two test rows")
        distances = NearestNeighbors(n_neighbors=k_eff).fit(reference).kneighbors(query)[0]
        return distances[:, -1]
    k_eff = min(k, len(reference))
    distances = NearestNeighbors(n_neighbors=k_eff).fit(reference).kneighbors(query)[0]
    return distances[:, -1]


def score_applicability(
    spec_train: DatasetSpec,
    fit_idx: np.ndarray,
    cal_idx: np.ndarray,
    spec_test: DatasetSpec,
    test_idx: np.ndarray,
    seed: int,
    threshold_quantile: float,
    workspace: Workspace,
) -> ADBundle:
    core = workspace.core
    fit_idx, cal_idx, test_idx = (np.asarray(a, dtype=int) for a in (fit_idx, cal_idx, test_idx))
    scaler = StandardScaler().fit(spec_train.features(fit_idx))
    fit_s = scaler.transform(spec_train.features(fit_idx))
    cal_s = scaler.transform(spec_train.features(cal_idx))
    test_s = scaler.transform(spec_test.features(test_idx))
    cal: dict[str, np.ndarray] = {}
    test: dict[str, np.ndarray] = {}

    cal["knn_5"] = core._knn_evaluation_scores(fit_s, cal_s, 5)
    test["knn_5"] = core._knn_evaluation_scores(fit_s, test_s, 5)
    cal["mahalanobis_oas"], test["mahalanobis_oas"] = core._mahalanobis_evaluation_scores(
        fit_s, cal_s, test_s
    )
    isolation = IsolationForest(
        n_estimators=400, contamination="auto", random_state=seed, n_jobs=1
    ).fit(fit_s)
    cal["isolation_forest"] = -isolation.score_samples(cal_s)
    test["isolation_forest"] = -isolation.score_samples(test_s)

    # Transductive domain classifier: fit inputs (0) versus test inputs (1).
    classifier = LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=seed
    ).fit(
        np.vstack([fit_s, test_s]),
        np.concatenate([np.zeros(len(fit_s), dtype=int), np.ones(len(test_s), dtype=int)]),
    )
    cal["domain_classifier"] = classifier.predict_proba(cal_s)[:, 1]
    test["domain_classifier"] = classifier.predict_proba(test_s)[:, 1]

    # kNN density ratio: far from fit set and close to the other test inputs → high.
    eps = 1e-12
    cal["density_ratio_knn"] = _knn_distance(fit_s, cal_s, 5, False) / (
        _knn_distance(test_s, cal_s, 5, False) + eps
    )
    test["density_ratio_knn"] = _knn_distance(fit_s, test_s, 5, False) / (
        _knn_distance(test_s, test_s, 5, True) + eps
    )

    rng = np.random.default_rng(seed + 9_173)
    cal["random_score"] = rng.uniform(size=len(cal_idx))
    test["random_score"] = rng.uniform(size=len(test_idx))

    size_fit = spec_train.size_proxy[fit_idx]
    centre = float(np.median(size_fit))
    scale = max(float(np.quantile(size_fit, 0.75) - np.quantile(size_fit, 0.25)), 1e-9)
    cal["size_only"] = np.abs(spec_train.size_proxy[cal_idx] - centre) / scale
    test["size_only"] = np.abs(spec_test.size_proxy[test_idx] - centre) / scale

    thresholds: dict[str, float] = {}
    for key in ALL_SCORES:
        if len(cal[key]) != len(cal_idx) or len(test[key]) != len(test_idx):
            raise RuntimeError(f"score length mismatch for {key}")
        if not (np.isfinite(cal[key]).all() and np.isfinite(test[key]).all()):
            raise RuntimeError(f"non-finite applicability score for {key}")
        thresholds[key] = core.threshold_from_training(cal[key], threshold_quantile)
    metadata = {
        "threshold_quantile": float(threshold_quantile),
        "threshold_source": "held_out_calibration_partition_scores",
        "transductive_methods": ["domain_classifier", "density_ratio_knn"],
        "n_fit": int(len(fit_idx)),
        "n_calibration": int(len(cal_idx)),
        "n_test": int(len(test_idx)),
    }
    return ADBundle(calibration=cal, test=test, thresholds=thresholds, metadata=metadata)


def global_threshold_mask(scores: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(scores, dtype=float) <= threshold


def within_group_median_mask(scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """D4 rule (b): keep the lower-scoring half of every test group (stable ranks)."""

    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups)
    keep = np.zeros(len(scores), dtype=bool)
    for group in np.unique(groups):
        positions = np.flatnonzero(groups == group)
        if len(positions) < 2:
            keep[positions] = True
            continue
        order = positions[np.argsort(scores[positions], kind="mergesort")]
        keep[order[: len(positions) // 2]] = True
    return keep
