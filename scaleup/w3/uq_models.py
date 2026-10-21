"""Uncertainty-aware comparison models for W4 — design W4_DESIGN.md §2.1.

Quantile regression forest, Gaussian process and a deep ensemble of multilayer
perceptrons. Each fit returns a point prediction, a native nominal-0.90 interval and
a native uncertainty score, all in model-response space. Fitting functions receive
indices into `spec.frame` and never see rows outside the supplied fit index (K1).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .datasets import DatasetSpec

UQ_MODELS = ("qrf", "gp", "deep_ensemble")
Z90 = 1.6448536269514722  # two-sided 90% normal quantile
ENSEMBLE_SIZE = 5
QRF_TREES = 400
QRF_MIN_LEAF = 5
GP_RESTARTS = 2
NATIVE_QUANTILES = (0.05, 0.95)


@dataclass
class UQPrediction:
    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    score: np.ndarray  # native uncertainty, response units


@dataclass
class UQFit:
    name: str
    n_fit: int
    _predict: object
    extra: dict = field(default_factory=dict)

    def predict(self, x: np.ndarray) -> UQPrediction:
        out: UQPrediction = self._predict(np.asarray(x, dtype=float))
        for label, arr in (("point", out.point), ("lower", out.lower), ("upper", out.upper), ("score", out.score)):
            if not np.isfinite(arr).all():
                raise RuntimeError(f"{self.name}: non-finite {label}")
        if np.any(out.lower > out.upper):
            raise RuntimeError(f"{self.name}: lower exceeds upper")
        if np.any(out.score < 0):
            raise RuntimeError(f"{self.name}: negative uncertainty score")
        return out


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    """Inverse weighted empirical CDF (smallest value whose cumulative weight >= q)."""

    order = np.argsort(values, kind="mergesort")
    v = np.asarray(values, dtype=float)[order]
    w = np.asarray(weights, dtype=float)[order]
    total = w.sum()
    if total <= 0:
        raise ValueError("weights must have positive total")
    cumulative = np.cumsum(w) / total
    position = int(np.searchsorted(cumulative, quantile, side="left"))
    return float(v[min(position, len(v) - 1)])


class QuantileForest:
    """Meinshausen (2006) quantile regression forest on top of a sklearn random forest."""

    def __init__(self, seed: int) -> None:
        self.forest = RandomForestRegressor(
            n_estimators=QRF_TREES, min_samples_leaf=QRF_MIN_LEAF, random_state=seed, n_jobs=1
        )
        self.y_fit: np.ndarray | None = None
        self.leaf_fit: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "QuantileForest":
        self.forest.fit(x, y)
        self.y_fit = np.asarray(y, dtype=float).copy()
        self.leaf_fit = self.forest.apply(x)  # (n_fit, n_trees)
        return self

    def weights(self, x: np.ndarray) -> np.ndarray:
        """Row-stochastic (n_query, n_fit) weight matrix."""

        leaf_query = self.forest.apply(x)  # (n_query, n_trees)
        n_trees = leaf_query.shape[1]
        weights = np.zeros((len(x), len(self.y_fit)), dtype=float)
        for tree in range(n_trees):
            leaves_fit = self.leaf_fit[:, tree]
            counts = np.bincount(leaves_fit)
            match = leaf_query[:, tree][:, None] == leaves_fit[None, :]
            weights += match / counts[leaves_fit][None, :]
        weights /= n_trees
        row_sums = weights.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-9):
            raise RuntimeError("quantile-forest weights do not sum to one")
        return weights

    def predict(self, x: np.ndarray) -> UQPrediction:
        point = np.asarray(self.forest.predict(x), dtype=float)
        weights = self.weights(x)
        lower = np.array([weighted_quantile(self.y_fit, w, NATIVE_QUANTILES[0]) for w in weights])
        upper = np.array([weighted_quantile(self.y_fit, w, NATIVE_QUANTILES[1]) for w in weights])
        lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
        return UQPrediction(point=point, lower=lower, upper=upper, score=upper - lower)


class GaussianProcessModel:
    def __init__(self, seed: int, n_features: int) -> None:
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(1e-2, (1e-6, 1e1))
        )
        self.scaler = StandardScaler()
        self.gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=GP_RESTARTS,
            random_state=seed, alpha=1e-10,
        )
        self.convergence_warnings = 0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GaussianProcessModel":
        xs = self.scaler.fit_transform(x)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            self.gp.fit(xs, y)
        self.convergence_warnings = int(sum(issubclass(w.category, ConvergenceWarning) for w in caught))
        return self

    def predict(self, x: np.ndarray) -> UQPrediction:
        mean, sd = self.gp.predict(self.scaler.transform(x), return_std=True)
        mean = np.asarray(mean, dtype=float).reshape(-1)
        sd = np.asarray(sd, dtype=float).reshape(-1)
        return UQPrediction(point=mean, lower=mean - Z90 * sd, upper=mean + Z90 * sd, score=sd)


class DeepEnsemble:
    def __init__(self, seed: int, standardise_target: bool) -> None:
        self.members = []
        for member in range(ENSEMBLE_SIZE):
            pipeline = Pipeline([
                ("scale", StandardScaler()),
                ("mlp", MLPRegressor(
                    hidden_layer_sizes=(64, 64), activation="relu", solver="adam",
                    learning_rate_init=1e-3, max_iter=2000, early_stopping=True,
                    validation_fraction=0.15, n_iter_no_change=30,
                    random_state=1000 * seed + member,
                )),
            ])
            if standardise_target:
                pipeline = TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())
            self.members.append(pipeline)
        self.n_iter: list[int] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DeepEnsemble":
        self.n_iter = []
        for member in self.members:
            member.fit(x, y)
            mlp = member.regressor_.named_steps["mlp"] if isinstance(member, TransformedTargetRegressor) else member.named_steps["mlp"]
            self.n_iter.append(int(mlp.n_iter_))
        return self

    def predict(self, x: np.ndarray) -> UQPrediction:
        stack = np.vstack([np.asarray(m.predict(x), dtype=float).reshape(-1) for m in self.members])
        mean = stack.mean(axis=0)
        sd = stack.std(axis=0, ddof=1)
        return UQPrediction(point=mean, lower=mean - Z90 * sd, upper=mean + Z90 * sd, score=sd)


def fit_uq(name: str, spec: DatasetSpec, fit_idx: np.ndarray, seed: int) -> UQFit:
    if name not in UQ_MODELS:
        raise ValueError(f"unknown uncertainty-aware model {name}")
    fit_idx = np.asarray(fit_idx, dtype=int)
    x = spec.features(fit_idx)
    y = spec.response[fit_idx]
    extra: dict = {}
    if name == "qrf":
        model = QuantileForest(seed).fit(x, y)
    elif name == "gp":
        model = GaussianProcessModel(seed, x.shape[1]).fit(x, y)
        extra["convergence_warnings"] = model.convergence_warnings
        extra["kernel"] = str(model.gp.kernel_)
    else:
        model = DeepEnsemble(seed, standardise_target=(spec.response_kind == "mm")).fit(x, y)
        extra["member_n_iter"] = list(model.n_iter)
    return UQFit(name=name, n_fit=int(len(fit_idx)), _predict=model.predict, extra=extra)


def predict_uq(fit: UQFit, spec: DatasetSpec, index: np.ndarray) -> UQPrediction:
    return fit.predict(spec.features(np.asarray(index, dtype=int)))
