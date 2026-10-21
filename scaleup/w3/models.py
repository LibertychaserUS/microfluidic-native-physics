"""Point predictors — design D4 (point models). Fitting functions take indices into
`spec.frame` and never see rows outside the supplied fit index (K1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .datasets import DatasetSpec
from .workspace import Workspace

POINT_MODELS = ("xgboost", "random_forest", "mlp")


@dataclass
class PointFit:
    name: str
    estimator: object
    n_fit: int
    extra: dict


def _make_estimator(name: str, spec: DatasetSpec, seed: int, workspace: Workspace):
    if name == "xgboost":
        # D4: published D1 configuration; sealed-V1/B1 configuration for D2.
        return workspace.dafd.build_model(seed) if spec.name == "D1" else workspace.v2.make_xgb(seed)
    if name == "random_forest":
        return workspace.core.build_regressor("random_forest", seed)
    if name == "mlp":
        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 64),
                        activation="relu",
                        solver="adam",
                        learning_rate_init=1e-3,
                        max_iter=2000,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=30,
                        random_state=seed,
                    ),
                ),
            ]
        )
        if spec.response_kind == "mm":
            return TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())
        return pipeline
    raise ValueError(f"unknown point model {name}")


def fit_point(
    name: str, spec: DatasetSpec, fit_idx: np.ndarray, seed: int, workspace: Workspace
) -> PointFit:
    estimator = _make_estimator(name, spec, seed, workspace)
    x = spec.features(fit_idx)
    y = spec.response[np.asarray(fit_idx, dtype=int)]
    estimator.fit(x, y)
    extra: dict = {}
    if name == "mlp":
        mlp = estimator.regressor_.named_steps["mlp"] if spec.response_kind == "mm" else estimator.named_steps["mlp"]
        extra["n_iter"] = int(mlp.n_iter_)
    return PointFit(name=name, estimator=estimator, n_fit=int(len(fit_idx)), extra=extra)


def predict_response(fit: PointFit, spec: DatasetSpec, index: np.ndarray) -> np.ndarray:
    prediction = np.asarray(fit.estimator.predict(spec.features(index)), dtype=float).reshape(-1)
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"{fit.name}: non-finite point prediction")
    return prediction


def predict_um(fit: PointFit, spec: DatasetSpec, index: np.ndarray) -> np.ndarray:
    return spec.to_um(predict_response(fit, spec, index), index)
