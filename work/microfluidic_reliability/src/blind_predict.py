"""Generate the complete pre-reveal artifact for the sealed TX100 holdout.

The command has no sealed-label argument by design.  It consumes only the
labeled non-TX100 development table and the input-only TX100 table created by
``blind_custodian.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from sklearn.covariance import OAS
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


from blind_custodian import (
    NORMALIZED_CONCENTRATION,
    RAW_INPUT_COLUMNS,
    ROW_ID_COLUMN,
    TARGET_COLUMN,
    sha256,
)


MODEL_FEATURES = [
    "continuous_flow_ml_min",
    "dispersed_flow_ml_min",
    "equilibrium_interfacial_tension_mn_m",
    NORMALIZED_CONCENTRATION,
]
METADATA_COLUMNS = [ROW_ID_COLUMN, "source_index", "source_row_index"]
PUBLIC_INPUT_COLUMNS = METADATA_COLUMNS + RAW_INPUT_COLUMNS + [NORMALIZED_CONCENTRATION]
TARGETS_MM = np.round(np.arange(0.06, 0.181, 0.01), 2)
TOP_K = 5
RANKING_COLUMNS = [
    "strategy",
    "target_mm",
    "rank",
    ROW_ID_COLUMN,
    "predicted_diameter_mm",
    "predicted_relative_gap",
]
RANDOM_STATE = 42
ALPHA = 0.10
ANSWER_RELATIVE_TOLERANCE = 0.10


def _exclusive_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(text)


def _exclusive_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(child) for child in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if np.isnan(value):
            return None
        if not np.isfinite(value):
            raise ValueError("infinite value cannot enter the frozen artifact")
        return value
    if isinstance(value, Path):
        return str(value)
    return value


def _json_text(payload: Any) -> str:
    return json.dumps(
        _json_ready(payload),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def build_locked_model() -> XGBRegressor:
    """Return the exact point-model declaration locked in the protocol."""

    return XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.3,
        random_state=RANDOM_STATE,
        objective="reg:squarederror",
        n_jobs=1,
    )


def validate_input_tables(
    development: pd.DataFrame, blind_features: pd.DataFrame
) -> None:
    required_development = PUBLIC_INPUT_COLUMNS + [TARGET_COLUMN]
    if list(development.columns) != required_development:
        raise ValueError(
            "development table schema differs from the sealed custodian schema"
        )
    if list(blind_features.columns) != PUBLIC_INPUT_COLUMNS:
        raise ValueError(
            "blind feature table must contain only the sealed public-input schema"
        )
    if TARGET_COLUMN in blind_features.columns:
        raise ValueError("blind target leakage detected")
    for name, frame in (("development", development), ("blind", blind_features)):
        if not len(frame):
            raise ValueError(f"{name} table is empty")
        if frame[ROW_ID_COLUMN].isna().any() or frame[ROW_ID_COLUMN].duplicated().any():
            raise ValueError(f"{name} opaque row identifiers are missing or duplicated")
        numeric = frame.drop(columns=[ROW_ID_COLUMN])
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"{name} table contains non-finite values")
    if set(development[ROW_ID_COLUMN]).intersection(blind_features[ROW_ID_COLUMN]):
        raise ValueError("development and blind row identifiers overlap")
    if (development[TARGET_COLUMN] <= 0).any():
        raise ValueError("development targets must be positive")


def _higher_quantile(values: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("quantile input must be finite and non-empty")
    return float(np.quantile(values, quantile, method="higher"))


def split_conformal(
    development: pd.DataFrame, blind_features: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    groups = development["cmc_mm"].to_numpy(dtype=float)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("grouped conformal calibration requires at least two CMC groups")
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=RANDOM_STATE
    )
    proper_indices, calibration_indices = next(
        splitter.split(development, groups=groups)
    )
    if set(groups[proper_indices]).intersection(groups[calibration_indices]):
        raise RuntimeError("CMC group leakage in conformal split")
    model = build_locked_model()
    model.fit(
        development.iloc[proper_indices][MODEL_FEATURES],
        development.iloc[proper_indices][TARGET_COLUMN],
    )
    calibration_prediction = model.predict(
        development.iloc[calibration_indices][MODEL_FEATURES]
    )
    residuals = np.abs(
        development.iloc[calibration_indices][TARGET_COLUMN].to_numpy(dtype=float)
        - calibration_prediction
    )
    n_calibration = len(residuals)
    finite_sample_level = min(
        1.0,
        float(np.ceil((n_calibration + 1) * (1.0 - ALPHA)) / n_calibration),
    )
    radius = _higher_quantile(residuals, finite_sample_level)
    center = model.predict(blind_features[MODEL_FEATURES])
    lower = center - radius
    upper = center + radius
    metadata = {
        "alpha": ALPHA,
        "proper_training_rows": int(len(proper_indices)),
        "calibration_rows": int(n_calibration),
        "proper_training_groups": int(len(np.unique(groups[proper_indices]))),
        "calibration_groups": int(len(np.unique(groups[calibration_indices]))),
        "finite_sample_quantile_level": finite_sample_level,
        "absolute_residual_radius_mm": radius,
        "group_overlap_count": 0,
    }
    return center, lower, upper, metadata


def applicability_scores(
    development: pd.DataFrame, blind_features: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_x = development[MODEL_FEATURES].to_numpy(dtype=float)
    blind_x = blind_features[MODEL_FEATURES].to_numpy(dtype=float)
    lower = np.min(train_x, axis=0)
    upper = np.max(train_x, axis=0)
    span = np.maximum(upper - lower, 1e-12)
    minmax_accept = np.logical_and(blind_x >= lower, blind_x <= upper).all(axis=1)
    minmax_violation = np.max(
        np.maximum(np.maximum(lower - blind_x, blind_x - upper), 0.0) / span,
        axis=1,
    )

    scaler = StandardScaler().fit(train_x)
    train_scaled = scaler.transform(train_x)
    blind_scaled = scaler.transform(blind_x)

    if len(train_scaled) < 2:
        raise ValueError("at least two development rows are required")
    neighbors = min(5, len(train_scaled) - 1)
    train_knn = NearestNeighbors(n_neighbors=neighbors + 1).fit(train_scaled)
    train_knn_score = train_knn.kneighbors(train_scaled)[0][:, -1]
    blind_knn = NearestNeighbors(n_neighbors=neighbors).fit(train_scaled)
    blind_knn_score = blind_knn.kneighbors(blind_scaled)[0][:, -1]
    knn_threshold = _higher_quantile(train_knn_score, 0.95)

    oas = OAS().fit(train_scaled)
    train_delta = train_scaled - oas.location_
    blind_delta = blind_scaled - oas.location_
    train_mahalanobis = np.sqrt(
        np.einsum("ij,jk,ik->i", train_delta, oas.precision_, train_delta)
    )
    blind_mahalanobis = np.sqrt(
        np.einsum("ij,jk,ik->i", blind_delta, oas.precision_, blind_delta)
    )
    mahalanobis_threshold = _higher_quantile(train_mahalanobis, 0.95)

    isolation = IsolationForest(
        n_estimators=400,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=1,
    ).fit(train_scaled)
    train_isolation = -isolation.score_samples(train_scaled)
    blind_isolation = -isolation.score_samples(blind_scaled)
    isolation_threshold = _higher_quantile(train_isolation, 0.95)

    result = pd.DataFrame(
        {
            "minmax_score": minmax_violation,
            "minmax_accept": minmax_accept,
            "knn5_distance": blind_knn_score,
            "knn5_q95_accept": blind_knn_score <= knn_threshold,
            "mahalanobis_oas_distance": blind_mahalanobis,
            "mahalanobis_q95_accept": blind_mahalanobis <= mahalanobis_threshold,
            "isolation_forest_outlier_score": blind_isolation,
            "isolation_forest_q95_accept": blind_isolation <= isolation_threshold,
        }
    )
    metadata = {
        "feature_columns": MODEL_FEATURES,
        "standardization": "development mean and population standard deviation",
        "knn": {
            "requested_neighbors": 5,
            "effective_neighbors": int(neighbors),
            "leave_one_out_training_q95": knn_threshold,
        },
        "mahalanobis": {
            "covariance": "OAS shrinkage",
            "training_q95": mahalanobis_threshold,
        },
        "isolation_forest": {
            "n_estimators": 400,
            "random_state": RANDOM_STATE,
            "training_q95": isolation_threshold,
        },
        "retained_counts": {
            "strict_minmax": int(result["minmax_accept"].sum()),
            "knn_q95": int(result["knn5_q95_accept"].sum()),
            "mahalanobis_q95": int(result["mahalanobis_q95_accept"].sum()),
            "isolation_forest_q95": int(
                result["isolation_forest_q95_accept"].sum()
            ),
        },
    }
    return result, metadata


def build_target_search(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategies = {
        "point_only": np.ones(len(predictions), dtype=bool),
        "strict_minmax": predictions["minmax_accept"].to_numpy(dtype=bool),
        "knn_q95": predictions["knn5_q95_accept"].to_numpy(dtype=bool),
        "mahalanobis_q95": predictions[
            "mahalanobis_q95_accept"
        ].to_numpy(dtype=bool),
        "isolation_forest_q95": predictions[
            "isolation_forest_q95_accept"
        ].to_numpy(dtype=bool),
    }
    point = predictions["predicted_diameter_mm"].to_numpy(dtype=float)
    query_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    for target in TARGETS_MM:
        target = float(target)
        target_strategies = dict(strategies)
        target_strategies["conformal_target_inclusion"] = (
            predictions["conformal_lower_mm"].to_numpy(dtype=float) <= target
        ) & (predictions["conformal_upper_mm"].to_numpy(dtype=float) >= target)
        for strategy, mask in target_strategies.items():
            eligible = np.flatnonzero(mask)
            if len(eligible):
                gap = np.abs(point[eligible] - target)
                identifiers = predictions.iloc[eligible][ROW_ID_COLUMN].to_numpy(
                    dtype=str
                )
                order = eligible[np.lexsort((identifiers, gap))]
                best_relative_gap = float(abs(point[order[0]] - target) / target)
                answered = best_relative_gap <= ANSWER_RELATIVE_TOLERANCE
            else:
                order = np.array([], dtype=int)
                best_relative_gap = None
                answered = False
            query_rows.append(
                {
                    "strategy": strategy,
                    "target_mm": target,
                    "eligible_candidates": int(len(eligible)),
                    "candidate_reduction_fraction": float(
                        1.0 - len(eligible) / len(predictions)
                    ),
                    "answered": bool(answered),
                    "best_predicted_relative_gap": best_relative_gap,
                }
            )
            if answered:
                for rank, row_index in enumerate(order[:TOP_K], start=1):
                    ranking_rows.append(
                        {
                            "strategy": strategy,
                            "target_mm": target,
                            "rank": rank,
                            ROW_ID_COLUMN: predictions.iloc[row_index][ROW_ID_COLUMN],
                            "predicted_diameter_mm": float(point[row_index]),
                            "predicted_relative_gap": float(
                                abs(point[row_index] - target) / target
                            ),
                        }
                    )
    return pd.DataFrame(query_rows), pd.DataFrame(
        ranking_rows, columns=RANKING_COLUMNS
    )


def environment_manifest() -> dict[str, str]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    }


def generate_prereveal_artifact(
    development_path: Path,
    blind_features_path: Path,
    output_dir: Path,
    protocol_path: Path,
    custodian_manifest_path: Path,
) -> dict[str, Any]:
    development = pd.read_csv(development_path)
    blind_features = pd.read_csv(blind_features_path)
    validate_input_tables(development, blind_features)

    custodian_manifest = json.loads(custodian_manifest_path.read_text(encoding="utf-8"))
    expected_development_hash = custodian_manifest["artifacts"][
        "development_labeled"
    ]["sha256"]
    expected_blind_hash = custodian_manifest["artifacts"]["blind_features"][
        "sha256"
    ]
    if sha256(development_path) != expected_development_hash:
        raise ValueError("development table hash differs from custodian manifest")
    if sha256(blind_features_path) != expected_blind_hash:
        raise ValueError("blind feature hash differs from custodian manifest")
    if sha256(protocol_path) != custodian_manifest["protocol"]["sha256"]:
        raise ValueError("protocol hash differs from custodian manifest")

    point_model = build_locked_model()
    point_model.fit(development[MODEL_FEATURES], development[TARGET_COLUMN])
    point_prediction = point_model.predict(blind_features[MODEL_FEATURES])
    conformal_center, conformal_lower, conformal_upper, conformal_metadata = (
        split_conformal(development, blind_features)
    )
    scores, score_metadata = applicability_scores(development, blind_features)

    predictions = blind_features.copy()
    predictions["predicted_diameter_mm"] = point_prediction
    predictions["conformal_model_prediction_mm"] = conformal_center
    predictions["conformal_lower_mm"] = conformal_lower
    predictions["conformal_upper_mm"] = conformal_upper
    predictions = pd.concat([predictions, scores], axis=1)
    if any(TARGET_COLUMN in column for column in predictions.columns):
        raise RuntimeError("target column entered the pre-reveal prediction table")
    query_plan, rankings = build_target_search(predictions)

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_csv = output_dir / "blind_predictions.csv"
    query_csv = output_dir / "blind_target_queries.csv"
    ranking_csv = output_dir / "blind_candidate_rankings.csv"
    environment_path = output_dir / "blind_environment_manifest.json"
    bundle_path = output_dir / "blind_prediction_bundle.json"
    hash_path = output_dir / "blind_prediction_bundle.sha256"
    _exclusive_csv(prediction_csv, predictions)
    _exclusive_csv(query_csv, query_plan)
    _exclusive_csv(ranking_csv, rankings)
    _exclusive_text(environment_path, _json_text(environment_manifest()))

    source_dir = Path(__file__).resolve().parent
    code_files = [
        source_dir / "blind_custodian.py",
        source_dir / "blind_predict.py",
        source_dir / "blind_reveal.py",
    ]
    bundle: dict[str, Any] = {
        "schema_version": "blind-prediction-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "file": protocol_path.name,
            "sha256": sha256(protocol_path),
        },
        "input_hashes": {
            "development_labeled": sha256(development_path),
            "blind_features": sha256(blind_features_path),
            "custodian_manifest": sha256(custodian_manifest_path),
        },
        "code_hashes": {
            path.name: sha256(path) if path.is_file() else None for path in code_files
        },
        "environment": environment_manifest(),
        "model": {
            "library": "xgboost.XGBRegressor",
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.3,
            "random_state": RANDOM_STATE,
            "objective": "reg:squarederror",
            "n_jobs": 1,
            "trained_on_all_development_rows": True,
            "feature_columns": MODEL_FEATURES,
        },
        "conformal": conformal_metadata,
        "applicability": score_metadata,
        "target_search": {
            "targets_mm": TARGETS_MM,
            "answer_relative_tolerance": ANSWER_RELATIVE_TOLERANCE,
            "maximum_recorded_rank": TOP_K,
            "ranking_rule": "absolute predicted target gap, stable opaque-row order for ties",
        },
        "table_hashes": {
            "blind_predictions.csv": sha256(prediction_csv),
            "blind_target_queries.csv": sha256(query_csv),
            "blind_candidate_rankings.csv": sha256(ranking_csv),
            "blind_environment_manifest.json": sha256(environment_path),
        },
        "predictions": predictions.to_dict(orient="records"),
        "queries": query_plan.to_dict(orient="records"),
        "rankings": rankings.to_dict(orient="records"),
    }
    bundle = _json_ready(bundle)
    _exclusive_text(bundle_path, _json_text(bundle))
    bundle_hash = sha256(bundle_path)
    _exclusive_text(hash_path, f"{bundle_hash}  {bundle_path.name}\n")
    return {
        "bundle": bundle,
        "bundle_path": bundle_path,
        "bundle_sha256": bundle_hash,
        "hash_path": hash_path,
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    blind_dir = root / "work" / "microfluidic_reliability" / "blind_artifacts"
    parser = argparse.ArgumentParser(
        description="Generate frozen blind predictions without reading sealed labels"
    )
    parser.add_argument(
        "--development", type=Path, default=blind_dir / "development_labeled.csv"
    )
    parser.add_argument(
        "--blind-features", type=Path, default=blind_dir / "blind_features.csv"
    )
    parser.add_argument(
        "--custodian-manifest",
        type=Path,
        default=blind_dir / "custodian_manifest.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=root / "work" / "microfluidic_reliability" / "BLIND_VALIDATION_PROTOCOL.md",
    )
    parser.add_argument("--output-dir", type=Path, default=blind_dir / "prereveal")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_prereveal_artifact(
        args.development,
        args.blind_features,
        args.output_dir,
        args.protocol,
        args.custodian_manifest,
    )
    print(
        json.dumps(
            {
                "status": "pre-reveal artifact frozen",
                "blind_rows": len(result["bundle"]["predictions"]),
                "queries": len(result["bundle"]["queries"]),
                "ranked_candidates": len(result["bundle"]["rankings"]),
                "retained_counts": result["bundle"]["applicability"][
                    "retained_counts"
                ],
                "bundle_sha256": result["bundle_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
