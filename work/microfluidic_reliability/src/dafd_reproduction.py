"""Independent DAFD 3.0 XGBoost reproduction from the published methods.

The paper specifies 100 trees, squared-error loss, L2 regularization of 1,
learning rate 0.3, maximum depth 6, and minimum child weight 1. The model maps
eight geometry/fluid/flow descriptors to normalized droplet diameter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from strict_json import dumps_strict, write_strict_json


FEATURES = [
    "Orifice width (um)",
    "viscosity ratio",
    "Capillary number",
    "Flow rate ratio",
    "Normalized channel depth",
    "Normalized dispersed inlet",
    "Normalized continuous inlet",
    "Normalized outlet width",
]
TARGET_NORMALIZED = "Normalized droplet diameter"
TARGET_UM = "Observed droplet diameter (um)"
HYDRAULIC_DIAMETER = "Hydraulic diameter"
DISPERSED_FLOW = "Dispersed flow rate ul/h"
TARGET_RATE = "Observed generation rate (Hz)"
REFERENCE = "Ref"


def load_dataset(path: Path, require_rate: bool) -> pd.DataFrame:
    frame = pd.read_excel(path)
    required = FEATURES + [TARGET_NORMALIZED, TARGET_UM, HYDRAULIC_DIAMETER]
    if require_rate:
        required += [DISPERSED_FLOW, TARGET_RATE]
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {path}: {missing_columns}")
    invalid = frame[required].isna().any(axis=1)
    if invalid.any():
        raise ValueError(f"{path} has {int(invalid.sum())} rows missing required values")
    return frame.reset_index(drop=True)


def build_model(seed: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=100,
        objective="reg:squarederror",
        reg_lambda=1.0,
        learning_rate=0.3,
        max_depth=6,
        min_child_weight=1.0,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=seed,
        n_jobs=1,
    )


def generation_rate_hz(flow_ul_h: np.ndarray, diameter_um: np.ndarray) -> np.ndarray:
    """Apply the volume-conservation relation F = 6 Qd / (pi D^3)."""

    flow_m3_s = np.asarray(flow_ul_h, dtype=float) * 1e-9 / 3600.0
    diameter_m = np.asarray(diameter_um, dtype=float) * 1e-6
    return 6.0 * flow_m3_s / (np.pi * diameter_m**3)


def regression_metrics(
    actual_normalized: np.ndarray,
    predicted_normalized: np.ndarray,
    hydraulic_diameter: np.ndarray,
) -> dict[str, float]:
    actual_um = actual_normalized * hydraulic_diameter
    predicted_um = predicted_normalized * hydraulic_diameter
    return {
        "mape_percent": float(
            100.0 * mean_absolute_percentage_error(actual_um, predicted_um)
        ),
        "mae_um": float(mean_absolute_error(actual_um, predicted_um)),
        "rmse_um": float(np.sqrt(mean_squared_error(actual_um, predicted_um))),
        "r2": float(r2_score(actual_um, predicted_um)),
    }


def summarize(values: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = values[0].keys()
    return {
        key: {
            "mean": float(np.mean([row[key] for row in values])),
            "std": float(np.std([row[key] for row in values], ddof=1)),
            "min": float(np.min([row[key] for row in values])),
            "max": float(np.max([row[key] for row in values])),
        }
        for key in keys
    }


def run_reproduction(
    comprehensive_path: Path,
    external_path: Path,
    output_dir: Path,
    seeds: range = range(15),
) -> dict[str, object]:
    comprehensive = load_dataset(comprehensive_path, require_rate=True)
    external = load_dataset(external_path, require_rate=False)

    internal_runs: list[dict[str, float]] = []
    external_runs: list[dict[str, float]] = []
    rate_runs: list[dict[str, float]] = []

    x = comprehensive[FEATURES]
    y = comprehensive[TARGET_NORMALIZED]

    for seed in seeds:
        train_indices, test_indices = train_test_split(
            np.arange(len(comprehensive)),
            test_size=0.2,
            random_state=seed,
            shuffle=True,
        )
        model = build_model(seed)
        model.fit(x.iloc[train_indices], y.iloc[train_indices])

        internal_prediction = model.predict(x.iloc[test_indices])
        internal = comprehensive.iloc[test_indices]
        internal_runs.append(
            regression_metrics(
                internal[TARGET_NORMALIZED].to_numpy(),
                internal_prediction,
                internal[HYDRAULIC_DIAMETER].to_numpy(),
            )
        )

        predicted_um = internal_prediction * internal[HYDRAULIC_DIAMETER].to_numpy()
        predicted_rate = generation_rate_hz(
            internal[DISPERSED_FLOW].to_numpy(), predicted_um
        )
        actual_rate = internal[TARGET_RATE].to_numpy()
        rate_runs.append(
            {
                "mape_percent": float(
                    100.0 * mean_absolute_percentage_error(actual_rate, predicted_rate)
                ),
                "mae_hz": float(mean_absolute_error(actual_rate, predicted_rate)),
                "rmse_hz": float(np.sqrt(mean_squared_error(actual_rate, predicted_rate))),
                "r2": float(r2_score(actual_rate, predicted_rate)),
            }
        )

        external_prediction = model.predict(external[FEATURES])
        external_runs.append(
            regression_metrics(
                external[TARGET_NORMALIZED].to_numpy(),
                external_prediction,
                external[HYDRAULIC_DIAMETER].to_numpy(),
            )
        )

    final_model = build_model(2024)
    final_model.fit(x, y)
    external_prediction = final_model.predict(external[FEATURES])
    external_predictions = external[
        FEATURES
        + [HYDRAULIC_DIAMETER, TARGET_NORMALIZED, TARGET_UM]
        + ([REFERENCE] if REFERENCE in external.columns else [])
    ].copy()
    external_predictions["Predicted normalized droplet diameter"] = external_prediction
    external_predictions["Predicted droplet diameter (um)"] = (
        external_prediction * external[HYDRAULIC_DIAMETER].to_numpy()
    )
    external_predictions["Absolute error (um)"] = np.abs(
        external_predictions["Predicted droplet diameter (um)"]
        - external_predictions[TARGET_UM]
    )
    external_predictions["Absolute percentage error"] = (
        100.0
        * external_predictions["Absolute error (um)"]
        / external_predictions[TARGET_UM]
    )

    by_reference: dict[str, dict[str, float]] = {}
    if REFERENCE in external_predictions.columns:
        for reference, group in external_predictions.groupby(REFERENCE):
            by_reference[str(reference)] = regression_metrics(
                group[TARGET_NORMALIZED].to_numpy(),
                group["Predicted normalized droplet diameter"].to_numpy(),
                group[HYDRAULIC_DIAMETER].to_numpy(),
            )

    result: dict[str, object] = {
        "paper": "https://doi.org/10.1038/s41467-023-44068-3",
        "data_repository": "https://osf.io/938rs/",
        "xgboost_version": __import__("xgboost").__version__,
        "n_comprehensive": int(len(comprehensive)),
        "n_external": int(len(external)),
        "n_random_splits": len(list(seeds)),
        "internal_random_split": summarize(internal_runs),
        "internal_rate_from_mass_conservation": summarize(rate_runs),
        "external_literature_set": summarize(external_runs),
        "external_full_training": regression_metrics(
            external[TARGET_NORMALIZED].to_numpy(),
            external_prediction,
            external[HYDRAULIC_DIAMETER].to_numpy(),
        ),
        "external_full_training_by_reference": by_reference,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_strict_json(output_dir / "reproduction_metrics.json", result)
    external_predictions.to_csv(
        output_dir / "external_predictions.csv", index=False, encoding="utf-8"
    )
    final_model.save_model(output_dir / "dafd3_xgb_reproduction.ubj")
    return result


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    default_data = root / "work" / "data" / "osf_938rs"
    default_output = root / "work" / "microfluidic_reliability" / "artifacts"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comprehensive",
        type=Path,
        default=default_data / "Comprehensive_normalized.xlsx",
    )
    parser.add_argument(
        "--external",
        type=Path,
        default=default_data / "Generalizability_data_normalized.xlsx",
    )
    parser.add_argument("--output-dir", type=Path, default=default_output)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_reproduction(args.comprehensive, args.external, args.output_dir)
    print(dumps_strict(result))


if __name__ == "__main__":
    main()
