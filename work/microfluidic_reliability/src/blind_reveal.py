"""One-time scoring gate for the prospective sealed-label validation.

Do not invoke this program until the pre-reveal bundle hash has been recorded.
The program creates an exclusive reveal gate *before* opening the label file,
then verifies every declared hash and opaque identifier before scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from blind_custodian import ROW_ID_COLUMN, TARGET_COLUMN, sha256
from blind_predict import RANKING_COLUMNS


RELIABLE_RELATIVE_ERROR = 0.10
RANDOM_COMPARISON_REPETITIONS = 10_000
RANDOM_STATE = 42
TOP_K_VALUES = (1, 3, 5)


def _exclusive_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(text)


def _json_text(payload: Any) -> str:
    def default(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            value = float(value)
            return value if np.isfinite(value) else None
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"cannot serialize {type(value)!r}")

    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
        default=default,
    ) + "\n"


def verify_bundle_hash(bundle_path: Path, hash_path: Path) -> str:
    parts = hash_path.read_text(encoding="utf-8").strip().split()
    if len(parts) != 2:
        raise ValueError("prediction hash sidecar must contain digest and filename")
    expected_hash, expected_name = parts
    if expected_name != bundle_path.name:
        raise ValueError("prediction hash sidecar names a different bundle")
    actual_hash = sha256(bundle_path)
    if len(expected_hash) != 64 or expected_hash.lower() != actual_hash:
        raise ValueError("pre-reveal prediction bundle SHA-256 mismatch")
    return actual_hash


def verify_opaque_ids(
    predictions: pd.DataFrame, sealed_labels: pd.DataFrame
) -> None:
    if list(sealed_labels.columns) != [ROW_ID_COLUMN, TARGET_COLUMN]:
        raise ValueError("sealed label schema is not the locked two-column schema")
    for name, frame in (("predictions", predictions), ("sealed labels", sealed_labels)):
        if frame[ROW_ID_COLUMN].isna().any() or frame[ROW_ID_COLUMN].duplicated().any():
            raise ValueError(f"{name} contain missing or duplicate opaque row IDs")
    prediction_ids = set(predictions[ROW_ID_COLUMN].astype(str))
    label_ids = set(sealed_labels[ROW_ID_COLUMN].astype(str))
    if prediction_ids != label_ids:
        missing = len(prediction_ids - label_ids)
        extra = len(label_ids - prediction_ids)
        raise ValueError(
            f"opaque row-ID set mismatch (missing labels={missing}, extra labels={extra})"
        )


def verify_prereveal_files(
    bundle_path: Path,
    hash_path: Path,
    protocol_path: Path,
    custodian_manifest_path: Path,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    bundle_hash = verify_bundle_hash(bundle_path, hash_path)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("schema_version") != "blind-prediction-v1":
        raise ValueError("unsupported pre-reveal bundle schema")
    if bundle["protocol"]["sha256"] != sha256(protocol_path):
        raise ValueError("protocol changed after predictions were frozen")
    if not bundle.get("predictions"):
        raise ValueError("pre-reveal bundle contains no prediction rows")
    if any(TARGET_COLUMN in str(column) for column in bundle["predictions"][0].keys()):
        raise ValueError("target leakage detected inside pre-reveal predictions")

    source_dir = Path(__file__).resolve().parent
    for filename, expected_hash in bundle.get("code_hashes", {}).items():
        if expected_hash is None:
            raise ValueError(f"frozen code file was missing at prediction time: {filename}")
        current_path = source_dir / filename
        if not current_path.is_file() or sha256(current_path) != expected_hash:
            raise ValueError(f"frozen code hash mismatch: {filename}")

    custodian = json.loads(custodian_manifest_path.read_text(encoding="utf-8"))
    if bundle["input_hashes"]["custodian_manifest"] != sha256(
        custodian_manifest_path
    ):
        raise ValueError("custodian manifest changed after predictions were frozen")
    if custodian["protocol"]["sha256"] != sha256(protocol_path):
        raise ValueError("custodian protocol hash mismatch")
    for filename, expected_hash in bundle["table_hashes"].items():
        path = bundle_path.parent / filename
        if not path.is_file() or sha256(path) != expected_hash:
            raise ValueError(f"frozen pre-reveal table hash mismatch: {filename}")
    return bundle, bundle_hash, custodian


def _regression_endpoints(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    error = predicted - actual
    relative = np.abs(error) / actual
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = (
        float(1.0 - np.sum(error**2) / denominator)
        if denominator > 0
        else None
    )
    mape = float(100.0 * np.mean(relative))
    return {
        "n": int(len(actual)),
        "mape_percent": mape,
        "mae_mm": float(np.mean(np.abs(error))),
        "median_absolute_error_mm": float(np.median(np.abs(error))),
        "rmse_mm": float(np.sqrt(np.mean(error**2))),
        "r2": r2,
        "reproduction_success_below_11_4_percent": bool(mape < 11.4),
        "strong_replication_at_or_below_6_percent": bool(mape <= 6.0),
    }


def _matched_random_comparison(
    errors: np.ndarray, retained: np.ndarray, rng: np.random.Generator
) -> dict[str, Any]:
    retained_count = int(np.sum(retained))
    if retained_count == 0:
        return {
            "repetitions": RANDOM_COMPARISON_REPETITIONS,
            "retained_count": 0,
            "random_mean_risk": None,
            "random_risk_95_interval": [None, None],
            "one_sided_p_random_risk_le_observed": None,
            "survives_at_0_05": False,
        }
    observed = float(np.mean(errors[retained]))
    random_risk = np.empty(RANDOM_COMPARISON_REPETITIONS, dtype=float)
    for repeat in range(RANDOM_COMPARISON_REPETITIONS):
        indices = rng.choice(len(errors), size=retained_count, replace=False)
        random_risk[repeat] = float(np.mean(errors[indices]))
    p_value = float(
        (1 + np.sum(random_risk <= observed))
        / (RANDOM_COMPARISON_REPETITIONS + 1)
    )
    return {
        "repetitions": RANDOM_COMPARISON_REPETITIONS,
        "retained_count": retained_count,
        "random_mean_risk": float(np.mean(random_risk)),
        "random_risk_95_interval": [
            float(np.quantile(random_risk, 0.025)),
            float(np.quantile(random_risk, 0.975)),
        ],
        "one_sided_p_random_risk_le_observed": p_value,
        "survives_at_0_05": bool(p_value <= 0.05),
    }


def score_applicability(
    joined: pd.DataFrame,
) -> dict[str, Any]:
    actual = joined[TARGET_COLUMN].to_numpy(dtype=float)
    predicted = joined["predicted_diameter_mm"].to_numpy(dtype=float)
    errors = np.abs(predicted - actual) / actual
    reliable = errors <= RELIABLE_RELATIVE_ERROR
    reliable_total = int(np.sum(reliable))
    full_risk = float(np.mean(errors))
    masks = {
        "strict_minmax": joined["minmax_accept"].to_numpy(dtype=bool),
        "knn_q95": joined["knn5_q95_accept"].to_numpy(dtype=bool),
        "mahalanobis_q95": joined["mahalanobis_q95_accept"].to_numpy(dtype=bool),
        "isolation_forest_q95": joined[
            "isolation_forest_q95_accept"
        ].to_numpy(dtype=bool),
    }
    minmax_count = int(masks["strict_minmax"].sum())
    minmax_risk = (
        float(np.mean(errors[masks["strict_minmax"]])) if minmax_count else None
    )
    rng = np.random.default_rng(RANDOM_STATE)
    result: dict[str, Any] = {
        "full_pool_relative_risk": full_risk,
        "reliable_definition": "absolute relative error <= 0.10",
        "full_pool_reliable_candidates": reliable_total,
        "rules": {},
    }
    comparable_margin = max(1, int(np.ceil(0.10 * len(joined))))
    for name, mask in masks.items():
        retained_count = int(mask.sum())
        retained_risk = float(np.mean(errors[mask])) if retained_count else None
        retained_reliable = int(np.sum(reliable & mask))
        random_comparison = _matched_random_comparison(errors, mask, rng)
        comparable_to_minmax = abs(retained_count - minmax_count) <= comparable_margin
        not_weaker_than_minmax = bool(
            comparable_to_minmax
            and retained_risk is not None
            and minmax_risk is not None
            and retained_risk <= minmax_risk
        )
        lower_than_full = bool(retained_risk is not None and retained_risk < full_risk)
        beneficial = bool(
            retained_count > 0
            and lower_than_full
            and not_weaker_than_minmax
            and random_comparison["survives_at_0_05"]
        )
        result["rules"][name] = {
            "retained_count": retained_count,
            "retention_fraction": float(retained_count / len(joined)),
            "candidate_reduction_fraction": float(1.0 - retained_count / len(joined)),
            "retained_relative_risk": retained_risk,
            "risk_lower_than_full_pool": lower_than_full,
            "retained_reliable_candidates": retained_reliable,
            "reliable_candidate_recall": (
                float(retained_reliable / reliable_total) if reliable_total else None
            ),
            "reliable_candidate_precision": (
                float(retained_reliable / retained_count) if retained_count else None
            ),
            "strict_minmax_retained_count": minmax_count,
            "strict_minmax_relative_risk": minmax_risk,
            "retention_comparable_to_strict_minmax_within_10pct_pool": comparable_to_minmax,
            "not_weaker_than_strict_minmax_when_comparable": not_weaker_than_minmax,
            "matched_budget_random_retention": random_comparison,
            "beneficial_under_all_locked_criteria": beneficial,
        }
    return result


def score_target_search(
    joined: pd.DataFrame,
    queries: pd.DataFrame,
    rankings: pd.DataFrame,
) -> dict[str, Any]:
    label_by_id = joined.set_index(ROW_ID_COLUMN)[TARGET_COLUMN].to_dict()
    actual_all = joined[TARGET_COLUMN].to_numpy(dtype=float)
    per_query: list[dict[str, Any]] = []
    for query in queries.to_dict(orient="records"):
        strategy = query["strategy"]
        target = float(query["target_mm"])
        oracle_errors = np.abs(actual_all - target) / target
        oracle_best = float(np.min(oracle_errors))
        oracle_achievable = bool(oracle_best <= RELIABLE_RELATIVE_ERROR)
        selected = rankings[
            (rankings["strategy"] == strategy)
            & np.isclose(rankings["target_mm"].to_numpy(dtype=float), target)
        ].sort_values("rank")
        selected_errors = np.array(
            [
                abs(float(label_by_id[row_id]) - target) / target
                for row_id in selected[ROW_ID_COLUMN].astype(str)
            ],
            dtype=float,
        )
        row: dict[str, Any] = {
            **query,
            "oracle_best_relative_target_error": oracle_best,
            "oracle_achievable_within_10_percent": oracle_achievable,
            "top1_relative_target_error": (
                float(selected_errors[0]) if len(selected_errors) else None
            ),
            "top1_oracle_regret": (
                float(selected_errors[0] - oracle_best)
                if len(selected_errors)
                else None
            ),
        }
        for top_k in TOP_K_VALUES:
            row[f"success_at_{top_k}"] = bool(
                len(selected_errors)
                and np.any(selected_errors[:top_k] <= RELIABLE_RELATIVE_ERROR)
            )
        per_query.append(row)

    per_query_frame = pd.DataFrame(per_query)
    strategies: dict[str, Any] = {}
    for strategy, group in per_query_frame.groupby("strategy", sort=True):
        answered = group["answered"].astype(bool)
        oracle_achievable = group["oracle_achievable_within_10_percent"].astype(bool)
        oracle_count = int(oracle_achievable.sum())
        regrets = pd.to_numeric(group["top1_oracle_regret"], errors="coerce").dropna()
        summary: dict[str, Any] = {
            "queries": int(len(group)),
            "answered_queries": int(answered.sum()),
            "answered_query_coverage": float(answered.mean()),
            "mean_candidate_reduction_fraction": float(
                group["candidate_reduction_fraction"].mean()
            ),
            "oracle_achievable_queries": oracle_count,
            "mean_top1_oracle_regret_answered": (
                float(regrets.mean()) if len(regrets) else None
            ),
        }
        for top_k in TOP_K_VALUES:
            success = group[f"success_at_{top_k}"].astype(bool)
            summary[f"success_at_{top_k}_all_queries"] = float(success.mean())
            summary[f"success_at_{top_k}_answered_queries"] = (
                float(success[answered].mean()) if answered.any() else None
            )
            summary[f"oracle_achievable_target_recall_at_{top_k}"] = (
                float(np.sum(success & oracle_achievable) / oracle_count)
                if oracle_count
                else None
            )
        strategies[str(strategy)] = summary
    return {
        "success_definition": "selected actual diameter within 10% of fixed target",
        "strategies": strategies,
        "per_query": per_query,
    }


def reveal_once(
    bundle_path: Path,
    hash_path: Path,
    sealed_labels_path: Path,
    custodian_manifest_path: Path,
    protocol_path: Path,
    output_path: Path,
    gate_path: Path | None = None,
) -> dict[str, Any]:
    bundle, bundle_hash, custodian = verify_prereveal_files(
        bundle_path, hash_path, protocol_path, custodian_manifest_path
    )
    gate_path = gate_path or output_path.with_suffix(output_path.suffix + ".reveal_gate")
    reveal_time = datetime.now(timezone.utc).isoformat()
    _exclusive_text(
        gate_path,
        _json_text(
            {
                "status": "reveal gate consumed",
                "reveal_utc": reveal_time,
                "prediction_bundle_sha256": bundle_hash,
            }
        ),
    )

    expected_label_hash = custodian["artifacts"]["sealed_blind_labels"]["sha256"]
    if sha256(sealed_labels_path) != expected_label_hash:
        raise ValueError("sealed label hash differs from custodian manifest")
    sealed_labels = pd.read_csv(sealed_labels_path)
    predictions = pd.DataFrame(bundle["predictions"])
    queries = pd.DataFrame(bundle["queries"])
    rankings = pd.DataFrame(bundle["rankings"], columns=RANKING_COLUMNS)
    verify_opaque_ids(predictions, sealed_labels)
    joined = predictions.merge(
        sealed_labels, on=ROW_ID_COLUMN, how="left", validate="one_to_one"
    )
    actual = joined[TARGET_COLUMN].to_numpy(dtype=float)
    predicted = joined["predicted_diameter_mm"].to_numpy(dtype=float)
    if not np.isfinite(actual).all() or (actual <= 0).any():
        raise ValueError("sealed targets are non-finite or non-positive")

    coverage = (
        (actual >= joined["conformal_lower_mm"].to_numpy(dtype=float))
        & (actual <= joined["conformal_upper_mm"].to_numpy(dtype=float))
    )
    result: dict[str, Any] = {
        "schema_version": "blind-reveal-v1",
        "reveal_utc": reveal_time,
        "prediction_bundle_sha256": bundle_hash,
        "sealed_labels_sha256": expected_label_hash,
        "row_id_verification": "exact set equality and one-to-one join passed",
        "point_prediction": _regression_endpoints(actual, predicted),
        "conformal": {
            "nominal_coverage": 0.90,
            "observed_coverage": float(np.mean(coverage)),
            "mean_interval_width_mm": float(
                np.mean(
                    joined["conformal_upper_mm"].to_numpy(dtype=float)
                    - joined["conformal_lower_mm"].to_numpy(dtype=float)
                )
            ),
            "guarantee_under_unseen_chemistry_asserted": False,
        },
        "applicability": score_applicability(joined),
        "target_search": score_target_search(joined, queries, rankings),
    }
    _exclusive_text(output_path, _json_text(result))
    return result


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    blind_dir = root / "work" / "microfluidic_reliability" / "blind_artifacts"
    prereveal = blind_dir / "prereveal"
    parser = argparse.ArgumentParser(description="Consume the one-time blind reveal gate")
    parser.add_argument(
        "--bundle", type=Path, default=prereveal / "blind_prediction_bundle.json"
    )
    parser.add_argument(
        "--bundle-hash", type=Path, default=prereveal / "blind_prediction_bundle.sha256"
    )
    parser.add_argument(
        "--sealed-labels", type=Path, default=blind_dir / "sealed_blind_labels.csv"
    )
    parser.add_argument(
        "--custodian-manifest", type=Path, default=blind_dir / "custodian_manifest.json"
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=root / "work" / "microfluidic_reliability" / "BLIND_VALIDATION_PROTOCOL.md",
    )
    parser.add_argument(
        "--output", type=Path, default=blind_dir / "blind_reveal_results.json"
    )
    parser.add_argument("--gate", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = reveal_once(
        args.bundle,
        args.bundle_hash,
        args.sealed_labels,
        args.custodian_manifest,
        args.protocol,
        args.output,
        args.gate,
    )
    print(
        _json_text(
            {
                "status": "blind labels revealed and scored exactly once",
                "reveal_utc": result["reveal_utc"],
                "prediction_bundle_sha256": result["prediction_bundle_sha256"],
                "point_prediction": result["point_prediction"],
                "conformal": result["conformal"],
            }
        )
    )


if __name__ == "__main__":
    main()
