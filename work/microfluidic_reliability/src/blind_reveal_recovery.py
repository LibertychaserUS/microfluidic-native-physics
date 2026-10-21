"""Recover the frozen blind evaluation after a JSON-only write failure.

The original one-time reveal gate was successfully consumed before labels were
read, but the final strict-JSON serialization rejected NaN values representing
unanswered target queries.  This recovery does not refit, rerank, retune, or
modify any pre-reveal artifact.  It verifies the original frozen hashes, reads
the already-unsealed labels, recomputes the same declared endpoints, converts
undefined non-finite query fields to JSON null, and writes one exclusive result.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from blind_custodian import ROW_ID_COLUMN, TARGET_COLUMN, sha256
from blind_predict import RANKING_COLUMNS
from blind_reveal import (
    _regression_endpoints,
    score_applicability,
    score_target_search,
    verify_opaque_ids,
    verify_prereveal_files,
)


ORIGINAL_FROZEN_REVEAL_SHA256 = (
    "46a289ad4369b10b56f8d1c233ba3a401769dec8fbcf392b1235a0baebd27a48"
)
ORIGINAL_FAILURE = (
    "Strict JSON serialization rejected NaN fields generated for unanswered "
    "target-search queries after all locked metrics had been computed."
)


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(child) for child in value]
    if isinstance(value, np.ndarray):
        return [_sanitize(child) for child in value.tolist()]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return None
        if not math.isfinite(number):
            raise ValueError("infinite value cannot enter the recovery artifact")
        return number
    if value is pd.NA:
        return None
    return value


def _exclusive_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        json.dump(
            _sanitize(payload),
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")


def recover(
    bundle_path: Path,
    hash_path: Path,
    sealed_labels_path: Path,
    custodian_manifest_path: Path,
    protocol_path: Path,
    gate_path: Path,
    original_output_path: Path,
    recovery_output_path: Path,
) -> dict[str, Any]:
    if not gate_path.is_file():
        raise FileNotFoundError("the original reveal gate was not consumed")
    if original_output_path.exists():
        raise FileExistsError(
            "the original reveal output exists; recovery is neither needed nor allowed"
        )
    source_dir = Path(__file__).resolve().parent
    if sha256(source_dir / "blind_reveal.py") != ORIGINAL_FROZEN_REVEAL_SHA256:
        raise ValueError("the original frozen reveal implementation has changed")

    bundle, bundle_hash, custodian = verify_prereveal_files(
        bundle_path, hash_path, protocol_path, custodian_manifest_path
    )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("prediction_bundle_sha256") != bundle_hash:
        raise ValueError("consumed gate does not belong to the frozen prediction bundle")

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
        raise ValueError("unsealed targets are non-finite or non-positive")

    lower = joined["conformal_lower_mm"].to_numpy(dtype=float)
    upper = joined["conformal_upper_mm"].to_numpy(dtype=float)
    covered = (actual >= lower) & (actual <= upper)
    recovered_utc = datetime.now(timezone.utc).isoformat()
    result: dict[str, Any] = {
        "schema_version": "blind-reveal-v1-with-serialization-recovery",
        "reveal_utc": gate.get("reveal_utc"),
        "recovered_utc": recovered_utc,
        "prediction_bundle_sha256": bundle_hash,
        "sealed_labels_sha256": expected_label_hash,
        "row_id_verification": "exact set equality and one-to-one join passed",
        "recovery_audit": {
            "reason": ORIGINAL_FAILURE,
            "original_gate_file": gate_path.name,
            "original_gate_sha256": sha256(gate_path),
            "original_frozen_reveal_sha256": ORIGINAL_FROZEN_REVEAL_SHA256,
            "original_output_absent": True,
            "pre_reveal_artifacts_modified": False,
            "model_refit_or_retuned": False,
            "ranking_recomputed": False,
            "metric_definitions_changed": False,
            "only_behavioral_change": "convert undefined NaN query fields to JSON null",
        },
        "point_prediction": _regression_endpoints(actual, predicted),
        "conformal": {
            "nominal_coverage": 0.90,
            "observed_coverage": float(np.mean(covered)),
            "mean_interval_width_mm": float(np.mean(upper - lower)),
            "guarantee_under_unseen_chemistry_asserted": False,
        },
        "applicability": score_applicability(joined),
        "target_search": score_target_search(joined, queries, rankings),
    }
    _exclusive_json(recovery_output_path, result)
    return result


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    blind_dir = root / "work" / "microfluidic_reliability" / "blind_artifacts"
    prereveal = blind_dir / "prereveal"
    parser = argparse.ArgumentParser(
        description="Recover the frozen blind result after the recorded JSON write failure"
    )
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
        "--gate", type=Path, default=blind_dir / "BLIND_REVEAL_CONSUMED.json"
    )
    parser.add_argument(
        "--original-output", type=Path, default=blind_dir / "blind_reveal_results.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=blind_dir / "blind_reveal_results_recovered.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = recover(
        args.bundle,
        args.bundle_hash,
        args.sealed_labels,
        args.custodian_manifest,
        args.protocol,
        args.gate,
        args.original_output,
        args.output,
    )
    print(
        json.dumps(
            _sanitize(
                {
                    "status": "blind result recovered without changing frozen predictions",
                    "reveal_utc": result["reveal_utc"],
                    "recovered_utc": result["recovered_utc"],
                    "prediction_bundle_sha256": result["prediction_bundle_sha256"],
                    "point_prediction": result["point_prediction"],
                    "conformal": result["conformal"],
                }
            ),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
