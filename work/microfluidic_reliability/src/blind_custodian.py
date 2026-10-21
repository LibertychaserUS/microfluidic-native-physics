"""Create sealed-label artifacts for the prospective TX100 validation.

This is the only module in the blind workflow that is allowed to read the raw
MAT targets.  It deliberately emits no target values or target summaries to
stdout.  Development labels and blind labels are written to separate files;
the model-development program never accepts the sealed-label path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.io import loadmat


RAW_INPUT_COLUMNS = [
    "continuous_flow_ml_min",
    "dispersed_flow_ml_min",
    "equilibrium_interfacial_tension_mn_m",
    "surfactant_concentration_mm",
    "cmc_mm",
]
NORMALIZED_CONCENTRATION = "concentration_over_cmc"
TARGET_COLUMN = "target_diameter_mm"
ROW_ID_COLUMN = "opaque_row_id"
TX100_CMC_MM = 3.5
DEFAULT_EXPECTED_BLIND_COUNT = 87
UUID_NAMESPACE = uuid.UUID("8d4cf0c4-2d5e-4b8a-bef5-5c02a24fe9d7")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exclusive_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(text)


def _exclusive_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)


def _numeric_arrays(value: Any, prefix: str = "") -> Iterable[tuple[str, np.ndarray]]:
    """Yield numeric arrays from ordinary scipy ``loadmat`` structures.

    Keys and shapes may be used in ambiguity errors, but array values are never
    included.  MATLAB v7.3 files are rejected with an actionable message rather
    than silently interpreted with a different storage convention.
    """

    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).startswith("__"):
                continue
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _numeric_arrays(child, child_prefix)
        return
    if isinstance(value, np.ndarray) and value.dtype.names:
        for name in value.dtype.names:
            yield from _numeric_arrays(value[name], f"{prefix}.{name}")
        return
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        squeezed = np.asarray(value).squeeze()
        if squeezed.size:
            yield prefix, squeezed


def _orient_matrix(values: np.ndarray, columns: int) -> np.ndarray | None:
    values = np.asarray(values)
    if values.ndim != 2:
        return None
    if values.shape[1] == columns:
        return values
    if values.shape[0] == columns and values.shape[1] != columns:
        return values.T
    return None


def extract_six_column_table(path: Path, matrix_key: str | None = None) -> np.ndarray:
    """Extract five published inputs plus the output from one MAT file.

    The preferred representation is one numeric n-by-6 matrix.  A separate
    n-by-5 feature matrix and n-vector target are also accepted when their
    pairing is unambiguous.  Explicit ``matrix_key`` resolves files containing
    multiple six-column matrices.
    """

    try:
        payload = loadmat(path, simplify_cells=True)
    except NotImplementedError as exc:
        raise ValueError(
            f"{path.name} uses MATLAB v7.3/HDF5; convert it losslessly to a "
            "standard MAT file inside the custodian boundary before sealing"
        ) from exc

    arrays = list(_numeric_arrays(payload))
    if matrix_key is not None:
        matches = [values for key, values in arrays if key == matrix_key]
        if len(matches) != 1:
            raise ValueError(
                f"matrix key {matrix_key!r} was not unique in {path.name}; "
                f"available keys/shapes: {[(key, value.shape) for key, value in arrays]}"
            )
        matrix = _orient_matrix(matches[0], 6)
        if matrix is None:
            raise ValueError(f"matrix key {matrix_key!r} is not an n-by-6 array")
        return np.asarray(matrix, dtype=float)

    six_column = [
        (key, matrix)
        for key, value in arrays
        if (matrix := _orient_matrix(value, 6)) is not None
    ]
    if len(six_column) == 1:
        return np.asarray(six_column[0][1], dtype=float)
    if len(six_column) > 1:
        raise ValueError(
            f"ambiguous six-column arrays in {path.name}: "
            f"{[(key, matrix.shape) for key, matrix in six_column]}; use --matrix-key"
        )

    five_column = [
        (key, matrix)
        for key, value in arrays
        if (matrix := _orient_matrix(value, 5)) is not None
    ]
    vectors = [
        (key, np.asarray(value, dtype=float).reshape(-1))
        for key, value in arrays
        if np.asarray(value).ndim in (1, 2)
        and 1 in np.atleast_2d(value).shape
    ]
    pairings: list[tuple[str, str, np.ndarray]] = []
    for feature_key, feature_values in five_column:
        for target_key, target_values in vectors:
            if len(feature_values) == len(target_values):
                pairings.append(
                    (
                        feature_key,
                        target_key,
                        np.column_stack([feature_values, target_values]),
                    )
                )
    if len(pairings) == 1:
        return pairings[0][2]
    description = [(key, value.shape) for key, value in arrays]
    raise ValueError(
        f"could not identify one unambiguous five-input/one-target table in "
        f"{path.name}; available keys/shapes: {description}"
    )


def _validated_frame(matrix: np.ndarray, source_index: int) -> pd.DataFrame:
    if matrix.ndim != 2 or matrix.shape[1] != 6 or not len(matrix):
        raise ValueError("each source must contain a non-empty n-by-6 table")
    if not np.isfinite(matrix).all():
        raise ValueError("raw source contains non-finite values")
    frame = pd.DataFrame(matrix, columns=RAW_INPUT_COLUMNS + [TARGET_COLUMN])
    if (frame[[RAW_INPUT_COLUMNS[0], RAW_INPUT_COLUMNS[1]]] < 0).any().any():
        raise ValueError("flow rates must be non-negative")
    if (frame["equilibrium_interfacial_tension_mn_m"] < 0).any():
        raise ValueError("interfacial tension must be non-negative")
    if (frame[["surfactant_concentration_mm", "cmc_mm"]] < 0).any().any():
        raise ValueError("concentration and CMC must be non-negative")
    if (frame[TARGET_COLUMN] <= 0).any():
        raise ValueError("target diameters must be positive")
    frame.insert(0, "source_row_index", np.arange(len(frame), dtype=int))
    frame.insert(0, "source_index", int(source_index))
    frame.insert(
        0,
        ROW_ID_COLUMN,
        [
            str(uuid.uuid5(UUID_NAMESPACE, f"source-{source_index}:row-{row}"))
            for row in range(len(frame))
        ],
    )
    cmc = frame["cmc_mm"].to_numpy(dtype=float)
    concentration = frame["surfactant_concentration_mm"].to_numpy(dtype=float)
    normalized = np.divide(
        concentration,
        cmc,
        out=np.zeros_like(concentration, dtype=float),
        where=cmc != 0,
    )
    frame[NORMALIZED_CONCENTRATION] = normalized
    return frame


def create_sealed_artifacts(
    mat_paths: list[Path],
    output_dir: Path,
    protocol_path: Path,
    *,
    matrix_keys: list[str | None] | None = None,
    source_urls: list[str | None] | None = None,
    source_md5: list[str | None] | None = None,
    expected_blind_count: int | None = DEFAULT_EXPECTED_BLIND_COUNT,
) -> dict[str, Any]:
    """Seal TX100 labels and return a value-free custodian manifest."""

    if len(mat_paths) != 2:
        raise ValueError("the locked protocol requires exactly two MAT files")
    matrix_keys = matrix_keys or [None] * len(mat_paths)
    source_urls = source_urls or [None] * len(mat_paths)
    source_md5 = source_md5 or [None] * len(mat_paths)
    if not (len(matrix_keys) == len(source_urls) == len(source_md5) == 2):
        raise ValueError("per-source metadata lists must have two entries")
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)

    frames = [
        _validated_frame(extract_six_column_table(path, key), index)
        for index, (path, key) in enumerate(zip(mat_paths, matrix_keys))
    ]
    combined = pd.concat(frames, ignore_index=True)
    if combined[ROW_ID_COLUMN].duplicated().any():
        raise RuntimeError("opaque row identifiers are not unique")
    blind_mask = np.isclose(
        combined["cmc_mm"].to_numpy(dtype=float),
        TX100_CMC_MM,
        rtol=0.0,
        atol=1e-12,
    )
    blind_count = int(blind_mask.sum())
    if blind_count == 0:
        raise ValueError("no TX100 rows were identified by the locked CMC=3.5 rule")
    if expected_blind_count is not None and blind_count != expected_blind_count:
        raise ValueError(
            f"locked TX100 count mismatch: expected {expected_blind_count}, got {blind_count}"
        )

    metadata_columns = [ROW_ID_COLUMN, "source_index", "source_row_index"]
    public_input_columns = metadata_columns + RAW_INPUT_COLUMNS + [NORMALIZED_CONCENTRATION]
    development = combined.loc[~blind_mask, public_input_columns + [TARGET_COLUMN]].copy()
    blind_features = combined.loc[blind_mask, public_input_columns].copy()
    sealed_labels = combined.loc[blind_mask, [ROW_ID_COLUMN, TARGET_COLUMN]].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    development_path = output_dir / "development_labeled.csv"
    blind_features_path = output_dir / "blind_features.csv"
    sealed_labels_path = output_dir / "sealed_blind_labels.csv"
    manifest_path = output_dir / "custodian_manifest.json"
    _exclusive_csv(development_path, development)
    _exclusive_csv(blind_features_path, blind_features)
    _exclusive_csv(sealed_labels_path, sealed_labels)
    os.chmod(sealed_labels_path, 0o600)

    artifacts = {
        "development_labeled": {
            "file": development_path.name,
            "rows": int(len(development)),
            "sha256": sha256(development_path),
        },
        "blind_features": {
            "file": blind_features_path.name,
            "rows": int(len(blind_features)),
            "sha256": sha256(blind_features_path),
        },
        "sealed_blind_labels": {
            "file": sealed_labels_path.name,
            "rows": int(len(sealed_labels)),
            "sha256": sha256(sealed_labels_path),
            "permissions": "0600",
        },
    }
    manifest: dict[str, Any] = {
        "schema_version": "blind-custodian-v1",
        "protocol": {
            "file": protocol_path.name,
            "sha256": sha256(protocol_path),
        },
        "blind_rule": {"column": "cmc_mm", "equals_mm": TX100_CMC_MM},
        "raw_sources": [
            {
                "source_index": index,
                "file": path.name,
                "url": url,
                "zenodo_md5": md5,
                "local_sha256": sha256(path),
                "rows": int(len(frames[index])),
            }
            for index, (path, url, md5) in enumerate(
                zip(mat_paths, source_urls, source_md5)
            )
        ],
        "counts": {
            "combined": int(len(combined)),
            "development": int(len(development)),
            "blind": int(len(blind_features)),
        },
        "artifacts": artifacts,
    }
    _exclusive_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Seal TX100 labels without printing any target value or summary"
    )
    parser.add_argument("mat_files", nargs=2, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "work" / "microfluidic_reliability" / "blind_artifacts",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=root / "work" / "microfluidic_reliability" / "BLIND_VALIDATION_PROTOCOL.md",
    )
    parser.add_argument("--matrix-key", action="append", default=[])
    parser.add_argument("--source-url", action="append", default=[])
    parser.add_argument("--source-md5", action="append", default=[])
    parser.add_argument("--expected-blind-count", type=int, default=87)
    return parser.parse_args()


def _two_optional(values: list[str]) -> list[str | None]:
    if len(values) > 2:
        raise ValueError("option may be supplied at most twice")
    return values + [None] * (2 - len(values))


def main() -> None:
    args = parse_args()
    manifest = create_sealed_artifacts(
        args.mat_files,
        args.output_dir,
        args.protocol,
        matrix_keys=_two_optional(args.matrix_key),
        source_urls=_two_optional(args.source_url),
        source_md5=_two_optional(args.source_md5),
        expected_blind_count=args.expected_blind_count,
    )
    print(
        json.dumps(
            {
                "status": "sealed",
                "counts": manifest["counts"],
                "protocol_sha256": manifest["protocol"]["sha256"],
                "artifact_sha256": {
                    key: value["sha256"]
                    for key, value in manifest["artifacts"].items()
                },
                "raw_sha256": [
                    source["local_sha256"] for source in manifest["raw_sources"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
