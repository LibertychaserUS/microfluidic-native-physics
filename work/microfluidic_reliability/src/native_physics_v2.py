"""Locked native-variable physics-guided V2 experiment.

This module implements ``PHYSICS_CONSTRAINED_V2_PROTOCOL.md``.  It is a
retrospective, nested leave-one-CMC-group-out study.  It is intentionally named
physics-guided rather than PINN: the public table has no PDE field observations
or collocation coordinates.

``--fast`` exists only to exercise the complete pipeline.  Fast artifacts are
marked non-inferential and must not be used as study results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

# Set deterministic CPU thread defaults before importing numerical runtimes.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from strict_json import dumps_strict, write_strict_json


PROTOCOL_NAME = "PHYSICS_CONSTRAINED_V2_PROTOCOL.md"
LOCKED_PROTOCOL_SHA256 = (
    "5e4a1e4c4a8bb344ed0d1393477a7807b69cc57517eb359a03aec3cbf6471d55"
)

ROW_ID = "opaque_row_id"
QC = "continuous_flow_ml_min"
QD = "dispersed_flow_ml_min"
GAMMA = "equilibrium_interfacial_tension_mn_m"
CONCENTRATION = "surfactant_concentration_mm"
CMC = "cmc_mm"
CONCENTRATION_OVER_CMC = "concentration_over_cmc"
TARGET = "target_diameter_mm"
SOURCE_INDEX = "source_index"
SOURCE_ROW_INDEX = "source_row_index"

MAIN_GROUPS = (0.0, 2.0, 3.5, 4.33, 11.0, 20.0)
QUARANTINED_GROUP = 0.60
LOCKED_SEEDS = (17, 42, 2024, 3407, 9001)
FULL_MAX_EPOCHS = 2000
FULL_PATIENCE = 150
FAST_SEEDS = (42,)
FAST_MAX_EPOCHS = 20
FAST_PATIENCE = 5
LAMBDA_P_GRID = (0.01, 0.1, 1.0)
LAMBDA_M_GRID = (0.01, 0.1)
MODEL_IDS = ("B0", "B1", "B2", "B3", "V2-H", "V2-P", "V2-M")
TARGETS_MM = tuple(np.round(np.arange(0.06, 0.181, 0.01), 2).tolist())
TOP_KS = (1, 3, 5)
DEVICE_QC_LIMITS_ML_MIN = (0.002, 0.4)
DEVICE_QD_LIMITS_ML_MIN = (0.001, 0.1)

RHO_C = 920.0
MU_C = 4.6e-3
RHO_D = 1132.0
MU_D = 6.8e-3
D_PRIMARY_M = 195e-6
D_PRIMARY_MM = D_PRIMARY_M * 1e3
D_SENSITIVITY_M = 190e-6
OVAL_AREA_M2 = math.pi * 195e-6 * 190e-6 / 4.0
RECTANGULAR_AREA_M2 = 195e-6 * 190e-6
EPSILON = 1e-4

NATIVE_COLUMNS = (QC, QD, GAMMA, CONCENTRATION, CMC, "is_no_surfactant")
NATIVE_TRANSFORM_COLUMNS = (
    "log_Qc",
    "log_Qd",
    "log_gamma_eq",
    "log1p_c",
    "log1p_CMC",
    "is_no_surfactant",
)
PHYSICS_MODEL_COLUMNS = (
    "log_flow_ratio",
    "log_Ca_c",
    "log_Ca_d",
    "log_Re_c",
    "log_Re_d",
    "log1p_c_star",
)
PUBLIC_INPUT_COLUMNS = (
    ROW_ID,
    SOURCE_INDEX,
    SOURCE_ROW_INDEX,
    QC,
    QD,
    GAMMA,
    CONCENTRATION,
    CMC,
    CONCENTRATION_OVER_CMC,
)
UNIT_CONTRACT = {
    QC: "mL/min",
    QD: "mL/min",
    GAMMA: "mN/m",
    CONCENTRATION: "mM",
    CMC: "mM",
    CONCENTRATION_OVER_CMC: "dimensionless",
    TARGET: "mm",
}


@dataclass(frozen=True)
class StudySettings:
    seeds: tuple[int, ...]
    max_epochs: int
    patience: int
    run_mode: str
    protocol_compliant: bool


def study_settings(fast: bool) -> StudySettings:
    if fast:
        return StudySettings(
            seeds=FAST_SEEDS,
            max_epochs=FAST_MAX_EPOCHS,
            patience=FAST_PATIENCE,
            run_mode="non_inferential_fast_smoke",
            protocol_compliant=False,
        )
    return StudySettings(
        seeds=LOCKED_SEEDS,
        max_epochs=FULL_MAX_EPOCHS,
        patience=FULL_PATIENCE,
        run_mode="locked_full_study",
        protocol_compliant=True,
    )


def beneficial_claim_allowed(
    *, protocol_compliant: bool, locked_conditions: dict[str, bool]
) -> bool:
    return bool(protocol_compliant and all(locked_conditions.values()))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    write_strict_json(path, payload, sort_keys=True)


def assert_protocol_lock(project_dir: Path) -> None:
    observed = sha256_file(project_dir / PROTOCOL_NAME)
    if observed != LOCKED_PROTOCOL_SHA256:
        raise RuntimeError(
            f"protocol hash mismatch: expected {LOCKED_PROTOCOL_SHA256}, got {observed}"
        )
    sidecar = (project_dir / "PHYSICS_CONSTRAINED_V2_PROTOCOL.sha256").read_text(
        encoding="utf-8"
    )
    if LOCKED_PROTOCOL_SHA256 not in sidecar:
        raise RuntimeError("protocol SHA sidecar does not contain the locked digest")


@dataclass(frozen=True)
class JoinedData:
    full: pd.DataFrame
    main: pd.DataFrame
    quarantine: pd.DataFrame
    audit: dict[str, Any]


def _validate_feature_schema(frame: pd.DataFrame, *, target_required: bool) -> None:
    expected = list(PUBLIC_INPUT_COLUMNS) + ([TARGET] if target_required else [])
    if list(frame.columns) != expected:
        raise ValueError(
            f"schema/order mismatch: expected {expected}, observed {list(frame.columns)}"
        )
    if frame[ROW_ID].isna().any() or frame[ROW_ID].duplicated().any():
        raise ValueError("row identifiers must be present and unique")
    numeric = frame.drop(columns=[ROW_ID]).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("input table contains non-finite numeric values")
    if (frame[[QC, QD, GAMMA]] <= 0).any().any():
        raise ValueError("flows and equilibrium interfacial tension must be positive")
    if (frame[[CONCENTRATION, CMC]] < 0).any().any():
        raise ValueError("concentration and CMC cannot be negative")
    if target_required and (frame[TARGET] <= 0).any():
        raise ValueError("diameter outcomes must be positive")


def _validate_concentration_identity(frame: pd.DataFrame) -> None:
    cmc = frame[CMC].to_numpy(float)
    concentration = frame[CONCENTRATION].to_numpy(float)
    expected = np.divide(
        concentration,
        cmc,
        out=np.zeros_like(concentration),
        where=cmc > 0,
    )
    observed = frame[CONCENTRATION_OVER_CMC].to_numpy(float)
    if not np.allclose(expected, observed, rtol=1e-10, atol=1e-12):
        mismatch = np.flatnonzero(~np.isclose(expected, observed, rtol=1e-10, atol=1e-12))
        raise ValueError(f"concentration/CMC identity failed at {len(mismatch)} rows")
    no_surf = cmc == 0
    if np.any(concentration[no_surf] != 0):
        raise ValueError("CMC=0 rows must have zero surfactant concentration")


def _strict_json_load(path: Path) -> Any:
    def reject_constant(token: str) -> None:
        raise ValueError(f"non-standard JSON constant in {path.name}: {token}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _verify_custodian_lock(
    blind_artifact_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest_path = blind_artifact_dir / "custodian_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("custodian manifest is required before joining data")
    manifest = _strict_json_load(manifest_path)
    if manifest.get("schema_version") != "blind-custodian-v1":
        raise ValueError("unsupported custodian manifest schema")
    expected_entries = {
        "development_labeled": "development_labeled.csv",
        "blind_features": "blind_features.csv",
        "sealed_blind_labels": "sealed_blind_labels.csv",
    }
    artifact_evidence: dict[str, Any] = {}
    for key, locked_file in expected_entries.items():
        entry = manifest.get("artifacts", {}).get(key)
        if not isinstance(entry, dict) or entry.get("file") != locked_file:
            raise ValueError(f"custodian artifact declaration mismatch for {key}")
        path = blind_artifact_dir / locked_file
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_sha = str(entry.get("sha256", ""))
        observed_sha = sha256_file(path)
        if observed_sha != expected_sha:
            raise ValueError(
                f"custodian artifact lock mismatch for {locked_file}: "
                f"expected {expected_sha}, observed {observed_sha}"
            )
        artifact_evidence[key] = {
            "file": locked_file,
            "expected_sha256": expected_sha,
            "observed_sha256": observed_sha,
            "sha256_match": True,
            "expected_rows": int(entry.get("rows")),
        }

    raw_dir = blind_artifact_dir.parent / "blind_raw"
    raw_evidence: dict[str, Any] = {}
    raw_entries = manifest.get("raw_sources")
    if not isinstance(raw_entries, list) or len(raw_entries) != 2:
        raise ValueError("custodian manifest must declare exactly two raw MAT sources")
    for entry in raw_entries:
        file_name = str(entry.get("file", ""))
        path = raw_dir / file_name
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_sha = str(entry.get("local_sha256", ""))
        observed_sha = sha256_file(path)
        if observed_sha != expected_sha:
            raise ValueError(
                f"raw MAT lock mismatch for {file_name}: expected {expected_sha}, "
                f"observed {observed_sha}"
            )
        raw_evidence[file_name] = {
            "expected_sha256": expected_sha,
            "observed_sha256": observed_sha,
            "sha256_match": True,
            "expected_rows": int(entry.get("rows")),
            "source_index": int(entry.get("source_index")),
            "url": entry.get("url"),
            "zenodo_md5": entry.get("zenodo_md5"),
        }
    return manifest, artifact_evidence, raw_evidence


def load_joined_public_data(blind_artifact_dir: Path) -> JoinedData:
    manifest, artifact_evidence, raw_evidence = _verify_custodian_lock(
        blind_artifact_dir
    )
    development_path = blind_artifact_dir / "development_labeled.csv"
    features_path = blind_artifact_dir / "blind_features.csv"
    labels_path = blind_artifact_dir / "sealed_blind_labels.csv"
    development = pd.read_csv(development_path)
    blind_features = pd.read_csv(features_path)
    sealed_labels = pd.read_csv(labels_path)

    frames_by_key = {
        "development_labeled": development,
        "blind_features": blind_features,
        "sealed_blind_labels": sealed_labels,
    }
    for key, frame in frames_by_key.items():
        observed_rows = int(len(frame))
        expected_rows = int(artifact_evidence[key]["expected_rows"])
        if observed_rows != expected_rows:
            raise ValueError(
                f"custodian row lock mismatch for {key}: expected {expected_rows}, "
                f"observed {observed_rows}"
            )
        artifact_evidence[key]["observed_rows"] = observed_rows
        artifact_evidence[key]["rows_match"] = True

    _validate_feature_schema(development, target_required=True)
    _validate_feature_schema(blind_features, target_required=False)
    if list(sealed_labels.columns) != [ROW_ID, TARGET]:
        raise ValueError("sealed-label schema must be exactly row ID plus target")
    if sealed_labels[ROW_ID].isna().any() or sealed_labels[ROW_ID].duplicated().any():
        raise ValueError("sealed labels have missing or duplicate row identifiers")
    if not np.isfinite(sealed_labels[TARGET].to_numpy(float)).all():
        raise ValueError("sealed labels contain non-finite targets")
    if set(blind_features[ROW_ID]) != set(sealed_labels[ROW_ID]):
        raise ValueError("blind feature and sealed label row IDs do not match exactly")
    if set(development[ROW_ID]).intersection(blind_features[ROW_ID]):
        raise ValueError("development and blind row identifiers overlap")

    revealed = blind_features.merge(
        sealed_labels,
        on=ROW_ID,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    full = pd.concat([development, revealed], ignore_index=True, sort=False)
    full = full.sort_values([SOURCE_INDEX, SOURCE_ROW_INDEX], kind="stable").reset_index(
        drop=True
    )
    _validate_feature_schema(full, target_required=True)
    _validate_concentration_identity(full)

    expected_counts = {0.0: 63, 0.6: 2, 2.0: 30, 3.5: 87, 4.33: 15, 11.0: 178, 20.0: 93}
    counts = {float(k): int(v) for k, v in full[CMC].value_counts().items()}
    if counts != expected_counts:
        raise ValueError(f"unexpected CMC group counts: {counts}")
    quarantine = full.loc[np.isclose(full[CMC], QUARANTINED_GROUP)].copy()
    main = full.loc[~np.isclose(full[CMC], QUARANTINED_GROUP)].copy()
    if tuple(sorted(float(x) for x in main[CMC].unique())) != MAIN_GROUPS:
        raise ValueError("main chemistry groups differ from the locked protocol")

    audit = {
        "join_type": "one_to_one opaque_row_id merge, followed by stable source row order",
        "development_rows": int(len(development)),
        "revealed_tx100_rows": int(len(revealed)),
        "joined_rows": int(len(full)),
        "main_rows": int(len(main)),
        "quarantined_rows": int(len(quarantine)),
        "group_counts": {format(k, ".12g"): counts[k] for k in sorted(counts)},
        "input_hashes": {
            development_path.name: sha256_file(development_path),
            features_path.name: sha256_file(features_path),
            labels_path.name: sha256_file(labels_path),
        },
        "custodian_lock": {
            "manifest_sha256": sha256_file(
                blind_artifact_dir / "custodian_manifest.json"
            ),
            "schema_version": manifest["schema_version"],
            "artifacts": artifact_evidence,
            "raw_sources": raw_evidence,
            "all_expected_observed_checks_passed": True,
        },
        "schema_evidence": {
            "development_expected": list(PUBLIC_INPUT_COLUMNS) + [TARGET],
            "development_observed": list(development.columns),
            "blind_features_expected": list(PUBLIC_INPUT_COLUMNS),
            "blind_features_observed": list(blind_features.columns),
            "sealed_labels_expected": [ROW_ID, TARGET],
            "sealed_labels_observed": list(sealed_labels.columns),
        },
        "unit_evidence": {
            "declared_units": UNIT_CONTRACT,
            "positive_columns_checked": [QC, QD, GAMMA, TARGET],
            "nonnegative_columns_checked": [CONCENTRATION, CMC],
            "SI_conversion_contract": {
                "mL/min_to_m3/s": 1e-6 / 60.0,
                "mN/m_to_N/m": 1e-3,
            },
        },
        "tx100_status": "post_reveal_retrospective",
    }
    return JoinedData(full=full, main=main, quarantine=quarantine, audit=audit)


def native_state(frame: pd.DataFrame) -> pd.DataFrame:
    state = frame[[QC, QD, GAMMA, CONCENTRATION, CMC]].copy()
    state["is_no_surfactant"] = frame[CMC].eq(0.0).astype(float)
    return state.loc[:, NATIVE_COLUMNS]


def native_transforms(frame: pd.DataFrame) -> pd.DataFrame:
    state = native_state(frame)
    if (state[[QC, QD, GAMMA]] <= 0).any().any():
        raise ValueError("native flow and tension variables must be positive")
    if (state[[CONCENTRATION, CMC]] < 0).any().any():
        raise ValueError("native material variables cannot be negative")
    return pd.DataFrame(
        {
            "log_Qc": np.log(state[QC].to_numpy(float)),
            "log_Qd": np.log(state[QD].to_numpy(float)),
            "log_gamma_eq": np.log(state[GAMMA].to_numpy(float)),
            "log1p_c": np.log1p(state[CONCENTRATION].to_numpy(float)),
            "log1p_CMC": np.log1p(state[CMC].to_numpy(float)),
            "is_no_surfactant": state["is_no_surfactant"].to_numpy(float),
        },
        index=frame.index,
    )


def physics_layer(
    frame: pd.DataFrame,
    *,
    diameter_m: float = D_PRIMARY_M,
    area_m2: float = OVAL_AREA_M2,
) -> pd.DataFrame:
    """Recompute every derived physical quantity from native variables."""

    state = native_state(frame)
    qc = state[QC].to_numpy(float) * 1e-6 / 60.0
    qd = state[QD].to_numpy(float) * 1e-6 / 60.0
    gamma = state[GAMMA].to_numpy(float) * 1e-3
    if np.any(qc <= 0) or np.any(qd <= 0) or np.any(gamma <= 0):
        raise ValueError("SI physics layer requires positive flow and tension")
    cmc = state[CMC].to_numpy(float)
    concentration = state[CONCENTRATION].to_numpy(float)
    c_star = np.divide(concentration, cmc, out=np.zeros_like(cmc), where=cmc > 0)
    result = pd.DataFrame(
        {
            "Qc_m3_s": qc,
            "Qd_m3_s": qd,
            "gamma_N_m": gamma,
            "flow_ratio": qd / qc,
            "Ca_c": MU_C * qc / (gamma * area_m2),
            "Ca_d": MU_D * qd / (gamma * area_m2),
            "Re_c": RHO_C * qc * diameter_m / (MU_C * area_m2),
            "Re_d": RHO_D * qd * diameter_m / (MU_D * area_m2),
            "c_star": c_star,
        },
        index=frame.index,
    )
    if not np.isfinite(result.to_numpy()).all() or (result <= 0).loc[:, [
        "Qc_m3_s", "Qd_m3_s", "gamma_N_m", "flow_ratio", "Ca_c", "Ca_d", "Re_c", "Re_d"
    ]].any().any():
        raise ValueError("deterministic physics layer produced invalid values")
    return result


def physics_model_transforms(frame: pd.DataFrame) -> pd.DataFrame:
    phys = physics_layer(frame)
    return pd.DataFrame(
        {
            "log_flow_ratio": np.log(phys["flow_ratio"]),
            "log_Ca_c": np.log(phys["Ca_c"]),
            "log_Ca_d": np.log(phys["Ca_d"]),
            "log_Re_c": np.log(phys["Re_c"]),
            "log_Re_d": np.log(phys["Re_d"]),
            "log1p_c_star": np.log1p(phys["c_star"]),
        },
        index=frame.index,
    )


def semi_empirical_anchor(
    frame: pd.DataFrame,
    *,
    diameter_m: float = D_PRIMARY_M,
    area_m2: float = OVAL_AREA_M2,
) -> tuple[np.ndarray, np.ndarray]:
    phys = physics_layer(frame, diameter_m=diameter_m, area_m2=area_m2)
    y_unclipped = 0.642 * np.power(phys["flow_ratio"].to_numpy(float), 0.188) * np.power(
        1.0 / phys["Ca_c"].to_numpy(float), 0.161
    )
    valid = np.isfinite(y_unclipped) & (y_unclipped > 0.0) & (y_unclipped < 1.0)
    return y_unclipped, valid


def stable_sigmoid(value: np.ndarray | float) -> np.ndarray:
    x = np.asarray(value, dtype=float)
    out = np.empty_like(x)
    nonnegative = x >= 0
    out[nonnegative] = 1.0 / (1.0 + np.exp(-np.minimum(x[nonnegative], 745.0)))
    exp_x = np.exp(np.maximum(x[~nonnegative], -745.0))
    out[~nonnegative] = exp_x / (1.0 + exp_x)
    return out


def bounded_normalized_from_logit(value: np.ndarray | float) -> np.ndarray:
    return EPSILON + (1.0 - 2.0 * EPSILON) * stable_sigmoid(value)


def bounded_diameter_from_logit(value: np.ndarray | float) -> np.ndarray:
    return D_PRIMARY_MM * bounded_normalized_from_logit(value)


def bounded_interval_from_logit(
    center: np.ndarray | float, radius: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    center_array = np.asarray(center, dtype=float)
    radius_array = np.asarray(radius, dtype=float)
    lower = bounded_diameter_from_logit(center_array - radius_array)
    upper = bounded_diameter_from_logit(center_array + radius_array)
    # Extreme logits can saturate both endpoints to the same representable
    # float. Preserve the mathematical strict ordering in the serialized
    # artifact with the nearest representable value toward zero.
    lower = np.where(lower < upper, lower, np.nextafter(upper, 0.0))
    return lower, upper


def normalized_to_logit(y: np.ndarray | float) -> np.ndarray:
    value = np.asarray(y, dtype=float)
    scaled = (np.clip(value, EPSILON, 1.0 - EPSILON) - EPSILON) / (
        1.0 - 2.0 * EPSILON
    )
    scaled = np.clip(scaled, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    return np.log(scaled / (1.0 - scaled))


def outer_loco_split(frame: pd.DataFrame, heldout_group: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_mask = np.isclose(frame[CMC].to_numpy(float), heldout_group)
    train = frame.loc[~test_mask].copy()
    test = frame.loc[test_mask].copy()
    if test.empty or train.empty:
        raise ValueError(f"empty outer LOCO partition for group {heldout_group}")
    if set(train[ROW_ID]).intersection(test[ROW_ID]):
        raise AssertionError("outer train/test row overlap")
    if any(np.isclose(train[CMC], heldout_group)):
        raise AssertionError("held-out chemistry leaked into outer training")
    return train, test


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting interop threads only before parallel work starts.
        pass


class TwoBranchRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.native = nn.Sequential(
            nn.Linear(6, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU()
        )
        self.physics = nn.Sequential(
            nn.Linear(6, 16), nn.SiLU(), nn.Linear(16, 16), nn.SiLU()
        )
        self.head = nn.Sequential(nn.Linear(48, 16), nn.SiLU(), nn.Linear(16, 1))

    def forward(self, native: torch.Tensor, physics: torch.Tensor) -> torch.Tensor:
        joined = torch.cat((self.native(native), self.physics(physics)), dim=1)
        return self.head(joined).squeeze(1)


@dataclass
class NeuralFit:
    model: TwoBranchRegressor
    native_scaler: StandardScaler
    physics_scaler: StandardScaler
    kind: str
    best_epoch: int
    best_validation_mape: float
    lambda_p: float
    lambda_m: float
    train_support: dict[str, tuple[float, float]]


def _tensor(values: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32, device="cpu")


def _scaled_inputs(
    frame: pd.DataFrame,
    native_scaler: StandardScaler,
    physics_scaler: StandardScaler,
) -> tuple[torch.Tensor, torch.Tensor]:
    native = native_transforms(frame).to_numpy(float)
    phys = physics_model_transforms(frame).to_numpy(float)
    return _tensor(native_scaler.transform(native)), _tensor(physics_scaler.transform(phys))


def _torch_anchor(frame: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    y_anchor, valid = semi_empirical_anchor(frame)
    # An out-of-regime semi-empirical value is not merely exempt from the
    # residual penalty: it must not enter the prediction path at all.  Zero is
    # the neutral no-anchor logit (the bounded midpoint) and the validity mask
    # is applied again inside ``_network_output`` as a second leakage barrier.
    anchor_logit = np.where(
        valid,
        normalized_to_logit(np.clip(y_anchor, EPSILON, 1.0 - EPSILON)),
        0.0,
    )
    return _tensor(anchor_logit), torch.as_tensor(valid, dtype=torch.bool, device="cpu")


def _network_output(
    model: TwoBranchRegressor,
    kind: str,
    native_x: torch.Tensor,
    phys_x: torch.Tensor,
    anchor_logit: torch.Tensor,
    anchor_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = model(native_x, phys_x)
    if kind == "B3":
        y_norm = raw
        z_pred = torch.logit(torch.clamp((y_norm - EPSILON) / (1 - 2 * EPSILON), 1e-7, 1 - 1e-7))
        delta = raw
    elif kind == "V2-H":
        z_pred = raw
        y_norm = EPSILON + (1 - 2 * EPSILON) * torch.sigmoid(z_pred)
        delta = raw
    elif kind in {"V2-P", "V2-M"}:
        if anchor_valid is None:
            raise ValueError("physics-residual models require an anchor-validity mask")
        delta = raw
        gated_anchor = torch.where(
            anchor_valid, anchor_logit, torch.zeros_like(anchor_logit)
        )
        z_pred = gated_anchor + delta
        y_norm = EPSILON + (1 - 2 * EPSILON) * torch.sigmoid(z_pred)
    else:
        raise ValueError(f"unknown neural kind: {kind}")
    return y_norm, z_pred, delta


def _support(frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    return {
        name: (float(frame[name].min()), float(frame[name].max()))
        for name in (QC, QD, GAMMA, CONCENTRATION, CMC)
    }


@dataclass(frozen=True)
class MonotonicPair:
    low_native: torch.Tensor
    low_physics: torch.Tensor
    low_anchor: torch.Tensor
    low_anchor_valid: torch.Tensor
    high_native: torch.Tensor
    high_physics: torch.Tensor
    high_anchor: torch.Tensor
    high_anchor_valid: torch.Tensor
    desired: str


def _prepare_monotonic_inputs(
    train_frame: pd.DataFrame,
    native_scaler: StandardScaler,
    physics_scaler: StandardScaler,
    support: dict[str, tuple[float, float]],
) -> list[MonotonicPair]:
    prepared: list[MonotonicPair] = []
    for variable, desired in ((QD, "increasing"), (QC, "decreasing")):
        lower_frame = train_frame.copy()
        upper_frame = train_frame.copy()
        lower_frame[variable] *= 0.95
        upper_frame[variable] *= 1.05
        minimum, maximum = support[variable]
        mask = (
            (lower_frame[variable].to_numpy(float) >= minimum)
            & (upper_frame[variable].to_numpy(float) <= maximum)
        )
        if not np.any(mask):
            continue
        low = lower_frame.loc[mask]
        high = upper_frame.loc[mask]
        low_native, low_phys = _scaled_inputs(low, native_scaler, physics_scaler)
        high_native, high_phys = _scaled_inputs(high, native_scaler, physics_scaler)
        low_anchor, low_anchor_valid = _torch_anchor(low)
        high_anchor, high_anchor_valid = _torch_anchor(high)
        prepared.append(
            MonotonicPair(
                low_native=low_native,
                low_physics=low_phys,
                low_anchor=low_anchor,
                low_anchor_valid=low_anchor_valid,
                high_native=high_native,
                high_physics=high_phys,
                high_anchor=high_anchor,
                high_anchor_valid=high_anchor_valid,
                desired=desired,
            )
        )
    return prepared


def _monotonic_penalty_precomputed(
    model: TwoBranchRegressor,
    kind: str,
    prepared: Sequence[MonotonicPair],
) -> torch.Tensor:
    penalties: list[torch.Tensor] = []
    for pair in prepared:
        low_y, _, _ = _network_output(
            model,
            kind,
            pair.low_native,
            pair.low_physics,
            pair.low_anchor,
            pair.low_anchor_valid,
        )
        high_y, _, _ = _network_output(
            model,
            kind,
            pair.high_native,
            pair.high_physics,
            pair.high_anchor,
            pair.high_anchor_valid,
        )
        if pair.desired == "increasing":
            violation = torch.relu(low_y - high_y)
        else:
            violation = torch.relu(high_y - low_y)
        penalties.append(torch.mean(violation.square()))
    if not penalties:
        return torch.zeros((), dtype=torch.float32)
    return torch.stack(penalties).mean()


def _mape_percent(y_true_mm: np.ndarray, y_pred_mm: np.ndarray) -> float:
    truth = np.asarray(y_true_mm, dtype=float)
    pred = np.asarray(y_pred_mm, dtype=float)
    return float(100.0 * np.mean(np.abs(pred - truth) / truth))


def fit_neural_with_validation(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    *,
    kind: str,
    seed: int,
    lambda_p: float = 0.0,
    lambda_m: float = 0.0,
    max_epochs: int = 2000,
    patience: int = 150,
) -> NeuralFit:
    """Fit using only the supplied inner training and validation chemistries."""

    if set(train_frame[CMC].unique()).intersection(validation_frame[CMC].unique()):
        raise ValueError("inner chemistry leakage")
    set_deterministic_seed(seed)
    native_scaler = StandardScaler().fit(native_transforms(train_frame).to_numpy(float))
    physics_scaler = StandardScaler().fit(
        physics_model_transforms(train_frame).to_numpy(float)
    )
    train_native, train_phys = _scaled_inputs(train_frame, native_scaler, physics_scaler)
    val_native, val_phys = _scaled_inputs(validation_frame, native_scaler, physics_scaler)
    train_anchor, train_anchor_valid = _torch_anchor(train_frame)
    val_anchor, val_anchor_valid = _torch_anchor(validation_frame)
    y_train = _tensor(train_frame[TARGET].to_numpy(float) / D_PRIMARY_MM)
    y_validation_mm = validation_frame[TARGET].to_numpy(float)
    z_train = _tensor(normalized_to_logit(y_train.numpy()))
    support = _support(train_frame)
    monotonic_inputs = (
        _prepare_monotonic_inputs(
            train_frame, native_scaler, physics_scaler, support
        )
        if kind == "V2-M"
        else []
    )

    model = TwoBranchRegressor().cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    huber = nn.SmoothL1Loss(beta=0.1)
    best_state: dict[str, torch.Tensor] | None = None
    best_mape = float("inf")
    best_epoch = 0
    stale = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        y_pred, z_pred, delta = _network_output(
            model, kind, train_native, train_phys, train_anchor, train_anchor_valid
        )
        if kind in {"V2-P", "V2-M"}:
            loss = huber(z_pred, z_train)
            if bool(train_anchor_valid.any()):
                loss = loss + lambda_p * torch.mean(delta[train_anchor_valid].square())
        else:
            loss = huber(y_pred, y_train)
        if kind == "V2-M":
            loss = loss + lambda_m * _monotonic_penalty_precomputed(
                model, kind, monotonic_inputs
            )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_y, _, _ = _network_output(
                model, kind, val_native, val_phys, val_anchor, val_anchor_valid
            )
            val_pred_mm = val_y.numpy().astype(float) * D_PRIMARY_MM
        validation_mape = _mape_percent(y_validation_mm, val_pred_mm)
        if validation_mape < best_mape - 1e-12:
            best_mape = validation_mape
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is None:
        raise RuntimeError("neural early stopping failed to capture a finite state")
    model.load_state_dict(best_state)
    model.eval()
    return NeuralFit(
        model=model,
        native_scaler=native_scaler,
        physics_scaler=physics_scaler,
        kind=kind,
        best_epoch=best_epoch,
        best_validation_mape=best_mape,
        lambda_p=lambda_p,
        lambda_m=lambda_m,
        train_support=support,
    )


def fit_neural_fixed_epochs(
    train_frame: pd.DataFrame,
    *,
    kind: str,
    seed: int,
    epochs: int,
    lambda_p: float = 0.0,
    lambda_m: float = 0.0,
) -> NeuralFit:
    if epochs < 1:
        raise ValueError("fixed training epochs must be positive")
    set_deterministic_seed(seed)
    native_scaler = StandardScaler().fit(native_transforms(train_frame).to_numpy(float))
    physics_scaler = StandardScaler().fit(
        physics_model_transforms(train_frame).to_numpy(float)
    )
    train_native, train_phys = _scaled_inputs(train_frame, native_scaler, physics_scaler)
    train_anchor, train_anchor_valid = _torch_anchor(train_frame)
    y_train = _tensor(train_frame[TARGET].to_numpy(float) / D_PRIMARY_MM)
    z_train = _tensor(normalized_to_logit(y_train.numpy()))
    support = _support(train_frame)
    monotonic_inputs = (
        _prepare_monotonic_inputs(
            train_frame, native_scaler, physics_scaler, support
        )
        if kind == "V2-M"
        else []
    )
    model = TwoBranchRegressor().cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    huber = nn.SmoothL1Loss(beta=0.1)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        y_pred, z_pred, delta = _network_output(
            model, kind, train_native, train_phys, train_anchor, train_anchor_valid
        )
        if kind in {"V2-P", "V2-M"}:
            loss = huber(z_pred, z_train)
            if bool(train_anchor_valid.any()):
                loss = loss + lambda_p * torch.mean(delta[train_anchor_valid].square())
        else:
            loss = huber(y_pred, y_train)
        if kind == "V2-M":
            loss = loss + lambda_m * _monotonic_penalty_precomputed(
                model, kind, monotonic_inputs
            )
        loss.backward()
        optimizer.step()
    model.eval()
    return NeuralFit(
        model=model,
        native_scaler=native_scaler,
        physics_scaler=physics_scaler,
        kind=kind,
        best_epoch=epochs,
        best_validation_mape=float("nan"),
        lambda_p=lambda_p,
        lambda_m=lambda_m,
        train_support=support,
    )


def predict_neural(fit: NeuralFit, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    native_x, physics_x = _scaled_inputs(frame, fit.native_scaler, fit.physics_scaler)
    anchor, anchor_valid = _torch_anchor(frame)
    fit.model.eval()
    with torch.no_grad():
        y_norm, z_pred, delta = _network_output(
            fit.model, fit.kind, native_x, physics_x, anchor, anchor_valid
        )
    return {
        "prediction_mm": y_norm.numpy().astype(float) * D_PRIMARY_MM,
        "prediction_normalized": y_norm.numpy().astype(float),
        "prediction_logit": z_pred.numpy().astype(float),
        "residual_delta": delta.numpy().astype(float),
        "anchor_valid": anchor_valid.numpy().astype(bool),
    }


def monotonic_violation_rate(fit: NeuralFit, frame: pd.DataFrame) -> float:
    violations: list[np.ndarray] = []
    for variable, desired in ((QD, "increasing"), (QC, "decreasing")):
        low = frame.copy()
        high = frame.copy()
        low[variable] *= 0.95
        high[variable] *= 1.05
        minimum, maximum = fit.train_support[variable]
        mask = (
            (low[variable].to_numpy(float) >= minimum)
            & (high[variable].to_numpy(float) <= maximum)
        )
        if not np.any(mask):
            continue
        low_pred = predict_neural(fit, low.loc[mask])["prediction_mm"]
        high_pred = predict_neural(fit, high.loc[mask])["prediction_mm"]
        if desired == "increasing":
            violations.append(low_pred > high_pred)
        else:
            violations.append(high_pred > low_pred)
    if not violations:
        return float("nan")
    return float(np.mean(np.concatenate(violations)))


def make_xgb(seed: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.3,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=1,
    )


def b1_features(frame: pd.DataFrame) -> np.ndarray:
    cmc = frame[CMC].to_numpy(float)
    concentration = frame[CONCENTRATION].to_numpy(float)
    ratio = np.divide(concentration, cmc, out=np.zeros_like(cmc), where=cmc > 0)
    return np.column_stack(
        [
            frame[QC].to_numpy(float),
            frame[QD].to_numpy(float),
            frame[GAMMA].to_numpy(float),
            ratio,
        ]
    )


@dataclass
class XGBFit:
    model: XGBRegressor
    scaler: StandardScaler | None
    kind: str


def fit_xgb(train: pd.DataFrame, *, kind: str, seed: int) -> XGBFit:
    model = make_xgb(seed)
    if kind == "B1":
        scaler = None
        features = b1_features(train)
        # Exact sealed-V1 contract: the frozen implementation fitted raw
        # diameter in millimetres.  Scaling d/D and multiplying predictions
        # back is not equivalent under XGBoost regularization.
        target = train[TARGET].to_numpy(float)
    elif kind == "B2":
        scaler = StandardScaler().fit(native_transforms(train).to_numpy(float))
        features = scaler.transform(native_transforms(train).to_numpy(float))
        # Keep the response contract identical to B1 so this ablation isolates
        # only the replacement of c/CMC by separate native c and CMC inputs.
        target = train[TARGET].to_numpy(float)
    else:
        raise ValueError(f"unknown XGBoost kind: {kind}")
    model.fit(features, target)
    return XGBFit(model=model, scaler=scaler, kind=kind)


def predict_xgb(fit: XGBFit, frame: pd.DataFrame) -> np.ndarray:
    if fit.kind == "B1":
        features = b1_features(frame)
        return fit.model.predict(features).astype(float)
    else:
        assert fit.scaler is not None
        features = fit.scaler.transform(native_transforms(frame).to_numpy(float))
        return fit.model.predict(features).astype(float)


def conformal_higher_quantile(values: np.ndarray, *, alpha: float = 0.10) -> float:
    residuals = np.asarray(values, dtype=float)
    if not len(residuals) or not np.isfinite(residuals).all():
        raise ValueError("conformal residuals must be finite and non-empty")
    rank = int(math.ceil((len(residuals) + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), len(residuals))
    return float(np.sort(residuals)[rank - 1])


@dataclass(frozen=True)
class InnerSelection:
    lambda_p: float
    lambda_m: float
    selected_epoch: int
    conformal_radius_logit: float
    records: list[dict[str, Any]]


def _hyperparameter_grid(kind: str) -> list[tuple[float, float]]:
    if kind in {"B3", "V2-H"}:
        return [(0.0, 0.0)]
    if kind == "V2-P":
        return [(value, 0.0) for value in LAMBDA_P_GRID]
    if kind == "V2-M":
        return [(p, m) for p in LAMBDA_P_GRID for m in LAMBDA_M_GRID]
    raise ValueError(kind)


def choose_hyperparameter_candidate(
    macro_mape_by_candidate: dict[tuple[float, float], float]
) -> tuple[float, float]:
    """Choose minimum MAPE; exact ties prefer larger lambda_p then lambda_m."""

    if not macro_mape_by_candidate:
        raise ValueError("hyperparameter candidate scores cannot be empty")
    if not np.isfinite(list(macro_mape_by_candidate.values())).all():
        raise ValueError("hyperparameter candidate scores must be finite")
    return min(
        macro_mape_by_candidate,
        key=lambda key: (macro_mape_by_candidate[key], -key[0], -key[1]),
    )


def select_neural_inner_loco(
    outer_train: pd.DataFrame,
    *,
    outer_group: float,
    kind: str,
    seed: int,
    max_epochs: int,
    patience: int,
) -> InnerSelection:
    groups = tuple(sorted(float(value) for value in outer_train[CMC].unique()))
    if len(groups) != 5:
        raise ValueError(f"outer training must contain five chemistry groups, got {groups}")
    candidates: dict[tuple[float, float], dict[str, Any]] = {}
    all_records: list[dict[str, Any]] = []
    for lambda_p, lambda_m in _hyperparameter_grid(kind):
        fold_mapes: list[float] = []
        fold_epochs: list[int] = []
        oof_residuals: list[np.ndarray] = []
        for inner_group in groups:
            inner_mask = np.isclose(outer_train[CMC].to_numpy(float), inner_group)
            inner_train = outer_train.loc[~inner_mask].copy()
            inner_validation = outer_train.loc[inner_mask].copy()
            row_intersection = set(inner_train[ROW_ID]).intersection(
                inner_validation[ROW_ID]
            )
            group_intersection = set(inner_train[CMC].unique()).intersection(
                inner_validation[CMC].unique()
            )
            if row_intersection or group_intersection:
                raise AssertionError("computed inner LOCO intersection is non-empty")
            fit = fit_neural_with_validation(
                inner_train,
                inner_validation,
                kind=kind,
                seed=seed,
                lambda_p=lambda_p,
                lambda_m=lambda_m,
                max_epochs=max_epochs,
                patience=patience,
            )
            predicted = predict_neural(fit, inner_validation)
            predicted_norm_for_logit = np.clip(
                predicted["prediction_mm"] / D_PRIMARY_MM,
                EPSILON,
                1.0 - EPSILON,
            )
            truth_norm = inner_validation[TARGET].to_numpy(float) / D_PRIMARY_MM
            residual = np.abs(
                normalized_to_logit(truth_norm)
                - normalized_to_logit(predicted_norm_for_logit)
            )
            oof_residuals.append(residual)
            fold_mapes.append(fit.best_validation_mape)
            fold_epochs.append(fit.best_epoch)
            all_records.append(
                {
                    "outer_group_cmc_mm": outer_group,
                    "inner_validation_group_cmc_mm": inner_group,
                    "model": kind,
                    "seed": seed,
                    "lambda_p": lambda_p,
                    "lambda_m": lambda_m,
                    "best_epoch": fit.best_epoch,
                    "inner_validation_mape_percent": fit.best_validation_mape,
                    "inner_train_rows": int(len(inner_train)),
                    "inner_validation_rows": int(len(inner_validation)),
                    "computed_inner_row_overlap_count": int(len(row_intersection)),
                    "computed_inner_group_overlap_count": int(len(group_intersection)),
                    "computed_inner_row_overlap_ids": sorted(row_intersection),
                    "computed_inner_group_overlap_values": sorted(
                        float(value) for value in group_intersection
                    ),
                    "outer_outcomes_used": False,
                    "tx100_status": (
                        "post_reveal_retrospective"
                        if np.isclose(outer_group, 3.5) or np.isclose(inner_group, 3.5)
                        else "not_tx100"
                    ),
                }
            )
        candidates[(lambda_p, lambda_m)] = {
            "macro_mape": float(np.mean(fold_mapes)),
            "epochs": fold_epochs,
            "residuals": np.concatenate(oof_residuals),
        }
    chosen = choose_hyperparameter_candidate(
        {key: float(value["macro_mape"]) for key, value in candidates.items()}
    )
    selected_epochs = candidates[chosen]["epochs"]
    selected_epoch = max(1, int(math.floor(float(np.median(selected_epochs)) + 0.5)))
    radius = conformal_higher_quantile(candidates[chosen]["residuals"], alpha=0.10)
    for record in all_records:
        record["selected_configuration"] = bool(
            np.isclose(record["lambda_p"], chosen[0])
            and np.isclose(record["lambda_m"], chosen[1])
        )
        record["selected_outer_refit_epoch"] = selected_epoch if record[
            "selected_configuration"
        ] else None
        record["loco_oof_heuristic_radius_logit"] = radius if record[
            "selected_configuration"
        ] else None
        record["interval_claim"] = (
            "LOCO_OOF_bounded_uncertainty_heuristic_no_formal_coverage_guarantee"
        )
    return InnerSelection(
        lambda_p=chosen[0],
        lambda_m=chosen[1],
        selected_epoch=selected_epoch,
        conformal_radius_logit=radius,
        records=all_records,
    )


def _base_prediction_rows(
    frame: pd.DataFrame,
    *,
    outer_group: float,
    model: str,
    seed: int,
    prediction_mm: np.ndarray,
    interval_lower_mm: np.ndarray | None = None,
    interval_upper_mm: np.ndarray | None = None,
    anchor_valid: np.ndarray | None = None,
    residual_delta: np.ndarray | None = None,
    monotonic_violation: float = float("nan"),
) -> pd.DataFrame:
    n = len(frame)
    if interval_lower_mm is None:
        interval_lower_mm = np.full(n, np.nan)
    if interval_upper_mm is None:
        interval_upper_mm = np.full(n, np.nan)
    if anchor_valid is None:
        anchor_valid = semi_empirical_anchor(frame)[1]
    if residual_delta is None:
        residual_delta = np.full(n, np.nan)
    anchor_norm, anchor_mask = semi_empirical_anchor(frame)
    result = frame[
        [
            ROW_ID,
            SOURCE_INDEX,
            SOURCE_ROW_INDEX,
            QC,
            QD,
            GAMMA,
            CONCENTRATION,
            CMC,
            TARGET,
        ]
    ].copy()
    result["heldout_group_cmc_mm"] = outer_group
    result["model"] = model
    result["seed"] = seed
    result["prediction_mm"] = np.asarray(prediction_mm, dtype=float)
    result["prediction_normalized"] = result["prediction_mm"] / D_PRIMARY_MM
    result["interval_lower_mm"] = np.asarray(interval_lower_mm, dtype=float)
    result["interval_upper_mm"] = np.asarray(interval_upper_mm, dtype=float)
    result["anchor_prediction_mm"] = anchor_norm * D_PRIMARY_MM
    result["anchor_valid"] = np.asarray(anchor_valid, dtype=bool) & anchor_mask
    result["residual_delta"] = np.asarray(residual_delta, dtype=float)
    result["monotonic_violation_rate"] = monotonic_violation
    result["tx100_status"] = np.where(
        np.isclose(result[CMC].to_numpy(float), 3.5),
        "post_reveal_retrospective",
        "not_tx100",
    )
    return result


def _point_invalid(prediction_mm: np.ndarray) -> np.ndarray:
    pred = np.asarray(prediction_mm, dtype=float)
    return (~np.isfinite(pred)) | (pred <= 0.0) | (pred >= D_PRIMARY_MM)


def _interval_invalid(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    missing = np.isnan(lo) & np.isnan(hi)
    invalid = (
        (~np.isfinite(lo))
        | (~np.isfinite(hi))
        | (lo <= 0.0)
        | (lo >= hi)
        | (hi >= D_PRIMARY_MM)
    )
    invalid[missing] = False
    return invalid


def run_outer_fold(
    main: pd.DataFrame,
    *,
    outer_group: float,
    seeds: Sequence[int],
    max_epochs: int,
    patience: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer_train, outer_test = outer_loco_split(main, outer_group)
    manifest_rows: list[dict[str, Any]] = []
    row_intersection = set(outer_train[ROW_ID]).intersection(outer_test[ROW_ID])
    group_intersection = set(outer_train[CMC].unique()).intersection(
        outer_test[CMC].unique()
    )
    if row_intersection or group_intersection:
        raise AssertionError("computed outer LOCO intersection is non-empty")
    for role, partition in (("outer_train", outer_train), ("outer_test", outer_test)):
        for _, source_row in partition[[ROW_ID, CMC]].iterrows():
            row_id = source_row[ROW_ID]
            row_group = float(source_row[CMC])
            manifest_rows.append(
                {
                    ROW_ID: row_id,
                    "outer_group_cmc_mm": outer_group,
                    "role": role,
                    "group_overlap": row_group in group_intersection,
                    "row_overlap": row_id in row_intersection,
                    "computed_group_overlap_count": int(len(group_intersection)),
                    "computed_row_overlap_count": int(len(row_intersection)),
                    "computed_group_overlap_values": sorted(
                        float(value) for value in group_intersection
                    ),
                    "computed_row_overlap_ids": sorted(row_intersection),
                    "tx100_status": (
                        "post_reveal_retrospective"
                        if np.isclose(outer_group, 3.5)
                        else "not_tx100"
                    ),
                }
            )

    prediction_frames: list[pd.DataFrame] = []
    inner_records: list[dict[str, Any]] = []
    b0_norm, b0_valid = semi_empirical_anchor(outer_test)
    for seed in seeds:
        prediction_frames.append(
            _base_prediction_rows(
                outer_test,
                outer_group=outer_group,
                model="B0",
                seed=seed,
                prediction_mm=b0_norm * D_PRIMARY_MM,
                anchor_valid=b0_valid,
            )
        )
        for kind in ("B1", "B2"):
            fit = fit_xgb(outer_train, kind=kind, seed=seed)
            prediction_frames.append(
                _base_prediction_rows(
                    outer_test,
                    outer_group=outer_group,
                    model=kind,
                    seed=seed,
                    prediction_mm=predict_xgb(fit, outer_test),
                )
            )
        for kind in ("B3", "V2-H", "V2-P", "V2-M"):
            selection = select_neural_inner_loco(
                outer_train,
                outer_group=outer_group,
                kind=kind,
                seed=seed,
                max_epochs=max_epochs,
                patience=patience,
            )
            inner_records.extend(selection.records)
            fit = fit_neural_fixed_epochs(
                outer_train,
                kind=kind,
                seed=seed,
                epochs=selection.selected_epoch,
                lambda_p=selection.lambda_p,
                lambda_m=selection.lambda_m,
            )
            predicted = predict_neural(fit, outer_test)
            lower, upper = bounded_interval_from_logit(
                predicted["prediction_logit"], selection.conformal_radius_logit
            )
            prediction_frames.append(
                _base_prediction_rows(
                    outer_test,
                    outer_group=outer_group,
                    model=kind,
                    seed=seed,
                    prediction_mm=predicted["prediction_mm"],
                    interval_lower_mm=lower,
                    interval_upper_mm=upper,
                    anchor_valid=predicted["anchor_valid"],
                    residual_delta=predicted["residual_delta"],
                    monotonic_violation=monotonic_violation_rate(fit, outer_test),
                )
            )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions["point_physical_invalid"] = _point_invalid(
        predictions["prediction_mm"].to_numpy(float)
    )
    predictions["interval_physical_invalid"] = _interval_invalid(
        predictions["interval_lower_mm"].to_numpy(float),
        predictions["interval_upper_mm"].to_numpy(float),
    )
    has_interval = predictions["interval_lower_mm"].notna()
    predictions["interval_contains_outcome"] = np.where(
        has_interval,
        (predictions[TARGET] >= predictions["interval_lower_mm"])
        & (predictions[TARGET] <= predictions["interval_upper_mm"]),
        np.nan,
    )
    return (
        predictions,
        pd.DataFrame(inner_records),
        pd.DataFrame(manifest_rows),
    )


def _safe_r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    if len(truth) < 2 or np.allclose(truth, truth[0]):
        return float("nan")
    return float(r2_score(truth, prediction))


def metric_record(frame: pd.DataFrame, *, level: str) -> dict[str, Any]:
    truth = frame[TARGET].to_numpy(float)
    prediction = frame["prediction_mm"].to_numpy(float)
    absolute_percentage = 100.0 * np.abs(prediction - truth) / truth
    interval_mask = frame["interval_lower_mm"].notna().to_numpy()
    if np.any(interval_mask):
        interval_coverage = float(
            np.mean(frame.loc[interval_mask, "interval_contains_outcome"].astype(float))
        )
        interval_width = float(
            np.mean(
                frame.loc[interval_mask, "interval_upper_mm"].to_numpy(float)
                - frame.loc[interval_mask, "interval_lower_mm"].to_numpy(float)
            )
        )
        interval_invalid = float(
            np.mean(frame.loc[interval_mask, "interval_physical_invalid"].astype(float))
        )
    else:
        interval_coverage = float("nan")
        interval_width = float("nan")
        interval_invalid = float("nan")
    residual = frame["residual_delta"].to_numpy(float)
    residual_finite = np.isfinite(residual)
    seed_value = frame["seed"].iloc[0] if frame["seed"].nunique() == 1 else "ensemble"
    seed_output: int | str = (
        int(seed_value) if isinstance(seed_value, (int, np.integer)) else str(seed_value)
    )
    return {
        "level": level,
        "model": str(frame["model"].iloc[0]),
        "seed": seed_output,
        "heldout_group_cmc_mm": float(frame["heldout_group_cmc_mm"].iloc[0]),
        "n": int(len(frame)),
        "mape_percent": float(np.mean(absolute_percentage)),
        "mdape_percent": float(np.median(absolute_percentage)),
        "mae_mm": float(mean_absolute_error(truth, prediction)),
        "rmse_mm": float(np.sqrt(mean_squared_error(truth, prediction))),
        "r2_unstable_for_small_or_narrow_groups": _safe_r2(truth, prediction),
        "signed_bias_mm": float(np.mean(prediction - truth)),
        "point_physical_invalid_rate": float(
            np.mean(frame["point_physical_invalid"].astype(float))
        ),
        "interval_physical_invalid_rate": interval_invalid,
        "interval_coverage_transfer_stress_test": interval_coverage,
        "interval_mean_width_mm": interval_width,
        "anchor_invalid_fraction": float(np.mean(~frame["anchor_valid"].astype(bool))),
        "mean_absolute_residual_delta": (
            float(np.mean(np.abs(residual[residual_finite])))
            if np.any(residual_finite)
            else float("nan")
        ),
        "local_monotonic_violation_rate": float(
            frame["monotonic_violation_rate"].dropna().mean()
        ),
        "interval_claim": (
            "LOCO_OOF_bounded_uncertainty_heuristic_no_formal_coverage_guarantee"
            if np.any(interval_mask)
            else "not_reported"
        ),
        "tx100_status": (
            "post_reveal_retrospective"
            if np.isclose(frame["heldout_group_cmc_mm"].iloc[0], 3.5)
            else "not_tx100"
        ),
    }


def aggregate_predictions(
    per_seed_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_metric_rows: list[dict[str, Any]] = []
    for (_, _, _), part in per_seed_predictions.groupby(
        ["heldout_group_cmc_mm", "model", "seed"], sort=True
    ):
        seed_metric_rows.append(metric_record(part, level="seed_group"))
    seed_metrics = pd.DataFrame(seed_metric_rows)

    identity = [
        ROW_ID,
        SOURCE_INDEX,
        SOURCE_ROW_INDEX,
        QC,
        QD,
        GAMMA,
        CONCENTRATION,
        CMC,
        TARGET,
        "heldout_group_cmc_mm",
        "model",
        "tx100_status",
    ]
    numeric_mean = [
        "prediction_mm",
        "prediction_normalized",
        "interval_lower_mm",
        "interval_upper_mm",
        "anchor_prediction_mm",
        "residual_delta",
        "monotonic_violation_rate",
    ]
    ensemble = (
        per_seed_predictions.groupby(identity, as_index=False, dropna=False, sort=True)[
            numeric_mean
        ]
        .mean()
        .sort_values(["heldout_group_cmc_mm", "model", SOURCE_INDEX, SOURCE_ROW_INDEX])
        .reset_index(drop=True)
    )
    ensemble["seed"] = "ensemble_mean"
    ensemble["anchor_valid"] = ensemble["anchor_prediction_mm"].between(
        0.0, D_PRIMARY_MM, inclusive="neither"
    )
    ensemble["point_physical_invalid"] = _point_invalid(ensemble["prediction_mm"])
    ensemble["interval_physical_invalid"] = _interval_invalid(
        ensemble["interval_lower_mm"], ensemble["interval_upper_mm"]
    )
    interval_present = ensemble["interval_lower_mm"].notna()
    ensemble["interval_contains_outcome"] = np.where(
        interval_present,
        (ensemble[TARGET] >= ensemble["interval_lower_mm"])
        & (ensemble[TARGET] <= ensemble["interval_upper_mm"]),
        np.nan,
    )

    group_rows: list[dict[str, Any]] = []
    for (_, _), part in ensemble.groupby(["heldout_group_cmc_mm", "model"], sort=True):
        group_rows.append(metric_record(part, level="ensemble_group"))
    group_metrics = pd.DataFrame(group_rows)

    summary_rows: list[dict[str, Any]] = []
    for model, parts in group_metrics.groupby("model", sort=True):
        model_ensemble = ensemble.loc[ensemble["model"].eq(model)]
        truth = model_ensemble[TARGET].to_numpy(float)
        pred = model_ensemble["prediction_mm"].to_numpy(float)
        seed_subset = seed_metrics.loc[seed_metrics["model"].eq(model)]
        summary_rows.append(
            {
                "model": model,
                "chemistry_macro_mape_percent": float(parts["mape_percent"].mean()),
                "worst_group_mape_percent": float(parts["mape_percent"].max()),
                "row_micro_mape_percent": _mape_percent(truth, pred),
                "row_micro_mae_mm": float(mean_absolute_error(truth, pred)),
                "row_micro_rmse_mm": float(np.sqrt(mean_squared_error(truth, pred))),
                "row_micro_r2": _safe_r2(truth, pred),
                "point_physical_invalid_rate": float(
                    model_ensemble["point_physical_invalid"].mean()
                ),
                "interval_physical_invalid_rate": float(
                    model_ensemble.loc[
                        model_ensemble["interval_lower_mm"].notna(),
                        "interval_physical_invalid",
                    ].mean()
                ),
                "mean_group_interval_coverage_transfer_stress_test": float(
                    parts["interval_coverage_transfer_stress_test"].mean()
                ),
                "seed_group_mape_sd_percent": float(
                    seed_subset.groupby("heldout_group_cmc_mm")["mape_percent"]
                    .std(ddof=0)
                    .mean()
                ),
                "n_main_groups": int(len(parts)),
            }
        )
    summary = pd.DataFrame(summary_rows)
    return ensemble, seed_metrics, group_metrics, summary


def add_compiler_support(
    ensemble_without_target: pd.DataFrame, main: pd.DataFrame
) -> pd.DataFrame:
    if TARGET in ensemble_without_target.columns:
        raise ValueError("outcome leakage: compiler support cannot receive outcomes")
    result = ensemble_without_target.copy()
    compiled_values = np.zeros(len(result), dtype=bool)
    reason_values = np.full(len(result), "", dtype=object)
    device_values = np.zeros(len(result), dtype=bool)
    training_flow_values = np.zeros(len(result), dtype=bool)
    native_values = np.zeros(len(result), dtype=bool)
    cmc_fixed_values = np.zeros(len(result), dtype=bool)
    material_values = np.zeros(len(result), dtype=bool)
    derived_values = np.zeros(len(result), dtype=bool)
    train_qc_min = np.full(len(result), np.nan)
    train_qc_max = np.full(len(result), np.nan)
    train_qd_min = np.full(len(result), np.nan)
    train_qd_max = np.full(len(result), np.nan)
    for group in MAIN_GROUPS:
        outer_train, outer_test = outer_loco_split(main, group)
        support = _support(outer_train)
        mask = np.isclose(result["heldout_group_cmc_mm"].to_numpy(float), group)
        part = result.loc[mask]
        device_supported = (
            part[QC].between(*DEVICE_QC_LIMITS_ML_MIN, inclusive="both")
            & part[QD].between(*DEVICE_QD_LIMITS_ML_MIN, inclusive="both")
        )
        training_supported = (
            part[QC].between(*support[QC], inclusive="both")
            & part[QD].between(*support[QD], inclusive="both")
        )
        native_valid = (
            (part[[QC, QD, GAMMA]] > 0).all(axis=1)
            & (part[[CONCENTRATION, CMC]] >= 0).all(axis=1)
            & (
                (~np.isclose(part[CMC].to_numpy(float), 0.0))
                | np.isclose(part[CONCENTRATION].to_numpy(float), 0.0)
            )
        )
        cmc_fixed = pd.Series(
            np.isclose(part[CMC].to_numpy(float), group), index=part.index
        )
        observed_material_records = {
            (float(row[CMC]), float(row[GAMMA]), float(row[CONCENTRATION]))
            for _, row in outer_test[[CMC, GAMMA, CONCENTRATION]].drop_duplicates().iterrows()
        }
        material_observed = part.apply(
            lambda row: (
                float(row[CMC]), float(row[GAMMA]), float(row[CONCENTRATION])
            )
            in observed_material_records,
            axis=1,
        )
        try:
            physical = physics_layer(part)
            derived_valid = pd.Series(
                np.isfinite(physical.to_numpy(float)).all(axis=1)
                & (physical[
                    [
                        "Qc_m3_s",
                        "Qd_m3_s",
                        "gamma_N_m",
                        "flow_ratio",
                        "Ca_c",
                        "Ca_d",
                        "Re_c",
                        "Re_d",
                    ]
                ].to_numpy(float)
                > 0)
                .all(axis=1)
                & (physical["c_star"].to_numpy(float) >= 0),
                index=part.index,
            )
        except ValueError:
            derived_valid = pd.Series(False, index=part.index)
        compiled_series = (
            device_supported
            & training_supported
            & native_valid
            & cmc_fixed
            & material_observed
            & derived_valid
        )
        compiled = compiled_series.to_numpy(bool)
        positions = np.flatnonzero(mask)
        compiled_values[positions] = compiled
        device_values[positions] = device_supported.to_numpy(bool)
        training_flow_values[positions] = training_supported.to_numpy(bool)
        native_values[positions] = native_valid.to_numpy(bool)
        cmc_fixed_values[positions] = cmc_fixed.to_numpy(bool)
        material_values[positions] = material_observed.to_numpy(bool)
        derived_values[positions] = derived_valid.to_numpy(bool)
        train_qc_min[positions] = support[QC][0]
        train_qc_max[positions] = support[QC][1]
        train_qd_min[positions] = support[QD][0]
        train_qd_max[positions] = support[QD][1]
        failures = np.select(
            [
                ~device_supported,
                ~training_supported,
                ~native_valid,
                ~cmc_fixed,
                ~material_observed,
                ~derived_valid,
            ],
            [
                "outside_explicit_device_flow_limits",
                "outside_outer_training_marginal_flow_support",
                "invalid_native_units_or_sign",
                "heldout_cmc_not_fixed",
                "gamma_concentration_not_observed_for_heldout_chemistry",
                "invalid_recomputed_derived_physics",
            ],
            default="accepted",
        )
        reason_values[positions] = failures
    result["compiled"] = compiled_values
    result["compiler_reason"] = reason_values
    result["device_flow_limits_pass"] = device_values
    result["outer_training_marginal_flow_support_pass"] = training_flow_values
    result["native_unit_and_positivity_pass"] = native_values
    result["heldout_cmc_fixed_pass"] = cmc_fixed_values
    result["same_chemistry_gamma_concentration_record_pass"] = material_values
    result["derived_finite_positive_pass"] = derived_values
    result["device_qc_min_ml_min"] = DEVICE_QC_LIMITS_ML_MIN[0]
    result["device_qc_max_ml_min"] = DEVICE_QC_LIMITS_ML_MIN[1]
    result["device_qd_min_ml_min"] = DEVICE_QD_LIMITS_ML_MIN[0]
    result["device_qd_max_ml_min"] = DEVICE_QD_LIMITS_ML_MIN[1]
    result["outer_train_qc_min_ml_min"] = train_qc_min
    result["outer_train_qc_max_ml_min"] = train_qc_max
    result["outer_train_qd_min_ml_min"] = train_qd_min
    result["outer_train_qd_max_ml_min"] = train_qd_max
    result["compiler_scope_note"] = (
        "Explicit device Qc/Qd limits plus outer-training marginal Qc/Qd support; "
        "heldout CMC fixed; gamma/concentration must be an observed same-chemistry "
        "record; native units/signs and recomputed derived physics must be valid. "
        "Heldout CMC is intentionally not rejected merely because LOCO training lacks it."
    )
    return result


def compile_label_free_search_selections(
    predictions: pd.DataFrame,
    *,
    targets_mm: Sequence[float] = TARGETS_MM,
    top_ks: Sequence[int] = TOP_KS,
) -> pd.DataFrame:
    if TARGET in predictions.columns:
        raise ValueError("outcome leakage: label-free compiler cannot receive target outcomes")
    required = {
        ROW_ID,
        CMC,
        QC,
        QD,
        GAMMA,
        CONCENTRATION,
        "model",
        "prediction_mm",
        "interval_lower_mm",
        "interval_upper_mm",
        "compiled",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"search compiler missing fields: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    tolerance_specs = (("absolute_0.01_mm", 0.01), ("relative_10_percent", None))
    for (group, model), candidates_all in predictions.groupby([CMC, "model"], sort=True):
        candidates = candidates_all.loc[candidates_all["compiled"].astype(bool)].copy()
        candidates = candidates.sort_values(ROW_ID, kind="stable")
        for target in targets_mm:
            for tolerance_name, fixed in tolerance_specs:
                tolerance = float(fixed if fixed is not None else 0.10 * target)
                if candidates.empty:
                    for top_k in top_ks:
                        rows.append(
                            {
                                "heldout_group_cmc_mm": float(group),
                                "model": model,
                                "target_mm": float(target),
                                "tolerance_rule": tolerance_name,
                                "tolerance_mm": tolerance,
                                "top_k": int(top_k),
                                "rank": None,
                                ROW_ID: None,
                                "prediction_mm": None,
                                "interval_lower_mm": None,
                                "interval_upper_mm": None,
                                "answered_query": False,
                                "point_within_tolerance": False,
                                "interval_fully_contained_in_tolerance": False,
                                "abstention_rule": "conservative_containment_abstention",
                                "compiled_candidate_count": 0,
                                "total_candidate_count": int(len(candidates_all)),
                            }
                        )
                    continue
                ranked = candidates.assign(
                    predicted_gap=np.abs(candidates["prediction_mm"].to_numpy(float) - target)
                ).sort_values(["predicted_gap", ROW_ID], kind="stable")
                best = ranked.iloc[0]
                point_within = bool(float(best["predicted_gap"]) <= tolerance)
                interval_available = bool(
                    pd.notna(best.get("interval_lower_mm"))
                    and pd.notna(best.get("interval_upper_mm"))
                )
                interval_contained = bool(
                    interval_available
                    and float(best["interval_lower_mm"]) >= target - tolerance
                    and float(best["interval_upper_mm"]) <= target + tolerance
                )
                answered = bool(point_within and interval_contained)
                for top_k in top_ks:
                    selected = ranked.head(int(top_k))
                    for rank, (_, chosen) in enumerate(selected.iterrows(), start=1):
                        rows.append(
                            {
                                "heldout_group_cmc_mm": float(group),
                                "model": model,
                                "target_mm": float(target),
                                "tolerance_rule": tolerance_name,
                                "tolerance_mm": tolerance,
                                "top_k": int(top_k),
                                "rank": rank,
                                ROW_ID: chosen[ROW_ID],
                                "prediction_mm": float(chosen["prediction_mm"]),
                                "interval_lower_mm": float(chosen["interval_lower_mm"])
                                if pd.notna(chosen["interval_lower_mm"])
                                else None,
                                "interval_upper_mm": float(chosen["interval_upper_mm"])
                                if pd.notna(chosen["interval_upper_mm"])
                                else None,
                                "answered_query": answered,
                                "point_within_tolerance": point_within,
                                "interval_fully_contained_in_tolerance": interval_contained,
                                "abstention_rule": "conservative_containment_abstention",
                                "compiled_candidate_count": int(len(candidates)),
                                "total_candidate_count": int(len(candidates_all)),
                            }
                        )
    result = pd.DataFrame(rows)
    if TARGET in result.columns:
        raise AssertionError("search selection artifact contains forbidden outcomes")
    return result


def score_search_selections(
    selections: pd.DataFrame,
    labels: pd.DataFrame,
    compiled_candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if list(labels.columns) != [ROW_ID, TARGET]:
        raise ValueError("scoring labels must be isolated to row ID and target")
    scored = selections.merge(labels, on=ROW_ID, how="left", validate="many_to_one")
    scored["selected_success"] = (
        scored["answered_query"].astype(bool)
        & scored[TARGET].notna()
        & ((scored[TARGET] - scored["target_mm"]).abs() <= scored["tolerance_mm"])
    )
    query_keys = [
        "heldout_group_cmc_mm",
        "model",
        "target_mm",
        "tolerance_rule",
        "top_k",
    ]
    query_rows: list[dict[str, Any]] = []
    for keys, part in scored.groupby(query_keys, dropna=False, sort=True):
        group, model, target, tolerance_rule, top_k = keys
        answered = bool(part["answered_query"].iloc[0])
        success = bool(part["selected_success"].any()) if answered else False
        tolerance = float(part["tolerance_mm"].iloc[0])
        pool = compiled_candidates.loc[
            np.isclose(compiled_candidates[CMC].to_numpy(float), float(group))
            & compiled_candidates["model"].eq(model)
            & compiled_candidates["compiled"].astype(bool)
        ]
        oracle_gap = (
            float(np.min(np.abs(pool[TARGET].to_numpy(float) - float(target))))
            if len(pool)
            else float("nan")
        )
        selected_gap = (
            float(np.min(np.abs(part.loc[part[TARGET].notna(), TARGET] - float(target))))
            if part[TARGET].notna().any()
            else float("nan")
        )
        query_rows.append(
            {
                "heldout_group_cmc_mm": float(group),
                "model": model,
                "target_mm": float(target),
                "tolerance_rule": tolerance_rule,
                "top_k": int(top_k),
                "answered_query": answered,
                "success": success,
                "oracle_feasible": bool(np.isfinite(oracle_gap) and oracle_gap <= tolerance),
                "oracle_gap_mm": oracle_gap,
                "selected_best_gap_mm": selected_gap,
                "oracle_regret_mm": selected_gap - oracle_gap
                if np.isfinite(selected_gap) and np.isfinite(oracle_gap)
                else float("nan"),
                "compiled_candidate_count": int(part["compiled_candidate_count"].iloc[0]),
                "total_candidate_count": int(part["total_candidate_count"].iloc[0]),
            }
        )
    queries = pd.DataFrame(query_rows)
    summary_rows: list[dict[str, Any]] = []
    for (model, top_k, tolerance_rule), part in queries.groupby(
        ["model", "top_k", "tolerance_rule"], sort=True
    ):
        answered = int(part["answered_query"].sum())
        successes = int(part["success"].sum())
        oracle_feasible = int(part["oracle_feasible"].sum())
        summary_rows.append(
            {
                "model": model,
                "top_k": int(top_k),
                "tolerance_rule": tolerance_rule,
                "queries": int(len(part)),
                "answered_queries": answered,
                "successful_queries": successes,
                "oracle_feasible_queries": oracle_feasible,
                "query_coverage": answered / len(part) if len(part) else float("nan"),
                "answered_query_precision": successes / answered if answered else float("nan"),
                "successful_target_recall": successes / oracle_feasible
                if oracle_feasible
                else float("nan"),
                "candidate_reduction": float(
                    1.0 - part["compiled_candidate_count"].sum() / part["total_candidate_count"].sum()
                )
                if part["total_candidate_count"].sum()
                else float("nan"),
                "mean_oracle_regret_mm": float(part["oracle_regret_mm"].mean()),
            }
        )
    return queries, pd.DataFrame(summary_rows)


def physics_sensitivity_table(main: pd.DataFrame) -> pd.DataFrame:
    primary_norm, primary_valid = semi_empirical_anchor(main)
    d190_norm, d190_valid = semi_empirical_anchor(main, diameter_m=D_SENSITIVITY_M)
    rectangular_norm, rectangular_valid = semi_empirical_anchor(
        main, area_m2=RECTANGULAR_AREA_M2
    )
    result = main[[ROW_ID, CMC, TARGET]].copy()
    result["B0_primary_D195_oval_mm"] = primary_norm * D_PRIMARY_MM
    result["B0_D190_oval_mm"] = d190_norm * (D_SENSITIVITY_M * 1e3)
    result["B0_D195_rectangular_area_mm"] = rectangular_norm * D_PRIMARY_MM
    result["primary_anchor_valid"] = primary_valid
    result["D190_anchor_valid"] = d190_valid
    result["rectangular_anchor_valid"] = rectangular_valid
    result["tx100_status"] = np.where(
        np.isclose(result[CMC].to_numpy(float), 3.5),
        "post_reveal_retrospective",
        "not_tx100",
    )
    return result


def _choose_anomaly_neural_configuration(
    inner_records: pd.DataFrame, *, kind: str, seed: int
) -> tuple[float, float, int]:
    selected = inner_records.loc[
        inner_records["model"].eq(kind)
        & inner_records["seed"].eq(seed)
        & inner_records["selected_configuration"].astype(bool)
    ].copy()
    if selected.empty:
        raise ValueError(f"no selected inner configurations for {kind}, seed {seed}")
    # Each selected configuration is repeated once per inner fold. Collapse to
    # one record per outer fold before deterministic consensus selection.
    per_outer = selected.groupby("outer_group_cmc_mm", as_index=False).agg(
        lambda_p=("lambda_p", "first"),
        lambda_m=("lambda_m", "first"),
        epoch=("selected_outer_refit_epoch", "first"),
    )
    combos = (
        per_outer.groupby(["lambda_p", "lambda_m"], as_index=False)
        .size()
        .sort_values(["size", "lambda_p", "lambda_m"], ascending=[False, False, False])
    )
    chosen = combos.iloc[0]
    epoch = max(1, int(math.floor(float(np.median(per_outer["epoch"])) + 0.5)))
    return float(chosen["lambda_p"]), float(chosen["lambda_m"]), epoch


def evaluate_quarantined_anomaly(
    main: pd.DataFrame,
    quarantine: pd.DataFrame,
    inner_records: pd.DataFrame,
    *,
    seeds: Sequence[int],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    anchor_norm, anchor_valid = semi_empirical_anchor(quarantine)
    for seed in seeds:
        frames.append(
            _base_prediction_rows(
                quarantine,
                outer_group=QUARANTINED_GROUP,
                model="B0",
                seed=seed,
                prediction_mm=anchor_norm * D_PRIMARY_MM,
                anchor_valid=anchor_valid,
            )
        )
        for kind in ("B1", "B2"):
            fit = fit_xgb(main, kind=kind, seed=seed)
            frames.append(
                _base_prediction_rows(
                    quarantine,
                    outer_group=QUARANTINED_GROUP,
                    model=kind,
                    seed=seed,
                    prediction_mm=predict_xgb(fit, quarantine),
                )
            )
        for kind in ("B3", "V2-H", "V2-P", "V2-M"):
            lambda_p, lambda_m, epoch = _choose_anomaly_neural_configuration(
                inner_records, kind=kind, seed=seed
            )
            fit = fit_neural_fixed_epochs(
                main,
                kind=kind,
                seed=seed,
                epochs=epoch,
                lambda_p=lambda_p,
                lambda_m=lambda_m,
            )
            prediction = predict_neural(fit, quarantine)
            frames.append(
                _base_prediction_rows(
                    quarantine,
                    outer_group=QUARANTINED_GROUP,
                    model=kind,
                    seed=seed,
                    prediction_mm=prediction["prediction_mm"],
                    anchor_valid=prediction["anchor_valid"],
                    residual_delta=prediction["residual_delta"],
                    monotonic_violation=monotonic_violation_rate(fit, quarantine),
                )
            )
    result = pd.concat(frames, ignore_index=True)
    result["analysis_status"] = "quarantined_unlisted_CMC_0.60_sensitivity_only"
    result["point_physical_invalid"] = _point_invalid(result["prediction_mm"])
    return result


def quarantined_anomaly_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, part in predictions.groupby("model", sort=True):
        ensemble = part.groupby(ROW_ID, as_index=False).agg(
            target_diameter_mm=(TARGET, "first"),
            prediction_mm=("prediction_mm", "mean"),
        )
        truth = ensemble[TARGET].to_numpy(float)
        prediction = ensemble["prediction_mm"].to_numpy(float)
        rows.append(
            {
                "model": model,
                "n_quarantined_rows": int(len(ensemble)),
                "mape_percent_descriptive_only": _mape_percent(truth, prediction),
                "mae_mm_descriptive_only": float(mean_absolute_error(truth, prediction)),
                "prediction_seed_sd_mm": float(
                    part.groupby(ROW_ID)["prediction_mm"].std(ddof=0).mean()
                ),
                "point_physical_invalid_rate": float(np.mean(_point_invalid(prediction))),
                "analysis_status": "quarantined_unlisted_CMC_0.60_sensitivity_only",
            }
        )
    return pd.DataFrame(rows)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")


def _hash_artifacts(output_dir: Path) -> dict[str, str]:
    excluded = {"artifact_hashes.json"}
    return {
        str(path.relative_to(output_dir)): sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name not in excluded
    }


def verify_artifact_hash_manifest(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "artifact_hashes.json"
    manifest = _strict_json_load(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("artifact hash manifest must be a JSON object")
    mismatches: dict[str, dict[str, str]] = {}
    for relative, expected in manifest.items():
        candidate = (output_dir / relative).resolve()
        try:
            candidate.relative_to(output_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"artifact manifest path escapes output directory: {relative}") from exc
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        observed = sha256_file(candidate)
        if observed != expected:
            mismatches[relative] = {"expected": str(expected), "observed": observed}
    if mismatches:
        raise ValueError(f"artifact hash mismatch: {mismatches}")
    json_files = sorted(output_dir.glob("*.json"))
    for path in json_files:
        _strict_json_load(path)
    return {
        "manifest_entries": int(len(manifest)),
        "strict_json_files": int(len(json_files)),
        "hashes_match": True,
        "strict_json_parse": True,
    }


def run_study(
    *,
    project_dir: Path,
    output_dir: Path,
    fast: bool,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    assert_protocol_lock(project_dir)
    set_deterministic_seed(42)

    data = load_joined_public_data(project_dir / "blind_artifacts")
    settings = study_settings(fast)
    seeds = settings.seeds
    max_epochs = settings.max_epochs
    patience = settings.patience
    run_mode = settings.run_mode
    protocol_compliant = settings.protocol_compliant

    _write_csv(output_dir / "quarantined_cmc_0p60_rows.csv", data.quarantine)
    write_json(output_dir / "data_join_audit.json", data.audit)
    raw_dir = project_dir / "blind_raw"
    raw_files = {
        path.name: sha256_file(path)
        for path in sorted(raw_dir.glob("*.mat"))
        if path.is_file()
    }
    write_json(
        output_dir / "source_provenance.json",
        {
            "dataset_url": "https://zenodo.org/records/7055018",
            "dataset_license": "CC BY 4.0",
            "source_article_doi": "10.1039/D2LC00416J",
            "raw_mat_sha256": raw_files,
            "joined_input_sha256": data.audit["input_hashes"],
        },
    )
    write_json(
        output_dir / "anomaly_register.json",
        {
            "anomaly": "recorded CMC 0.60 mM not listed among nominal source surfactants",
            "action": "never reassigned; excluded from 466-row main analysis; sensitivity only",
            "rows": data.quarantine.to_dict(orient="records"),
            "main_group_counts": {
                format(float(group), ".12g"): int(np.isclose(data.main[CMC], group).sum())
                for group in MAIN_GROUPS
            },
        },
    )
    physics = pd.concat(
        [
            data.main[[ROW_ID]].reset_index(drop=True),
            native_state(data.main).reset_index(drop=True),
            physics_layer(data.main).reset_index(drop=True),
            physics_model_transforms(data.main).reset_index(drop=True),
        ],
        axis=1,
    )
    physics["tx100_status"] = np.where(
        np.isclose(physics[CMC].to_numpy(float), 3.5),
        "post_reveal_retrospective",
        "not_tx100",
    )
    _write_csv(output_dir / "native_physics_layer_audit.csv", physics)

    prediction_parts: list[pd.DataFrame] = []
    inner_parts: list[pd.DataFrame] = []
    split_parts: list[pd.DataFrame] = []
    for outer_group in MAIN_GROUPS:
        predictions, inner, manifest = run_outer_fold(
            data.main,
            outer_group=outer_group,
            seeds=seeds,
            max_epochs=max_epochs,
            patience=patience,
        )
        prediction_parts.append(predictions)
        inner_parts.append(inner)
        split_parts.append(manifest)
    per_seed = pd.concat(prediction_parts, ignore_index=True)
    inner_records = pd.concat(inner_parts, ignore_index=True)
    split_manifest = pd.concat(split_parts, ignore_index=True)
    if split_manifest["group_overlap"].any() or split_manifest["row_overlap"].any():
        raise AssertionError("leakage assertion failed")
    _write_csv(output_dir / "outer_split_manifest.csv", split_manifest)
    _write_csv(output_dir / "inner_selection_records.csv", inner_records)
    _write_csv(output_dir / "per_row_predictions_all_seeds.csv", per_seed)

    ensemble, seed_metrics, group_metrics, summary = aggregate_predictions(per_seed)
    _write_csv(output_dir / "ensemble_per_row_predictions.csv", ensemble)
    _write_csv(output_dir / "seed_group_metrics.csv", seed_metrics)
    _write_csv(output_dir / "ensemble_group_metrics.csv", group_metrics)
    _write_csv(output_dir / "model_summary_metrics.csv", summary)

    b1_groups = group_metrics.loc[group_metrics["model"].eq("B1")].set_index(
        "heldout_group_cmc_mm"
    )
    v2p_groups = group_metrics.loc[group_metrics["model"].eq("V2-P")].set_index(
        "heldout_group_cmc_mm"
    )
    differences = pd.DataFrame(
        {
            "heldout_group_cmc_mm": MAIN_GROUPS,
            "B1_mape_percent": [b1_groups.loc[group, "mape_percent"] for group in MAIN_GROUPS],
            "V2_P_mape_percent": [v2p_groups.loc[group, "mape_percent"] for group in MAIN_GROUPS],
        }
    )
    differences["V2_P_minus_B1_mape_percentage_points"] = (
        differences["V2_P_mape_percent"] - differences["B1_mape_percent"]
    )
    differences["V2_P_beats_B1"] = differences[
        "V2_P_minus_B1_mape_percentage_points"
    ] < 0
    differences["tx100_status"] = np.where(
        np.isclose(differences["heldout_group_cmc_mm"], 3.5),
        "post_reveal_retrospective",
        "not_tx100",
    )
    _write_csv(output_dir / "paired_group_differences_v2p_vs_b1.csv", differences)

    sensitivity = physics_sensitivity_table(data.main)
    _write_csv(output_dir / "locked_physics_sensitivities.csv", sensitivity)
    sensitivity_summary = []
    for column, bound_mm in (
        ("B0_primary_D195_oval_mm", D_PRIMARY_MM),
        ("B0_D190_oval_mm", D_SENSITIVITY_M * 1e3),
        ("B0_D195_rectangular_area_mm", D_PRIMARY_MM),
    ):
        sensitivity_summary.append(
            {
                "variant": column,
                "mape_percent": _mape_percent(data.main[TARGET], sensitivity[column]),
                "point_invalid_rate_using_variant_diameter_bound": float(
                    np.mean(
                        (sensitivity[column].to_numpy(float) <= 0)
                        | (sensitivity[column].to_numpy(float) >= bound_mm)
                    )
                ),
                "selection_role": "locked_sensitivity_never_used_for_model_selection",
            }
        )
    _write_csv(output_dir / "locked_physics_sensitivity_summary.csv", pd.DataFrame(sensitivity_summary))

    anomaly_predictions = evaluate_quarantined_anomaly(
        data.main,
        data.quarantine,
        inner_records,
        seeds=seeds,
    )
    _write_csv(output_dir / "quarantined_anomaly_predictions.csv", anomaly_predictions)
    _write_csv(
        output_dir / "quarantined_anomaly_metrics.csv",
        quarantined_anomaly_metrics(anomaly_predictions),
    )

    label_free = ensemble.drop(columns=[TARGET]).copy()
    label_free = add_compiler_support(label_free, data.main)
    label_free_selection = compile_label_free_search_selections(label_free)
    selection_path = output_dir / "retrospective_search_selections_label_free.csv"
    _write_csv(selection_path, label_free_selection)
    selection_digest = sha256_file(selection_path)
    write_json(
        output_dir / "retrospective_search_selection_freeze.json",
        {
            "selection_sha256": selection_digest,
            "outcome_column_present": False,
            "selection_status": "written_and_hashed_before_outcome_scoring",
            "interpretation": "retrospective ranking/retrieval of observed historical candidates",
            "tx100_status": "post_reveal_retrospective",
        },
    )
    labels = data.main[[ROW_ID, TARGET]].copy()
    compiled_with_labels = label_free.merge(labels, on=ROW_ID, validate="many_to_one")
    search_queries, search_summary = score_search_selections(
        label_free_selection, labels, compiled_with_labels
    )
    _write_csv(output_dir / "retrospective_search_query_results.csv", search_queries)
    _write_csv(output_dir / "retrospective_search_summary.csv", search_summary)

    summary_index = summary.set_index("model")
    condition_1 = bool(
        summary_index.loc["V2-P", "chemistry_macro_mape_percent"]
        < summary_index.loc["B1", "chemistry_macro_mape_percent"]
    )
    condition_2 = bool(differences["V2_P_beats_B1"].sum() >= 4)
    condition_3 = bool(
        summary_index.loc["V2-P", "worst_group_mape_percent"]
        <= summary_index.loc["B1", "worst_group_mape_percent"] + 2.0
    )
    condition_4 = bool(
        summary_index.loc["V2-P", "point_physical_invalid_rate"] == 0.0
        and summary_index.loc["V2-P", "interval_physical_invalid_rate"] == 0.0
    )
    condition_5 = bool(
        {"query_coverage", "answered_query_precision", "successful_target_recall"}.issubset(
            search_summary.columns
        )
    )
    condition_6 = TARGET not in label_free_selection.columns
    locked_conditions = {
        "1_macro_mape_lower_than_B1": condition_1,
        "2_improves_at_least_four_of_six_groups": condition_2,
        "3_worst_group_no_more_than_2pp_worse": condition_3,
        "4_zero_point_and_interval_invalid_rate": condition_4,
        "5_search_reports_coverage_recall_and_precision": condition_5,
        "6_no_outer_outcome_used_for_selection": condition_6,
    }
    decision = {
        "run_mode": run_mode,
        "protocol_compliant_full_run": protocol_compliant,
        "locked_conditions": locked_conditions,
        "all_locked_conditions_hold": bool(all(locked_conditions.values())),
        "beneficial_claim_allowed": beneficial_claim_allowed(
            protocol_compliant=protocol_compliant,
            locked_conditions=locked_conditions,
        ),
        "fast_smoke_warning": (
            "Fast mode changes seed count, epoch cap, and patience; never use its outcomes inferentially."
            if fast
            else None
        ),
        "tx100_status": "post_reveal_retrospective",
        "new_blind_validation_required": (
            "A genuinely new V2 blind validation requires a different never-inspected dataset."
        ),
    }
    write_json(output_dir / "locked_decision_rule.json", decision)

    configuration = {
        "protocol_sha256": LOCKED_PROTOCOL_SHA256,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "implementation_test_sha256": sha256_file(
            project_dir / "tests" / "test_native_physics_v2.py"
        ),
        "run_mode": run_mode,
        "protocol_compliant_full_run": protocol_compliant,
        "seeds": list(seeds),
        "locked_full_seeds": list(LOCKED_SEEDS),
        "max_epochs": max_epochs,
        "patience": patience,
        "neural_architecture": "native 6-32-32; physics 6-16-16; concat 48-16-1; SiLU",
        "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4), full batch",
        "lambda_p_grid": list(LAMBDA_P_GRID),
        "lambda_m_grid": list(LAMBDA_M_GRID),
        "outer_groups_cmc_mm": list(MAIN_GROUPS),
        "interval_claim": (
            "LOCO OOF bounded uncertainty heuristic only; no formal coverage guarantee"
        ),
        "device": "cpu",
        "tx100_status": "post_reveal_retrospective",
    }
    write_json(output_dir / "configuration.json", configuration)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "torch": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_device": "cpu",
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }
    write_json(output_dir / "environment.json", environment)
    hashes = _hash_artifacts(output_dir)
    write_json(output_dir / "artifact_hashes.json", hashes)
    verify_artifact_hash_manifest(output_dir)
    return {
        "output_dir": str(output_dir),
        "run_mode": run_mode,
        "artifact_count": len(hashes),
        "decision": decision,
        "summary": summary.to_dict(orient="records"),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_project = Path(__file__).resolve().parents[1]
    parser.add_argument("--project-dir", type=Path, default=default_project)
    parser.add_argument(
        "--output-dir", type=Path, default=default_project / "v2_artifacts_full_run1"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Non-inferential pipeline smoke test with one seed and shorter training.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.monotonic()
    result = run_study(
        project_dir=args.project_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        fast=bool(args.fast),
    )
    elapsed = time.monotonic() - started
    result["elapsed_seconds_not_part_of_hashed_artifacts"] = elapsed
    print(dumps_strict(result, sort_keys=True))


if __name__ == "__main__":
    main()
