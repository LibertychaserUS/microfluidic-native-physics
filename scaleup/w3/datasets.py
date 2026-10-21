"""Dataset specifications for D1 (DAFD) and D2 (Chagot) — design D1 (§2.1).

Both datasets are expressed through one `DatasetSpec` so that cells, models,
intervals and applicability scores never branch on the dataset name except
inside the two loaders below.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .workspace import Workspace, sha256_file

D1_WORKBOOKS = (
    "Comprehensive_normalized.xlsx",
    "Generalizability_data_normalized.xlsx",
)
D2_B1_COLUMNS = ("b1_Qc", "b1_Qd", "b1_gamma", "b1_c_over_cmc")
D2_ANCHOR_SCALE_UM = 195.0  # D_PRIMARY_MM * 1000 (locked V2 protocol §5)
D2_GROUP_ORDER = ("0", "2", "3.5", "4.33", "11", "20")


@dataclass
class DatasetSpec:
    """One table plus everything a cell needs; see W3_DESIGN.md §2.1 (D1)."""

    name: str
    frame: pd.DataFrame
    feature_cols: list[str]
    target_um: np.ndarray
    response: np.ndarray
    response_kind: str  # "normalized_by_hydraulic" | "mm"
    size_proxy: np.ndarray
    cluster_id: np.ndarray
    groups: dict[str, np.ndarray]
    row_id: np.ndarray
    audit: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.frame)
        for label, array in (
            ("target_um", self.target_um),
            ("response", self.response),
            ("size_proxy", self.size_proxy),
            ("cluster_id", self.cluster_id),
            ("row_id", self.row_id),
        ):
            if len(array) != n:
                raise ValueError(f"{self.name}: {label} length {len(array)} != {n}")
        for key, array in self.groups.items():
            if len(array) != n:
                raise ValueError(f"{self.name}: group {key} length mismatch")
        if len(np.unique(self.row_id)) != n:
            raise ValueError(f"{self.name}: row identifiers are not unique")
        if not np.isfinite(self.target_um).all() or np.any(self.target_um <= 0):
            raise ValueError(f"{self.name}: physical targets must be positive and finite")

    def features(self, index: np.ndarray) -> np.ndarray:
        return self.frame.iloc[np.asarray(index, dtype=int)][self.feature_cols].to_numpy(
            dtype=float
        )

    def to_um(self, response_pred: np.ndarray, index: np.ndarray) -> np.ndarray:
        """Convert a model-response prediction to micrometres (monotone, positive)."""

        response_pred = np.asarray(response_pred, dtype=float)
        if self.response_kind == "normalized_by_hydraulic":
            hydraulic = self.frame.iloc[np.asarray(index, dtype=int)][
                "Hydraulic diameter"
            ].to_numpy(dtype=float)
            return response_pred * hydraulic
        if self.response_kind == "mm":
            return response_pred * 1000.0
        raise ValueError(f"unknown response kind {self.response_kind}")


def _manifest_digests(workspace: Workspace) -> dict[str, str]:
    text = (workspace.root / "SOURCE_MANIFEST.md").read_text(encoding="utf-8")
    pattern = re.compile(r"^\|\s*([A-Za-z0-9_.]+\.xlsx)\s*\|\s*([0-9a-f]{64})\s*\|", re.M)
    return {name: digest for name, digest in pattern.findall(text)}


def load_d1(workspace: Workspace) -> tuple[DatasetSpec, DatasetSpec]:
    """D1.1 — comprehensive (868) and external (64) tables with digest audit."""

    dafd, core, intervals = workspace.dafd, workspace.core, workspace.intervals
    expected = _manifest_digests(workspace)
    digests: dict[str, str] = {}
    for name in D1_WORKBOOKS:
        path = workspace.data_dir / name
        observed = sha256_file(path)
        if expected.get(name) != observed:
            raise ValueError(
                f"{name}: sha256 {observed[:12]} does not match SOURCE_MANIFEST.md"
            )
        digests[name] = observed

    comprehensive = dafd.load_dataset(
        workspace.data_dir / D1_WORKBOOKS[0], require_rate=True
    )
    external = dafd.load_dataset(workspace.data_dir / D1_WORKBOOKS[1], require_rate=False)
    if len(comprehensive) != 868 or len(external) != 64:
        raise ValueError("D1 row counts differ from the locked 868/64")

    comp_clusters = core.comprehensive_experiment_clusters(comprehensive)
    ext_clusters = core.external_experiment_clusters(external)
    comp_groups = {
        "geometry": core.dependency_group_ids(
            comprehensive, core.GEOMETRY_FEATURES, comp_clusters
        ),
        "event_type": intervals._event_types(868, "comprehensive").astype(str),
    }
    ext_groups = {
        "geometry": core.dependency_group_ids(external, core.GEOMETRY_FEATURES, ext_clusters),
        "event_type": intervals._event_types(64, "external").astype(str),
        "source": external[dafd.REFERENCE].astype(str).to_numpy(),
    }
    audit = {
        "workbook_sha256": digests,
        "n_comprehensive": 868,
        "n_external": 64,
        "n_se_rows": int((comp_groups["event_type"] == "SE").sum()),
        "n_de_rows": int((comp_groups["event_type"] == "DE").sum()),
        "n_geometry_groups": int(len(np.unique(comp_groups["geometry"]))),
        "external_sources": {
            str(k): int(v) for k, v in pd.Series(ext_groups["source"]).value_counts().items()
        },
    }
    comp_spec = DatasetSpec(
        name="D1",
        frame=comprehensive,
        feature_cols=list(dafd.FEATURES),
        target_um=comprehensive[dafd.TARGET_UM].to_numpy(dtype=float),
        response=comprehensive[dafd.TARGET_NORMALIZED].to_numpy(dtype=float),
        response_kind="normalized_by_hydraulic",
        size_proxy=comprehensive[dafd.HYDRAULIC_DIAMETER].to_numpy(dtype=float),
        cluster_id=np.asarray(comp_clusters, dtype=int),
        groups=comp_groups,
        row_id=np.array([f"D1:comp:{i}" for i in range(868)], dtype=object),
        audit=audit,
    )
    ext_spec = DatasetSpec(
        name="D1",
        frame=external,
        feature_cols=list(dafd.FEATURES),
        target_um=external[dafd.TARGET_UM].to_numpy(dtype=float),
        response=external[dafd.TARGET_NORMALIZED].to_numpy(dtype=float),
        response_kind="normalized_by_hydraulic",
        size_proxy=external[dafd.HYDRAULIC_DIAMETER].to_numpy(dtype=float),
        cluster_id=np.asarray(ext_clusters, dtype=int),
        groups=ext_groups,
        row_id=np.array([f"D1:ext:{i}" for i in range(64)], dtype=object),
        audit=audit,
    )
    return comp_spec, ext_spec


def load_d2(workspace: Workspace) -> DatasetSpec:
    """D1.2 — Chagot main table (466 rows) via the custodian-lock-verified join."""

    v2 = workspace.v2
    data = v2.load_joined_public_data(workspace.root / "blind_artifacts")
    main = data.main.reset_index(drop=True).copy()
    if len(main) != 466:
        raise ValueError("D2 main table must contain exactly 466 rows")
    b1 = v2.b1_features(main)
    for column, values in zip(D2_B1_COLUMNS, b1.T):
        main[column] = values
    anchor_unclipped, anchor_valid = v2.semi_empirical_anchor(main)
    chemistry = np.array(
        [format(float(value), ".12g") for value in main[v2.CMC].to_numpy(dtype=float)],
        dtype=object,
    )
    if set(chemistry.tolist()) != set(D2_GROUP_ORDER):
        raise ValueError(f"unexpected D2 chemistry groups: {sorted(set(chemistry))}")
    audit = {
        **data.audit,
        "anchor_invalid_rows": int((~anchor_valid).sum()),
        "feature_contract": "B1 four published inputs (Qc, Qd, gamma_eq, c/CMC)",
        "size_proxy": "Kalli/Chagot anchor diameter (label-free), um",
    }
    return DatasetSpec(
        name="D2",
        frame=main,
        feature_cols=list(D2_B1_COLUMNS),
        target_um=main[v2.TARGET].to_numpy(dtype=float) * 1000.0,
        response=main[v2.TARGET].to_numpy(dtype=float),
        response_kind="mm",
        size_proxy=np.asarray(anchor_unclipped, dtype=float) * D2_ANCHOR_SCALE_UM,
        cluster_id=np.arange(466, dtype=int),
        groups={"chemistry": chemistry},
        row_id=main[v2.ROW_ID].astype(str).to_numpy(dtype=object),
        audit=audit,
    )
