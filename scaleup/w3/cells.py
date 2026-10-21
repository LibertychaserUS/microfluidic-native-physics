"""Evaluation cells (shift type × dataset) and the fit/calibration split — D2, D3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .datasets import D2_GROUP_ORDER, DatasetSpec


@dataclass(frozen=True)
class Settings:
    fast: bool
    n_random_seeds: int
    n_geometry_seeds: int
    permutations: int
    ensemble_members: int = 20
    nominal_coverage: float = 0.90
    threshold_quantile: float = 0.95

    @classmethod
    def for_mode(cls, fast: bool) -> "Settings":
        if fast:
            return cls(fast=True, n_random_seeds=2, n_geometry_seeds=2, permutations=200)
        return cls(fast=False, n_random_seeds=15, n_geometry_seeds=30, permutations=2000)


@dataclass(frozen=True)
class Cell:
    cell_id: str
    family: str
    shift_type: str
    repeat: int
    spec_train: DatasetSpec
    train_idx: np.ndarray
    spec_test: DatasetSpec
    test_idx: np.ndarray
    test_group_key: str
    cal_rule: str  # "dependency_group" | "row"
    grouped: bool
    status_note: str

    @property
    def test_groups(self) -> np.ndarray:
        return self.spec_test.groups[self.test_group_key][self.test_idx]

    @property
    def cal_group_key(self) -> str:
        """Grouping available on the training table (the external table's
        `source` key does not exist on the comprehensive table)."""

        return "geometry" if self.spec_train.name == "D1" else "chemistry"

    def assert_integrity(self) -> dict[str, int]:
        """D2 — zero overlap of rows (same table) and of grouping units when grouped."""

        same_table = self.spec_train is self.spec_test
        if same_table and set(self.train_idx.tolist()) & set(self.test_idx.tolist()):
            raise AssertionError(f"{self.cell_id}: train/test rows overlap")
        overlap = 0
        if self.grouped and same_table:
            key = self.test_group_key
            train_units = set(self.spec_train.groups[key][self.train_idx].tolist())
            test_units = set(self.spec_test.groups[key][self.test_idx].tolist())
            overlap = len(train_units & test_units)
            if overlap:
                raise AssertionError(f"{self.cell_id}: grouping unit crossed train/test")
        return {
            "n_train": int(len(self.train_idx)),
            "n_test": int(len(self.test_idx)),
            "group_unit_overlap": int(overlap),
            "n_test_groups": int(len(np.unique(self.test_groups))),
        }


def _row_random(n_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    train, test = train_test_split(
        np.arange(n_rows), test_size=0.2, random_state=seed, shuffle=True
    )
    return np.sort(train), np.sort(test)


def _group_random(groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train, test = next(splitter.split(np.zeros(len(groups)), groups=groups))
    return np.sort(train), np.sort(test)


def make_cells(
    d1_comp: DatasetSpec, d1_ext: DatasetSpec, d2: DatasetSpec, settings: Settings
) -> list[Cell]:
    """D2 — the seven cell families of W3_PROTOCOL.md §3."""

    cells: list[Cell] = []
    for seed in range(settings.n_random_seeds):  # D2.1 D1_random
        train, test = _row_random(len(d1_comp.frame), seed)
        cells.append(
            Cell("D1_random", "D1_random", "reference", seed, d1_comp, train, d1_comp, test,
                 "geometry", "dependency_group", False, "row_random_reference")
        )
    for seed in range(settings.n_geometry_seeds):  # D2.2 D1_geometry
        train, test = _group_random(d1_comp.groups["geometry"], seed)
        cells.append(
            Cell("D1_geometry", "D1_geometry", "device_geometry", seed, d1_comp, train,
                 d1_comp, test, "geometry", "dependency_group", True, "grouped_by_geometry")
        )
    cells.append(  # D2.3 D1_external
        Cell("D1_external", "D1_external", "cross_literature", 0, d1_comp,
             np.arange(len(d1_comp.frame)), d1_ext, np.arange(len(d1_ext.frame)),
             "source", "dependency_group", False, "external_literature_sources_inspected_before")
    )
    event = d1_comp.groups["event_type"]
    se_idx = np.flatnonzero(event == "SE")
    de_idx = np.flatnonzero(event == "DE")
    cells.append(  # D2.4 D1_se_to_de
        Cell("D1_se_to_de", "D1_se_to_de", "device_type", 0, d1_comp, se_idx, d1_comp, de_idx,
             "geometry", "dependency_group", True, "single_emulsion_to_double_emulsion")
    )
    cells.append(  # D2.4 D1_de_to_se
        Cell("D1_de_to_se", "D1_de_to_se", "device_type", 0, d1_comp, de_idx, d1_comp, se_idx,
             "geometry", "dependency_group", True, "double_emulsion_to_single_emulsion")
    )
    for seed in range(settings.n_random_seeds):  # D2.5 D2_random
        train, test = _row_random(len(d2.frame), seed)
        cells.append(
            Cell("D2_random", "D2_random", "reference", seed, d2, train, d2, test,
                 "chemistry", "row", False, "row_random_reference")
        )
    chemistry = d2.groups["chemistry"]
    for fold, group in enumerate(D2_GROUP_ORDER):  # D2.6 D2_chemistry_loco
        test = np.flatnonzero(chemistry == group)
        train = np.flatnonzero(chemistry != group)
        note = "post_reveal_retrospective" if group == "3.5" else "not_tx100"
        cells.append(
            Cell("D2_chemistry_loco", "D2_chemistry_loco", "whole_chemistry", fold, d2, train,
                 d2, test, "chemistry", "row", True, f"heldout_cmc_{group}:{note}")
        )
    for cell in cells:
        cell.assert_integrity()
    return cells


MIN_CALIBRATION_GROUPS = 5


def split_fit_calibration(cell: Cell, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    """D3 — divide the training rows into fit and calibration partitions.

    Returns the rule actually used. A dependency-group rule falls back to the
    row rule when the training rows contain fewer than five dependency groups
    (the DE→SE cell: all DE rows form two connected components); the fallback is
    reported in `cells.csv` and `deviations.json`, never applied silently.
    """

    train = cell.train_idx
    rule = cell.cal_rule
    if rule == "dependency_group":
        groups = cell.spec_train.groups["geometry"][train]
        if len(np.unique(groups)) < MIN_CALIBRATION_GROUPS:
            rule = "row"
    if rule == "dependency_group":
        groups = cell.spec_train.groups["geometry"][train]
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        fit_pos, cal_pos = next(splitter.split(np.zeros(len(train)), groups=groups))
        if set(groups[fit_pos].tolist()) & set(groups[cal_pos].tolist()):
            raise AssertionError("dependency group crossed fit/calibration partitions")
    elif rule == "row":
        permutation = np.random.default_rng(seed).permutation(len(train))
        n_cal = max(10, int(round(0.20 * len(train))))
        cal_pos, fit_pos = permutation[:n_cal], permutation[n_cal:]
    else:
        raise ValueError(f"unknown calibration rule {rule}")
    fit_idx = np.sort(train[fit_pos])
    cal_idx = np.sort(train[cal_pos])
    if len(fit_idx) < 20 or len(cal_idx) < 5:
        raise ValueError(f"{cell.cell_id}: fit/calibration partition too small")
    if set(fit_idx.tolist()) & set(cal_idx.tolist()):
        raise AssertionError("fit and calibration rows overlap")
    return fit_idx, cal_idx, rule
