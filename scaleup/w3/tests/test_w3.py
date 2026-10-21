"""W3 invariants (Step 5 adversarial checks encoded as tests).

Data-dependent tests need MICROFLUIDIC_WORK pointing at work/microfluidic_reliability.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from w3.applicability import ALL_SCORES, score_applicability, within_group_median_mask  # noqa: E402
from w3.cells import Settings, make_cells, split_fit_calibration  # noqa: E402
from w3.evaluate import aurc_fast, benjamini_hochberg  # noqa: E402

WORK = os.environ.get("MICROFLUIDIC_WORK", "")


class PureFunctionTests(unittest.TestCase):
    def test_aurc_fast_matches_workspace_risk_curve_with_ties(self) -> None:
        if not WORK:
            self.skipTest("MICROFLUIDIC_WORK not set")
        from w3.workspace import load_workspace

        core = load_workspace(Path(WORK)).core
        rng = np.random.default_rng(3)
        for _ in range(20):
            errors = rng.uniform(size=40)
            scores = rng.integers(0, 6, size=40).astype(float)  # heavy ties
            _, expected = core.risk_curve(errors, scores)
            self.assertAlmostEqual(aurc_fast(errors, scores), expected, places=12)

    def test_within_group_median_keeps_half_per_group(self) -> None:
        scores = np.array([5.0, 1.0, 3.0, 2.0, 9.0, 8.0, 7.0])
        groups = np.array(["a", "a", "a", "a", "b", "b", "b"])
        keep = within_group_median_mask(scores, groups)
        self.assertEqual(keep[groups == "a"].sum(), 2)
        self.assertEqual(keep[groups == "b"].sum(), 1)
        self.assertTrue(keep[1] and keep[3] and keep[6])

    def test_benjamini_hochberg_is_monotone_and_bounded(self) -> None:
        adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.5, float("nan")])
        finite = [v for v in adjusted if np.isfinite(v)]
        self.assertTrue(all(0.0 <= v <= 1.0 for v in finite))
        self.assertTrue(np.isnan(adjusted[4]))
        order = np.argsort([0.01, 0.04, 0.03, 0.5])
        sorted_adj = [adjusted[i] for i in order]
        self.assertEqual(sorted_adj, sorted(sorted_adj))


@unittest.skipUnless(WORK, "MICROFLUIDIC_WORK not set")
class DataDependentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from w3.datasets import load_d1, load_d2
        from w3.workspace import load_workspace

        cls.ws = load_workspace(Path(WORK))
        cls.d1c, cls.d1e = load_d1(cls.ws)
        cls.d2 = load_d2(cls.ws)
        cls.cells = make_cells(cls.d1c, cls.d1e, cls.d2, Settings.for_mode(fast=True))

    def test_grouped_cells_have_zero_unit_overlap_and_expected_sizes(self) -> None:
        by_family = {}
        for cell in self.cells:
            by_family.setdefault(cell.family, []).append(cell)
            cell.assert_integrity()
        self.assertEqual(len(by_family["D2_chemistry_loco"]), 6)
        loco_sizes = sorted(len(c.test_idx) for c in by_family["D2_chemistry_loco"])
        self.assertEqual(loco_sizes, [15, 30, 63, 87, 93, 178])
        se_de = by_family["D1_se_to_de"][0]
        self.assertEqual((len(se_de.train_idx), len(se_de.test_idx)), (474, 394))

    def test_fit_calibration_partition_is_disjoint_and_grouped(self) -> None:
        geometry_cell = next(c for c in self.cells if c.family == "D1_geometry")
        fit_idx, cal_idx, rule = split_fit_calibration(geometry_cell, 0)
        self.assertEqual(rule, "dependency_group")
        self.assertFalse(set(fit_idx) & set(cal_idx))
        groups = geometry_cell.spec_train.groups["geometry"]
        self.assertFalse(set(groups[fit_idx]) & set(groups[cal_idx]))
        self.assertTrue(set(fit_idx) | set(cal_idx) == set(geometry_cell.train_idx))

    def test_scores_and_intervals_are_label_free_and_deterministic(self) -> None:
        from w3.intervals import INTERVAL_METHODS, fit_interval
        from w3.models import fit_point, predict_um

        cell = next(c for c in self.cells if c.family == "D1_external")
        fit_idx, cal_idx, _ = split_fit_calibration(cell, 0)
        cal_groups = cell.spec_train.groups[cell.cal_group_key][cal_idx]
        bundle_a = score_applicability(cell.spec_train, fit_idx, cal_idx, cell.spec_test,
                                       cell.test_idx, 0, 0.95, self.ws)
        # Permute every test label: nothing on the fitting/scoring path may change.
        original = cell.spec_test.target_um.copy()
        cell.spec_test.target_um = original[np.random.default_rng(1).permutation(len(original))]
        try:
            bundle_b = score_applicability(cell.spec_train, fit_idx, cal_idx, cell.spec_test,
                                           cell.test_idx, 0, 0.95, self.ws)
            for key in ALL_SCORES:
                np.testing.assert_array_equal(bundle_a.test[key], bundle_b.test[key])
            point = fit_point("xgboost", cell.spec_train, fit_idx, 0, self.ws)
            pred_a = predict_um(point, cell.spec_test, cell.test_idx)
            for method in INTERVAL_METHODS:
                fitted = fit_interval(method, point, cell.spec_train, fit_idx, cal_idx, cal_groups,
                                      0, 0.9, 4, self.ws)
                lower, upper = fitted.predict_um(cell.spec_test, cell.test_idx)
                self.assertTrue(np.all(lower <= upper))
                self.assertTrue(np.isfinite(lower).all() and np.isfinite(upper).all())
        finally:
            cell.spec_test.target_um = original
        point_again = fit_point("xgboost", cell.spec_train, fit_idx, 0, self.ws)
        np.testing.assert_array_equal(pred_a, predict_um(point_again, cell.spec_test, cell.test_idx))

    def test_d2_models_predict_in_micrometres(self) -> None:
        from w3.models import fit_point, predict_um

        cell = next(c for c in self.cells if c.family == "D2_random")
        fit_idx, _, _ = split_fit_calibration(cell, 0)
        for name in ("xgboost", "mlp"):
            point = fit_point(name, cell.spec_train, fit_idx, 0, self.ws)
            pred = predict_um(point, cell.spec_test, cell.test_idx)
            self.assertTrue(np.all(pred > 20) and np.all(pred < 400), name)


if __name__ == "__main__":
    unittest.main()
