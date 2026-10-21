from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[3]
SRC = WORKSPACE / "work" / "microfluidic_reliability" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import study_core as core


class StudyCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        data = WORKSPACE / "work" / "data" / "osf_938rs"
        self.comprehensive = core.DAFD.load_dataset(
            data / "Comprehensive_normalized.xlsx", require_rate=True
        )
        self.external = core.DAFD.load_dataset(
            data / "Generalizability_data_normalized.xlsx", require_rate=False
        )

    def test_double_emulsion_pairs_share_physical_cluster(self) -> None:
        clusters = core.comprehensive_experiment_clusters(self.comprehensive)
        self.assertEqual(len(np.unique(clusters)), 671)
        np.testing.assert_array_equal(clusters[:197], clusters[671:868])
        self.assertEqual(len(np.unique(clusters[197:671])), 474)

    def test_external_blind_double_emulsion_pairs_share_cluster(self) -> None:
        clusters = core.external_experiment_clusters(self.external)
        self.assertEqual(len(np.unique(clusters)), 54)
        for inner, outer in zip(range(44, 49), range(49, 54)):
            self.assertEqual(clusters[inner], clusters[outer])
        for inner, outer in zip(range(54, 59), range(59, 64)):
            self.assertEqual(clusters[inner], clusters[outer])

    def test_dependency_groups_join_geometry_and_pairs(self) -> None:
        physical = core.comprehensive_experiment_clusters(self.comprehensive)
        groups = core.dependency_group_ids(
            self.comprehensive, core.GEOMETRY_FEATURES, physical
        )
        for inner, outer in zip(range(197), range(671, 868)):
            self.assertEqual(groups[inner], groups[outer])
        geometry = core.group_ids(self.comprehensive, core.GEOMETRY_FEATURES)
        sample_group = geometry[250]
        members = np.flatnonzero(geometry == sample_group)
        self.assertTrue(np.all(groups[members] == groups[members[0]]))

    def test_risk_curve_orders_low_scores_first(self) -> None:
        errors = np.array([1.0, 2.0, 8.0, 9.0])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        curve, aurc = core.risk_curve(errors, scores, minimum_count=1)
        self.assertAlmostEqual(float(curve.iloc[0]["risk"]), 1.0)
        self.assertAlmostEqual(float(curve.iloc[-1]["risk"]), 5.0)
        expected = np.mean([1.0, 1.5, 11.0 / 3.0, 5.0])
        self.assertAlmostEqual(aurc, expected)

    def test_risk_curve_is_full_and_tie_order_invariant(self) -> None:
        errors = np.array([1.0, 9.0, 3.0, 7.0])
        scores = np.array([0.0, 0.0, 1.0, 1.0])
        curve, aurc = core.risk_curve(errors, scores, minimum_count=99)
        reversed_curve, reversed_aurc = core.risk_curve(
            errors[::-1], scores[::-1], minimum_count=1
        )
        self.assertEqual(curve["retained"].tolist(), [2, 4])
        pd.testing.assert_frame_equal(
            curve.reset_index(drop=True), reversed_curve.reset_index(drop=True)
        )
        self.assertAlmostEqual(aurc, reversed_aurc)
        self.assertAlmostEqual(aurc, 5.0)

    def test_risk_score_calibration_split_preserves_dependencies(self) -> None:
        physical = core.comprehensive_experiment_clusters(self.comprehensive)
        bundle = core.fit_risk_scores(
            self.comprehensive,
            self.external,
            seed=17,
            group_columns=core.GEOMETRY_FEATURES,
            experiment_clusters=physical,
            include_expensive=True,
        )
        self.assertIsNotNone(bundle.metadata)
        metadata = bundle.metadata or {}
        self.assertEqual(metadata["dependency_group_overlap_count"], 0)
        self.assertTrue(metadata["experiment_clusters_supplied"])
        fit_positions = set(metadata["fit_row_positions"])
        calibration_positions = set(metadata["calibration_row_positions"])
        self.assertFalse(fit_positions & calibration_positions)
        self.assertEqual(
            fit_positions | calibration_positions,
            set(range(len(self.comprehensive))),
        )
        for inner in range(197):
            outer = inner + 671
            self.assertEqual(inner in fit_positions, outer in fit_positions)
            self.assertEqual(
                inner in calibration_positions,
                outer in calibration_positions,
            )
        for method, values in bundle.training.items():
            with self.subTest(partition="calibration", method=method):
                self.assertEqual(len(values), metadata["n_calibration_rows"])
                self.assertTrue(np.isfinite(values).all())
        for method, values in bundle.candidate.items():
            with self.subTest(partition="candidate", method=method):
                self.assertEqual(len(values), len(self.external))
                self.assertTrue(np.isfinite(values).all())
        self.assertEqual(
            metadata["lof_training_score_source"],
            "novelty_scores_on_held_out_calibration",
        )
        self.assertEqual(
            metadata["bootstrap_resampling_unit"],
            "fit_dependency_group",
        )
        self.assertTrue(metadata["oof_residual_risk_dependency_aware"])

    def test_unsupervised_candidate_scores_are_label_independent(self) -> None:
        physical = core.comprehensive_experiment_clusters(self.comprehensive)
        original = core.fit_risk_scores(
            self.comprehensive,
            self.external,
            seed=23,
            experiment_clusters=physical,
            include_expensive=False,
        )
        changed_development = self.comprehensive.copy()
        changed_candidate = self.external.copy()
        changed_development[core.DAFD.TARGET_NORMALIZED] = (
            changed_development[core.DAFD.TARGET_NORMALIZED].to_numpy()[::-1]
        )
        changed_development[core.DAFD.TARGET_UM] = (
            changed_development[core.DAFD.TARGET_UM].to_numpy()[::-1]
        )
        changed_candidate[core.DAFD.TARGET_NORMALIZED] = np.linspace(
            10.0, 20.0, len(changed_candidate)
        )
        changed_candidate[core.DAFD.TARGET_UM] = np.linspace(
            1_000.0, 2_000.0, len(changed_candidate)
        )
        changed = core.fit_risk_scores(
            changed_development,
            changed_candidate,
            seed=23,
            experiment_clusters=physical,
            include_expensive=False,
        )
        unsupervised = {
            "random_score",
            "axis_aligned",
            "knn_3",
            "knn_5",
            "knn_10",
            "knn_20",
            "mahalanobis_oas",
            "isolation_forest",
            "local_outlier_factor",
            "one_class_svm",
            "hydraulic_diameter_only",
        }
        for method in sorted(unsupervised):
            with self.subTest(method=method):
                np.testing.assert_allclose(
                    original.training[method],
                    changed.training[method],
                    rtol=0.0,
                    atol=0.0,
                )
                np.testing.assert_allclose(
                    original.candidate[method],
                    changed.candidate[method],
                    rtol=0.0,
                    atol=0.0,
                )
        np.testing.assert_array_equal(
            original.strict_minmax_candidate,
            changed.strict_minmax_candidate,
        )

    def test_candidate_ranking_cannot_depend_on_observed_labels(self) -> None:
        predicted = np.array([50.0, 52.0, 80.0])
        allowed = np.array([True, True, True])
        first = core.rank_candidates(predicted, 51.0, allowed, 5.0, 2)
        # There is deliberately no observed-label argument. Two hypothetical
        # label vectors therefore cannot affect the result.
        labels_a = np.array([1.0, 2.0, 3.0])
        labels_b = labels_a[::-1]
        second = core.rank_candidates(predicted, 51.0, allowed, 5.0, 2)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, np.array([0, 1]))


if __name__ == "__main__":
    unittest.main()
