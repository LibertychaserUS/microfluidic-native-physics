from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    WORKSPACE
    / "work"
    / "microfluidic_reliability"
    / "src"
    / "interval_experiments.py"
)
SPEC = importlib.util.spec_from_file_location("interval_experiments", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class IntervalExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        data_dir = WORKSPACE / "work" / "data" / "osf_938rs"
        cls.comprehensive = MODULE.DAFD.load_dataset(
            data_dir / "Comprehensive_normalized.xlsx", require_rate=True
        )
        cls.external = MODULE.DAFD.load_dataset(
            data_dir / "Generalizability_data_normalized.xlsx",
            require_rate=False,
        )

    def test_finite_sample_quantile_uses_exact_order_statistic(self) -> None:
        scores = np.array([1.0, 2.0, 3.0, 4.0])
        # ceil((4 + 1) * .5) = 3, so the third order statistic is 3.
        self.assertEqual(MODULE.finite_sample_quantile(scores, 0.50), 3.0)
        # ceil((4 + 1) * .8) = 4, capped at n.
        self.assertEqual(MODULE.finite_sample_quantile(scores, 0.80), 4.0)
        with self.assertRaises(ValueError):
            MODULE.finite_sample_quantile(np.array([]), 0.90)

    def test_group_calibration_uses_group_maxima_and_group_quantile(self) -> None:
        scores = np.array([1.0, 100.0, 2.0, 3.0, 4.0, 5.0])
        groups = np.array([0, 0, 1, 1, 2, 2])
        row_state = MODULE.calibrate_conformal_scores(scores, 0.50)
        group_state = MODULE.calibrate_conformal_scores(scores, 0.50, groups)
        # Row rank: ceil(7 * .5) = 4 -> 4. Group maxima are [100, 3, 5];
        # group rank: ceil(4 * .5) = 2 -> 5.
        self.assertEqual(row_state.threshold, 4.0)
        self.assertEqual(group_state.threshold, 5.0)
        self.assertEqual(group_state.calibration_unit, "dependency_group")
        self.assertEqual(group_state.n_calibration_units, 3)
        self.assertEqual(group_state.score_aggregation, "group_max")
        self.assertTrue(group_state.nominal_finite_threshold_attainable)

        unattainable = MODULE.calibrate_conformal_scores(scores, 0.90, groups)
        self.assertFalse(unattainable.nominal_finite_threshold_attainable)
        self.assertEqual(unattainable.requested_rank, 4)
        self.assertEqual(unattainable.used_rank, 3)
        self.assertEqual(unattainable.finite_threshold_coverage_ceiling, 0.75)

    def test_dependency_aware_split_keeps_groups_and_de_pairs_together(self) -> None:
        split = MODULE.make_interval_split(
            self.comprehensive,
            split_mode="geometry_group",
            seed=17,
        )
        split.validate(len(self.comprehensive))
        membership = np.empty(len(self.comprehensive), dtype=int)
        membership[split.train_indices] = 0
        membership[split.calibration_indices] = 1
        membership[split.test_indices] = 2

        for inner in range(197):
            self.assertEqual(membership[inner], membership[inner + 671])

        group_partition: dict[int, int] = {}
        for row_index, group in enumerate(split.dependency_groups.tolist()):
            partition = int(membership[row_index])
            if group in group_partition:
                self.assertEqual(group_partition[group], partition)
            else:
                group_partition[group] = partition

    def test_all_methods_produce_finite_ordered_endpoints(self) -> None:
        split = MODULE.make_interval_split(
            self.comprehensive,
            split_mode="row_random",
            seed=3,
        )
        train = self.comprehensive.iloc[split.train_indices]
        calibration = self.comprehensive.iloc[split.calibration_indices]
        for method in MODULE.INTERVAL_METHODS:
            with self.subTest(method=method):
                predictor = MODULE.fit_interval_predictor(
                    train,
                    calibration,
                    method=method,
                    coverage=0.90,
                    seed=3,
                )
                endpoints = predictor.predict(self.external)
                self.assertEqual(len(endpoints), len(self.external))
                self.assertTrue(
                    np.isfinite(endpoints[MODULE.ENDPOINT_COLUMNS]).all().all()
                )
                self.assertTrue(
                    np.all(
                        endpoints["lower_normalized"].to_numpy()
                        <= endpoints["upper_normalized"].to_numpy()
                    )
                )

    def test_group_protocol_uses_group_calibration_metadata(self) -> None:
        result = MODULE.run_interval_experiment(
            self.comprehensive,
            self.external,
            method="split_conformal",
            coverage=0.90,
            split_mode="geometry_group",
            seed=19,
        )
        expected_units = len(
            np.unique(
                result.split.dependency_groups[
                    result.split.calibration_indices
                ]
            )
        )
        self.assertEqual(result.predictor.calibration_unit, "dependency_group")
        self.assertEqual(result.predictor.n_calibration_units, expected_units)
        self.assertGreaterEqual(expected_units, 19)
        self.assertTrue(
            result.predictor.nominal_finite_threshold_attainable
        )
        self.assertEqual(
            result.predictor.calibration_score_aggregation, "group_max"
        )
        self.assertTrue(
            (result.external_rows["calibration_unit"] == "dependency_group").all()
        )

    def test_normalized_scale_uses_group_oof_residuals_without_leakage(self) -> None:
        split = MODULE.make_interval_split(
            self.comprehensive,
            split_mode="geometry_group",
            seed=23,
        )
        training_groups = split.dependency_groups[split.train_indices]
        calibration_groups = split.dependency_groups[split.calibration_indices]
        self.assertFalse(
            set(training_groups.tolist()) & set(calibration_groups.tolist())
        )
        predictor = MODULE.fit_interval_predictor(
            self.comprehensive.iloc[split.train_indices],
            self.comprehensive.iloc[split.calibration_indices],
            method="normalized_split_conformal",
            coverage=0.90,
            seed=23,
            training_groups=training_groups,
            calibration_groups=calibration_groups,
        )
        self.assertEqual(predictor.scale_residual_protocol, "group_oof")
        assignments = predictor.scale_oof_fold_assignments
        self.assertIsNotNone(assignments)
        assert assignments is not None
        self.assertEqual(len(assignments), len(training_groups))
        for group in np.unique(training_groups):
            self.assertEqual(
                len(np.unique(assignments[training_groups == group])),
                1,
            )

    def test_external_label_changes_do_not_change_endpoints(self) -> None:
        split = MODULE.make_interval_split(
            self.comprehensive,
            split_mode="row_random",
            seed=11,
        )
        predictor = MODULE.fit_interval_predictor(
            self.comprehensive.iloc[split.train_indices],
            self.comprehensive.iloc[split.calibration_indices],
            method="split_conformal",
            coverage=0.95,
            seed=11,
        )
        original = predictor.predict(self.external)
        changed = self.external.copy()
        changed[MODULE.DAFD.TARGET_NORMALIZED] = np.linspace(
            10.0, 20.0, len(changed)
        )
        changed[MODULE.DAFD.TARGET_UM] = np.linspace(
            1_000.0, 2_000.0, len(changed)
        )
        after_change = predictor.predict(changed)
        np.testing.assert_allclose(
            original[MODULE.ENDPOINT_COLUMNS].to_numpy(),
            after_change[MODULE.ENDPOINT_COLUMNS].to_numpy(),
            rtol=0.0,
            atol=0.0,
        )

    def test_fast_suite_returns_analysis_ready_tables(self) -> None:
        suite = MODULE.run_interval_suite(
            self.comprehensive,
            self.external,
            fast=True,
        )
        self.assertEqual(
            set(suite),
            {
                "internal_predictions",
                "external_predictions",
                "summary",
                "splits",
                "split_composition",
                "split_de_pair_audit",
                "calibration_scores",
                "calibration_unit_scores",
                "physical_bound_diagnostics",
            },
        )
        external = suite["external_predictions"]
        required = {
            "row_index",
            "method",
            "protocol",
            "nominal",
            "seed",
            "lower_um",
            "upper_um",
        }
        self.assertTrue(required.issubset(external.columns))
        self.assertEqual(set(external["method"]), set(MODULE.INTERVAL_METHODS))
        self.assertEqual(set(external["protocol"]), set(MODULE.SPLIT_MODES))
        self.assertEqual(set(external["nominal"]), set(MODULE.SUPPORTED_COVERAGES))
        self.assertEqual(len(external), 12 * len(self.external))
        calibration = suite["calibration_scores"]
        self.assertTrue(
            {
                "development_row_position",
                "dependency_group",
                "event_type",
                "nonconformity_score",
            }.issubset(calibration.columns)
        )
        self.assertTrue(np.isfinite(calibration["nonconformity_score"]).all())
        for keys, group in suite["calibration_unit_scores"].groupby(
            ["method", "protocol", "nominal", "seed"]
        ):
            selected = group["selected_threshold"].unique()
            self.assertEqual(len(selected), 1, keys)
            reconstructed = MODULE.finite_sample_quantile(
                group["unit_nonconformity_score"].to_numpy(dtype=float),
                float(keys[2]),
            )
            self.assertAlmostEqual(float(selected[0]), reconstructed, places=14)
        self.assertTrue(
            {
                "n_train_rows",
                "n_calibration_rows",
                "n_internal_test_rows",
                "train_n_dependency_groups",
                "calibration_n_dependency_groups",
                "internal_test_n_dependency_groups",
            }.issubset(suite["summary"].columns)
        )

    def test_seed_zero_de_pair_and_event_composition_is_explicit(self) -> None:
        records = []
        compositions = []
        for protocol in MODULE.SPLIT_MODES:
            split = MODULE.make_interval_split(
                self.comprehensive, split_mode=protocol, seed=0
            )
            membership = np.empty(len(self.comprehensive), dtype=object)
            membership[split.train_indices] = "train"
            membership[split.calibration_indices] = "calibration"
            membership[split.test_indices] = "test"
            records.append(
                {
                    "protocol": protocol,
                    **MODULE._de_pair_allocation(
                        membership, len(self.comprehensive)
                    ),
                }
            )
            physical = MODULE._physical_clusters(
                self.comprehensive, "comprehensive"
            )
            event = MODULE._event_types(len(self.comprehensive), "comprehensive")
            for partition, indices in (
                ("train", split.train_indices),
                ("calibration", split.calibration_indices),
                ("test", split.test_indices),
            ):
                compositions.append(
                    {
                        "protocol": protocol,
                        "partition": partition,
                        **MODULE._partition_audit(
                            indices,
                            split.dependency_groups,
                            physical,
                            event,
                        ),
                    }
                )
        by_protocol = {row["protocol"]: row for row in records}
        self.assertEqual(
            by_protocol["row_random"]["n_de_pairs_crossing_partitions"],
            110,
        )
        grouped = by_protocol["geometry_group"]
        self.assertEqual(grouped["n_de_pairs_crossing_partitions"], 0)
        self.assertEqual(grouped["n_de_pairs_train"], 11)
        self.assertEqual(grouped["n_de_pairs_calibration"], 186)
        self.assertEqual(grouped["n_de_pairs_test"], 0)
        grouped_test = next(
            row
            for row in compositions
            if row["protocol"] == "geometry_group"
            and row["partition"] == "test"
        )
        self.assertEqual(grouped_test["n_de_rows"], 0)

    def test_interval_summary_exports_physical_scale_diagnostics(self) -> None:
        result = MODULE.run_interval_experiment(
            self.comprehensive,
            self.external,
            method="split_conformal",
            coverage=0.90,
            split_mode="row_random",
            seed=0,
        )
        summary = result.summary["external"]["overall"]
        self.assertIn("n_negative_lower_endpoints", summary)
        self.assertIn("n_upper_endpoints_above_hydraulic_diameter", summary)
        self.assertIn("max_width_um", summary)
        self.assertGreaterEqual(summary["max_width_um"], 0.0)


if __name__ == "__main__":
    unittest.main()
