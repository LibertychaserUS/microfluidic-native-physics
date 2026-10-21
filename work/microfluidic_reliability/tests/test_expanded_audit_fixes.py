from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[3]
SRC = WORKSPACE / "work" / "microfluidic_reliability" / "src"
sys.path.insert(0, str(SRC))

import expanded_study as MODULE  # noqa: E402


class ExpandedAuditFixTests(unittest.TestCase):
    def test_reproducibility_manifest_is_output_directory_name_invariant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence_a = root / "run_a" / "test_evidence.json"
            evidence_b = root / "run_b" / "test_evidence.json"
            evidence_a.parent.mkdir()
            evidence_b.parent.mkdir()
            payload = '{"status":"not_executed_by_analysis_script"}\n'
            evidence_a.write_text(payload, encoding="utf-8")
            evidence_b.write_text(payload, encoding="utf-8")
            manifest_a = MODULE.reproducibility_manifest(
                MODULE.Path(__file__).resolve().parents[1], evidence_a
            )
            manifest_b = MODULE.reproducibility_manifest(
                MODULE.Path(__file__).resolve().parents[1], evidence_b
            )
            self.assertEqual(manifest_a, manifest_b)
            paths = [item["path"] for item in manifest_a["governed_files"]]
            self.assertEqual(paths.count("test_evidence.json"), 1)

    def test_cluster_permutations_never_split_physical_de_blocks(self) -> None:
        scored = pd.DataFrame(
            {
                "source": ["A"] * 8,
                "source_context": [
                    "inner",
                    "outer",
                    "inner",
                    "outer",
                    "single",
                    "single",
                    "single",
                    "single",
                ],
                "physical_cluster": [0, 0, 1, 1, 2, 3, 4, 5],
            }
        )
        cluster_values = np.array([10, 10, 20, 20, 30, 40, 50, 60])
        for include_context in (False, True):
            with self.subTest(include_context=include_context):
                strata = MODULE._cluster_blocks(scored, include_context)
                blocks = [block for stratum in strata for block in stratum]
                self.assertIn({0, 1}, [set(block.tolist()) for block in blocks])
                self.assertIn({2, 3}, [set(block.tolist()) for block in blocks])
                permuted = MODULE._permute_cluster_blocks(
                    cluster_values,
                    strata,
                    np.random.default_rng(4),
                )
                for _, indices in scored.groupby("physical_cluster").groups.items():
                    positions = np.asarray(list(indices), dtype=int)
                    self.assertEqual(len(np.unique(permuted[positions])), 1)

    def test_source_macro_averages_endpoints_not_metadata(self) -> None:
        rows = pd.DataFrame(
            {
                "method": ["cqr", "cqr"],
                "protocol": ["geometry_group", "geometry_group"],
                "nominal": [0.9, 0.9],
                "seed": [0, 0],
                "dataset": ["external", "external"],
                "source": ["A", "B"],
                "empirical_coverage": [0.5, 0.9],
                "mean_width_um": [10.0, 20.0],
                "n_rows": [10, 20],
                "n_dependency_groups": [8, 15],
                "n_negative_lower_endpoints": [1, 2],
                "n_upper_endpoints_above_hydraulic_diameter": [3, 4],
                "calibration_unit": ["dependency_group"] * 2,
                "n_calibration_units": [19, 19],
                "calibration_score_aggregation": ["group_max"] * 2,
                "calibration_requested_rank": [18, 18],
                "calibration_used_rank": [18, 18],
                "nominal_finite_threshold_attainable": [True, True],
            }
        )
        macro = MODULE.source_macro_interval_summary(rows).iloc[0]
        self.assertAlmostEqual(float(macro["empirical_coverage"]), 0.7)
        self.assertAlmostEqual(float(macro["mean_width_um"]), 15.0)
        self.assertEqual(int(macro["sum_source_n_rows"]), 30)
        self.assertEqual(int(macro["sum_source_n_dependency_groups"]), 23)
        self.assertEqual(int(macro["n_calibration_units"]), 19)
        self.assertTrue(bool(macro["nominal_finite_threshold_attainable"]))
        self.assertNotIn("n_rows", macro.index)

    def test_context_conditioned_cluster_permutation_aligns_member_contexts(self) -> None:
        scored = pd.DataFrame(
            {
                "source": ["A"] * 4,
                # The second physical pair deliberately has reversed row order.
                "source_context": ["inner", "outer", "outer", "inner"],
                "physical_cluster": [0, 0, 1, 1],
            }
        )
        values = np.array([11, 12, 22, 21])
        strata = MODULE._cluster_blocks(scored, include_context=True)
        permuted = MODULE._permute_cluster_blocks(
            values,
            strata,
            np.random.default_rng(4),
        )
        permitted_context_vectors = {
            (("inner", 11), ("outer", 12)),
            (("inner", 21), ("outer", 22)),
        }
        for _, positions in scored.groupby("physical_cluster").groups.items():
            vector = tuple(
                sorted(
                    (
                        str(scored.loc[position, "source_context"]),
                        int(permuted[position]),
                    )
                    for position in positions
                )
            )
            self.assertIn(vector, permitted_context_vectors)

    def test_hierarchical_retention_ci_reports_conditional_validity(self) -> None:
        scored = pd.DataFrame(
            {
                "source": ["A", "A", "B", "B"],
                "physical_cluster": [0, 1, 2, 3],
                "absolute_relative_error": [0.1, 0.2, 0.3, 0.4],
                "absolute_normalized_error": [0.01, 0.02, 0.03, 0.04],
                "absolute_error_um": [1.0, 2.0, 3.0, 4.0],
                "accepted__strict_minmax": [True, False, False, False],
                "accepted__knn_5__q0_95": [True, False, False, False],
            }
        )
        summary = MODULE.hierarchical_bootstrap_external(scored, fast=True)
        retained = summary[
            summary["estimand"].astype(str).str.startswith("conditional_")
        ]
        self.assertGreater(len(retained), 0)
        self.assertTrue((retained["repetitions_requested"] == 500).all())
        self.assertTrue(
            (retained["repetitions_valid"] <= retained["repetitions_requested"]).all()
        )
        np.testing.assert_allclose(
            retained["valid_repetition_fraction"].to_numpy(dtype=float),
            retained["repetitions_valid"].to_numpy(dtype=float) / 500.0,
        )
        self.assertTrue(retained["conditional_on_nonempty_retention"].all())


if __name__ == "__main__":
    unittest.main()
