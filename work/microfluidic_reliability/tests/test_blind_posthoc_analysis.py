from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[3]
SOURCE_DIR = WORKSPACE / "work" / "microfluidic_reliability" / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import blind_posthoc_analysis as posthoc


class BlindPosthocAnalysisTests(unittest.TestCase):
    def test_tie_aware_aurc_admits_complete_tie_blocks(self) -> None:
        errors = np.array([0.1, 0.3, 0.2])
        scores = np.array([0.0, 0.0, 1.0])
        curve, aurc = posthoc.tie_aware_risk_curve(errors, scores)
        self.assertEqual(curve["retained"].tolist(), [2, 3])
        self.assertEqual(curve["tie_block_size"].tolist(), [2, 1])
        self.assertTrue(np.allclose(curve["relative_risk"], [0.2, 0.2]))
        self.assertAlmostEqual(aurc, 0.2)

        permuted_curve, permuted_aurc = posthoc.tie_aware_risk_curve(
            errors[[1, 0, 2]], scores[[1, 0, 2]]
        )
        self.assertTrue(
            np.allclose(curve["relative_risk"], permuted_curve["relative_risk"])
        )
        self.assertAlmostEqual(aurc, permuted_aurc)

    def test_benjamini_hochberg_is_monotone_in_sorted_p_values(self) -> None:
        adjusted = posthoc.benjamini_hochberg(
            {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.002}
        )
        self.assertAlmostEqual(adjusted["d"], 0.008)
        self.assertAlmostEqual(adjusted["a"], 0.02)
        self.assertAlmostEqual(adjusted["c"], 0.04)
        self.assertAlmostEqual(adjusted["b"], 0.04)

    def test_wilson_interval_contains_observed_fraction(self) -> None:
        lower, upper = posthoc.wilson_interval(39, 87)
        observed = 39 / 87
        self.assertLess(lower, observed)
        self.assertGreater(upper, observed)
        self.assertGreaterEqual(lower, 0.0)
        self.assertLessEqual(upper, 1.0)

    def test_full_retention_matched_budget_null_is_degenerate(self) -> None:
        corrected, reason = posthoc.correct_matched_budget_p(0.444155, 87, 87)
        self.assertEqual(corrected, 1.0)
        self.assertIn("degenerate null", reason)
        unchanged, reason = posthoc.correct_matched_budget_p(0.0265, 78, 87)
        self.assertEqual(unchanged, 0.0265)
        self.assertIn("nondegenerate", reason)

    def test_independent_diagnostics_preserve_signed_bias_direction(self) -> None:
        predictions = pd.DataFrame(
            {
                posthoc.ROW_ID: ["a", "b"],
                "predicted_diameter_mm": [0.12, 0.18],
                "conformal_lower_mm": [0.10, 0.15],
                "conformal_upper_mm": [0.14, 0.20],
            }
        )
        labels = pd.DataFrame(
            {posthoc.ROW_ID: ["a", "b"], posthoc.TARGET: [0.10, 0.20]}
        )
        merged = posthoc.point_diagnostics(predictions, labels)
        summary = posthoc.summarize_point(merged)
        self.assertTrue(np.allclose(merged["signed_error_mm"], [0.02, -0.02]))
        self.assertAlmostEqual(
            summary["signed_bias_mm_prediction_minus_observation"], 0.0
        )
        self.assertAlmostEqual(summary["overprediction_fraction"], 0.5)
        self.assertAlmostEqual(summary["underprediction_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
