"""W5 invariants: weighted conformal radius reduces to the finite-sample quantile when
weights are constant, is monotone in the test weight, and returns +inf when the
calibration mass cannot reach the coverage level."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from w3.weighted_conformal import (  # noqa: E402
    domain_odds, effective_sample_size, knn_density_ratio, weighted_radii, weighted_radius,
)


def finite_sample_quantile(scores: np.ndarray, coverage: float) -> float:
    values = np.sort(np.asarray(scores, dtype=float))
    rank = min(int(math.ceil((len(values) + 1) * coverage)), len(values))
    return float(values[rank - 1])


class WeightedRadiusTests(unittest.TestCase):
    def test_constant_weights_reduce_to_finite_sample_quantile(self) -> None:
        rng = np.random.default_rng(0)
        for n in (7, 20, 53, 200):
            residuals = rng.exponential(size=n)
            for coverage in (0.8, 0.9, 0.95):
                observed = weighted_radius(residuals, np.ones(n), 1.0, coverage)
                if math.ceil((n + 1) * coverage) <= n:
                    # attainable: identical to the clamped finite-sample quantile used by W3
                    self.assertAlmostEqual(observed, finite_sample_quantile(residuals, coverage), places=12)
                else:
                    # not attainable: the test point's atom at +inf makes the interval infinite,
                    # where W3's quantile clamps and flags the coverage ceiling instead
                    self.assertTrue(math.isinf(observed))

    def test_infinite_when_test_weight_dominates(self) -> None:
        residuals = np.arange(1.0, 11.0)
        self.assertTrue(math.isinf(weighted_radius(residuals, np.ones(10), 5.0, 0.9)))
        self.assertFalse(math.isinf(weighted_radius(residuals, np.ones(10), 1.0, 0.9)))

    def test_monotone_in_test_weight(self) -> None:
        rng = np.random.default_rng(1)
        residuals = rng.gamma(2.0, size=60)
        cal_w = rng.uniform(0.5, 2.0, size=60)
        radii = weighted_radii(residuals, cal_w, np.array([0.1, 1.0, 5.0, 50.0]), 0.9)
        self.assertTrue(all(radii[i] <= radii[i + 1] for i in range(3)))

    def test_upweighting_large_residuals_widens(self) -> None:
        residuals = np.linspace(0.1, 1.0, 50)
        flat = weighted_radius(residuals, np.ones(50), 1.0, 0.9)
        tilted = weighted_radius(residuals, np.linspace(0.1, 5.0, 50), 1.0, 0.9)
        self.assertGreaterEqual(tilted, flat)


class WeightHelperTests(unittest.TestCase):
    def test_domain_odds_clipped_and_monotone(self) -> None:
        p = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        w = domain_odds(p)
        self.assertTrue(np.all(np.diff(w) > 0))
        self.assertTrue(np.isfinite(w).all())
        self.assertAlmostEqual(w[2], 1.0)

    def test_knn_density_ratio_scaling(self) -> None:
        w = knn_density_ratio(np.array([1.0, 2.0]), n_features=3, n_fit=100, n_test=50)
        self.assertAlmostEqual(w[0], 2.0)
        self.assertAlmostEqual(w[1], 16.0)

    def test_effective_sample_size_bounds(self) -> None:
        self.assertAlmostEqual(effective_sample_size(np.ones(40)), 40.0)
        self.assertAlmostEqual(effective_sample_size(np.array([1.0, 0.0, 0.0])), 1.0)


if __name__ == "__main__":
    unittest.main()
