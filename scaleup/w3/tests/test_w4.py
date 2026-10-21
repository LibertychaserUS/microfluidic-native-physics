"""W4 invariants: quantile-forest weights and quantiles, interval ordering, determinism."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from w3.uq_models import (  # noqa: E402
    DeepEnsemble, GaussianProcessModel, QuantileForest, Z90, weighted_quantile,
)


def _toy(n: int = 120, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, size=(n, 3))
    y = 1.5 + x[:, 0] ** 2 - 0.7 * x[:, 1] + 0.1 * rng.normal(size=n)
    return x, y


class WeightedQuantileTests(unittest.TestCase):
    def test_matches_brute_force_inverse_cdf(self) -> None:
        rng = np.random.default_rng(1)
        for _ in range(50):
            values = rng.normal(size=25)
            weights = rng.uniform(size=25)
            q = float(rng.uniform(0.01, 0.99))
            order = np.argsort(values, kind="mergesort")
            cdf = np.cumsum(weights[order]) / weights.sum()
            expected = values[order][int(np.argmax(cdf >= q))]
            self.assertAlmostEqual(weighted_quantile(values, weights, q), expected, places=12)

    def test_uniform_weights_reduce_to_empirical_quantile(self) -> None:
        values = np.arange(10, dtype=float)
        self.assertEqual(weighted_quantile(values, np.ones(10), 0.5), 4.0)
        self.assertEqual(weighted_quantile(values, np.ones(10), 0.05), 0.0)
        self.assertEqual(weighted_quantile(values, np.ones(10), 0.95), 9.0)


class QuantileForestTests(unittest.TestCase):
    def test_weights_are_row_stochastic_and_quantiles_ordered(self) -> None:
        x, y = _toy()
        qrf = QuantileForest(seed=3).fit(x[:90], y[:90])
        weights = qrf.weights(x[90:])
        self.assertTrue(np.allclose(weights.sum(axis=1), 1.0))
        self.assertTrue((weights >= 0).all())
        pred = qrf.predict(x[90:])
        self.assertTrue(np.all(pred.lower <= pred.upper))
        self.assertTrue(np.all(pred.score >= 0))
        # native 90% interval should cover most in-distribution held-out rows
        covered = np.mean((y[90:] >= pred.lower) & (y[90:] <= pred.upper))
        self.assertGreater(covered, 0.6)

    def test_deterministic(self) -> None:
        x, y = _toy()
        a = QuantileForest(seed=5).fit(x, y).predict(x[:10])
        b = QuantileForest(seed=5).fit(x, y).predict(x[:10])
        self.assertTrue(np.array_equal(a.point, b.point) and np.array_equal(a.lower, b.lower))


class GaussianProcessAndEnsembleTests(unittest.TestCase):
    def test_gp_interval_is_symmetric_and_deterministic(self) -> None:
        x, y = _toy(n=80)
        a = GaussianProcessModel(seed=2, n_features=3).fit(x, y).predict(x[:7])
        b = GaussianProcessModel(seed=2, n_features=3).fit(x, y).predict(x[:7])
        self.assertTrue(np.allclose(a.upper - a.point, Z90 * a.score))
        self.assertTrue(np.allclose(a.point - a.lower, Z90 * a.score))
        self.assertTrue(np.array_equal(a.point, b.point))

    def test_ensemble_sd_is_between_member_spread_and_deterministic(self) -> None:
        x, y = _toy(n=100)
        a = DeepEnsemble(seed=4, standardise_target=False).fit(x, y)
        b = DeepEnsemble(seed=4, standardise_target=False).fit(x, y)
        pa, pb = a.predict(x[:6]), b.predict(x[:6])
        self.assertTrue(np.array_equal(pa.point, pb.point))
        self.assertTrue(np.all(pa.score >= 0))
        self.assertEqual(len(a.n_iter), 5)
        self.assertTrue(np.allclose(pa.upper - pa.lower, 2 * Z90 * pa.score))


if __name__ == "__main__":
    unittest.main()
