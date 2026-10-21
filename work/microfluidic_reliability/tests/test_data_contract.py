from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[3]
MODULE_PATH = WORKSPACE / "work" / "microfluidic_reliability" / "src" / "dafd_reproduction.py"
SPEC = importlib.util.spec_from_file_location("dafd_reproduction", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DataContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data_dir = WORKSPACE / "work" / "data" / "osf_938rs"

    def test_public_datasets_have_expected_rows_and_complete_features(self) -> None:
        comprehensive = MODULE.load_dataset(
            self.data_dir / "Comprehensive_normalized.xlsx", require_rate=True
        )
        external = MODULE.load_dataset(
            self.data_dir / "Generalizability_data_normalized.xlsx", require_rate=False
        )
        self.assertEqual(len(comprehensive), 868)
        self.assertEqual(len(external), 64)
        self.assertFalse(comprehensive[MODULE.FEATURES].isna().any().any())
        self.assertFalse(external[MODULE.FEATURES].isna().any().any())
        self.assertEqual(external[MODULE.REFERENCE].nunique(), 4)

    def test_mass_conservation_rate_recovers_observed_values_with_rounding_error(self) -> None:
        comprehensive = MODULE.load_dataset(
            self.data_dir / "Comprehensive_normalized.xlsx", require_rate=True
        )
        reconstructed = MODULE.generation_rate_hz(
            comprehensive[MODULE.DISPERSED_FLOW].to_numpy(),
            comprehensive[MODULE.TARGET_UM].to_numpy(),
        )
        observed = comprehensive[MODULE.TARGET_RATE].to_numpy()
        percentage_error = 100.0 * np.abs(reconstructed - observed) / observed
        # Diameters in the spreadsheet are rounded, so exact equality is not
        # expected. The conservation formula nevertheless reconstructs nearly
        # all rates to substantially below one percent.
        self.assertLess(float(np.median(percentage_error)), 0.1)
        self.assertLess(float(np.quantile(percentage_error, 0.95)), 0.5)
        self.assertLess(float(np.max(percentage_error)), 5.0)


if __name__ == "__main__":
    unittest.main()
