from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat


WORKSPACE = Path(__file__).resolve().parents[3]
SOURCE_DIR = WORKSPACE / "work" / "microfluidic_reliability" / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import blind_custodian as custodian
import blind_predict as predictor
import blind_reveal as revealer


class BlindWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.protocol = (
            WORKSPACE
            / "work"
            / "microfluidic_reliability"
            / "BLIND_VALIDATION_PROTOCOL.md"
        )
        rng = np.random.default_rng(314159)
        groups = np.repeat(np.array([0.0, 1.0, 2.0, 4.0, 3.5]), [18, 18, 18, 18, 12])
        n_rows = len(groups)
        continuous = rng.uniform(0.02, 0.22, n_rows)
        dispersed = rng.uniform(0.003, 0.055, n_rows)
        tension = rng.uniform(2.0, 28.0, n_rows)
        concentration = np.where(
            groups == 0,
            0.0,
            rng.uniform(0.1, 2.2, n_rows) * groups,
        )
        normalized = np.divide(
            concentration,
            groups,
            out=np.zeros_like(concentration),
            where=groups != 0,
        )
        diameter = (
            0.045
            + 0.22 * dispersed
            + 0.055 * continuous
            + 0.0014 * tension
            - 0.004 * normalized
            + rng.normal(0.0, 0.0015, n_rows)
        )
        matrix = np.column_stack(
            [continuous, dispersed, tension, concentration, groups, diameter]
        )
        permutation = rng.permutation(n_rows)
        first_indices = permutation[: n_rows // 2]
        second_indices = permutation[n_rows // 2 :]
        self.mat_a = self.root / "source_a.mat"
        self.mat_b = self.root / "source_b.mat"
        savemat(self.mat_a, {"published_table": matrix[first_indices]})
        savemat(self.mat_b, {"published_table": matrix[second_indices]})
        self.sealed_dir = self.root / "sealed"
        custodian.create_sealed_artifacts(
            [self.mat_a, self.mat_b],
            self.sealed_dir,
            self.protocol,
            source_urls=["https://example.invalid/a", "https://example.invalid/b"],
            source_md5=["a" * 32, "b" * 32],
            expected_blind_count=None,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _predict(self, output_name: str) -> dict:
        result = predictor.generate_prereveal_artifact(
            self.sealed_dir / "development_labeled.csv",
            self.sealed_dir / "blind_features.csv",
            self.root / output_name,
            self.protocol,
            self.sealed_dir / "custodian_manifest.json",
        )
        return result

    def test_custodian_separates_blind_labels_from_model_inputs(self) -> None:
        development = pd.read_csv(self.sealed_dir / "development_labeled.csv")
        blind = pd.read_csv(self.sealed_dir / "blind_features.csv")
        labels = pd.read_csv(self.sealed_dir / "sealed_blind_labels.csv")
        manifest = json.loads(
            (self.sealed_dir / "custodian_manifest.json").read_text(encoding="utf-8")
        )

        self.assertIn(custodian.TARGET_COLUMN, development.columns)
        self.assertNotIn(custodian.TARGET_COLUMN, blind.columns)
        self.assertEqual(
            list(labels.columns),
            [custodian.ROW_ID_COLUMN, custodian.TARGET_COLUMN],
        )
        self.assertEqual(len(blind), 12)
        self.assertTrue(np.allclose(blind["cmc_mm"], 3.5))
        self.assertEqual(manifest["counts"]["blind"], 12)
        self.assertNotIn("mean", json.dumps(manifest).lower())
        self.assertNotIn("minimum", json.dumps(manifest).lower())
        self.assertNotIn("maximum", json.dumps(manifest).lower())

    def test_changing_sealed_labels_cannot_change_predictions_or_rankings(self) -> None:
        first = self._predict("prediction_first")["bundle"]
        labels_path = self.sealed_dir / "sealed_blind_labels.csv"
        labels = pd.read_csv(labels_path)
        labels[custodian.TARGET_COLUMN] = labels[custodian.TARGET_COLUMN] * 7.0 + 0.123
        labels.to_csv(labels_path, index=False)

        second = self._predict("prediction_second")["bundle"]
        self.assertEqual(first["predictions"], second["predictions"])
        self.assertEqual(first["queries"], second["queries"])
        self.assertEqual(first["rankings"], second["rankings"])

    def test_hash_gate_and_opaque_id_gate_reject_tampering(self) -> None:
        result = self._predict("prediction_hash_gate")
        bundle_path = result["bundle_path"]
        hash_path = result["hash_path"]
        self.assertEqual(
            revealer.verify_bundle_hash(bundle_path, hash_path),
            result["bundle_sha256"],
        )
        original = bundle_path.read_text(encoding="utf-8")
        bundle_path.write_text(original + " ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            revealer.verify_bundle_hash(bundle_path, hash_path)

        predictions = pd.DataFrame(
            {custodian.ROW_ID_COLUMN: ["opaque-a", "opaque-b"]}
        )
        wrong_labels = pd.DataFrame(
            {
                custodian.ROW_ID_COLUMN: ["opaque-a", "opaque-c"],
                custodian.TARGET_COLUMN: [0.1, 0.2],
            }
        )
        with self.assertRaisesRegex(ValueError, "row-ID set mismatch"):
            revealer.verify_opaque_ids(predictions, wrong_labels)

    def test_predictor_rejects_any_blind_target_column(self) -> None:
        development = pd.read_csv(self.sealed_dir / "development_labeled.csv")
        blind = pd.read_csv(self.sealed_dir / "blind_features.csv")
        blind[custodian.TARGET_COLUMN] = 0.1
        with self.assertRaisesRegex(ValueError, "public-input schema"):
            predictor.validate_input_tables(development, blind)


if __name__ == "__main__":
    unittest.main()
