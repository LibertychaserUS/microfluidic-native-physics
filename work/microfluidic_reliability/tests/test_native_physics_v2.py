from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


WORKSPACE = Path(__file__).resolve().parents[3]
SRC = WORKSPACE / "work" / "microfluidic_reliability" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import native_physics_v2 as v2
import blind_predict


class NativePhysicsV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = v2.load_joined_public_data(
            WORKSPACE / "work" / "microfluidic_reliability" / "blind_artifacts"
        )

    def test_locked_protocol_hash_and_auditable_join(self) -> None:
        protocol = WORKSPACE / "work" / "microfluidic_reliability" / v2.PROTOCOL_NAME
        self.assertEqual(v2.sha256_file(protocol), v2.LOCKED_PROTOCOL_SHA256)
        self.assertEqual(len(self.data.full), 468)
        self.assertEqual(len(self.data.main), 466)
        self.assertEqual(len(self.data.quarantine), 2)
        self.assertEqual(set(self.data.quarantine[v2.CMC]), {0.6})
        self.assertFalse(self.data.full[v2.ROW_ID].duplicated().any())
        lock = self.data.audit["custodian_lock"]
        self.assertTrue(lock["all_expected_observed_checks_passed"])
        self.assertTrue(
            all(item["sha256_match"] for item in lock["artifacts"].values())
        )
        self.assertTrue(
            all(item["sha256_match"] for item in lock["raw_sources"].values())
        )

    def test_custodian_hash_lock_rejects_one_byte_artifact_mutation(self) -> None:
        project = WORKSPACE / "work" / "microfluidic_reliability"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "blind_artifacts"
            raw = root / "blind_raw"
            artifacts.mkdir()
            raw.mkdir()
            for name in (
                "development_labeled.csv",
                "blind_features.csv",
                "sealed_blind_labels.csv",
                "custodian_manifest.json",
            ):
                shutil.copyfile(project / "blind_artifacts" / name, artifacts / name)
            for name in ("matTrainingDataSet.mat", "matTestDataSet.mat"):
                shutil.copyfile(project / "blind_raw" / name, raw / name)
            target = artifacts / "development_labeled.csv"
            target.write_bytes(target.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "artifact lock mismatch"):
                v2.load_joined_public_data(artifacts)

    def test_si_conversion_and_physical_identities(self) -> None:
        sample = self.data.main.iloc[:32].copy()
        phys = v2.physics_layer(sample)
        qc_si = sample[v2.QC].to_numpy(float) * 1.0e-6 / 60.0
        qd_si = sample[v2.QD].to_numpy(float) * 1.0e-6 / 60.0
        gamma_si = sample[v2.GAMMA].to_numpy(float) * 1.0e-3
        np.testing.assert_allclose(phys["Qc_m3_s"], qc_si, rtol=0, atol=1e-18)
        np.testing.assert_allclose(phys["Qd_m3_s"], qd_si, rtol=0, atol=1e-18)
        np.testing.assert_allclose(phys["gamma_N_m"], gamma_si, rtol=0, atol=1e-15)
        np.testing.assert_allclose(phys["flow_ratio"], qd_si / qc_si, rtol=1e-12)
        expected_ca = v2.MU_C * qc_si / (gamma_si * v2.OVAL_AREA_M2)
        np.testing.assert_allclose(phys["Ca_c"], expected_ca, rtol=1e-12)
        expected_re = v2.RHO_C * qc_si * v2.D_PRIMARY_M / (v2.MU_C * v2.OVAL_AREA_M2)
        np.testing.assert_allclose(phys["Re_c"], expected_re, rtol=1e-12)

    def test_derived_columns_are_ignored_and_recomputed(self) -> None:
        sample = self.data.main.iloc[:5].copy()
        reference = v2.physics_layer(sample)
        sample["flow_ratio"] = -999.0
        sample["Ca_c"] = 123456.0
        changed = v2.physics_layer(sample)
        pd.testing.assert_frame_equal(reference, changed)

    def test_native_identity_and_no_surfactant_contract(self) -> None:
        native = v2.native_state(self.data.main)
        no_surf = native[v2.CMC].eq(0.0)
        self.assertTrue(native.loc[no_surf, v2.CONCENTRATION].eq(0.0).all())
        self.assertTrue(native.loc[no_surf, "is_no_surfactant"].eq(1.0).all())
        self.assertTrue(native.loc[~no_surf, "is_no_surfactant"].eq(0.0).all())
        ratio = (
            native.loc[~no_surf, v2.CONCENTRATION]
            / native.loc[~no_surf, v2.CMC]
        )
        np.testing.assert_allclose(
            ratio,
            self.data.main.loc[~no_surf, v2.CONCENTRATION_OVER_CMC],
            rtol=1e-12,
            atol=1e-12,
        )

    def test_hard_parameterization_is_strictly_bounded(self) -> None:
        latent = np.array([-1e6, -100.0, 0.0, 100.0, 1e6])
        diameter = v2.bounded_diameter_from_logit(latent)
        self.assertTrue(np.all(diameter > 0.0))
        self.assertTrue(np.all(diameter < v2.D_PRIMARY_MM))
        lower, upper = v2.bounded_interval_from_logit(latent, 12.0)
        self.assertTrue(np.all(lower > 0.0))
        self.assertTrue(np.all(lower < upper))
        self.assertTrue(np.all(upper < v2.D_PRIMARY_MM))

    def test_b1_is_exactly_the_sealed_v1_raw_mm_xgboost(self) -> None:
        artifact_dir = WORKSPACE / "work" / "microfluidic_reliability" / "blind_artifacts"
        development = pd.read_csv(artifact_dir / "development_labeled.csv")
        blind_features = pd.read_csv(artifact_dir / "blind_features.csv")

        sealed = blind_predict.build_locked_model()
        candidate = v2.make_xgb(blind_predict.RANDOM_STATE)
        self.assertEqual(candidate.get_params(), sealed.get_params())
        self.assertIsNone(candidate.get_params()["tree_method"])

        sealed.fit(
            development[blind_predict.MODEL_FEATURES],
            development[v2.TARGET].to_numpy(float),
        )
        v2_fit = v2.fit_xgb(
            development,
            kind="B1",
            seed=blind_predict.RANDOM_STATE,
        )
        sealed_prediction = sealed.predict(blind_features[blind_predict.MODEL_FEATURES])
        v2_prediction = v2.predict_xgb(v2_fit, blind_features)
        np.testing.assert_array_equal(v2_prediction, sealed_prediction)

    def test_b2_keeps_raw_mm_target_so_ablation_changes_only_inputs(self) -> None:
        artifact_dir = WORKSPACE / "work" / "microfluidic_reliability" / "blind_artifacts"
        development = pd.read_csv(artifact_dir / "development_labeled.csv")
        blind_features = pd.read_csv(artifact_dir / "blind_features.csv")
        scaler = v2.StandardScaler().fit(
            v2.native_transforms(development).to_numpy(float)
        )
        train_x = scaler.transform(v2.native_transforms(development).to_numpy(float))
        blind_x = scaler.transform(v2.native_transforms(blind_features).to_numpy(float))
        reference = v2.make_xgb(42)
        reference.fit(train_x, development[v2.TARGET].to_numpy(float))

        candidate = v2.fit_xgb(development, kind="B2", seed=42)
        np.testing.assert_array_equal(
            v2.predict_xgb(candidate, blind_features),
            reference.predict(blind_x),
        )

    def test_invalid_anchor_uses_neutral_branch_and_cannot_affect_prediction(self) -> None:
        anchor, valid = v2._torch_anchor(self.data.main)
        self.assertEqual(int((~valid).sum()), 6)
        self.assertTrue(torch.equal(anchor[~valid], torch.zeros_like(anchor[~valid])))

        torch.manual_seed(42)
        model = v2.TwoBranchRegressor().cpu().eval()
        native = torch.zeros((len(anchor), 6), dtype=torch.float32)
        physics = torch.zeros((len(anchor), 6), dtype=torch.float32)
        tampered = anchor.clone()
        tampered[~valid] = 1.0e6
        with torch.no_grad():
            first, first_z, first_delta = v2._network_output(
                model, "V2-P", native, physics, anchor, valid
            )
            second, second_z, _ = v2._network_output(
                model, "V2-P", native, physics, tampered, valid
            )
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        torch.testing.assert_close(first_z, second_z, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            first_z[~valid], first_delta[~valid], rtol=0.0, atol=0.0
        )

    def test_loco_split_has_zero_chemistry_and_row_overlap(self) -> None:
        for group in v2.MAIN_GROUPS:
            train, test = v2.outer_loco_split(self.data.main, group)
            self.assertNotIn(group, set(train[v2.CMC].unique()))
            self.assertEqual(set(test[v2.CMC].unique()), {group})
            self.assertFalse(set(train[v2.ROW_ID]).intersection(test[v2.ROW_ID]))

    def test_exact_hyperparameter_tie_prefers_larger_penalties(self) -> None:
        scores = {
            (0.01, 0.01): 5.0,
            (0.1, 0.1): 5.0,
            (1.0, 0.01): 5.0,
            (1.0, 0.1): 5.0,
        }
        self.assertEqual(v2.choose_hyperparameter_candidate(scores), (1.0, 0.1))
        scores[(0.01, 0.01)] = 4.999999999
        self.assertEqual(v2.choose_hyperparameter_candidate(scores), (0.01, 0.01))

    def test_full_and_fast_settings_and_claim_gate_are_exact(self) -> None:
        full = v2.study_settings(False)
        self.assertEqual(full.seeds, (17, 42, 2024, 3407, 9001))
        self.assertEqual(full.max_epochs, 2000)
        self.assertEqual(full.patience, 150)
        self.assertTrue(full.protocol_compliant)
        fast = v2.study_settings(True)
        self.assertEqual(fast.seeds, (42,))
        self.assertEqual(fast.max_epochs, 20)
        self.assertEqual(fast.patience, 5)
        self.assertFalse(fast.protocol_compliant)
        self.assertFalse(
            v2.beneficial_claim_allowed(
                protocol_compliant=fast.protocol_compliant,
                locked_conditions={"all": True},
            )
        )

    def test_support_compiler_exports_every_conservative_gate(self) -> None:
        candidate = self.data.main.drop(columns=[v2.TARGET]).copy()
        candidate["heldout_group_cmc_mm"] = candidate[v2.CMC]
        candidate["model"] = "V2-P"
        candidate["prediction_mm"] = 0.12
        candidate["interval_lower_mm"] = 0.11
        candidate["interval_upper_mm"] = 0.13
        compiled = v2.add_compiler_support(candidate, self.data.main)
        self.assertNotIn(v2.TARGET, compiled.columns)
        gates = [
            "device_flow_limits_pass",
            "outer_training_marginal_flow_support_pass",
            "native_unit_and_positivity_pass",
            "heldout_cmc_fixed_pass",
            "same_chemistry_gamma_concentration_record_pass",
            "derived_finite_positive_pass",
        ]
        self.assertTrue(set(gates).issubset(compiled.columns))
        expected = compiled[gates].all(axis=1)
        pd.testing.assert_series_equal(
            compiled["compiled"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )
        self.assertTrue(compiled.groupby(v2.CMC)["compiled"].any().all())

    def test_label_free_search_selection_cannot_receive_outcomes(self) -> None:
        predictions = pd.DataFrame(
            {
                v2.ROW_ID: ["a", "b"],
                v2.CMC: [2.0, 2.0],
                v2.QC: [0.1, 0.2],
                v2.QD: [0.01, 0.02],
                v2.GAMMA: [4.0, 4.0],
                v2.CONCENTRATION: [1.0, 1.0],
                "model": ["V2-P", "V2-P"],
                "prediction_mm": [0.10, 0.12],
                "interval_lower_mm": [0.08, 0.10],
                "interval_upper_mm": [0.12, 0.14],
                "compiled": [True, True],
            }
        )
        chosen = v2.compile_label_free_search_selections(predictions, top_ks=(1,))
        self.assertNotIn(v2.TARGET, chosen.columns)
        self.assertGreater(len(chosen), 0)
        leaked = predictions.assign(**{v2.TARGET: [0.1, 0.2]})
        with self.assertRaisesRegex(ValueError, "outcome leakage"):
            v2.compile_label_free_search_selections(leaked, top_ks=(1,))

    def test_search_requires_point_and_full_interval_containment(self) -> None:
        predictions = pd.DataFrame(
            {
                v2.ROW_ID: ["narrow", "wide", "missing"],
                v2.CMC: [2.0, 2.0, 2.0],
                v2.QC: [0.1, 0.1, 0.1],
                v2.QD: [0.01, 0.01, 0.01],
                v2.GAMMA: [4.0, 4.0, 4.0],
                v2.CONCENTRATION: [1.0, 1.0, 1.0],
                "model": ["narrow", "wide", "missing"],
                "prediction_mm": [0.10, 0.10, 0.10],
                "interval_lower_mm": [0.095, 0.08, np.nan],
                "interval_upper_mm": [0.105, 0.12, np.nan],
                "compiled": [True, True, True],
            }
        )
        selected = v2.compile_label_free_search_selections(
            predictions, targets_mm=(0.10,), top_ks=(1,)
        )
        absolute = selected.loc[selected["tolerance_rule"].eq("absolute_0.01_mm")]
        answered = absolute.groupby("model")["answered_query"].first().to_dict()
        self.assertEqual(answered, {"missing": False, "narrow": True, "wide": False})
        self.assertTrue(
            absolute["abstention_rule"].eq("conservative_containment_abstention").all()
        )

    def test_json_summary_sanitizes_undefined_r_squared(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            v2.write_json(path, {"r2": float("nan"), "valid": 1.0})
            text = path.read_text(encoding="utf-8")
        self.assertNotIn("NaN", text)
        parsed = json.loads(text, parse_constant=lambda token: self.fail(token))
        self.assertIsNone(parsed["r2"])

    def test_hash_manifest_rehashes_and_strictly_parses_every_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "payload.json"
            v2.write_json(payload, {"undefined": float("nan"), "valid": 1.0})
            v2.write_json(
                root / "artifact_hashes.json",
                {payload.name: v2.sha256_file(payload)},
            )
            evidence = v2.verify_artifact_hash_manifest(root)
            self.assertTrue(evidence["hashes_match"])
            self.assertTrue(evidence["strict_json_parse"])
            payload.write_text('{"valid": 2}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                v2.verify_artifact_hash_manifest(root)


if __name__ == "__main__":
    unittest.main()
