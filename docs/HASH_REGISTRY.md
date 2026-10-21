# Hash registry

SHA-256 registry for the preprint 'Reliability Audits for Machine-Learned Droplet-Microfluidic Design Surrogates'. Section 'appendix_d' lists the 24 digests of the manuscript's Appendix D in order; 'supplementary' lists further frozen digests recorded in the artifact manifests. status: released = the file in release_path hashes to sha256; released_superset / released_redacted = the released file differs from the frozen bytes for the stated reason (released_sha256 gives the digest of the released file); not_redistributed_* = the bytes are not in this repository (third-party raw data, sealed labels, re-cut rows or a model binary); release_path is then where a local copy is expected after following work/data/README.md. Verify with scripts/verify_hash_registry.py.

Status values: `released` (bytes in this repository hash to the digest), `released_superset` / `released_redacted` (released file differs from the frozen bytes for the reason given in `hashes/appendix_d_registry.json`), `not_redistributed_*` (bytes are not in this repository).

## Appendix D of the manuscript (24 digests, in order)

| Object | SHA-256 | Status | Path |
|---|---|---|---|
| Comprehensive_normalized.xlsx | `aebb0136ff812f7233c8fd57ebc0f27f1b935736e8a9c1d4fdc20585f06b9c06` | not_redistributed_third_party_raw | `work/data/osf_938rs/Comprehensive_normalized.xlsx` |
| Generalizability_data_normalized.xlsx | `e90996da3e866e64922dc3b8aaef4b092ee6845fe95aac356c73e90ed108b303` | not_redistributed_third_party_raw | `work/data/osf_938rs/Generalizability_data_normalized.xlsx` |
| Final_Comprehensive_plus_Generalizability_data_normalized.xlsx | `14ebfaae93a7bbc7e1a73be91a629c57b1720dffac0dd45cef6d7df4e7eb6ac3` | not_redistributed_third_party_raw | `work/data/osf_938rs/Final_Comprehensive_plus_Generalizability_data_normalized.xlsx` |
| Raw_SE_dataset.xlsx | `996862065c33a3f17a36ad2df825179937ddbd7f3c8da41765946a9c1db5e579` | not_redistributed_third_party_raw | `work/data/osf_938rs/Raw_SE_dataset.xlsx` |
| matTrainingDataSet.mat | `7fab0693a7065198468a5d654226f5e2164985e334ed105f91355bdd49fb1a17` | not_redistributed_third_party_raw | `work/microfluidic_reliability/blind_raw/matTrainingDataSet.mat` |
| matTestDataSet.mat | `55f202bd19433331c86ddc65507e834036d294f7f9b4db2275f21ad91f6729da` | not_redistributed_third_party_raw | `work/microfluidic_reliability/blind_raw/matTestDataSet.mat` |
| microfluidics_dataset.zip (Zenodo 17867478) | `54272086fb390c788c25780856439ba32bfceeae3d40f19bc72af76529af13ee` | not_redistributed_third_party_raw | `work/data/zenodo_17867478/microfluidics_dataset.zip` |
| V1 protocol (BLIND_VALIDATION_PROTOCOL.md) | `1402752425f111a97222ae5c4cb7d13bd68ca1fdfabab978594ac79370e68b1c` | released | `work/microfluidic_reliability/BLIND_VALIDATION_PROTOCOL.md` |
| V1 pre-registration lock | `1f6901bc79de2cd189c7813d63e16ba561b7c502ce42721f4ed9e15a2357edc7` | released | `work/microfluidic_reliability/BLIND_PREREGISTRATION_LOCK.json` |
| V1 prediction bundle | `2b6eb42d709b367c563e62ca7c82ee764c06ac2b67dffc9f1127b3b732d6932d` | released | `work/microfluidic_reliability/blind_artifacts/prereveal/blind_prediction_bundle.json` |
| V1 sealed labels | `0e4c9a2577449446875d69ac51713aba510f7f492d7dffc4b6e3595746cc4fb9` | not_redistributed_sealed_labels | `work/microfluidic_reliability/blind_artifacts/sealed_blind_labels.csv` |
| V1 frozen reveal implementation | `46a289ad4369b10b56f8d1c233ba3a401769dec8fbcf392b1235a0baebd27a48` | released | `work/microfluidic_reliability/src/blind_reveal.py` |
| V1 consumed gate | `e285c4cb4ad7db74b9a3aca4b6240759081095c821865ae76852486f6fb13344` | released | `work/microfluidic_reliability/blind_artifacts/BLIND_REVEAL_CONSUMED.json` |
| V1 recovery implementation | `06b6099e0d7c754681bcee7de8f16b8eec8c6c902d4ca996d3e9d0086e263a74` | released | `work/microfluidic_reliability/src/blind_reveal_recovery.py` |
| V1 recovered result | `008d026163af37b7ecddf220d2ffc5aea0818784561a389419e088cf0eaa1624` | released | `work/microfluidic_reliability/blind_artifacts/blind_reveal_results_recovered.json` |
| V1 incident record | `75615dd3fc6f1d3ea26224c5243b56dd25003dcba48d935e502a6fa72e7db162` | released | `work/microfluidic_reliability/BLIND_REVEAL_INCIDENT.md` |
| V2 protocol (PHYSICS_CONSTRAINED_V2_PROTOCOL.md) | `5e4a1e4c4a8bb344ed0d1393477a7807b69cc57517eb359a03aec3cbf6471d55` | released | `work/microfluidic_reliability/PHYSICS_CONSTRAINED_V2_PROTOCOL.md` |
| V2 implementation (native_physics_v2.py) | `21ab9ea25eb1bfb8421c49d6a21b216242283286d33999eea88550c5f63b08c7` | released | `work/microfluidic_reliability/src/native_physics_v2.py` |
| Method-matrix protocol (W3_PROTOCOL.md) | `8f058306de4d51de1c3708443ed46d7c1556fef95ba3479a1cc7389ee46cb3f8` | released | `scaleup/W3_PROTOCOL.md` |
| Uncertainty-aware comparison protocol (W4_PROTOCOL.md) | `23dc7d0ae5431aeb89e3fe9a9174552e0d0258824658f97a674ccb216e951aa8` | released | `scaleup/W4_PROTOCOL.md` |
| Covariate-shift weighting protocol (W5_PROTOCOL.md) | `03af554431259a7971ca695db3a80e8cddc270814365c2f4c0c1f63960e60b40` | released | `scaleup/W5_PROTOCOL.md` |
| D3 sealed protocol (D3_BLIND_PROTOCOL.md) | `1fa4b404c964c1cfd2d78d2c1fbf5a9965e11560e4d07d2f4e005e15adc91419` | released | `scaleup/D3_BLIND_PROTOCOL.md` |
| D3 parameter binding (D3_BINDING.json) | `c943ce8bfab2e2db225e7d3fa3947b8f15ca54ac9063b866af2c1ca9157d284f` | released | `scaleup/D3_BINDING.json` |
| D3 prediction bundle | `79be728c7673803e329ce85574def2c34a869f52d3e2526a1a3161343d296bd2` | released | `work/microfluidic_reliability/d3_blind_artifacts/prereveal/blind_prediction_bundle.json` |

## Supplementary frozen digests

| Object | SHA-256 | Status | Path |
|---|---|---|---|
| V1 prediction program (blind_predict.py; digest recorded in the pre-registration lock) | `91ae4b08aaaee07fa94a5dfb51af234c0a17250ca1656e71b21ddcc9fb7a64cb` | released | `work/microfluidic_reliability/src/blind_predict.py` |
| V1 workflow test (tests/test_blind_workflow.py; digest recorded in the pre-registration lock) | `d32d63ca29bfbc70967177930c84d88ea45c6d3151ac1e88a076f5e38bba784a` | released | `work/microfluidic_reliability/tests/test_blind_workflow.py` |
| V2 implementation test (tests/test_native_physics_v2.py; digest recorded in v2 configuration.json) | `e64f22fc33860b32d77cbfe77d6c4947632b918a1d8d7655ef9d310f0024b8b7` | released | `work/microfluidic_reliability/tests/test_native_physics_v2.py` |
| D3 frozen prediction program (d3_predict.py; code_and_lock_hashes.json) | `30d99a478625d1a97d3e30453380d7ff4e900f2130b91625f836ac8cb04316e7` | released | `scaleup/w3/d3_predict.py` |
| D3 frozen reveal program (d3_reveal.py) | `630d08d9eb0d1b7c239afa130d5335ada974d49a2145e2f301bf43f996857328` | released | `scaleup/w3/d3_reveal.py` |
| D3 custodian program (d3_custodian.py) | `26f9afeff8b458c2296b41218fd1d9c90263377766686825340aa3a05887c639` | released | `scaleup/w3/d3_custodian.py` |
| D3 shared evaluation module (evaluate.py) | `ee595738a1f0e2e4673bc0b3de87066cf81102cbcb71a8e4ef6f5b8e68bff802` | released | `scaleup/w3/evaluate.py` |
| D3 custodian manifest | `6f5503b19cdc1ea029319dc4e1d8afceb33f66564e148e32fcd64c99d7f4e1f0` | released | `work/microfluidic_reliability/d3_blind_artifacts/custodian_manifest.json` |
| D3 deviation register as frozen before fitting (digest in code_and_lock_hashes.json) | `9cdd8374ff12edb430e7a4dd32b979f206f07917466d7aff5c24a684f921a3f1` | released_superset | `scaleup/D3_DEVIATION_REGISTER.md` |
| D3 sealed primary labels (laboratory B, 57 rows) | `30e15e490d77a59b4a2d3b0a451d0fda4a2b13ac56d5db4e3d141525c29fb797` | not_redistributed_sealed_labels | `work/microfluidic_reliability/d3_blind_artifacts/sealed_primary_sealed_labels.csv` |
| D3 sealed secondary labels (operator B, 12 rows) | `d07d094b545b596c2df34916b2ce119627dd8fdff052d055bf2a0f7ce0c62569` | not_redistributed_sealed_labels | `work/microfluidic_reliability/d3_blind_artifacts/sealed_secondary_sealed_labels.csv` |
| D3 development table (246 labelled laboratory-A rows) | `5e57bbee8e18b811d209c7924c0377e13ba44ab77351b35dfd8f88ca01c2ada2` | not_redistributed_derived_rows | `work/microfluidic_reliability/d3_blind_artifacts/development_labeled.csv` |
| D3 sealed primary features (inputs only, 57 rows) | `e26c7313596974188f2d34a0aacd95ee701cef774f4ad67fd6381b227b0cf00d` | not_redistributed_derived_rows | `work/microfluidic_reliability/d3_blind_artifacts/sealed_primary_features.csv` |
| D3 sealed secondary features (inputs only, 12 rows) | `4d88b424c696b768a6b54ef51a303c0739d03fb73a87169df5450b53379c5d38` | not_redistributed_derived_rows | `work/microfluidic_reliability/d3_blind_artifacts/sealed_secondary_features.csv` |
| V1 development table (381 labelled rows) | `b2889726960774acb2b25319f7544c31a1be4f22f319b1a1f4df88f98c7d7ab0` | not_redistributed_derived_rows | `work/microfluidic_reliability/blind_artifacts/development_labeled.csv` |
| V1 blind features (inputs only, 87 rows) | `6f76c9aa91b4f6e01c6b6e600b17f8a2b6600fde4085b0253c68f6b1df3aa29b` | not_redistributed_derived_rows | `work/microfluidic_reliability/blind_artifacts/blind_features.csv` |
| Zenodo 17867478 README.pdf | `a474c9c842aef90e20f870a3a0eb064235225e3bbef6c982454e6fef33c1af35` | not_redistributed_third_party_raw | `work/data/zenodo_17867478/README.pdf` |
| E0 reproduction model (dafd3_xgb_reproduction.ubj) | `72103e5c1185e382de890530679b8d74649a4f7ba7cf8438ff53c05b345f70f4` | not_redistributed_model_binary | `work/microfluidic_reliability/artifacts/dafd3_xgb_reproduction.ubj` |
| V1 corrected post-hoc summary as written by blind_posthoc_analysis.py | `eac78b1f943b9204920c30964b8193ef17990ddc951e1a903e522675cfaba5d0` | released_redacted | `work/microfluidic_reliability/blind_artifacts/posthoc_analysis_corrected/blind_posthoc_summary.json` |
| Zenodo 17867478 bound raw file microfluidics_dataset/raw_data/initial_formulations_raw.csv | `e2e00376eb840a05db97369925c037e93ae7132d07b591f2715c5649d311deec` | not_redistributed_third_party_raw | `work/data/zenodo_17867478/microfluidics_dataset/raw_data/initial_formulations_raw.csv` |
| Zenodo 17867478 bound raw file microfluidics_dataset/raw_data/initial_formulations_raw.xlsx | `a82740eb7f9d905698c09e5b3d1dbdce17a6db2f113900e8e0fe12e26a9b9760` | not_redistributed_third_party_raw | `work/data/zenodo_17867478/microfluidics_dataset/raw_data/initial_formulations_raw.xlsx` |
| Zenodo 17867478 bound raw file microfluidics_dataset/raw_data/new_formulations_raw.csv | `40134c4c0e5b54a949b762cab8a6a4cf801f9a66bc018979ea0d6d5dce55068e` | not_redistributed_third_party_raw | `work/data/zenodo_17867478/microfluidics_dataset/raw_data/new_formulations_raw.csv` |
| Zenodo 17867478 bound raw file microfluidics_dataset/raw_data/new_formulations_raw.xlsx | `e21315eab48c1f78e091d9f62e3083e3831a6152b4cef66b4a6ed0d03dee7069` | not_redistributed_third_party_raw | `work/data/zenodo_17867478/microfluidics_dataset/raw_data/new_formulations_raw.xlsx` |
| Zenodo 17867478 bound raw file microfluidics_dataset/raw_data/wet_lab_validation_raw.csv | `ea57d4bbddb131662193ea86de018656079fcad4ed6ad180f9ee77d7c239f897` | not_redistributed_third_party_raw | `work/data/zenodo_17867478/microfluidics_dataset/raw_data/wet_lab_validation_raw.csv` |
| Zenodo 17867478 bound raw file microfluidics_dataset/raw_data/wet_lab_validation_raw.xlsx | `89c4431b926e805c2615af06a0102b7a689f9cd40e6d0751c1154a450e2fa759` | not_redistributed_third_party_raw | `work/data/zenodo_17867478/microfluidics_dataset/raw_data/wet_lab_validation_raw.xlsx` |

