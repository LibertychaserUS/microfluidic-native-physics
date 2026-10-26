# Hash registry

SHA-256 registry for the preprint 'Reliability Audits for Machine-Learned Droplet-Microfluidic Design Surrogates'. Section 'appendix_d' lists the 46 digests of the manuscript's Appendix D (v0.5.4) in order; 'supplementary' lists further frozen digests recorded in the artifact manifests. status: released = the file in release_path hashes to sha256; released_superset / released_redacted = the released file differs from the frozen bytes for the stated reason (released_sha256 gives the digest of the released file); not_redistributed_* = the bytes are not in this repository (third-party raw data, sealed labels, re-cut rows or a model binary); release_path is then where a local copy is expected after following work/data/README.md. Verify with scripts/verify_hash_registry.py.

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
| Fidelity protocol (FIDELITY_PROTOCOL_v1.md) | `a273102cbde3e69137a0a24c2192c34b50637e2c21694668698b940b20eb701c` | released | `scaleup/fidelity/FIDELITY_PROTOCOL_v1.md` |
| Deployed upstream model (20231001_DAFD3_XGB.json) | `91140b28dcf309892cbef105d34132ab5e0285473faaa9a75094b67ef8d93c62` | not_redistributed_third_party_code | `work/data/upstream/dafd-website/app/mod_dafd/models/forward_models/saved/20231001_DAFD3_XGB.json` |
| Closed-loop protocol E0--E2 (E0_E1_E2_CLOSED_LOOP_PROTOCOL_v1.md, revision 1.1) | `022aa19d3acc3bf9afcfac99ff53bf244ed54254adcb759cf7010dd120d6fb5f` | released | `scaleup/validation_plan/E0_E1_E2_CLOSED_LOOP_PROTOCOL_v1.md` |
| Closed-loop runner (closed_loop.py) | `645b73b2a0afa82154ee54b4018e33973427ea1640358878395afebc5ee13776` | released | `scaleup/w7/closed_loop.py` |
| Closed-loop analyzer (closed_loop_analyze.py) | `4715eb029191c3266517ca3ef78c03cbd29444234b12504a61e7c30fe3b28c7a` | released | `scaleup/w7/closed_loop_analyze.py` |
| E1b protocol (E1B_DIFFICULTY_PROTOCOL_v1.md) | `7fa2a69c8c59901fc00669ac544d63d09e014afe94e49cffc65b335eb8826c8e` | released | `scaleup/validation_plan/E1B_DIFFICULTY_PROTOCOL_v1.md` |
| E1b runner wrapper | `6f65d59941fac6e39312dd281914bf8f6174fc9e08b37c95ad23ba7606f36661` | released | `scaleup/w7/closed_loop_e1b.py` |
| E1b analyzer | `c8e7dba8213d24e3379ec3da47296bce7a153c62a555a260aaebcb26b97626c6` | released | `scaleup/w7/closed_loop_e1b_analyze.py` |
| E3 protocol (E3_DECISION_POLICY_PROTOCOL_v1.md, revision 1.1) | `71166fca292884e45e7585c03e2323fd8b2936455b53934a89b8236d06c1e310` | released | `scaleup/validation_plan/E3_DECISION_POLICY_PROTOCOL_v1.md` |
| E3 runner wrapper | `ddf25617321bff59fbd2fec4f8307a842a9e08f93f39cbe69635b677bd2c40b0` | released | `scaleup/w7/closed_loop_e3.py` |
| E3 analyzer | `5a091d1641873917df2ea094b6f39cd9c206f7e60e0807389a162e8d26fff76c` | released | `scaleup/w7/closed_loop_e3_analyze.py` |
| E4 protocol (E4_MODEL_FAMILY_PROTOCOL_v1.md) | `0b0b68d4e1119bc785a631b20b3d2b1c3af9709c3a4d402138eb5839597c59ce` | released | `scaleup/validation_plan/E4_MODEL_FAMILY_PROTOCOL_v1.md` |
| E4 runner wrapper | `6bb3772a1290d4c7f8d3e1e3cad2a05df1fffe8b5f4367ccdddb3633e4b23ab4` | released | `scaleup/w7/closed_loop_e4.py` |
| E4 analyzer | `27cdb8c7bee460bd4f39d4680b478beb01f9284d539da36d8a1356c97e9eaa34` | released | `scaleup/w7/closed_loop_e4_analyze.py` |
| E5 protocol (E5_BUDGET_TRANSFER_PROTOCOL_v1.md) | `52fd522ed0d9dbac962db91dfe732df75ea5c630f22e52d4d88f13ce9a34e1bc` | released | `scaleup/validation_plan/E5_BUDGET_TRANSFER_PROTOCOL_v1.md` |
| E5 analyzer | `a724a9b6d872e14f4d3f4fca3fc99f11a0a7a1eb2ad2f69f52b976929af2e494` | released | `scaleup/w7/closed_loop_e5.py` |
| E6 protocol (E6_NEGATIVE_CONTROLS_PROTOCOL_v1.md, revision 1.1) | `d28896aea17d28d427857c8db862209cffb50616a0446d4a9e9ae910092b8a06` | released | `scaleup/validation_plan/E6_NEGATIVE_CONTROLS_PROTOCOL_v1.md` |
| E6 runner wrapper | `8d76252092eab9b903c836dac05149f8d450871d365eca9ae4f8470b7932c6fc` | released | `scaleup/w7/closed_loop_e6.py` |
| E6 analyzer | `edc6f1c50c2bff397538eb214e32b62e8ab09221155397358c76fb2e912aa056` | released | `scaleup/w7/closed_loop_e6_analyze.py` |
| E7 protocol (E7_HARD_TASK_PROTOCOL_v1.md, revision 1.1) | `130a150965ef742753e63692da5474d6fe9fa925987a16d9914fa13b44f71425` | released | `scaleup/validation_plan/E7_HARD_TASK_PROTOCOL_v1.md` |
| E7 runner wrapper | `09ae934d9878a8230c3ce7ebfdbe00f09a2986215d4fe758ecf5758e7c654eb7` | released | `scaleup/w7/closed_loop_e7.py` |
| E7 analyzer | `f64edbeeae25fec8e29c82f65cddf26e0beb44f543f38d672b5302e43f39a48a` | released | `scaleup/w7/closed_loop_e7_analyze.py` |

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
| FIDELITY_DRIVER_SCRIPT.sha256 -> fidelity_driver.py | `522b9d1ae048192c10e45f94275cedf02c20da80183b4eeb982021a7eb9a3852` | released | `scaleup/fidelity/fidelity_driver.py` |
| FIDELITY_WORKER_SCRIPT.sha256 -> fidelity_worker133.py | `dd4e1550c2ee19379eb69bcb9b7cd6c529b89090e81552765e7e1646c3a8c9ec` | released | `scaleup/fidelity/fidelity_worker133.py` |
| G0_POSTHOC.sha256 -> fidelity_g0_posthoc.py | `41cb9cfe698d6a1310871c36be74cf88da2b1668ce8b70aaa750a387860c6aeb` | released | `scaleup/fidelity/fidelity_g0_posthoc.py` |
| G0_POSTHOC.sha256 -> g0_rowwise_diagnostic.json | `3ae6fffccdb0063570ebc9f89b43ba58f29d3be6b7defbade37e3939f2013f3d` | released | `scaleup/fidelity/posthoc/g0_rowwise_diagnostic.json` |
| H_J_POSTHOC.sha256 -> h_fewshot_rule_posthoc.py | `641fc4fc1cf0b0098f5e50267ed4aff27d47efe907f535d98881d79d043e1516` | released | `scaleup/w6/h_fewshot_rule_posthoc.py` |
| H_J_POSTHOC.sha256 -> descriptive.json | `b07893f1c1ebf7d06870ec4daf9871e2151f0c0e830096a3217254302c3a9d4f` | released | `scaleup/h_fewshot_rule/posthoc/descriptive.json` |
| H_BINDING_A1.json.sha256 -> H_BINDING_A1.json | `fd8418fa0e173782ca3daabba96531f61ad9362b7d92f490fd99211fe8046358` | released | `scaleup/validation_plan/H_BINDING_A1.json` |
| H_BINDING_A1_v2.json.sha256 -> H_BINDING_A1_v2.json | `985ce9de0b50ef683aebaab86e4107e7c028934a6cd484a6b0294df92c21cf29` | released | `scaleup/validation_plan/H_BINDING_A1_v2.json` |
| H_BINDING_R2A1.json.sha256 -> H_BINDING_R2A1.json | `88bfb1ca94fa366aa6b01095bb9b5fd3d886f57f6410c4873a170c35d28e9294` | released | `scaleup/validation_plan/H_BINDING_R2A1.json` |
| H_BINDING_R2A2.json.sha256 -> H_BINDING_R2A2.json | `8c6be2b33ebde4a643ddae7e5b51fac410b3d393b56d25ccef1e404b40945eae` | released | `scaleup/validation_plan/H_BINDING_R2A2.json` |
| H_COMBINE_SCRIPT.sha256 -> h_combine.py | `88826ebb99e4ce5c34060c2f5ae637344722516a58fe2ac8d105afaaf1b7f583` | released | `scaleup/w6/h_combine.py` |
| H_CUSTODIAN2_SCRIPT.sha256 -> h_custodian2.py | `e6b24dd36b9bb116160ab2d37fdd13276d6fe0eefeeffc812aca3faac1f0bf21` | released | `scaleup/w6/h_custodian2.py` |
| H_CUSTODIAN_SCRIPT.sha256 -> h_custodian.py | `8b16a3c70928a2b5b0edf7692d7b38f7f49b45a52ec8420ee92e99d8d247f557` | released | `scaleup/w6/h_custodian.py` |
| H_DOWNLOAD_SCRIPT.sha256 -> h_download.py | `274f5de48ea059452e1f524a4e4336ac4969a26737f87849faa4c8ce187c05fb` | released | `scaleup/w6/h_download.py` |
| H_F_PORTFOLIO_PROTOCOL_v1.md.sha256 -> H_F_PORTFOLIO_PROTOCOL_v1.md | `2d3c78e0480114dc9503f5175e04bea5b8f611040f7e524a076e416942ea3cd5` | released | `scaleup/validation_plan/H_F_PORTFOLIO_PROTOCOL_v1.md` |
| H_F_PORTFOLIO_PROTOCOL_v2.md.sha256 -> H_F_PORTFOLIO_PROTOCOL_v2.md | `b55669c08131aa88f3b3376f1f9a8e57b4035191ff6c63dc715f92038437319e` | released | `scaleup/validation_plan/H_F_PORTFOLIO_PROTOCOL_v2.md` |
| H_G_WARMSTART_PROTOCOL_v1.md.sha256 -> H_G_WARMSTART_PROTOCOL_v1.md | `a6836d9e5427af8129872ae0f293f66a9a07ffaec9893416252c24cbef152058` | released | `scaleup/validation_plan/H_G_WARMSTART_PROTOCOL_v1.md` |
| H_G_WARMSTART_PROTOCOL_v2.md.sha256 -> H_G_WARMSTART_PROTOCOL_v2.md | `ad9e442d2a98047ad8da362bd2b89212deace07d857e62be7ce89ca8023ab562` | released | `scaleup/validation_plan/H_G_WARMSTART_PROTOCOL_v2.md` |
| H_J_ANALYZER_SCRIPT.sha256 -> h_fewshot_rule_analyze.py | `7183a825983d635d0b09f45f9b9a4dc3a7409fa44e37a8e213fea029332ddfaf` | released | `scaleup/w6/h_fewshot_rule_analyze.py` |
| H_J_FEWSHOT_RULE_PROTOCOL_v1.sha256 -> H_J_FEWSHOT_RULE_PROTOCOL_v1.md | `8641c20538cdec2585e1b352b0350f0555b3d8b3a88d56faaa38fc67029f47c5` | released | `scaleup/validation_plan/H_J_FEWSHOT_RULE_PROTOCOL_v1.md` |
| H_J_RUNNER_SCRIPT.sha256 -> h_fewshot_rule.py | `519938f6238a2b1c5884d8c6e818c86bca4747d2d54fa04a0a3d1bcb63ff79fe` | released | `scaleup/w6/h_fewshot_rule.py` |
| H_PORTFOLIO2_SCRIPT.sha256 -> h_portfolio2.py | `97291c9085e7a80f13305920b746f5bb1fc0b81cffe2fde8a77fb90aba234e61` | released | `scaleup/w6/h_portfolio2.py` |
| H_PORTFOLIO_SCRIPT.sha256 -> h_portfolio.py | `66dea2602e498c0ffff8b1d582254fa72857512a422b431efecb29f25de35ec3` | released | `scaleup/w6/h_portfolio.py` |
| H_PROTOCOL_v1.md.sha256 -> H_PROTOCOL_v1.md | `c0663a84c6d228f945f352feeb02034378f3d3993a0b9365b3d817bd581ef821` | released | `scaleup/validation_plan/H_PROTOCOL_v1.md` |
| H_PROTOCOL_v2.md.sha256 -> H_PROTOCOL_v2.md | `2adbd18623a848d3179c89d0ee49cd9106fb573144d7fabb47db12f906b873b3` | released | `scaleup/validation_plan/H_PROTOCOL_v2.md` |
| H_REVEAL2_SCRIPT.sha256 -> h_reveal2.py | `ed4d78a2842fb16251ec4300d48587d618c8b0509007dea595bf4e0524005e9a` | released | `scaleup/w6/h_reveal2.py` |
| H_REVEAL_SCRIPT.sha256 -> h_reveal.py | `5ac1f2774f625a7d7b30f0568e50ad4c47e37f2b44d2258932ca79609e1a7956` | released | `scaleup/w6/h_reveal.py` |
| H_RUN2_SCRIPT.sha256 -> h_run2.py | `15b1215993c97704d0337924ca45fbc54d7fe2a4b4bb005d1c81630ddb06abce` | released | `scaleup/w6/h_run2.py` |
| H_RUN_SCRIPT.sha256 -> h_run.py | `270c0da20065c1d39339d8fd028779a84c21a308c367895da4cb0e4d5b29ce7d` | released | `scaleup/w6/h_run.py` |
| POSTHOC_ROUND3_RECALIBRATION_PROTOCOL_v1.md.sha256 -> POSTHOC_ROUND3_RECALIBRATION_PROTOCOL_v1.md | `4eedba5b756630f00e72e16977bf21e39ed53c244ff9c77fe4b9de65fac60f77` | released | `scaleup/validation_plan/POSTHOC_ROUND3_RECALIBRATION_PROTOCOL_v1.md` |
| POSTHOC_ROUND3_SCRIPT.sha256 -> posthoc_round3.py | `954ce3f5fd1f4a92269b3fc94915fd2a606e77e0eb507825cd079375822a0771` | released | `scaleup/w3/posthoc_round3.py` |
| POSTHOC_ROUND4_WARMSTART2_SCRIPT.sha256 -> posthoc_round4_warmstart2.py | `a9fe1308be11ec8127f9e64f2625f0042efff9f7cf7802ba449e931f376d37a0` | released | `scaleup/w3/posthoc_round4_warmstart2.py` |
| POSTHOC_ROUND4_WARMSTART_SCRIPT.sha256 -> posthoc_round4_warmstart.py | `8565105334aeee81fc9277a2e0ae7578a9e2b90c270fc1a22ad85ad5c49f8a50` | released | `scaleup/w3/posthoc_round4_warmstart.py` |
| R10 A1 sealed prediction bundle (per-row, inputs + predictions) | `45d641d27caa225cf15f5ff1527baae918698ec27e5f4fe3422001cf968a1fbd` | not_redistributed_derived_rows | `work/microfluidic_reliability/h_A1_run1/sealed_prediction_bundle.csv` |
| R10 R2A1 sealed prediction bundle (per-row, inputs + predictions) | `1f357ad566b780361c407a0069b65d0089d659a9dbf59f8f95fd660e80cebfcb` | not_redistributed_derived_rows | `work/microfluidic_reliability/h_R2A1_run1/sealed_prediction_bundle.csv` |
| R10 R2A2 sealed prediction bundle (per-row, inputs + predictions) | `bed2b12a3c9ec0bf447a122c13e56aa01ac80623b75e3d92787c8beb5cf753c1` | not_redistributed_derived_rows | `work/microfluidic_reliability/h_R2A2_run1/sealed_prediction_bundle.csv` |
| R10 A1 source file (PMC11834948_supplementaryFiles.zip) | `ea9d5003bdcc665f666a60679656ddb3e29af3ec0adecd3259c8363c8dd52063` | not_redistributed_third_party_raw | `work/data/h_A1/PMC11834948_supplementaryFiles.zip` |
| R10 A1 development table (development_labeled.csv, 661 rows) | `8b439565f7bac2e9e06f030a1f6167e4de8b56488eccb3a1ea47603cf11d949e` | not_redistributed_derived_rows | `work/microfluidic_reliability/h_A1_artifacts/development_labeled.csv` |
| R10 A1 sealed features (inputs only) (sealed_features.csv, 240 rows) | `feb9c268f79795b7ec35949165a4459bcd39fc902c1bd217390a2cd949482068` | not_redistributed_derived_rows | `work/microfluidic_reliability/h_A1_artifacts/sealed_features.csv` |
| R10 A1 sealed labels (sealed_labels.csv, 240 rows) | `0dfdd77fec3acd28723a8b32d7c8847e6909221b23f68eb7b02a45583172597d` | not_redistributed_sealed_labels | `work/microfluidic_reliability/h_A1_artifacts/sealed_labels.csv` |
| R10 R2A1 source file (Data for FrameworkA.xlsx) | `991b450207c2d1398bf8d9c77784a9b8504341fe3e0b4d2b8f5dccf2482d9be6` | not_redistributed_third_party_raw | `work/data/h_R2A1/Data for FrameworkA.xlsx` |
| R10 R2A1 development table (development_labeled.csv, 557 rows) | `fca77a0de5f50581776eea91647e295f5e0bdc246dbb3824c4989bb368f909a9` | not_redistributed_derived_rows | `work/microfluidic_reliability/h_R2A1_artifacts/development_labeled.csv` |
| R10 R2A1 sealed features (inputs only) (sealed_features.csv, 273 rows) | `4616cd38319573225d017a9dcccd19dcd975459c4164533200493eb96774ff55` | not_redistributed_derived_rows | `work/microfluidic_reliability/h_R2A1_artifacts/sealed_features.csv` |
| R10 R2A1 sealed labels (sealed_labels.csv, 273 rows) | `ecafd931d4dce256ac489a43a48bd2e442e4c95c44eecf0655ff53127aa3b497` | not_redistributed_sealed_labels | `work/microfluidic_reliability/h_R2A1_artifacts/sealed_labels.csv` |
| R10 R2A2 source file (LNP_Atlas_DB_202509_v1.csv) | `624ba8424cc89855251ff20d8a3edf81f36e87a89b0f8c789c375860bbe83cb0` | not_redistributed_third_party_raw | `work/data/h_R2A2/LNP_Atlas_DB_202509_v1.csv` |
| R10 R2A2 development table (development_labeled.csv, 523 rows) | `d98d372b836094b282504ac5e1d25a1dbe0ef81e7f299cc3b9758ea43a637b2c` | not_redistributed_derived_rows | `work/microfluidic_reliability/h_R2A2_artifacts/development_labeled.csv` |
| R10 R2A2 sealed features (inputs only) (sealed_features.csv, 396 rows) | `ec232439f2aa4502c67fb1304d96bec3f4c88c96d8f864a6a9035e3db202bf63` | not_redistributed_derived_rows | `work/microfluidic_reliability/h_R2A2_artifacts/sealed_features.csv` |
| R10 R2A2 sealed labels (sealed_labels.csv, 396 rows) | `47d97e4a2aff243a1695b4c30e519927f9b350a682e26b1399370e91ad721ca7` | not_redistributed_sealed_labels | `work/microfluidic_reliability/h_R2A2_artifacts/sealed_labels.csv` |
| Header-only intake manifest h_A1_download_manifest.json as written by h_download.py | `2080e9fbdaec9bdaab089f721e10a337c24124f38b7138f29841b490ec489c5f` | released_redacted | `scaleup/h_intake/h_A1_download_manifest.json` |
| Header-only intake manifest h_A2_download_manifest.json as written by h_download.py | `070367bb17bd5ccdeccf74c59b587e393f546fc2c8b2740dee79c71f70bd368f` | released_redacted | `scaleup/h_intake/h_A2_download_manifest.json` |
| Header-only intake manifest h_B1_download_manifest.json as written by h_download.py | `81ba90db18a7d9127196ea5e265ca5ae7fa7481e5cacee85d50c764376a8b959` | released_redacted | `scaleup/h_intake/h_B1_download_manifest.json` |
| Header-only intake manifest h_B2_download_manifest.json as written by h_download.py | `be98cc31d6a8f48ff58711a4ac45d779fa2726d458f1818f40b158ea28847cb3` | released_redacted | `scaleup/h_intake/h_B2_download_manifest.json` |
| Header-only intake manifest h_R2A1_download_manifest.json as written by h_download.py | `54d89ffe2f49a4895f6a8e7fa2a998eb0c12e1725dbfa159487e35fd160982ea` | released_redacted | `scaleup/h_intake/h_R2A1_download_manifest.json` |
| Header-only intake manifest h_R2A2_download_manifest.json as written by h_download.py | `c20f8a840fa4b2975b385a50abf2a033df0261ec220f2582662ad2e946018a34` | released_redacted | `scaleup/h_intake/h_R2A2_download_manifest.json` |

