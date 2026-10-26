# FIDELITY_PROTOCOL_v1 — Reproduction fidelity of the XGBoost baselines

Frozen on 2026-10-22 before any fidelity run. The sha256 of this file is recorded in
`FIDELITY_PROTOCOL_v1.sha256` and in every run's `artifact_hashes.json`. Any change after the
first run is a new version with a new sidecar; the executed version is the one whose digest
appears in the run directories.

中文摘要:本协议回答"我们在 D1 上的 XGBoost(E0 构造函数,R1–R4、R7、R8 的 XGB 行)与 D2 上的
B1 是否忠实于上游实现",办法是(i)在完全相同的划分上,把我们的构造函数与上游的构造函数
(上游训练脚本原样:`XGBRegressor(tree_method="hist")` 全默认 + `StandardScaler`,在上游钉死的
xgboost 1.3.3 里运行)逐单元对照;(ii)直接加载上游部署的已训练模型文件,不重训,按预先写死的
判读规则与我们的模型对照;(iii)在 Chagot 给定的 392/76 划分上跑我们的 B1,对照论文报告的 3.9%。
容差与判定规则在第 7 节,先于任何数据写死;正负结果一律进报告、附录与台账。

## 0. Question and scope

Q1 (D1). Does the XGBoost constructor used for every D1 result in the manuscript
(`dafd_reproduction.build_model`, called by `study_core.build_regressor("xgboost")` and by
`scaleup/w3/models.py` for D1) reproduce what the upstream DAFD 3.0 code trains, to within the
pre-registered tolerance, on identical splits?

Q2 (D1, deployed model). Does the trained model file shipped in the upstream repository agree
with a model trained by our constructor on the same rows?

Q3 (D2). Does our B1 constructor (`native_physics_v2.make_xgb`, `blind_predict.build_locked_model`)
reproduce the published random-split XGBoost figure of the D2 source paper on the source paper's
own train/test tables?

Out of scope: any change to a published manuscript number. If a fidelity decision is negative, or
if any check reveals an implementation error on our side, the result is reported to the
coordinating session first; which results are affected is assessed afterwards, under a separate
decision.

## 1. Pinned upstream artifacts

Source: the paper's Code availability statement names `https://github.com/CIDARLAB/DAFD-website`
(citable at 10.5281/zenodo.10156784) and the OSF deposit 938rs for data.

| Item | Value |
|---|---|
| Repository clone | `CIDARLAB/dafd-website`, HEAD `4902f8ce79f73251282b75a3b12b9efd90420f81` (2024-08-05) |
| Paper snapshot | Zenodo 10156784, version v3.0.0 (2023-11-19) = git tag `v3.0.0` = commit `d1e62963309d8bcd31e2272aded54d71939a1943`; zip md5 `b4eb1ff1806cf777b25e0325953b5904` |
| Upstream xgboost | 1.3.3 (`environment.yml`, `req2.txt`: `xgboost=1.3.3`, `python=3.6.13`, `scikit-learn=0.21.2`, `numpy=1.16.4`, `pandas=0.24.2`) |
| Training script | `Ali_work/Code/XGBoost_gloval_stability.py` sha256 `d9f6aa72f755f4703f30b8f4f971ed8eec410a71e3712018bc02c12cf342b36d` |
| App model class | `app/mod_dafd/models/forward_models/DAFD3_models.py` sha256 `2a4a0f083e14b3cf46fc1549ce5e38e0f15e95a0dea13cb569833d0e550dd76a` (`XGBoost_DAFD3.build_model` = `xgb.XGBRegressor(tree_method="hist")`) |
| App forward model | `app/mod_dafd/core_logic/ForwardModel3.py` sha256 `3e787c928c26fe5c4a761da01759771379e6d35ead37f06da78826c1c2c8e087` |
| App data helper | `app/mod_dafd/helper_scripts/ModelHelper3.py` sha256 `8537fb006655c1d558e7392777c21abb3e07fbc5c99b14286ee6c03e833d07f2` (feature order `orifice_width, aspect_ratio, flow_rate_ratio, capillary_number, normalized_oil_inlet, normalized_water_inlet, expansion_ratio, viscosity_ratio`; target `normalized_hyd_size`; `hyd_d` restores micrometres) |
| Deployed model (primary) | `app/mod_dafd/models/forward_models/saved/20231001_DAFD3_XGB.json` sha256 `91140b28dcf309892cbef105d34132ab5e0285473faaa9a75094b67ef8d93c62` (xgboost JSON version [1,3,3]; 100 trees; all depth 6; 8 features; `base_score` 0.5; `reg:squarederror`; committed 2023-10-03, d1e6296 "Added new weights") |
| Upstream training tables | `app/mod_dafd/helper_scripts/experimental_data/20231001_all_data_dafd3.csv` (932 rows) sha256 `7735baefec0bc35dbd05c26eaf580110051880e540383eabf81d536c72037782`; `.../DAFD3_data.csv` (868 rows) sha256 `f6596ebc8c3152f5344043ccdfd2067999071b3944adc1922acc627d5d57f0fe` |
| Our data | OSF 938rs `Comprehensive_normalized.xlsx` (868 rows) and `Generalizability_data_normalized.xlsx` (64 rows), sha256 as in `SOURCE_MANIFEST.md` |
| D2 tables | Zenodo 7055018 `matTrainingDataSet.mat` sha256 `7fab0693a7065198468a5d654226f5e2164985e334ed105f91355bdd49fb1a17` (392 × 5 inputs + output), `matTestDataSet.mat` sha256 `55f202bd19433331c86ddc65507e834036d294f7f9b4db2275f21ad91f6729da` (76 × 5 + output) |

Neither upstream repository carries a LICENSE file (GitHub licence API returns 404 for both). No
upstream code or model file is copied into our repository; the fidelity scripts load the pinned
files from a local clone by path after re-verifying the sha256 above, and the repository stores
only our scripts, the digests, and the result files.

## 2. Environments

- ENV-A (ours): the manuscript environment, `~/.local/share/mf-w3-venv` — CPython 3.12.13 arm64,
  xgboost 3.3.0, scikit-learn 1.9.0, numpy 2.5.1, pandas 3.0.3.
- ENV-B (upstream pin): CPython 3.9 x86_64 under Rosetta 2 with xgboost **1.3.3** (the
  upstream-pinned version), numpy 1.26.x, scikit-learn 1.3.2, pandas 2.1.x, and the conda-forge
  `llvm-openmp` 15.0.5 osx-64 runtime loaded through `DYLD_LIBRARY_PATH`. Upstream's Python 3.6,
  scikit-learn 0.21.2 and numpy 1.16.4 cannot be installed on this machine; only the xgboost
  version is matched (deviation F-D3). Both environments are recorded in `environment.json`.
- If ENV-B cannot import xgboost at run time, every ENV-B item is recorded as "not executable" and
  the decisions in Section 7 fall back to the emulation variant M3 with the label
  "emulated 1.3.3 defaults"; this fallback is pre-declared here and would be stated in the report.

## 3. Gates (executed first; a failed gate stops the run)

G0 — data equivalence. The upstream CSVs are matched row-wise to the OSF tables by the tuple of the
eight features rounded to six significant digits (column map: `orifice_width`↔Orifice width (um),
`aspect_ratio`↔Normalized channel depth, `flow_rate_ratio`↔Flow rate ratio,
`capillary_number`↔Capillary number, `normalized_oil_inlet`↔Normalized continuous inlet,
`normalized_water_inlet`↔Normalized dispersed inlet, `expansion_ratio`↔Normalized outlet width,
`viscosity_ratio`↔viscosity ratio; `hyd_d`↔Hydraulic diameter; `droplet_size`↔Observed droplet
diameter (um); `normalized_hyd_size`↔Normalized droplet diameter; `q_in`↔Dispersed flow rate ul/h;
`generation_rate`↔Observed generation rate (Hz)). Reported: matched / unmatched counts in both
directions, duplicate feature tuples, and the maximum relative difference of the five mapped
non-feature columns on matched rows. Declared "same table" when at least 99% of rows match in both
directions and the maximum relative difference is at most 1e-6. The 932-row table is matched
against the union of the 868-row comprehensive table and the 64-row generalisability table. G0 is
informative: a mismatch is reported and the run continues on the OSF tables, which are the tables
behind every manuscript number.

G1 — self-reproduction. Before any comparison, variant M0 (Section 5) must reproduce, to within
1e-9 of `mape_percent`, (a) the 15 `row_random_reproduction` rows, the 15 `exact_input_dependency_group`
rows, the 30 `geometry_dependency_group` rows and the 30 `device_fluid_dependency_group` rows of
`expanded_artifacts_canonical_run2/split_metrics.csv` with `model == "xgboost"`; (b) the external
`row_micro` xgboost summary of `expanded_artifacts_canonical_run2/external_model_summary.csv`
(11.258371856167436); and (c) the E0 random-split mean of `artifacts/reproduction_metrics.json`
(5.267946109018772). If G1 fails the run stops with a report that the canonical environment is not
reproducing itself; no fidelity decision is issued.

## 4. Units and splits (fixed; identical for every variant)

The splits are generated by the canonical code (`expanded_study.split_protocols()` and
`expanded_study.make_split()` with `study_core.comprehensive_experiment_clusters`), so that every
variant is fitted and scored on exactly the rows the manuscript used:

| Unit | Splits | Train / test |
|---|---|---|
| U1 random | `row_random_reproduction`, seeds 0–14 | 80/20 row-random |
| U2 exact-input group | `exact_input_dependency_group`, seeds 0–14 | GroupShuffleSplit 80/20 |
| U3 geometry group | `geometry_dependency_group`, seeds 0–29 | GroupShuffleSplit 80/20 |
| U4 geometry + fluid group | `device_fluid_dependency_group`, seeds 0–29 | GroupShuffleSplit 80/20 |
| U5 external | one fit on all 868 rows, seed 2024 (as `evaluate_external_models`) | 64 literature rows |

Per-split metrics are the canonical ones (`study_core.prediction_frame`,
`study_core.summarize_prediction_rows`): predictions in normalised diameter are multiplied by the
hydraulic diameter and scored in micrometres; `mape_percent`, `mae_um`, `rmse_um`, `r2`. The unit
value is the mean over its splits. For U1 the generation-rate MAPE from volume conservation is
also computed twice, with π and with the upstream constant 3.143 (F-D4).

## 5. Model variants (same splits, same features, same scoring)

| Variant | Environment | Constructor | Input scaling |
|---|---|---|---|
| M0 (ours) | ENV-A | `dafd_reproduction.build_model(seed)` | none |
| M1 | ENV-A | `XGBRegressor(tree_method="hist", random_state=seed, n_jobs=1)`, all other arguments default | none |
| M2 | ENV-A | as M1 | `StandardScaler` fitted on the training rows (upstream script) |
| M3 | ENV-A | as M2 plus `base_score=0.5` (the 1.3.3 default) | as M2 |
| M4 (upstream constructor, upstream version) | ENV-B | `XGBRegressor(tree_method="hist", random_state=seed, n_jobs=1)`, all other arguments default in xgboost 1.3.3 | `StandardScaler` fitted on the training rows |
| M5 (diagnostic) | ENV-B | as M4 | none |

M4 is the upstream training script's model line executed verbatim except for two additions that
do not change the fitted function: `random_state=seed` (upstream passes no seed; F-D1) and
`n_jobs=1` (F-D2). Predictions of every variant are stored per row and per split so that
prediction-level differences can be computed, not only metric-level ones.

## 6. Deployed-model check (Q2; descriptive, rules fixed here)

1. Load `20231001_DAFD3_XGB.json` in ENV-A and, if available, in ENV-B; the two prediction vectors
   on the 932-row upstream table must agree to a relative 1e-5 (otherwise reported as a loader
   discrepancy).
2. Feeding conventions, both evaluated: C-raw — the eight features in `ModelHelper3.input_headers`
   order, unscaled; C-std — the same features standardised by a `StandardScaler` fitted on all
   932 rows of the upstream table, which is what `ModelHelper3.get_data` fits and
   `ForwardModel3.predict_xgb` applies at inference. (Upstream's `ModelHelper3.train_test_split`
   scales and then overwrites the scaled arrays with the unscaled ones, so the convention used
   when the shipped model was trained is not determinable from the code.) Rule: the convention
   with the lower in-sample MAPE on the 932 rows is called the training convention.
3. Training-set inference, under the training convention: rows with absolute percentage error
   below 1% are called likely training rows. If at least 95% of the 932 rows qualify, the model is
   treated as trained on all 932 rows and no out-of-sample unit exists; if between 70% and 90%
   qualify, the model is treated as trained on a random 80% split; otherwise inconclusive. The
   same fractions are reported separately for the 868 comprehensive rows and the 64
   generalisability rows.
4. Comparison with our constructor: M0 (ENV-A) and M4 (ENV-B) are fitted on the inferred training
   rows (all 932 rows if rule 3 says so; seeds 0–14 are reported as a range) and their predictions
   on all 932 rows are compared with the deployed model's: Spearman correlation of predictions,
   Spearman correlation of per-row absolute percentage errors, mean and maximum absolute
   difference in normalised diameter, and MAPE of each on the same rows. The 64 generalisability
   rows are reported separately with an explicit flag stating whether they are inferred to be
   inside the deployed model's training set; if they are, the comparison with E0's
   full-training external MAPE (11.258%) is marked in-sample and is not used for any decision.

## 7. Pre-registered decisions and tolerances

- D1 (random split): |mean MAPE(M4, U1) − mean MAPE(M0, U1)| ≤ 0.5 percentage points → "faithful
  on the random split".
- D2 (direction and ranking across shift units): with unit means m(·) over U1–U5, for M4 and M0,
  (a) sign(m(Uk) − m(U1)) is identical for k = 2, 3, 4, 5, and (b) the rank order of the five unit
  means is identical (Kendall τ = 1) → "faithful in direction and ranking".
- Overall D1 ∧ D2 → **faithful**; otherwise **not faithful**, and the source is attributed by the
  chain M0→M1→M2→M3→M4 on the same splits: prediction-level maximum absolute difference and
  unit-mean MAPE shift for each step (constructor identity; input scaling; `base_score` default;
  library version). The step with the largest absolute unit-mean shift is named the primary
  source. M0 vs M1 is expected to be exactly zero: both are the same explicit parameters.
- D3 (D2, B1): `make_xgb(seed=42)` with `b1_features` fitted on the 392 training rows (raw
  diameter in millimetres, as in the sealed V1 contract) and scored on the 76 test rows;
  |MAPE − 3.9| ≤ 0.5 percentage points → "faithful to the published random-split figure". R²,
  RMSE and the MAPE range over seeds {17, 42, 2024, 3407, 9001} are reported descriptively.
  The published 3.9% is the figure cited in the manuscript (Section 4, threshold provenance).
- D4 (deployed model): descriptive only; the agreement statistics of Section 6 are reported with
  the pre-declared inference labels and no threshold.
- If ENV-B is not executable, D1 and D2 use M3 in place of M4 and carry the label "emulated 1.3.3
  defaults".

Secondary descriptive outputs, no threshold: per-unit |Δ| for every variant; per-split paired
differences; the generation-rate MAPE under π and under 3.143.

## 8. Outputs and integrity

- Two independent runs in separate directories, `WORK/fidelity_run1` and `WORK/fidelity_run2`;
  each writes `splits_manifest.csv`, per-variant per-row predictions, `fidelity_results.json`,
  `environment.json` (ENV-A and ENV-B), `upstream_pins.json` (every pinned file re-hashed at run
  time), and `artifact_hashes.json`. The two `artifact_hashes.json` must be byte-identical.
- All JSON is written through the strict serialiser that rejects NaN and infinities.
- Repository copies (`microfluidic/scaleup/fidelity/`): this protocol and its sidecar, the driver
  and worker scripts with sidecars, the deviation register, `FIDELITY_REPORT.md`, and the result
  and hash files of both runs. No upstream file is copied.

## 9. Reporting

`FIDELITY_REPORT.md` (bilingual summary) and a new manuscript appendix "Reproduction fidelity of
the XGBoost baselines" in v0.4.3 (English and Chinese) with: the pinned upstream table, the unit
table (M0, M4 or M3, Δ), the decisions, D3, and the deployed-model summary. Every printed number
enters `numbers_ledger.json`. Positive and negative outcomes are reported alike. No published
number is changed by this protocol.

## 10. Deviation register

Deviations are numbered F-D1, F-D2, … in `FIDELITY_DEVIATION_REGISTER.md`. Pre-declared:
F-D1 fixed seeds where upstream had none; F-D2 `n_jobs=1`; F-D3 ENV-B matches only the xgboost
version; F-D4 upstream rate constant 3.143 reported alongside π; F-D5 column-name mapping between
the upstream CSV and the OSF tables (Section 3, G0).
