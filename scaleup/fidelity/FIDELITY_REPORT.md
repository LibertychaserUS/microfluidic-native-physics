# FIDELITY_REPORT — Reproduction fidelity of the XGBoost baselines (protocol v1, two runs)

Protocol: `FIDELITY_PROTOCOL_v1.md` (sha256 `a273102cbde3e69137a0a24c2192c34b50637e2c21694668698b940b20eb701c`), frozen
before the first run. Runs: `WORK/fidelity_run1` and `WORK/fidelity_run2` (2026-10-22); the two
`artifact_hashes.json` are byte-identical; the result files are copied under `run1/` and `run2/`.
Scripts: `fidelity_driver.py` (ENV-A) and `fidelity_worker133.py` (ENV-B), sidecars in this folder.

中文摘要:在完全相同的 91 个划分/单元上,我们的 D1 构造函数(M0)与上游训练脚本原样的构造函数
(`XGBRegressor(tree_method="hist")` 全默认 + StandardScaler,在上游钉死的 xgboost 1.3.3 里运行,M4)
逐单元对照:随机划分 MAPE 差 0.021 pp(≤0.5 pp),五个单元的方向与排序完全一致 → 按预注册规则
**保真**。消融链表明 M0=M1=M2 逐位相同(显式参数=默认;StandardScaler 零影响),唯一差异来自
`base_score` 的默认行为(3.3.0 由数据估计截距,1.3.3 固定 0.5),M3(3.3.0 手设 0.5)与 M4 逐位相同。
上游部署模型等于在包含 64 行文献"泛化集"的全部 932 行上、以标准化输入训练的 1.3.3 默认模型(预测逐位
相同);D2 的 B1 在 Chagot 自己的 392/76 划分上得到 3.17%,与论文的 3.9% 相差 0.73 pp,按预注册规则
不在容差内(方向为更低),原因待核对论文原文。

## 1. Gates

| Gate | Pre-registered rule | Outcome |
|---|---|---|
| G1 self-reproduction | M0 reproduces the 90 canonical xgboost split rows, the external row and the E0 mean within 1e-9 | **passed**: max |Δ MAPE| over the 90 rows 0; external 11.258371856167436 both; E0 mean 5.267946109018772 both |
| G0 data equivalence (pre-registered first-match rule on the eight rounded features) | ≥ 99% rows matched both ways and max relative difference ≤ 1e-6 on mapped columns | rows matched 100% both ways for the 868-row and the 932-row upstream tables; the max-relative-difference condition **failed** because 57 (868) / 59 (932) duplicate feature tuples were paired to the first duplicate, which is a different replicate measurement — a defect of the rule, not of the data |
| G0 post-hoc diagnostic (not pre-registered; rows matched on the eight features + measured diameter + dispersed flow) | — | 868-row upstream table = OSF comprehensive table (all 868 rows, max relative difference 4e-10 in hydraulic diameter, 1.3e-9 in normalised size, 4.3e-10 in generation rate). 932-row table = the 868 rows + the 64 literature rows; on the 64 literature rows 26 differ from the OSF generalisability file beyond 1e-6 in hydraulic diameter and normalised size (6 beyond 1e-3; maximum relative difference 0.153 in hydraulic diameter, 0.132 in normalised size), and the upstream file carries generation rates that the OSF file leaves blank |

## 2. Unit table (mean MAPE in %, identical splits; M0 = ours; M4 = upstream constructor in xgboost 1.3.3)

| Unit | splits | M0 | M1 | M2 | M3 | M4 | M5 |
|---|---|---|---|---|---|---|---|
| U1 random 80/20 | 15 | 5.268 | 5.268 | 5.268 | 5.247 | 5.247 | 5.247 |
| U2 exact-input group | 15 | 5.742 | 5.742 | 5.742 | 5.699 | 5.699 | 5.699 |
| U3 geometry group | 30 | 11.689 | 11.689 | 11.689 | 11.661 | 11.661 | 11.661 |
| U4 geometry + fluid group | 30 | 10.176 | 10.176 | 10.176 | 10.287 | 10.287 | 10.287 |
| U5 external 64 rows | 1 | 11.258 | 11.258 | 11.258 | 10.814 | 10.814 | 10.814 |

M1 = 3.3.0 defaults; M2 = M1 + StandardScaler; M3 = M2 + `base_score=0.5`; M5 = M4 without the
scaler. Generation-rate MAPE on U1 (volume conservation): M0 16.697 (π) / 16.691 (3.143); M4 16.547 /
16.540.

## 3. Decisions (pre-registered)

- D1 random split: |5.247 − 5.268| = 0.021 pp ≤ 0.5 pp → faithful.
- D2 direction and ranking: signs of (unit − random) all positive for both; rank order identical
  (random < exact-input < geometry+fluid < external < geometry) → faithful.
- **Overall: faithful.** ENV-B executed (xgboost 1.3.3 under Rosetta 2); decision label "xgboost 1.3.3 (ENV-B)".
- Attribution chain (prediction-level max |Δ| in normalised diameter; unit-mean MAPE shift in pp):
  M0→M1 0 (constructor identity); M1→M2 0 (input scaling has no effect on the trees); M2→M3 0.271
  max |Δ|, shifts U1 −0.021, U2 −0.043, U3 −0.028, U4 +0.111, U5 −0.444 (`base_score` default);
  M3→M4 2.2e-16 (library version has no effect beyond `base_score`); M4→M5 0 (scaling inside 1.3.3).
- D3 (D2, B1 on the source paper's 392/76 split): MAPE 3.170%, RMSE 0.00772 mm, R² 0.927, identical
  for all five seeds; published random-split figure 3.9%; |Δ| = 0.730 pp > 0.5 pp → **not within
  tolerance** (our figure is lower). The published configuration was transcribed from the paper's
  Methods (`BLIND_VALIDATION_PROTOCOL.md`); the paper is not open access to this session, so the
  input set, hyper-parameters and the definition behind the 3.9% could not be re-read. Open item.
- D4 (deployed model `20231001_DAFD3_XGB.json`): the model loads identically in both environments
  (max relative difference 2e-16). Under C-std (StandardScaler fitted on the 932-row table, as
  `ModelHelper3` does) its in-sample MAPE is 1.150% (C-raw: 40.2%), so the shipped model was trained
  on standardised inputs. The pre-registered training-set rule returns "inconclusive" (65.0% of rows
  below 1% APE; 63.0% of the 868, 92.2% of the 64). Outside that rule: M4 fitted on all 932 rows
  reproduces the deployed predictions bit for bit (max |Δ| 2.2e-16, Spearman 1.000), so the deployed
  model is the 1.3.3 default-parameter fit on all 932 rows including the 64 literature rows; for it
  those rows are in-sample (MAPE 0.313%) and are not comparable with our external evaluation
  (E0 full-training external MAPE 11.258%). M0 fitted on the same 932 rows: Spearman of predictions
  0.99975, mean |Δ| 0.0032, in-sample MAPE 1.164%.

## 4. Deviations

See `FIDELITY_DEVIATION_REGISTER.md` (F-D1 to F-D7). F-D6: the driver's task files embedded the run
directory path, which made the two hash manifests differ although every computed file was
byte-identical; the paths were made relative, the scripts re-frozen, and both runs repeated
(`fidelity_run*_attempt0` kept in the workspace, results identical). F-D7: the pre-registered G0 rule
pairs duplicate input tuples to their first occurrence; the post-hoc diagnostic above is the honest
reading.

## 5. What this does and does not establish

It establishes that every D1 XGBoost number in the manuscript is what the upstream constructor
produces on the same rows up to the `base_score` default of the library version (largest effect
−0.44 pp on the 64 literature rows), with identical direction and ranking of the shift units; and
that the upstream deployed model was trained on the literature rows that the paper reports as a
generalisability test. It does not establish the provenance of the D2 figure of 3.9%; that requires
the source paper's Methods.
