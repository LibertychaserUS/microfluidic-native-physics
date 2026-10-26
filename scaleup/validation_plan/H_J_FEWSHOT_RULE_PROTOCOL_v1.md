# H_J_FEWSHOT_RULE_PROTOCOL_v1 — Cross-tier meta-analysis of when few-shot adaptation helps

Frozen on 2026-10-22 before any run; sha256 in `H_J_FEWSHOT_RULE_PROTOCOL_v1.sha256`. Retrospective:
every label used here is public or was revealed by a consumed gate; nothing is sealed. No expected
outcome is written anywhere in this protocol. Positive and negative outcomes are reported alike.

中文摘要:在 D1、D2、A1、R2-A1、R2-A2 五张表上回顾性构造全部可构造的"留一组"档位(≥12 行),
每档位用该表既有的锁定 XGBoost 在其余行上训练源模型,目标域为留出组;k∈{2,5,10},每 k 随机抽 100 次,
方法 = 不校准 / log 仿射 / warm-start(λ=0.1)/ 从头训练(与 round 3/4 定义逐字相同);结果变量
G_k(method) = 跨抽样中位数的 (MAPE_uncal − MAPE_method);决策时可得的预测变量只用 k 个实测点、源模型
与目标输入;检验 = 跨档位 Spearman(表级块 bootstrap 区间,BH 校正)与阈值规则的五折轮换遗憾;判定规则
第 5 节先写死;不用 AUC/AUROC。

## 0. Question

Across every leave-one-group tier that the five tables allow, does a diagnostic computed at decision
time — from the k measured target rows, the frozen source model and the target inputs only — predict
the gain of few-shot adaptation over the uncalibrated source model, and does a threshold rule on such
a diagnostic reduce the regret relative to the post-hoc best method?

## 1. Tables, features, targets and locked source constructors (fixed)

| Table | Rows used | Features | Target and scale | Source constructor (existing locked configuration) |
|---|---|---|---|---|
| D1 | 868 comprehensive rows; 64 literature rows for the pool tiers | the eight OSF columns (`dafd_reproduction.FEATURES`) | normalised droplet diameter, raw scale | `dafd_reproduction.build_model(42)` (100 trees, depth 6, lr 0.3) |
| D2 | 468 rows (392 + 76 Zenodo tables, custodian extraction) | the four B1 inputs (`native_physics_v2.b1_features`) | diameter in mm, raw scale (sealed-V1 contract) | `native_physics_v2.make_xgb(42)` (100 trees, depth 3, lr 0.3) |
| A1 | 901 rows (661 development + 240 revealed sealed) | the seven native columns of `H_BINDING_A1_v2.json` | `L_um`, log scale (devdiag) | `h_run2.make_point("xgboost", 42)` (100 trees, depth 3, lr 0.3) |
| R2-A1 | 830 rows (557 + 273) | the nine native columns of the custodian manifest | `target`, raw scale (devdiag) | same |
| R2-A2 | 919 rows (523 + 396; main table + sealed S4 class) | the 118 native columns of the custodian manifest | `target` (nm), log scale (devdiag) | same |

Log scale means the source model is fitted on log y and predictions are exponentiated; the few-shot
methods always work in log space on positive predictions (round 3 convention).

## 2. Tier construction (rules; the runner reports the realised counts)

A tier is one held-out group with at least 12 rows; its source rows are all other rows of the same
table except where stated. Shift-type labels are fixed here.

| Code | Table | Held-out group | Source rows | Shift type |
|---|---|---|---|---|
| D1-pool | D1 | literature pool by `Ref` with ≥ 12 rows (Galogahi, Lashkaripour, Roumpea; Mazutis has 6 rows and is excluded) | the 868 comprehensive rows | laboratory |
| D1-geo | D1 | geometry dependency group (`study_core.group_ids` on `GEOMETRY_FEATURES`) with ≥ 12 rows | the other comprehensive rows | device |
| D1-type | D1 | SE rows / DE rows (`expanded_study.dafd_event_types`) | the other type | emulsion type |
| D2-chem | D2 | CMC group ∈ {0, 2.0, 3.5, 4.33, 11, 20} mM (0.60 mM has 2 rows and is excluded) | the other groups | chemistry |
| A1-fluid | A1 | `fluid_family` level with ≥ 12 rows | the other rows | chemistry |
| A1-device | A1 | `device` level | the other device | device |
| R2A1-fluid | R2-A1 | `s2_group` level with ≥ 12 rows | the other rows | chemistry |
| R2A1-device | R2-A1 | `s3_group` level | the other device | device |
| R2A2-lipid | R2-A2 | `s2_group` level with ≥ 12 rows | the other rows | composition |
| R2A2-paper | R2-A2 | `s3_group` level with ≥ 12 rows | the other rows | laboratory |
| R2A2-prep | R2-A2 | `s4_group` level | the other class | preparation |

The source model of a tier is fitted once on its source rows; its in-sample predictions on the
source rows are the warm-start base margins (round 4 convention). Expected order of magnitude:
about one hundred tiers.

## 3. Few-shot methods (definitions imported from the frozen scripts, not re-implemented)

k ∈ {2, 5, 10}. A tier is eligible for k iff n − k ≥ 10. For each eligible (tier, k): 100 random
draws, seed s = 0…99, rows `numpy.random.default_rng(s).choice(n, k, replace=False)` (round 3).
Methods, scored by MAPE on the n − k evaluation rows:

- uncal: the source prediction.
- affine: two-parameter affine map in log space fitted on the k rows (`posthoc_round3.calibrate`,
  key `affine`; when the k source predictions are constant the one-parameter shift `scale` is used
  and the substitution is counted).
- warm: warm-start residual boosting with λ = 0.1 (`posthoc_round4_warmstart.warm_start` with
  `WARM_PARAMS`: 100 trees, depth 2, lr 0.1, seed 42; source rows weighted 0.1; base margin log f).
- scratch (k ≥ 3): XGBoost 100 trees, depth 3, lr 0.3, seed 42 on the k rows with the table's
  target scale (`posthoc_round3.calibrate`, key `scratch`).

Per tier and k: G_k(m) = median over draws of (MAPE_uncal − MAPE_m), m ∈ {affine, warm, scratch}.
Per draw: regret(choice) = MAPE_choice − min over the available methods including uncal.

## 4. Decision-time predictors

With r_i = log y_i − log max(f_i, 10⁻⁶) on the k drawn rows (per draw):

- P1 bias ratio = |mean r| / SD(r) (SD with one degree of freedom; SD = 0 → capped at 10).
- P2 sign consistency = max(#{r > 0}, #{r < 0}) / k.
- P3 magnitude ratio = mean |r| / E_src, where E_src is the mean |log residual| of the source
  constructor under 5-fold GroupKFold on the source rows (tier-level constant).

Per tier, from the target inputs and the source rows only:

- P4 support overlap = fraction of the tier's rows whose 5-NN distance to the source rows (features
  standardised by source mean and SD; constant columns dropped) is at most the 95th percentile of the
  source rows' leave-one-out 5-NN distances.
- P5 width ratio = median over the tier's rows of (q95 − q05) divided by the median over the source
  OOF rows of (q95 − q05), where q05 and q95 are `GradientBoostingRegressor(loss="quantile",
  alpha=0.05/0.95, n_estimators=300, max_depth=3, random_state=42)` fitted on the source rows with
  log y (all tables, so widths are in log units); OOF uses the same 5-fold grouping as P3.
- P6 shift type: the categorical label of Section 2; used to stratify descriptive summaries only.

Grouping for the 5-fold OOF: D1 experiment clusters (`comprehensive_experiment_clusters`; for the
pool tiers the source is the comprehensive table), D2 CMC group (n_splits = min(5, number of
source groups)), A1 `fluid_family`, R2-A1 `s1_group`, R2-A2 `s1_group`.

Tier-level predictor values for P1–P3 are the medians over the 100 draws.

## 5. Pre-registered analyses and decision

A1 (association). For each k and method m (primary m = warm; affine and scratch reported) and each
numeric predictor P1–P5: ρ = Spearman(G_k(m), P) across eligible tiers. Confidence interval:
table-level block bootstrap (resample the five tables with replacement, keep every tier of a drawn
table, B = 2000, seed 0, percentile 2.5–97.5). Two-sided p-value p = 2·min(F(ρ* ≤ 0), F(ρ* ≥ 0));
Benjamini–Hochberg over the five predictors within each (k, m); "significant" = adjusted p < 0.05.
Secondary interval: stratified tier bootstrap (resample tiers within each table). Direction
consistency: within every table with ≥ 5 eligible tiers, the within-table Spearman has the same
sign as the pooled ρ.

A2 (threshold rule). For a predictor P and k, rule R(c): choose warm if s·P > s·c, where s is the
sign of the pooled ρ on the training tables; otherwise choose uncal (variant U) or scratch when
k ≥ 3 (variant S; uncal at k = 2). Five-fold table rotation: c is fitted on the four training tables
by minimising the mean over their tiers of the tier-median draw-level regret, over the grid of c
= the 5th, 10th, …, 95th percentiles of the training tiers' predictor values (decisions are taken
per draw with that draw's predictor value); the fitted rule is evaluated on the held-out table.
Pooling the five held-out evaluations gives one regret per tier. Baselines: always warm; never
calibrate (uncal); descriptive: always affine, always scratch, oracle (regret 0). D = regret_rule −
regret_baseline per tier; mean with primary interval by stratified tier bootstrap (B = 2000, seed 0)
and secondary interval by table-level block bootstrap. "Reduced" = mean D < 0 and the primary
interval excludes 0.

A3 (decision). At a given k the rule holds iff there is a predictor P such that (i) in A1 the
BH-adjusted block-bootstrap test for ρ(G_k(warm), P) is significant and P is direction-consistent,
and (ii) in A2 the rule built on P (variant U or S) has reduced regret against both baselines.
Overall: "holds" iff it holds at ≥ 2 of the 3 values of k with the same predictor; otherwise
"does not hold". Both outcomes are reported with the full tables.

Not used anywhere: AUC/AUROC; tier labels beyond the k drawn rows in any predictor; any
expectation about direction.

## 6. Wording constraints

If the rule holds: "in these five tables, a decision-time diagnostic computed from the k measured
rows predicted whether few-shot adaptation helped", followed by the rule (predictor, threshold,
k) and the regret numbers with intervals. If it does not: the same table with the statement that no
predictor met the pre-registered criteria. Summaries by shift type are descriptive only; no
per-type claim. No paradigm, probe, wall or artefact vocabulary.

## 7. Outputs and integrity

Simulation runner `scaleup/w6/h_fewshot_rule.py` and analysis `scaleup/w6/h_fewshot_rule_analyze.py`,
both hashed with sidecars before the first run. Two independent simulation runs in
`WORK/h_fewshot_rule_run1` and `_run2`; `artifact_hashes.json` must be byte-identical; the analysis is
deterministic from the run files. Files: `tiers.csv` (table, tier, type, n, n_source, k eligibility,
E_src, P4, P5), `draws.csv` (tier, k, seed, MAPE per method, P1–P3, regrets), `results.json`
(A1–A3 with intervals), `environment.json`, `artifact_hashes.json`. Repository copies in
`scaleup/h_fewshot_rule/`; ledger prefix `hj.`; deviations in `H_J_DEVIATION_REGISTER.md`.
Smoke runs with fewer draws or tiers are not counted and are not written into the run directories.
