# E4_MODEL_FAMILY_PROTOCOL_v1 — Source-model families inside the closed loop: does the random-split ranking predict the loop ranking?

Frozen on 2026-10-22 before any counted E4 run; sha256 in `E4_MODEL_FAMILY_PROTOCOL_v1.sha256`. Separate
from the E0/E1/E2, E1b and E3 protocols, whose scripts stay unchanged; E4 is run by a wrapper
(`w7/closed_loop_e4.py`) that imports the frozen E1 runner for tiers, targets, the loop and the adaptation
methods, and analysed by `w7/closed_loop_e4_analyze.py`; both hashed (`E4_RUNNER_SCRIPT.sha256`,
`E4_ANALYZER_SCRIPT.sha256`) before the first counted run. Every threshold and decision form precedes the
data. Positive and negative outcomes are reported alike. Nothing wet.

中文摘要:R10 的 H-E 已证明六个模型族在随机划分下的排名不能预测密封档位上的排名;E4 问的是同一件事在闭环
里是否成立:以六个模型族分别作源模型(XGBoost 为各表锁定的构造器,其余五族为 H 协议的通用构造器),在 107 个
档位上回放 P0/P1/P2 三条策略,比较各族的命中时间 T*,并用每个档位上的 Kendall τ 检验"表级随机划分误差排名"与
"回路 T* 排名"是否一致;另报"零样本档位误差排名"(需要标签,作上限参照)。判定:τ 的档位均值与表级块 bootstrap
区间,区间不含 0 且均值为正才算"随机划分排名能预测回路排名";各族相对 XGBoost 的 ΔT* 用 E1 的判定形式与 BH。

## 1. Families

Six source-model families, each fitted on the tier's source rows on the table's target scale (log where the
E1 runner uses log, raw otherwise) with seed 42:

| family | constructor | notes |
|---|---|---|
| xgboost | the table's locked constructor of E1 (`closed_loop.make_model`) | reference; identical to E1's source model |
| random_forest | `RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=1)` | H-protocol constructor |
| mlp | `StandardScaler` + `MLPRegressor((64, 64), relu, adam, lr 1e-3, max_iter 2000, early stopping 0.15, n_iter_no_change 30, random_state=42)` with a standardised target | H-protocol `mlp_log` pipeline; bounded by the log target where the table is log |
| qrf | `w3.uq_models.QuantileForest(42)`, point prediction | 400 trees, min leaf 5 |
| gp | `w3.uq_models.GaussianProcessModel(42, p)`, mean prediction | ARD RBF + white kernel, 2 restarts; convergence warnings recorded |
| deep_ensemble | `w3.uq_models.DeepEnsemble(42, standardise_target=True)`, mean prediction | 5 members |

Per tier and family, computed once: the source fit, its predictions on the source rows and on the tier rows,
the zero-shot MAPE on the tier rows, and the fit time. Adaptation inside the loop is the frozen warm-start
residual boosting of the E1 runner with the family's predictions as the base margin; the family itself is not
refitted inside the loop.

## 2. Loops

Policies P0 (no adaptation), P1 (two random rows, warm-start, greedy; E1 reference) and P2 (k₀ = 0,
warm-start, greedy; E1's best) for every family on the 107 tiers of E0: three targets per tier (as E1),
seeds 0–9, τ = 0.10, B = 20; drift δ = ±0.2 with seeds 0–2 for every family and policy (fragility). No
other policy; the acquisition, proposal and decision questions are E1's and E3's.

## 3. Random-split and zero-shot rankings

Table-level random-split error: for every table and family, 5-fold row-random cross-validation
(`KFold(5, shuffle=True, random_state=0)`) on the table's development rows, MAPE on the held-out rows in the
original scale, averaged over folds — the number a developer would report without any target label.
Zero-shot tier error: the family's MAPE on the tier rows before any measurement (needs the tier labels; a
label-informed upper reference, not a deployable predictor).

## 4. Decisions

Tier value of a family under a policy = mean T* over 3 targets × 10 seeds.

- (D1) Family effect in the loop: Δ_F = mean over tiers of [T*_F(P2) − T*_xgboost(P2)], table-level block
  bootstrap (B = 2000, seed 0); "better" / "worse" / "not distinguishable" as E1 §3.6 with
  Benjamini–Hochberg over the five non-reference families; fragile = sign change under δ = ±0.2. The same is
  reported for P1 and P0 (descriptive).
- (D2) Ranking prediction: per tier, Kendall τ_b between the ranking of the six families by table-level
  random-split MAPE and their ranking by loop T* under P2 (both ascending); τ̄ = mean over tiers with a
  table-level block-bootstrap interval. "The random-split ranking predicts the loop ranking" iff τ̄ > 0 and the
  interval excludes 0; "contradicts" iff τ̄ < 0 and the interval excludes 0; else "does not predict". The
  same statistic is reported for the zero-shot ranking vs the loop ranking (label-informed reference), for the
  random-split ranking vs the zero-shot ranking (the H-E analogue), and under P1 (secondary).
- Descriptive: mean T* and zero-shot MAPE per family, by shift type; the family with the lowest loop T* per
  tier and how often each family is best; GP convergence warnings and deep-ensemble iterations.

## 5. Outputs, integrity, compute

`WORK/e4_run1`, `WORK/e4_run2`: `loops.csv`, `families.csv`, `table_cv.csv`, `run_manifest.json`,
`environment.json`, `artifact_hashes.json` (byte-identical across the two runs); the analyzer adds
`results.json` and `family_table.csv`. Ledger prefix `e4.`; repository copies without `loops.csv` under
`scaleup/e_closed_loop/e4_run1/2/`. Compute ≈ 6 families × 107 tiers × 3 policies × 3 × 10 = 57.8 k main
loops + 23.1 k drift loops, plus 642 source fits and 180 cross-validation fits; the Gaussian process on the
118-column R2-A2 table dominates. If a counted run exceeds 6 h on 8 processes, the drift loops are dropped
first and recorded as not run. Smoke runs with reduced sizes are not counted.
