# H_J_REPORT — Cross-tier meta-analysis of when few-shot adaptation helps (protocol v1, two runs)

Protocol: `validation_plan/H_J_FEWSHOT_RULE_PROTOCOL_v1.md` (sha256 `8641c20538cdec2585e1b352b0350f0555b3d8b3a88d56faaa38fc67029f47c5`), frozen
before the first run; runner and analyzer sidecars `H_J_RUNNER_SCRIPT.sha256`, `H_J_ANALYZER_SCRIPT.sha256`.
Runs: `WORK/h_fewshot_rule_run1` and `_run2` (2026-10-22), `artifact_hashes.json` byte-identical; result
files copied under `run1/` and `run2/`. Deviations: `H_J_DEVIATION_REGISTER.md` (H-J-D1 to H-J-D4).

中文摘要:五张表共构造 106 个留一组档位(k=2 全部可用;k=5 有 81 个;k=10 有 67 个),每档位 100 次抽样、
四种方法。warm-start 的增益在 70.8% / 85.2% / 88.1% 的档位为正(k=2/5/10;中位增益 1.53 / 3.15 / 4.60 pp);
log 仿射 37.7% / 67.9% / 77.6%;从头训练 44.4% / 52.2%(k=5/10)。五个决策时预测变量里,偏置比、符号一致率、
残差量级比三者在所有 k 上与 warm-start 增益显著正相关(表级块 bootstrap、BH 校正、各表内方向一致;
偏置比 ρ = 0.80 / 0.72 / 0.67),5-NN 支撑重叠与区间宽度比不相关。阈值规则(四表拟合、第五表检验)只在
k=2 时同时优于"总是 warm-start"与"从不校准"(遗憾 1.75 vs 1.97 vs 11.30 pp;相对总是 warm-start 的差
−0.22 [−0.45, −0.03]);k=5、10 时"总是 warm-start"的遗憾已只有 2.60 / 1.63 pp,无规则能再降低。
按预注册判定(≥2 个 k)**规则不成立**;如实报告:决策时诊断量能预测增益大小,但按阈值在 warm-start 与
不校准之间选择只在两个标签时胜过总是 warm-start。

## 1. Tiers

| Code | n tiers | Shift type |
|---|---|---|
| D1-pool | 3 | laboratory |
| D1-geo | 26 | device |
| D1-type | 2 | emulsion type |
| D2-chem | 6 | chemistry |
| A1-fluid | 22 | chemistry |
| A1-device | 2 | device |
| R2A1-fluid | 5 | chemistry |
| R2A1-device | 2 | device |
| R2A2-lipid | 11 | composition |
| R2A2-paper | 25 | laboratory |
| R2A2-prep | 2 | preparation |

Total 106; eligible for k = 5: 81, for k = 10: 67 (n − k ≥ 10). 25,400 draw rows.

## 2. Gains (per-tier medians over 100 draws; `posthoc/descriptive.json`)

| k | tiers | warm-start gain > 0 | median warm gain (pp) | affine gain > 0 | scratch gain > 0 | median regret: never / always warm |
|---|---|---|---|---|---|---|
| 2 | 106 | 70.8% | 1.53 | 37.7% | — | 2.06 / 0.04 |
| 5 | 81 | 85.2% | 3.15 | 67.9% | 44.4% | 4.39 / 0.41 |
| 10 | 67 | 88.1% | 4.60 | 77.6% | 52.2% | 6.42 / 0.08 |

## 3. A1 — association of decision-time diagnostics with the warm-start gain (Spearman ρ across tiers)

| Diagnostic | k = 2 | k = 5 | k = 10 | BH-significant and direction-consistent |
|---|---|---|---|---|
| P1 bias ratio |mean r|/SD | 0.796 [0.613, 0.866] | 0.717 [0.604, 0.773] | 0.667 [0.601, 0.712] | all k |
| P2 sign consistency | 0.509 [0.244, 0.650] | 0.645 [0.492, 0.739] | 0.626 [0.543, 0.727] | all k |
| P3 magnitude ratio | 0.387 [0.349, 0.574] | 0.488 [0.386, 0.742] | 0.548 [0.429, 0.813] | all k |
| P4 5-NN support overlap | −0.071 [−0.360, 0.151] | −0.096 [−0.407, 0.117] | −0.131 [−0.368, 0.020] | none |
| P5 quantile width ratio | −0.026 [−0.175, 0.241] | −0.117 [−0.360, 0.251] | −0.083 [−0.322, 0.250] | none |

Intervals: table-level block bootstrap (B = 2000). The same three diagnostics also correlate with
the affine and scratch gains (P1: 0.605 / 0.713 / 0.704 with affine; 0.428 / 0.507 with scratch at k = 5, 10).

## 4. A2 — threshold rule, five-fold table rotation (mean held-out regret, percentage points)

| k | never calibrate | always affine | always scratch | always warm | bias-ratio rule (variant U) | rule − always warm [95% CI] | rule − never [95% CI] |
|---|---|---|---|---|---|---|---|
| 2 | 11.30 | 7.84 | — | 1.97 | 1.75 | −0.22 [−0.45, −0.03] | −9.55 [−14.72, −5.11] |
| 5 | 15.89 | 3.20 | 9.65 | 2.60 | 2.80 | +0.20 [+0.04, +0.39] | −13.10 [−19.66, −7.77] |
| 10 | 15.71 | 3.92 | 5.70 | 1.63 | 1.80 | +0.17 [−0.01, +0.39] | −13.91 [−20.09, −8.65] |

No other diagnostic's rule (variants U or S) reduced regret against both baselines at any k.

## 5. A3 — decision

k = 2: holds (P1). k = 5: does not hold. k = 10: does not hold. Overall (≥ 2 of 3 k with the same
diagnostic): **does not hold**. Wording used in the manuscript: a decision-time diagnostic computed
from the k measured rows predicted the size of the warm-start gain, but a threshold on it beat
always applying warm-start only with two labels.

## 6. Descriptive by shift type (median warm-start gain, pp; fraction of tiers positive)

| type | tiers (k=2/5/10) | k = 2 | k = 5 | k = 10 |
|---|---|---|---|---|
| chemistry | 33/33/29 | 1.97 (0.85) | 4.20 (0.91) | 4.32 (0.93) |
| device | 30/16/10 | 1.02 (0.67) | 1.65 (0.75) | 3.58 (0.90) |
| laboratory | 28/20/16 | 1.73 (0.71) | 5.76 (0.85) | 7.39 (0.81) |
| composition | 11/8/8 | −0.07 (0.45) | 2.26 (0.75) | 3.79 (0.75) |
| emulsion type | 2/2/2 | 17.94 (0.50) | 28.22 (1.00) | 32.59 (1.00) |
| preparation | 2/2/2 | 0.86 (0.50) | 3.23 (1.00) | 7.23 (1.00) |

Descriptive only; no per-type claim is made.
