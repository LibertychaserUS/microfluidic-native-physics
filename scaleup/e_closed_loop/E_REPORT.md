# E_REPORT — Offline closed-loop benchmark E0/E1/E2 (protocol v1 revision 1.1), E1b difficulty sensitivity, E3 decision policy, E4 model families, E5 budget transfer, E6 negative controls, E7 hard targets; two runs each

Protocol `validation_plan/E0_E1_E2_CLOSED_LOOP_PROTOCOL_v1.md` (sha256 `022aa19d…fb5f`), runner
`w7/closed_loop.py` (`645b73b2…`), analyzer `w7/closed_loop_analyze.py` (`4715eb02…`). Runs 2026-10-22:
`e0_run1/2` (hash manifests identical) and `e1e2_run1/2` (identical). 107 tiers (D1 31, D2 6, A1 24,
R2-A1 7, R2-A2 38, D3 1); 80,250 loops; 194,700 budget rows. Deviations E-D1–E-D13 in
`E_DEVIATION_REGISTER.md`. E-D1 (the V1 gap became a density label instead of an eligibility gate) is a
protocol revision triggered by the uncounted smoke run and registered before any counted run; Section 6
reports every E1 decision recomputed under the original gate (18 dense tiers) beside the all-tier decision.

中文摘要:在 107 个离线实验室档位上回放闭环。模拟器通过打乱标签对照(P1−P0 = −0.05 [−0.63, +0.38])。
点任务(命中目标尺寸 ±10%,预算 20 次):参考策略(先随机测 2 行、warm-start、贪心提议)平均 3.31 次命中
(中位 2.87,成功率 99.5%);天花板 1.23。**不先做校准测量、直接边测边 warm-start(k₀=0)最好:比参考少
0.50 次 [−0.87, −0.10],且在 ±20% 漂移下稳定**;多做校准测量(k₀=5)不优;三种采集规则(最远点、k-means、
不确定性宽度)都没有实质优势(k-means −0.075 次但漂移下不稳);仿射与从头训练比 warm-start 差 0.49/0.79 次;
"组合"提议比贪心差 0.39–1.11 次;H-J 的阈值规则预览比总是 warm-start 差 0.17 次。漂移 ±20% 下所有策略排序
不变,而"完美先验但不更新"的天花板模型反而被自适应回路击败(7.28/5.81 对 P2 3.24/3.25)。模型任务
(误差回到 1.25×开发参照):warm-start 在 20 行内达标的档位比例 76%(仿射 68%、从头 71%);按偏移类型,
化学 94%(90 分位 k* = 5)、实验室 90%(k* = 10.6)、配方 91%(k* = 5),器件只有 40%(中位 k* 超预算)。

## 1. E0 — simulator

| Item | Value |
|---|---|
| tiers | 107 (dense 18, sparse 89; density = median nearest-row relative gap ≤ 0.05) |
| noise floor (pooled relative SD of replicate rows) | D1 0.035, D2 0.095, D3 0.102, A1 0.374, R2-A2 0.201, R2-A1 0 (no replicates) |
| dev-ref (median OOF MAPE of the source constructor, by type) | chemistry 25.6, composition 35.9, device 5.4, emulsion 5.4, laboratory 35.1, preparation 36.0 |
| shuffled-label control, mean T*(P1) − T*(P0) | −0.05, table-block interval [−0.63, +0.38] → simulator accepted |

## 2. E1 — point task (mean T* over 3 targets × 10 seeds per tier; τ = 0.10; B = 20)

| Policy | k₀ | acquisition | adaptation | proposal | mean T* | median | success | Δ vs P1 [block 95%] | verdict (BH) | fragile ±0.2 |
|---|---|---|---|---|---|---|---|---|---|---|
| CEIL | — | — | refit on all tier labels | greedy | 1.23 | 1.00 | 1.000 | −2.07 [−2.76, −1.54] | better | yes (loses under drift) |
| P0 | 0 | — | none | greedy | 3.79 | 2.67 | 0.988 | +0.48 [−0.38, +1.47] | n.d. | no |
| P1 | 2 | random | warm | greedy | 3.31 | 2.87 | 0.995 | reference | — | — |
| P2 | 0 | — | warm | greedy | 2.80 | 2.00 | 1.000 | −0.50 [−0.87, −0.10] | better | no |
| P3 | 5 | random | warm | greedy | 3.75 | 3.43 | 0.998 | +0.44 [−0.07, +0.95] | n.d. | no |
| P4 | 2 | farthest | warm | greedy | 3.43 | 3.00 | 0.997 | +0.12 [−0.01, +0.29] | n.d. | no |
| P5 | 2 | k-means | warm | greedy | 3.23 | 2.67 | 1.000 | −0.075 [−0.137, −0.027] | better | yes |
| P6 | 2 | uncertainty | warm | greedy | 3.51 | 3.00 | 0.997 | +0.21 [−0.15, +0.53] | n.d. | no |
| P7 | 2 | random | affine | greedy | 3.79 | 3.20 | 0.990 | +0.49 [+0.33, +0.86] | worse | no |
| P8 | 5 | random | scratch | greedy | 4.10 | 3.80 | 0.996 | +0.79 [+0.13, +1.62] | worse | no |
| P9 | 2 | random | warm | portfolio | 3.70 | 2.97 | 0.991 | +0.39 [+0.11, +0.89] | worse | no |
| P10 | 5 | farthest | warm | portfolio | 4.41 | 4.00 | 0.981 | +1.11 [+0.93, +1.43] | worse | no |
| P11 | 2 | uncertainty | warm | portfolio | 3.81 | 3.00 | 1.000 | +0.51 [−0.04, +0.98] | n.d. | no |
| P12 | 2 | random | preview rule (bias ratio > 0.63 → warm) | greedy | 3.48 | 2.97 | 0.991 | +0.17 [+0.04, +0.31] | worse | no |

Pre-registered questions: (Q3) no acquisition rule beats random by more than 0.1 measurement
(k-means −0.075, fragile; farthest and uncertainty not distinguishable; with portfolio proposals
farthest is worse by +0.71); (Q2) spending measurements on calibration before searching costs
steps: k₀ = 0 is better (−0.50) and k₀ = 5 is not better (+0.44); (Q4) warm-start beats affine (+0.49)
and scratch (+0.79); the preview switching rule is worse than always warm-start (+0.17); the
portfolio proposal is worse than greedy in every pairing (+0.39, +0.66, +0.30 n.d.).

By shift type (mean T*; P0 / P1 / P2 / CEIL): chemistry 3.52 / 3.33 / 2.53 / 1.00; composition 5.76 /
3.95 / 3.39 / 1.09; device 2.60 / 2.93 / 2.10 / 1.03; laboratory 4.48 / 3.16 / 3.52 / 1.76; emulsion
type 2.83 / 3.85 / 1.83 / 1.00; preparation 6.00 / 6.55 / 5.33 / 1.50. By density: dense 4.06 / 2.45 /
2.31 / 1.06; sparse 3.73 / 3.48 / 2.90 / 1.27.

Drift. δ = ±0.2 (target in the drifted scale): mean T* P2 3.24 / 3.25, P1 3.47 / 3.40, P0 5.16 / 4.50,
CEIL 7.28 / 5.81 — the policy ranking among adaptive policies is unchanged, while the ceiling (perfect
prior, no update) falls behind every warm-start policy; δ = ±0.5 (descriptive): P1 3.66 / 3.53, P3 3.70
/ 3.64, P6 4.00 / 3.65, P0 7.51 / 6.21.

## 3. E2 — model task (k*(ε), ε = 1.25 × dev-ref; 100 random draws per k)

| Stratum | n | affine: med / p90 / reached | warm: med / p90 / reached | scratch: med / p90 / reached |
|---|---|---|---|---|
| all | 107 | 1 / 21 / 0.68 | 1 / 21 / 0.76 | 1 / 21 / 0.71 |
| chemistry | 33 | 1 / 17.8 / 0.88 | 1 / 5.0 / 0.94 | 1 / 15.0 / 0.94 |
| laboratory | 29 | 1 / 21 / 0.83 | 1 / 10.6 / 0.90 | 1 / 21 / 0.86 |
| composition | 11 | 1 / 21 / 0.82 | 1 / 5.0 / 0.91 | 1 / 21 / 0.82 |
| device | 30 | 21 / 21 / 0.30 | 21 / 21 / 0.40 | 21 / 21 / 0.30 |
| emulsion type | 2 | 12 / 19.2 / 0.50 | 18 / 20.4 / 0.50 | 12 / 19.2 / 0.50 |
| preparation | 2 | 11 / 19 / 0.50 | 11 / 19 / 0.50 | 11 / 19 / 0.50 |
| dense | 18 | 1 / 9.8 / 0.89 | 1 / 8.0 / 0.94 | 1 / 9.5 / 0.94 |
| sparse | 89 | 1 / 21 / 0.64 | 1 / 21 / 0.72 | 1 / 21 / 0.66 |

"21" = not reached within 20. The distribution is bimodal: many tiers are already within ε at k = 1
(their uncalibrated error is near or below 1.25 × dev-ref), and the device tiers mostly never reach
it within 20 rows — partly because the D1 geometry tiers have a small dev-ref (median 5.4%), so ε is
strict in absolute terms. The E0 noise-floor rule (no policy credited with an error below the floor) is not applied in this table; the floors of A1 and D2
(3 and 4 replicate groups) exceed most of their adapted errors, and under the strict reading of the rule the warm-start
chemistry share falls from 94% to 27% (all tiers 76% to 55%; E-D13, `e_floor_check/floor.json`).

## 4. What this adds to the framework (F-list numbering continued)

- F14 (Q2/Q3, point task): for reaching a target property within 10% on these tables, measuring
  before searching does not pay — the best loop starts proposing immediately and warm-starts on every
  measurement (2.80 measurements on average vs 3.31 with two calibration rows and 3.75 with five).
  Density sensitivity (Section 6): on the 18 dense tiers alone the gain is −0.14 [−0.67, +0.45], not
  distinguishable and fragile under drift — F14 rests on the sparse tiers. Difficulty (E1b, Section 8): robust —
  P2 is better in all three harder cells (−0.66 / −0.45 / −0.50, none fragile).
- F15 (Q3): label-free acquisition rules (farthest-point, k-means, source-interval width) do not
  shorten the search beyond 0.1 measurement on all tiers; the one nominal gain (k-means) is fragile under
  drift. Density sensitivity (Section 6): on the 18 dense tiers k-means acquisition gains 0.34 measurements
  [−0.51, −0.19], stable under drift (2.13 / 2.15 vs P1 2.85 / 2.64) — F15 holds for sparse tiers and is
  reversed for dense ones, where seeding at k-means centroids pays. Difficulty (E1b): robust on all tiers in all
  three harder cells (k-means −0.14 / −0.04 / −0.07, none better); on the dense subset k-means is better in every
  cell (−0.46 / −0.27 / −0.46, none fragile), so the dense-tier reversal is itself robust to difficulty.
- F16 (Q4): on the point task warm-start dominates affine, scratch and the threshold-switching
  preview; the retrieval portfolio proposal is worse than greedy — spreading proposals wastes
  measurements when a single-row target is reachable.
  Density sensitivity: affine stays worse on the dense tiers (+0.54 [+0.22, +1.11]); scratch (+0.30 [−0.08,
  +0.84]) and the portfolio (+0.18 [−0.04, +0.65]) are not distinguishable there. Difficulty (E1b): robust — affine
  and scratch are worse and the portfolio is worse in all three harder cells.
- F17 (robustness): under a ±20% oracle offset the adaptive loop's ranking is unchanged and a perfect
  but non-updating prior model falls behind it. Difficulty (E1b): robust — the P2 < P1 < P0 ordering holds at
  δ = ±0.2 in all three harder cells and CEIL falls to 5.1–9.6 under drift.
- F18 (Q2, model task): within 20 target rows, warm-start brings the surrogate back to 1.25 × its
  development reference in 94% of chemistry tiers (90th-percentile budget 5 rows), 90% of laboratory
  tiers (10.6) and 91% of composition tiers (5), but in only 40% of device tiers.
- F19 (Q4, decision rule; E3, Section 7): after two random measurements, no threshold rule on the bias
  ratio, the residual magnitude, the source interval width or the support overlap beats "always
  warm-start" under five-fold table rotation at an escalation cost of 5 or 10 measurements (rule − continue
  +0.01 [+0.00, +0.03]; the in-sample best rule is "always continue"); a perfect per-unit decision would
  save 0.54 measurements and 17.4% of units prefer another action (mostly "stop adapting"), so the
  information exists but is not in these k = 2 diagnostics. At λ = 0 "always escalate" wins trivially
  (−0.96 [−1.37, −0.48]) and is fragile under drift (+2.00 / +1.10 at δ = ∓0.2). The H-J preview rule (bias ratio > 0.63
  → warm-start, else none) is worse than always warm-start by 0.17 [0.04, 0.30].

## 5. Limits

Row-indexed oracle (the lab can realise exactly a recorded condition); targets are recorded labels; τ
= 10% makes the point task easy for most tiers (median T* ≤ 3), so differences are fractions of a
measurement; laboratory shift has 29 tiers only through R2-A2 source papers plus D3; device tiers are
dominated by D1 geometry groups with a strict ε; the preview rule's threshold is in-sample; no wet
experiment.

## 6. E-D1 transparency and the original-gate (dense-subset) sensitivity

Protocol v1 was frozen before the uncounted smoke run with the V1 median nearest-row gap ≤ τ/2 = 0.05 as an
eligibility gate. The smoke run (12 tiers, one per code) showed the gate would exclude 9 of 12 tiers, including
the D3 laboratory-B tier; because proposals are row indices, the gap measures how densely a tier samples its
input space, not how far the oracle's answer is from the request. Revision 1.1 made the gap a descriptive
label (dense/sparse) and kept every tier with ≥ 12 rows; the change was registered (E-D1) on the same day,
before any counted run, and no counted value existed when it was made. On the 107 counted tiers the original
gate would have kept 18 (D1 7, D2 3, R2-A1 3, R2-A2 5; no A1 or D3 tier). `E1B_DIFFICULTY_PROTOCOL_v1.md` §4
pre-specifies the recomputation of every E1 decision on those 18 tiers (table blocks = the 4 tables present);
output `e1_dense_sensitivity/results_dense.json`. Shuffled control on the subset: +0.03 [−0.16, +0.33], accepted.

| Policy | all 107 tiers: Δ vs P1 [block 95%], verdict | 18 dense tiers: Δ vs P1 [block 95%], verdict | fragile (dense) | reading |
|---|---|---|---|---|
| P0 | +0.48 [−0.38, +1.47] n.d. | +1.60 [−0.53, +4.34] n.d. | no | — |
| P2 (k₀ = 0) | −0.50 [−0.87, −0.10] better | −0.14 [−0.67, +0.45] n.d. | yes | F14 depends on the sparse tiers |
| P3 (k₀ = 5) | +0.44 [−0.07, +0.95] n.d. | +0.15 [−0.10, +0.40] n.d. | yes | — |
| P4 farthest | +0.12 [−0.01, +0.29] n.d. | +0.42 [+0.24, +0.77] worse | no | — |
| P5 k-means | −0.075 [−0.137, −0.027] better, fragile | −0.34 [−0.51, −0.19] better | no | F15 reversed on dense tiers |
| P6 uncertainty | +0.21 [−0.15, +0.53] n.d. | +0.25 [+0.00, +0.51] n.d. (BH) | yes | — |
| P7 affine | +0.49 [+0.33, +0.86] worse | +0.54 [+0.22, +1.11] worse | no | F16 affine part reproduced |
| P8 scratch | +0.79 [+0.13, +1.62] worse | +0.30 [−0.08, +0.84] n.d. | yes | F16 scratch part not reproduced |
| P9 portfolio | +0.39 [+0.11, +0.89] worse | +0.18 [−0.04, +0.65] n.d. | no | — |
| P10 | +1.11 [+0.93, +1.43] worse | +1.08 [+0.42, +2.33] worse | no | — |
| P11 | +0.51 [−0.04, +0.98] n.d. | +0.62 [+0.17, +1.19] worse | no | — |
| P12 preview | +0.17 [+0.04, +0.31] worse | +0.13 [+0.01, +0.34] worse | yes | — |
| CEIL | −2.07 [−2.76, −1.54] better | −1.40 [−2.44, −0.97] better | yes | F17 reproduced (CEIL 9.07 / 9.78 under ±0.2 vs P1 2.85 / 2.64) |

Mean T* on the dense tiers: P0 4.06, P1 2.45, P2 2.31, P5 2.11, CEIL 1.06. The all-tier analysis stays
primary (sparse tiers are legitimate target domains for a row-indexed oracle), and the two findings that do
not survive the subset are marked in Section 4. E1b (`E1B_DIFFICULTY_PROTOCOL_v1.md`) repeats this subset
analysis for every (τ, B) cell.

## 7. E3 — decision policy after two measurements (protocol `E3_DECISION_POLICY_PROTOCOL_v1.md`, revision 1.1; sha256 `71166fca…`)

Runner `w7/closed_loop_e3.py`, analyzer `w7/closed_loop_e3_analyze.py` (wrappers around the frozen E1 runner;
E-D4), runs `e3_run1/2` (hash manifests identical; 20,544 loops; the "continue" arm reproduces E1's P1 loop for
loop on all 3,210 main and 1,926 drift units). Four arms share the same two random phase-A rows per (tier,
seed): continue = warm-start (P1), none = stop adapting (arm A_none), switch = abandon the source model (arm A_switch:
unadapted proposal at k = 2, then XGBoost on the measured rows only), escalate = the ceiling model after
phase A (arm A_esc) at a cost λ ∈ {0, 5, 10} measurement-equivalents. Rule family: one diagnostic (d1 bias ratio,
d2 mean |log residual| / q_src, d3 mean interval width / q_src, d4 fraction of tier rows outside the source
support) with a threshold at a training decile and an action on each side, plus the four constant rules;
learned on four tables, applied to the fifth (D3 held out with D2). Decision form: rule holds at λ iff its
held-out cost is below both "always continue" and "never adapt" with block intervals excluding 0; E3 positive
iff holds at ≥ 2 of 3 λ.

| λ | fold rules (A1 / D1 / D2 / R2-A1 / R2-A2 held out) | rule cost | continue | none | switch | escalate + λ | oracle | rule − continue [block 95%] | rule − none | holds | drift (Δ vs continue at δ = −0.2 / +0.2) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | always escalate × 5 | 2.34 | 3.31 | 3.91 | 3.71 | 2.34 | 2.31 | −0.96 [−1.37, −0.48] | −1.57 [−2.48, −0.67] | yes, fragile | +2.00 / +1.10 |
| 5 | d1 > 9.41 → switch else continue / d1 > 9.54 → switch else continue / always continue × 3 | 3.32 | 3.31 | 3.91 | 3.71 | 7.34 | 2.77 | +0.01 [+0.00, +0.03] | −0.59 [−1.09, −0.24] | no | +0.08 / +0.06 |
| 10 | same as λ = 5 | 3.32 | 3.31 | 3.91 | 3.71 | 12.34 | 2.82 | +0.01 [+0.00, +0.03] | −0.59 [−1.10, −0.24] | no | +0.08 / +0.06 |

E3 is negative (holds at 1 of 3 λ, and only the trivial free-escalation level). Headroom of a perfect
per-unit decision over always-continue: 0.54 (λ = 5) and 0.49 (λ = 10) measurements; the oracle-best action
differs from "continue" in 17.4% / 16.7% of units. Tie-aware (E-D12, `e_tie_check/ties.json`, λ = 5): "stop adapting" and "switch"
attain the minimum in 11.7% and 11.2% of all units and are tied with each other in 7.3%; the share of units with an action
strictly better than continuing is largest for preparation (45.0%) and emulsion-type (31.7%) units; escalation attains the
minimum in 1.3% (device) to 5.0% (preparation) of units by type. (The earlier per-type "stop adapting" shares credited ties
to "none" before "switch".) The in-sample
best rule over all 107 tiers at λ = 5 or 10 is "always continue": the family cannot exploit the headroom even
without rotation. P12's preview rule (d1 > 0.63 → continue else none), evaluated as one member of the
family, costs +0.17 [+0.04, +0.30] relative to always continue — the E1 preview is confirmed as a loss.

Limits: the diagnostics are evaluated at k = 2 only; "switch" is the frozen scratch definition (no model
until the third row); escalation is modelled as the ceiling (an oracle-informed model), which is favourable
to escalation; rule family is single-split; λ in measurement-equivalents is a modelling choice.

## 8. E1b — difficulty sensitivity of the E1 decisions (protocol `E1B_DIFFICULTY_PROTOCOL_v1.md`, sha256 `7fa2a69c…`)

Wrapper `w7/closed_loop_e1b.py` over the frozen runner (overrides τ and B only; E2 and the ±0.5 runs off), analyzer
`w7/closed_loop_e1b_analyze.py`; runs `e1b_run1/2` (3 × 75,114 loops each, 21 min on 8 processes; the two runs' hash manifests are byte-identical
before and after analysis, at cell level and run level). Cells (τ, B) = (0.05, 20),
(0.10, 10), (0.05, 10); the fourth cell (0.10, 20) is E1. Shuffled control accepted in every cell (−0.25 [−1.30,
+0.31]; −0.04 [−0.52, +0.38]; −0.14 [−0.62, +0.22]).

| Policy | E1 (0.10, 20) | (0.05, 20) | (0.10, 10) | (0.05, 10) | grade (§3) |
|---|---|---|---|---|---|
| P1 mean T* (success) | 3.31 (0.995) | 4.75 (0.976) | 3.17 (0.964) | 4.31 (0.907) | — |
| CEIL mean T* | 1.23 | 1.55 | 1.21 | 1.45 | — |
| P2 k₀ = 0 | −0.50 [−0.87, −0.10] better | −0.66 [−1.07, −0.40] better | −0.45 [−0.76, −0.04] better | −0.50 [−0.87, −0.20] better | F14 robust to difficulty |
| P3 k₀ = 5 | +0.44 n.d. | +0.57 n.d. | +0.49 worse | +0.62 worse | — |
| P4 farthest | +0.12 n.d. | +0.05 worse | +0.09 n.d. | +0.07 n.d. | F15 robust |
| P5 k-means | −0.075 better, fragile | −0.14 [−0.40, +0.07] n.d. | −0.04 n.d., fragile | −0.07 n.d., fragile | F15 robust |
| P6 uncertainty | +0.21 n.d. | +0.12 n.d. | +0.24 n.d. | +0.21 n.d. | F15 robust |
| P7 affine | +0.49 worse | +0.42 [+0.29, +0.83] worse | +0.37 [+0.26, +0.59] worse | +0.35 [+0.26, +0.62] worse | F16 robust |
| P8 scratch | +0.79 worse | +1.11 [+0.24, +2.37] worse | +0.73 [+0.16, +1.38] worse | +0.94 [+0.25, +1.76] worse | F16 robust |
| P9 portfolio | +0.39 worse | +0.77 [+0.25, +1.85] worse | +0.26 [+0.07, +0.61] worse | +0.45 [+0.16, +1.06] worse | F16′ robust |
| P10 / P11 | +1.11 worse / +0.51 n.d. | +1.52 worse / +0.89 worse | +0.88 worse / +0.53 n.d. | +1.13 worse / +0.71 worse | — |
| P12 preview | +0.17 worse | +0.19 worse | +0.07 worse | +0.11 worse | — |
| ordering P2 < P1 < P0 at δ = 0, ±0.2 | yes | yes | yes | yes | F17 robust |
| CEIL under δ = −0.2 / +0.2 | 7.28 / 5.81 | 9.63 / 7.94 | 6.17 / 5.07 | 7.53 / 6.48 | F17 robust |

Dense subset (18 tiers) per cell: P2 −0.61 [−1.30, −0.08] better but fragile / −0.07 n.d. / −0.48 [−0.88, −0.10]
better but fragile; P5 k-means −0.46 [−0.97, −0.14] / −0.27 [−0.41, −0.16] / −0.46 [−0.83, −0.17], better and
non-fragile in every cell; P8 scratch not distinguishable in every dense cell (+0.84 / +0.32 / +0.67); P7 affine
worse in two of three. Reading: on all tiers every E1 decision survives a halved tolerance and a halved budget
("robust to difficulty", 3 of 3 cells, for F14, F15, F16, F16′ and F17); the dense-tier picture of Section 6 —
k-means seeding pays, k₀ = 0 is not reliably better, scratch is not reliably worse — is equally stable. The two
pictures are not in conflict: on sparsely sampled tiers the first two measurements carry little information about
where the target lies, so starting the warm-started search at once wins; on densely sampled tiers two
well-spread rows already localise the response and a k-means start is worth 0.3–0.5 measurements.

## 9. E5 — do measurement budgets and their planning rules transfer across tables? (protocol `E5_BUDGET_TRANSFER_PROTOCOL_v1.md`, sha256 `52fd522e…`; analysis of frozen E0/E2/E3 outputs, two identical runs `e5_run1/2`)

The quantity planned is the warm-start budget k*(ε) of E2 (ε = 1.25 × dev-ref; 21 = not reached): of the
107 tiers, 81 reach ε within 20 rows and 26 do not. A planning rule maps
pre-measurement features to a planned budget B̂ (80th percentile of k* over the training tiers sharing the
feature); the cost of a plan is the rows measured beyond need (B̂ ≥ k*) or the full budget of 20 (B̂ < k*: the
surrogate is unusable when the loop stops). Rules: R0 = always the full budget; R1 = by shift type; R2 = shift
type × density; R3 = shift type × label-free support diagnostic (d4 above/below the training median); R4 = shift
type × two-row bias ratio (median d1 over the E3 seeds). Five-fold table rotation (D3 with D2): the transferred
rule is learned on four tables and applied to the fifth; the table-specific rule is learned on the held-out
table's own tiers (in sample). Decision: a rule transfers when its transfer regret (transferred − table-specific
cost) has a block interval containing or below 0 and it beats R0 with an interval excluding 0.

| rule | transferred cost | table-specific cost | R0 cost | transfer regret [block 95%] | transferred − R0 [95%] | under-planned | verdict |
|---|---|---|---|---|---|---|---|
| R0 full budget | 18.07 | 18.07 | 18.07 | 0 | 0 | 0% | no advantage over the full budget |
| R1 shift type | 4.86 | 2.78 | 18.07 | 2.09 [-0.48, 7.62] | -13.21 [-15.71, -6.50] | 14% | transfers |
| R2 type × density | 4.77 | 2.31 | 18.07 | 2.46 [0.50, 7.53] | -13.31 [-15.71, -7.19] | 14% | does not transfer |
| R3 type × support diagnostic | 4.32 | 2.48 | 18.07 | 1.84 [0.08, 6.05] | -13.75 [-15.74, -8.18] | 17% | does not transfer |
| R4 type × bias ratio | 5.28 | 2.36 | 18.07 | 2.93 [0.44, 9.17] | -12.79 [-15.50, -5.55] | 19% | does not transfer |

Oracle cost (B̂ = k*) is 0.00 by construction. Reading: every lookup rule saves about 13 rows per tier against
planning the full budget, and the plainest one transfers — a budget planned from the shift type alone costs
4.86 rows per tier when learned on other tables against 2.78 when learned on the table itself,
a regret whose interval contains zero (2.09 [-0.48, 7.62]); every refinement by density, by the
label-free support diagnostic or by the two-row bias ratio fits the training tables and does not transfer
(regret intervals above zero), although each still beats the full budget. The verdicts are unchanged at
q = 0.50 and q = 0.90. Budgets by shift type (80th percentile of k*): chemistry 2.2, composition
1.0, laboratory 2.0, device 21.0 (not reached), with 60% of device tiers never
reaching ε. Not-reachable classification by R1: 77% of the 26 unreached tiers are predicted "not reachable",
with 28% false alarms among reached tiers. Shape transfer (descriptive): the gain curves
g_k = median MAPE(k)/uncalibrated MAPE agree across tables to within 0.2–0.4 for chemistry and 0.2–0.5 for device
at every k, and within 0.03–0.17 for laboratory where defined.

- F20 (Q2, transfer; F): the budget a new target domain needs is predictable from its shift type with a rule
  learned on other tables (transfer regret 2.09 rows, interval containing zero, against a saving of
  -13.21 rows over planning the full budget); refinements by density or by diagnostics do not transfer.

## 10. E4 — source-model families inside the loop (protocol `E4_MODEL_FAMILY_PROTOCOL_v1.md`, sha256 `0b0b68d4…`; runner `6bb3772a…`, analyzer `27cdb8c7…`)

Runs `e4_mac_run1/2` on the workstation (E-D7–E-D9): 107 tiers, 92448 loops, 642 family fits and
180 cross-validation fits per run; every output file is identical between the runs except the wall-clock
column `fit_seconds` (E-D10; `e4_identity/e4_identity.json`). The XGBoost arm reproduces E1's P0, P1 and P2 loops
exactly (3,210 of 3,210 main loops each). Mean T* (three targets × ten seeds per tier; τ = 0.10, B = 20), Δ vs
XGBoost with table-level block interval and Benjamini–Hochberg verdict, share of tiers where the family has the lowest or
tied-lowest P2 mean T* / the strictly lowest (tie-aware, E-D12; the frozen analyzer's argmin credited ties to XGBoost), and
median zero-shot tier MAPE:

| family | P0 (no adaptation) | P1 (two rows + warm-start) | P2 (warm-start at once) | Δ P2 vs XGBoost [95%] | lowest or tied / strictly lowest (P2) | zero-shot MAPE (median) |
|---|---|---|---|---|---|---|
| xgboost (reference) | 3.79 | 3.31 | 2.80 | — | 42.1% / 15.9% | 20.2 |
| random_forest | 3.30 (-0.49, better) | 3.26 (-0.04, n.d.) | 2.83 | 0.02 [-0.18, 0.17] n.d., fragile | 30.8% / 3.7% | 18.8 |
| mlp | 4.23 (0.45, n.d.) | 3.34 (0.03, n.d.) | 3.16 | 0.36 [0.01, 0.67] n.d., fragile | 30.8% / 13.1% | 23.5 |
| qrf | 3.10 (-0.69, better) | 3.24 (-0.07, n.d.) | 2.83 | 0.02 [-0.07, 0.11] n.d. | 36.4% / 8.4% | 18.4 |
| gp | 4.73 (0.95, n.d.) | 3.55 (0.24, worse) | 3.41 | 0.61 [0.32, 1.01] worse | 22.4% / 8.4% | 22.7 |
| deep_ensemble | 4.37 (0.58, n.d.) | 3.41 (0.10, n.d.) | 3.25 | 0.45 [-0.02, 1.38] n.d. | 24.3% / 6.5% | 22.9 |

Table-level random-split MAPE (5-fold, row-random; the developer's ranking):

| table | xgboost | random_forest | mlp | qrf | gp | deep_ensemble |
|---|---|---|---|---|---|---|
| A1 | 13.37 | 13.33 | 16.45 | 15.49 | 15.64 | 16.52 |
| D1 | 5.52 | 6.04 | 8.87 | 6.41 | 6.52 | 8.83 |
| D2 | 3.61 | 4.91 | 4.75 | 6.06 | 3.47 | 4.51 |
| D3 | 26.20 | 22.63 | 33.11 | 27.53 | 23.20 | 30.10 |
| R2A1 | 6.09 | 5.57 | 14.98 | 7.60 | 12.39 | 15.21 |
| R2A2 | 19.69 | 19.07 | 22.93 | 20.52 | 21.94 | 21.18 |

Ranking prediction (Kendall τ_b per tier, mean over tiers, table-level block bootstrap):

| comparison | P2 | P1 | verdict (P2) |
|---|---|---|---|
| random-split ranking vs loop ranking | 0.08 [-0.04, 0.19] | 0.09 [-0.02, 0.26] | does not predict |
| zero-shot ranking (label-informed) vs loop ranking | 0.23 [0.04, 0.40] | 0.21 [0.07, 0.33] | predicts |
| random-split ranking vs zero-shot ranking (H-E analogue) | 0.18 [0.03, 0.32] | — | predicts |

Reading. Once the loop measures and warm-starts, the choice of source family matters little: no family beats XGBoost
under P2 or P1, the forests are indistinguishable from it, the perceptron and the deep ensemble trail by 0.36–0.45
measurements, and the Gaussian process is worse by 0.61 [0.32, 1.01]. Without adaptation (P0) the
family does matter — the quantile forest and the random forest beat XGBoost by -0.69 and -0.49 measurements —
so adaptation absorbs most of the difference between families. The ranking a developer would compute from a
random split carries only weak information: it agrees with the zero-shot ranking on shifted tiers (0.18, interval
above zero) but does not predict which family needs the fewest measurements in the loop (0.08, interval
containing zero; positive in 49% of tiers). Even the label-informed zero-shot ranking predicts the
loop ranking only weakly (0.23). The Gaussian process raised 739 optimiser convergence warnings
over its 107 tier fits (none in its 30 cross-validation fits); fit time is not reported as a result (E-D10).

- F21 (Q1/Q4, model choice; F): inside an adaptive loop the source family matters less than adaptation — no
  family beats XGBoost once the loop warm-starts, while without adaptation forests are better by about half a
  measurement — and the random-split ranking of families does not predict their loop ranking.

## 11. E6 — negative controls and the fragility ledger (protocol `E6_NEGATIVE_CONTROLS_PROTOCOL_v1.md` revision 1.1, sha256 `d28896ae…`)

Runs `e6_run1/2`: shuffled-label replay of all 14 E1 policies on the 107 tiers (22470 loops; manifests identical
before and after analysis). Under permutation the reference policy needs 4.11 measurements and every
source-model policy lies between 3.89 and 4.36; the ceiling, refitted on the permuted labels, needs
1.37 and is "better" by design (positive control, E-D6). Of the 12 source-model policies, 11 are not
distinguishable from the reference under permutation; P5 (k-means seeding) is "better" by -0.22
[-0.51, -0.09] and is therefore an artefact candidate by the pre-registered rule. Post hoc (not
pre-registered; `e6_posthoc`), the same check with permutation seeds 1–3 gives seed 1: P5 +0.05 [-0.20, +0.71] not distinguishable; seed 2: P5 +0.24 [+0.03, +0.52] worse; seed 3: P5 +0.13 [-0.25, +0.38] not distinguishable — the seed-0 result is
one permutation's draw, but the pre-registered status of the k-means verdicts stays "artefact".

Chance level (analytic expected T* of uniformly random proposals) beside the E1 policies, by shift type:

| shift type | tiers | chance | P0 | P1 | P2 | CEIL |
|---|---|---|---|---|---|---|
| all | 107 | 4.39 | 3.79 | 3.31 | 2.80 | 1.23 |
| chemistry | 33 | 5.68 | 3.52 | 3.33 | 2.53 | 1.00 |
| composition | 11 | 4.33 | 5.76 | 3.95 | 3.39 | 1.09 |
| device | 30 | 4.04 | 2.60 | 2.93 | 2.10 | 1.03 |
| emulsion type | 2 | 4.06 | 2.83 | 3.85 | 1.83 | 1.00 |
| laboratory | 29 | 3.18 | 4.48 | 3.16 | 3.52 | 1.76 |
| preparation | 2 | 6.24 | 6.00 | 6.55 | 5.33 | 1.50 |

Fragility ledger — every "better" (Benjamini–Hochberg) verdict of E1, E1b, E3 and E4 with its drift and shuffled
status (20 verdicts: 5 artefact, 4 confirmed, 3 fragile, 8 positive control):

| source | unit | Δ [95%] | fragile (±0.2) | shuffled | status |
|---|---|---|---|---|---|
| E1 all tiers | CEIL | -2.072 [-2.758, -1.537] | yes | better | positive control |
| E1 all tiers | P2 | -0.502 [-0.870, -0.105] | no | not distinguishable | confirmed |
| E1 all tiers | P5 | -0.075 [-0.137, -0.027] | yes | better | artefact |
| E1 dense tiers | CEIL | -1.398 [-2.440, -0.967] | yes | better | positive control |
| E1 dense tiers | P5 | -0.343 [-0.512, -0.190] | no | better | artefact |
| E1b cell_tau0.05_B20 all tiers | CEIL | -3.197 [-4.036, -2.514] | yes | better | positive control |
| E1b cell_tau0.05_B20 all tiers | P2 | -0.661 [-1.068, -0.397] | no | not distinguishable | confirmed |
| E1b cell_tau0.05_B20 dense tiers | CEIL | -2.720 [-4.333, -1.873] | yes | better | positive control |
| E1b cell_tau0.05_B20 dense tiers | P2 | -0.609 [-1.300, -0.080] | yes | not distinguishable | fragile |
| E1b cell_tau0.05_B20 dense tiers | P5 | -0.461 [-0.972, -0.140] | no | better | artefact |
| E1b cell_tau0.10_B10 all tiers | CEIL | -1.955 [-2.637, -1.475] | yes | better | positive control |
| E1b cell_tau0.10_B10 all tiers | P2 | -0.450 [-0.755, -0.039] | no | not distinguishable | confirmed |
| E1b cell_tau0.10_B10 dense tiers | CEIL | -1.328 [-2.207, -0.967] | yes | better | positive control |
| E1b cell_tau0.10_B10 dense tiers | P5 | -0.272 [-0.408, -0.163] | no | better | artefact |
| E1b cell_tau0.05_B10 all tiers | CEIL | -2.860 [-3.620, -2.361] | yes | better | positive control |
| E1b cell_tau0.05_B10 all tiers | P2 | -0.502 [-0.874, -0.200] | no | not distinguishable | confirmed |
| E1b cell_tau0.05_B10 dense tiers | CEIL | -2.480 [-3.700, -1.870] | yes | better | positive control |
| E1b cell_tau0.05_B10 dense tiers | P2 | -0.480 [-0.878, -0.100] | yes | not distinguishable | fragile |
| E1b cell_tau0.05_B10 dense tiers | P5 | -0.461 [-0.825, -0.170] | no | better | artefact |
| E3 rule vs continue | lambda=0 | -0.963 [-1.375, -0.478] | yes | — | fragile |

Reading. Every confirmed "better" verdict in the closed-loop programme is the same finding — start the warm-started
search at once (P2) — on all tiers in E1 and in each of the three harder E1b cells. The k-means acquisition gains
(all tiers in E1; dense tiers in E1 and every E1b cell) fail the shuffled-label control by the pre-registered rule;
the dense-tier P2 gains are fragile; the free-escalation E3 rule is fragile; no E4 family is "better" than XGBoost.
The loops beat chance where the surrogate carries information (chemistry: P2 2.53 vs chance 5.68) and are close to
or above chance on the laboratory tiers (P1 3.16 vs 3.18), where the surrogate's prior misleads the first proposals.

- F22 (robustness; F): of 20 "better" verdicts in E1–E4, four survive both negative controls and all four are the
  same finding (P2, start at once); the k-means acquisition gains do not survive the shuffled-label control, and
  the remaining gains are fragile under drift or are positive controls.

## 12. E7 — the closed-loop questions on hard targets (protocol `E7_HARD_TASK_PROTOCOL_v1.md` revision 1.1, sha256 `130a1509…`; runner `09ae934d…`, analyzer `f64edbee…`)

Motivation (post hoc, from E1/E3/E6 outputs): at τ = 0.10 with central targets a median 23% of tier rows hit the
target, random proposals hit in 4.39 measurements, and on the easiest third of units no policy beat chance — the
benchmark could not resolve Q3/Q4. E7 uses tail targets (labels nearest the 5/10/90/95th percentiles), τ = 0.03 and
B = 20; the primary analysis is on hard units (hit fraction ≤ 10%). Runs `e7_run1/2` on the workstation:
124976 loops each, manifests byte-identical before and after analysis (no wall-clock value written; 29 min
per run, two in parallel). 428 units, of which 301 hard in 92 tiers (median hit fraction
5.4%); extrapolative units: 5 in 3 tiers (below the 20-tier threshold — descriptive only).
Shuffled-label control accepted on the hard units.

Hard units — mean T* (chance level 11.35), success within 20, Δ vs P1 [table-block 95%] with
Benjamini–Hochberg verdict, fragility under ±20% drift, shuffled-label artefact flag:

| policy | mean T* | success | Δ vs P1 [95%] | fragile | shuffled |
|---|---|---|---|---|---|
| P0 no adaptation | 6.58 | 90% | 0.07 [-0.60, 0.88] n.d. | no | — |
| P1 two rows + warm-start (reference) | 6.52 | 94% | reference | — | — |
| P2 warm-start at once | 5.72 | 93% | -0.79 [-1.37, -0.09] better | no | — |
| P3 five rows | 7.97 | 94% | 1.46 [0.82, 1.94] worse | no | — |
| P4 farthest-point | 6.64 | 93% | 0.12 [-0.26, 0.41] n.d. | yes | — |
| P5 k-means | 7.05 | 93% | 0.53 [0.30, 0.97] worse | no | — |
| P6 widest interval | 6.77 | 93% | 0.25 [-0.07, 0.45] n.d. | yes | artefact |
| P7 log-affine | 7.56 | 89% | 1.05 [0.85, 1.35] worse | no | — |
| P8 scratch | 8.89 | 91% | 2.38 [1.62, 2.92] worse | no | — |
| P9 portfolio | 7.29 | 93% | 0.78 [0.43, 1.29] worse | no | — |
| P13 interval-aware proposal (new) | 5.88 | 94% | -0.63 [-1.22, 0.11] n.d. | no | — |
| CEIL ceiling | 2.31 | 98% | -4.20 [-5.53, -3.41] better | yes | — |

Pre-registered decisions (hard units):
- H7a start at once: holds — P2 -0.79 [-1.37, -0.09], not fragile, not an artefact.
- H7b label-free acquisition matters: does not hold — farthest-point and widest-interval seeding not distinguishable
  (the latter also flagged by the shuffled control), k-means worse by 0.53.
- H7c interval-aware proposal helps: does not hold — P13 − P2 = 0.16 [0.10, 0.33]
  (slightly worse than the greedy proposal; better than P1 only because it starts at once).
- H7d warm-start dominates: holds — log-affine 1.05, scratch 2.38, five calibration rows 1.46,
  portfolio 0.78, all worse.
- H7f the loop beats chance: holds — P2 − chance = -5.63 [-7.64, -3.34] measurements.
- H7e decision rule from four rows: negative (1 of 3 λ, only λ = 0, where every fold learns "always escalate" and the
  verdict is fragile: +1.78 / +1.48 under drift). At λ = 5 and 10 four folds learn "always continue"
  (one learns a label-free width threshold); rule − continue = 0.08 [0.00, 0.39];
  the in-sample best rule is "always continue". Headroom of a perfect per-unit decision: 1.37 measurements at λ = 5
  (36.7% of units prefer another action) — larger than at k = 2 in E3 (0.54), still out of reach of these diagnostics.

Descriptive: on all 428 units the verdicts are the same, and P13 is "better" than P1 (−0.59) for the same reason as P2.
By hit-fraction tercile the hardest third (chance 14.77) shows the largest gaps (P2 -0.84, n.d. at 57 tiers; scratch
2.75); the easiest third (chance 5.45) the smallest. On the hard units by shift type, P2 helps on chemistry
and device tiers and not on laboratory tiers (P1 and P2 both 8.74). Three tiers have extrapolative targets; there P0 fails
most (16.50 of 21), P2 (−1.25) and farthest-point seeding (−1.92) are below P1, and widest-interval seeding has the
lowest mean (9.50) with a very wide interval — too few tiers for a verdict.

- F23 (Q2–Q4 on hard targets; F): on tail targets with a 3% tolerance, where random proposals need 11.35 measurements,
  the loop that starts at once and warm-starts needs 5.72 — and every closed-loop answer of E1 holds again: starting at
  once is better than calibrating first, no label-free acquisition rule or interval-aware proposal beats the greedy
  warm-started proposal, warm-start dominates the alternatives, and no rule on four-row diagnostics beats always
  continuing once escalation has a price. The null results on Q3 and Q4 are therefore not an artefact of easy tasks.
