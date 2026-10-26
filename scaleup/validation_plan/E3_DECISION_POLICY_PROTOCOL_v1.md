# E3_DECISION_POLICY_PROTOCOL_v1 (revision 1.1) — Q4: a diagnostic-driven decision rule after two measurements (continue / stop adapting / switch / escalate)

Frozen on 2026-10-22 before any counted E3 run; sha256 in `E3_DECISION_POLICY_PROTOCOL_v1.sha256`. Separate
from `E0_E1_E2_CLOSED_LOOP_PROTOCOL_v1.md`, whose runner and analyzer stay unchanged; E3 is run by a wrapper
(`w7/closed_loop_e3.py`) that imports the frozen runner and adds policies, and analysed by
`w7/closed_loop_e3_analyze.py`; both hashed (`E3_RUNNER_SCRIPT.sha256`, `E3_ANALYZER_SCRIPT.sha256`) before
the first counted run. E1's P12 was a preview with an in-sample threshold; E3 is the test. Every threshold,
rule family and decision form precedes the data. Positive and negative outcomes are reported alike.

中文摘要:闭环先随机测 2 行,然后由这两行的诊断量决定四个动作之一:继续(warm-start)、停止自适应(用源模型
原样提议)、切换(放弃源模型,从第 3 次起只用测量数据从头训练)、升级(付出 λ 次测量等价的代价换取"天花板"
模型,即源构造器在该档位全部标签上重拟合;λ ∈ {0, 5, 10})。规则族 = 单一诊断量的阈值规则(高于阈值一个动
作、低于另一个)加四条常数规则;在四张表上学、在第五张表上测(五折表轮换,D3 随 D2 一起留出、从不训练)。判
定:规则的留出成本相对"总是继续"和"从不自适应"都更低且表级块 bootstrap 区间不含 0,并且在 λ 的三个水平中至
少两个成立,E3 才算正面。另报"完美决策的上限空间" H_λ。

## 1. Units, arms and cost

Units: the 107 tiers of E0 (`e0_run1`), 3 targets per tier (as E1), seeds 0–9, δ = 0; drift δ = ±0.2 with
seeds 0–2 for the fragility check. Phase A for every arm = 2 random rows (`choose_phase_a` with the frozen
seeded generator, so the two rows are identical across arms for a given tier and seed and identical to E1's
P1 rows). Success is checked after each phase-A row as in E1. Arms (phase B after k₀ = 2, greedy proposal):

| arm | code | adaptation after k = 2 | meaning |
|---|---|---|---|
| continue | P1 | warm-start residual boosting (frozen) | keep adapting the source model |
| none | P15 = (2, random, none, greedy) | unadapted source predictions | stop adapting; trust the source as is |
| switch | P13 = (2, random, scratch, greedy) | frozen `scratch`: unadapted source for the proposal at k = 2, from k ≥ 3 XGBoost on the measured rows only | abandon the source model |
| escalate | P14 = (2, random, ceiling, greedy) | the ceiling model (source constructor refitted on all tier labels), as E1's CEIL but after phase A | call for a full recalibration at a price |

Cost of an arm on a unit u = (tier, target, seed): c_a(u) = T*_a(u) + λ·1[a = escalate], with
T* = B + 1 = 21 on failure (τ = 0.10, B = 20 as E1) and λ ∈ {0, 5, 10} measurement-equivalents. Units where a
phase-A row already succeeds have equal T* across arms (the decision is never taken).

## 2. Diagnostics at k = 2 (computed from the two phase-A rows and the pre-computed source quantities only)

With r_i = log y_i − log f(x_i) for the two measured rows (y in the oracle's scale, so under drift r shifts
by log(1 + δ)):
- d1 bias ratio = |mean r| / SD(r) (ddof 1; 10 if SD = 0) — the H-J/P12 statistic;
- d2 magnitude ratio = mean |r| / q_src, q_src = the source model's 90th-percentile OOF absolute log residual;
- d3 (label-free) = mean over tier rows of the source quantile-forest interval width / q_src;
- d4 (label-free) = fraction of tier rows with max_j |z_j| > 3, z the inputs standardised by the source mean and SD
  (support overlap: a row is outside the source's support when any column is more than 3 source SDs from the source
  mean). Revision 1.1 (same day, after the uncounted smoke run and before any counted run): the draft used the mean
  ‖z‖/√p, which is dominated by columns that are nearly constant in the source (values of 10^16 in the smoke run);
  the fraction-outside form is bounded and ordinal.
d3 and d4 depend on the tier only; d1 and d2 on (tier, seed).

## 3. Rule family and learning

A rule is (d, θ, a_lo, a_hi): action a_hi if d > θ else a_lo, with d ∈ {d1, d2, d3, d4}, θ in the
10th–90th percentiles (step 10) of d over the training units at δ = 0, (a_lo, a_hi) ∈ {continue, none,
switch, escalate}²; plus the four constant rules. Training objective = mean c_a(u) over training units under
the rule's action; ties → the constant "continue" rule, then the rule with fewer escalate actions, then the
lexicographically smaller (d, θ, a_lo, a_hi). Five-fold table rotation: folds = the five public tables D1,
D2, A1, R2-A1, R2-A2; the D3 tier is held out with D2 and never trained on. One rule per λ per fold; the
held-out costs of the five folds are assembled over all 107 tiers.

## 4. Decision form

Tier value = mean over targets × seeds. For each λ: Δ_cont = mean over tiers of [c_rule − c_continue],
Δ_none = mean [c_rule − c_none]; table-level block bootstrap (B = 2000, seed 0) intervals as E1. The rule
"holds at λ" iff Δ_cont < 0 and Δ_none < 0 with both intervals excluding 0. E3 is "positive" iff the rule
holds at ≥ 2 of the 3 λ levels; otherwise "negative", with the λ levels where it holds named. Reported
beside: the constant arms' mean costs (always continue / none / switch / escalate + λ), the oracle headroom
H_λ = mean over tiers of [c_continue − min_a c_a(u)] (what a perfect per-unit decision could save), the
fraction of units whose oracle-best action is not "continue", and the rule chosen in every fold. Fragility:
the fold rules learned at δ = 0 are applied to the δ = ±0.2 loops (seeds 0–2, diagnostics recomputed from
the drifted labels); the sign of Δ_cont under drift is reported, and a "holds" verdict whose sign flips is
marked fragile. P12's preview threshold (0.63) is one point of this family and is reported as such, not
tested separately.

## 5. Outputs, integrity, compute

`WORK/e3_run1`, `WORK/e3_run2` with `loops.csv` (one row per arm × unit × kind, with the four diagnostics),
`run_manifest.json`, `environment.json`, `artifact_hashes.json` (byte-identical across the two runs); the
analyzer adds `results.json` and `rules_table.csv`. Ledger prefix `e3.`; repository copies without
`loops.csv` under `scaleup/e_closed_loop/e3_run1/2/`. Compute ≈ 4 arms × 107 × 3 × 10 = 12.8 k main loops +
7.7 k drift loops. Smoke runs with reduced sizes are not counted.
