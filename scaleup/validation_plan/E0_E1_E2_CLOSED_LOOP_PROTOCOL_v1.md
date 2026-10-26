# E0_E1_E2_CLOSED_LOOP_PROTOCOL_v1 (revision 1.1) — Offline closed-loop benchmark on the public tables

Frozen on 2026-10-22 before any counted run; sha256 in `E0_E1_E2_CLOSED_LOOP_PROTOCOL_v1.sha256`.
Supersedes the draft v0 with the five open choices settled (Section 8). Revision 1.1 (same day, after the
uncounted smoke run and before any counted run) changes two items, both recorded in
`E_DEVIATION_REGISTER.md` as E-D1 and E-D2: (i) the V1 gap is a descriptive density stratifier, not an
eligibility gate, because proposals are row-indexed (Section 3.2) and the gap measures sampling density
rather than oracle distortion — in the smoke run the gate would have excluded 9 of 12 tiers; (ii) under V3
drift the target is expressed in the drifted oracle's scale, y*·(1 + δ), so that drift stresses the
surrogate's calibration and not the reachability of the target. Every threshold and decision
rule below precedes the data. Positive and negative outcomes are reported alike. Nothing wet.

中文摘要:107 个留一组档位 = 107 个离线实验室。E0 先验证模拟器(查询失真、噪声底线、漂移、三条对照线)
并产出淘汰档位清单;E1 回放闭环(14 条策略 × 3 个目标 × 10 个种子,预算 20 次测量;指标 = 命中目标所需
测量次数 T*、累计遗憾),判定 = 相对参考策略(随机采集 + warm-start + 贪心,k₀=2)的 ΔT* 表级块
bootstrap 区间;E2 从随机抽样 k 曲线得到各类偏移的测量预算 k*(ε)。决策规则预览 P12 只作预览,正式判定留给
E3。双跑;台账前缀 e0./e1./e2.。

## 1. Units and source models

Target domains: the 106 leave-one-group tiers of `H_J_FEWSHOT_RULE_PROTOCOL_v1.md` plus the D3
laboratory-B tier (57 rows, source = the laboratory-A development rows with SIZE > 0, inputs
`d3_design`, log target, locked regressor of `posthoc_round2`), 107 tiers. Source constructor per
table as in H-J (D1 build_model(42); D2 make_xgb(42); A1/R2-A1/R2-A2 make_point("xgboost", 42); D3
locked_regressor). Features, target scales and OOF groupings as in H-J; D3 OOF groups = row folds.

Per tier, computed once before any loop: the source model on the source rows; its in-sample
predictions (warm-start base margins) and its predictions on the tier rows; 5-fold grouped OOF
predictions on the source rows giving dev-ref (OOF MAPE) and q_src (90th percentile of the OOF
absolute log residuals); a quantile forest (`w3.uq_models.QuantileForest`, seed 42) on the source rows
with log target giving the interval width (upper − lower) of every tier row; the standardised
tier inputs z (source mean and SD, constant columns dropped).

## 2. E0 — simulator validity (runs first; gates E1/E2)

- V1 query distortion. For every tier row, the nearest other row of the tier in z space; record the
  distance d and the relative output gap g = |y_nn − y| / y. Per tier: median g, 90th percentile of g,
  95th percentile of d (= ρ). Tiers with median g ≤ τ/2 = 0.05 are labelled "dense", the others
  "sparse"; every tier with ≥ 12 rows is eligible (revision 1.1), and E1/E2 results are reported for all
  tiers and stratified by this label. The label list is an output of E0.
- V2 noise floor. Rows of a table with identical inputs (six significant digits) are replicates; the
  pooled relative SD of y across replicate groups is the table's noise floor (0 if none; stated). No
  policy is credited with an error below it (E2 reporting).
- V3 drift. Loops are repeated with oracle labels multiplied by (1 + δ) and the target expressed in the
  same scale, y*·(1 + δ) (revision 1.1): δ ∈ {−0.2, +0.2} for every policy with seeds 0–2 (pre-registered
  fragility check); δ ∈ {−0.5, +0.5} for P0, P1, P3, P6 with seeds 0–1 (descriptive only).
- Controls. (a) Shuffled-label oracle: labels permuted within tier (seed 0); P0 and P1 are run on it
  with seeds 0–4; the simulator is accepted iff the table-block bootstrap interval of mean
  [T*(P1) − T*(P0)] under shuffling contains 0. (b) Floor = P0. (c) Ceiling = the source constructor
  refitted on all tier labels, greedy proposal.

## 3. E1 — closed-loop benchmark (point task)

3.1 Task. For each tier, three targets: the labels of the rows nearest the 20th, 50th and
80th percentiles of the tier's labels (so every target is reachable). τ = 0.10 relative (primary).
Budget B = 20 measurements. Success at step t: the measured row's returned label satisfies
|y_t − y*| / y* ≤ τ. T* = t at first success, else B + 1. Cumulative regret = Σ_t max(0, |y_t − y*| /
y* − τ) over steps before success. Secondary descriptive success criterion: absolute tolerance 0.10 ×
the table's development median label; not used in any decision.

3.2 Oracle. Proposals are tier row indices; the oracle returns that row's label (times 1 + δ under
V3) and consumes the row. Because proposals are rows, the query distortion of a row-indexed loop is
zero; V1 therefore gates tier eligibility rather than scoring proposals.

3.3 Loop. Phase A: measure k₀ rows chosen by the acquisition rule (success is checked after each).
Phase B: until success or budget: adapt the surrogate with all measured rows, propose one unmeasured
row, measure it.

3.4 Components.
- k₀ ∈ {0, 2, 5}.
- Acquisition: random (seeded uniform without replacement); farthest-point (`posthoc_round3.
  farthest_point_rows` on z with the source mean as origin); k-means (`kmeans_rows`, k₀ centroids);
  uncertainty = the k₀ tier rows with the widest source quantile-forest interval (label-free; ties by
  row index). An adaptive variant (re-ranking by the adapted model's conformal radius each step) is
  deferred to E1b (Section 8).
- Adaptation (round-3/4 definitions): none; affine (two-parameter log-affine for k ≥ 2 with
  non-constant source predictions, one-parameter log shift for k = 1 or constant predictions, none for
  k = 0); warm-start residual boosting (`WARM_PARAMS`, λ = 0.1, base margin log f, for k ≥ 1);
  scratch (XGBoost 100/3/0.3 seed 42 on the measured rows with the table's target scale, for k ≥ 3,
  none below).
- Proposal: greedy = argmin over unmeasured rows of |f_adapted − y*| (ties: lowest index);
  portfolio = greedy toward the sub-target y*·exp(s·q), s cycling 0, −1, +1 with the step index,
  where q = q_src until 10 rows are measured and thereafter the conformal radius of the source
  model's absolute log residuals on the measured rows (rank ⌈(k+1)·0.9⌉).

3.5 Policies (each run on every eligible tier × 3 targets × seeds 0–9; δ = 0):
P0 floor (0, —, none, greedy) · P1 reference (2, random, warm, greedy) · P2 (0, —, warm, greedy) ·
P3 (5, random, warm, greedy) · P4 (2, farthest, warm, greedy) · P5 (2, k-means, warm, greedy) ·
P6 (2, uncertainty, warm, greedy) · P7 (2, random, affine, greedy) · P8 (5, random, scratch, greedy) ·
P9 (2, random, warm, portfolio) · P10 (5, farthest, warm, portfolio) · P11 (2, uncertainty, warm,
portfolio) · P12 preview rule (2, random; bias ratio |mean r|/SD of the two log residuals > 0.63 →
warm, else none; greedy; the threshold is in-sample from H-J and this policy is a preview, not a
test) · CEIL ceiling.

3.6 Decisions. Tier value of a policy = mean T* over its 3 targets × 10 seeds. Δ(P) = mean over all
tiers of [T*(P) − T*(P1)], table-level block bootstrap interval (B = 2000, seed 0; stratified
tier bootstrap secondary). P is "better than the reference" iff Δ < 0 and the primary interval
excludes 0; "worse" iff Δ > 0 and it excludes 0; else "not distinguishable". Benjamini–Hochberg over
the 13 non-reference policies. Reported with P0 and CEIL on the same scale and, descriptively, by
shift type, by density label and by k₀. Fragile = the sign of Δ(P) changes under δ = −0.2 or +0.2 (seeds 0–2 for both
P and P1). Pre-registered questions answered by E1: (Q3) does any acquisition rule beat random
(P4, P5, P6 vs P1; P10, P11 vs P9); (Q2) does k₀ matter (P2, P3 vs P1); (Q4) does the adaptation
method matter (P7, P8 vs P1); does the portfolio proposal help (P9 vs P1; P10 vs P3; P11 vs P6).

## 4. E2 — measurement budget (model task)

For every tier, k ∈ {1, 2, 3, 5, 8, 10, 15, 20} with n − k ≥ 10, 100 random draws (seeds
0–99), adaptation ∈ {affine, warm, scratch}: MAPE on the unmeasured rows. k*(ε) = the smallest k whose
median MAPE over draws is ≤ ε = 1.25 × dev-ref of the tier; "not reached within 20" is a value.
Reported per shift type and per density label as the median and 90th percentile of k*(ε) across tiers,
per adaptation; also per tier. Descriptive; no threshold beyond ε. The table noise floor (V2) is printed beside each
table's curves.

## 5. Outputs and integrity

`WORK/e0_run1`, `WORK/e0_run2` (E0), `WORK/e1e2_run1`, `WORK/e1e2_run2` (E1 + E2); each pair must have
byte-identical `artifact_hashes.json`. Files: `tiers_e0.csv`, `exclusions.json`, `loops.csv`,
`budget.csv`, `results.json`, `run_manifest.json`, `environment.json`. Runner
`scaleup/w7/closed_loop.py` and analyzer `scaleup/w7/closed_loop_analyze.py` hashed with sidecars
before the first counted run; smoke runs with reduced sizes are not counted. Deviation register
`E_DEVIATION_REGISTER.md`. Ledger prefixes `e0.`, `e1.`, `e2.`. Repository copies under
`scaleup/e_closed_loop/`.

## 6. Compute budget and what is dropped if exceeded

Primary E1 ≈ 14 policies × 107 tiers × 3 targets × 10 seeds ≈ 45 k loops; drift runs ≈ 14 × 107 × 3 ×
3 × 2 ≈ 27 k; ±0.5 descriptive ≈ 4 × 107 × 3 × 2 × 2 ≈ 5 k; E2 ≈ 107 × 8 × 100 × 3 ≈ 0.26 M fits. If a
counted run exceeds 6 h of wall time on 8 processes, the ±0.5 descriptive runs are dropped first, then
the manuscript-grid secondary targets (not implemented in this version; recorded as not run).

## 7. Later experiments pre-registered here in outline

E3 decision policy: actions {continue, switch to scratch, escalate}; escalation cost λ ∈ {0, 5, 10}
measurement-equivalents; five-fold table rotation; separate protocol. E1b adaptive uncertainty
acquisition: only if its extra compute is ≤ 40% of E1. E4–E6 as in `CLOSED_LOOP_FRAMEWORK_v0.md`.

## 8. Settled choices (from the draft's open items)

1. τ = 0.10 relative for the decision; absolute 0.10 × development median descriptive only.
2. Targets = labels of the rows nearest the 20/50/80th label percentiles; the D1/D2 manuscript grids
   are not run in this version.
3. B = 20; k₀ ∈ {0, 2, 5}.
4. V3 δ = ±0.2 pre-registered (seeds 0–2); ±0.5 descriptive (P0, P1, P3, P6; seeds 0–1).
5. Uncertainty acquisition arm included as the label-free source quantile-forest width ranking (P6,
   P11); the adaptive variant deferred to E1b.
6. ρ = 95th percentile of within-tier leave-one-row-out nearest-neighbour distance (reported; not a
   gate for row-indexed proposals).
7. P12 is a preview with the in-sample threshold 0.63; the decision test is E3.
