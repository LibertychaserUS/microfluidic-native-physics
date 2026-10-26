# E7_HARD_TASK_PROTOCOL_v1 (revision 1.1) — The closed-loop questions on hard and extrapolative targets

Frozen on 2026-10-24 before any counted E7 run; sha256 in `E7_HARD_TASK_PROTOCOL_v1.sha256`. E7 is run by a
wrapper (`w7/closed_loop_e7.py`) that imports the frozen E1 runner unchanged and overrides only the constants and
policies named below, and analysed by `w7/closed_loop_e7_analyze.py` (which reuses the frozen E1 analyzer's
bootstrap and the E3 analyzer's rule learning); both hashed (`E7_RUNNER_SCRIPT.sha256`,
`E7_ANALYZER_SCRIPT.sha256`) before the first counted run. Every threshold and decision rule precedes the data.
Positive and negative outcomes are reported alike. Nothing wet. Revision 1.1 (same day, after the uncounted
smoke run, before any counted run): the multiplicity family is the 11 non-reference, non-arm policies (P0, P2–P9,
P13, CEIL); the frozen text said 12 (deviation E-D11).

中文摘要:E1–E6 的点任务太容易——档位中位只有 28 行,平均 23% 的行落在目标 ±10% 以内,随机乱选平均 4.39 次
就能命中,三分之一的单元上任何策略都不比随机好(事后分层,E7 的动机)。E7 把任务变难、变得更像真实使用:目标取
标签分布尾部(最接近 5、10、90、95 分位数的标签),容差 τ = 3%,预算 20;主分析只在"难单元"上做(命中行占比
h ≤ 10%);另报"外推单元"(目标在源表标签范围之外)。问题:k₀ = 0 是否仍最好;采集规则在难目标上是否有用(并
内置打乱标签对照,防止 E6 发现的产物);新增"区间感知提议"是否优于贪心;自适应方法排序;以及用 4 行(而非 2
行)诊断量学到的决策规则是否优于"总是继续"。判定形式同 E1/E3,全部正负并列。

## 1. Units, targets and difficulty

Tiers: the 107 tiers of E0 (`e0_run1`), with the source models, out-of-fold quantities and quantile-forest widths
of the frozen E1 runner (`precompute`). Targets per tier: the labels nearest the 5th, 10th, 90th and 95th
percentiles of the tier's labels (`targets_for` with these quantiles; a label that is nearest to two quantiles is
used twice). Tolerance τ = 0.03 relative; budget B = 20; T* = B + 1 on failure; regret as E1.

Unit = (tier, target). For every unit, from the tier labels only (no policy outcome): m = number of tier rows with
|y − y*|/y* ≤ τ, n = tier rows, hit fraction h = m/n, the analytic chance level of uniformly random proposals
(E6 formula), and the extrapolation flag: y* below the minimum or above the maximum of the tier's source labels.

- Hard units: h ≤ 0.10. Primary analyses use the hard units; a tier enters the primary analysis with the mean over
  its hard units. Fallback (pre-registered): if fewer than 30 tiers have at least one hard unit, the hard set is the
  lowest tercile of h over all units.
- Extrapolative units: the extrapolation flag is set; analysed descriptively (and with the same decision form if
  at least 20 tiers have one).
- All units: reported beside, with h-tercile stratification.

## 2. Policies

From E1 (definitions unchanged): P0 (no adaptation), P1 (2 random rows, warm-start, greedy; reference), P2 (k₀ = 0,
warm-start, greedy), P3 (k₀ = 5), P4 (farthest-point), P5 (k-means), P6 (widest source interval), P7 (log-affine),
P8 (scratch, k₀ = 5), P9 (portfolio), CEIL (ceiling). New:
- P13 interval-aware proposal (k₀ = 0, warm-start): the next row minimises max(0, |log f̂ − log y*| − w/2), where
  w is the 90% interval width (log scale) of the source quantile forest for that row; ties by smaller
  |log f̂ − log y*|, then lower row index. A row whose source interval covers the target counts as distance 0.
- Decision arms after k₀ = 4 random rows (the same rows for all four arms for a given tier and seed): A_cont
  (warm-start), A_none (no adaptation), A_switch (scratch from k ≥ 3), A_esc (ceiling model at cost λ).
  Diagnostics from the four rows: d1 bias ratio |mean r| / SD(r) (ddof 1), d2 mean |r| / q_src, d5 sign
  consistency (share of the four residuals with the majority sign); tier-level label-free d3 (mean interval width
  / q_src) and d4 (share of rows with any |z| > 3), as E3.

Runs: every policy on every unit with seeds 0–9 (δ = 0); drift δ = ±0.2 with seeds 0–2 (target in the drifted
scale); shuffled-label replay (labels permuted within tier with seed 0; targets = the permuted labels nearest the
original targets; CEIL refitted on the permuted labels) for the 12 non-arm policies with seeds 0–2.

## 3. Decisions (hard units; table-level block bootstrap B = 2000, seed 0; Benjamini–Hochberg over the 11 non-reference, non-arm policies)

Tier value = mean T* over the tier's hard units × seeds. Δ(P) = mean over tiers of [T*(P) − T*(P1)]; "better" /
"worse" / "not distinguishable" as E1 §3.6; fragile = sign change under δ = −0.2 or +0.2; artefact = "better"
under the shuffled replay (CEIL is the positive control). Simulator check: the shuffled P1 − P0 interval must
contain 0 on the hard units.

- H7a (Q2): "start at once still wins on hard targets" iff P2 is better than P1, not fragile, not an artefact.
- H7b (Q3): "label-free acquisition matters on hard targets" iff at least one of P4, P5, P6 is better than P1,
  not fragile and not an artefact; otherwise "does not".
- H7c (Q3, proposal): "interval-aware proposal helps" iff P13 − P2 < 0 with a block interval excluding 0 and
  the sign unchanged under δ = ±0.2, and P13 is not an artefact.
- H7d (Q4, adaptation): "warm-start dominates on hard targets" iff P7 and P8 are both worse than P1.
- H7e (Q4, decision): within the hard units, the E3 rule family (one diagnostic among d1, d2, d3, d4, d5,
  threshold at a training decile, one action on each side, plus the four constant rules) learned by five-fold
  table rotation (D3 held out with D2) at λ ∈ {0, 5, 10}; the rule holds at λ iff its held-out cost is below both
  "always continue" (A_cont) and "never adapt" (A_none) with block intervals excluding 0; H7e is positive iff it
  holds at ≥ 2 of 3 λ. Headroom (A_cont − per-unit oracle) reported.
- H7f (loop vs chance): mean over tiers of [T*(P2) − chance], block interval; "the loop beats chance on hard
  targets" iff the interval lies below 0.
- Descriptive: every policy's mean T* and success rate on all, hard and extrapolative units and by h tercile;
  extrapolative units with the H7a–d decision form if ≥ 20 tiers have one.

## 4. Outputs, integrity, compute

`WORK/e7_run1`, `WORK/e7_run2`: `loops.csv`, `units.csv`, `run_manifest.json`, `environment.json`,
`artifact_hashes.json` (byte-identical across the two runs; no wall-clock value is written); the analyzer adds
`results.json` and `policy_table.csv`. Ledger prefix `e7.`; repository copies without `loops.csv` under
`scaleup/e_closed_loop/e7_run1/2/`. Compute ≈ 16 policies × 107 tiers × 4 targets × 10 seeds = 68 k main loops,
41 k drift loops, 15 k shuffled loops; workstation, two runs in parallel. Smoke runs with reduced sizes are not
counted.
