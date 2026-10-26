# E1B_DIFFICULTY_PROTOCOL_v1 — Difficulty sensitivity of the E1 point-task decisions (τ × B) and the E-D1 density sensitivity

Frozen on 2026-10-22 after E1/E2 were complete and before any counted E1b run; sha256 in
`E1B_DIFFICULTY_PROTOCOL_v1.sha256`. This protocol does not modify `E0_E1_E2_CLOSED_LOOP_PROTOCOL_v1.md`
(revision 1.1), its runner or its analyzer; those files keep the hashes in `E_RUNNER_SCRIPT.sha256` and
`E_ANALYZER_SCRIPT.sha256`. E1b is run by a wrapper (`w7/closed_loop_e1b.py`) that imports the frozen
runner unchanged and overrides only the two constants named below; the wrapper and its analyzer
(`w7/closed_loop_e1b_analyze.py`) are hashed in `E1B_RUNNER_SCRIPT.sha256` and
`E1B_ANALYZER_SCRIPT.sha256` before the first counted run. Every threshold and decision rule precedes
the data. Positive and negative outcomes are reported alike. Nothing wet.

中文摘要:E1 的三条结论(k₀=0 最好;采集规则无用;warm-start 优于仿射/从头)是在 τ=10%、预算 20 下得到
的,大多数档位的中位 T* ≤ 3,差异只有零点几次测量。E1b 把任务变难:τ ∈ {0.05, 0.10} × B ∈ {10, 20},
三个新格(第四格就是 E1 本身,不重跑),同一 14 条策略、同一判定形式(相对 P1 的 ΔT* 表级块 bootstrap
区间 + BH + 漂移 ±20% 脆弱性)。每条 E1 结论按"在三个新格中成立几个"分级。另外登记 E-D1 透明度:协议 v1
原本把 V1 间隙当淘汰门(会淘汰 107 档位中的 89 个疏档位),修订 1.1 在冒烟运行后、任何计数运行之前把它改
成描述性分层;本协议预先规定在仅保留 18 个密档位(即原门之下)时重算全部 E1 与 E1b 判定并并列报告。

## 1. Cells

(τ, B) ∈ {(0.05, 20), (0.10, 10), (0.05, 10)}. The fourth cell (0.10, 20) is E1 (`e1e2_run1/2`) and is
not rerun. τ enters only the success test |y_t − y*|/y* ≤ τ and the regret increment max(0, err − τ);
B enters only the loop length and T* = B + 1 for failure. Nothing else changes: tiers (107, from
`e0_run1/tiers_e0.csv` and `exclusions.json`), source models, targets (labels nearest the 20/50/80th
label percentiles), the 14 policies P0–P12 and CEIL with their definitions, seeds 0–9 (main), drift
δ = ±0.2 with seeds 0–2 (fragility), shuffled-label control with P0/P1 and seeds 0–4.

Dropped in E1b (compute): the ±0.5 descriptive drift runs and the E2 budget curves (E2 does not depend on
τ or B). The density label (dense = median nearest-row relative gap ≤ 0.05) is an E0 output and is kept
as it is for every cell, including τ = 0.05; it is a stratifier, not a gate.

## 2. Decisions per cell (identical to E0_E1_E2 protocol §3.6)

Tier value = mean T* over 3 targets × 10 seeds. Δ(P) = mean over tiers of [T*(P) − T*(P1)]; table-level
block bootstrap (B = 2000, seed 0; stratified tier bootstrap secondary). "better" iff Δ < 0 and the
primary interval excludes 0; "worse" iff Δ > 0 and it excludes 0; else "not distinguishable".
Benjamini–Hochberg over the 13 non-reference policies (verdict_bh). Fragile = sign of Δ changes under
δ = −0.2 or +0.2. The shuffled-label control is recomputed per cell (the simulator is accepted iff the
block interval of mean [T*(P1) − T*(P0)] under shuffling contains 0); a cell whose control fails is
reported but its verdicts are not used for the grading below.

## 3. Pre-registered questions and grading

Each E1 finding is graded by the three new cells (E1 itself already holds by construction):

- (a) F14, k₀ = 0: "holds" in a cell iff P2 is "better" (verdict_bh) and not fragile.
- (b) F15, acquisition: "holds" in a cell iff none of P4, P5, P6 is "better" (verdict_bh) with Δ < −0.10
  and non-fragile. (A fragile or sub-0.10 nominal gain, as k-means in E1, keeps F15.)
- (c) F16, adaptation: "holds" in a cell iff P7 (affine) and P8 (scratch) are both "worse" (verdict_bh).
- (d) also graded, same form: F16′ portfolio — P9 "worse"; F17 — the ranking of {P0, P1, P2, CEIL} by mean T*
  at δ = 0 is unchanged at δ = ±0.2 except for CEIL, which may fall (as in E1).

Grade: "robust to difficulty" = holds in 3 of 3 new cells; "mostly robust" = 2 of 3; "specific to the
E1 setting" = 0 or 1 of 3. The cells where a finding fails are named with their Δ and interval. No
finding is dropped or upgraded by this grading; the grade is appended to the F-list entry.

## 4. E-D1 transparency and the original-gate (dense-subset) sensitivity

Protocol v1 (frozen before the smoke run) made the V1 median nearest-row gap ≤ τ/2 an eligibility gate.
After the uncounted smoke run (12 tiers, one per code) showed the gate would exclude 9 of 12 tiers, and
before any counted run, revision 1.1 turned the gate into a descriptive label (E-D1 in
`E_DEVIATION_REGISTER.md`, same day). On the 107 counted tiers the original gate would have kept 18
tiers (D1 7, D2 3, R2-A1 3, R2-A2 5; no A1 or D3 tier). Pre-specified sensitivity: every decision of §2
(and E1's own decisions on `e1e2_run1`) is recomputed on the 18 dense tiers only, with the same bootstrap
(table blocks = the 4 tables present), and printed beside the all-tier verdicts as `results_dense.json`.
A finding whose all-tier verdict is not reproduced on the dense subset is reported as "depends on the
sparse tiers", with both verdicts; the all-tier analysis remains primary because the sparse tiers are
legitimate target domains for a row-indexed oracle. The E1 dense recomputation is written to
`WORK/e1_dense_sensitivity/` and never into the counted `e1e2_run1/2` directories.

## 5. Outputs and integrity

`WORK/e1b_run1/`, `WORK/e1b_run2/`, each with `cell_tau0.05_B20/`, `cell_tau0.10_B10/`,
`cell_tau0.05_B10/` containing `loops.csv`, `run_manifest.json` (records the overrides, the frozen runner's
sha256 and this protocol's sha256), `environment.json`, `artifact_hashes.json`; the two runs must have
byte-identical hash manifests. The analyzer adds `results.json`, `policy_table.csv`, `results_dense.json`,
`policy_table_dense.csv` per cell and a run-level `e1b_summary.json` with the §3 grades. Smoke runs with
reduced sizes are not counted. Ledger prefix `e1b.`; the dense sensitivity of E1 uses `e1.dense.`.
Repository copies (without `loops.csv`) under `scaleup/e_closed_loop/e1b_run1/2/` and
`scaleup/e_closed_loop/e1_dense_sensitivity/`.

## 6. Compute

Per cell ≈ 45 k main + 27 k drift + 3 k control loops; three cells ≈ 225 k loops per run (E1: 80 k loops
plus E2). τ = 0.05 lengthens loops. If a counted run exceeds 3 h on 8 processes, the (0.05, 10) cell is
dropped first and recorded as not run.
