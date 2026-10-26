# E5_BUDGET_TRANSFER_PROTOCOL_v1 — Do measurement budgets and their planning rules transfer across tables?

Frozen on 2026-10-22 before any counted E5 run; sha256 in `E5_BUDGET_TRANSFER_PROTOCOL_v1.sha256`. E5 is an
analysis of frozen outputs that already exist (E2 budget curves in `e1e2_run1/budget.csv`, E0 tier
descriptors in `e0_run1/tiers_e0.csv`, E3 two-row diagnostics in `e3_run1/loops.csv`); it runs no new loop.
The analyzer `w7/closed_loop_e5.py` is hashed (`E5_ANALYZER_SCRIPT.sha256`) before the counted run and run
twice into separate directories (byte-identical outputs). Every threshold and decision form precedes the data.
Positive and negative outcomes are reported alike. Nothing wet.

中文摘要:E2 给出每个档位的预算 k*(ε)(warm-start 在 20 行内回到 1.25×开发参照所需的行数,未达到记 21)。E5 问:
一个实验室在新的目标域上该计划多少次测量,能否从别的表学到?规划规则 = 按特征(偏移类型;偏移类型×密/疏;无标签
支持域诊断量;两行偏置比)对训练档位 k* 取 80 分位数作为计划预算 B̂;代价 = 超量测量的行数(B̂ ≥ k*)或整个预算
20 行(B̂ < k*,模型不可用)。五折表轮换:在四张表上学规则、用于第五张表(迁移),与在该表自身档位上学到的规则
(表内)比较;迁移遗憾 = 迁移代价 − 表内代价,表级块 bootstrap 区间。另描述各偏移类型的预算曲线形状在表间的一致性。

## 1. Units and the quantity to plan

Tiers: the 107 tiers of E0. For every tier, from E2 (warm-start adaptation): the median MAPE over the 100
draws at each k ∈ {1, 2, 3, 5, 8, 10, 15, 20}, the uncalibrated MAPE, and k*(ε) = the smallest k whose
median MAPE ≤ ε = 1.25 × dev-ref; "not reached within 20" is coded 21 and such tiers are handled separately
(Section 4). The analyzer recomputes k* from `budget.csv` and checks it against `budget_table.csv`.

## 2. Planning rules (each maps a tier's pre-measurement features to a planned budget B̂)

Lookup rules take, over the training tiers sharing the feature value, the 80th percentile of k* (q = 0.80,
pre-registered: a planner accepts under-planning on one tier in five); a feature value absent from the
training tiers falls back to the global 80th percentile of the training tiers. Rules:
- R0 constant: B̂ = 20 (the full budget).
- R1 shift type.
- R2 shift type × density label (E0).
- R3 shift type × label-free support diagnostic: d4 (fraction of tier rows outside the source support, E3)
  above or below the training median of d4.
- R4 shift type × two-row bias ratio: the tier's median d1 over the E3 seeds above or below the training median.
- ORACLE: B̂ = min(k*, 20).

## 3. Cost of a planned budget

For a tier with true k* ≤ 20: C(B̂) = B̂ − k* if B̂ ≥ k* (rows measured beyond need), else 20 (the surrogate
is not usable when the loop stops; the laboratory has to spend the full budget). Tiers with k* = 21 are
excluded from the cost comparison, because every rule with B̂ ≤ 20 pays 20 on them; for those tiers each
rule's "not reachable" classification is reported instead (a rule predicts "not reachable" when its lookup
value is 21).

## 4. Transfer test and decisions

Five-fold table rotation: folds = D1, D2 (with D3 held out together, never trained on), A1, R2-A1, R2-A2.
For each rule and fold: the transferred rule is learned on the four training tables and applied to the
held-out tiers; the table-specific rule is learned on the held-out table's own tiers (in sample) and applied
to the same tiers; both costs are assembled over all 107 tiers.
- Transfer regret per rule = mean over tiers of [C_transferred − C_table-specific], table-level block
  bootstrap (B = 2000, seed 0). A rule "transfers" iff the regret interval contains 0 or lies below 0, and
  the transferred rule's cost is below the constant rule R0 with an interval excluding 0; it "does not
  transfer" iff the regret is positive with an interval excluding 0; otherwise "no advantage over the full
  budget".
- Also reported: mean cost of every rule (transferred, table-specific, R0, oracle), the fraction of tiers
  under-planned, and the sensitivity of the regret to q ∈ {0.50, 0.90} (descriptive).
- Shape transfer (descriptive): per shift type and table, the median over tiers of the gain curve
  g_k = median MAPE(k) / uncalibrated MAPE; the between-table range of g_k at each k for shift types present in
  at least two tables.

## 5. Outputs, integrity

`WORK/e5_run1`, `WORK/e5_run2`: `tier_budgets.csv`, `rules_table.csv`, `results.json`, `run_manifest.json`,
`artifact_hashes.json` (byte-identical across runs). Ledger prefix `e5.`; repository copies under
`scaleup/e_closed_loop/e5_run1/2/`.
