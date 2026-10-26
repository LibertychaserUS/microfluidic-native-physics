# E6_NEGATIVE_CONTROLS_PROTOCOL_v1 (revision 1.1) — Negative controls and fragility summary for every closed-loop "better" verdict

Frozen on 2026-10-22 before any counted E6 run; sha256 in `E6_NEGATIVE_CONTROLS_PROTOCOL_v1.sha256`. E6 has one
new replay (the shuffled-label oracle for all 14 E1 policies) run by a wrapper over the frozen E1 runner
(`w7/closed_loop_e6.py`) and one assembly step (`w7/closed_loop_e6_analyze.py`) over the frozen outputs of
E1, E1b, E3 and E4; both hashed (`E6_RUNNER_SCRIPT.sha256`, `E6_ANALYZER_SCRIPT.sha256`) before the counted
runs, which are repeated in separate directories with byte-identical manifests. Every rule precedes the data.

中文摘要:E6 不问新问题,只问已得到的"更优"判定是否经得起三类负对照:(i)打乱标签的 oracle 上全部 14 条策略
相对参考策略都应不可区分(否则该策略的"优势"是回放的产物);(ii)解析的随机提议机会水平(不看模型、随机抽行
直到命中)与底线 P0、天花板 CEIL 并列,给每个偏移类型一个尺度;(iii)E1、E1b 各格、密档位子集、E3(λ=0)、
E4 中每一个"更优"判定在 ±20% 漂移下是否保持符号。判定:一个"更优"判定被"确认"当且仅当漂移下不脆弱且同一策略
在打乱标签下不是"更优";否则标为"脆弱"或"产物"。

## 1. Shuffled-label replay (new loops)

For every tier of E0, labels permuted within the tier (seed 0, as E1's control); all 14 E1 policies (P0–P12,
CEIL, the ceiling refitted on the permuted labels); three targets (the permuted labels nearest the original
targets, as E1); seeds 0–4; τ = 0.10, B = 20; δ = 0. Decision: Δ_shuffled(P) = mean over tiers of
[T*(P) − T*(P1)] under permutation, table-level block bootstrap (B = 2000, seed 0), Benjamini–Hochberg over
the 13 non-reference policies; a policy is an "artefact candidate" if it is "better" under permutation.
Revision 1.1 (same day, after the uncounted smoke run and before any counted run): the ceiling policy CEIL is
refitted on the permuted labels and is therefore label-informed by construction; under permutation it must be
"better" (the simulator responds to label knowledge) and it serves as the positive control of the replay, not as
an artefact candidate. Only the 12 source-model policies P0, P2–P12 are artefact candidates; CEIL's verdicts
enter the fragility ledger with the status "positive control".

## 2. Chance level (analytic)

For every tier and target, with n tier rows of which m lie within τ of the target, the expected
time-to-target of a proposer that draws unmeasured rows uniformly at random is
E[min(T, B + 1)] = Σ_{t=0}^{B} P(T > t), P(T > t) = C(n − m, t) / C(n, t) (hypergeometric; 0 when t > n − m).
Reported per shift type beside P0, P1, P2 and CEIL from E1 (mean over tiers and targets), as the scale against
which the loop gains are read.

## 3. Fragility ledger

Every "better" verdict (Benjamini–Hochberg) of E1 (all tiers and dense subset), of each E1b cell (all tiers
and dense subset), of E3 (rule vs continue at each λ) and of E4 (family vs xgboost under P2) is listed with:
its Δ and interval, its sign under δ = −0.2 and +0.2, its fragility flag, and, for E1 policies, its shuffled
verdict from Section 1. Status: "confirmed" iff not fragile and not an artefact candidate; "fragile" iff
fragile; "artefact" iff better under permutation. The count of confirmed / fragile / artefact verdicts is the
E6 result; no verdict is removed or upgraded by E6.

## 4. Outputs, integrity

`WORK/e6_run1`, `WORK/e6_run2`: `loops.csv`, `chance.csv`, `run_manifest.json`, `environment.json`,
`artifact_hashes.json`; the analyzer adds `results.json` and `fragility_ledger.csv`. Ledger prefix `e6.`;
repository copies without `loops.csv` under `scaleup/e_closed_loop/e6_run1/2/`. Compute ≈ 14 × 107 × 3 × 5 =
22.5 k loops. Smoke runs with reduced sizes are not counted.
