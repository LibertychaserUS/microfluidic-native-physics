# CLAIMS_REGISTRY_v0 — every defensible sub-conclusion in the result base, synthesised for the framework

Date 2026-10-22. Basis: four read-only extraction passes over manuscript v0.4.4 and every report in
`scaleup/` (files `claims_A1_stage1_D1.md` 70 items, `claims_A2_sealed_constructive.md` 78,
`claims_A3_multitable_hj_fidelity.md` 79, `claims_A4_stated_and_patterns.md` 90 stated-claim checks +
12 cross-cutting patterns + 10 gaps + 7 merge groups), all in this folder. Item ids below refer to those
files; every item there carries evidence location, numbers, ledger ids, tier, scope, caveats.

中文摘要:227 条子结论 + 90 条已写结论核对,综合成 13 条框架级结论(F1–F13)、按 Q1–Q4/P/M 的目录、
10 条空白(对应 E0–E6)、重写时必须处理的 11 处张力/措辞,以及新稿的章节—结论映射。证据等级计数:
密封预注册 25、预注册多表 26、冻结回顾性 96、事后 59、纯方法论 19、综合 2。

Evidence tiers: S = sealed pre-registered single reveal; M = pre-registered multi-table; R = frozen
retrospective; P = post hoc. "Tables" counts the public tables that carry evidence for the item.

---

## 1. Framework-level conclusions (F1–F13)

### F1 — Shift cost is stratified by shift type; the multiplier is not a constant (Q1)
Statement (allowed strength): in these six tables the error of a frozen surrogate relative to its own
grouped development reference rises in the order row-random ≈ exact-input < geometry ≈ cross-literature
< whole chemistry < device type / laboratory; the multiplier depends on how much heterogeneity the
development reference already contains (LNP compilation tiers: r 0.62–0.86 against a reference that
already spans 33 papers).
Evidence: A1-34, A1-35, A1-37, A1-50 (three UQ families reproduce the order), A1-11, A1-14 (three tree
models), A3-01–A3-08 (H-A falsified on six tiers: r 1.32/1.86 fluid, 5.03/4.95 device, 0.62–0.86 LNP),
A2-01 (TX100 19.84%), A2-24 (lab B 240.39% vs 39.41%). Tiers S, M, R. Tables 6.
Limits: MdAPE moves cross-literature above chemistry (A1-35); the A1/R2-A1 device tiers are compound
shifts (device + unseen fluids, A3-10); laboratory shift has one tier (D3).

### F2 — With zero target labels, no input-side statistic tells reliable rows from unreliable ones at candidate or query level; the only gains are source-level (Q1)
Evidence: A1-25, A1-27 (5-NN gain disappears under source–context permutation, p = 1.0, and retention
is all-or-nothing per source), A1-29, A1-31 (hydraulic-diameter control beats every score), A1-45, A1-47
(pre-registered candidate-level criterion 0/5 cross-literature), A1-64, A1-65 (batch-detectable ≠
row-rankable), A2-04 (TX100: strict min–max accepts all 87 rows), A2-05 (IF suggestive only, BH p 0.106),
A2-29, A2-30 (D3: 0 reliable rows; size control best ranker), A2-64 (AD filtering never raises
Success@K), A3-24 (H-D: six sealed tiers answer 0 queries), A3-53 (label-free diagnostics ρ −0.03 to
−0.13 with the few-shot gain). Tiers S, M, R. Tables 6.
Limits: in-distribution 5-NN is the best pool-level score (A1-46); the transductive domain classifier
reaches the geometry criterion in 12/30 seeds (A1-47); IF improves precision on D2 LOCO (A1-49).

### F3 — Under shift, interval coverage is recovered only by widening or abstaining (Q1/Q2)
Evidence: A1-15–A1-22 (CQR external 0.52–0.55; grouped calibration 0.848 at 3× width; Mondrian
0.970 at 78 vs 44 µm), A1-52 (DE→SE: no interval survives), A1-53 (GP native 0.959 but 49 µm vs 35 µm
median diameter), A1-61–A1-63 (weighted conformal: infinite intervals 1.000 in device/literature cells;
chemistry 0.848 with 0.326 infinite), A2-02, A2-26 (sealed coverages 0.448, 0.719), A2-20, A2-61
(certification answers 0–1 of 78 queries), A3-24, A3-28–A3-30 (device tiers 0.225/0.330; ensemble width
×1.1·10⁵ with coverage 1.000). Tiers S, M, R. Tables 6.
Limits: QRF native in chemistry LOCO 0.830 at 51 µm is the best coverage–width trade-off seen (A1-56);
conformalised ensemble external 0.906 at 84 µm (A1-55).

### F4 — Random-split accuracy and rankings do not predict shifted accuracy or rankings; grouped retrospective CV predicts chemistry shift but underestimates device and laboratory shift (Q1/M)
Evidence: A1-07 (5.27 → 10.97/11.26%), A1-10, A1-13 (row-random leaks 79% of DE pairs, 99.5% of test
geometries), A1-51 (D2: best two random models fall to 4th/5th under LOCO), A3-13 (S0 vs dev-ref 10.7 vs
21.8, 4.3 vs 13.6, 22.2 vs 40.9), A3-26, A3-27 (H-E: six ρ ≤ 0.5, GP best→worst on A1), A2-07, A2-11,
A1-43 (TX100 LOCO 20.75 vs sealed 19.84; macro 10.24 vs 10.27), A3-09 (LOGO 0.77/0.99/0.75 vs sealed
device 5.03/4.95), A2-24 (D3 dev-ref 39.41 vs 240.39). Tiers S, M, R. Tables 6.
Consequence for the framework: a retrospective grouped estimate can stand in for a sealed test only
when the development set already contains the same kind of heterogeneity (chemistry, LNP), not for
device or laboratory shift.

### F5 — Output constraints change admissibility, not accuracy; the constraint form must match the regime (P/Q4)
Evidence: A2-13–A2-16 (V2: invalid 0 but macro 12.05 vs 10.27; worst group narrowed), A2-25 (D3
raw-scale XGBoost: 22/57 negative diameters), A2-32–A2-34 (log target removes all negatives, 2.55× dev-ref
remains; log already cheaper inside lab A: 16.89 vs 39.41), A3-14–A3-17 (H-B B1 holds on R2-A1, R2-A2),
A3-15 (A1 device tier: log-MLP 1.66·10⁶%), A2-18, A2-21 (anchor model facts). Tiers S, M, R. Tables 5.
Tension to reconcile (see §5): R8 reads trees as "bounded by the training response range" (A1-39),
but boosted trees are not bounded in general and D3 shows negative outputs (A2-25); the honest
statement is observational per cell.

### F6 — A few target labels recover chemistry shift but not device or laboratory shift within 20 rows; the gain is usually positive but its size depends on the domain (Q2)
Evidence: A2-41–A2-44 (TX100: shift k=2 13.9%, scratch crosses 11.4% at k=10; lab B: nothing within
k ≤ 20), A2-47–A2-49 (warm-start TX100 8.93% at k=2, 4.21% at k=20; lab B never meets R1), A2-51–A2-54
(five new tiers: R2-A1 fluid pair recovers at k=2–3; device tier scratch best, no arm reaches dev-ref; LNP
tiers already below dev-ref, calibration only adds variance; T6 affine 5.4% at k=3), A2-68, A3-48, A3-49
(106 tiers: warm-start gain positive in 70.8/85.2/88.1%, median 1.53/3.15/4.60 pp; affine negative at
k=2; scratch ≈ coin flip at k=5), A3-59 (regret of never calibrating 11–16 pp vs always warm-start
1.6–2.6 pp), A2-77, A3-77. Tiers P (protocols frozen before runs), M (H-J). Tables 5 + 106 tiers.
Limits: generalisation rule (≥ 3 domains) not met for "warm-start beats scratch" (A2-54); laboratory
type has one tier; composition tiers at k=2 median −0.07 pp (A3-60).

### F7 — A diagnostic computed from the k measured rows predicts the size of the few-shot gain; label-free diagnostics do not; a threshold rule on it beats always-warm-start only with two labels (Q4)
Evidence: A3-50–A3-54 (bias ratio ρ 0.796/0.717/0.667, sign consistency 0.51/0.65/0.63, magnitude ratio
0.39/0.49/0.55, all BH-significant and same sign within every table; label-free ρ ≈ 0), A3-55–A3-58
(rule at k=2 regret 1.75 vs 1.97 vs 11.30; at k=5/10 always-warm-start regret 2.60/1.63 leaves no room;
pre-registered verdict "does not hold"), A2-69–A2-71. Tier M. Tables 5, 106 tiers.
Limits: the k=2 success rests on the stratified bootstrap interval; the table-block interval includes 0
(A3-55 caveat); the rule has two actions only (no scratch/escalate arm, no escalation cost) — gap G2.

### F8 — Where to measure first is the least answered question: label-free space-filling selection did not beat random, applicability filtering before retrieval never helped, and the retrieval portfolio helped in one unit only (Q3)
Evidence: A2-45 (k-means/farthest-point vs random: thresholds not met in either domain), A2-64, A3-25
(AD filtering never raises Success@K; rejects all 109 rows on R2-A1 device), A2-58–A2-63, A3-78
(portfolio: A1 fluid-family Success@3 0.30 → 0.70; three units null; two LNP tiers worse; cross-unit
mechanism test ρ = 0.00), A1-32 (replay: every screen trades coverage for precision), A2-06 (V1: six
strategies answer the same 8 of 13 queries). Tiers P, S. Tables 5.
Consequence: Q3 needs the closed-loop experiments (E1 acquisition axis) — gap G1.

### F9 — Failure has two layers: a support gap visible from inputs, and a conditional shift inside the marginal ranges that no input-side screen sees (Q1)
Evidence: A2-04 (TX100 inside all marginal ranges), A2-27, A2-28 (D3: 18/57 in support, in-support
median error still 0.884), A2-78, A1-40–A1-42 (device-type errors largest on sizes the training set
covers; DE→SE systematic over-prediction), A1-61, A1-64 (two independent input-side tools agree on the
support gap for DE rows), A3-10, A3-12 (R2-A1 device tier: systematic under-prediction by trees).
Tiers S, R, P. Tables 4.
Consequence: only the first layer can be handled label-free; the second needs target labels (F6) or
state variables that were not recorded (the H-H gate exited, A2-66).

### F10 — The evaluation key is itself a result: grouping choices change conclusions at four stages (M)
Evidence: A1-66 (split 5.27 → 11.69; calibration 0.521 → 0.848; threshold 26 → 32 retained; significance
p 0.020 → 1.0), A1-10, A1-12, A1-13, A1-18, A1-26, A1-67 (macro vs micro 13.21 vs 11.26), A1-68
(reference cell depends on fit share: 5.27 vs 6.71 — multipliers only comparable within a protocol),
A3-31, A3-32 (sealing count rule leaves R2-A1 with two fluid pairs → 2-fold dev-ref; r must be reported
with its grouping granularity). Tier R, M. Tables 4.

### F11 — Sealing adds exclusion of tuning and isolation of modelling choices, at a documented cost: rules must be fixed before any label is read, and defects surface after the reveal (M)
Evidence: A2-08 (V1 thresholds anchored in-distribution: a design flaw fixed in D3), A2-10 (reveal
incident: NaN serialisation; recovery without refit), A2-37–A2-40 (D3 register: signed-error defect in
one block, heavy tail noticed in a dry run, buffer–laboratory confounding, binding by file identity),
A2-55–A2-57 (round-4 reproduction asserts; H-D14 fit-partition mismatch; H-D15 vacuous threshold),
A2-65, A2-72–A2-76 (H-D6/7/9/10/11/12/13: zero-target rows, header leak, label source, encoding,
GroupKFold fold count, calibration fallback, mlp_log/mlp_raw identity — the last changed a protocol
decision), A3-31, A3-33–A3-38, A3-42, A3-61–A3-62, F-D1–F-D7. Tier M/method. Tables 6.
Consequence for a self-driving loop: the custodian/gate machinery is reusable as the loop's
measurement interface, and every rule that a loop applies must be frozen before the first measurement.

### F12 — Provenance: the surrogates are independent re-implementations whose D1 fidelity to the upstream code is established up to one library default; the deployed upstream model was trained on the literature rows; dataset admission needs hard filters (P)
Evidence: A1-01–A1-06, A3-65–A3-76 (M0 vs upstream 1.3.3: 0.021 pp; only `base_score`; deployed model =
default constructor + scaler on 932 rows, literature rows in-sample 0.313% vs our held-out 11.258%;
D2 B1 3.17% vs published 3.9% left open), A3-39–A3-46 (admission: readable row table as hard filter;
five exclusion classes; independence checks catch re-used deposits). Tier R, P. Tables 6.

### F13 — Model family matters under shift in a way random splits do not reveal: forests and quantile forests degrade least under chemistry shift, unbounded MLPs and ensembles extrapolate by orders of magnitude, and every family's native uncertainty is beaten by a size control in at least one shift cell (P/Q4)
Evidence: A1-37, A1-38, A1-50, A1-51, A1-53–A1-58, A3-15, A3-18, A3-26–A3-29. Tiers R, S, M. Tables 4.
Limits: ranking of source models inside a few-shot loop is untested (gap G6).

---

### F14–F19 — added by the offline closed-loop benchmark (E0–E3, E1b; full statements, numbers and grades in `scaleup/e_closed_loop/E_REPORT.md`)

- F14 (Q2, F): measuring before searching does not pay; the best loop proposes at once and warm-starts on every measurement; rests on the sparse tiers; robust to a halved tolerance and budget.
- F15 (Q3, F): label-free acquisition rules do not shorten the search by more than 0.1 measurement on all tiers; on dense tiers k-means seeding pays, stably under drift and difficulty.
- F16 (Q4, F): warm-start dominates log-affine, scratch and the threshold preview; the portfolio proposal is worse than greedy.
- F17 (robustness, F): under a ±20% oracle offset the adaptive ranking is unchanged and a perfect non-updating prior falls behind.
- F18 (Q2, F): within 20 rows warm-start returns the surrogate to 1.25× dev-ref in 94% chemistry / 90% laboratory / 91% composition tiers and 40% device tiers.
- F19 (Q4, F): after two measurements no threshold rule on d1–d4 beats always-continue once escalation has a price (λ = 5, 10), under rotation or in sample; headroom 0.54 measurements; the P12 preview is a loss (+0.17).
- Gaps closed: G1 (acquisition order; E1), G2 (decision rule; E3), G3 (k*(ε) by shift type; E2), G4 (simulator validity; E0), G5 (point-task metrics; E1). Open: G6 (E4), G7 (partly: E3 diagnostics vs cost), G8 (E5), G9, G10.
- Manuscript v0.5.0 restructures the paper around Q1–Q4 (`manuscript/RESTRUCTURE_PLAN_v0.5.0.md`); Section 7 (Synthesis) carries F1–F19 as `tab:findings` and the practitioner table `tab:loopguide`.

- F20 (Q2, transfer, F): a budget planned from the shift type alone transfers across tables (regret 2.09 [−0.48, 7.62] rows; saves 13.2 rows vs the full budget); refinements by density or diagnostics do not transfer. (E5)
- F21 (Q1/Q4, model choice, F): inside an adaptive loop no source family beats XGBoost; without adaptation forests are better by ~0.5–0.7 measurement; the random-split ranking of families does not predict the loop ranking (τ 0.08 [−0.04, 0.19]). (E4)
- F22 (robustness, F): of 20 "better" verdicts in E1–E4, four survive drift and shuffled-label controls — all P2 (start at once); k-means acquisition gains fail the shuffled control (pre-registered), dense-tier P2 and free-escalation gains are fragile. (E6)
- Gaps closed by E4–E6: G6 (E4), G8 (E5); negative-control summary (E6). Still open: G9, G10, acquisition on sparse tiers.

- F23 (Q2–Q4 on hard targets, F): on tail targets with τ = 3% (chance 11.35 measurements) the start-at-once warm-started loop needs 5.72; starting at once, no acquisition or interval-aware proposal gain, warm-start dominance and the negative decision-rule result (now with four-row diagnostics, headroom 1.37) all hold again — the Q3/Q4 nulls are not artefacts of easy tasks. (E7)

## 2. Catalogue by framework question (ids; strongest first)

| Question | Count | Strongest items | Framework conclusions |
|---|---|---|---|
| Q1 competence boundary | 84 | A2-01, A2-24, A3-01, A1-34, A2-04, A3-24, A3-26, A2-78 | F1, F2, F3, F4, F9, F13 |
| Q2 measurement budget | 31 | A2-47, A2-44, A3-48, A3-59, A1-17, A2-46 | F6, F3 |
| Q3 acquisition | 14 | A2-45, A2-64, A1-33, A2-27, A1-13 | F8, F9 |
| Q4 switching / decision | 46 | A3-50, A3-57, A3-58, A1-27, A1-29, A2-43, A2-58 | F7, F5, F6, F8 |
| P provenance and model choice | 44 | A3-65, A3-72, A1-50, A1-51, A2-16, A3-39 | F12, F13, F5 |
| M methodology | 71 | A1-66, A2-10, A3-31, A2-73, A3-69, A1-44 | F10, F11 |

(Items map to more than one question; counts are of mappings.)

---

## 3. Gaps the framework needs (from A4 §2.2, mapped to the experiment plan)

| Gap | What is missing | Nearest existing evidence | Planned |
|---|---|---|---|
| G1 | acquisition order of the first k rows; budget curves by acquisition rule | H-J random draws; round-3 k-means/farthest-point (artifacts only, A2-45) | E1 acquisition axis |
| G2 | decision rule with actions {continue, switch, escalate} and an escalation cost | H-J two-action rule (A3-55) | E3 |
| G3 | k*(ε) distributions by shift type with ε = 1.25 × dev-ref | H-J gains in pp; round-3 k points | E2 |
| G4 | simulator validity: query distortion, noise floor, drift, loop-level controls | exact-input vs random 5.74 vs 5.27 (A1-10); shuffled-label model control (A1-09) | E0 |
| G5 | point-task metrics: time-to-target, failed proposals, cumulative regret | single-step retrieval endpoints (A2-06, A2-58) | E1 |
| G6 | source-model ranking inside the loop vs random-split ranking | H-E zero-shot rankings (A3-26) | E4 |
| G7 | diagnostics ↔ r directly (H-J predicts the gain, not r) | A3-50 | E1/E3 side table |
| G8 | transfer of k*(ε) shapes and thresholds across tables | H-J leave-one-table-out for one diagnostic | E5 |
| G9 | provenance for D2 (3.17 vs 3.9) and for the three Stage 2 locked models | Appendix D (D1 only) | open item; E4 records model identity |
| G10 | laboratory shift has one tier | D3 lab B (A2-24), operator B n=12 (A2-36) | report with n = 1; seek a second laboratory table |

---

## 4. Section-to-conclusion map for the rewritten paper

| Section | Conclusions carried | Status |
|---|---|---|
| 1 Problem and framework (Q1–Q4) | definitions; F-list as the paper's claims | new |
| 2 Data, provenance, admission | F12 | existing text, re-ordered |
| 3 Evaluation methodology as a result | F10, F11 | existing (§4, §8, registers), condensed |
| 4 Q1 Competence boundary | F1, F2, F3, F4, F9, F13 | existing R1–R10 numbers, regrouped |
| 5 Q2 Measurement budget | F6, F3 (k-row recalibration A2-46) | existing round 3/4, H-J + E2 |
| 6 Q3 Acquisition | F8 + E1 | mostly new |
| 7 Q4 Switching and escalation | F7, F5 + E3 | existing H-J + new |
| 8 Closed-loop benchmark | E0–E6 | new |
| 9 What a self-driving loop needs | synthesis of F6–F8 with budgets | new |
| Limitations, Reproducibility, appendices | as now + E registers | existing |

---

## 5. Tensions and wording to fix in the rewrite (from A4 §1 and the A2/A3 caveats)

1. P9 tension: R8's "trees bounded by the training response range" (A1-39) vs D3 raw-scale XGBoost
   negative diameters (A2-25) — state the R8 observation per cell; boosting carries no bound.
2. R9 "never beats the uncalibrated model" refers to the R1 double condition (median and p95); by the
   median alone λ = 0.1 is slightly below baseline at k ≤ 10 and worse at k ≥ 15 (A2-48) — say so.
3. UNBACKED numbers in §6 (A4-52 kNN rejected 109/109; A4-61 ensemble width factors) — add to R10 text
   with ledger entries or cite the artifact explicitly.
4. A4-70, A4-82 (speculative sentences) — mark as hypotheses for the framework section or drop.
5. Over-strong: A4-12 add "in these tables"; A4-49 "requires either …" → "only the label arm was tested";
   A4-54 specify which models produced impossible outputs; A4-58 "every interval method" → add the two
   exceptions (Mondrian geometry 0.970, GP SE→DE 0.959, both by widening); A4-65 "cannot reconstruct" →
   "did not in these tables"; A4-76 drop "which is why"; A4-87 "sensible default" → "best in the chemistry
   cell only".
6. Under-claims to strengthen: A4-16 (hydraulic-diameter control beats every score, not merely matches);
   A4-72 (candidate-level tests were run in R8 geometry and R9 lipid strata and lost to the size control).
7. H-J k=2 rule: report both intervals (stratified excludes 0; table-block includes 0).
8. F1 wording: "device type / laboratory" largest, with the compound-shift caveat for A1/R2-A1 device
   tiers and n = 1 for laboratory.
9. "Reproduction" reserved for the upstream-as-is step (done in v0.4.4); V1/D3 endpoint names kept with
   the explanatory sentence.
10. Fidelity G0: the pre-registered verdict "not equivalent" is a rule defect; keep the verdict and the
    post-hoc diagnosis side by side (A3-69).
11. Framework §5 mapping: round-3 acquisition comparison and the kNN/width numbers are artifact-only;
    either promote them into the text with ledger entries or label them "artifact only".

---

## 6. Writing rules for strength

- S/M-tier conclusions may be stated as findings for the named tables ("in these six tables …").
- R-tier conclusions are stated with the protocol named and "retrospective".
- P-tier conclusions are stated as observations after the reveal, never as tests.
- Any sentence that generalises beyond the tables is a framework definition or a hypothesis for E0–E6,
  not a result. No absolute words; no paradigm/probe/wall/artefact vocabulary.

## v0.5.2 wording changes (2026-10-26)

Manuscript v0.5.2 rewords F4–F8, F12 and F14–F19 to the evidence after a review round (see `manuscript/RESPONSE_v0.5.2.md` §4):
F5 bounds remove invalid predictions and narrow the worst group but do not beat the published baseline; F6 separates the sealed sets
from the 107 tiers; F7 tier P → F and "short of its two-of-three criterion"; F8 "did not systematically help"; F12 cites the post-hoc
refit; F14 adds the sparse-tier and laboratory qualifiers; F15 notes the single pre-registered permutation; F16 adds the dense-tier
scratch result; F18 adds the unapplied noise-floor rule (E-D13); F19 states the headroom, not that the information exists.

v0.5.3 (2026-10-26): F20 adds "by the pre-registered criterion (its transfer regret interval includes zero)"; F1 punctuation only.
