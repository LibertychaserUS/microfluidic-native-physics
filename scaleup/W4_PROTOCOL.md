# Frozen W4 protocol: uncertainty-aware comparison models on the W3 cells

Status: locked before any W4 run (hash in `W4_PROTOCOL.md.sha256`; recorded in each run's `configuration.json`)
Lock date: 2026-10-17
Study type: retrospective, exploratory; no sealed holdout (identical data and cells to `W3_PROTOCOL.md`)
Design: `W4_DESIGN.md`. Trigger: reviewer-style concern that the W3 interval findings hold only for intervals built around XGBoost.

## 1. Claim boundary

W4 asks one question: do the W3 findings about intervals under shift — nominal-0.90
coverage collapsing under device-type and whole-chemistry shift, near-nominal coverage
only on the random reference cells — depend on the point model being XGBoost? Three
model families that carry their own predictive uncertainty are run on the same seven
cells, seeds and fit/calibration partitions as W3. Nothing here is a blind validation
(the TX100 fold stays `post_reveal_retrospective`), nothing is tuned per cell, and no
candidate-level permutation test is run (the W3 §7 rule is evaluated only in its
descriptive first condition). No wet-lab or prospective claim.

## 2. Data, cells and partitions

Exactly as `W3_PROTOCOL.md` §2–§4: D1 comprehensive/external tables and D2 main
table loaded through the same digest-checked loaders; the seven cell families with
seeds 0–14 (random), 0–29 (geometry), single splits (external, SE→DE, DE→SE) and six
LOCO folds; the same `split_fit_calibration` rule (dependency group for D1, row for
D2, with the documented DE→SE fallback to rows). Features, responses, size proxies and
the micrometre map are the W3 `DatasetSpec` objects.

## 3. Models (fixed; never tuned per cell)

| model | point prediction | native 0.90 interval | native uncertainty |
|---|---|---|---|
| `qrf` | mean of `RandomForestRegressor(n_estimators=400, min_samples_leaf=5, random_state=seed)` | Meinshausen quantile forest: training-row weights 1/(leaf size) per tree averaged over trees, weighted empirical quantiles at 0.05 and 0.95 | q0.95 − q0.05 |
| `gp` | posterior mean of `GaussianProcessRegressor` on standardised inputs, kernel `Constant × RBF(ARD, bounds 1e-2..1e2) + White(1e-2, bounds 1e-6..1e1)`, `normalize_y=True`, `n_restarts_optimizer=2`, `random_state=seed` | mean ± 1.6449 σ, σ the predictive standard deviation including the white-noise term | σ |
| `deep_ensemble` | mean of five `MLPRegressor(64-64, ReLU, Adam 1e-3, early stopping on 15%, patience 30, max_iter 2000)` members with `random_state = 1000·seed + m`, each fitted on the full fit partition (D2 response standardised as in W3) | mean ± 1.6449 sd, sd over members with ddof 1 | sd |

The deep ensemble has no heteroscedastic likelihood head: it is the mean–variance of
independently initialised point networks, and is labelled as such.

## 4. Intervals at nominal 0.90 (calibration partition only)

- `native`: the model's own interval above.
- `split_conformal`: centre = point prediction; radius = finite-sample 0.90 quantile of |calibration residual| (W3 definition).
- `conformalised_native`: δ = finite-sample 0.90 quantile of the conformity score max(lo − y, y − hi) on the calibration partition; interval [lo − δ, hi + δ]. For `qrf` this is conformalised quantile regression; for `gp`/`deep_ensemble` it is an additive correction of ±1.6449σ.

All calibration is in model-response space; endpoints map to micrometres through
the dataset's monotone positive map (coverage preserved).

## 5. Uncertainty as a selective-prediction score

`uq_rel` = native uncertainty / max(|point prediction|, 1e-12) and `uq_abs` = native
uncertainty, both in response space, are evaluated as scores for ranking test rows by
expected relative error, with the tie-aware AURC and Spearman ρ against |relative
error|. Comparators in the same cell, computed exactly as in W3 (`score_applicability`,
same seed offsets and thresholds): `random_score`, `size_only`, `knn_5`,
`mahalanobis_oas`, `isolation_forest`, `domain_classifier`, `density_ratio_knn`.

## 6. Endpoints

Point: MAPE, MAE (µm), R² (flagged unstable below 15 rows), signed relative bias,
equal-weight group macro (W3 `point_rows`). Interval: empirical coverage, coverage
gap, mean/median width (µm), mean relative width, simultaneous group coverage (W3
`interval_row`). Score: overall AURC and Spearman per score; `uq_rel_beats_random`,
`uq_rel_beats_size_only`, `uq_rel_beats_best_ad` booleans per (cell, model).

## 7. Pre-registered reading rules (descriptive; no significance claim)

1. **Native interval survives a cell** if its empirical coverage is ≥ 0.80 (ten points
   below nominal). Expectation H-W4-1: no model's native interval survives either
   device-type cell; expectation H-W4-2: on the two reference cells every
   conformalised interval has |coverage gap| ≤ 0.05 while native intervals may not.
2. **Conformalisation restores nominal coverage on the reference cells but not on
   shift cells**: report the gap per cell and method; no threshold beyond rule 1.
3. **Uncertainty as score is useful in a cell** only if `uq_rel` has lower AURC than
   both `random_score` and `size_only` there (W3 §7 condition 1). Expectation
   H-W4-3: this holds on the reference cells and fails on at least one of the
   cross-literature and device-type cells for every model. Without a permutation
   test this is a description, not a candidate-level claim.
4. Point-error ratios to the same model's reference cell are reported next to the
   XGBoost ratios of W3; the ordering of the five shift cells is compared, not
   the magnitudes.

## 8. Matrix summary

For each cell family and model: mean MAPE over repeats and its ratio to the same
model's reference cell; mean coverage and width for the three interval methods;
AURC of `uq_rel`, `random_score`, `size_only` and the best W3 applicability score;
the three booleans of §6; the §7 rule-1 and rule-3 verdicts.

## 9. Determinism and release gate

Fixed seeds; single-threaded numerical libraries; `n_jobs=1`; GP optimiser is
L-BFGS-B with deterministic restarts from `random_state`. Two complete runs in
separate empty directories; `artifact_hashes.json` must be byte-identical and every
file must re-hash to its entry (`run_w3.compare_runs`). CSV floats `%.12g`, JSON
through the strict writer. `--fast` runs (2 seeds) are labelled non-inferential and
never summarised as results. GP convergence warnings are recorded in
`deviations.json` and do not stop the run.
