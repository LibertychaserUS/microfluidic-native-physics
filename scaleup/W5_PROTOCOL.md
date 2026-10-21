# Frozen W5 protocol: covariate-shift-weighted conformal intervals on the W3 cells

Status: locked before any W5 run (hash in `W5_PROTOCOL.md.sha256`; recorded in each run's `configuration.json`)
Lock date: 2026-10-17
Study type: retrospective, exploratory; same data, cells, seeds and partitions as `W3_PROTOCOL.md`
Design: `W5_DESIGN.md`

## 1. Question and claim boundary

W3 and W4 show nominal-0.90 intervals collapsing under device-type and whole-chemistry
shift. W5 asks whether that collapse is a covariate shift — inputs move, the conditional
law does not, so a conformal method reweighted by the input density ratio should recover
coverage (Tibshirani, Barber, Candès and Ramdas, 2019) — or a conditional shift or
support mismatch, which no input-only reweighting can repair. Nothing here is a blind
validation; the TX100 fold stays `post_reveal_retrospective`; no wet-lab or prospective
claim; no new tuning.

## 2. Data, cells, partitions, point model

Exactly as `W3_PROTOCOL.md` §2–§4 (seven cell families, seeds 0–14 / 0–29, single
splits, six LOCO folds; `split_fit_calibration` with the documented DE→SE fallback).
Point model: the W3 reference `xgboost` fitted on the fit partition. Residual score
R_i = |y_i − μ(x_i)| on the calibration partition, in model-response space; endpoints
mapped to micrometres by the dataset's monotone positive map.

## 3. Weights (inputs only; W3 scores reused unchanged)

- `unweighted`: w ≡ 1 — must reproduce the W3 `split_conformal` coverage cell by cell
  (built-in consistency check; any discrepancy is a defect, not a result).
- `w_domain`: w(x) = p̂/(1 − p̂), p̂ = P(test | x) from the W3 transductive domain
  classifier (class-balanced logistic regression on fit-partition inputs versus test
  inputs, as in `score_applicability`), p̂ clipped to [1e-6, 1 − 1e-6]. Constant
  factors cancel in the normalisation below.
- `w_knn`: w(x) = (r_fit(x) / r_test(x))^d · n_fit / n_test, where r_fit is the 5-NN
  distance from x to the fit inputs and r_test the 5-NN distance to the other test
  inputs (the W3 `density_ratio_knn` score is r_fit / r_test), d the number of input
  features (8 for D1, 4 for D2). A crude k-NN density-ratio estimate, deliberately
  unsmoothed.

Weights are evaluated on calibration rows and on test rows from the same fitted
scorer; no label enters.

## 4. Weighted split conformal (Tibshirani et al. 2019)

For test input x with calibration weights w_i and test weight w(x):
p_i = w_i / (Σ_j w_j + w(x)); the radius is the smallest R_(k) such that the cumulative
mass of calibration residuals up to R_(k) is at least 0.90; if the total calibration mass
Σ_i p_i is below 0.90 the radius is +∞. Interval: μ(x) ± radius. An infinite interval
counts as covering (it does) and is reported separately.

## 5. Endpoints per (cell, repeat, weight)

Empirical coverage (infinite intervals covering); `fraction_infinite`; coverage and
mean width in micrometres over finite intervals only; mean over test rows of the
effective sample size ratio ESS/n_cal with ESS = (Σ w_i)² / Σ w_i² over calibration
weights; mean of log10 w(x) over test rows. Means over repeats in the summary.

## 6. Pre-registered expectations (descriptive; no significance test)

- H-W5-0 (consistency): `unweighted` reproduces W3 `split_conformal` coverage in every
  cell to 1e-9.
- H-W5-1 (reference cells): weighting changes coverage by at most 0.03 (weights ≈ 1).
- H-W5-2 (SE→DE): at least one weighting raises coverage above the unweighted value;
  whether it reaches 0.80 is reported together with `fraction_infinite` and finite-only
  width.
- H-W5-3 (DE→SE): the support mismatch makes weights extreme (ESS/n_cal small); any
  coverage recovery comes mainly through infinite intervals, and finite-only coverage
  stays below 0.80.
- H-W5-4 (whole-chemistry LOCO): weighting does not reach 0.80 coverage in a majority of
  folds — the chemistry failure is not a covariate-shift problem (consistent with the
  TX100 finding that every row lay inside the marginal ranges).
- H-W5-5 (cross-literature): reported; no directional expectation.

## 7. Determinism and release gate

Fixed seeds; single-threaded; `n_jobs=1`. Two complete runs in separate empty
directories; `artifact_hashes.json` byte-identical and every file re-hashes
(`run_w3.compare_runs`). CSV floats `%.12g`; JSON through the strict writer, with
infinite radii stored as `null` in JSON and as the string `inf` in CSV. `--fast` runs are
labelled non-inferential and never summarised.
