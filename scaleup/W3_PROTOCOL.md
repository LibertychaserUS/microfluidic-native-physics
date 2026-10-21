# Frozen W3 protocol: method-matrix reliability benchmark on in-hand public data

Status: locked before any W3 run (hash in `W3_PROTOCOL.md.sha256`; recorded in each run's `configuration.json`)
Lock date: 2026-10-16
Study type: retrospective, exploratory; no sealed holdout (every label in D1 and D2 has been inspected by earlier work)
Design: `W3_DESIGN.md` (same directory); plan: `../08_scaleup_experiment_plan.md` §2–§4

## 1. Claim boundary

This run evaluates how point surrogates, interval estimators and label-free
applicability screens behave under four shift types that the in-hand public
tables can express. It cannot provide a new blind validation: D1 (DAFD, OSF
938rs) rows were used for training and grouped evaluation in the expanded
suite, and all 468 D2 (Chagot, Zenodo 7055018) labels were used as V1
development data or as V2 outer LOCO folds. The TX100 fold is labelled
`post_reveal_retrospective`. Nothing here is wet-lab validation, prospective
discovery, or a universal applicability domain. Cross-laboratory shift (D3)
and real-versus-simulation shift (D5) are not available in hand and are left
empty in the matrix rather than approximated.

## 2. Data

- D1 comprehensive table (868 rows: 474 SE, 197 DE conditions as 394 paired
  rows) and external table (64 rows, four literature sources), loaded by the
  reproduction loader with the four workbook SHA-256 digests checked against
  `SOURCE_MANIFEST.md`.
- D2 main analysis table (466 rows, six CMC groups 0/2/3.5/4.33/11/20 mM),
  loaded through the custodian-lock-verified join; the two CMC 0.60 mM rows
  stay quarantined.
- Physical target in micrometres for both datasets (D2 mm × 1000). Model
  response: D1 normalised diameter (published contract, converted with the
  hydraulic diameter); D2 diameter in mm (sealed-V1/B1 contract).
- Features: D1 the eight published inputs; D2 the four published inputs
  (Qc, Qd, equilibrium interfacial tension, c/CMC). No other feature set.
- Size-only control proxy: D1 hydraulic diameter; D2 the deterministic
  Kalli/Chagot anchor diameter (label-free).

## 3. Cells (shift type × dataset)

| cell | train → test | repeats | shift type | test group key |
|---|---|---|---|---|
| D1_random | row-random 80/20 of 868 | seeds 0–14 | reference | geometry dependency group |
| D1_geometry | dependency-group (geometry + DE pair) 80/20 | seeds 0–29 | device geometry | geometry dependency group |
| D1_external | all 868 → 64 external | 1 | cross-literature | literature source |
| D1_se_to_de | 474 SE → 394 DE | 1 | device type | geometry dependency group |
| D1_de_to_se | 394 DE → 474 SE | 1 | device type | geometry dependency group |
| D2_random | row-random 80/20 of 466 | seeds 0–14 | reference | chemistry group |
| D2_chemistry_loco | leave one CMC group out | 6 folds | whole chemistry | chemistry group |

Every cell asserts zero overlap of the grouping unit between train and test
where the protocol is grouped, and records train/calibration/test sizes.

## 4. Calibration partition inside the training rows

- D1: `GroupShuffleSplit(test_size=0.20, random_state=seed)` on dependency
  groups → fit and calibration partitions; zero group overlap asserted.
- D2: row-level 20% calibration (`random_state=seed`); Mondrian calibration
  aggregates calibration scores by chemistry group (≤5 units), so a 90%
  finite-sample threshold is not attainable and is reported as such.
- Interval calibration and applicability thresholds share this partition.

## 5. Method matrix

Point models: `xgboost` (D1 published configuration; D2 depth-3 B1
configuration), `random_forest` (400 trees, min leaf 2), `mlp`
(standardised inputs; hidden layers 64-64; ReLU; Adam, learning rate 1e-3;
early stopping on a 15% internal validation split; `n_iter_no_change` 30;
`max_iter` 2000; D2 response standardised). TabPFN is excluded because its
weights require a network download that this run does not perform.

Interval methods at nominal 0.90: `split_conformal`; `cqr` (gradient-boosted
quantile regressors 0.05/0.95, 300 trees, depth 3, learning rate 0.03, min
leaf 5); `mondrian_group_conformal` (absolute-residual scores aggregated by
group maximum over the calibration partition's test-group key, finite-sample
order statistic); `ensemble_variance` (20-member cluster-bootstrap refits of
the point model give a per-row standard deviation; radius = q × max(sd,
floor) with q the finite-sample 0.90 quantile of |residual|/sd on the
calibration partition). Endpoints are reported in micrometres.

Applicability scores (fit-partition inputs plus test inputs; thresholds from
calibration-partition scores): `knn_5`, `mahalanobis_oas`, `isolation_forest`
(expanded-suite hyperparameters), `domain_classifier` (balanced logistic
regression, fit inputs versus test inputs, score = P(test); transductive,
label-free), `density_ratio_knn` (5-NN distance to the fit set divided by
5-NN distance to the other test inputs). Controls: `random_score`,
`size_only` (|size proxy − fit median| / fit IQR).

Binary rules: (a) global threshold at the 0.95 calibration quantile;
(b) within-group median (retain the lower-scoring half of every test group).

## 6. Endpoints

- Point: MAPE, MAE (µm), R² (flagged unstable when a group has fewer than 15
  rows); equal-weight macro over test groups.
- Interval: empirical coverage, mean width (µm), simultaneous group coverage,
  attainability flag and ceiling.
- Applicability: Spearman ρ(score, relative error) overall and mean within
  groups; tie-invariant AURC overall; within-group AURC (groups with ≥4 rows,
  equal weight); precision, recall and retention for rules (a) and (b) at 10%
  relative error and at 10 µm absolute error.

## 7. Pre-registered candidate-level decision rule

An applicability method shows candidate-level (within-group) benefit in a
cell only if all of the following hold:
1. its within-group AURC is lower than both the `random_score` and the
   `size_only` within-group AURC in that cell;
2. the within-group score permutation test (scores permuted inside each test
   group, 2000 permutations, one-sided) gives p < 0.05;
3. that p-value survives Benjamini–Hochberg adjustment across the five
   applicability methods of the same cell.
Cells with a single test group (LOCO folds, SE→DE/DE→SE) report the
within-group quantities as null with reason `single_test_group`; the rule is
then not evaluable there. Failure of any condition is reported as "no
candidate-level benefit".

## 8. Matrix summary

For each cell: MAPE doubling factor (cell xgboost MAPE ÷ same-dataset
reference-cell xgboost MAPE); nominal-90% coverage gap for `split_conformal`
and `cqr`; best within-group AURC reduction relative to `random_score`;
`knn_5` rule (a) precision/recall pair at 10 µm and at 10% relative error;
candidate-level verdict per method. Empty cells for shift types not in hand.

## 9. Determinism and release gate

Fixed seeds; single-threaded numerical libraries; `n_jobs=1`. Two complete
runs in separate empty directories; `artifact_hashes.json` must be
byte-identical and every file must re-hash to its entry. CSV written with
`%.12g` floats and `\n` line endings; JSON through the strict writer that
rejects non-finite values. `--fast` runs (2 seeds, 200 permutations) are
labelled non-inferential and never summarised as results.
