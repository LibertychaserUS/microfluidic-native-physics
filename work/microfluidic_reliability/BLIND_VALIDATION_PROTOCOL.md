# Prospective sealed-label blind-validation protocol

Status: locked before downloading or opening the MAT artifacts  
Lock date: 2026-07-16  
Primary source: Chagot et al., Lab on a Chip (2022)  
DOI: 10.1039/D2LC00416J  
Data record: https://zenodo.org/records/7055018 (v1, CC BY 4.0)

## 1. Purpose and blindness boundary

The DAFD external outcomes have already been inspected and therefore cannot be
used as a new blind validation. This second public experimental dataset will be
handled prospectively: its target values will be sealed before model fitting,
risk scoring, interval construction, or target-search replay.

Only the published schema and aggregate metadata were inspected before this
lock. No MAT file has been downloaded or opened, and no target value from either
MAT file has been observed.

## 2. Published schema and physical representation

The Zenodo record defines five input columns:

1. continuous-phase flow rate, mL/min;
2. dispersed-phase flow rate, mL/min;
3. equilibrium interfacial tension, mN/m;
4. surfactant concentration, mM;
5. critical micelle concentration (CMC), mM.

The output is droplet diameter in mm. Following the paper, the model uses four
physics-structured predictors: continuous and dispersed flow rates, equilibrium
interfacial tension, and concentration divided by CMC. For the no-surfactant
group (published CMC = 0), the normalized concentration is defined as 0 rather
than dividing by zero.

## 3. Blind chemistry holdout

All rows from the two author-provided MAT files are combined before splitting.
The entire Triton X-100 (TX100) chemistry is the blind holdout, identified only
from the input CMC value 3.5 mM reported in the paper. Published aggregate
metadata state that this group contains 87 configurations. No TX100 target is
available to model-development code.

All other CMC groups form the development set. The custodian program may read
the raw target arrays only to write two separate artifacts:

- a labeled non-TX100 development table;
- an input-only TX100 blind table plus a separately sealed target artifact.

The custodian prints counts and cryptographic hashes, never target values or
target summaries.

## 4. Locked predictive model

The point model reproduces the paper's published XGBoost settings:

- 100 estimators;
- maximum depth 3;
- learning rate 0.3;
- random state 42;
- squared-error regression objective;
- otherwise library defaults recorded in the environment manifest.

The point model is fit once on all non-TX100 development rows. No blind-label
hyperparameter search, early stopping, or method selection is permitted.

## 5. Locked uncertainty and applicability analyses

### Prediction interval

A 90% split-conformal interval is fitted only on the development set. CMC is
the grouping variable. A deterministic GroupShuffleSplit with random state 42
places approximately 20% of development chemistry groups in calibration; all
remaining groups are proper training data. Absolute residual scores and the
finite-sample higher quantile are used.

### Applicability scores

The blind feature table receives the following scores fitted only on development
inputs:

- strict per-feature min--max inclusion;
- standardized 5-nearest-neighbor distance, thresholded at the 95th percentile
  of leave-one-out development scores;
- OAS-shrinkage Mahalanobis distance, thresholded at the 95th percentile;
- Isolation Forest score, thresholded at the 95th percentile.

All four methods and all zero-retention outcomes are reported. No blind result
is used to choose a winning rule.

## 6. Locked blind target-search replay

The TX100 input rows form one same-chemistry candidate pool. Requested target
diameters are fixed at 0.06, 0.07, ..., 0.18 mm from the paper's published
physical range (all measured droplets are below the 0.195 mm channel depth).

Strategies are point-only, strict min--max, kNN-95%, Mahalanobis-95%, Isolation
Forest-95%, and point-plus-conformal-target-inclusion. Candidate ranking uses
only predicted diameter and the locked screening mask. A query is answered only
if a candidate is predicted within 10% of its target. Results report success@1,
success@3, success@5, answered-query coverage, oracle-achievable-target recall,
candidate reduction, and oracle regret. Blind labels are used only after the
ranked candidate file is hashed.

## 7. Predeclared evaluation criteria

Primary point-prediction endpoint: MAPE on the full TX100 blind group.

- Reproduction success: blind MAPE below the published semi-empirical benchmark
  of 11.4%.
- Strong replication: blind MAPE at or below 6.0%, a tolerance around the
  paper's 3.9% in-distribution four-input XGBoost result.

These thresholds assess the point model only. They do not determine whether an
applicability rule is useful.

An applicability rule is described as beneficial only if, on relative error:

1. retained-candidate risk is lower than the full blind-pool risk;
2. both retention and reliable-candidate recall are reported;
3. the effect is not weaker than strict min--max at a comparable retention;
4. the result survives a matched-budget random-retention comparison.

The 90% conformal result is reported as observed coverage and width, with no
guarantee asserted for the unseen TX100 chemistry.

## 8. Reveal gate and audit trail

The blind target may be revealed exactly once, only after all of the following
exist:

1. this protocol and its SHA-256;
2. raw-file URLs, MD5 values from Zenodo, and local SHA-256 values;
3. custodian manifest and blind-feature SHA-256;
4. frozen code and environment manifest;
5. blind point predictions, interval endpoints, risk scores, screening masks,
   and ranked target-query candidates;
6. a SHA-256 for the complete pre-reveal prediction artifact.

The reveal program joins predictions and labels by an opaque row identifier,
scores every predeclared endpoint, records UTC reveal time, and saves the result
regardless of sign. The unrevealed prediction artifact is never overwritten.

