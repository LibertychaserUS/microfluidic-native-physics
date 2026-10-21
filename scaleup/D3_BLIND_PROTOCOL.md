# Prospective sealed cross-laboratory validation protocol — D3 (Smart Microfluidics, Zenodo 17867478)

Status: locked before the dataset is downloaded or opened (hash in `D3_BLIND_PROTOCOL.md.sha256`)
Lock date: 2026-10-17
Template: this file instantiates the rotating-sealed-holdout template of `../08_scaleup_experiment_plan.md` §4.1. Parameters written as ⟨…⟩ are not known before download; they are bound mechanically from input-only metadata into `D3_BINDING.json` (hashed and committed before the custodian runs), never from target values.
Data record: https://zenodo.org/records/17867478 (v1.0, 2025-12-09, CC BY 4.0, access `open`; related publication DOI 10.1016/j.ijpharm.2025.126362)

## 1. Purpose and blindness boundary

D1 (DAFD) and D2 (Chagot) cannot provide a new blind validation because every
label in them has been inspected. D3 is the first dataset this project has
never touched. Its cross-laboratory structure (laboratory A, Latina;
laboratory B, Rome) is the sealed dimension: **every target value of
laboratory B rows is sealed** before any model is fitted, calibrated,
thresholded or ranked. Laboratory A rows form the development set.

Before this lock, only the public Zenodo record page and the Zenodo API
metadata were read. They state: 304 cleaned microfluidic liposome formulations
(Micromixer chip only); inputs are formulation components (ESM, HSPC, CHOL,
PEG), operating conditions (total flow rate, flow-rate ratio, aqueous medium)
and hardware type; outputs are liposome size, polydispersity index (PDI) and a
binary formation indicator; data were collected by operator A on equipment A
and replicated in laboratory B in Rome; the zip holds five folders (`code`,
`data`, `data_extensions`, `metadata`, `raw_data`) plus a `README.pdf`. No
file content, no column name and no number from the dataset has been seen.

The task differs from D1/D2 (liposome size, not droplet diameter). D3 is
therefore a sealed test of a *microfluidic formulation surrogate* under
laboratory shift; its results are never pooled with the D1/D2 estimands.

## 2. Download, licence and provenance gate

1. Download the two record files into a fresh, empty directory outside any
   source tree; verify the record MD5 values (`28fcc5379016033c8f7a634a9eac0795`
   for the zip, `c33f49a3e24bd726a96a98ee307fd7db` for the README) and record
   SHA-256 digests in `source_registry_scaleup.json`.
2. Licence check: the record licence id must read `cc-by-4.0` (verified via the
   API before this lock); the README's licence statement, if any, is quoted in
   the registry. Attribution to the dataset creators and the related article is
   mandatory in every output.
3. Schema inventory is restricted to: the zip file list, the first line (header)
   of every tabular file, any data-dictionary or metadata files that describe
   columns, and the README's variable definitions. **No row of any file that
   contains a target column is printed or loaded for inspection** before the
   custodian has sealed laboratory B.

## 3. Parameter binding (`D3_BINDING.json`)

The binding file fixes, from inputs and metadata only:

- ⟨data_file⟩: the cleaned table used for analysis (expected: the 304-row file
  in `data/`), with its SHA-256;
- ⟨laboratory_column⟩ and ⟨laboratory_B_value⟩: how laboratory B rows are
  identified; if no input/metadata field identifies the laboratory, the study
  stops here and that is reported — there is no fallback that reads targets;
- ⟨input_columns⟩: ESM, HSPC, CHOL, PEG, total flow rate, flow-rate ratio,
  aqueous medium, hardware type, with units as documented;
- ⟨target_columns⟩: size, PDI, formation indicator (names only);
- ⟨cluster_column⟩: a formulation or replicate identifier if one exists, else
  row identity; it is the unit for calibration splitting and bootstraps;
- ⟨stratum_column⟩: the categorical input used for within-laboratory-B strata
  in the candidate-level test (default: aqueous medium);
- ⟨quarantine_rules⟩: rows whose hardware type is not Micromixer or that lack a
  required input are quarantined by row index (targets never written).

The binding is hashed (`D3_BINDING.json.sha256`) and committed before the
custodian program runs.

## 4. Custodian

A custodian program reads ⟨data_file⟩ once and writes three artifacts:

- `development_labeled.csv`: laboratory A rows, inputs and targets;
- `blind_features.csv`: laboratory B rows, inputs only, with an opaque row id
  (`uuid5` of file digest and row index);
- `sealed_blind_labels.csv`: laboratory B targets keyed by opaque id, mode 0600,
  kept outside the repository.

It prints only row counts and SHA-256 digests (manifest
`custodian_manifest.json`), never a target value or a target summary. The
manifest records the binding digest, the protocol digest and the raw-file
digests.

## 5. Locked predictive model

Point model for the primary target (size): XGBoost regression with 100
estimators, maximum depth 3, learning rate 0.3, squared-error objective,
random state 42 — the same configuration as the sealed V1 model, chosen before
seeing any D3 value. Inputs: the four lipid-component values as recorded,
total flow rate, flow-rate ratio, and one-hot aqueous medium with levels taken
from the development set (an unseen level maps to the all-zero vector and is
counted). Hardware type is constant after quarantine and is dropped. The model
is fitted once on all development rows; no hyper-parameter search, early
stopping or model selection may use sealed labels.

Secondary targets: PDI (same regressor) and formation indicator (XGBoost
classifier, 100/3/0.3, seed 42) are predicted and scored but do not enter the
primary criteria.

## 6. Locked uncertainty and applicability analyses

- Interval: 90% split conformal on absolute residuals, calibrated on a 20%
  calibration partition of the development set drawn by `GroupShuffleSplit`
  over ⟨cluster_column⟩ with random state 42; finite-sample order statistic;
  reported as observed coverage and width with **no guarantee asserted across
  laboratories**.
- Applicability scores fitted on development inputs, thresholds at the 95th
  percentile of calibration-partition scores: strict per-feature min–max;
  standardised 5-NN distance; OAS-shrinkage Mahalanobis; Isolation Forest;
  transductive domain classifier (development inputs versus laboratory B
  inputs; labelled as transductive). Controls: random score; predicted-size-only
  score. All rules and all zero-retention outcomes are reported.
- Candidate-level test: within laboratory B, strata are the levels of
  ⟨stratum_column⟩ with at least four rows; within-stratum AURC of each score
  must be lower than both controls, the within-stratum score permutation
  (2000 permutations) must give p < 0.05, and that p must survive
  Benjamini–Hochberg across the four non-control scores. If fewer than two
  strata qualify, the test is reported as not evaluable.

## 7. Pre-declared evaluation criteria

Primary endpoint: MAPE of predicted size on all sealed laboratory B rows.

Because no published held-out size error for this dataset was retrieved
before the lock, the criteria are relative to a development reference that is
computed **before reveal** from laboratory A only and frozen in
`D3_DEVELOPMENT_REFERENCE.json` (hashed): the mean MAPE of 5-fold grouped
cross-validation over ⟨cluster_column⟩ with the locked model.

- Reproduction success: sealed MAPE ≤ 1.25 × development reference MAPE.
- Strong replication: sealed MAPE ≤ 1.00 × development reference MAPE.

Also reported, without thresholds: fraction of sealed rows within 10% relative
error; signed relative bias; nominal-90% interval coverage and mean width;
strict min–max acceptance fraction (conditional-shift diagnostic); PDI MAPE
and formation accuracy/F1.

An applicability rule is called beneficial only under the four V1 conditions
(retained risk lower than full-pool risk; retention and reliable-candidate
recall reported; not weaker than strict min–max at comparable retention;
survives a matched-budget random-retention comparison) plus the §6
candidate-level test.

## 8. Reveal gate and audit trail

Laboratory B targets may be revealed exactly once, only after all of the
following exist and are hashed: this protocol and its sidecar; the binding and
its sidecar; the custodian manifest; the development reference file; frozen
prediction code and an environment manifest; the complete pre-reveal
prediction bundle (point predictions, interval endpoints, applicability scores
and masks, stratum labels) and its SHA-256. The reveal program joins by opaque
row id, scores every pre-declared endpoint, records UTC time and saves the
result regardless of sign. Serialization goes through the strict JSON writer
that converts undefined values to `null` and rejects infinities — the lesson
of the V1 incident is part of this template, so a write failure cannot
truncate a reveal. The unrevealed bundle is never overwritten. Any deviation is
logged in a numbered register and disclosed.

## 9. Claim boundary

A positive or negative D3 result speaks to one surrogate configuration under a
shift between two Italian laboratories on one microfluidic system. It does not
establish wet-lab feasibility, formulation quality, clinical relevance,
generalisation to other laboratories or instruments, or anything about droplet
microfluidics. The TX100-style wording discipline applies: the result is
reported as a sealed prediction with its full reveal chronology.
