# Frozen V2 protocol: native-variable physics-guided droplet prediction

Status: locked before V2 model fitting or outcome comparison  
Lock date: 2026-07-16  
Study type: retrospective, nested leave-one-chemistry-out analysis of public data  
Primary source: Chagot et al., *Lab on a Chip* (2022), DOI 10.1039/D2LC00416J

## 1. Terminology and claim boundary

The method is a **native-variable physics-guided constrained regressor**, not a
classical physics-informed neural network (PINN). A classical PINN evaluates
differential-equation, boundary-condition, or initial-condition residuals at
collocation points. The public table contains no velocity field, pressure field,
interface coordinates, time coordinates, or boundary-condition observations.

This study asks whether deterministic physical transformations, a semi-empirical
residual prior, and hard output constraints improve reliability under held-out
surfactant-chemistry groups. It does not prove a Navier--Stokes solution, stable
dripping, device compatibility, chemical compatibility, or wet-lab feasibility.
TX100 outcomes have already been revealed in V1; all V2 comparisons on TX100 are
post-reveal retrospective analyses and must never be called new blind validation.

## 2. Frozen research questions and decision rule

Primary question: does V2-P reduce equal-weight chemistry-group MAPE relative to
the published four-input XGBoost baseline B1 under nested leave-one-chemistry-out
(LOCO) evaluation?

V2-P may be described as beneficial only if all conditions hold:

1. chemistry-macro MAPE is lower than B1;
2. V2-P improves MAPE in at least four of the six main held-out groups;
3. worst-group MAPE is no more than 2 percentage points worse than B1;
4. point-prediction and interval physical-invalid rates are zero;
5. retrospective search reports query coverage and successful-target recall
   together with precision; and
6. no result is selected using the outer held-out outcomes.

Failure of any condition is retained as a negative or mixed result. No model is
renamed, removed, or retuned after outer outcomes are scored.

## 3. Data contract and anomaly quarantine

The raw public tables contain 468 rows with these required fields:

- continuous-phase flow rate, `Qc`, mL/min;
- dispersed-phase flow rate, `Qd`, mL/min;
- equilibrium interfacial tension, `gamma_eq`, mN/m;
- surfactant concentration, `c`, mM;
- critical micelle concentration, `CMC`, mM; and
- observed droplet diameter, `d`, mm.

Main-analysis chemistry groups are defined only by the recorded CMC value:

| Recorded CMC (mM) | Rows | Main-analysis status |
|---:|---:|---|
| 0.00 | 63 | included, no-surfactant group |
| 0.60 | 2 | quarantined anomaly |
| 2.00 | 30 | included |
| 3.50 | 87 | included; TX100, already revealed |
| 4.33 | 15 | included |
| 11.00 | 178 | included |
| 20.00 | 93 | included |

The published table lists five surfactants with nominal CMC values 4.3, 11.0,
20.0, 2.0, and 3.5 mM, plus no surfactant; it does not list a 0.60 mM group. The
two 0.60 mM rows are never reassigned by guesswork. They are excluded from the
466-row main analysis and evaluated only as a separately labeled sensitivity
case. Their source rows, values, and exclusion reason are exported.

All raw-file hashes, row counts, units, positivity constraints, and the identity
`concentration_over_cmc = c / CMC` (when CMC > 0) must pass before fitting.

## 4. Native input state

The only user- or search-supplied state is

\[
x_{native}=[Q_c,Q_d,\gamma_{eq},c,CMC,I_{no\ surfactant}].
\]

For the no-surfactant group, `c = CMC = 0` and the indicator is one. Flow rates
are converted to m3/s and interfacial tension to N/m before physics calculations.
Models receive training-fold-standardized transforms of the native values:

\[
[\log Q_c,\log Q_d,\log \gamma_{eq},\log(1+c),\log(1+CMC),I_{no\ surfactant}].
\]

Scaling parameters are fitted inside each outer training fold only. CMC is a
material property, not an optimizable process variable. A caller cannot submit
flow ratio, capillary number, or Reynolds number independently.

## 5. Frozen physical constants and deterministic physics layer

The source paper reports a fixed oval cross-junction of 195 um by 190 um and
fixed phases:

- continuous phase: `rho_c = 920 kg/m3`, `mu_c = 4.6e-3 Pa s`;
- dispersed phase: `rho_d = 1132 kg/m3`, `mu_d = 6.8e-3 Pa s`;
- primary diameter scale: `D = 195e-6 m`;
- primary oval area: `S = pi * 195e-6 * 190e-6 / 4`.

The paper's wording is internally awkward because it reports width x depth as
195 x 190 um but later normalizes the diameter using `D = 195 um`. The primary
analysis follows the published normalization. A locked sensitivity analysis uses
`D = 190 um`; a second area sensitivity uses `S = 195e-6 * 190e-6`. Neither
sensitivity participates in model selection.

The non-trainable physics layer recomputes

\[
r_Q=Q_d/Q_c,
\quad Ca_i=\mu_i Q_i/(\gamma S),
\quad Re_i=\rho_iQ_iD/(\mu_iS),
\]

and

\[
c^*=\begin{cases}c/CMC,&CMC>0\\0,&CMC=0.\end{cases}
\]

Its model-facing values are

\[
x_{phys}=[\log r_Q,\log Ca_c,\log Ca_d,\log Re_c,\log Re_d,\log(1+c^*)].
\]

Unit conversion, identities, positivity, and deterministic recomputation receive
dedicated tests. Derived quantities in an input file are ignored and recomputed.

## 6. Semi-empirical anchor and hard output parameterization

The locked Kalli/Chagot anchor is

\[
y_{phys}=d_{phys}/D
=0.642\,r_Q^{0.188}(1/Ca_c)^{0.161}.
\]

With `epsilon = 1e-4`, V2 predictions use

\[
z_{pred}=logit(clip(y_{phys},\epsilon,1-\epsilon))+\delta_\theta,
\]

\[
\hat d=D[\epsilon+(1-2\epsilon)\,sigmoid(z_{pred})].
\]

Therefore every V2-H/P/M point prediction satisfies `0 < d_hat < D`. The anchor
prior is disabled for a row when un-clipped `y_phys` is outside `(0,1)`; the row
remains in the data loss and is counted in an anchor-validity diagnostic.

This bound is specific to the spherical-droplet regime in the Chagot data and is
not transferred to DAFD, where normalized diameter may exceed one.

## 7. Frozen model and ablation matrix

| ID | Model | Purpose |
|---|---|---|
| B0 | deterministic Kalli/Chagot equation | physics-only baseline |
| B1 | four-input XGBoost: Qc, Qd, gamma, c/CMC | V1/published-feature baseline |
| B2 | native-input XGBoost | effect of retaining c and CMC separately |
| B3 | native plus physics-feature MLP, unconstrained output | physics-feature baseline and invalid-output audit |
| V2-H | B3 architecture with bounded sigmoid output | hard-bound ablation |
| V2-P | bounded residual MLP plus semi-empirical prior | primary V2 model |
| V2-M | V2-P plus local flow monotonicity penalty | sensitivity only |

B1 and B2 use 100 estimators, maximum depth 3, learning rate 0.3, squared-error
objective, and seeds `{17, 42, 2024, 3407, 9001}`. No outer-fold tuning occurs.

The neural models use two branches: native branch `6 -> 32 -> 32`, physics branch
`6 -> 16 -> 16`, SiLU activations, concatenation, `48 -> 16 -> 1`, AdamW with
learning rate `1e-3`, weight decay `1e-4`, full-batch updates, maximum 2000 epochs,
and early-stopping patience 150. Seeds are `{17, 42, 2024, 3407, 9001}`.

V2-P minimizes

\[
L=L_{Huber}(z_{obs},z_{pred})+\lambda_p\,mean(I_{anchor}\delta_\theta^2),
\]

where `z_obs` is the bounded-logit transform of observed `d/D`, Huber delta is
0.1 in logit space, and `lambda_p` is selected only from `{0.01, 0.1, 1.0}` by
inner LOCO chemistry-macro MAPE. Ties choose the larger lambda. The selected
epoch is the median best epoch across inner folds.

V2-M adds local finite-difference penalties for `d_hat(Qd*1.05) >= d_hat(Qd*0.95)`
and `d_hat(Qc*1.05) <= d_hat(Qc*0.95)`, but only when both perturbations remain
inside the outer-training native support. `lambda_m` is selected from `{0.01,
0.1}` by the same inner procedure. No monotonic rule is applied to gamma or c.

## 8. Nested validation and leakage controls

The six main CMC groups `{0, 2, 3.5, 4.33, 11, 20}` are outer LOCO folds. For
each outer fold:

1. the held-out chemistry labels are inaccessible to preprocessing, early
   stopping, hyperparameter selection, thresholds, and candidate ranking;
2. each remaining chemistry group serves once as the inner validation group;
3. all scalers and early-stopping decisions are fitted inside the inner training
   groups;
4. fixed choices are refitted on all outer-training rows for every locked seed;
5. outer predictions are written before outer metrics are computed; and
6. row IDs and chemistry-group manifests prove zero group overlap.

The primary estimand first averages seeds within each held-out group and then
weights the six chemistry groups equally. Row-micro metrics are secondary.

## 9. Metrics and uncertainty

Primary endpoints:

- chemistry-macro MAPE;
- worst-group MAPE;
- per-group MAPE and MdAPE; and
- number of groups in which each V2 model beats B1.

Secondary endpoints:

- MAE and RMSE in mm;
- row-micro MAPE;
- R-squared, labeled unstable for small or narrow-range groups;
- point and interval physical-invalid rates;
- anchor-invalid fraction and residual-to-anchor magnitude;
- local monotonic violation rate; and
- seed sensitivity.

Paired group-level model differences are reported with all six values. With only
six chemistry groups, bootstrap or permutation intervals are descriptive and no
asymptotic significance claim is made.

## 10. Bounded uncertainty bands

Any interval is formed in bounded-logit space and mapped through the same sigmoid,
so endpoints obey `0 < lower < upper < D`. Because only six chemistry groups are
available, 90% or 95% group-max conformal calibration is not finite-sample
attainable at the chemistry-group level. Therefore:

- row-calibrated split-conformal bands within outer-training chemistries may be
  reported as bounded uncertainty heuristics;
- their held-out-chemistry coverage is a transfer stress test, not a formal
  coverage guarantee; and
- group-simultaneous coverage and band width are shown without claiming
  exchangeability across new chemistry.

## 11. Retrospective constrained target search

The search compiler accepts only native controllable variables. Within a held-out
chemistry replay, CMC is fixed; gamma must come from an observed same-chemistry
concentration record or a locked within-chemistry interpolation; and concentration
is restricted to the observed grid. The search varies only Qc and Qd inside both
device limits and outer-training support, then recomputes all derived quantities.

Candidates are rejected for invalid units, nonpositive flow/tension, inconsistent
material fields, out-of-support native variables, or invalid derived values. The
model ranks only compiled candidates and abstains when uncertainty/support rules
fail. Targets are 0.06 to 0.18 mm in 0.01 mm increments, with top-k in `{1,3,5}`
and tolerances `{0.01 mm, 10%}`. Report success, answered-query precision, query
coverage, successful-target recall, candidate reduction, and oracle regret.

This is retrospective retrieval among historically observed conditions, not
prospective materials discovery or manufacturing validation.

## 12. Secondary DAFD device experiment

DAFD may be used only as a separately labeled partial-native-input experiment.
Its raw geometry/flow table provides `W,H,DIW,CIW,OW,Qc,Qd`, while the released
normalized table lacks row-level mu, rho, and gamma needed to reconstruct a full
native physicochemical state. Leave-one-geometry-out evaluation is permitted,
but its results cannot be pooled with the Chagot chemistry estimand or described
as a fully native PINN.

## 13. Reproducibility and release gate

Before any result is labeled VERIFIED, the release must contain:

1. raw data URLs, licenses, hashes, schema audit, and anomaly register;
2. this protocol hash recorded before model fitting;
3. source-code, configuration, environment, and split-manifest hashes;
4. per-row predictions for every model, fold, and seed;
5. inner-selection records and outer-fold leakage assertions;
6. all metrics and negative results, including every locked ablation;
7. tests for units, physical identities, hard bounds, label isolation, and strict
   JSON serialization;
8. two complete runs in separate directories with deterministic artifact hash
   comparison; and
9. an explicit statement that a genuinely new V2 blind validation requires a
   different, never-inspected dataset.

