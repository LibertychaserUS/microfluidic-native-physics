# Literature evidence matrix

Status: working evidence ledger for the manuscript. This file is not itself a
peer-reviewed source. Every manuscript claim must cite the linked primary source
and remain within the boundary recorded here.

## Chagot et al. (2022): fixed-device surfactant data

- Primary article: https://doi.org/10.1039/D2LC00416J
- Official HTML: https://pubs.rsc.org/en/content/articlehtml/2022/lc/d2lc00416j
- Public data record: https://zenodo.org/records/7055018
- Article license: CC BY 3.0 as stated by RSC.
- Dataset license: CC BY 4.0 as stated by Zenodo.
- Supported facts: the Methods and released 392 + 76 matrices support 468
  measured configurations; the Introduction contains an isolated 476-row
  statement. The fixed oval flow-focusing inlet is 195 x 190 um. Continuous and
  dispersed flow rates, equilibrium interfacial tension, and c/CMC are the four
  published XGBoost inputs. The source reports a 3.9% MAPE averaged over 50
  randomized runs with dimensionless d/D as its target. The deterministic
  comparison uses flow ratio and continuous-phase capillary number. The paper
  states that dynamic rather than equilibrium interfacial tension is preferred
  for universal flow-pattern maps and warns that equilibrium measurements based
  on a fixed interface may not represent the interface during droplet formation.
- Boundary: the source's randomized split does not establish transfer to an
  unseen surfactant chemistry. The present TX100 chemistry holdout is a new
  analysis of the released data. Its fixed-seed raw-mm XGBoost is a
  feature-aligned locked implementation, not a reproduction of the paper's
  50-run dimensionless-target estimate.

## Lashkaripour et al. (2024): DAFD 3.0

- Primary article: https://doi.org/10.1038/s41467-023-44068-3
- Official HTML: https://www.nature.com/articles/s41467-023-44068-3
- Supported facts: the comprehensive table contains 868 entries spanning
  single- and double-emulsion rows; the published system combines predictive
  models with a constrained search procedure; the article reports wet-lab
  testing for selected unseen-fluid designs, including unsuccessful
  recommendations. In its DE test, three of nine recommended flow-rate
  combinations did not form stable single-core double emulsions; stable cases
  are summarized separately by the source.
- Boundary: the present reanalysis does not reproduce those wet experiments. A
  project schema audit identifies 197 DE flow conditions represented by paired
  inner- and outer-diameter rows (394 rows) plus 474 SE rows in the 868-row
  workbook. This project treats each pair as one dependency cluster so one
  physical condition cannot cross partitions. The downloaded normalized
  workbook exposes capillary number and viscosity ratio but not the row-level
  raw viscosities, interfacial tensions, and densities needed to reconstruct a
  complete native physicochemical state. It is therefore a separate
  device-level stress study rather than evidence for the Chagot V2 model.

## Raissi et al. (2019): classical PINNs

- Primary article: https://doi.org/10.1016/j.jcp.2018.10.045
- Supported fact: classical physics-informed neural networks encode governing
  differential-equation residuals and associated conditions in the learning
  problem.
- Boundary: the present V2 model has deterministic dimensionless transforms,
  a semi-empirical anchor, hard output bounds, and a local monotonic penalty. It
  does not solve or enforce a governing PDE and must not be called a classical
  PINN.

## Romano et al. (2019): conformalized quantile regression

- Primary paper: https://proceedings.neurips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html
- Supported fact: conformal prediction can provide finite-sample marginal
  coverage under the required exchangeability conditions.
- Boundary: chemistry-held-out transfer violates the simple iid interpretation,
  and the V2 inner outcomes are reused for early stopping, model selection, and
  a residual-radius heuristic. V2 intervals are therefore bounded LOCO OOF
  uncertainty heuristics, not formal split-conformal guarantees.

## Tibshirani et al. (2019): conformal prediction under covariate shift

- Primary paper: https://proceedings.neurips.cc/paper_files/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html
- Supported fact: coverage under covariate shift requires additional structure;
  the paper studies weighted conformal prediction when an appropriate density
  ratio is known or accurately estimated.
- Boundary: the project neither establishes invariant P(Y|X) across surfactant
  chemistry nor estimates a justified test/train covariate likelihood ratio;
  weighted-conformal validity is therefore not claimed.

## Materials and self-driving-laboratory context

- Moosavi, Jablonka, and Smit (2020), JACS:
  https://doi.org/10.1021/jacs.0c09105
- Tom et al. (2024), Chemical Reviews:
  https://doi.org/10.1021/acs.chemrev.4c00055
- Supported context: Moosavi et al. motivate exploration of very large
  application-conditioned materials spaces and inverse design. Tom et al.
  describe self-driving-laboratory loops that connect data-driven planning,
  automated execution, measurement, and feedback. Treating this project as a
  constrained search problem is our synthesis of that context.
- Boundary: these sources motivate the long-term workflow. They do not turn a
  public-data simulation into experimental validation or a newly discovered
  material.

## Mandatory manuscript wording

- Use "sealed chemistry-held-out evaluation followed by a disclosed
  serialization-only recovery" for the TX100 episode.
- Use "post-reveal retrospective" for all later TX100 V2 analyses.
- Use "native-variable physics-guided constrained regressor" for V2.
- Use "bounded LOCO OOF uncertainty heuristic" for V2 intervals.
- Use "retrospective constrained retrieval" for target search.
- Never use "wet-lab validated", "new material discovered", "formal conformal
  coverage", or "classical PINN" for the present results.
