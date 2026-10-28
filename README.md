<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Microfluidic Native Physics — code and artifacts for the preprint

Code, frozen protocols, deviation registers, number ledger, hash manifests and
summary artifacts behind the preprint

> Zhiyuan Zhang, *When Can a Learned Design Surrogate Be Trusted in a
> Self-Driving Microfluidics Laboratory? An Offline Closed-Loop Framework Built
> and Tested on Six Public Tables* (2026). arXiv:2610.36395,
> <https://doi.org/10.48550/arXiv.2610.36395>.

The work is retrospective and built on public tables only. Stage 1 is a
reliability audit and mechanism diagnostics on three tables (a sealed
chemistry holdout V1, a frozen physics-guided constrained regressor V2, an
expanded shift audit on the DAFD tables, the method matrix W3–W5 and a sealed
cross-laboratory validation D3); Stage 2 a pre-registered multi-table test of
hypotheses H-A to H-E on three further public tables (R10); Stage 3
constructive adaptation experiments under frozen protocols; Stage 4 an offline
closed-loop benchmark (E0–E7) that replays the six tables as a laboratory
would query them. Most pre-registered results are negative and are reported
as such. Nothing here is evidence of wet-lab feasibility of any device,
material or formulation.

## Release status

- Tag `v0.5.5-preprint` accompanies manuscript v0.5.5 (the version named in `CITATION.cff`).
- The preprint is arXiv:2610.36395 (v1, submitted 26 October 2026, primary
  category cs.LG, licence CC BY 4.0; DOI 10.48550/arXiv.2610.36395). The
  source uploaded to arXiv is byte-identical to `manuscript/main.tex` and
  `manuscript/main.bbl` at this tag.
  The code, protocols, registers, ledger and hash manifests promised in the
  manuscript's *Reproducibility* section are in this repository; the compiled
  PDFs (English and Chinese mirror) are attached to the GitHub release.
- Every SHA-256 digest cited in the manuscript (Appendix D) is listed in
  [`hashes/appendix_d_registry.json`](hashes/appendix_d_registry.json) with the
  path of the released file or the reason the bytes are not redistributed.
  `python3 scripts/verify_hash_registry.py` re-hashes the released files.
- Every number printed in the manuscript is recomputed from the released
  artifact files by the ledger script (standard library only):

  ```bash
  python3 manuscript/tools/check_numbers.py --work work/microfluidic_reliability --skip-runpair
  python3 scripts/verify_run_manifests.py
  python3 scripts/verify_hash_registry.py
  python3 scripts/release_manifest.py verify --root . --manifest hashes/release_manifest.json
  ```

  `--skip-runpair` only skips the re-hashing of the per-row files that are not
  in this repository; `verify_run_manifests.py` performs the same check on the
  released part of each run pair and reports the manifests as byte-identical.

## What is in the repository

```text
work/microfluidic_reliability/   the experiment workspace, path-for-path as cited in the manuscript (Appendix C)
  src/                           E0 reproduction, reliability and interval audits, expanded DAFD suite,
                                 V1 custodian / prediction / reveal / recovery / post-hoc programs, V2 implementation
  tests/                         unit, leakage and strict-JSON tests of the audited suite
  *_PROTOCOL.md, *.sha256        frozen protocols (expanded suite, V1, V2) and the V1 pre-registration lock
  PROTOCOL_DEVIATION_REGISTER.md deviation register D01–D19 of the expanded suite
  BLIND_REVEAL_INCIDENT.md       V1 reveal incident record and JSON_SERIALIZATION_POSTMORTEM.md
  artifacts/                     E0 / reliability summary artifacts and frozen figures
  blind_artifacts/               V1 pre-reveal bundle, consumed gate, recovered result, corrected post-hoc analysis
  d3_blind_artifacts/            D3 custodian manifest, pre-reveal bundle, reveal result, post-reveal correction
  *_artifacts_canonical_run{1,2}/ the five canonical run pairs: full hash manifests, configuration, environment,
                                 deviations and every summary artifact (per-row files > 1 MB stay in the workspace)
  posthoc_review_round2/         the four post-hoc analyses of Section "Post-hoc analyses"
  e0_run1/ … e7_run1/, e1b_run1/, e1_reanalysis/, e1_dense_sensitivity/, e6_posthoc/, e_tie_check/, e_floor_check/,
  e4_identity/, figures_closed_loop/   closed-loop benchmark E0–E7: run manifests, environments, results, summary
                                 tables (per-loop tables stay in the workspace); second runs keep manifests only
  h_A1_*/, h_R2A1_*/, h_R2A2_*/  R10 sealed validations: custodian manifests, development diagnostics, run summaries,
                                 reveal results and consumed gates (no per-row file)
  h_portfolio*_run1/, h_fewshot_rule_run1/, h_fewshot_rule_posthoc/, posthoc_review_round3_run1/, posthoc_review_round4*/
                                 Stage 3 constructive experiments and post-hoc rounds 3–4 (summaries and manifests)
  fidelity_run1/, fidelity_posthoc/   reproduction fidelity of the XGBoost baselines (results, pins, environments)
scaleup/                         W3/W4/W5 protocols and runners, D3 protocol, binding, deviation register and programs;
  validation_plan/               frozen protocols, bindings, digest sidecars and deviation registers of Stages 2–4
                                 (H-A–H-E, H-F, H-G, H-J, closed loop E0–E7, post-hoc round 3)
  w6/, w7/, fidelity/            R10 custodian/runner/reveal/combination programs; closed-loop runner, analyzer and
                                 wrappers; fidelity driver, worker and post-hoc script
  e_closed_loop/, h_A1/, h_R2A1/, h_R2A2/, h_combined/, h_portfolio/, h_fewshot_rule/, posthoc_round3/,
  posthoc_round4/, fidelity/run{1,2}/, h_intake/, synthesis/
                                 byte copies of the manifests, summaries, reveal results, reports and registers
                                 exactly where the manuscript's Section 8 names them (the same bytes as under work/)
manuscript/                      LaTeX source (English and Chinese mirror), bibliography, figures, numbers_ledger.json,
                                 tools/check_numbers.py (ledger script) and tools/render_figures.py
hashes/                          appendix_d_registry.json and release_manifest.json (SHA-256 of every released file)
scripts/                         release tooling: manifests, run comparison, policy and strict-JSON validators,
                                 public-data fetcher, hash-registry and run-manifest verifiers
data/source_registry.json        authoritative URLs, sizes and digests of the three public datasets
requirements/                    pinned analysis environment (the one that produced the artifacts) and CI subset
```

The workspace keeps its original layout (`work/microfluidic_reliability/`,
`work/data/`) because the frozen programs and tests resolve their paths from
it and because the manuscript cites artifact paths relative to that root.

## What is deliberately not in the repository

| Item | Why | Where the digest is |
|---|---|---|
| Raw third-party tables (OSF 938rs workbooks in `work/data/`, Zenodo 7055018 `.mat` files in `work/microfluidic_reliability/blind_raw/`, Zenodo 17867478 archive in `work/data/`) | public under their own licences; not redistributed | `data/source_registry.json`, Appendix D registry |
| Sealed label files of V1 and D3 | rows of public datasets that the released custodian programs reconstruct deterministically; never released | custodian manifests, Appendix D registry |
| Re-cut development / feature tables of V1 and D3 | same reason; the custodians regenerate them byte-for-byte (required before running V2, W3–W5 or the post-hoc analyses) | custodian manifests |
| Per-row artifacts larger than 1 MB (predictions, split manifests, interval rows; about 26 MB per matrix run) | kept in the authors' workspace | each run's `artifact_hashes.json` |
| Per-loop tables of the closed loop (`loops.csv`, `budget.csv`, shuffled-label loop tables), the per-row files of the R10 runs (sealed prediction bundles, scored rows, point predictions, interval rows), the fidelity per-row predictions, split manifest and worker task files, and the post-hoc warm-start curves | regenerated byte for byte by the frozen runners; third-party rows | each run's `artifact_hashes.json`; sidecars for the sealed bundles |
| Stage 2 source tables (A1 ESI, Wang et al. deposit, LNP Atlas) and the custodian re-cuts of them; the upstream DAFD code and model used by the fidelity protocol | public under their own licences, or without a licence (upstream code); not redistributed | `data/source_registry.json`, custodian manifests, `fidelity_run1/upstream_pins.json` |
| E0 XGBoost model binary | regenerated by re-running E0 | Appendix D registry |
| Superseded fast/smoke runs, pre-fix outputs, the uncorrected V1 post-hoc directory | excluded from the release inventory by the protocols | — |

Edited before release: `blind_artifacts/posthoc_analysis_corrected/blind_posthoc_summary.json`
carried fourteen absolute workspace paths as keys of its read-only audit block,
and each of the six header-only intake manifests under `scaleup/h_intake/`
carried one absolute path in its `directory` field; the prefixes were replaced
by `<WORK>/`. No value changed; original and released digests are in the
registry. Internal working documents (Chinese design notes and reports, review
responses, candidate lists, drafts) are not part of the release; every frozen
protocol, binding, register and sidecar that the manuscript cites is.

## Reproducing the runs

### Environment

The artifacts were produced with Python 3.12 on macOS (arm64) with the pinned
versions in [`requirements/analysis.in`](requirements/analysis.in)
(numpy 2.5.1, pandas 3.0.3, scipy 1.18.0, scikit-learn 1.9.0, XGBoost 3.3.0,
PyTorch 2.13.0, matplotlib 3.11.0, seaborn 0.13.2, openpyxl 3.1.5), single-threaded
numerical libraries and `n_jobs=1`. Each run writes its own `environment.json`.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements/analysis.in
```

### Data

Download the three public datasets into `work/data/` as described in
[`work/data/README.md`](work/data/README.md), verify them against
`data/source_registry.json`, then run the two custodian programs once: they
regenerate, byte-for-byte, the re-cut tables that V2, W3–W5 and the post-hoc
analyses read through the custodian locks. Nothing in `work/data/` and none of
the regenerated tables is tracked by git.

### Commands

All commands run from the repository root; `<WORK>` is `work/microfluidic_reliability`.

| Suite | Command | Output compared with |
|---|---|---|
| Unit tests (audited suite) | `python -m unittest discover -s <WORK>/tests` | — |
| Unit tests (W3–W5) | `cd scaleup && MICROFLUIDIC_WORK=../<WORK> python -m unittest w3.tests.test_w3 w3.tests.test_w4 w3.tests.test_w5` | — |
| E0 reproduction | `python <WORK>/src/dafd_reproduction.py` | `<WORK>/artifacts/reproduction_metrics.json` |
| Reliability / interval audits | `python <WORK>/src/reliability_analysis.py`, `python <WORK>/src/interval_experiments.py` | `<WORK>/artifacts/reliability_metrics.json` |
| Expanded DAFD suite (two runs) | `python <WORK>/src/expanded_study.py --output-dir <WORK>/expanded_artifacts_canonical_run1` (and `run2`) | `expanded_artifacts_canonical_run{1,2}/artifact_hashes.json` |
| V2 frozen protocol (two runs) | `python <WORK>/src/native_physics_v2.py --project-dir <WORK> --output-dir <WORK>/v2_artifacts_canonical_run1` (and `run2`) | `v2_artifacts_canonical_run{1,2}/artifact_hashes.json` |
| Method matrix W3 (two runs, about 25 min each) | `cd scaleup && python -m w3.run_w3 run --work ../<WORK> --output-dir ../<WORK>/w3_artifacts_canonical_run1` (and `run2`), then `python -m w3.run_w3 compare --a … --b …` | `w3_artifacts_canonical_run{1,2}/artifact_hashes.json` |
| W4, W5 | same pattern with `w3.run_w4` and `w3.run_w5` | `w4_…`, `w5_…/artifact_hashes.json` |
| Post-hoc analyses | `cd scaleup && python -m w3.posthoc_round2 --work ../<WORK> --output-dir ../<WORK>/posthoc_review_round2_repro` | `posthoc_review_round2/artifact_hashes.json` |

The sealed validations are one-time events and are not re-run: V1's gate was
consumed on 2026-07-16 and D3's on 2026-10-17. Their programs are released so
that the chronology can be audited: the custodians reproduce the splits and the
sealed-label digests from the public files, the prediction programs reproduce
the hashed pre-reveal bundles, and the reveal programs refuse a second reveal
because the gate files exist. See [`scaleup/README.md`](scaleup/README.md) and
[`work/microfluidic_reliability/README.md`](work/microfluidic_reliability/README.md).

#### Stage 2 (R10), Stage 3 and the closed loop

All from `scaleup/`, with `<WORK>` = `../work/microfluidic_reliability` and
`<DATA>` = `../work/data`; every program writes a hash manifest and an
environment record, and the counted runs were executed twice into separate
directories.

| Programme | Command | Output compared with |
|---|---|---|
| R10 sealed run (A1 under protocol v1; R2-A1 and R2-A2 under v2 with `h_run2.py`) | `python -m w6.h_run --binding validation_plan/H_BINDING_A1_v2.json --protocol validation_plan/H_PROTOCOL_v1.md --artifacts <WORK>/h_A1_artifacts --output <WORK>/h_A1_run1` | `h_A1_run{1,2}/artifact_hashes.json`, `sealed_prediction_bundle.sha256` |
| R10 reveal (one-time; consumed) | `python -m w6.h_reveal … --run <WORK>/h_A1_run1 --output <WORK>/h_A1_reveal` (`h_reveal2.py` for R2) | `h_A1_reveal/h_reveal_results.json` |
| R10 combination | `python -m w6.h_combine --result A1=<WORK>/h_A1_reveal/h_reveal_results.json --result R2A1=… --result R2A2=… --output h_combined/H_COMBINED_DECISIONS.json` | `scaleup/h_combined/H_COMBINED_DECISIONS.json` |
| Stage 3: portfolio (H-F), warm start (H-G), recalibration | `python -m w6.h_portfolio --work <WORK> --output-dir <WORK>/h_portfolio_run1` (and `h_portfolio2`); `python -m w3.posthoc_round4_warmstart --work <WORK> --output-dir …` (and `_warmstart2`); `python -m w3.posthoc_round3 --work <WORK> --output-dir …` | `h_portfolio*_run{1,2}/`, `posthoc_review_round4_hg*_run{1,2}/`, `posthoc_review_round3_run{1,2}/artifact_hashes.json` |
| Few-shot rule meta-analysis (H-J) | `python -m w6.h_fewshot_rule --work <WORK> --data <DATA> --out <WORK>/h_fewshot_rule_run1`, then `h_fewshot_rule_analyze.py`, `h_fewshot_rule_posthoc.py` | `h_fewshot_rule_run{1,2}/artifact_hashes.json` |
| Closed loop E0 and E1/E2 | `python -I w7/closed_loop.py --stage e0 --work <WORK> --data <DATA> --out <WORK>/e0_run1`; `python -I w7/closed_loop.py --stage e1e2 --work <WORK> --data <DATA> --out <WORK>/e1e2_run1 --e0 <WORK>/e0_run1`; then `closed_loop_analyze.py` | `e0_run{1,2}/`, `e1e2_run{1,2}/artifact_hashes.json` |
| E1b, E3, E4, E6, E7 wrappers; E5 analyzer | `python -I w7/closed_loop_e1b.py …` (each wrapper documents its arguments in its docstring); `closed_loop_e5.py`, `closed_loop_tie_check.py`, `closed_loop_floor_check.py`, `figures_closed_loop.py` | the corresponding `e*_run{1,2}/` manifests, `e4_identity/e4_identity.json`, `e_tie_check/`, `e_floor_check/`, `figures_closed_loop/values.json` |
| Reproduction fidelity (two environments) | `python fidelity/fidelity_driver.py --work <WORK> --data <DATA>/osf_938rs --upstream <DATA>/upstream/dafd-website --d2-raw <WORK>/blind_raw --envb-python <python3.9 of requirements/envB-xgboost133.txt> --envb-dyld <its library path> --protocol fidelity/FIDELITY_PROTOCOL_v1.md --worker fidelity/fidelity_worker133.py --out <WORK>/fidelity_run1` | `fidelity_run{1,2}/artifact_hashes.json` |

`scripts/verify_run_manifests.py` checks all 27 run pairs (the E4 pair differs
only in a wall-clock timing column, as `e4_identity/e4_identity.json` records)
and the single-run directories.

### Manuscript

```bash
cd manuscript && tectonic -X compile main.tex      # English
cd manuscript && tectonic -X compile main_zh.tex   # Chinese mirror (XeTeX, system CJK fonts)
```

## Claim boundaries

The intended contribution is a decision framework for learned surrogates in a
self-driving loop, instantiated offline on public tables, and the evidence
for and against its assumptions. The sealed validations (V1, D3, the three R10
tables) are negative or mixed results and are retained as such; V2 is a
physics-guided constrained regressor, not a classical physics-informed neural
network; every V2 result on the TX100 group is retrospective because those
labels had already been revealed; the closed loop replays recorded
measurements and performs no experiment. Nothing in this repository
establishes chemical compatibility, manufacturability, stable dripping, device
safety, or prospective discovery.

## Licences and citation

Original software, tests and configuration: [Apache License 2.0](LICENSE).
Manuscript text, original figures and repository-authored result summaries:
[CC BY 4.0](LICENSES/CC-BY-4.0.txt); see [`LICENSES/README.md`](LICENSES/README.md)
for the file-level boundary. Third-party data, papers and upstream software
keep their own terms ([`THIRD_PARTY_DATA.md`](THIRD_PARTY_DATA.md)). This
independent work reproduces, stress-tests or builds on public work by
Lashkaripour et al., Chagot et al., Lavagna et al., Hormozinezhad et al.,
Wang et al. and Song et al.; it does not imply their endorsement. Citation metadata: [`CITATION.cff`](CITATION.cff).

To cite the preprint:

```bibtex
@misc{zhang2026surrogate,
  author        = {Zhang, Zhiyuan},
  title         = {When Can a Learned Design Surrogate Be Trusted in a Self-Driving Microfluidics Laboratory? An Offline Closed-Loop Framework Built and Tested on Six Public Tables},
  year          = {2026},
  eprint        = {2610.36395},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2610.36395}
}
```
