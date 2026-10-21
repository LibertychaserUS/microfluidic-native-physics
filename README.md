<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Microfluidic Native Physics — code and artifacts for the preprint

Code, frozen protocols, deviation registers, number ledger, hash manifests and
summary artifacts behind the preprint

> Zhiyuan Zhang, *Reliability Audits for Machine-Learned Droplet-Microfluidic
> Design Surrogates* (2026; arXiv identifier to be added to `CITATION.cff` once
> assigned).

The study is retrospective and reports mostly negative results: a sealed
chemistry holdout (V1), a sealed cross-laboratory validation (D3), a frozen
physics-guided constrained regressor (V2), an expanded shift audit on the DAFD
tables, and a method-matrix benchmark with uncertainty-aware and
covariate-shift-weighted comparison models (W3–W5). Nothing here is evidence of
wet-lab feasibility of any device, material or formulation.

## Release status

- Tag `v0.3-preprint` accompanies manuscript draft v0.3.x. The code, registers,
  ledger and hash manifests promised in the manuscript's *Data and code
  availability* paragraph are in this repository; the compiled PDF is attached
  to the GitHub release.
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
scaleup/                         W3/W4/W5 protocols and runners, D3 protocol, binding, deviation register and programs
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
| E0 XGBoost model binary | regenerated by re-running E0 | Appendix D registry |
| Superseded fast/smoke runs, pre-fix outputs, the uncorrected V1 post-hoc directory | excluded from the release inventory by the protocols | — |

One released file was edited: `blind_artifacts/posthoc_analysis_corrected/blind_posthoc_summary.json`
carried fourteen absolute workspace paths as keys of its read-only audit block;
the path prefix was replaced by `<WORK>/`. No value changed; both digests are in
the registry.

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

### Manuscript

```bash
cd manuscript && tectonic -X compile main.tex      # English
cd manuscript && tectonic -X compile main_zh.tex   # Chinese mirror (XeTeX, system CJK fonts)
```

## Claim boundaries

The intended contribution is a reliability-under-shift evaluation of
machine-learned surrogates on public droplet-microfluidics data. The sealed
chemistry holdout (V1) and the sealed cross-laboratory validation (D3) are
negative results and are retained as such; V2 is a physics-guided constrained
regressor, not a classical physics-informed neural network; every V2 result on
the TX100 group is retrospective because those labels had already been
revealed. Nothing in this repository establishes chemical compatibility,
manufacturability, stable dripping, device safety, or prospective discovery.

## Licences and citation

Original software, tests and configuration: [Apache License 2.0](LICENSE).
Manuscript text, original figures and repository-authored result summaries:
[CC BY 4.0](LICENSES/CC-BY-4.0.txt); see [`LICENSES/README.md`](LICENSES/README.md)
for the file-level boundary. Third-party data, papers and upstream software
keep their own terms ([`THIRD_PARTY_DATA.md`](THIRD_PARTY_DATA.md)). This
independent work reproduces, stress-tests or builds on public work by
Lashkaripour et al., Chagot et al. and Lavagna et al.; it does not imply their
endorsement. Citation metadata: [`CITATION.cff`](CITATION.cff).
