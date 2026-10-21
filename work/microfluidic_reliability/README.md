# Experiment workspace (E0, expanded DAFD suite, V1, V2)

This directory is the released part of the experiment workspace, kept in its
original layout so that every artifact path cited in the manuscript (Appendix
C) resolves here and so that the frozen programs and tests, which derive their
paths from this layout, run unchanged. `work/data/` next to it receives the
public datasets (see `../data/README.md`).

## Frozen documents

| File | Role |
|---|---|
| `SOURCE_MANIFEST.md` | DAFD source, licence and digests; implementation boundary |
| `EXPERIMENT_PROTOCOL.md` | frozen protocol of the expanded DAFD suite |
| `PROTOCOL_DEVIATION_REGISTER.md` | deviation register D01–D19 |
| `BLIND_VALIDATION_PROTOCOL.md`, `BLIND_PREREGISTRATION_LOCK.json` (+ `.sha256`) | V1 sealed chemistry holdout: protocol and pre-registration lock |
| `BLIND_REVEAL_INCIDENT.md`, `JSON_SERIALIZATION_POSTMORTEM.md` | V1 reveal incident and the serialisation-only recovery |
| `PHYSICS_CONSTRAINED_V2_PROTOCOL.md` (+ `.sha256`) | V2 frozen protocol |
| `LITERATURE_EVIDENCE_MATRIX.md` | working evidence ledger of the cited sources |

Digests of these files are cross-referenced in the manuscript's Appendix D and
verified by `scripts/verify_hash_registry.py`.

## Programs (`src/`)

| Program | Suite | Notes |
|---|---|---|
| `dafd_reproduction.py` | E0 | reproduces the published XGBoost baseline from the OSF workbooks; writes `artifacts/` |
| `reliability_analysis.py`, `interval_experiments.py` | reliability audit | applicability screens, CQR, external-shift coverage; `study_core.py` holds the shared core |
| `expanded_study.py` | expanded DAFD suite | grouped splits, intervals, applicability, inverse replay, negative controls; `--output-dir`, `--fast` (non-inferential) |
| `blind_custodian.py`, `blind_predict.py`, `blind_reveal.py`, `blind_reveal_recovery.py`, `blind_posthoc_analysis.py` | V1 | custodian splits the public `.mat` tables and seals the TX100 labels (mode 0600); prediction writes the hashed bundle; reveal consumed the one-time gate; recovery re-serialised the same locked endpoints; post-hoc analysis is marked post hoc |
| `native_physics_v2.py` | V2 | native-variable physics-guided constrained regressor under the frozen protocol; `--project-dir`, `--output-dir` |
| `strict_json.py` | all | JSON writer that rejects `NaN` and infinities |

Tests in `tests/` are run with `python -m unittest discover -s work/microfluidic_reliability/tests`
from the repository root (three modules need the OSF workbooks in
`work/data/osf_938rs/`; `test_native_physics_v2.py` needs PyTorch).

## Artifacts

- `artifacts/`: E0 and reliability summaries and frozen figures (the XGBoost
  model binary is not committed; its digest is in `hashes/appendix_d_registry.json`).
- `blind_artifacts/`: V1 `custodian_manifest.json`, `prereveal/` bundle,
  `BLIND_REVEAL_CONSUMED.json`, `blind_reveal_results_recovered.json`,
  `posthoc_analysis_corrected/`. The sealed label file and the re-cut
  development/feature tables are not released (digests in the custodian
  manifest); `src/blind_custodian.py` regenerates them byte-for-byte from the
  public `.mat` files, and V2, W3–W5 and the post-hoc programs need them
  (see `../data/README.md`).
- `d3_blind_artifacts/`: D3 `custodian_manifest.json`, `prereveal/` and
  `reveal/` (programs in `../../scaleup/w3/`). Sealed labels and re-cut tables
  are not released; `w3.d3_custodian` regenerates them from the public archive.
- `expanded_artifacts_canonical_run{1,2}/`, `v2_…`, `w3_…`, `w4_…`, `w5_…`: the
  five canonical run pairs. Each `run1` directory holds the manifest and
  metadata, each `run2` directory every artifact up to 1 MB (plus
  `w3_…/point_predictions.csv`, which the ledger reads). Per-row files above
  1 MB stay in the workspace; `artifact_hashes.json` lists all of them and
  `scripts/verify_run_manifests.py` re-hashes the released ones.
- `posthoc_review_round2/`: the four post-hoc analyses with their own manifest.

The uncorrected V1 post-hoc directory, fast/smoke runs and pre-fix outputs are
excluded from the release inventory, as the protocols require.
