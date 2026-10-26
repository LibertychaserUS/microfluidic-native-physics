# Scale-up suites: W3 method matrix, W4/W5 comparison models, D3 sealed validation

`w3/` is one package whose modules mirror the W3 protocol: `workspace.py`
(imports the audited suite's frozen loaders from `../work/microfluidic_reliability/src`
and pins single-threaded numerics), `datasets.py`, `cells.py`, `models.py`,
`intervals.py`, `applicability.py`, `evaluate.py` (the only place test labels
enter), `run_w3.py` (orchestrator, strict JSON, hash manifest, `compare`),
`uq_models.py` + `run_w4.py` (quantile forest, Gaussian process, deep ensemble),
`weighted_conformal.py` + `run_w5.py` (covariate-shift-weighted split
conformal), the four D3 programs `d3_custodian.py`, `d3_predict.py`,
`d3_reveal.py`, `d3_posthoc_correction.py`, and `posthoc_round2.py`.
`tests/` holds the invariant tests (seven for W3, six for W4, seven for W5;
data-dependent cases need `MICROFLUIDIC_WORK`).

## Frozen files

| File | Digest sidecar | Frozen |
|---|---|---|
| `W3_PROTOCOL.md` | `W3_PROTOCOL.md.sha256` | before any W3 run |
| `W4_PROTOCOL.md` | `W4_PROTOCOL.md.sha256` | before any W4 run |
| `W5_PROTOCOL.md` | `W5_PROTOCOL.md.sha256` | before any W5 run |
| `D3_BLIND_PROTOCOL.md` | `D3_BLIND_PROTOCOL.md.sha256` | before the D3 dataset was downloaded |
| `D3_BINDING.json` | `D3_BINDING.json.sha256` | after inspecting headers and metadata only |
| `D3_DEVIATION_REGISTER.md` | pre-fit digest in `d3_blind_artifacts/prereveal/code_and_lock_hashes.json` | entries D3-R1 and D3-P1 were appended after the reveal and say so |

The protocols refer to a planning document (`08_scaleup_experiment_plan.md`)
of the authors' working repository; it is not part of the release and the
frozen bytes are kept unchanged.

## Commands (from this directory; `<WORK>` = `../work/microfluidic_reliability`)

```bash
python -m unittest w3.tests.test_w3 w3.tests.test_w4 w3.tests.test_w5      # MICROFLUIDIC_WORK=<WORK> for the data cases
python -m w3.run_w3 run --work <WORK> --output-dir <WORK>/w3_artifacts_canonical_run1
python -m w3.run_w3 run --work <WORK> --output-dir <WORK>/w3_artifacts_canonical_run2
python -m w3.run_w3 compare --a <WORK>/w3_artifacts_canonical_run1 --b <WORK>/w3_artifacts_canonical_run2
# W4 and W5: the same three commands with w3.run_w4 / w3.run_w5
python -m w3.posthoc_round2 --work <WORK> --output-dir <WORK>/posthoc_review_round2_repro
```

D3 chronology (one-time; released for audit, the reveal cannot be repeated):

```bash
python -m w3.d3_custodian --download-dir ../work/data/zenodo_17867478 --binding D3_BINDING.json \
    --protocol D3_BLIND_PROTOCOL.md --output-dir <WORK>/d3_blind_artifacts_repro
python -m w3.d3_predict --artifacts-dir <WORK>/d3_blind_artifacts_repro --protocol D3_BLIND_PROTOCOL.md \
    --binding D3_BINDING.json --register D3_DEVIATION_REGISTER.md --output-dir <WORK>/d3_blind_artifacts_repro/prereveal
python -m w3.d3_reveal --artifacts-dir <WORK>/d3_blind_artifacts_repro --prereveal-dir <WORK>/d3_blind_artifacts_repro/prereveal \
    --output-dir <WORK>/d3_blind_artifacts_repro/reveal --protocol D3_BLIND_PROTOCOL.md --binding D3_BINDING.json \
    --register D3_DEVIATION_REGISTER.md
python -m w3.d3_posthoc_correction --reveal-dir <WORK>/d3_blind_artifacts/reveal --output <path>.json
```

The custodian reproduces the row counts and the digests recorded in
`<WORK>/d3_blind_artifacts/custodian_manifest.json`; the prediction program
re-verifies the register digest, so a fresh run against the released register
(which now contains the post-reveal entries) reports the digest difference
explicitly rather than silently proceeding.

## Stages 2–4 (v0.5): protocols, programs and byte copies

| Directory | Content |
|---|---|
| `validation_plan/` | frozen protocols with digest sidecars: `H_PROTOCOL_v1.md`/`v2.md` (pre-registered hypotheses H-A–H-E on A1, R2-A1, R2-A2), the per-table bindings `H_BINDING_*.json`, `H_F_PORTFOLIO_PROTOCOL_v1/v2.md`, `H_G_WARMSTART_PROTOCOL_v1/v2.md`, `H_J_FEWSHOT_RULE_PROTOCOL_v1.md`, `POSTHOC_ROUND3_RECALIBRATION_PROTOCOL_v1.md`, the closed-loop protocols `E0_E1_E2_…`, `E1B_…`, `E3_…`, `E4_…`, `E5_…`, `E6_…`, `E7_…`; the registers `H_DEVIATION_REGISTER.md` (H-D1–H-D15) and `E_DEVIATION_REGISTER.md` (E-D1–E-D13); the feasibility gate `H_I_POOL_FEASIBILITY_GATE.md`; and one `*_SCRIPT.sha256` sidecar per frozen program (`scripts/verify_hash_registry.py` checks every one against the released program). The H-A–H-E, H-F, H-G and round-3 protocols and the H register are written in Chinese; they are released byte for byte because the manuscript cites their digests. |
| `w6/` | R10 programs: `h_download.py` (header-only intake), `h_custodian.py` / `h_custodian2.py` (seal by tier, mode 0600 labels), `h_run.py` / `h_run2.py` (frozen runner), `h_reveal.py` / `h_reveal2.py` (one-time gate), `h_combine.py` (protocol-level decisions), `h_portfolio.py` / `h_portfolio2.py` (H-F), `h_fewshot_rule.py`, `h_fewshot_rule_analyze.py`, `h_fewshot_rule_posthoc.py` (H-J). |
| `w7/` | closed loop: `closed_loop.py` (E0, E1/E2 runner), `closed_loop_analyze.py`, the wrappers `closed_loop_e1b.py`, `_e3.py`, `_e4.py`, `_e6.py`, `_e7.py` with their analyzers, `closed_loop_e5.py`, `closed_loop_e4_identity.py`, `closed_loop_e6_posthoc.py`, `closed_loop_tie_check.py`, `closed_loop_floor_check.py`, `figures_closed_loop.py`. |
| `w3/posthoc_round3.py`, `posthoc_round4*.py` | post-hoc rounds 3 (small-sample recalibration) and 4 (mechanism checks, warm start H-G). |
| `fidelity/` | reproduction-fidelity protocol, driver (ENV-A), worker (ENV-B, xgboost 1.3.3), post-hoc G0 script, deviation register, report, and byte copies of both runs' manifests, environments, results and upstream pins. |
| `e_closed_loop/`, `h_A1/`, `h_R2A1/`, `h_R2A2/`, `h_combined/`, `h_portfolio/`, `h_fewshot_rule/`, `posthoc_round3/`, `posthoc_round4/` | byte copies of manifests, environments, run summaries, reveal results, consumed gates, reports and registers at the paths the manuscript's Section 8 names. The same bytes live in the workspace layout under `../work/microfluidic_reliability/` (table below), which is what the ledger script reads. |
| `h_intake/` | header-only intake manifests of the six candidate tables (A1, A2, B1, B2, R2-A1, R2-A2); the `directory` field is redacted to `<WORK>/../data/…`. |
| `synthesis/CLAIMS_REGISTRY_v0.md` | the sub-conclusion registry cited in Section 7. |

Mapping between the byte copies here and the workspace directories:

| `scaleup/…` | `work/microfluidic_reliability/…` |
|---|---|
| `e_closed_loop/e0_run{1,2}`, `e1e2_run{1,2}`, `e1b_run{1,2}`, `e3_run{1,2}`, `e4_mac_run{1,2}`, `e5_run{1,2}`, `e6_run{1,2}`, `e7_run{1,2}`, `e1_reanalysis`, `e1_dense_sensitivity`, `e6_posthoc`, `e4_identity`, `e_tie_check`, `e_floor_check`, `figures_closed_loop` | the same names |
| `h_A1/custodian`, `h_A1/run{1,2}`, `h_A1/reveal` | `h_A1_artifacts`, `h_A1_run{1,2}`, `h_A1_reveal` |
| `h_R2A1/` (flat: `custodian_manifest.json`, `run{1..4}_artifact_hashes.json`, `h_run_summary*.json`, `reveal_artifact_hashes.json`, `h_reveal_results*.json`) | `h_R2A1_artifacts`, `h_R2A1_run{1..4}`, `h_R2A1_reveal`, `h_R2A1_reveal_b1` |
| `h_R2A2/` | `h_R2A2_artifacts`, `h_R2A2_run{1,2}`, `h_R2A2_reveal` |
| `h_portfolio/`, `h_portfolio/v2/` | `h_portfolio_run{1,2}`, `h_portfolio2_run{1,2}` |
| `h_fewshot_rule/run{1,2}`, `posthoc` | `h_fewshot_rule_run{1,2}`, `h_fewshot_rule_posthoc` |
| `posthoc_round3/` | `posthoc_review_round3_run{1,2}` |
| `posthoc_round4/`, `posthoc_round4/v2/` | `posthoc_review_round4`, `posthoc_review_round4_hg_run{1,2}`, `posthoc_review_round4_hg2_run{1,2}` |
| `fidelity/run{1,2}`, `fidelity/posthoc` | `fidelity_run{1,2}`, `fidelity_posthoc` |

Not released from these directories: per-loop tables (`loops.csv`, `budget.csv`,
shuffled-label loop tables), the per-row files of the R10 runs (development
tables, sealed features and labels, sealed prediction bundles, scored rows,
point predictions, interval rows), the fidelity per-row predictions, split
manifest, worker task files and third-party table copies, and the warm-start
curves above 1 MB. Each is listed with its digest in the directory's
`artifact_hashes.json`, and the frozen programs regenerate them.
