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
