<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Third-party data and attribution

The Apache-2.0 licence of this repository applies to repository-authored
software, tests and configuration; repository-authored manuscript text, original
figures and result summaries use CC BY 4.0. Neither licence relicenses the
source papers, the public datasets, upstream code or upstream figures.

**No raw third-party data are included in this repository.** The programs read
the public files from `work/data/` after the user downloads them
(`work/data/README.md`); `data/source_registry.json` records URL, size and
digest of every raw file, and the manuscript's Appendix D (mirrored in
`hashes/appendix_d_registry.json`) cites the same digests.

## DAFD data (Lashkaripour et al.)

- Article: Lashkaripour et al., *Design automation of microfluidic single and
  double emulsion droplets with machine learning*, **Nature Communications** 15,
  83 (2024), DOI 10.1038/s41467-023-44068-3 (CC BY 4.0 article).
- Public deposit used: [OSF project 938rs](https://osf.io/938rs/), four
  workbooks retrieved 2026-07-16 (digests in `data/source_registry.json`).
- Object-level redistribution terms of the deposit were not verified, so the
  workbooks are not mirrored and no derived table that reproduces them row by
  row is released. The implementation is written from the published method
  description and copies no code from the DAFD web repositories.

## Chagot droplet dataset

- Article: Chagot et al., *Surfactant-laden droplet size prediction in a
  flow-focusing microchannel: a data-driven approach*, **Lab on a Chip** (2022),
  DOI 10.1039/D2LC00416J.
- Dataset: [Zenodo record 7055018](https://zenodo.org/records/7055018)
  (`matTrainingDataSet.mat`, `matTestDataSet.mat`, expected locally at
  `work/microfluidic_reliability/blind_raw/`), licence **CC BY 4.0** as
  stated on the record; the record's MD5 values are reproduced in the registry
  and in `work/microfluidic_reliability/blind_artifacts/custodian_manifest.json`.
- The V1 custodian re-cuts these rows into a development table, a blind-feature
  table and a sealed-label file. None of the three is released (they are rows
  of the public dataset); their digests are in the custodian manifest and the
  custodian program reproduces them deterministically from the public files.

## Smart Microfluidics liposome dataset (Lavagna et al.)

- Dataset: [Zenodo record 17867478](https://zenodo.org/records/17867478),
  DOI 10.5281/zenodo.17867478, v1.0, licence **CC BY 4.0**; related publication
  DOI 10.1016/j.ijpharm.2025.126362.
- Files used: `microfluidics_dataset.zip` (the three bound raw tables
  `raw_data/initial_formulations_raw.csv`, `raw_data/new_formulations_raw.csv`,
  `raw_data/wet_lab_validation_raw.csv`) and `README.pdf`; digests in the
  registry and in
  `work/microfluidic_reliability/d3_blind_artifacts/custodian_manifest.json`.
- The D3 custodian re-cuts these rows into a development table, input-only
  sealed feature tables and sealed-label files. None of them is released; the
  released D3 artifacts (prediction bundle, scored rows, reveal result) contain
  the inputs and predictions of the sealed rows together with the labels
  revealed on 2026-10-17, attributed to this record.

## Derived release artifacts

Released derived tables (per-row predictions, scores, intervals, split
summaries) carry the inputs of the public rows they score. They are published
because their source rows are traceable to the cited records, their generating
program and configuration are released alongside, their schema is the one the
programs write, and they pass the strict-JSON and policy gates in `scripts/`.
No derived artifact reproduces an upstream dataset in full.

No source author or institution is affiliated with or endorses this independent
project.
