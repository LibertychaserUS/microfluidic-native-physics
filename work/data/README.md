# Public source data (not tracked by git)

The three datasets are public under their own licences and are **not
redistributed** here. Place them in this directory so that the frozen programs
find them at the paths they were run with; `data/source_registry.json` holds
the authoritative URLs, sizes and digests, and `scripts/verify_hash_registry.py`
re-hashes whatever is present.

```text
work/data/
  osf_938rs/                      DAFD deposit (Lashkaripour et al. 2024), https://osf.io/938rs/
    Comprehensive_normalized.xlsx
    Generalizability_data_normalized.xlsx
    Final_Comprehensive_plus_Generalizability_data_normalized.xlsx
    Raw_SE_dataset.xlsx
  zenodo_17867478/                Smart Microfluidics liposome dataset (Lavagna et al.; CC BY 4.0)
    microfluidics_dataset.zip
    README.pdf
    microfluidics_dataset/        the archive extracted in place (raw_data/*.csv are the bound files)
```

The two Chagot MATLAB tables (Zenodo 7055018, CC BY 4.0) go to
`work/microfluidic_reliability/blind_raw/` instead, because the V1 custodian
lock in `native_physics_v2.py` verifies them at that frozen path:

```text
work/microfluidic_reliability/blind_raw/
  matTrainingDataSet.mat
  matTestDataSet.mat
```

Download and verify:

```bash
python3 scripts/fetch_public_data.py --dataset chagot_zenodo_7055018 --destination work/microfluidic_reliability
python3 scripts/fetch_public_data.py --dataset smart_microfluidics_zenodo_17867478
unzip -q work/data/zenodo_17867478/microfluidics_dataset.zip -d work/data/zenodo_17867478/
# OSF objects have no stable direct download URL recorded here: download the four
# workbooks from the project page into work/data/osf_938rs/ and verify them:
python3 scripts/fetch_public_data.py --dataset dafd_osf_938rs --verify-only
```

Expected SHA-256 digests are in `data/source_registry.json` and in the
manuscript's Appendix D (`hashes/appendix_d_registry.json`).

## Regenerating the custodian tables

The V2, W3–W5 and post-hoc programs read the Chagot rows through the V1
custodian lock (`blind_artifacts/development_labeled.csv`,
`blind_features.csv`, `sealed_blind_labels.csv`), and the post-hoc programs
read the D3 tables (`d3_blind_artifacts/development_labeled.csv`,
`sealed_*_features.csv`). These re-cuts of public rows are not committed; the
custodian programs regenerate them byte-for-byte (the digests are verified
against the committed custodian manifests, and both manifests are themselves
reproduced byte-identically):

```bash
python work/microfluidic_reliability/src/blind_custodian.py \
    work/microfluidic_reliability/blind_raw/matTrainingDataSet.mat work/microfluidic_reliability/blind_raw/matTestDataSet.mat \
    --source-url 'https://zenodo.org/records/7055018/files/matTrainingDataSet.mat?download=1' \
    --source-url 'https://zenodo.org/records/7055018/files/matTestDataSet.mat?download=1' \
    --source-md5 c5dd5f3fbfd7c218cea734b205ae502e --source-md5 a6c41bba18e4f232e7f25e2207fd183c
cd scaleup && python -m w3.d3_custodian --download-dir ../work/data/zenodo_17867478 \
    --binding D3_BINDING.json --protocol D3_BLIND_PROTOCOL.md --output-dir ../work/microfluidic_reliability/d3_blind_artifacts
```

Both commands write into the committed artifact directories; git ignores the
regenerated tables and the sealed label files (mode 0600). Compare the written
`custodian_manifest.json` with the committed one (`git diff` must be empty) and
run `python3 scripts/verify_hash_registry.py`, which then reports the
regenerated tables as verified.

## Stage 2 tables and the upstream clone (v0.5)

```text
work/data/
  h_A1/PMC11834948_supplementaryFiles.zip      Hormozinezhad et al. 2025 ESI (Europe PMC bundle; CC BY 3.0)
  h_R2A1/Data for FrameworkA.xlsx              Wang et al. 2024, Zenodo 10526441 (CC BY 4.0)
  h_R2A2/LNP_Atlas_DB_202509_v1.csv            LNP Atlas, Zenodo 17243733 (CC BY 4.0)
  upstream/dafd-website/                       git clone of CIDARLAB/dafd-website at commit 4902f8ce (fidelity protocol only)
```

```bash
python3 scripts/fetch_public_data.py --dataset hormozinezhad_2025_esi_A1
python3 scripts/fetch_public_data.py --dataset wang_2024_zenodo_10526441_R2A1
python3 scripts/fetch_public_data.py --dataset lnp_atlas_zenodo_17243733_R2A2
git clone https://github.com/CIDARLAB/dafd-website work/data/upstream/dafd-website && git -C work/data/upstream/dafd-website checkout 4902f8ce
```

`scaleup/w6/h_download.py` is the program that produced the header-only intake
manifests under `scaleup/h_intake/`; it downloads the same files and prints no
cell value. The R10 custodians then write `work/microfluidic_reliability/h_<code>_artifacts/`
(development table, sealed features, sealed labels with mode 0600) whose
digests are in the released custodian manifests:

```bash
cd scaleup && python -m w6.h_custodian  --binding validation_plan/H_BINDING_A1_v2.json --protocol validation_plan/H_PROTOCOL_v1.md \
    --zip ../work/data/h_A1/PMC11834948_supplementaryFiles.zip --output ../work/microfluidic_reliability/h_A1_artifacts
cd scaleup && python -m w6.h_custodian2 --binding validation_plan/H_BINDING_R2A1.json --protocol validation_plan/H_PROTOCOL_v2.md \
    --source "../work/data/h_R2A1/Data for FrameworkA.xlsx" --output ../work/microfluidic_reliability/h_R2A1_artifacts
cd scaleup && python -m w6.h_custodian2 --binding validation_plan/H_BINDING_R2A2.json --protocol validation_plan/H_PROTOCOL_v2.md \
    --source ../work/data/h_R2A2/LNP_Atlas_DB_202509_v1.csv --output ../work/microfluidic_reliability/h_R2A2_artifacts
```

The closed-loop runners (`scaleup/w7/`) read the Stage 1 and Stage 2 tables
through these custodian outputs and the `--data` directory (`work/data`).
