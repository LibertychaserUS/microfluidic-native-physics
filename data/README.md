# Public source data registry

`source_registry.json` records the authoritative URL, size, SHA-256 and (where
the record publishes one) MD5 of every raw file the study used. The files
themselves are **not** in this repository; `scripts/fetch_public_data.py`
downloads the Zenodo files into `work/data/` and verifies them, and verifies
manually downloaded OSF workbooks with `--verify-only`. See
[`work/data/README.md`](../work/data/README.md) for the expected layout.

```bash
python3 scripts/fetch_public_data.py --dataset chagot_zenodo_7055018 --destination work/microfluidic_reliability
python3 scripts/fetch_public_data.py --dataset smart_microfluidics_zenodo_17867478
python3 scripts/fetch_public_data.py --dataset dafd_osf_938rs --verify-only
python3 scripts/fetch_public_data.py --dataset hormozinezhad_2025_esi_A1
python3 scripts/fetch_public_data.py --dataset wang_2024_zenodo_10526441_R2A1
python3 scripts/fetch_public_data.py --dataset lnp_atlas_zenodo_17243733_R2A2
```

Downloading a public file does not relicense it. See
[`THIRD_PARTY_DATA.md`](../THIRD_PARTY_DATA.md).
