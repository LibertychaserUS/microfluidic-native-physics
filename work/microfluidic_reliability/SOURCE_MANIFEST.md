# Source and reuse manifest

## Primary paper and data

- Paper: Lashkaripour et al., *Design automation of microfluidic single and
  double emulsion droplets with machine learning*, Nature Communications 15,
  83 (2024), https://doi.org/10.1038/s41467-023-44068-3.
- Article license: Creative Commons Attribution 4.0.
- Public data deposit: https://osf.io/938rs/.
- Data access date: 2026-07-16.

| File | SHA-256 |
|---|---|
| Comprehensive_normalized.xlsx | aebb0136ff812f7233c8fd57ebc0f27f1b935736e8a9c1d4fdc20585f06b9c06 |
| Generalizability_data_normalized.xlsx | e90996da3e866e64922dc3b8aaef4b092ee6845fe95aac356c73e90ed108b303 |
| Final_Comprehensive_plus_Generalizability_data_normalized.xlsx | 14ebfaae93a7bbc7e1a73be91a629c57b1720dffac0dd45cef6d7df4e7eb6ac3 |
| Raw_SE_dataset.xlsx | 996862065c33a3f17a36ad2df825179937ddbd7f3c8da41765946a9c1db5e579 |

The OSF data are not redistributed in this package. Users should retrieve the
files from the public deposit and verify the hashes above.

## Implementation boundary

The scripts in this package are an independent implementation from the method
parameters stated in the paper. No code was copied from the DAFD website
repository.

At the time of inspection, the root of `CIDARLAB/DAFD-website` did not contain a
license file. It was therefore treated as a readable reference, not a source of
reusable code. `CIDARLAB/DAFD_Metrics` has a GPL-3.0 license, but its code is not
used here.

## Optional secondary dataset

- Chagot et al., *Surfactant-laden droplet size prediction in a flow-focusing
  microchannel: a data-driven approach*, Lab on a Chip (2022),
  https://doi.org/10.1039/D2LC00416J.
- Dataset: https://zenodo.org/records/7055018.
- Dataset license: Creative Commons Attribution 4.0.
- Status: identified for a later independent case study; not used in the
  reported DAFD reproduction or pruning results.

