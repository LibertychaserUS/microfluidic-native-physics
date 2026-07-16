<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Third-party data and attribution

The Apache-2.0 license in this repository applies to repository-authored
software code, tests, and configuration unless an individual file says
otherwise. Repository-authored paper/report text and original figures/results
use CC BY 4.0. Neither license relicenses source papers, public datasets,
upstream code, or upstream figures.

No raw third-party data are included in the documentation-only release staging
tree. Before any data or derived artifact is published, the release must record
the source URL, governing terms, raw-file hash, retrieval date, schema, and
transformation provenance.

## Chagot droplet dataset

- Article: Chagot et al., *Surfactant-laden droplet size prediction in a
  flow-focusing microchannel: a data-driven approach*, **Lab on a Chip**
  (2022), DOI 10.1039/D2LC00416J.
  [Publisher page](https://pubs.rsc.org/en/content/articlehtml/2022/lc/d2lc00416j)
- Dataset: Zenodo record 7055018, version 1.
  [Zenodo record](https://zenodo.org/records/7055018)
- The Zenodo record reports the dataset license as **Creative Commons
  Attribution 4.0 International (CC BY 4.0)**. Users must follow the record's
  current terms and cite the source dataset and article.
- The release must verify the source files against the recorded checksums before
  analysis. Repository copies, if any, must remain clearly separated from
  Apache-2.0 code.

## DAFD data

- Article: Lashkaripour et al., *Design automation of microfluidic single and
  double emulsion droplets with machine learning*, **Nature Communications**
  (2024). [Publisher page](https://www.nature.com/articles/s41467-023-44068-3)
- Public deposit used by the independent reproduction:
  [OSF project 938rs](https://osf.io/938rs/)
- Redistribution terms for each downloaded OSF object must be verified from the
  source record before it is mirrored. Until that check is complete, raw DAFD
  files must not be committed to GitHub or uploaded to Hugging Face.

## Derived release artifacts

Derived tables may include split manifests, per-row predictions, uncertainty
bands, metrics, and hashes. A derived artifact is publishable only when:

1. its source rows can be traced to a cited public record;
2. its contents are permitted by the source terms;
3. it does not silently embed the entire upstream dataset;
4. its generation script and configuration are identified;
5. its schema and missing-value policy are documented; and
6. it passes strict JSON/CSV validation and the privacy/secrets gate.

No source author or institution is affiliated with or endorses this independent
project unless an explicit statement is added with evidence.
