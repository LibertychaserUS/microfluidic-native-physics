<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Public release checklist

GitHub is the primary release. Hugging Face is published only after the GitHub
commit/tag is fixed and all shared artifacts can point back to it.

Nothing is marked VERIFIED merely because a script completed. Each box requires
the named evidence in the public repository or release record.

## 1. Identity and repository metadata

- [x] Public author name is exactly `Zhiyuan Zhang`.
- [x] GitHub target is
  `https://github.com/LibertychaserUS/microfluidic-native-physics`.
- [x] HF target is
  `https://huggingface.co/datasets/Chaser111/microfluidic-native-physics`.
- [x] `CITATION.cff` contains no invented DOI, arXiv ID, affiliation, ORCID, or
  release date.
- [ ] Final paper/software title approved; update provisional title everywhere.
- [ ] Release version and Git tag approved.
- [ ] Optional ORCID supplied by the author, or intentionally omitted.

## 2. Claim and protocol integrity

- [ ] V1 is described as a negative held-out-chemistry result.
- [ ] V1 reveal wording says "sealed prediction followed by disclosed
  serialization-only recovery" rather than "clean atomic one-shot".
- [ ] V1 frozen source, lock, input, label, prediction, and result hashes are
  included and cross-checked.
- [ ] The JSON incident report and recovery boundary are included.
- [ ] V2 frozen protocol and pre-fit hash are included unchanged.
- [ ] V2 is called physics-guided/constrained, not a classical PINN.
- [ ] TX100 V2 results are called post-reveal retrospective analyses.
- [ ] No wet-lab, chemical-compatibility, manufacturability, stable-dripping,
  or prospective-discovery claim appears.
- [ ] All locked negative/mixed results and ablations remain in the release.
- [ ] Manuscript, README, abstract, figure captions, and HF card use the same
  claim boundaries.

## 3. Code, environment, and tests

- [ ] Audited source copied only after active fixes finish.
- [ ] Frozen blind files copied byte-for-byte; do not rewrite formatting.
- [ ] Dependency lock generated with hashes and tested in a clean environment.
- [ ] Environment/platform manifest included.
- [ ] Unit and data-contract tests pass from a clean checkout.
- [ ] Leakage tests verify chemistry/source/device grouping as applicable.
- [ ] Native-to-physics identities, unit conversions, and hard output bounds
  have regression tests.
- [ ] Undefined analytical quantities serialize as JSON `null`.
- [ ] Positive/negative infinity fails serialization loudly.
- [ ] Every released JSON file passes JavaScript `JSON.parse`.
- [ ] Every released JSON file also passes Python `json.loads` with a
  `parse_constant` callback that rejects `NaN`, `Infinity`, and `-Infinity`;
  Python's permissive default is not accepted as evidence.
- [ ] No credentials, tokens, private paths, private URLs, or personal metadata
  appear in tracked files or history.

## 4. Data provenance and licensing

- [ ] Every raw file has source URL, retrieval date, source version, license or
  governing terms, size, and cryptographic hash.
- [ ] Chagot Zenodo files verify against the source record.
- [ ] The anomalous two-row `CMC = 0.60 mM` group is quarantined and documented;
  it is not reassigned by guesswork.
- [ ] DAFD OSF object-level redistribution terms are verified before mirroring.
- [ ] Raw third-party data are not covered by Apache-2.0 wording.
- [ ] Derived artifacts do not silently republish an upstream dataset contrary
  to its terms.
- [ ] Source article and dataset citations appear in README, manuscript, and HF
  card.

## 5. Artifact completeness and determinism

- [ ] Per-row predictions included for every released model/fold/seed.
- [ ] Split manifests prove zero prohibited group overlap.
- [ ] Inner model-selection records and early-stopping choices included.
- [ ] Applicability-domain calibration metadata included.
- [ ] Label-free retrospective selections are written and hashed before scoring.
- [ ] Point, interval, pruning, and target-retrieval endpoints are all included.
- [ ] Two full runs completed in separate clean output directories.
- [ ] Hash comparison is exact for deterministic artifacts.
- [ ] Any excluded volatile field is listed with a narrow justification.
- [ ] Superseded runs, recovery scratch files, old `fast`/`finalfast` outputs,
  and pre-fix JSON artifacts are absent from the public inventory.
- [ ] Figures are regenerated from released tables and visually inspected.
- [ ] Artifact hash manifest itself is validated and signed/tagged by commit.

## 6. Paper and documentation

- [ ] English manuscript completed.
- [ ] Supplement/method appendix completed.
- [ ] Chinese evidence audit completed.
- [ ] All numbers in prose and captions are programmatically checked against
  released artifacts.
- [ ] Primary-source links resolve.
- [ ] No "first" claim appears without a documented systematic basis.
- [ ] Limitations explain missing dynamic interfacial tension, adsorption
  kinetics, molecular descriptors, rheology, wetting, and wet-lab validation.
- [ ] No DOI/arXiv badge or citation is shown before an identifier exists.

## 7. GitHub-first publication

- [ ] Create GitHub repository under `LibertychaserUS`.
- [ ] Confirm default branch and repository visibility.
- [ ] Push the audited tree; review the full diff on GitHub.
- [ ] Run clean CI; only then replace the static pre-release badge with a real
  workflow badge.
- [ ] Protect the release tag from later mutation.
- [ ] Create a GitHub release with checksums and known-limitations notes.
- [ ] Record the exact Git commit and release tag for the HF mirror.

## 8. Hugging Face artifact mirror

- [ ] Copy `huggingface/README.md` to the HF dataset repository root.
- [ ] Replace all pending fields with the Git commit/tag and artifact inventory.
- [ ] Upload only license-cleared artifacts.
- [ ] Add cryptographic hashes and schemas for each uploaded file.
- [ ] Link every artifact to the generating GitHub code/config/commit.
- [ ] Link the HF dataset card back to GitHub as canonical.
- [ ] Link the GitHub README to the live HF dataset repository.
- [ ] Verify both links in an unauthenticated browser session.

## 9. Final go/no-go

- [ ] Independent claim audit finds no result stronger than its evidence.
- [ ] Reproduction from a clean clone matches the released hash manifest.
- [ ] GitHub and HF inventories match their respective roles.
- [ ] Known issues and negative results are prominent, not buried.
- [ ] Author approves public release.

If any mandatory item is unresolved, retain the static **pre-release audit**
status and do not label the repository or paper VERIFIED.
