<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Microfluidic Native Physics

[![Status: pre-release audit](https://img.shields.io/badge/status-pre--release%20audit-orange)](#release-status)
[![Evidence: V1 negative result retained](https://img.shields.io/badge/evidence-V1%20negative%20result%20retained-blue)](#sealed-v1-result)
[![License: Apache-2.0](https://img.shields.io/badge/code%20license-Apache--2.0-green)](LICENSE)

Reliability under chemistry shift for public droplet-microfluidics data: a
sealed-label negative validation, applicability-domain audits, and a
native-variable, physics-guided constrained regression protocol.

**GitHub is the canonical project and code repository.** The Hugging Face
dataset repository is a secondary mirror for released tables, manifests, and
reproducibility artifacts.

- GitHub (primary):
  [LibertychaserUS/microfluidic-native-physics](https://github.com/LibertychaserUS/microfluidic-native-physics)
- Hugging Face (artifact mirror; publication pending):
  [Chaser111/microfluidic-native-physics](https://huggingface.co/datasets/Chaser111/microfluidic-native-physics)

> **Pre-release warning:** this staging tree intentionally contains
> documentation and release metadata only. Audited source code, frozen
> protocols, per-row predictions, figures, and deterministic-run evidence must
> pass the [release checklist](docs/RELEASE_CHECKLIST.md) before publication.
> The badges above report this pre-release state; they do not claim that CI or
> the final reproduction has passed.

## What this project asks

Can public historical data, native experimental variables, physical-chemical
transformations, uncertainty estimates, and explicit feasibility constraints
reduce the search space for droplet-microfluidic experiments without running a
new wet experiment?

The project tests that question in a deliberately narrow setting:

1. reproduce and stress-test public tabular ML baselines;
2. hold out an entire surfactant-chemistry group rather than relying only on a
   random row split;
3. audit whether applicability-domain screens and uncertainty bands actually
   identify safer predictions under shift;
4. constrain a second-generation model using native inputs, deterministic
   dimensionless features, a semi-empirical prior, and hard output bounds; and
5. evaluate retrospective target retrieval without calling it prospective
   discovery.

This is **not** evidence that a material, formulation, device, or operating
path is wet-lab feasible. It does not establish chemical compatibility,
manufacturability, stable dripping, device safety, or clinical/industrial
readiness.

## Evidence status and claim boundaries

| Component | Current status | Permitted claim | Not permitted |
|---|---|---|---|
| V1 chemistry holdout | Sealed prediction completed; negative result retained | The published-feature baseline failed to transfer reliably to the held-out TX100 chemistry in this test | General failure of ML, a new wet-lab result, or a clean atomic one-shot reveal |
| Applicability/uncertainty audit | Retrospective public-data analysis; final release audit pending | Screens and intervals can be compared as shift diagnostics | Guaranteed safety or distribution-free coverage for a new chemistry |
| V2 native-physics protocol | Frozen before V2 fitting; audited results pending | Physics-guided constrained regression with deterministic native-to-physics transforms | A classical PINN, PDE solution, or validated physical simulator |
| Target search | Retrospective ranking/retrieval among historical conditions | Search-space pruning can be measured against already observed outcomes | Prospective materials discovery or experiment-free proof of feasibility |
| Wet-lab validation | Not performed | None | Any claim of experimental validation |

The narrow intended contribution is a reliability-under-shift evaluation. We
do not claim to be the first microfluidic inverse-design or ML system.

## Sealed V1 result

V1 combined the 468 public Chagot rows, sealed the complete TX100 group
(`CMC = 3.5 mM`, 87 rows), and trained only on the remaining 381 rows. Its fixed
four inputs were continuous-flow rate, dispersed-flow rate, equilibrium
interfacial tension, and concentration divided by CMC. Model settings and
evaluation endpoints were frozen before label reveal.

The held-out result was negative:

| Endpoint | Held-out TX100 result |
|---|---:|
| MAPE | 19.8436% |
| MAE | 0.02261 mm |
| RMSE | 0.02868 mm |
| R-squared | -0.2436 |
| Nominal 90% interval coverage | 44.83% |
| Mean interval width | 0.03281 mm |

Predictions were biased high on 85.06% of the held-out rows. The strict
training-range screen accepted all 87 rows, showing that this failure was not a
simple marginal-range violation. The six locked V1 target-retrieval strategies
did not improve the search endpoints: each answered 8 of 13 targets, with top-1,
top-3, and top-5 success on 3, 6, and 8 targets respectively.

The honest interpretation is conditional-shift failure: values of the four
recorded predictors overlap training support, but the relationship between
those predictors and droplet diameter changes with unseen surfactant chemistry.
The public table omits potentially relevant state such as dynamic interfacial
tension, adsorption kinetics, molecular chemistry, interfacial rheology, and
wetting behavior. More model capacity alone cannot reconstruct unobserved
physics.

### Reveal incident disclosure

The blind labels and prediction bundle were sealed and hash-checked before
reveal. The original reveal gate was consumed and the locked scores were
computed, but writing the result failed because a nested IEEE `NaN` reached a
strict JSON writer. A separate serialization-only recovery revalidated the
frozen hashes and recomputed the same locked endpoints while mapping undefined
values to JSON `null`. The frozen prediction bundle, labels, and analysis code
were not changed.

Accordingly, this repository describes V1 as **a sealed prediction followed by
a disclosed serialization-only recovery**, not as an uninterrupted atomic
one-shot reveal. The incident report, frozen hashes, recovery code, and strict
cross-parser regression tests are release requirements.

## V2: native-variable physics-guided constraints

V2 is intentionally called a **native-variable physics-guided constrained
regressor**. It is not a classical physics-informed neural network (PINN): the
public table has no pressure or velocity fields, interface coordinates,
time-dependent observations, boundary-condition fields, or PDE collocation
points.

The user/search-facing state is restricted to native recorded quantities:

```text
[Qc, Qd, equilibrium interfacial tension, concentration, CMC,
 no-surfactant indicator]
```

Flow ratio, capillary numbers, Reynolds numbers, and normalized concentration
are never trusted as caller-supplied inputs. They are recomputed deterministically
with fixed units and documented device/fluid constants. The primary V2 model
adds:

- a semi-empirical Kalli/Chagot anchor;
- a learned residual around that anchor;
- a hard sigmoid parameterization enforcing `0 < predicted diameter < D`;
- nested leave-one-chemistry-out model selection;
- bounded uncertainty heuristics whose held-out-chemistry coverage is reported
  as a transfer stress test, not a formal guarantee; and
- a constrained target-search compiler that varies only allowed native process
  variables and can abstain.

TX100 labels have already been revealed. Every V2 result on TX100 is therefore
retrospective and must not be described as a second blind validation. A new
blind claim requires a different, never-inspected external dataset.

## Authoritative public sources

This independent project builds on, reproduces, or stress-tests public work; it
does not imply endorsement by the source authors.

- Chagot et al., *Surfactant-laden droplet size prediction in a flow-focusing
  microchannel: a data-driven approach*, **Lab on a Chip** (2022):
  [publisher article](https://pubs.rsc.org/en/content/articlehtml/2022/lc/d2lc00416j)
- Chagot et al. public MATLAB tables, Zenodo record 7055018:
  [dataset record](https://zenodo.org/records/7055018)
- Lashkaripour et al., *Design automation of microfluidic single and double
  emulsion droplets with machine learning*, **Nature Communications** (2024):
  [publisher article](https://www.nature.com/articles/s41467-023-44068-3)
- DAFD public deposit used by the reproduction:
  [OSF project 938rs](https://osf.io/938rs/)
- Romano, Patterson, and Candès, conformalized quantile regression:
  [NeurIPS paper](https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html)
- Tibshirani et al., conformal prediction under covariate shift:
  [NeurIPS paper](https://proceedings.neurips.cc/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html)

Raw third-party data are not covered by this repository's Apache-2.0 code
license. See [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md) before downloading,
redistributing, or publishing any data-derived artifact.

## Reproducibility

### Release state

The current staging tree is not executable. After the audited source and lock
files are synchronized, a verified release will support the following core
workflow from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements-lock.txt
python -m unittest discover -s tests
python src/dafd_reproduction.py
python src/expanded_study.py --output-dir artifacts/expanded
```

The final V2 entry point and exact data-staging command will be documented only
after implementation, protocol-lock verification, and two deterministic full
runs. Until then, do not infer a command or a V2 result from this README.

Every release-labeled result must include:

- source URLs, source licenses/terms, raw-file hashes, and schema checks;
- frozen protocol and split-manifest hashes;
- per-row predictions rather than summary metrics alone;
- inner-selection records and outer-fold leakage assertions;
- strict JSON validation with JavaScript `JSON.parse` or Python
  `json.loads(..., parse_constant=reject)`;
- all locked negative results and ablations;
- two complete runs in separate directories; and
- a deterministic artifact comparison, with any expected non-deterministic
  metadata explicitly excluded and justified.

### Planned public repository layout

Items marked `release sync` are intentionally absent from this documentation-only
staging tree.

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── THIRD_PARTY_DATA.md
├── requirements-lock.txt              # release sync
├── protocols/                         # frozen protocols + hashes; release sync
├── src/                               # audited implementation; release sync
├── tests/                             # unit/leakage/strict-JSON tests; release sync
├── artifacts/
│   ├── manifests/                     # hashes and environment records; release sync
│   ├── predictions/                   # per-row outputs; release sync
│   ├── metrics/                       # strict JSON/CSV summaries; release sync
│   └── figures/                       # manuscript figures; release sync
├── manuscript/                        # paper and supplement; release sync
├── docs/
│   └── RELEASE_CHECKLIST.md
└── huggingface/
    └── README.md                       # staged HF dataset card
```

## 中文快速开始

本项目研究的是：在不做新湿实验的前提下，公开历史数据、原生实验变量、
物理化学约束和不确定性筛选，能否缩小微流控液滴实验的候选空间。

当前目录仍是**发布前文档骨架**，不能当作可运行版本。正式发布后：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements-lock.txt
python -m unittest discover -s tests
```

请先注意三条边界：

1. V1 对整组 TX100 的盲留出结果是负结果，必须保留；
2. V2 是“原生变量 + 物理引导 + 硬约束”的回归器，不是经典 PINN；
3. 所有搜索都是公开历史数据上的回放，不代表材料、配方或实验路径已经
   通过湿实验验证。

GitHub 是主仓库；HF 仅镜像经审计的数据表、预测、清单和结果卡片。

## License and citation

Original software code, tests, and configuration are released under the
[Apache License 2.0](LICENSE). Repository-authored paper/report text and
original figures/results are released under
[CC BY 4.0](LICENSES/CC-BY-4.0.txt), subject to any upstream rights in source
data represented by a derived artifact. See [LICENSES/README.md](LICENSES/README.md)
for the file-level boundary. Third-party data, papers, and upstream software
retain their own terms and are not relicensed by this repository.

The public author name is **Zhiyuan Zhang**. Citation metadata are provided in
[CITATION.cff](CITATION.cff). The project does not yet have an arXiv identifier
or DOI; none should be inferred from the working title or repository URL.

## Release status

- [x] Public repository names and author name fixed.
- [x] Claim boundaries and V1 negative result documented.
- [x] Apache-2.0 code license selected.
- [ ] Audited code and tests synchronized.
- [ ] V2 implementation and result audit completed.
- [ ] Two deterministic full runs compared.
- [ ] Manuscript and supplement claim-checked against artifacts.
- [ ] GitHub repository published and tagged.
- [ ] HF artifact repository published from the matching GitHub commit/tag.
- [ ] Cross-links verified in both repositories.

See the complete [release checklist](docs/RELEASE_CHECKLIST.md).
