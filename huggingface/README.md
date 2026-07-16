---

<!-- SPDX-License-Identifier: CC-BY-4.0 -->
pretty_name: Microfluidic Native Physics - Reproducibility Artifacts
language:
  - en
  - zh
license: other
task_categories:
  - tabular-regression
tags:
  - microfluidics
  - droplet-generation
  - distribution-shift
  - physics-guided-machine-learning
  - applicability-domain
  - conformal-prediction
---

# Microfluidic Native Physics — reproducibility artifacts

> **Publication pending.** This card is staged for the future Hugging Face
> dataset repository
> [Chaser111/microfluidic-native-physics](https://huggingface.co/datasets/Chaser111/microfluidic-native-physics).
> No artifact upload is VERIFIED until the matching GitHub commit/tag, hashes,
> source terms, and schemas are filled below.

The canonical project, implementation, protocols, tests, and issue history are
hosted on GitHub:

**[LibertychaserUS/microfluidic-native-physics](https://github.com/LibertychaserUS/microfluidic-native-physics)**

Author: **Zhiyuan Zhang**

## Scope

This repository is intended to mirror license-cleared reproducibility artifacts
for a retrospective public-data study of reliability under chemistry shift in
droplet microfluidics. Planned content includes split manifests, per-row model
predictions, applicability scores, uncertainty bands, strict metrics tables,
figure-source tables, and artifact hashes.

It is not a new wet-lab dataset. It does not establish chemical compatibility,
manufacturability, stable dripping, device safety, or prospective material or
formulation discovery.

## Evidence boundary

- **V1:** a complete TX100 chemistry group was sealed. The held-out result was
  negative (MAPE 19.8436%, R-squared -0.2436, nominal 90% interval coverage
  44.83%). The reveal required a disclosed serialization-only recovery after a
  strict JSON write rejected an undefined value. Frozen predictions and labels
  were not changed.
- **V2:** a native-variable physics-guided constrained regressor, not a
  classical PINN. TX100 was already revealed, so V2 results on TX100 are
  retrospective, not a second blind validation.
- **Search:** retrospective ranking/retrieval of historical candidates, not
  prospective experiment-free validation.

## Planned files

| Path | Description | Status |
|---|---|---|
| `manifests/` | Source, split, environment, and artifact hashes | pending |
| `predictions/` | Per-row predictions, intervals, and risk scores | pending |
| `metrics/` | Strict JSON/CSV endpoint summaries | pending |
| `figure_data/` | Tables used to regenerate paper figures | pending |
| `README.md` | This dataset card | staged |

Before publication, each file will list its schema, row unit, generating GitHub
commit, source-data dependency, license boundary, and SHA-256 hash.

## Source data

This mirror does not relicense upstream data.

- Chagot et al. article:
  [Lab on a Chip](https://pubs.rsc.org/en/content/articlehtml/2022/lc/d2lc00416j)
- Chagot public tables:
  [Zenodo 7055018](https://zenodo.org/records/7055018), whose record reports
  CC BY 4.0
- Lashkaripour et al. DAFD article:
  [Nature Communications](https://www.nature.com/articles/s41467-023-44068-3)
- DAFD deposit used by the reproduction:
  [OSF 938rs](https://osf.io/938rs/)

Only artifacts cleared under the applicable source terms will be uploaded.
The GitHub project's Apache-2.0 license applies to original software code,
tests, and configuration. Repository-authored report text and original
figures/results use CC BY 4.0. Neither license automatically applies to upstream
data or overrides rights retained in data-derived content.

## Reproducibility pointer

Pending release metadata:

```text
GitHub commit: PENDING
GitHub release/tag: PENDING
Artifact manifest SHA-256: PENDING
Schema version: PENDING
```

Reproduction commands and dependency locks will live in the canonical GitHub
repository. This HF repository will not contain an independent, drifting copy
of the implementation.

## Citation

Use the canonical GitHub
[`CITATION.cff`](https://github.com/LibertychaserUS/microfluidic-native-physics/blob/main/CITATION.cff)
after the first tagged release. No DOI or arXiv identifier has been assigned.

## 中文说明

这里计划只发布经过许可核验的复现实验产物，例如逐行预测、数据切分清单、
指标表和哈希。GitHub 是唯一主仓库，HF 不单独维护算法代码。

V1 的整组 TX100 留出测试是负结果；V2 是“原生变量 + 物理引导 + 硬约束”，
不是经典 PINN；所有搜索均为历史公开数据回放，没有做新的湿实验验证。
