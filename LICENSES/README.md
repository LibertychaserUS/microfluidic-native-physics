<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# License map

This repository uses a deliberate dual-license boundary.

## Apache-2.0

The root [`LICENSE`](../LICENSE) applies to repository-authored software,
including:

- `work/microfluidic_reliability/src/**` and `work/microfluidic_reliability/tests/**`;
- `scaleup/w3/**`;
- `scripts/**` and `manuscript/tools/**`;
- `tests/**`, `requirements/**`, `pyproject.toml` and the workflow configuration; and
- code or configuration snippets that are explicitly marked `Apache-2.0`.

## CC BY 4.0

[`CC-BY-4.0.txt`](CC-BY-4.0.txt) applies to repository-authored research
content, including:

- `README.md`, the readme files of the subdirectories and the prose under `docs/**`;
- the frozen protocols, deviation registers and incident records (`*.md` under
  `work/microfluidic_reliability/` and `scaleup/`);
- the manuscript text, bibliography and original figures under `manuscript/**`; and
- repository-generated result summaries, manifests and figures under
  `work/microfluidic_reliability/**` and `hashes/**`, only to the extent
  Zhiyuan Zhang holds the relevant rights (they carry inputs of public
  third-party rows; see below).

Attribution should identify **Zhiyuan Zhang**, the work title, the repository
URL, the CC BY 4.0 license, and whether changes were made.

## Upstream material is excluded

These grants do not cover third-party papers, raw source data, upstream code,
upstream figures, trademarks, or any third-party content embedded in a derived
artifact. Upstream terms continue to apply. See
[`THIRD_PARTY_DATA.md`](../THIRD_PARTY_DATA.md).

When a file contains mixed material, its own attribution/notice controls and
must identify the excluded third-party portion. If the boundary is unclear, do
not redistribute the file until rights review is complete.
