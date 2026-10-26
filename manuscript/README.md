# Manuscript source

`main.tex` / `references.bib` / `main.bbl` — *When Can a Learned Design Surrogate
Be Trusted in a Self-Driving Microfluidics Laboratory? An Offline Closed-Loop
Framework Built and Tested on Six Public Tables* (English);
`main_zh.tex` / `main_zh.bbl` — paragraph-by-paragraph Chinese mirror (the
English file is authoritative where they differ). Text and original figures are
released under CC BY 4.0 (`../LICENSES/CC-BY-4.0.txt`).

```bash
tectonic -X compile main.tex
tectonic -X compile main_zh.tex     # ctexart; XeTeX with the system CJK fonts
```

## Number reconciliation

`numbers_ledger.json` lists every printed number together with the artifact
file and the JSON path or CSV selector it is recomputed from.
`tools/check_numbers.py` (standard library only) fails if a printed value
disagrees with the artifact at the stated precision, if it is missing from the
`.tex`, if a percentage that needs its denominator lacks the denominator
wording nearby, or if any truncated digest `\hashtrunc{prefix}{suffix}` does
not match one registered SHA-256 on both ends.

```bash
python3 tools/check_numbers.py --work ../work/microfluidic_reliability --skip-runpair
python3 tools/check_numbers.py --work ../work/microfluidic_reliability --skip-runpair \
    --tex main_zh.tex --literal-field literal_zh --cooccur-field cooccur_zh
```

`--skip-runpair` skips only the re-hashing of per-row files that are not in
this repository; `../scripts/verify_run_manifests.py` performs that check on
the released part of each run pair.

## Figures

`figures/` holds byte copies of frozen workspace figures
(`data/copied_artifacts.sha256`), the figures re-rendered from the same
artifact data by `tools/render_figures.py` (`data/rerendered_figures.sha256`)
and the four closed-loop figures drawn by `../scaleup/w7/figures_closed_loop.py`
(plotted values in `../scaleup/e_closed_loop/figures_closed_loop/values.json`);
the figure of V2 paired differences is drawn by pgfplots from
`data/paired_group_differences_v2p_vs_b1.csv`. Appendix C of the manuscript
names the artifact file behind every table and figure.
