# H_I_POOL_FEASIBILITY_GATE — Can the four droplet tables be aligned for pooled pre-training?

Written on 2026-10-22. The gate criteria (Section 1) were fixed before the per-table evidence
(Section 2) was tabulated; the evidence uses only column definitions, units and constants recorded
in the bindings, custodian manifests, source papers and deposits — no target value and no model
output. If the gate is not passed, H-I is registered as exited here and no pooled model is trained.

中文摘要:H-I(跨表汇合预训练)的可行性门。四张液滴表要对齐到交集物理特征(连续相毛细管数、流量比、
黏度比、特征尺寸)与可比的无量纲目标(液滴直径/特征尺寸),单位与定义逐表核对。判据先写死,再填证据。

## 1. Gate criteria (fixed before tabulation)

For each of D1, D2, A1 and R2-A1 the table is **fully aligned** if all of the following hold with
stated units: (a) a continuous-phase capillary number Ca_c computable from recorded columns or from
constants stated in the source paper, with an explicit velocity definition; (b) the flow-rate ratio
q = Q_d/Q_c; (c) the viscosity ratio λ = μ_d/μ_c that varies within the table or is a stated constant;
(d) a characteristic length L of the generating junction in micrometres; (e) a droplet **diameter**
of unconfined (dripping/jetting) droplets in a stated unit, so that d/L is dimensionless and comparable
across tables. A table is **partially aligned** if (a), (b), (d), (e) hold and only (c) is a constant.
A table is **not aligned** if any of (a), (b), (d) or (e) fails.

The gate **passes** iff at least three of the four tables are fully or partially aligned, so that a
leave-one-table-out pooled model has at least two source tables. Otherwise H-I exits; the exit is
registered and reported.

## 2. Evidence per table

| Table | (a) Ca_c | (b) q | (c) λ | (d) L | (e) target | Verdict |
|---|---|---|---|---|---|---|
| D1 (DAFD 3.0, OSF 938rs) | recorded column `Capillary number` (continuous-phase Ca as defined in the DAFD 3.0 Methods) | recorded `Flow rate ratio` (DAFD definition Φ) | recorded `viscosity ratio` (DAFD Table 1: continuous/dispersed; the reciprocal of μ_d/μ_c, invertible) | `Orifice width (um)`; hydraulic diameter also recorded | `Observed droplet diameter (um)` of dripping droplets; `Normalized droplet diameter` = d/D_h | **fully aligned** |
| D2 (Chagot et al. 2022, Zenodo 7055018) | computable: μ_c = 4.6 mPa·s (paper constant used by the manuscript's anchor model), U_c = Q_c/A with the oval junction area π·195·190/4 µm², γ recorded per row | computable from recorded Q_d, Q_c (ml/min) | constant: μ_d = 6.8 mPa·s, μ_c = 4.6 mPa·s (paper constants); no variation | 195 µm junction width (paper constant) | droplet diameter in mm; all droplets below the 0.195 mm channel depth (manuscript, Section 4) | **partially aligned** (λ and L constant) |
| A1 (Hormozinezhad et al. 2025, ESI) | recorded `Ca_c` (workbook column; the non-Newtonian phase viscosity is shear-dependent, so Ca_c of the HA sheets is an effective quantity defined by the authors) | recorded `q=Qd/Qc` | recorded `Eta_d`/`Eta_c` per sheet | `Width [um]` and `Etching Depth [um]` | `L_um` is the **droplet length** of confined plugs (may exceed the channel width; binding `hard_bound_note`); no diameter of an unconfined droplet exists; the regime is viscoelastic (Wi, El recorded), outside the Newtonian dripping regime of D1/D2 | **not aligned** — (e) fails |
| R2-A1 (Wang et al. 2024, Zenodo 10526441) | computable only up to an unknown constant: the deposit states no unit for `disp_flow`/`cont_flow` (binding `unit_note`); the deposited training script `FrameworkA.py` names the columns but states no units | computable (unit cancels) | recorded `lambda_mu` | `inner`, `outer`, `channel` recorded in metres (9e-05, 2e-04, 2e-03) | `diameter` column with **no stated unit** in the deposit or the script; d/L therefore has an unknown scale factor | **not aligned** — (a) and (e) fail on units |

## 3. Verdict

Fully aligned: D1. Partially aligned: D2. Not aligned: A1 (target is a plug length, not a droplet
diameter), R2-A1 (flow-rate and diameter units are not stated anywhere in the deposit). Two of four
tables are aligned, below the threshold of three. **The gate is not passed; H-I exits here.**

What would re-open it: a statement of the flow-rate and diameter units of R2-A1 from the source
paper (Chem. Eng. J. 485:149467, not open access to this session) would make R2-A1 fully aligned and
the gate would pass with D1, D2 and R2-A1. A1 cannot be aligned on (e) by any unit statement.

## 4. Register

| ID | Entry |
|---|---|
| H-I-G1 | 2026-10-22: gate evaluated on metadata only; verdict "not passed"; no pooled model trained; no pre-registration written. Re-evaluation requires the R2-A1 unit statement above. |
