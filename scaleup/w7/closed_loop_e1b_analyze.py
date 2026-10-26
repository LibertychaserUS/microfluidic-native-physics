"""Analyzer for E1B_DIFFICULTY_PROTOCOL_v1 (§2 decisions per cell, §3 grading, §4 dense-subset sensitivity).

    python -I scaleup/w7/closed_loop_e1b_analyze.py cell --run <cell dir | e1e2 run dir> --e0 <e0 dir> [--restrict-dense] [--out <dir>]
    python -I scaleup/w7/closed_loop_e1b_analyze.py summary --e1 <e1e2 run dir> --e1b <e1b run dir> --e1-dense <dir>

Reuses the frozen E1 analyzer's bootstrap, BH and aggregation functions (`closed_loop_analyze.py`, unchanged).
`cell` writes results[_dense].json and policy_table[_dense].csv into --out (default: the run dir) and records
their digests in that directory's artifact_hashes.json; `summary` writes e1b_summary.json into the E1b run dir.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE.parent / "validation_plan" / "E1B_DIFFICULTY_PROTOCOL_v1.md"
GRADE_MIN_GAIN = 0.10  # §3(b): an acquisition gain smaller than this keeps F15
F17_POLICIES = ("P0", "P1", "P2")
CELLS = ("cell_tau0.05_B20", "cell_tau0.10_B10", "cell_tau0.05_B10")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_analyzer():
    spec = importlib.util.spec_from_file_location("closed_loop_analyze", HERE / "closed_loop_analyze.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["closed_loop_analyze"] = module
    spec.loader.exec_module(module)
    return module


def analyze_e1(an, loops: pd.DataFrame, tiers_e0: pd.DataFrame, restrict_dense: bool) -> tuple[dict, pd.DataFrame]:
    key_e0 = tiers_e0["table"] + "|" + tiers_e0["tier"]
    dense_map = tiers_e0.set_index(key_e0)["dense"]
    keys = loops["table"] + "|" + loops["tier"]
    if restrict_dense:
        loops = loops[keys.map(dense_map).fillna(0).astype(int) == 1].reset_index(drop=True)
    rng_state = np.random.default_rng(an.BOOT_SEED)
    main_vals = an.tier_values(loops, "main", 0.0)
    wide = main_vals.pivot_table(index=["table", "tier", "shift_type"], columns="policy", values="t_star").reset_index()
    tables = wide["table"].to_numpy()
    policies = [p for p in wide.columns if p not in ("table", "tier", "shift_type")]
    deltas, pvals, names = {}, [], []
    for p in policies:
        if p == an.REFERENCE:
            continue
        d = wide[p].to_numpy(float) - wide[an.REFERENCE].to_numpy(float)
        deltas[p] = an.mean_ci(d, tables, rng_state)
        names.append(p)
        pvals.append(deltas[p]["p_block"])
    for p, a in zip(names, an.bh(pvals)):
        e = deltas[p]
        lo, hi = e["ci_table_block"]
        e["p_block_bh"] = a
        e["verdict"] = "better" if (e["mean"] < 0 and hi < 0) else ("worse" if (e["mean"] > 0 and lo > 0) else "not distinguishable")
        e["verdict_bh"] = e["verdict"] if a < an.ALPHA else "not distinguishable"
    drift_levels = {}
    for p in names:
        signs = []
        for delta in (-0.2, 0.2):
            dv = an.tier_values(loops, "drift_pre", delta).pivot_table(index=["table", "tier"], columns="policy", values="t_star")
            if p in dv.columns and an.REFERENCE in dv.columns:
                signs.append(float(np.sign(np.nanmean(dv[p] - dv[an.REFERENCE]))))
        base_sign = float(np.sign(deltas[p]["mean"]))
        deltas[p]["signs_under_drift"] = signs
        deltas[p]["fragile_under_drift_0p2"] = bool(any(s != base_sign and s != 0 for s in signs))
    for delta in (-0.2, 0.2):
        dv = an.tier_values(loops, "drift_pre", delta)
        drift_levels[str(delta)] = {p: float(dv[dv.policy == p]["t_star"].mean()) for p in sorted(set(dv.policy))}
    levels = {p: {"mean_t_star": float(np.nanmean(wide[p])), "median_t_star": float(np.nanmedian(wide[p])),
                  "success_rate": float(main_vals[main_vals.policy == p]["success"].mean())} for p in policies}
    by_type = {}
    for st, g in wide.groupby("shift_type"):
        by_type[st] = {p: float(np.nanmean(g[p])) for p in policies}
        by_type[st]["n_tiers"] = int(len(g))
    wide["dense"] = (wide["table"] + "|" + wide["tier"]).map(dense_map).to_numpy()
    by_density = {}
    for lab, g in wide.groupby("dense"):
        by_density["dense" if lab == 1 else "sparse"] = {p: float(np.nanmean(g[p])) for p in policies}
        by_density["dense" if lab == 1 else "sparse"]["n_tiers"] = int(len(g))
    questions = {}
    for qname, pairs in an.QUESTIONS.items():
        questions[qname] = {}
        for p, ref in pairs:
            if p in wide.columns and ref in wide.columns:
                d = wide[p].to_numpy(float) - wide[ref].to_numpy(float)
                questions[qname][f"{p}_vs_{ref}"] = an.mean_ci(d, tables, rng_state)
    sh = an.tier_values(loops, "shuffled", 0.0).pivot_table(index=["table", "tier"], columns="policy", values="t_star").reset_index()
    control = {}
    if "P1" in sh.columns and "P0" in sh.columns:
        control = an.mean_ci(sh["P1"].to_numpy(float) - sh["P0"].to_numpy(float), sh["table"].to_numpy(), rng_state)
        lo, hi = control["ci_table_block"]
        control["simulator_accepted"] = bool(math.isfinite(lo) and lo <= 0 <= hi)
    e1 = {"n_tiers": int(len(wide)), "n_tables": int(len(set(tables.tolist()))), "restricted_to_dense": bool(restrict_dense), "reference": an.REFERENCE,
          "levels": levels, "delta_vs_reference": deltas, "by_shift_type_mean_t_star": by_type, "by_density_mean_t_star": by_density,
          "questions": questions, "shuffled_control_P1_minus_P0": control, "drift_pre_mean_t_star": drift_levels}
    table = pd.DataFrame([{"policy": p, **levels[p], **({k: v for k, v in deltas[p].items() if k in ("mean", "ci_table_block", "verdict", "verdict_bh", "fragile_under_drift_0p2")} if p in deltas else {})}
                          for p in policies])
    return e1, table


def record_hashes(out: Path, names: list[str]) -> None:
    hp = out / "artifact_hashes.json"
    hashes = json.loads(hp.read_text(encoding="utf-8")) if hp.exists() else {}
    for name in names:
        hashes[name] = sha256_file(out / name)
    hp.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cmd_cell(args) -> int:
    an = load_frozen_analyzer()
    run = args.run.resolve()
    out = (args.out or run).resolve()
    out.mkdir(parents=True, exist_ok=True)
    loops = pd.read_csv(run / "loops.csv")
    tiers_e0 = pd.read_csv(args.e0 / "tiers_e0.csv")
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    e1, table = analyze_e1(an, loops, tiers_e0, args.restrict_dense)
    results = {"schema_version": "closed-loop-e1b-analysis-v1", "protocol": PROTOCOL.name, "protocol_sha256": sha256_file(PROTOCOL),
               "analyzer_sha256": sha256_file(Path(__file__).resolve()), "frozen_analyzer_sha256": sha256_file(HERE / "closed_loop_analyze.py"),
               "run": run.name, "run_manifest_sha256": sha256_file(run / "run_manifest.json"), "tau": manifest.get("tau"), "budget": manifest.get("budget"),
               "E1": e1}
    tag = "_dense" if args.restrict_dense else ""
    (out / f"results{tag}.json").write_text(json.dumps(an.sanitize(results), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    table.to_csv(out / f"policy_table{tag}.csv", index=False)
    record_hashes(out, [f"results{tag}.json", f"policy_table{tag}.csv"])
    d = e1["delta_vs_reference"]
    print(json.dumps({"run": run.name, "dense": args.restrict_dense, "n_tiers": e1["n_tiers"], "control_accepted": e1["shuffled_control_P1_minus_P0"].get("simulator_accepted"),
                      "P1_mean": round(e1["levels"]["P1"]["mean_t_star"], 3),
                      "verdicts": {p: (d[p]["verdict_bh"], round(d[p]["mean"], 3), d[p]["fragile_under_drift_0p2"]) for p in sorted(d)}}))
    return 0


def grade_cell(e1: dict) -> dict:
    d = e1["delta_vs_reference"]
    accepted = bool(e1["shuffled_control_P1_minus_P0"].get("simulator_accepted"))
    f14 = d["P2"]["verdict_bh"] == "better" and not d["P2"]["fragile_under_drift_0p2"]
    f15_breakers = [p for p in ("P4", "P5", "P6") if d[p]["verdict_bh"] == "better" and d[p]["mean"] < -GRADE_MIN_GAIN and not d[p]["fragile_under_drift_0p2"]]
    f16 = d["P7"]["verdict_bh"] == "worse" and d["P8"]["verdict_bh"] == "worse"
    f16p = d["P9"]["verdict_bh"] == "worse"
    lv = e1["levels"]
    order0 = sorted(F17_POLICIES, key=lambda p: lv[p]["mean_t_star"])
    f17_orders = {"0": order0}
    f17 = True
    for delta, levels in e1["drift_pre_mean_t_star"].items():
        order = sorted(F17_POLICIES, key=lambda p: levels[p])
        f17_orders[delta] = order
        f17 = f17 and order == order0
    return {"control_accepted": accepted, "F14_k0_zero": bool(f14), "F15_no_acquisition_gain": not f15_breakers, "F15_breakers": f15_breakers,
            "F16_warm_dominates": bool(f16), "F16p_portfolio_worse": bool(f16p), "F17_ranking_unchanged": bool(f17), "F17_orders": f17_orders,
            "ceil_mean_t_star": {"0": lv["CEIL"]["mean_t_star"], **{dl: v["CEIL"] for dl, v in e1["drift_pre_mean_t_star"].items()}},
            "P1_mean_t_star": lv["P1"]["mean_t_star"], "P2_delta": d["P2"]["mean"], "P2_ci": d["P2"]["ci_table_block"], "P2_verdict": d["P2"]["verdict_bh"],
            "P7_delta": d["P7"]["mean"], "P7_verdict": d["P7"]["verdict_bh"], "P8_delta": d["P8"]["mean"], "P8_verdict": d["P8"]["verdict_bh"],
            "P9_delta": d["P9"]["mean"], "P9_verdict": d["P9"]["verdict_bh"],
            "acq": {p: (d[p]["verdict_bh"], d[p]["mean"], d[p]["fragile_under_drift_0p2"]) for p in ("P4", "P5", "P6")}}


def grade_label(n_hold: int, n_cells: int) -> str:
    if n_cells == 3 and n_hold == 3:
        return "robust to difficulty"
    if n_cells == 3 and n_hold == 2:
        return "mostly robust"
    if n_cells == 3:
        return "specific to the E1 setting"
    return f"holds in {n_hold} of {n_cells} accepted cells"


def cmd_summary(args) -> int:
    an = load_frozen_analyzer()
    e1b = args.e1b.resolve()
    cells = {}
    e1_res = json.loads((args.e1.resolve() / "results.json").read_text(encoding="utf-8"))["E1"]
    cells["E1 (tau0.10_B20)"] = {"all": grade_cell(e1_res)}
    dense_path = args.e1_dense.resolve() / "results_dense.json"
    if dense_path.exists():
        cells["E1 (tau0.10_B20)"]["dense"] = grade_cell(json.loads(dense_path.read_text(encoding="utf-8"))["E1"])
    for cell in CELLS:
        cd = e1b / cell
        if not (cd / "results.json").exists():
            cells[cell] = {"not_run": True}
            continue
        cells[cell] = {"all": grade_cell(json.loads((cd / "results.json").read_text(encoding="utf-8"))["E1"])}
        if (cd / "results_dense.json").exists():
            cells[cell]["dense"] = grade_cell(json.loads((cd / "results_dense.json").read_text(encoding="utf-8"))["E1"])
    new_cells = [c for c in CELLS if "all" in cells.get(c, {})]
    accepted = [c for c in new_cells if cells[c]["all"]["control_accepted"]]
    grades = {}
    for key, label in (("F14_k0_zero", "F14"), ("F15_no_acquisition_gain", "F15"), ("F16_warm_dominates", "F16"), ("F16p_portfolio_worse", "F16p"), ("F17_ranking_unchanged", "F17")):
        holds = [c for c in accepted if cells[c]["all"][key]]
        fails = [c for c in accepted if not cells[c]["all"][key]]
        grades[label] = {"holds_in": holds, "fails_in": fails, "n_accepted_cells": len(accepted), "grade": grade_label(len(holds), len(accepted)),
                         "dense_subset_holds_in": [c for c in ["E1 (tau0.10_B20)"] + new_cells if "dense" in cells[c] and cells[c]["dense"][key]]}
    summary = {"schema_version": "closed-loop-e1b-summary-v1", "protocol_sha256": sha256_file(PROTOCOL), "analyzer_sha256": sha256_file(Path(__file__).resolve()),
               "cells": cells, "accepted_new_cells": accepted, "grades": grades}
    (e1b / "e1b_summary.json").write_text(json.dumps(an.sanitize(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record_hashes(e1b, ["e1b_summary.json"])
    print(json.dumps({k: (v["grade"], v["fails_in"]) for k, v in grades.items()}))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cell")
    c.add_argument("--run", type=Path, required=True)
    c.add_argument("--e0", type=Path, required=True)
    c.add_argument("--restrict-dense", action="store_true")
    c.add_argument("--out", type=Path, default=None)
    s = sub.add_parser("summary")
    s.add_argument("--e1", type=Path, required=True)
    s.add_argument("--e1b", type=Path, required=True)
    s.add_argument("--e1-dense", type=Path, required=True)
    args = ap.parse_args(argv)
    return cmd_cell(args) if args.cmd == "cell" else cmd_summary(args)


if __name__ == "__main__":
    raise SystemExit(main())
