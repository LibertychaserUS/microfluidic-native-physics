"""Analyzer for E6_NEGATIVE_CONTROLS_PROTOCOL_v1: shuffled-label decisions, chance level, fragility ledger over E1/E1b/E3/E4.

    python -I scaleup/w7/closed_loop_e6_analyze.py --run <e6 run dir> --work W [--e1 W/e1e2_run1 --e1-reanalysis W/e1_reanalysis --e1-dense W/e1_dense_sensitivity --e1b W/e1b_run1 --e3 W/e3_run1 --e4 W/e4_run1]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE.parent / "validation_plan" / "E6_NEGATIVE_CONTROLS_PROTOCOL_v1.md"
REFERENCE = "P1"
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


def read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    for k in ("e1", "e1-reanalysis", "e1-dense", "e1b", "e3", "e4"):
        ap.add_argument(f"--{k}", type=Path, default=None)
    args = ap.parse_args(argv)
    an = load_frozen_analyzer()
    run = args.run.resolve(); work = args.work.resolve()
    e1 = (args.e1 or work / "e1e2_run1").resolve(); e1r = (args.e1_reanalysis or work / "e1_reanalysis").resolve(); e1d = (args.e1_dense or work / "e1_dense_sensitivity").resolve()
    e1b = (args.e1b or work / "e1b_run1").resolve(); e3 = (args.e3 or work / "e3_run1").resolve(); e4 = (args.e4 or work / "e4_run1").resolve()
    rng_state = np.random.default_rng(an.BOOT_SEED)
    loops = pd.read_csv(run / "loops.csv"); chance = pd.read_csv(run / "chance.csv")
    results: dict = {"schema_version": "closed-loop-e6-analysis-v1", "protocol": PROTOCOL.name, "protocol_sha256": sha256_file(PROTOCOL),
                     "analyzer_sha256": sha256_file(Path(__file__).resolve()), "frozen_analyzer_sha256": sha256_file(HERE / "closed_loop_analyze.py"),
                     "inputs": {n: sha256_file(p / "results.json") for n, p in (("e1", e1), ("e1_reanalysis", e1r), ("e1_dense", e1d), ("e3", e3), ("e4", e4)) if (p / "results.json").exists()}}
    # ---- 1. shuffled-label decisions for all policies
    vals = loops.groupby(["table", "tier", "shift_type", "policy"])["t_star"].mean().reset_index()
    wide = vals.pivot_table(index=["table", "tier", "shift_type"], columns="policy", values="t_star").reset_index()
    tables = wide["table"].to_numpy(); policies = [p for p in wide.columns if p not in ("table", "tier", "shift_type")]
    sh, names, pvals = {}, [], []
    for p in policies:
        if p == REFERENCE:
            continue
        sh[p] = an.mean_ci(wide[p].to_numpy(float) - wide[REFERENCE].to_numpy(float), tables, rng_state); names.append(p); pvals.append(sh[p]["p_block"])
    for p, a in zip(names, an.bh(pvals)):
        e = sh[p]; lo, hi = e["ci_table_block"]; e["p_block_bh"] = a
        e["verdict"] = "better" if (e["mean"] < 0 and hi < 0) else ("worse" if (e["mean"] > 0 and lo > 0) else "not distinguishable")
        e["verdict_bh"] = e["verdict"] if a < an.ALPHA else "not distinguishable"
    results["shuffled"] = {"n_tiers": int(len(wide)), "levels_mean_t_star": {p: float(np.nanmean(wide[p])) for p in policies}, "delta_vs_P1": sh,
                           "artefact_candidates": [p for p in names if sh[p]["verdict_bh"] == "better" and p != "CEIL"],
                           "positive_control_CEIL": {"verdict_bh": sh["CEIL"]["verdict_bh"], "mean": sh["CEIL"]["mean"]} if "CEIL" in sh else None}
    # ---- 2. chance level by shift type beside E1 levels
    e1res = read_json(e1 / "results.json")["E1"]
    ch = chance.groupby(["table", "tier", "shift_type"])["chance_expected_t_star"].mean().reset_index()
    by_type = {}
    for st, g in ch.groupby("shift_type"):
        by_type[st] = {"chance": float(g["chance_expected_t_star"].mean()), "n_tiers": int(len(g)),
                       **{p: e1res["by_shift_type_mean_t_star"][st][p] for p in ("P0", "P1", "P2", "CEIL") if st in e1res["by_shift_type_mean_t_star"]}}
    results["chance_level"] = {"all": float(ch["chance_expected_t_star"].mean()), "by_shift_type": by_type, "E1_all": {p: e1res["levels"][p]["mean_t_star"] for p in ("P0", "P1", "P2", "CEIL")}}
    # ---- 3. fragility ledger
    rows = []
    def add(source, unit, d, shuffled_v=None):
        if d.get("verdict_bh") != "better":
            return
        rows.append({"source": source, "unit": unit, "delta": d["mean"], "ci_lo": d["ci_table_block"][0], "ci_hi": d["ci_table_block"][1],
                     "signs_under_drift": json.dumps(d.get("signs_under_drift")), "fragile": bool(d.get("fragile_under_drift_0p2")),
                     "shuffled_verdict": shuffled_v, "status": "positive control" if unit == "CEIL" else ("artefact" if shuffled_v == "better" else ("fragile" if d.get("fragile_under_drift_0p2") else "confirmed"))})
    e1rr = read_json(e1r / "results.json")["E1"]
    for p, d in e1rr["delta_vs_reference"].items():
        add("E1 all tiers", p, d, sh.get(p, {}).get("verdict_bh"))
    for p, d in read_json(e1d / "results_dense.json")["E1"]["delta_vs_reference"].items():
        add("E1 dense tiers", p, d, sh.get(p, {}).get("verdict_bh"))
    for cell in CELLS:
        cd = e1b / cell
        if (cd / "results.json").exists():
            for p, d in read_json(cd / "results.json")["E1"]["delta_vs_reference"].items():
                add(f"E1b {cell} all tiers", p, d, sh.get(p, {}).get("verdict_bh"))
        if (cd / "results_dense.json").exists():
            for p, d in read_json(cd / "results_dense.json")["E1"]["delta_vs_reference"].items():
                add(f"E1b {cell} dense tiers", p, d, sh.get(p, {}).get("verdict_bh"))
    if (e3 / "results.json").exists():
        for lam, v in read_json(e3 / "results.json")["per_lambda"].items():
            d = dict(v["delta_rule_minus_continue"]); d["verdict_bh"] = "better" if v["holds"] else "not better"
            d["signs_under_drift"] = [x["sign"] for x in v["drift"].values()]; d["fragile_under_drift_0p2"] = v["fragile_under_drift_0p2"]
            add("E3 rule vs continue", f"lambda={lam}", d)
    if (e4 / "results.json").exists():
        for f, d in read_json(e4 / "results.json")["E4_family_effect"]["P2"]["delta_vs_xgboost"].items():
            add("E4 family vs xgboost (P2)", f, d)
    ledger = pd.DataFrame(rows)
    ledger.to_csv(run / "fragility_ledger.csv", index=False)
    counts = ledger["status"].value_counts().to_dict() if len(ledger) else {}
    results["fragility_ledger"] = {"n_better_verdicts": int(len(ledger)), "counts": {k: int(v) for k, v in counts.items()},
                                   "entries": ledger.to_dict(orient="records")}
    (run / "results.json").write_text(json.dumps(an.sanitize(results), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hashes = json.loads((run / "artifact_hashes.json").read_text(encoding="utf-8"))
    for name in ("results.json", "fragility_ledger.csv"):
        hashes[name] = sha256_file(run / name)
    (run / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"artefact_candidates": results["shuffled"]["artefact_candidates"], "shuffled": {p: (sh[p]["verdict_bh"], round(sh[p]["mean"], 3)) for p in names},
                      "chance_all": round(results["chance_level"]["all"], 3), "ledger_counts": results["fragility_ledger"]["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
