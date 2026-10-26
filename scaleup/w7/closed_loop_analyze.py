"""Analyzer for E0_E1_E2_CLOSED_LOOP_PROTOCOL_v1 (Sections 2–4 decisions).

    python -I scaleup/w7/closed_loop_analyze.py --run <e1e2 run dir> --e0 <e0 run dir>

Deterministic from loops.csv, budget.csv and tiers_e0.csv; writes results.json, policy_table.csv,
budget_table.csv into the run directory and appends their digests to artifact_hashes.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

B = 2000
BOOT_SEED = 0
ALPHA = 0.05
REFERENCE = "P1"
EPS_MULT = 1.25
E2_K = (1, 2, 3, 5, 8, 10, 15, 20)
QUESTIONS = {"Q3_acquisition": [("P4", "P1"), ("P5", "P1"), ("P6", "P1"), ("P10", "P9"), ("P11", "P9")],
             "Q2_k0": [("P2", "P1"), ("P3", "P1")],
             "Q4_adaptation": [("P7", "P1"), ("P8", "P1"), ("P12", "P1")],
             "proposal": [("P9", "P1"), ("P10", "P3"), ("P11", "P6")],
             "bounds": [("P0", "P1"), ("CEIL", "P1")]}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sanitize(v):
    if isinstance(v, dict):
        return {str(k): sanitize(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [sanitize(x) for x in v]
    if isinstance(v, (np.floating, float)):
        return None if not math.isfinite(float(v)) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def boot_indices(tables: np.ndarray, rng: np.random.Generator, kind: str) -> np.ndarray:
    levels = sorted(set(tables.tolist()))
    if kind == "table":
        chosen = rng.choice(len(levels), size=len(levels), replace=True)
        return np.concatenate([np.flatnonzero(tables == levels[c]) for c in chosen])
    return np.concatenate([rng.choice(np.flatnonzero(tables == lv), size=int((tables == lv).sum()), replace=True) for lv in levels])


def mean_ci(d: np.ndarray, tables: np.ndarray, rng_state: np.random.Generator) -> dict:
    ok = np.isfinite(d)
    d, tables = d[ok], tables[ok]
    if len(d) == 0:
        return {"mean": float("nan"), "ci_table_block": [float("nan")] * 2, "ci_tier_stratified": [float("nan")] * 2, "n": 0, "p_block": float("nan")}
    rng = np.random.default_rng(rng_state.integers(0, 2**32 - 1))
    bt = np.array([np.mean(d[boot_indices(tables, rng, "table")]) for _ in range(B)])
    rng = np.random.default_rng(rng_state.integers(0, 2**32 - 1))
    bs = np.array([np.mean(d[boot_indices(tables, rng, "tier")]) for _ in range(B)])
    return {"mean": float(np.mean(d)), "ci_table_block": [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
            "ci_tier_stratified": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n": int(len(d)),
            "p_block": float(2.0 * min(np.mean(bt <= 0), np.mean(bt >= 0)))}


def bh(pvals: list[float]) -> list[float]:
    p = np.where(np.isfinite(np.asarray(pvals, float)), np.asarray(pvals, float), 1.0)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        running = min(running, p[i] * n / rank)
        adj[i] = running
    return [float(v) for v in adj]


def tier_values(loops: pd.DataFrame, kind: str, delta: float) -> pd.DataFrame:
    sub = loops[(loops.kind == kind) & (np.isclose(loops.delta, delta))]
    g = sub.groupby(["table", "tier", "shift_type", "policy"], sort=True)
    out = g.agg(t_star=("t_star", "mean"), regret=("regret", "mean"), success=("success", "mean"), n=("t_star", "size")).reset_index()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--e0", type=Path, required=True)
    args = ap.parse_args(argv)
    run = args.run.resolve()
    loops = pd.read_csv(run / "loops.csv")
    budget = pd.read_csv(run / "budget.csv")
    tiers_e0 = pd.read_csv(args.e0 / "tiers_e0.csv")
    rng_state = np.random.default_rng(BOOT_SEED)
    results: dict = {"schema_version": "closed-loop-analysis-v1", "protocol_sha256_from_run": json.loads((run / "run_manifest.json").read_text())["protocol_sha256"],
                     "analyzer_sha256": sha256_file(Path(__file__).resolve())}

    # ---- E1 main decisions
    main_vals = tier_values(loops, "main", 0.0)
    wide = main_vals.pivot_table(index=["table", "tier", "shift_type"], columns="policy", values="t_star").reset_index()
    tables = wide["table"].to_numpy()
    policies = [p for p in wide.columns if p not in ("table", "tier", "shift_type")]
    deltas = {}
    pvals = []
    names = []
    for p in policies:
        if p == REFERENCE:
            continue
        d = wide[p].to_numpy(float) - wide[REFERENCE].to_numpy(float)
        deltas[p] = mean_ci(d, tables, rng_state)
        names.append(p)
        pvals.append(deltas[p]["p_block"])
    adj = bh(pvals)
    for p, a in zip(names, adj):
        e = deltas[p]
        lo, hi = e["ci_table_block"]
        e["p_block_bh"] = a
        e["verdict"] = "better" if (e["mean"] < 0 and hi < 0) else ("worse" if (e["mean"] > 0 and lo > 0) else "not distinguishable")
        e["verdict_bh"] = e["verdict"] if a < ALPHA else "not distinguishable"
    # drift fragility
    fragile = {}
    for p in names:
        signs = []
        for delta in (-0.2, 0.2):
            dv = tier_values(loops, "drift_pre", delta).pivot_table(index=["table", "tier"], columns="policy", values="t_star")
            if p in dv.columns and REFERENCE in dv.columns:
                signs.append(float(np.sign(np.nanmean(dv[p] - dv[REFERENCE]))))
        base_sign = float(np.sign(deltas[p]["mean"]))
        fragile[p] = {"signs_under_drift": signs, "base_sign": base_sign, "fragile": bool(any(s != base_sign and s != 0 for s in signs))}
        deltas[p]["fragile_under_drift_0p2"] = fragile[p]["fragile"]
    # per-policy absolute levels
    levels = {p: {"mean_t_star": float(np.nanmean(wide[p])), "median_t_star": float(np.nanmedian(wide[p])),
                  "success_rate": float(main_vals[main_vals.policy == p]["success"].mean())} for p in policies}
    by_type = {}
    for st, g in wide.groupby("shift_type"):
        by_type[st] = {p: float(np.nanmean(g[p])) for p in policies}
        by_type[st]["n_tiers"] = int(len(g))
    dense_map = tiers_e0.set_index(tiers_e0["table"] + "|" + tiers_e0["tier"])["dense"] if "dense" in tiers_e0.columns else None
    by_density = {}
    if dense_map is not None:
        wide["dense"] = (wide["table"] + "|" + wide["tier"]).map(dense_map).to_numpy()
        for lab, g in wide.groupby("dense"):
            by_density["dense" if lab == 1 else "sparse"] = {p: float(np.nanmean(g[p])) for p in policies}
            by_density["dense" if lab == 1 else "sparse"]["n_tiers"] = int(len(g))
    questions = {}
    for qname, pairs in QUESTIONS.items():
        questions[qname] = {}
        for p, ref in pairs:
            if p in wide.columns and ref in wide.columns:
                d = wide[p].to_numpy(float) - wide[ref].to_numpy(float)
                questions[qname][f"{p}_vs_{ref}"] = mean_ci(d, tables, rng_state)
    # shuffled-label control
    sh = tier_values(loops, "shuffled", 0.0).pivot_table(index=["table", "tier"], columns="policy", values="t_star").reset_index()
    control = {}
    if "P1" in sh.columns and "P0" in sh.columns:
        control = mean_ci(sh["P1"].to_numpy(float) - sh["P0"].to_numpy(float), sh["table"].to_numpy(), rng_state)
        lo, hi = control["ci_table_block"]
        control["simulator_accepted"] = bool(math.isfinite(lo) and lo <= 0 <= hi)
    # descriptive ±0.5
    desc = {}
    for delta in (-0.5, 0.5):
        dv = tier_values(loops, "drift_desc", delta)
        if len(dv):
            desc[str(delta)] = {p: float(dv[dv.policy == p]["t_star"].mean()) for p in sorted(set(dv.policy))}
    results["E1"] = {"n_tiers": int(len(wide)), "reference": REFERENCE, "levels": levels, "delta_vs_reference": deltas,
                     "by_shift_type_mean_t_star": by_type, "by_density_mean_t_star": by_density, "questions": questions,
                     "shuffled_control_P1_minus_P0": control, "drift_descriptive_0p5": desc}
    pd.DataFrame([{"policy": p, **levels[p], **({k: v for k, v in deltas[p].items() if k in ("mean", "ci_table_block", "verdict", "verdict_bh", "fragile_under_drift_0p2")} if p in deltas else {})}
                  for p in policies]).to_csv(run / "policy_table.csv", index=False)

    # ---- E2 budgets
    ref = tiers_e0.set_index(tiers_e0["table"] + "|" + tiers_e0["tier"])["dev_ref_mape"]
    budget["key"] = budget["table"] + "|" + budget["tier"]
    med = budget.groupby(["key", "shift_type", "method", "k"])["mape"].median().reset_index()
    kstar_rows = []
    for (key, st, method), g in med.groupby(["key", "shift_type", "method"]):
        eps = EPS_MULT * float(ref[key])
        reached = sorted(int(k) for k, m in zip(g["k"], g["mape"]) if m <= eps)
        kstar_rows.append({"key": key, "shift_type": st, "method": method, "eps": eps, "k_star": reached[0] if reached else 21,
                           "reached": bool(reached), "mape_at_k": {str(int(k)): float(m) for k, m in zip(g["k"], g["mape"])}})
    ks = pd.DataFrame(kstar_rows)
    e2 = {}
    if dense_map is not None:
        ks["dense"] = ks["key"].map(dense_map).to_numpy()
        for (lab, method), g in ks.groupby(["dense", "method"]):
            e2[f"{'dense' if lab == 1 else 'sparse'}|{method}"] = {"n_tiers": int(len(g)), "median_k_star": float(np.median(g["k_star"])), "p90_k_star": float(np.quantile(g["k_star"], 0.9)),
                                                                    "fraction_reached_within_20": float(np.mean(g["reached"]))}
    for (st, method), g in ks.groupby(["shift_type", "method"]):
        e2[f"{st}|{method}"] = {"n_tiers": int(len(g)), "median_k_star": float(np.median(g["k_star"])), "p90_k_star": float(np.quantile(g["k_star"], 0.9)),
                                "fraction_reached_within_20": float(np.mean(g["reached"]))}
    for method, g in ks.groupby("method"):
        e2[f"all|{method}"] = {"n_tiers": int(len(g)), "median_k_star": float(np.median(g["k_star"])), "p90_k_star": float(np.quantile(g["k_star"], 0.9)),
                               "fraction_reached_within_20": float(np.mean(g["reached"]))}
    results["E2"] = {"eps_rule": "1.25 x dev-ref (OOF grouped MAPE of the source constructor on the source rows)", "summary": e2}
    ks.drop(columns=["mape_at_k"]).to_csv(run / "budget_table.csv", index=False)
    (run / "results.json").write_text(json.dumps(sanitize(results), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hashes = json.loads((run / "artifact_hashes.json").read_text(encoding="utf-8"))
    for name in ("results.json", "policy_table.csv", "budget_table.csv"):
        hashes[name] = sha256_file(run / name)
    (run / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({p: (deltas[p]["verdict_bh"], round(deltas[p]["mean"], 3)) for p in names}))
    print(json.dumps({"control_accepted": control.get("simulator_accepted"), "levels_P1": levels.get("P1"), "levels_CEIL": levels.get("CEIL")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
