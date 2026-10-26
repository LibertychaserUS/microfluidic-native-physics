"""Analyzer for E7_HARD_TASK_PROTOCOL_v1 (§3 decisions H7a–H7f and the descriptive tables).

    python -I scaleup/w7/closed_loop_e7_analyze.py --run <e7 run dir>

Deterministic from loops.csv and units.csv; writes results.json and policy_table.csv into the run directory and
records their digests in artifact_hashes.json.  Bootstrap, BH and sanitising come from the frozen E1 analyzer;
the rule family and table rotation come from the frozen E3 analyzer.
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
PROTOCOL = HERE.parent / "validation_plan" / "E7_HARD_TASK_PROTOCOL_v1.md"
REFERENCE = "P1"
NON_ARM = ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P13", "CEIL")
ARMS = {"A_cont": "continue", "A_none": "none", "A_switch": "switch", "A_esc": "escalate"}
DIAGS = ("d1_bias_ratio", "d2_magnitude_ratio", "d3_width_ratio", "d4_support_outside", "d5_sign_consistency")
HARD_MAX = 0.10
MIN_HARD_TIERS = 30
MIN_EXTRAP_TIERS = 20
LAMBDAS = (0, 5, 10)
UKEY = ["table", "tier", "target"]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tier_values(loops: pd.DataFrame, keys: set, kind: str, delta: float) -> pd.DataFrame:
    sub = loops[(loops.kind == kind) & np.isclose(loops.delta, delta) & loops.policy.isin(NON_ARM)]
    sub = sub[(sub.table + "|" + sub.tier + "|" + sub.target).isin(keys)]
    g = sub.groupby(["table", "tier", "shift_type", "policy"], sort=True).agg(t_star=("t_star", "mean"), success=("success", "mean")).reset_index()
    return g


def decision_block(an, loops: pd.DataFrame, keys: set, rng_state, chance_tier: pd.Series | None) -> dict:
    main = tier_values(loops, keys, "main", 0.0)
    wide = main.pivot_table(index=["table", "tier", "shift_type"], columns="policy", values="t_star").reset_index()
    tables = wide["table"].to_numpy()
    out: dict = {"n_tiers": int(len(wide)), "n_tables": int(len(set(tables.tolist())))}
    if len(wide) == 0:
        return out
    pols = [p for p in NON_ARM if p in wide.columns]
    out["levels_mean_t_star"] = {p: float(np.nanmean(wide[p])) for p in pols}
    out["success_rate"] = {p: float(main[main.policy == p]["success"].mean()) for p in pols}
    drift = {d: tier_values(loops, keys, "drift_pre", d).pivot_table(index=["table", "tier"], columns="policy", values="t_star") for d in (-0.2, 0.2)}
    shuf = tier_values(loops, keys, "shuffled", 0.0).pivot_table(index=["table", "tier"], columns="policy", values="t_star").reset_index()
    deltas, names, pvals = {}, [], []
    for p in pols:
        if p == REFERENCE:
            continue
        deltas[p] = an.mean_ci(wide[p].to_numpy(float) - wide[REFERENCE].to_numpy(float), tables, rng_state); names.append(p); pvals.append(deltas[p]["p_block"])
    for p, a in zip(names, an.bh(pvals)):
        e = deltas[p]; lo, hi = e["ci_table_block"]; e["p_block_bh"] = a
        e["verdict"] = "better" if (e["mean"] < 0 and hi < 0) else ("worse" if (e["mean"] > 0 and lo > 0) else "not distinguishable")
        e["verdict_bh"] = e["verdict"] if a < an.ALPHA else "not distinguishable"
        signs = [float(np.sign(np.nanmean(dv[p] - dv[REFERENCE]))) for dv in drift.values() if p in dv.columns and REFERENCE in dv.columns]
        e["signs_under_drift"] = signs
        e["fragile_under_drift_0p2"] = bool(any(s != np.sign(e["mean"]) and s != 0 for s in signs))
    sh, sn, sp = {}, [], []
    for p in pols:
        if p == REFERENCE or p not in shuf.columns:
            continue
        sh[p] = an.mean_ci(shuf[p].to_numpy(float) - shuf[REFERENCE].to_numpy(float), shuf["table"].to_numpy(), rng_state); sn.append(p); sp.append(sh[p]["p_block"])
    for p, a in zip(sn, an.bh(sp)):
        e = sh[p]; lo, hi = e["ci_table_block"]
        e["verdict_bh"] = ("better" if (e["mean"] < 0 and hi < 0) else ("worse" if (e["mean"] > 0 and lo > 0) else "not distinguishable")) if a < an.ALPHA else "not distinguishable"
    for p in names:
        deltas[p]["shuffled_verdict"] = sh.get(p, {}).get("verdict_bh")
        deltas[p]["artefact"] = bool(p != "CEIL" and deltas[p]["shuffled_verdict"] == "better")
    out["delta_vs_P1"] = deltas
    out["shuffled_delta_vs_P1"] = sh
    ctrl = sh.get("P0")
    if ctrl:
        lo, hi = ctrl["ci_table_block"]
        out["simulator_accepted"] = bool(lo <= 0 <= hi)  # shuffled P0 - P1 contains 0  <=>  P1 - P0 contains 0
    # P13 vs P2
    d = an.mean_ci(wide["P13"].to_numpy(float) - wide["P2"].to_numpy(float), tables, rng_state)
    d["signs_under_drift"] = [float(np.sign(np.nanmean(dv["P13"] - dv["P2"]))) for dv in drift.values()]
    out["P13_vs_P2"] = d
    if chance_tier is not None:
        ch = (wide["table"] + "|" + wide["tier"]).map(chance_tier).to_numpy(float)
        out["chance_mean"] = float(np.nanmean(ch))
        out["P2_minus_chance"] = an.mean_ci(wide["P2"].to_numpy(float) - ch, tables, rng_state)
    by_type = {}
    for st, g in wide.groupby("shift_type"):
        by_type[st] = {p: float(np.nanmean(g[p])) for p in pols}; by_type[st]["n_tiers"] = int(len(g))
    out["by_shift_type_mean_t_star"] = by_type
    return out


def hypotheses(blk: dict) -> dict:
    d = blk["delta_vs_P1"]
    ok = lambda p, v: d[p]["verdict_bh"] == v and not d[p]["fragile_under_drift_0p2"] and not d[p]["artefact"]
    h = {}
    h["H7a_start_at_once"] = bool(ok("P2", "better"))
    breakers = [p for p in ("P4", "P5", "P6") if ok(p, "better")]
    h["H7b_acquisition_matters"] = bool(breakers); h["H7b_policies"] = breakers
    x = blk["P13_vs_P2"]; lo, hi = x["ci_table_block"]
    h["H7c_interval_proposal_helps"] = bool(x["mean"] < 0 and hi < 0 and all(s < 0 for s in x["signs_under_drift"]) and not d["P13"]["artefact"])
    h["H7d_warm_dominates"] = bool(d["P7"]["verdict_bh"] == "worse" and d["P8"]["verdict_bh"] == "worse")
    if "P2_minus_chance" in blk:
        h["H7f_beats_chance"] = bool(blk["P2_minus_chance"]["ci_table_block"][1] < 0)
    return h


def decision_rule(an, e3, loops: pd.DataFrame, keys: set, rng_state) -> dict:
    e3.DIAGS = DIAGS
    def frame(kind: str, delta: float) -> pd.DataFrame:
        sub = loops[(loops.kind == kind) & np.isclose(loops.delta, delta) & loops.policy.isin(list(ARMS))]
        sub = sub[(sub.table + "|" + sub.tier + "|" + sub.target).isin(keys)]
        wide = sub.pivot_table(index=["table", "tier", "shift_type", "target", "seed"], columns="arm", values="t_star").reset_index()
        diag = sub[sub.policy == "A_cont"][["table", "tier", "shift_type", "target", "seed"] + list(DIAGS)]
        out = wide.merge(diag, on=["table", "tier", "shift_type", "target", "seed"], how="left")
        assert not out[list(e3.ACTIONS)].isna().any().any()
        return out
    main_u = frame("main", 0.0)
    drift_u = {d: frame("drift_pre", d) for d in (-0.2, 0.2)}
    res = {"n_units_x_seeds": int(len(main_u)), "n_tiers": int(main_u.groupby(["table", "tier"]).ngroups), "arms_mean_t_star": {a: float(main_u[a].mean()) for a in e3.ACTIONS}}
    n_hold = 0
    per = {}
    for lam in LAMBDAS:
        c = e3.cost_matrix(main_u, float(lam))
        c_rule = np.full(len(main_u), np.nan); c_rule_d = {d: np.full(len(du), np.nan) for d, du in drift_u.items()}
        folds = {}
        for held in e3.FOLD_TABLES:
            tr, te = e3.fold_masks(main_u, held)
            if te.sum() == 0:
                continue
            rule = e3.learn_rule(main_u[tr], float(lam))
            c_rule[te] = e3.rule_cost(rule, main_u[te], float(lam))
            for d, du in drift_u.items():
                _, te_d = e3.fold_masks(du, held)
                c_rule_d[d][te_d] = e3.rule_cost(rule, du[te_d], float(lam))
            folds[held] = {**rule, "description": e3.describe(rule), "n_train": int(tr.sum()), "n_test": int(te.sum())}
        assert np.isfinite(c_rule).all()
        oracle = c.min(axis=1)
        tm = e3.tier_means(main_u, {"rule": c_rule, "continue": c[:, 0], "none": c[:, 1], "switch": c[:, 2], "escalate": c[:, 3], "oracle": oracle})
        tables = tm["table"].to_numpy()
        dc = an.mean_ci((tm["rule"] - tm["continue"]).to_numpy(float), tables, rng_state)
        dn = an.mean_ci((tm["rule"] - tm["none"]).to_numpy(float), tables, rng_state)
        holds = bool(dc["mean"] < 0 and dc["ci_table_block"][1] < 0 and dn["mean"] < 0 and dn["ci_table_block"][1] < 0)
        drift = {}
        for d, du in drift_u.items():
            cd = e3.cost_matrix(du, float(lam)); tmd = e3.tier_means(du, {"rule": c_rule_d[d], "continue": cd[:, 0]})
            drift[str(d)] = float((tmd["rule"] - tmd["continue"]).mean())
        insample = e3.learn_rule(main_u, float(lam))
        per[str(lam)] = {"folds": folds, "mean_cost": {k: float(tm[k].mean()) for k in ("rule", "continue", "none", "switch", "escalate", "oracle")},
                         "delta_rule_minus_continue": dc, "delta_rule_minus_none": dn, "holds": holds, "drift_rule_minus_continue": drift,
                         "headroom_continue_minus_oracle": float((tm["continue"] - tm["oracle"]).mean()),
                         "fraction_units_oracle_not_continue": float(np.mean(oracle < c[:, 0] - 1e-9)),
                         "in_sample_best_rule": {**insample, "description": e3.describe(insample)}}
        n_hold += int(holds)
    res["per_lambda"] = per; res["n_lambda_holds"] = n_hold; res["H7e_positive"] = bool(n_hold >= 2)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args(argv)
    an = load("closed_loop_analyze", HERE / "closed_loop_analyze.py")
    e3 = load("closed_loop_e3_analyze", HERE / "closed_loop_e3_analyze.py")
    run = args.run.resolve()
    loops = pd.read_csv(run / "loops.csv"); units = pd.read_csv(run / "units.csv")
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    rng_state = np.random.default_rng(an.BOOT_SEED)
    units["key"] = units.table + "|" + units.tier + "|" + units.target
    hard = units[units.hit_frac <= HARD_MAX]
    hard_rule = "hit_frac <= 0.10"
    if hard.groupby(["table", "tier"]).ngroups < MIN_HARD_TIERS:
        cut = float(units.hit_frac.quantile(1 / 3)); hard = units[units.hit_frac <= cut]; hard_rule = f"fallback: lowest tercile, hit_frac <= {cut:.6g}"
    sets = {"hard": set(hard.key), "all": set(units.key), "extrapolative": set(units[units.extrapolative == 1].key)}
    terc = pd.qcut(units.hit_frac.rank(method="first"), 3, labels=["h_low", "h_mid", "h_high"])
    for lab in ("h_low", "h_mid", "h_high"):
        sets[lab] = set(units.key[terc == lab])
    results: dict = {"schema_version": "closed-loop-e7-analysis-v1", "protocol": PROTOCOL.name, "protocol_sha256": sha256_file(PROTOCOL),
                     "protocol_sha256_from_run": manifest["protocol_sha256"], "analyzer_sha256": sha256_file(Path(__file__).resolve()),
                     "frozen_analyzer_sha256": sha256_file(HERE / "closed_loop_analyze.py"), "e3_analyzer_sha256": sha256_file(HERE / "closed_loop_e3_analyze.py"),
                     "hard_rule": hard_rule, "n_units": int(len(units)), "n_hard_units": int(len(hard)), "n_hard_tiers": int(hard.groupby(["table", "tier"]).ngroups),
                     "n_extrapolative_units": int(len(sets["extrapolative"])),
                     "n_extrapolative_tiers": int(units[units.extrapolative == 1].groupby(["table", "tier"]).ngroups),
                     "hit_frac": {"median_all": float(units.hit_frac.median()), "median_hard": float(hard.hit_frac.median())}}
    blocks = {}
    for name, keys in sets.items():
        sub = units[units.key.isin(keys)]
        chance_tier = sub.groupby(sub.table + "|" + sub.tier)["chance_expected_t_star"].mean()
        blocks[name] = decision_block(an, loops, keys, rng_state, chance_tier)
    results["blocks"] = blocks
    results["hypotheses_hard"] = hypotheses(blocks["hard"])
    results["hypotheses_all"] = hypotheses(blocks["all"])
    if results["n_extrapolative_tiers"] >= MIN_EXTRAP_TIERS:
        results["hypotheses_extrapolative"] = hypotheses(blocks["extrapolative"])
    results["decision_rule_hard"] = decision_rule(an, e3, loops, sets["hard"], rng_state)
    (run / "results.json").write_text(json.dumps(an.sanitize(results), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = []
    for p in NON_ARM:
        r = {"policy": p}
        for name in ("all", "hard", "extrapolative"):
            b = blocks[name]
            r[f"mean_t_star_{name}"] = b.get("levels_mean_t_star", {}).get(p)
            e = b.get("delta_vs_P1", {}).get(p, {})
            r[f"delta_{name}"] = e.get("mean"); r[f"verdict_{name}"] = e.get("verdict_bh")
        rows.append(r)
    pd.DataFrame(rows).to_csv(run / "policy_table.csv", index=False)
    hashes = json.loads((run / "artifact_hashes.json").read_text(encoding="utf-8"))
    for name in ("results.json", "policy_table.csv"):
        hashes[name] = sha256_file(run / name)
    (run / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hb = blocks["hard"]
    print(json.dumps({"hard_rule": hard_rule, "n_hard_tiers": results["n_hard_tiers"], "simulator_accepted_hard": hb.get("simulator_accepted"),
                      "levels_hard": {p: round(v, 2) for p, v in hb["levels_mean_t_star"].items()},
                      "delta_hard": {p: (e["verdict_bh"], round(e["mean"], 2), e["fragile_under_drift_0p2"], e["artefact"]) for p, e in hb["delta_vs_P1"].items()},
                      "P13_vs_P2": round(hb["P13_vs_P2"]["mean"], 2), "P2_minus_chance": round(hb["P2_minus_chance"]["mean"], 2),
                      "hyp_hard": results["hypotheses_hard"], "H7e": (results["decision_rule_hard"]["n_lambda_holds"],
                      {l: (round(v["delta_rule_minus_continue"]["mean"], 2), round(v["headroom_continue_minus_oracle"], 2)) for l, v in results["decision_rule_hard"]["per_lambda"].items()})}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
