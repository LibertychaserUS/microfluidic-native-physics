"""Analyzer for E3_DECISION_POLICY_PROTOCOL_v1 (§3 rule learning by five-fold table rotation, §4 decision form).

    python -I scaleup/w7/closed_loop_e3_analyze.py --run <e3 run dir>

Deterministic from loops.csv; writes results.json and rules_table.csv into the run directory and records their
digests in artifact_hashes.json.  Bootstrap and sanitising functions come from the frozen E1 analyzer.
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
PROTOCOL = HERE.parent / "validation_plan" / "E3_DECISION_POLICY_PROTOCOL_v1.md"
ACTIONS = ("continue", "none", "switch", "escalate")
DIAGS = ("d1_bias_ratio", "d2_magnitude_ratio", "d3_width_ratio", "d4_support_outside")
LAMBDAS = (0, 5, 10)
FOLD_TABLES = ("D1", "D2", "A1", "R2A1", "R2A2")
HELD_WITH = {"D3": "D2"}  # D3 is held out together with D2 and never trained on
THETA_PERCENTILES = tuple(range(10, 100, 10))
UNIT = ["table", "tier", "shift_type", "target", "seed"]
PREVIEW_THRESHOLD = 0.63
EPS = 1e-9


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_analyzer():
    spec = importlib.util.spec_from_file_location("closed_loop_analyze", HERE / "closed_loop_analyze.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["closed_loop_analyze"] = module
    spec.loader.exec_module(module)
    return module


def unit_frame(loops: pd.DataFrame, kind: str, delta: float) -> pd.DataFrame:
    sub = loops[(loops.kind == kind) & np.isclose(loops.delta, delta)]
    wide = sub.pivot_table(index=UNIT, columns="arm", values="t_star").reset_index()
    diag = sub[sub.arm == "continue"][UNIT + list(DIAGS)]
    out = wide.merge(diag, on=UNIT, how="left")
    assert not out[list(ACTIONS)].isna().any().any(), "every arm must be present for every unit"
    return out


def cost_matrix(units: pd.DataFrame, lam: float) -> np.ndarray:
    c = units[list(ACTIONS)].to_numpy(float).copy()
    c[:, ACTIONS.index("escalate")] += lam
    return c


def rule_cost(rule: dict, units: pd.DataFrame, lam: float) -> np.ndarray:
    c = cost_matrix(units, lam)
    if rule["kind"] == "constant":
        return c[:, ACTIONS.index(rule["action"])]
    d = units[rule["diag"]].to_numpy(float)
    hi, lo = ACTIONS.index(rule["a_hi"]), ACTIONS.index(rule["a_lo"])
    return np.where(d > rule["theta"], c[:, hi], c[:, lo])


def learn_rule(train: pd.DataFrame, lam: float) -> dict:
    c = cost_matrix(train, lam)
    cands = []
    for ai, a in enumerate(ACTIONS):
        obj = float(c[:, ai].mean())
        cands.append(((round(obj, 9), 0 if a == "continue" else 1, int(a == "escalate"), 0, -1, float("-inf"), ai, ai),
                      {"kind": "constant", "action": a, "train_cost": obj}))
    for di, dname in enumerate(DIAGS):
        d = train[dname].to_numpy(float)
        ok = np.isfinite(d)
        if ok.sum() == 0:
            continue
        for theta in sorted(set(float(t) for t in np.percentile(d[ok], THETA_PERCENTILES))):
            mask = d > theta
            for lo_i, lo in enumerate(ACTIONS):
                for hi_i, hi in enumerate(ACTIONS):
                    if lo == hi:
                        continue
                    obj = float(np.where(mask, c[:, hi_i], c[:, lo_i]).mean())
                    n_esc = int(lo == "escalate") + int(hi == "escalate")
                    cands.append(((round(obj, 9), 1, n_esc, 1, di, theta, lo_i, hi_i),
                                  {"kind": "threshold", "diag": dname, "theta": theta, "a_lo": lo, "a_hi": hi, "train_cost": obj,
                                   "train_fraction_above": float(mask.mean())}))
    cands.sort(key=lambda kv: kv[0])
    return cands[0][1]


def describe(rule: dict) -> str:
    if rule["kind"] == "constant":
        return f"always {rule['action']}"
    return f"{rule['diag']} > {rule['theta']:.4g} -> {rule['a_hi']} else {rule['a_lo']}"


def fold_masks(units: pd.DataFrame, held: str) -> tuple[np.ndarray, np.ndarray]:
    tab = units["table"].to_numpy()
    held_set = {held} | {t for t, h in HELD_WITH.items() if h == held}
    test = np.isin(tab, list(held_set))
    train = ~np.isin(tab, [held] + list(HELD_WITH))
    return train, test


def tier_means(units: pd.DataFrame, values: dict[str, np.ndarray]) -> pd.DataFrame:
    df = units[["table", "tier", "shift_type"]].copy()
    for k, v in values.items():
        df[k] = v
    return df.groupby(["table", "tier", "shift_type"], sort=True).mean(numeric_only=True).reset_index()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args(argv)
    an = load_frozen_analyzer()
    run = args.run.resolve()
    loops = pd.read_csv(run / "loops.csv")
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    rng_state = np.random.default_rng(an.BOOT_SEED)
    main_u = unit_frame(loops, "main", 0.0)
    drift_u = {d: unit_frame(loops, "drift_pre", d) for d in (-0.2, 0.2)}
    results: dict = {"schema_version": "closed-loop-e3-analysis-v1", "protocol": PROTOCOL.name, "protocol_sha256": sha256_file(PROTOCOL),
                     "protocol_sha256_from_run": manifest["protocol_sha256"], "analyzer_sha256": sha256_file(Path(__file__).resolve()),
                     "frozen_analyzer_sha256": sha256_file(HERE / "closed_loop_analyze.py"), "n_units": int(len(main_u)),
                     "n_tiers": int(main_u.groupby(["table", "tier"]).ngroups), "arms_mean_t_star": {a: float(main_u[a].mean()) for a in ACTIONS}}
    # descriptive: oracle-best action by shift type (ties -> continue), mean T* per arm by type
    c0 = cost_matrix(main_u, 0.0)
    by_type = {}
    for st, g in main_u.groupby("shift_type"):
        cg = cost_matrix(g, 5.0)
        best = np.where(cg.min(axis=1) < cg[:, 0] - EPS, cg.argmin(axis=1), 0)
        by_type[st] = {"n_units": int(len(g)), "mean_t_star": {a: float(g[a].mean()) for a in ACTIONS},
                       "oracle_best_action_at_lambda5": {a: float(np.mean(best == i)) for i, a in enumerate(ACTIONS)}}
    results["by_shift_type"] = by_type
    # P12 preview rule as a fixed member of the family (descriptive)
    p12 = {"kind": "threshold", "diag": "d1_bias_ratio", "theta": PREVIEW_THRESHOLD, "a_lo": "none", "a_hi": "continue"}
    tm = tier_means(main_u, {"p12": rule_cost(p12, main_u, 0.0), "cont": c0[:, 0]})
    results["preview_rule_P12"] = {"rule": describe(p12), **an.mean_ci((tm["p12"] - tm["cont"]).to_numpy(float), tm["table"].to_numpy(), rng_state)}
    rules_rows = []
    per_lambda = {}
    n_hold = 0
    for lam in LAMBDAS:
        c = cost_matrix(main_u, float(lam))
        c_rule = np.full(len(main_u), np.nan)
        c_rule_drift = {d: np.full(len(drift_u[d]), np.nan) for d in drift_u}
        folds = {}
        for held in FOLD_TABLES:
            tr, te = fold_masks(main_u, held)
            rule = learn_rule(main_u[tr], float(lam))
            c_rule[te] = rule_cost(rule, main_u[te], float(lam))
            for d, du in drift_u.items():
                _, te_d = fold_masks(du, held)
                c_rule_drift[d][te_d] = rule_cost(rule, du[te_d], float(lam))
            folds[held] = {**rule, "n_train_units": int(tr.sum()), "n_test_units": int(te.sum()), "description": describe(rule)}
            rules_rows.append({"lambda": lam, "held_out": held, "rule": describe(rule), "train_cost": rule["train_cost"], "n_train_units": int(tr.sum()), "n_test_units": int(te.sum())})
        assert np.isfinite(c_rule).all(), "every unit must be covered by exactly one fold"
        oracle = c.min(axis=1)
        tm = tier_means(main_u, {"rule": c_rule, "continue": c[:, 0], "none": c[:, 1], "switch": c[:, 2], "escalate": c[:, 3], "oracle": oracle})
        tables = tm["table"].to_numpy()
        d_cont = an.mean_ci((tm["rule"] - tm["continue"]).to_numpy(float), tables, rng_state)
        d_none = an.mean_ci((tm["rule"] - tm["none"]).to_numpy(float), tables, rng_state)
        d_switch = an.mean_ci((tm["rule"] - tm["switch"]).to_numpy(float), tables, rng_state)
        d_esc = an.mean_ci((tm["rule"] - tm["escalate"]).to_numpy(float), tables, rng_state)
        holds = bool(d_cont["mean"] < 0 and d_cont["ci_table_block"][1] < 0 and d_none["mean"] < 0 and d_none["ci_table_block"][1] < 0)
        drift_signs = {}
        for d, du in drift_u.items():
            cd = cost_matrix(du, float(lam))
            tmd = tier_means(du, {"rule": c_rule_drift[d], "continue": cd[:, 0]})
            drift_signs[str(d)] = {"delta_cont_mean": float((tmd["rule"] - tmd["continue"]).mean()), "sign": float(np.sign((tmd["rule"] - tmd["continue"]).mean()))}
        fragile = bool(holds and any(v["sign"] > 0 for v in drift_signs.values()))
        in_sample = learn_rule(main_u, float(lam))
        per_lambda[str(lam)] = {"folds": folds, "mean_cost": {k: float(tm[k].mean()) for k in ("rule", "continue", "none", "switch", "escalate", "oracle")},
                                "delta_rule_minus_continue": d_cont, "delta_rule_minus_none": d_none, "delta_rule_minus_switch": d_switch, "delta_rule_minus_escalate": d_esc,
                                "headroom_continue_minus_oracle": float((tm["continue"] - tm["oracle"]).mean()),
                                "fraction_units_oracle_not_continue": float(np.mean(oracle < c[:, 0] - EPS)),
                                "holds": holds, "fragile_under_drift_0p2": fragile, "drift": drift_signs,
                                "in_sample_best_rule": {**in_sample, "description": describe(in_sample)}}
        n_hold += int(holds)
    results["per_lambda"] = per_lambda
    results["n_lambda_holds"] = n_hold
    results["E3_positive"] = bool(n_hold >= 2)
    (run / "results.json").write_text(json.dumps(an.sanitize(results), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame(rules_rows).to_csv(run / "rules_table.csv", index=False)
    hashes = json.loads((run / "artifact_hashes.json").read_text(encoding="utf-8"))
    for name in ("results.json", "rules_table.csv"):
        hashes[name] = sha256_file(run / name)
    (run / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"E3_positive": results["E3_positive"], "arms": {a: round(v, 3) for a, v in results["arms_mean_t_star"].items()},
                      **{f"lambda{l}": (v["holds"], round(v["delta_rule_minus_continue"]["mean"], 3), [f["description"] for f in v["folds"].values()],
                                        round(v["headroom_continue_minus_oracle"], 3)) for l, v in per_lambda.items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
