"""E5 budget-transfer analysis — E5_BUDGET_TRANSFER_PROTOCOL_v1 (analysis of frozen E0/E2/E3 outputs; no new loops).

    python -I scaleup/w7/closed_loop_e5.py --work W --out W/e5_run1 [--e1e2 W/e1e2_run1 --e0 W/e0_run1 --e3 W/e3_run1]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE.parent / "validation_plan" / "E5_BUDGET_TRANSFER_PROTOCOL_v1.md"
EPS_MULT = 1.25
BUDGET = 20
NOT_REACHED = 21
Q = 0.80
Q_SENS = (0.50, 0.90)
FOLD_TABLES = ("D1", "D2", "A1", "R2A1", "R2A2")
HELD_WITH = {"D3": "D2"}
RULES = ("R0", "R1", "R2", "R3", "R4")
K_GRID = (1, 2, 3, 5, 8, 10, 15, 20)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_analyzer():
    spec = importlib.util.spec_from_file_location("closed_loop_analyze", HERE / "closed_loop_analyze.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["closed_loop_analyze"] = module
    spec.loader.exec_module(module)
    return module


def features_of(rule: str, df: pd.DataFrame, med_d4: float, med_d1: float) -> pd.Series:
    if rule == "R0":
        return pd.Series(["all"] * len(df), index=df.index)
    if rule == "R1":
        return df["shift_type"]
    if rule == "R2":
        return df["shift_type"] + "|" + np.where(df["dense"] == 1, "dense", "sparse")
    if rule == "R3":
        return df["shift_type"] + "|" + np.where(df["d4"] > med_d4, "d4hi", "d4lo")
    if rule == "R4":
        return df["shift_type"] + "|" + np.where(df["d1"] > med_d1, "d1hi", "d1lo")
    raise ValueError(rule)


def learn(rule: str, train: pd.DataFrame, q: float) -> tuple[dict, float, float, float]:
    med_d4 = float(np.median(train["d4"])); med_d1 = float(np.median(train["d1"]))
    if rule == "R0":
        return {}, float(BUDGET), med_d4, med_d1
    feats = features_of(rule, train, med_d4, med_d1)
    table = {f: float(np.quantile(train.loc[feats == f, "k_star"], q)) for f in sorted(set(feats))}
    return table, float(np.quantile(train["k_star"], q)), med_d4, med_d1


def plan(rule: str, model: tuple, test: pd.DataFrame) -> np.ndarray:
    table, fallback, med_d4, med_d1 = model
    if rule == "R0":
        return np.full(len(test), float(BUDGET))
    feats = features_of(rule, test, med_d4, med_d1)
    return np.array([table.get(f, fallback) for f in feats], float)


def cost(b_hat: np.ndarray, k_star: np.ndarray) -> np.ndarray:
    b = np.minimum(np.ceil(b_hat), BUDGET)
    return np.where(b >= k_star, b - k_star, float(BUDGET))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--e1e2", type=Path, default=None)
    ap.add_argument("--e0", type=Path, default=None)
    ap.add_argument("--e3", type=Path, default=None)
    args = ap.parse_args(argv)
    an = load_frozen_analyzer()
    work = args.work.resolve()
    e1e2 = (args.e1e2 or work / "e1e2_run1").resolve(); e0 = (args.e0 or work / "e0_run1").resolve(); e3 = (args.e3 or work / "e3_run1").resolve()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    budget = pd.read_csv(e1e2 / "budget.csv"); budget = budget[budget.method == "warm"]
    tiers = pd.read_csv(e0 / "tiers_e0.csv")
    loops3 = pd.read_csv(e3 / "loops.csv")
    bt = pd.read_csv(e1e2 / "budget_table.csv"); bt = bt[bt.method == "warm"].set_index("key")
    rng_state = np.random.default_rng(an.BOOT_SEED)
    # ---- per-tier budgets
    med = budget.groupby(["table", "tier", "shift_type", "k"])["mape"].median().reset_index()
    unc = budget.groupby(["table", "tier"])["mape_uncal"].first()
    rows = []
    for (tb, tier, st), g in med.groupby(["table", "tier", "shift_type"]):
        key = f"{tb}|{tier}"
        dev_ref = float(tiers.set_index(tiers["table"] + "|" + tiers["tier"]).loc[key, "dev_ref_mape"])
        eps = EPS_MULT * dev_ref
        reached = sorted(int(k) for k, m in zip(g["k"], g["mape"]) if m <= eps)
        k_star = reached[0] if reached else NOT_REACHED
        assert int(bt.loc[key, "k_star"]) == k_star, (key, k_star, bt.loc[key, "k_star"])
        d = loops3[(loops3.table == tb) & (loops3.tier == tier) & (loops3.kind == "main") & (loops3.arm == "continue")]
        row = {"table": tb, "tier": tier, "shift_type": st, "dense": int(tiers.set_index(tiers["table"] + "|" + tiers["tier"]).loc[key, "dense"]),
               "dev_ref": dev_ref, "eps": eps, "mape_uncal": float(unc.loc[(tb, tier)]), "k_star": k_star,
               "d1": float(d.groupby("seed")["d1_bias_ratio"].first().median()), "d4": float(d["d4_support_outside"].iloc[0])}
        for k in K_GRID:
            m = g[g.k == k]["mape"]
            row[f"gain_k{k}"] = float(m.iloc[0] / row["mape_uncal"]) if len(m) and row["mape_uncal"] > 0 else float("nan")
        rows.append(row)
    tb_df = pd.DataFrame(rows).sort_values(["table", "tier"], kind="stable").reset_index(drop=True)
    tb_df.to_csv(out / "tier_budgets.csv", index=False, float_format="%.12g")
    reach = tb_df[tb_df.k_star <= BUDGET].reset_index(drop=True)
    results: dict = {"schema_version": "closed-loop-e5-analysis-v1", "protocol": PROTOCOL.name, "protocol_sha256": sha256_file(PROTOCOL),
                     "analyzer_sha256": sha256_file(Path(__file__).resolve()), "frozen_analyzer_sha256": sha256_file(HERE / "closed_loop_analyze.py"),
                     "inputs": {"budget.csv": sha256_file(e1e2 / "budget.csv"), "tiers_e0.csv": sha256_file(e0 / "tiers_e0.csv"), "e3_loops.csv": sha256_file(e3 / "loops.csv")},
                     "n_tiers": int(len(tb_df)), "n_reached": int(len(reach)), "n_not_reached": int((tb_df.k_star == NOT_REACHED).sum()), "q": Q}
    # ---- rotation
    rules_rows = []
    per_rule = {}
    for q in (Q,) + Q_SENS:
        for rule in RULES:
            c_tr = np.full(len(tb_df), np.nan); c_ts = np.full(len(tb_df), np.nan); bhat_tr = np.full(len(tb_df), np.nan)
            nr_pred_tr = np.zeros(len(tb_df), bool)
            for held in FOLD_TABLES:
                held_set = {held} | {t for t, h in HELD_WITH.items() if h == held}
                te = tb_df["table"].isin(held_set).to_numpy(); trn = ~tb_df["table"].isin([held] + list(HELD_WITH)).to_numpy()
                train = tb_df[trn & (tb_df.k_star <= BUDGET).to_numpy()]  # learn on reached tiers (k* defined)
                train_all = tb_df[trn]
                model_tr = learn(rule, train, q)
                # not-reachable classification: lookup on all training tiers incl. 21
                m_all = learn(rule, train_all, q)
                test = tb_df[te]
                b_tr = plan(rule, model_tr, test); bhat_tr[te] = b_tr
                nr_pred_tr[te] = plan(rule, m_all, test) > BUDGET
                own = tb_df[te & (tb_df.k_star <= BUDGET).to_numpy()]
                model_ts = learn(rule, own, q) if len(own) >= 2 else model_tr
                b_ts = plan(rule, model_ts, test)
                c_tr[te] = cost(b_tr, test["k_star"].to_numpy(float)); c_ts[te] = cost(b_ts, test["k_star"].to_numpy(float))
                if q == Q:
                    rules_rows.append({"rule": rule, "held_out": held, "n_test": int(te.sum()), "lookup_transferred": json.dumps(model_tr[0], sort_keys=True), "fallback": model_tr[1]})
            keep = (tb_df.k_star <= BUDGET).to_numpy()
            tables = tb_df["table"].to_numpy()[keep]
            regret = an.mean_ci(c_tr[keep] - c_ts[keep], tables, rng_state)
            c0 = cost(np.full(keep.sum(), float(BUDGET)), tb_df["k_star"].to_numpy(float)[keep])
            vs_r0 = an.mean_ci(c_tr[keep] - c0, tables, rng_state)
            lo, hi = regret["ci_table_block"]; lo0, hi0 = vs_r0["ci_table_block"]
            beats_r0 = bool(vs_r0["mean"] < 0 and hi0 < 0)
            if regret["mean"] > 0 and lo > 0:
                verdict = "does not transfer"
            elif beats_r0:
                verdict = "transfers"
            else:
                verdict = "no advantage over the full budget"
            entry = {"mean_cost_transferred": float(np.mean(c_tr[keep])), "mean_cost_table_specific": float(np.mean(c_ts[keep])), "mean_cost_R0": float(np.mean(c0)),
                     "transfer_regret": regret, "transferred_minus_R0": vs_r0, "verdict": verdict,
                     "fraction_underplanned_transferred": float(np.mean(np.minimum(np.ceil(bhat_tr[keep]), BUDGET) < tb_df["k_star"].to_numpy(float)[keep])),
                     "not_reachable_classification": {"n_not_reached": int((~keep).sum()), "tpr": float(np.mean(nr_pred_tr[~keep])) if (~keep).any() else float("nan"),
                                                      "fpr": float(np.mean(nr_pred_tr[keep]))}}
            per_rule.setdefault(str(q), {})[rule] = entry
    oracle_cost = cost(np.minimum(tb_df["k_star"].to_numpy(float), BUDGET), tb_df["k_star"].to_numpy(float))
    results["mean_cost_oracle"] = float(np.mean(oracle_cost[(tb_df.k_star <= BUDGET).to_numpy()]))
    results["rules"] = per_rule[str(Q)]; results["sensitivity_q"] = {q: per_rule[q] for q in per_rule if q != str(Q)}
    results["k_star_by_shift_type"] = {st: {"n": int(len(g)), "median": float(np.median(g.k_star)), "p80": float(np.quantile(g.k_star, 0.8)), "fraction_not_reached": float(np.mean(g.k_star == NOT_REACHED))}
                                       for st, g in tb_df.groupby("shift_type")}
    # ---- shape transfer (descriptive)
    shape = {}
    for st, g in tb_df.groupby("shift_type"):
        per_table = {tb: {f"k{k}": float(np.nanmedian(gg[f"gain_k{k}"])) for k in K_GRID} for tb, gg in g.groupby("table") if len(gg) >= 2}
        if len(per_table) >= 2:
            shape[st] = {"tables": per_table, "between_table_range": {f"k{k}": float(max(v[f"k{k}"] for v in per_table.values()) - min(v[f"k{k}"] for v in per_table.values())) for k in K_GRID}}
    results["shape_transfer"] = shape
    (out / "results.json").write_text(json.dumps(an.sanitize(results), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame(rules_rows).to_csv(out / "rules_table.csv", index=False)
    manifest = {"schema_version": "closed-loop-e5-run-v1", "stage": "e5", "protocol": PROTOCOL.name, "protocol_sha256": sha256_file(PROTOCOL), "analyzer_sha256": sha256_file(Path(__file__).resolve()),
                "inputs": results["inputs"], "q": Q, "q_sensitivity": list(Q_SENS), "budget": BUDGET, "eps_mult": EPS_MULT, "rules": list(RULES), "n_tiers": int(len(tb_df)),
                "python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    hashes = {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.name != "artifact_hashes.json"}
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({r: (v["verdict"], round(v["mean_cost_transferred"], 2), round(v["mean_cost_table_specific"], 2), round(v["transfer_regret"]["mean"], 2), [round(x, 2) for x in v["transfer_regret"]["ci_table_block"]]) for r, v in results["rules"].items()}))
    print(json.dumps({"oracle": round(results["mean_cost_oracle"], 2), "n_reached": results["n_reached"], "k_star_by_type": {k: v["p80"] for k, v in results["k_star_by_shift_type"].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
