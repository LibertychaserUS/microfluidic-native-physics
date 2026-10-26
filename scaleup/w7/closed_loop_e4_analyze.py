"""Analyzer for E4_MODEL_FAMILY_PROTOCOL_v1 (§4 decisions).

    python -I scaleup/w7/closed_loop_e4_analyze.py --run <e4 run dir>

Deterministic from loops.csv, families.csv and table_cv.csv; writes results.json and family_table.csv into the
run directory and records their digests in artifact_hashes.json.  Bootstrap, BH and sanitising come from the
frozen E1 analyzer.
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
from scipy.stats import kendalltau

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE.parent / "validation_plan" / "E4_MODEL_FAMILY_PROTOCOL_v1.md"
FAMILIES = ("xgboost", "random_forest", "mlp", "qrf", "gp", "deep_ensemble")
REFERENCE = "xgboost"
POLICIES = ("P0", "P1", "P2")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_analyzer():
    spec = importlib.util.spec_from_file_location("closed_loop_analyze", HERE / "closed_loop_analyze.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["closed_loop_analyze"] = module
    spec.loader.exec_module(module)
    return module


def tier_values(loops: pd.DataFrame, kind: str, delta: float, policy: str) -> pd.DataFrame:
    sub = loops[(loops.kind == kind) & np.isclose(loops.delta, delta) & (loops.policy == policy)]
    g = sub.groupby(["table", "tier", "shift_type", "family"], sort=True)["t_star"].mean().reset_index()
    return g.pivot_table(index=["table", "tier", "shift_type"], columns="family", values="t_star").reset_index()


def tau_rows(a: pd.DataFrame, b: pd.DataFrame, fams: list[str]) -> np.ndarray:
    """Kendall tau_b per tier between two family-valued frames aligned on table|tier (both ascending = better)."""
    key = lambda d: d["table"] + "|" + d["tier"]
    bb = b.set_index(key(b))
    out = []
    for _, row in a.iterrows():
        k = row["table"] + "|" + row["tier"]
        x = np.array([row[f] for f in fams], float)
        y = np.array([bb.loc[k, f] for f in fams], float)
        t, _ = kendalltau(x, y)
        out.append(float(t) if np.isfinite(t) else float("nan"))
    return np.array(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args(argv)
    an = load_frozen_analyzer()
    run = args.run.resolve()
    loops = pd.read_csv(run / "loops.csv")
    fams = pd.read_csv(run / "families.csv")
    cv = pd.read_csv(run / "table_cv.csv")
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    rng_state = np.random.default_rng(an.BOOT_SEED)
    families = [f for f in FAMILIES if f in set(loops.family)]
    results: dict = {"schema_version": "closed-loop-e4-analysis-v1", "protocol": PROTOCOL.name, "protocol_sha256": sha256_file(PROTOCOL),
                     "protocol_sha256_from_run": manifest["protocol_sha256"], "analyzer_sha256": sha256_file(Path(__file__).resolve()),
                     "frozen_analyzer_sha256": sha256_file(HERE / "closed_loop_analyze.py"), "families": families, "n_tiers": int(fams.groupby(["table", "tier"]).ngroups)}
    # ---- D1 family effect per policy
    effect = {}
    for policy in POLICIES:
        wide = tier_values(loops, "main", 0.0, policy)
        tables = wide["table"].to_numpy()
        deltas, names, pvals = {}, [], []
        for f in families:
            if f == REFERENCE:
                continue
            d = wide[f].to_numpy(float) - wide[REFERENCE].to_numpy(float)
            deltas[f] = an.mean_ci(d, tables, rng_state); names.append(f); pvals.append(deltas[f]["p_block"])
        for f, a in zip(names, an.bh(pvals)):
            e = deltas[f]; lo, hi = e["ci_table_block"]
            e["p_block_bh"] = a
            e["verdict"] = "better" if (e["mean"] < 0 and hi < 0) else ("worse" if (e["mean"] > 0 and lo > 0) else "not distinguishable")
            e["verdict_bh"] = e["verdict"] if a < an.ALPHA else "not distinguishable"
            signs = []
            for delta in (-0.2, 0.2):
                dv = tier_values(loops, "drift_pre", delta, policy)
                if f in dv.columns and REFERENCE in dv.columns:
                    signs.append(float(np.sign(np.nanmean(dv[f] - dv[REFERENCE]))))
            base = float(np.sign(e["mean"]))
            e["signs_under_drift"] = signs
            e["fragile_under_drift_0p2"] = bool(any(s != base and s != 0 for s in signs))
        levels = {f: float(np.nanmean(wide[f])) for f in families}
        best = wide[families].to_numpy(float).argmin(axis=1)
        by_type = {}
        for st, g in wide.groupby("shift_type"):
            by_type[st] = {f: float(np.nanmean(g[f])) for f in families}; by_type[st]["n_tiers"] = int(len(g))
        effect[policy] = {"levels_mean_t_star": levels, "delta_vs_xgboost": deltas, "fraction_tiers_best": {f: float(np.mean(best == i)) for i, f in enumerate(families)},
                          "by_shift_type_mean_t_star": by_type}
    results["E4_family_effect"] = effect
    # ---- rankings
    cvm = cv.groupby(["table", "family"])["mape"].mean().reset_index().pivot_table(index="table", columns="family", values="mape")
    zs = fams.pivot_table(index=["table", "tier", "shift_type"], columns="family", values="zero_shot_mape").reset_index()
    cv_tier = zs[["table", "tier", "shift_type"]].copy()
    for f in families:
        cv_tier[f] = cv_tier["table"].map(cvm[f]).to_numpy(float)
    rank = {}
    for policy in ("P2", "P1"):
        wide = tier_values(loops, "main", 0.0, policy)
        tables = wide["table"].to_numpy()
        key = lambda d: (d["table"] + "|" + d["tier"]).to_numpy()
        assert (key(wide) == key(cv_tier)).all() and (key(wide) == key(zs)).all()
        comps = {}
        for name, a in (("random_split_vs_loop", cv_tier), ("zero_shot_vs_loop", zs)):
            tau = tau_rows(a, wide, families)
            ci = an.mean_ci(tau, tables, rng_state)
            lo, hi = ci["ci_table_block"]
            ci["verdict"] = "predicts" if (ci["mean"] > 0 and lo > 0) else ("contradicts" if (ci["mean"] < 0 and hi < 0) else "does not predict")
            ci["fraction_tiers_tau_positive"] = float(np.nanmean(tau > 0))
            comps[name] = ci
        rank[policy] = comps
    tau = tau_rows(cv_tier, zs, families); ci = an.mean_ci(tau, zs["table"].to_numpy(), rng_state); lo, hi = ci["ci_table_block"]
    ci["verdict"] = "predicts" if (ci["mean"] > 0 and lo > 0) else ("contradicts" if (ci["mean"] < 0 and hi < 0) else "does not predict")
    rank["random_split_vs_zero_shot"] = ci
    results["E4_ranking"] = rank
    results["table_cv_mape"] = {t: {f: float(cvm.loc[t, f]) for f in families} for t in cvm.index}
    results["zero_shot_mape_mean"] = {f: float(np.nanmean(zs[f])) for f in families}
    results["zero_shot_mape_median"] = {f: float(np.nanmedian(zs[f])) for f in families}
    results["fit_seconds_mean"] = {f: float(fams[fams.family == f]["fit_seconds"].mean()) for f in families}
    if "extra_convergence_warnings" in fams.columns:
        results["gp_convergence_warnings_total"] = int(np.nansum(fams["extra_convergence_warnings"].to_numpy(float)))
    (run / "results.json").write_text(json.dumps(an.sanitize(results), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = []
    for f in families:
        e = effect["P2"]["delta_vs_xgboost"].get(f, {})
        rows.append({"family": f, "mean_t_star_P0": effect["P0"]["levels_mean_t_star"][f], "mean_t_star_P1": effect["P1"]["levels_mean_t_star"][f], "mean_t_star_P2": effect["P2"]["levels_mean_t_star"][f],
                     "delta_P2_vs_xgboost": e.get("mean"), "ci_table_block": e.get("ci_table_block"), "verdict_bh": e.get("verdict_bh"), "fragile": e.get("fragile_under_drift_0p2"),
                     "zero_shot_mape_median": results["zero_shot_mape_median"][f], "fraction_tiers_best_P2": effect["P2"]["fraction_tiers_best"][f]})
    pd.DataFrame(rows).to_csv(run / "family_table.csv", index=False)
    hashes = json.loads((run / "artifact_hashes.json").read_text(encoding="utf-8"))
    for name in ("results.json", "family_table.csv"):
        hashes[name] = sha256_file(run / name)
    (run / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"levels_P2": {f: round(v, 3) for f, v in effect["P2"]["levels_mean_t_star"].items()},
                      "delta_P2": {f: (e["verdict_bh"], round(e["mean"], 3), e["fragile_under_drift_0p2"]) for f, e in effect["P2"]["delta_vs_xgboost"].items()},
                      "ranking": {k: (v["verdict"], round(v["mean"], 3), [round(x, 3) for x in v["ci_table_block"]]) for k, v in rank["P2"].items()},
                      "rs_vs_zs": (rank["random_split_vs_zero_shot"]["verdict"], round(rank["random_split_vs_zero_shot"]["mean"], 3))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
