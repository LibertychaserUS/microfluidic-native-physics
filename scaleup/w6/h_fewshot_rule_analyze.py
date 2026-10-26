"""H-J analysis — Section 5 of H_J_FEWSHOT_RULE_PROTOCOL_v1.md (A1 association, A2 threshold rule, A3 decision).

    python -I scaleup/w6/h_fewshot_rule_analyze.py --run <run dir>

Deterministic from tiers.csv and draws.csv; writes results.json, tier_summary.csv and rule_summary.csv
into the run directory and appends them to artifact_hashes.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

K_VALUES = (2, 5, 10)
METHODS = ("affine", "warm", "scratch")
PREDICTORS = {"p1_bias_ratio": "P1", "p2_sign_consistency": "P2", "p3_magnitude_ratio": "P3", "p4_support_overlap": "P4", "p5_width_ratio": "P5"}
B = 2000
BOOT_SEED = 0
ALPHA = 0.05
MIN_TIERS_PER_TABLE_FOR_SIGN = 5
GRID_PERCENTILES = tuple(range(5, 100, 5))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sanitize(v):
    if isinstance(v, dict):
        return {str(k): sanitize(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [sanitize(x) for x in v]
    if isinstance(v, (np.floating, float)):
        return None if not math.isfinite(float(v)) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return float("nan")
    return float(spearmanr(a[ok], b[ok]).statistic)


def bh_adjust(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    p = np.where(np.isfinite(p), p, 1.0)  # an undefined p-value is treated as 1 (never significant)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n)
    running = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        running = min(running, p[i] * n / rank)
        adjusted[i] = running
    return [float(v) if math.isfinite(v) else float("nan") for v in adjusted]


def block_bootstrap_indices(tables: np.ndarray, rng: np.random.Generator, kind: str) -> np.ndarray:
    """kind='table': resample tables with replacement; kind='tier': resample tiers within each table."""
    levels = sorted(set(tables.tolist()))
    if kind == "table":
        chosen = rng.choice(len(levels), size=len(levels), replace=True)
        return np.concatenate([np.flatnonzero(tables == levels[c]) for c in chosen])
    parts = []
    for level in levels:
        idx = np.flatnonzero(tables == level)
        parts.append(rng.choice(idx, size=len(idx), replace=True))
    return np.concatenate(parts)


def percentile_ci(values: np.ndarray) -> tuple[float, float, int]:
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return float("nan"), float("nan"), 0
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), int(len(v))


def two_sided_p(values: np.ndarray) -> float:
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return float("nan")
    return float(2.0 * min(np.mean(v <= 0), np.mean(v >= 0)))


# ----------------------------------------------------------------------------- tier summary
def tier_summary(tiers: pd.DataFrame, draws: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (table, tier, k), g in draws.groupby(["table", "tier", "k"], sort=True):
        t = tiers[(tiers.table == table) & (tiers.tier == tier)].iloc[0]
        row = {"table": table, "tier": tier, "shift_type": t["shift_type"], "k": int(k), "n": int(t["n"]), "n_draws": int(len(g))}
        for m in METHODS:
            row[f"G_{m}"] = float(np.nanmedian(g["mape_uncal"] - g[f"mape_{m}"]))
        for m in ("uncal",) + METHODS:
            row[f"regret_{m}"] = float(np.nanmedian(g[f"regret_{m}"]))
        for col in ("p1_bias_ratio", "p2_sign_consistency", "p3_magnitude_ratio"):
            row[col] = float(np.nanmedian(g[col]))
        row["p4_support_overlap"] = float(t["p4_support_overlap"])
        row["p5_width_ratio"] = float(t["p5_width_ratio"])
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- A1 association
def association(summary: pd.DataFrame) -> dict:
    out: dict = {}
    rng_state = np.random.default_rng(BOOT_SEED)
    for k in K_VALUES:
        sk = summary[summary.k == k].reset_index(drop=True)
        tables = sk["table"].to_numpy()
        out[str(k)] = {"n_tiers": int(len(sk)), "methods": {}}
        for m in METHODS:
            g = sk[f"G_{m}"].to_numpy(float)
            entries = {}
            pvals = []
            for col, label in PREDICTORS.items():
                p = sk[col].to_numpy(float)
                rho = spearman(g, p)
                rng = np.random.default_rng(rng_state.integers(0, 2**32 - 1))
                boot_table = np.array([spearman(g[idx], p[idx]) for idx in (block_bootstrap_indices(tables, rng, "table") for _ in range(B))])
                rng = np.random.default_rng(rng_state.integers(0, 2**32 - 1))
                boot_tier = np.array([spearman(g[idx], p[idx]) for idx in (block_bootstrap_indices(tables, rng, "tier") for _ in range(B))])
                lo_t, hi_t, n_t = percentile_ci(boot_table)
                lo_r, hi_r, n_r = percentile_ci(boot_tier)
                pval = two_sided_p(boot_table)
                within = {}
                for table in sorted(set(tables.tolist())):
                    mask = tables == table
                    if mask.sum() >= MIN_TIERS_PER_TABLE_FOR_SIGN:
                        within[table] = spearman(g[mask], p[mask])
                pooled_sign = np.sign(rho) if math.isfinite(rho) else 0
                finite_within = {t: v for t, v in within.items() if math.isfinite(v)}
                direction_consistent = bool(pooled_sign != 0 and len(finite_within) > 0 and all(np.sign(v) == pooled_sign for v in finite_within.values()))
                entries[label] = {"predictor": col, "rho": rho, "ci_table_block": [lo_t, hi_t], "n_boot_finite_table": n_t,
                                  "ci_tier_stratified": [lo_r, hi_r], "n_boot_finite_tier": n_r, "p_block": pval,
                                  "within_table_rho": within, "direction_consistent": direction_consistent}
                pvals.append(pval)
            adjusted = bh_adjust(pvals)
            for (label, entry), adj in zip(entries.items(), adjusted):
                entry["p_block_bh"] = adj
                entry["significant_bh"] = bool(math.isfinite(adj) and adj < ALPHA)
                entry["ci_table_block_excludes_zero"] = bool(math.isfinite(entry["ci_table_block"][0]) and (entry["ci_table_block"][0] > 0 or entry["ci_table_block"][1] < 0))
            out[str(k)]["methods"][m] = entries
    return out


# ----------------------------------------------------------------------------- A2 threshold rule
def draw_regret_for_rule(draws_k: pd.DataFrame, col: str, c: float, s: float, variant: str) -> pd.Series:
    """Per-draw regret of rule R(c) on draws_k (one k); returns tier-median regret per tier."""
    p = draws_k[col].to_numpy(float)
    use_warm = (s * p) > (s * c)
    fallback = draws_k["regret_scratch"].to_numpy(float) if variant == "S" else draws_k["regret_uncal"].to_numpy(float)
    fallback = np.where(np.isfinite(fallback), fallback, draws_k["regret_uncal"].to_numpy(float))
    regret = np.where(use_warm, draws_k["regret_warm"].to_numpy(float), fallback)
    return pd.Series(regret, index=draws_k.index).groupby(draws_k["tier_key"]).median()


def threshold_rule(summary: pd.DataFrame, draws: pd.DataFrame) -> dict:
    out: dict = {}
    draws = draws.copy()
    draws["tier_key"] = draws["table"] + "|" + draws["tier"]
    for col, label in PREDICTORS.items():
        if col in ("p4_support_overlap", "p5_width_ratio"):
            tier_level = summary.drop_duplicates(["table", "tier"]).set_index(summary.drop_duplicates(["table", "tier"])["table"] + "|" + summary.drop_duplicates(["table", "tier"])["tier"])[col]
            draws[col] = draws["tier_key"].map(tier_level).to_numpy(float)
    rng_state = np.random.default_rng(BOOT_SEED + 1)
    for k in K_VALUES:
        sk = summary[summary.k == k].copy()
        sk["tier_key"] = sk["table"] + "|" + sk["tier"]
        dk = draws[draws.k == k]
        tables = sorted(set(sk["table"]))
        out[str(k)] = {"n_tiers": int(len(sk)), "baselines": {}, "rules": {}}
        baseline_regret = {m: sk.set_index("tier_key")[f"regret_{m}"] for m in ("uncal", "affine", "warm", "scratch")}
        out[str(k)]["baselines"] = {f"always_{m}" if m != "uncal" else "never_calibrate": {"mean_regret": float(np.nanmean(v.to_numpy(float))),
                                                                                           "median_regret": float(np.nanmedian(v.to_numpy(float)))}
                                    for m, v in baseline_regret.items()}
        for col, label in PREDICTORS.items():
            for variant in ("U", "S"):
                if variant == "S" and k < 3:
                    continue
                held_out = {}
                fold_info = {}
                for table in tables:
                    train = sk[sk.table != table]
                    test = sk[sk.table == table]
                    rho = spearman(train["G_warm"].to_numpy(float), train[col].to_numpy(float))
                    s = float(np.sign(rho)) if math.isfinite(rho) and rho != 0 else 1.0
                    values = train[col].to_numpy(float)
                    values = values[np.isfinite(values)]
                    grid = [float(np.percentile(values, q)) for q in GRID_PERCENTILES] if len(values) else [0.0]
                    train_draws = dk[dk.table != table]
                    best_c, best_obj = None, float("inf")
                    for c in grid:
                        reg = draw_regret_for_rule(train_draws, col, c, s, variant)
                        obj = float(reg.mean())
                        if obj < best_obj - 1e-12:
                            best_c, best_obj = c, obj
                    test_draws = dk[dk.table == table]
                    reg_test = draw_regret_for_rule(test_draws, col, best_c, s, variant)
                    held_out.update(reg_test.to_dict())
                    fold_info[table] = {"c": best_c, "sign": s, "train_rho": rho, "train_mean_regret": best_obj, "n_test_tiers": int(len(test))}
                rule_regret = pd.Series(held_out)
                keys = sk["tier_key"].to_numpy()
                rule_vec = rule_regret.reindex(keys).to_numpy(float)
                table_vec = sk["table"].to_numpy()
                comparisons = {}
                for base_name, base in (("always_warm", baseline_regret["warm"]), ("never_calibrate", baseline_regret["uncal"]),
                                        ("always_affine", baseline_regret["affine"]), ("always_scratch", baseline_regret["scratch"])):
                    base_vec = base.reindex(keys).to_numpy(float)
                    d = rule_vec - base_vec
                    ok = np.isfinite(d)
                    rng = np.random.default_rng(rng_state.integers(0, 2**32 - 1))
                    if ok.sum() == 0:
                        comparisons[base_name] = {"mean_diff": float("nan"), "n_tiers": 0, "ci_tier_stratified": [float("nan"), float("nan")],
                                                  "ci_table_block": [float("nan"), float("nan")], "reduced_primary": False, "reduced_secondary": False}
                        continue
                    d_ok, table_ok = d[ok], table_vec[ok]
                    boot_tier = np.array([np.mean(d_ok[idx]) for idx in (block_bootstrap_indices(table_ok, rng, "tier") for _ in range(B))])
                    rng = np.random.default_rng(rng_state.integers(0, 2**32 - 1))
                    boot_table = np.array([np.mean(d_ok[idx]) for idx in (block_bootstrap_indices(table_ok, rng, "table") for _ in range(B))])
                    lo_r, hi_r, _ = percentile_ci(boot_tier)
                    lo_t, hi_t, _ = percentile_ci(boot_table)
                    comparisons[base_name] = {"mean_diff": float(np.nanmean(d[ok])) if ok.any() else float("nan"), "n_tiers": int(ok.sum()),
                                              "ci_tier_stratified": [lo_r, hi_r], "ci_table_block": [lo_t, hi_t],
                                              "reduced_primary": bool(ok.any() and np.nanmean(d[ok]) < 0 and math.isfinite(hi_r) and hi_r < 0),
                                              "reduced_secondary": bool(ok.any() and np.nanmean(d[ok]) < 0 and math.isfinite(hi_t) and hi_t < 0)}
                out[str(k)]["rules"][f"{label}_{variant}"] = {"predictor": col, "variant": variant, "folds": fold_info,
                                                                "mean_regret_held_out": float(np.nanmean(rule_vec)), "median_regret_held_out": float(np.nanmedian(rule_vec)),
                                                                "comparisons": comparisons,
                                                                "reduced_vs_both_baselines_primary": bool(comparisons["always_warm"]["reduced_primary"] and comparisons["never_calibrate"]["reduced_primary"])}
    return out


# ----------------------------------------------------------------------------- A3 decision
def decision(assoc: dict, rules: dict) -> dict:
    per_k = {}
    for k in K_VALUES:
        holds = []
        for label in PREDICTORS.values():
            a = assoc[str(k)]["methods"]["warm"][label]
            cond_i = bool(a["significant_bh"] and a["direction_consistent"])
            cond_ii = any(rules[str(k)]["rules"].get(f"{label}_{v}", {}).get("reduced_vs_both_baselines_primary", False) for v in ("U", "S"))
            if cond_i and cond_ii:
                holds.append(label)
        per_k[str(k)] = {"holds": bool(holds), "predictors": holds}
    counts: dict[str, int] = {}
    for k in K_VALUES:
        for label in per_k[str(k)]["predictors"]:
            counts[label] = counts.get(label, 0) + 1
    overall_predictors = [label for label, c in counts.items() if c >= 2]
    return {"per_k": per_k, "overall_holds": bool(overall_predictors), "overall_predictors": overall_predictors,
            "rule_text": "holds" if overall_predictors else "does not hold"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    run = args.run.resolve()
    tiers = pd.read_csv(run / "tiers.csv")
    draws = pd.read_csv(run / "draws.csv")
    summary = tier_summary(tiers, draws)
    summary.to_csv(run / "tier_summary.csv", index=False, float_format="%.12g")
    assoc = association(summary)
    rules = threshold_rule(summary, draws)
    dec = decision(assoc, rules)
    by_type = {}
    for (k, st), g in summary.groupby(["k", "shift_type"]):
        by_type[f"{k}|{st}"] = {"n_tiers": int(len(g))}
        for m in METHODS:
            vals = g[f"G_{m}"].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            by_type[f"{k}|{st}"][f"median_G_{m}"] = float(np.median(vals)) if len(vals) else float("nan")
            by_type[f"{k}|{st}"][f"frac_G_{m}_positive"] = float(np.mean(vals > 0)) if len(vals) else float("nan")
    results = {"schema_version": "h-fewshot-rule-analysis-v1", "protocol_sha256_from_run": json.loads((run / "run_manifest.json").read_text())["protocol_sha256"],
               "analyzer_sha256": sha256_file(Path(__file__).resolve()), "n_tiers_total": int(len(tiers)),
               "tiers_per_code": json.loads((run / "run_manifest.json").read_text())["tiers_per_code"],
               "n_tiers_by_k": {str(k): int((summary.k == k).sum()) for k in K_VALUES},
               "A1_association": assoc, "A2_threshold_rule": rules, "A3_decision": dec, "descriptive_by_shift_type": by_type}
    (run / "results.json").write_text(json.dumps(sanitize(results), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = []
    for k, block in rules.items():
        for name, r in block["rules"].items():
            rows.append({"k": int(k), "rule": name, "mean_regret": r["mean_regret_held_out"],
                         **{f"diff_vs_{b}": c["mean_diff"] for b, c in r["comparisons"].items()},
                         **{f"ci_lo_vs_{b}": c["ci_tier_stratified"][0] for b, c in r["comparisons"].items()},
                         **{f"ci_hi_vs_{b}": c["ci_tier_stratified"][1] for b, c in r["comparisons"].items()},
                         "reduced_vs_both": r["reduced_vs_both_baselines_primary"]})
    pd.DataFrame(rows).to_csv(run / "rule_summary.csv", index=False, float_format="%.12g")
    hashes = json.loads((run / "artifact_hashes.json").read_text(encoding="utf-8"))
    for name in ("tier_summary.csv", "rule_summary.csv", "results.json"):
        hashes[name] = sha256_file(run / name)
    (run / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": dec, "n_tiers": int(len(tiers)), "n_tiers_by_k": results["n_tiers_by_k"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
