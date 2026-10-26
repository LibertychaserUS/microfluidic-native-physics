"""Post-hoc check of the E0 noise-floor rule in E2 reporting (deviation E-D13; not pre-registered).

    python -I scaleup/w7/closed_loop_floor_check.py --work W --out W/e_floor_check

The frozen E0-E2 protocol (V2) states that no policy is credited with an error below its table's noise floor
(pooled relative SD of replicate rows, from e0_run1/exclusions.json). The frozen analyzer did not apply this rule.
This script reports (1) the floors with their numbers of replicate groups, (2) per table, how many tier x method x k
medians of the E2 budget curves lie below the floor, and (3) the share of tiers reaching epsilon within 20 rows,
as reported and under the strict reading of the rule (a tier whose epsilon lies below its table's floor counts as
not reached), by method and shift type.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    W = a.work.resolve(); out = a.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    res: dict = {"schema_version": "closed-loop-floor-check-v1", "post_hoc": True, "deviation": "E-D13",
                 "script_sha256": sha(Path(__file__).resolve())}
    nf = json.loads((W / "e0_run1/exclusions.json").read_text())["noise_floor"]
    floor = {t: 100.0 * v["pooled_relative_sd"] for t, v in nf.items()}
    res["floors"] = {t: {"floor_pct": floor[t], "n_replicate_groups": nf[t]["n_replicate_groups"]} for t in sorted(nf)}

    b = pd.read_csv(W / "e1e2_run1/budget.csv")
    med = b.groupby(["table", "tier", "method", "k"]).mape.median().reset_index()
    med["below"] = med.apply(lambda r: r.mape < floor[r.table], axis=1)
    res["curve_medians_below_floor"] = {t: {"n_cells": int(len(g)), "n_below": int(g.below.sum())} for t, g in med.groupby("table")}

    bt = pd.read_csv(W / "e1e2_run1/budget_table.csv")
    bt["table"] = bt.key.str.split("|").str[0]
    bt["eps_below_floor"] = bt.apply(lambda r: r.eps < floor[r.table], axis=1)
    bt["reached_strict"] = bt.reached & ~bt.eps_below_floor
    shares: dict = {}
    for m, g in bt.groupby("method"):
        d = {"all": {"n_tiers": int(len(g)), "reached_pct": 100.0 * g.reached.mean(), "reached_strict_pct": 100.0 * g.reached_strict.mean(),
                     "n_reached_eps_below_floor": int((g.reached & g.eps_below_floor).sum())}}
        for st, h in g.groupby("shift_type"):
            d[st.replace(" ", "_")] = {"n_tiers": int(len(h)), "reached_pct": 100.0 * h.reached.mean(), "reached_strict_pct": 100.0 * h.reached_strict.mean(),
                                       "n_reached_eps_below_floor": int((h.reached & h.eps_below_floor).sum())}
        shares[m] = d
    res["reached_within_20"] = shares
    res["warm_reached_eps_below_floor_by_table"] = {t: int(((g.method == "warm") & g.reached & g.eps_below_floor).sum()) for t, g in bt.groupby("table")}
    res["inputs_sha256"] = {p: sha(W / p) for p in ("e0_run1/exclusions.json", "e1e2_run1/budget.csv", "e1e2_run1/budget_table.csv")}
    (out / "floor.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({m: {k: (round(v["reached_pct"], 1), round(v["reached_strict_pct"], 1)) for k, v in d.items()} for m, d in shares.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
