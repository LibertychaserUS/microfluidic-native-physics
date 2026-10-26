"""Post-hoc tie-aware versions of two descriptive statistics (deviation E-D12; not pre-registered).

    python -I scaleup/w7/closed_loop_tie_check.py --work W --out W/e_tie_check

(1) E4: share of tiers on which each source family has the lowest P2 mean T*. The frozen E4 analyzer used argmin,
    which credits a tie to the first family in its list (XGBoost); here a tie is reported explicitly: lowest or
    tied-lowest, strictly lowest, and ties split equally.
(2) E3: among units whose after-the-fact best action at lambda = 5 is not "continue", the share at which each other
    action attains the minimum (ties between non-continue actions counted for each), by shift type. The frozen E3
    analyzer credited ties to "none" before "switch".
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

FAMILIES = ("xgboost", "random_forest", "mlp", "qrf", "gp", "deep_ensemble")
EPS = 1e-9


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    W = a.work.resolve(); out = a.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    res: dict = {"schema_version": "closed-loop-tie-check-v1", "post_hoc": True, "deviation": "E-D12", "script_sha256": sha(Path(__file__).resolve())}
    l4 = pd.read_csv(W / "e4_mac_run1/loops.csv")
    w = l4[(l4.kind == "main") & (l4.policy == "P2")].groupby(["table", "tier", "family"]).t_star.mean().unstack()[list(FAMILIES)]
    mn = w.min(axis=1); tied = w.sub(mn, axis=0).abs().le(EPS); k = tied.sum(axis=1)
    res["E4"] = {"input_sha256": sha(W / "e4_mac_run1/loops.csv"), "n_tiers": int(len(w)), "n_tiers_all_six_tied": int((k == 6).sum()), "n_tiers_unique_best": int((k == 1).sum()),
                 "lowest_or_tied": {f: float(tied[f].mean()) for f in FAMILIES},
                 "strictly_lowest": {f: float((tied[f] & (k == 1)).mean()) for f in FAMILIES},
                 "ties_split": {f: float(tied[f].div(k).mean()) for f in FAMILIES}}
    e3 = pd.read_csv(W / "e3_run1/loops.csv"); e3 = e3[e3.kind == "main"]
    u = e3.pivot_table(index=["table", "tier", "shift_type", "target", "seed"], columns="arm", values="t_star").reset_index()
    C = np.column_stack([u["continue"], u["none"], u["switch"], u["escalate"] + 5.0]); m = C.min(axis=1)
    other = C[:, 0] > m + EPS; at = np.abs(C - m[:, None]) <= EPS
    def shares(idx):
        o = other[idx]; A = at[idx]
        return {"n_units": int(idx.sum()), "share_other_action_strictly_better": float(o.mean()),
                "share_none_at_min": float((o & A[:, 1]).mean()), "share_switch_at_min": float((o & A[:, 2]).mean()), "share_escalate_at_min": float((o & A[:, 3]).mean()),
                "share_none_and_switch_tied": float((o & A[:, 1] & A[:, 2]).mean()), "share_none_only": float((o & A[:, 1] & ~A[:, 2] & ~A[:, 3]).mean()),
                "share_switch_only": float((o & A[:, 2] & ~A[:, 1] & ~A[:, 3]).mean())}
    res["E3_lambda5"] = {"input_sha256": sha(W / "e3_run1/loops.csv"), "all": shares(np.ones(len(u), bool)),
                         "by_shift_type": {st: shares((u.shift_type == st).to_numpy()) for st in sorted(set(u.shift_type))}}
    (out / "ties.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "artifact_hashes.json").write_text(json.dumps({"ties.json": sha(out / "ties.json")}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"E4_lowest_or_tied": {f: round(v, 3) for f, v in res["E4"]["lowest_or_tied"].items()}, "E4_strict": {f: round(v, 3) for f, v in res["E4"]["strictly_lowest"].items()},
                      "E3_all": {k2: round(v, 3) for k2, v in res["E3_lambda5"]["all"].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
