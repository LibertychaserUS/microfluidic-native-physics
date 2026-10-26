"""Post-hoc descriptive summary for H-J (register entry H-J-D3): overall fractions and medians of the
per-tier gains, computed from tier_summary.csv of a counted run.  Not part of the frozen analysis;
every number is a plain summary of the frozen per-tier table and is written to a separate JSON.

    python -I scaleup/w6/h_fewshot_rule_posthoc.py --run <run dir> --out <json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = pd.read_csv(args.run / "tier_summary.csv")
    draws = pd.read_csv(args.run / "draws.csv")
    out = {"schema_version": "h-fewshot-rule-posthoc-v1",
           "tier_summary_sha256": hashlib.sha256((args.run / "tier_summary.csv").read_bytes()).hexdigest(),
           "draws_sha256": hashlib.sha256((args.run / "draws.csv").read_bytes()).hexdigest(),
           "n_draw_rows": int(len(draws)),
           "n_draws_affine_mape_infinite": int(np.isinf(draws["mape_affine"]).sum()),
           "n_draws_affine_fallback_to_shift": int(draws["affine_fallback"].sum()),
           "by_k": {}}
    for k in (2, 5, 10):
        sk = summary[summary.k == k]
        block = {"n_tiers": int(len(sk))}
        for m in ("affine", "warm", "scratch"):
            vals = sk[f"G_{m}"].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            block[f"median_G_{m}"] = float(np.median(vals)) if len(vals) else None
            block[f"fraction_tiers_G_{m}_positive"] = float(np.mean(vals > 0)) if len(vals) else None
        for m in ("uncal", "affine", "warm", "scratch"):
            vals = sk[f"regret_{m}"].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            block[f"median_regret_{m}"] = float(np.median(vals)) if len(vals) else None
        out["by_k"][str(k)] = block
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(out["by_k"], indent=1))


if __name__ == "__main__":
    main()
