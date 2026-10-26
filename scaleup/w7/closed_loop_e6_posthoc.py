"""E6 post-hoc (not pre-registered): permutation-seed sensitivity of the shuffled-label control for P0/P1/P2/P5.

    python -I scaleup/w7/closed_loop_e6_posthoc.py --work W --data D --out W/e6_posthoc --e0 W/e0_run1 --shuffle-seeds 1,2,3 [--jobs 2]

Imports the frozen E6 runner and overrides only the permutation seed and the policy subset; writes one loops file per
permutation seed and a summary with the block-bootstrap difference P5 − P1 (and P2 − P1) per seed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SUBSET = ("P0", "P1", "P2", "P5")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--e0", type=Path, required=True)
    ap.add_argument("--shuffle-seeds", default="1,2,3")
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args(argv)
    e6 = load("closed_loop_e6", HERE / "closed_loop_e6.py")
    cl = e6.load_frozen_runner()
    an = load("closed_loop_analyze", HERE / "closed_loop_analyze.py")
    cl.POLICIES = {k: v for k, v in cl.POLICIES.items() if k in SUBSET}
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    e0 = args.e0.resolve()
    tables, tiers = cl.build_all_tiers(args.work.resolve(), args.data.resolve())
    excluded = set(json.loads((e0 / "exclusions.json").read_text(encoding="utf-8"))["excluded_tiers"])
    tiers = [t for t in tiers if f"{t['table']}|{t['tier']}" not in excluded]
    summary = {"schema_version": "closed-loop-e6-posthoc-v1", "post_hoc": True, "policies": list(SUBSET), "e6_runner_sha256": sha256_file(HERE / "closed_loop_e6.py"),
               "frozen_runner_sha256": sha256_file(HERE / "closed_loop.py"), "n_tiers": int(len(tiers)), "per_seed": {}}
    for s in [int(x) for x in args.shuffle_seeds.split(",")]:
        e6.SHUFFLE_SEED = s
        payloads = [(cl.tier_payload(tables, t), False) for t in tiers]
        with get_context("fork").Pool(processes=max(1, args.jobs)) as pool:
            results = pool.map(e6._worker, payloads, chunksize=1)
        loops = pd.DataFrame([row for r in results for row in r["loops"]]).sort_values(["table", "tier", "policy", "target", "seed"], kind="stable").reset_index(drop=True)
        loops.to_csv(out / f"loops_shuffle{s}.csv", index=False, float_format="%.12g")
        wide = loops.groupby(["table", "tier", "policy"])["t_star"].mean().reset_index().pivot_table(index=["table", "tier"], columns="policy", values="t_star").reset_index()
        rng_state = np.random.default_rng(an.BOOT_SEED)
        per = {}
        for p in SUBSET:
            if p == "P1":
                continue
            d = an.mean_ci(wide[p].to_numpy(float) - wide["P1"].to_numpy(float), wide["table"].to_numpy(), rng_state)
            lo, hi = d["ci_table_block"]
            d["verdict"] = "better" if (d["mean"] < 0 and hi < 0) else ("worse" if (d["mean"] > 0 and lo > 0) else "not distinguishable")
            per[p] = d
        summary["per_seed"][str(s)] = {"n_loops": int(len(loops)), "levels": {p: float(np.nanmean(wide[p])) for p in SUBSET}, "delta_vs_P1": per}
        print(json.dumps({"shuffle_seed": s, **{p: (v["verdict"], round(v["mean"], 3), [round(x, 3) for x in v["ci_table_block"]]) for p, v in per.items()}}), flush=True)
    (out / "summary.json").write_text(json.dumps(an.sanitize(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hashes = {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.name != "artifact_hashes.json"}
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
