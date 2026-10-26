"""E6 negative-control runner — E6_NEGATIVE_CONTROLS_PROTOCOL_v1 (shuffled-label replay of all 14 E1 policies + analytic chance level).

    python -I scaleup/w7/closed_loop_e6.py --work W --data D --out W/e6_run1 --e0 W/e0_run1 [--jobs 8] [--smoke]
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
from math import comb  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import xgboost  # noqa: E402

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE.parent / "validation_plan" / "E6_NEGATIVE_CONTROLS_PROTOCOL_v1.md"
SHUFFLE_SEED = 0


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_runner():
    spec = importlib.util.spec_from_file_location("closed_loop", HERE / "closed_loop.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["closed_loop"] = module
    spec.loader.exec_module(module)
    return module


def chance_expectation(n: int, m: int, budget: int) -> float:
    """E[min(T, B+1)] for uniform draws without replacement from n rows of which m are hits."""
    total = 0.0
    for t in range(0, budget + 1):
        total += comb(n - m, t) / comb(n, t) if t <= n - m else 0.0
    return total


def e6_tier(cl, p: dict, smoke: bool) -> dict:
    pc = cl.precompute(p)
    y = pc["y_tgt"]; n = len(y)
    targets = cl.targets_for(y)
    shuffled = np.random.default_rng(SHUFFLE_SEED).permutation(y)
    f_ceiling = cl.predict(cl.fit_source(pc["constructor_id"], pc["x_tgt"], shuffled, pc["log"]), pc["x_tgt"], pc["log"])
    seeds = cl.SHUFFLE_SEEDS[:2] if smoke else cl.SHUFFLE_SEEDS
    loops, chance = [], []
    for tname, y_star in targets:
        m = int(np.sum(np.abs(y - y_star) / y_star <= cl.TAU))
        chance.append({"table": pc["table"], "tier": pc["tier"], "shift_type": pc["shift_type"], "target": tname, "y_star": y_star, "n": n, "m_within_tau": m,
                       "chance_expected_t_star": chance_expectation(n, m, cl.BUDGET)})
        y_star_eff = float(shuffled[int(np.argmin(np.abs(shuffled - y_star)))])
        for policy in cl.POLICIES:
            for seed in seeds:
                r = cl.run_loop(pc, policy, y_star_eff, seed, 0.0, shuffled, f_ceiling)
                loops.append({"table": pc["table"], "tier": pc["tier"], "shift_type": pc["shift_type"], "policy": policy, "target": tname, "y_star": y_star,
                              "seed": seed, "delta": 0.0, "kind": "shuffled", **r})
    return {"loops": loops, "chance": chance}


def _worker(args):
    p, smoke = args
    return e6_tier(sys.modules["closed_loop"], p, smoke)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--e0", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    cl = load_frozen_runner()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    e0 = args.e0.resolve()
    tables, tiers = cl.build_all_tiers(args.work.resolve(), args.data.resolve())
    if args.smoke:
        seen: dict[str, int] = {}
        keep = []
        for t in tiers:
            code = t["tier"].split(":")[0]
            if seen.get(code, 0) < 1:
                keep.append(t); seen[code] = seen.get(code, 0) + 1
        tiers = keep
    excluded = set(json.loads((e0 / "exclusions.json").read_text(encoding="utf-8"))["excluded_tiers"])
    tiers = [t for t in tiers if f"{t['table']}|{t['tier']}" not in excluded]
    payloads = [(cl.tier_payload(tables, t), args.smoke) for t in tiers]
    with get_context("fork").Pool(processes=max(1, args.jobs)) as pool:
        results = pool.map(_worker, payloads, chunksize=1)
    loops = pd.DataFrame([row for r in results for row in r["loops"]]).sort_values(["table", "tier", "policy", "target", "seed"], kind="stable").reset_index(drop=True)
    chance = pd.DataFrame([row for r in results for row in r["chance"]]).sort_values(["table", "tier", "target"], kind="stable").reset_index(drop=True)
    loops.to_csv(out / "loops.csv", index=False, float_format="%.12g")
    chance.to_csv(out / "chance.csv", index=False, float_format="%.12g")
    manifest = {"schema_version": "closed-loop-e6-run-v1", "stage": "e6", "protocol": PROTOCOL.name, "protocol_sha256": sha256_file(PROTOCOL),
                "frozen_runner": "closed_loop.py", "frozen_runner_sha256": sha256_file(HERE / "closed_loop.py"), "wrapper_sha256": sha256_file(Path(__file__).resolve()),
                "e1_protocol_sha256": sha256_file(cl.PROTOCOL), "policies": {k: list(v) for k, v in cl.POLICIES.items()}, "shuffle_seed": SHUFFLE_SEED,
                "shuffle_seeds": list(cl.SHUFFLE_SEEDS), "tau": cl.TAU, "budget": cl.BUDGET, "smoke": bool(args.smoke), "n_tiers": int(len(tiers)), "n_loops": int(len(loops)),
                "e0_sha256": {"tiers_e0.csv": sha256_file(e0 / "tiers_e0.csv"), "exclusions.json": sha256_file(e0 / "exclusions.json")}}
    environment = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
                   "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__}
    for name, payload in (("run_manifest.json", manifest), ("environment.json", environment)):
        (out / name).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    hashes = {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.name != "artifact_hashes.json"}
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"n_tiers": len(tiers), "n_loops": int(len(loops))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
