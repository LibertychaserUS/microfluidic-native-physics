"""E7 hard-task runner — E7_HARD_TASK_PROTOCOL_v1.

    python -I scaleup/w7/closed_loop_e7.py --work W --data D --out W/e7_run1 --e0 W/e0_run1 [--jobs 8] [--smoke]

Imports the frozen E1 runner (`closed_loop.py`, unchanged) and overrides only TAU, TARGET_QUANTILES and the policy
table; adds the interval-aware proposal by wrapping the frozen `propose`.  No wall-clock value is written.
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
PROTOCOL = HERE.parent / "validation_plan" / "E7_HARD_TASK_PROTOCOL_v1.md"
TAU_E7 = 0.03
QUANTILES_E7 = (0.05, 0.10, 0.90, 0.95)
E1_KEEP = ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "CEIL")
NEW = {"P13": (0, "random", "warm", "interval"),
       "A_cont": (4, "random", "warm", "greedy"), "A_none": (4, "random", "none", "greedy"),
       "A_switch": (4, "random", "scratch", "greedy"), "A_esc": (4, "random", "ceiling", "greedy")}
ARMS = {"A_cont": "continue", "A_none": "none", "A_switch": "switch", "A_esc": "escalate"}
SHUFFLE_POLICIES = E1_KEEP + ("P13",)
ARM_K0 = 4
Z_OUTSIDE = 3.0
_STATE: dict = {"width": None}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_runner():
    spec = importlib.util.spec_from_file_location("closed_loop", HERE / "closed_loop.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["closed_loop"] = module
    spec.loader.exec_module(module)
    return module


def configure(cl) -> None:
    cl.TAU = TAU_E7
    cl.TARGET_QUANTILES = QUANTILES_E7
    cl.POLICIES = {**{k: cl.POLICIES[k] for k in E1_KEEP}, **NEW}
    frozen_propose = cl.propose

    def propose(f_adapted, unmeasured, y_star, rule, step, q):
        if rule != "interval":
            return frozen_propose(f_adapted, unmeasured, y_star, rule, step, q)
        w = _STATE["width"]
        d = np.abs(np.log(np.maximum(f_adapted[unmeasured], cl.FLOOR)) - np.log(y_star))
        score = np.maximum(0.0, d - w[unmeasured] / 2.0)
        order = np.lexsort((unmeasured, d, score))
        return int(unmeasured[order[0]])

    cl.propose = propose


def chance_expectation(n: int, m: int, budget: int) -> float:
    return float(sum(comb(n - m, t) / comb(n, t) if t <= n - m else 0.0 for t in range(budget + 1)))


def diagnostics(cl, pc: dict, rows: list[int], delta: float) -> dict:
    lf = np.log(np.maximum(pc["f_tgt"], cl.FLOOR))
    r = np.log(np.maximum(pc["y_tgt"][rows] * (1.0 + delta), cl.FLOOR)) - lf[rows]
    sd = float(np.std(r, ddof=1))
    q = max(float(pc["q_src"]), 1e-12)
    z = pc["z"]
    return {"d1_bias_ratio": abs(float(np.mean(r))) / sd if sd > 0 else 10.0,
            "d2_magnitude_ratio": float(np.mean(np.abs(r))) / q,
            "d3_width_ratio": float(np.mean(pc["qrf_width"])) / q,
            "d4_support_outside": float(np.mean(np.max(np.abs(z), axis=1) > Z_OUTSIDE)),
            "d5_sign_consistency": float(max(int((r > 0).sum()), int((r < 0).sum()))) / len(r)}


def e7_tier(cl, p: dict, smoke: bool) -> dict:
    pc = cl.precompute(p)
    _STATE["width"] = np.asarray(pc["qrf_width"], float)
    y = pc["y_tgt"]; n = len(y)
    cid, log = pc["constructor_id"], pc["log"]
    f_ceiling = cl.predict(cl.fit_source(cid, pc["x_tgt"], y, log), pc["x_tgt"], log)
    shuffled = np.random.default_rng(0).permutation(y)
    f_ceiling_sh = cl.predict(cl.fit_source(cid, pc["x_tgt"], shuffled, log), pc["x_tgt"], log)
    targets = cl.targets_for(y)
    src_min, src_max = float(np.min(pc["y_src"])), float(np.max(pc["y_src"]))
    units = []
    for tname, y_star in targets:
        m = int(np.sum(np.abs(y - y_star) / y_star <= cl.TAU))
        units.append({"table": pc["table"], "tier": pc["tier"], "shift_type": pc["shift_type"], "target": tname, "y_star": y_star, "n": n, "m_within_tau": m,
                      "hit_frac": m / n, "chance_expected_t_star": chance_expectation(n, m, cl.BUDGET), "extrapolative": int(y_star < src_min or y_star > src_max),
                      "src_min": src_min, "src_max": src_max})
    seeds = cl.MAIN_SEEDS[:2] if smoke else cl.MAIN_SEEDS
    dseeds = cl.DRIFT_SEEDS[:1] if smoke else cl.DRIFT_SEEDS
    sseeds = (0,) if smoke else (0, 1, 2)
    plan = [(s, 0.0, "main") for s in seeds] + [(s, d, "drift_pre") for d in cl.DRIFT_PRE for s in dseeds]
    loops = []
    for seed, delta, kind in plan:
        arm_rows = [int(i) for i in np.random.default_rng(seed).choice(n, size=ARM_K0, replace=False)]  # = choose_phase_a(random, 4)
        diag = diagnostics(cl, pc, arm_rows, delta)
        for policy in cl.POLICIES:
            for tname, y_star in targets:
                r = cl.run_loop(pc, policy, y_star * (1.0 + delta), seed, delta, None, f_ceiling)
                row = {"table": pc["table"], "tier": pc["tier"], "shift_type": pc["shift_type"], "policy": policy, "arm": ARMS.get(policy, ""),
                       "target": tname, "y_star": y_star, "seed": seed, "delta": delta, "kind": kind, **r}
                if policy in ARMS:
                    row.update(diag)
                loops.append(row)
    for policy in SHUFFLE_POLICIES:
        for seed in sseeds:
            for tname, y_star in targets:
                y_eff = float(shuffled[int(np.argmin(np.abs(shuffled - y_star)))])
                r = cl.run_loop(pc, policy, y_eff, seed, 0.0, shuffled, f_ceiling_sh)
                loops.append({"table": pc["table"], "tier": pc["tier"], "shift_type": pc["shift_type"], "policy": policy, "arm": "", "target": tname,
                              "y_star": y_star, "seed": seed, "delta": 0.0, "kind": "shuffled", **r})
    return {"loops": loops, "units": units}


def _worker(args):
    p, smoke = args
    return e7_tier(sys.modules["closed_loop"], p, smoke)


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
    configure(cl)
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
    order = sorted(tiers, key=lambda t: -len(t["tgt_idx"]))  # scheduling only; results are order-independent
    payloads = [(cl.tier_payload(tables, t), args.smoke) for t in order]
    with get_context("fork").Pool(processes=max(1, args.jobs)) as pool:
        results = pool.map(_worker, payloads, chunksize=1)
    loops = pd.DataFrame([row for r in results for row in r["loops"]]).sort_values(["table", "tier", "policy", "kind", "delta", "target", "seed"], kind="stable").reset_index(drop=True)
    units = pd.DataFrame([row for r in results for row in r["units"]]).sort_values(["table", "tier", "target"], kind="stable").reset_index(drop=True)
    loops.to_csv(out / "loops.csv", index=False, float_format="%.12g")
    units.to_csv(out / "units.csv", index=False, float_format="%.12g")
    manifest = {"schema_version": "closed-loop-e7-run-v1", "stage": "e7", "protocol": PROTOCOL.name, "protocol_sha256": sha256_file(PROTOCOL),
                "frozen_runner": "closed_loop.py", "frozen_runner_sha256": sha256_file(HERE / "closed_loop.py"), "wrapper_sha256": sha256_file(Path(__file__).resolve()),
                "e1_protocol_sha256": sha256_file(cl.PROTOCOL), "tau": cl.TAU, "budget": cl.BUDGET, "target_quantiles": list(cl.TARGET_QUANTILES),
                "policies": {k: list(v) for k, v in cl.POLICIES.items()}, "arms": ARMS, "arm_k0": ARM_K0, "shuffle_policies": list(SHUFFLE_POLICIES),
                "main_seeds": list(cl.MAIN_SEEDS), "drift_pre": list(cl.DRIFT_PRE), "drift_seeds": list(cl.DRIFT_SEEDS), "smoke": bool(args.smoke),
                "n_tiers": int(len(tiers)), "n_units": int(len(units)), "n_loops": int(len(loops)),
                "e0_sha256": {"tiers_e0.csv": sha256_file(e0 / "tiers_e0.csv"), "exclusions.json": sha256_file(e0 / "exclusions.json")}}
    environment = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
                   "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__}
    for name, payload in (("run_manifest.json", manifest), ("environment.json", environment)):
        (out / name).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    hashes = {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.name != "artifact_hashes.json"}
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"n_tiers": len(tiers), "n_units": int(len(units)), "n_loops": int(len(loops)), "n_hard_units": int((units.hit_frac <= 0.10).sum()),
                      "tiers_with_hard": int(units[units.hit_frac <= 0.10].groupby(["table", "tier"]).ngroups), "n_extrapolative": int(units.extrapolative.sum())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
