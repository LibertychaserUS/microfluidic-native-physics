"""E3 decision-policy runner — E3_DECISION_POLICY_PROTOCOL_v1.

    python -I scaleup/w7/closed_loop_e3.py --work W --data D --out W/e3_run1 --e0 W/e0_run1 [--jobs 8] [--smoke]

Wraps the frozen E1 runner (`closed_loop.py`, unchanged): adds the arms P13 (switch), P14 (escalate) and P15
(none) to its policy table and reuses precompute/run_loop.  The four diagnostics of protocol §2 are computed
from the two phase-A rows, which are re-derived with the same seeded generator the frozen loop uses.
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
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import xgboost  # noqa: E402

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE.parent / "validation_plan" / "E3_DECISION_POLICY_PROTOCOL_v1.md"
ARMS = {"continue": "P1", "none": "P15", "switch": "P13", "escalate": "P14"}
NEW_POLICIES = {"P13": (2, "random", "scratch", "greedy"), "P14": (2, "random", "ceiling", "greedy"), "P15": (2, "random", "none", "greedy")}
LAMBDAS = (0, 5, 10)
K0 = 2
EPS = 1e-12
Z_OUTSIDE = 3.0  # protocol §2 d4: a row is outside the source support when any standardised column exceeds 3


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_runner():
    spec = importlib.util.spec_from_file_location("closed_loop", HERE / "closed_loop.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["closed_loop"] = module
    spec.loader.exec_module(module)
    return module


def diagnostics(cl, pc: dict, rows: list[int], delta: float) -> dict:
    lf = np.log(np.maximum(pc["f_tgt"], cl.FLOOR))
    y_obs = pc["y_tgt"][rows] * (1.0 + delta)
    r = np.log(np.maximum(y_obs, cl.FLOOR)) - lf[rows]
    sd = float(np.std(r, ddof=1))
    q = max(float(pc["q_src"]), EPS)
    z = pc["z"]
    return {"d1_bias_ratio": abs(float(np.mean(r))) / sd if sd > 0 else 10.0,
            "d2_magnitude_ratio": float(np.mean(np.abs(r))) / q,
            "d3_width_ratio": float(np.mean(pc["qrf_width"])) / q,
            "d4_support_outside": float(np.mean(np.max(np.abs(z), axis=1) > Z_OUTSIDE))}


def e3_tier(cl, p: dict, smoke: bool) -> list[dict]:
    pc = cl.precompute(p)
    cid, log = pc["constructor_id"], pc["log"]
    f_ceiling = cl.predict(cl.fit_source(cid, pc["x_tgt"], pc["y_tgt"], log), pc["x_tgt"], log)
    n = len(pc["y_tgt"])
    targets = cl.targets_for(pc["y_tgt"])
    plan = [(seed, 0.0, "main") for seed in (cl.MAIN_SEEDS[:2] if smoke else cl.MAIN_SEEDS)]
    for delta in cl.DRIFT_PRE:
        plan += [(seed, delta, "drift_pre") for seed in (cl.DRIFT_SEEDS[:1] if smoke else cl.DRIFT_SEEDS)]
    rows_out = []
    for seed, delta, kind in plan:
        rng = np.random.default_rng(seed)
        rows = [int(i) for i in rng.choice(n, size=K0, replace=False)]  # identical to choose_phase_a(..., "random", 2, rng)
        diag = diagnostics(cl, pc, rows, delta)
        for arm, code in ARMS.items():
            for tname, y_star in targets:
                r = cl.run_loop(pc, code, y_star * (1.0 + delta), seed, delta, None, f_ceiling)
                rows_out.append({"table": pc["table"], "tier": pc["tier"], "shift_type": pc["shift_type"], "arm": arm, "policy": code, "target": tname,
                                 "y_star": y_star, "seed": seed, "delta": delta, "kind": kind, "row_a": rows[0], "row_b": rows[1], **diag, **r})
    return rows_out


def _worker(args):
    p, smoke = args
    cl = sys.modules["closed_loop"]
    return e3_tier(cl, p, smoke)


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
    cl.POLICIES = {**cl.POLICIES, **NEW_POLICIES}
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    e0 = args.e0.resolve()
    tables, tiers = cl.build_all_tiers(args.work.resolve(), args.data.resolve())
    if args.smoke:
        seen: dict[str, int] = {}
        keep = []
        for t in tiers:
            code = t["tier"].split(":")[0]
            if seen.get(code, 0) < 1:
                keep.append(t)
                seen[code] = seen.get(code, 0) + 1
        tiers = keep
    excluded = set(json.loads((e0 / "exclusions.json").read_text(encoding="utf-8"))["excluded_tiers"])
    tiers = [t for t in tiers if f"{t['table']}|{t['tier']}" not in excluded]
    payloads = [(cl.tier_payload(tables, t), args.smoke) for t in tiers]
    with get_context("fork").Pool(processes=max(1, args.jobs)) as pool:
        results = pool.map(_worker, payloads, chunksize=1)
    loops = pd.DataFrame([row for r in results for row in r]).sort_values(["table", "tier", "arm", "kind", "delta", "target", "seed"], kind="stable").reset_index(drop=True)
    loops.to_csv(out / "loops.csv", index=False, float_format="%.12g")
    manifest = {"schema_version": "closed-loop-e3-run-v1", "stage": "e3", "protocol": PROTOCOL.name, "protocol_sha256": sha256_file(PROTOCOL),
                "frozen_runner": "closed_loop.py", "frozen_runner_sha256": sha256_file(HERE / "closed_loop.py"), "wrapper_sha256": sha256_file(Path(__file__).resolve()),
                "e1_protocol_sha256": sha256_file(cl.PROTOCOL), "arms": {a: [c, *cl.POLICIES[c]] for a, c in ARMS.items()}, "lambdas": list(LAMBDAS), "k0": K0,
                "tau": cl.TAU, "budget": cl.BUDGET, "lambda_warm": cl.LAMBDA, "target_quantiles": list(cl.TARGET_QUANTILES), "main_seeds": list(cl.MAIN_SEEDS),
                "drift_pre": list(cl.DRIFT_PRE), "drift_seeds": list(cl.DRIFT_SEEDS), "smoke": bool(args.smoke), "n_tiers": int(len(tiers)), "n_loops": int(len(loops)),
                "e0_sha256": {"tiers_e0.csv": sha256_file(e0 / "tiers_e0.csv"), "exclusions.json": sha256_file(e0 / "exclusions.json")},
                "tables": {c: {"n_rows": int(len(tb["y"])), "n_features": int(tb["x"].shape[1]), "log_target": bool(tb["log"]), "sources": tb["sources"]} for c, tb in tables.items()}}
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
