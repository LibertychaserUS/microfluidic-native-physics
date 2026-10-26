"""E1b difficulty-sensitivity runner — E1B_DIFFICULTY_PROTOCOL_v1.

    python -I scaleup/w7/closed_loop_e1b.py --work W --data D --out W/e1b_run1 --e0 W/e0_run1 [--jobs 8] [--smoke]
        [--cells 0.05:20,0.10:10,0.05:10]

Wraps the frozen E1 runner (`closed_loop.py`, unchanged) and overrides only TAU and BUDGET per cell; the
descriptive ±0.5 drift runs and the E2 budget curves are switched off by emptying the frozen module's
DRIFT_DESC_POLICIES and E2_K.  Workers are forked, so the overrides are inherited.  Output files never
contain absolute paths.
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
PROTOCOL = HERE.parent / "validation_plan" / "E1B_DIFFICULTY_PROTOCOL_v1.md"
DEFAULT_CELLS = "0.05:20,0.10:10,0.05:10"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_runner():
    spec = importlib.util.spec_from_file_location("closed_loop", HERE / "closed_loop.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["closed_loop"] = module
    spec.loader.exec_module(module)
    return module


def cell_name(tau: float, budget: int) -> str:
    return f"cell_tau{tau:.2f}_B{budget}"


def run_cell(cl, tables, tiers, tau: float, budget: int, out: Path, args, e0: Path) -> dict:
    cl.TAU = float(tau)
    cl.BUDGET = int(budget)
    cl.DRIFT_DESC_POLICIES = ()
    cl.E2_K = ()
    out.mkdir(parents=True, exist_ok=False)
    payloads = [(cl.tier_payload(tables, t), "e1e2", args.smoke) for t in tiers]
    with get_context("fork").Pool(processes=max(1, args.jobs)) as pool:
        results = pool.map(cl._worker, payloads, chunksize=1)
    loops = pd.DataFrame([row for r in results for row in r["loops"]]).sort_values(
        ["table", "tier", "policy", "kind", "delta", "target", "seed"], kind="stable").reset_index(drop=True)
    assert all(len(r["budget"]) == 0 for r in results), "E2 must be switched off in E1b"
    loops.to_csv(out / "loops.csv", index=False, float_format="%.12g")
    manifest = {"schema_version": "closed-loop-e1b-run-v1", "stage": "e1b", "protocol": PROTOCOL.name, "protocol_sha256": sha256_file(PROTOCOL),
                "frozen_runner": "closed_loop.py", "frozen_runner_sha256": sha256_file(HERE / "closed_loop.py"),
                "wrapper_sha256": sha256_file(Path(__file__).resolve()), "e1_protocol_sha256": sha256_file(cl.PROTOCOL),
                "overrides": {"TAU": cl.TAU, "BUDGET": cl.BUDGET, "DRIFT_DESC_POLICIES": [], "E2_K": []},
                "smoke": bool(args.smoke), "n_tiers": int(len(tiers)), "policies": {k: list(v) for k, v in cl.POLICIES.items()},
                "tau": cl.TAU, "budget": cl.BUDGET, "lambda": cl.LAMBDA, "target_quantiles": list(cl.TARGET_QUANTILES),
                "main_seeds": list(cl.MAIN_SEEDS), "drift_pre": list(cl.DRIFT_PRE), "drift_seeds": list(cl.DRIFT_SEEDS),
                "shuffle_seeds": list(cl.SHUFFLE_SEEDS), "n_loops": int(len(loops)),
                "e0_sha256": {"tiers_e0.csv": sha256_file(e0 / "tiers_e0.csv"), "exclusions.json": sha256_file(e0 / "exclusions.json")},
                "tables": {c: {"n_rows": int(len(tb["y"])), "n_features": int(tb["x"].shape[1]), "log_target": bool(tb["log"]),
                               "sources": tb["sources"]} for c, tb in tables.items()}}
    environment = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
                   "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__}
    for name, payload in (("run_manifest.json", manifest), ("environment.json", environment)):
        (out / name).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    hashes = {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.name != "artifact_hashes.json"}
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"cell": out.name, "tau": cl.TAU, "budget": cl.BUDGET, "n_loops": int(len(loops))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--e0", type=Path, required=True)
    ap.add_argument("--cells", default=DEFAULT_CELLS, help="comma-separated tau:budget cells")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    cl = load_frozen_runner()
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
    excl = json.loads((e0 / "exclusions.json").read_text(encoding="utf-8"))
    excluded = set(excl["excluded_tiers"])
    tiers = [t for t in tiers if f"{t['table']}|{t['tier']}" not in excluded]
    summary = []
    for cell in args.cells.split(","):
        tau_s, b_s = cell.split(":")
        tau, budget = float(tau_s), int(b_s)
        summary.append(run_cell(cl, tables, tiers, tau, budget, out / cell_name(tau, budget), args, e0))
        print(json.dumps(summary[-1]), flush=True)
    (out / "cells.json").write_text(json.dumps({"cells": summary, "smoke": bool(args.smoke)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hashes = {str(p.relative_to(out)): sha256_file(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "artifact_hashes.json"}
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
