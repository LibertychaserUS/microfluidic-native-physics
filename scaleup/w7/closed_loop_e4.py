"""E4 model-family runner — E4_MODEL_FAMILY_PROTOCOL_v1.

    python -I scaleup/w7/closed_loop_e4.py --work W --data D --out W/e4_run1 --e0 W/e0_run1 [--jobs 8] [--smoke]

Wraps the frozen E1 runner (`closed_loop.py`, unchanged) for tiers, targets, the loop and the warm-start
adaptation; adds six source-model families.  Output files never contain absolute paths.
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
import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import xgboost  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE.parent / "validation_plan" / "E4_MODEL_FAMILY_PROTOCOL_v1.md"
FAMILIES = ("xgboost", "random_forest", "mlp", "qrf", "gp", "deep_ensemble")
E4_POLICIES = ("P0", "P1", "P2")
FAMILY_SEED = 42
CV_FOLDS = 5
CV_SEED = 0


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_runner():
    spec = importlib.util.spec_from_file_location("closed_loop", HERE / "closed_loop.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["closed_loop"] = module
    spec.loader.exec_module(module)
    return module


class Family:
    """A source model fitted on the runner's target scale; predict() returns the original scale."""

    def __init__(self, name: str, cid: str, log: bool):
        self.name, self.cid, self.log = name, cid, log
        self.extra: dict = {}

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Family":
        cl = sys.modules["closed_loop"]
        yt = np.log(y) if self.log else np.asarray(y, float)
        t0 = time.perf_counter()
        if self.name == "xgboost":
            self.model = cl.make_model(self.cid).fit(x, yt)
            self._pred = lambda xx: np.asarray(self.model.predict(xx), float)
        elif self.name == "random_forest":
            self.model = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=FAMILY_SEED, n_jobs=1).fit(x, yt)
            self._pred = lambda xx: np.asarray(self.model.predict(xx), float)
        elif self.name == "mlp":
            self.scaler = StandardScaler().fit(yt.reshape(-1, 1))
            self.model = Pipeline([("scale", StandardScaler()),
                                   ("mlp", MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu", solver="adam", learning_rate_init=1e-3,
                                                        max_iter=2000, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30,
                                                        random_state=FAMILY_SEED))]).fit(x, self.scaler.transform(yt.reshape(-1, 1)).ravel())
            self.extra["n_iter"] = int(self.model.named_steps["mlp"].n_iter_)
            self._pred = lambda xx: self.scaler.inverse_transform(np.asarray(self.model.predict(xx), float).reshape(-1, 1)).ravel()
        elif self.name == "qrf":
            self.model = cl.QuantileForest(FAMILY_SEED).fit(x, yt)
            self._pred = lambda xx: np.asarray(self.model.predict(xx).point, float)
        elif self.name == "gp":
            uq = sys.modules["w3.uq_models"]
            self.model = uq.GaussianProcessModel(FAMILY_SEED, x.shape[1]).fit(x, yt)
            self.extra["convergence_warnings"] = int(self.model.convergence_warnings)
            self._pred = lambda xx: np.asarray(self.model.predict(xx).point, float)
        elif self.name == "deep_ensemble":
            uq = sys.modules["w3.uq_models"]
            self.model = uq.DeepEnsemble(FAMILY_SEED, standardise_target=True).fit(x, yt)
            self.extra["member_n_iter_mean"] = float(np.mean(self.model.n_iter))
            self._pred = lambda xx: np.asarray(self.model.predict(xx).point, float)
        else:
            raise ValueError(self.name)
        self.fit_seconds = time.perf_counter() - t0
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        p = self._pred(x)
        return np.exp(p) if self.log else p


def e4_tier(cl, p: dict, smoke: bool) -> dict:
    n = len(p["y_tgt"])
    targets = cl.targets_for(p["y_tgt"])
    loops, fam_rows = [], []
    for fam in FAMILIES:
        f = Family(fam, p["constructor_id"], p["log"]).fit(p["x_src"], p["y_src"])
        f_src, f_tgt = f.predict(p["x_src"]), f.predict(p["x_tgt"])
        pc = {**p, "f_src": f_src, "f_tgt": f_tgt, "q_src": 0.0, "qrf_width": np.zeros(n), "z": np.zeros((n, 1)), "dev_ref": float("nan"),
              "lf_src": np.log(np.maximum(f_src, cl.FLOOR)), "ly_src": np.log(np.maximum(p["y_src"], cl.FLOOR))}
        fam_rows.append({"table": p["table"], "tier": p["tier"], "shift_type": p["shift_type"], "family": fam, "n_source": int(len(p["y_src"])), "n_tier": n,
                         "zero_shot_mape": float(cl.mape(p["y_tgt"], f_tgt)), "fit_seconds": round(f.fit_seconds, 3), **{f"extra_{k}": v for k, v in f.extra.items()}})
        plan = [(seed, 0.0, "main") for seed in (cl.MAIN_SEEDS[:2] if smoke else cl.MAIN_SEEDS)]
        for delta in cl.DRIFT_PRE:
            plan += [(seed, delta, "drift_pre") for seed in (cl.DRIFT_SEEDS[:1] if smoke else cl.DRIFT_SEEDS)]
        for policy in E4_POLICIES:
            for seed, delta, kind in plan:
                for tname, y_star in targets:
                    r = cl.run_loop(pc, policy, y_star * (1.0 + delta), seed, delta, None, None)
                    loops.append({"table": p["table"], "tier": p["tier"], "shift_type": p["shift_type"], "family": fam, "policy": policy, "target": tname,
                                  "y_star": y_star, "seed": seed, "delta": delta, "kind": kind, **r})
    return {"loops": loops, "families": fam_rows}


def table_cv(cl, code: str, tb: dict, fam: str) -> list[dict]:
    x, y, log = tb["x"], tb["y"], tb["log"]
    rows = []
    for fold, (tr, te) in enumerate(KFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED).split(x)):
        f = Family(fam, tb["constructor_id"], log).fit(x[tr], y[tr])
        rows.append({"table": code, "family": fam, "fold": fold, "n_fit": int(len(tr)), "n_test": int(len(te)), "mape": float(cl.mape(y[te], f.predict(x[te]))),
                     "fit_seconds": round(f.fit_seconds, 3)})
    return rows


def _worker(args):
    cl = sys.modules["closed_loop"]
    if args[0] == "tier":
        _, p, smoke = args
        return {"kind": "tier", **e4_tier(cl, p, smoke)}
    _, code, tb, fam = args
    return {"kind": "cv", "rows": table_cv(cl, code, tb, fam)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--e0", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--heavy-jobs", type=int, default=3, help="concurrent R2-A2 Gaussian-process tasks (memory-bound)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    cl = load_frozen_runner()
    import w3.uq_models  # noqa: F401  (registers the module for the forked workers)
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
    cv_tables = {t["table"] for t in tiers} if args.smoke else set(tables)
    payloads = [("cv", code, tables[code], fam) for code in sorted(cv_tables) for fam in FAMILIES]  # expensive first
    # scheduling only (results are order-independent): the Gaussian process on the 118-column R2-A2 tiers dominates, so those go first
    order = sorted(tiers, key=lambda t: (t["table"] != "R2A2", -int(len(tables[t["table"]]["y"]) - len(t["tgt_idx"]))))
    payloads += [("tier", cl.tier_payload(tables, t), args.smoke) for t in order]
    # scheduling only (every task is independent and deterministic): the Gaussian process on the 118-column R2-A2 rows needs
    # about 2 GB per fit, so those tasks run in a small pool first and the light tasks in a wider pool afterwards (E-D7)
    heavy = [pl for pl in payloads if (pl[0] == "tier" and pl[1]["table"] == "R2A2") or (pl[0] == "cv" and pl[1] == "R2A2" and pl[3] == "gp")]
    light = [pl for pl in payloads if pl not in heavy]
    results = []
    with get_context("fork").Pool(processes=max(1, args.heavy_jobs)) as pool:
        results += pool.map(_worker, heavy, chunksize=1)
    with get_context("fork").Pool(processes=max(1, args.jobs)) as pool:
        results += pool.map(_worker, light, chunksize=1)
    loops = pd.DataFrame([row for r in results if r["kind"] == "tier" for row in r["loops"]]).sort_values(["table", "tier", "family", "policy", "kind", "delta", "target", "seed"], kind="stable").reset_index(drop=True)
    fams = pd.DataFrame([row for r in results if r["kind"] == "tier" for row in r["families"]]).sort_values(["table", "tier", "family"], kind="stable").reset_index(drop=True)
    cv = pd.DataFrame([row for r in results if r["kind"] == "cv" for row in r["rows"]]).sort_values(["table", "family", "fold"], kind="stable").reset_index(drop=True)
    loops.to_csv(out / "loops.csv", index=False, float_format="%.12g")
    fams.to_csv(out / "families.csv", index=False, float_format="%.12g")
    cv.to_csv(out / "table_cv.csv", index=False, float_format="%.12g")
    manifest = {"schema_version": "closed-loop-e4-run-v1", "stage": "e4", "protocol": PROTOCOL.name, "protocol_sha256": sha256_file(PROTOCOL),
                "frozen_runner": "closed_loop.py", "frozen_runner_sha256": sha256_file(HERE / "closed_loop.py"), "wrapper_sha256": sha256_file(Path(__file__).resolve()),
                "e1_protocol_sha256": sha256_file(cl.PROTOCOL), "families": list(FAMILIES), "policies": {k: list(cl.POLICIES[k]) for k in E4_POLICIES}, "family_seed": FAMILY_SEED,
                "cv_folds": CV_FOLDS, "cv_seed": CV_SEED, "tau": cl.TAU, "budget": cl.BUDGET, "lambda_warm": cl.LAMBDA, "target_quantiles": list(cl.TARGET_QUANTILES),
                "main_seeds": list(cl.MAIN_SEEDS), "drift_pre": list(cl.DRIFT_PRE), "drift_seeds": list(cl.DRIFT_SEEDS), "smoke": bool(args.smoke), "n_tiers": int(len(tiers)),
                "n_loops": int(len(loops)), "n_family_fits": int(len(fams)), "n_cv_fits": int(len(cv)),
                "e0_sha256": {"tiers_e0.csv": sha256_file(e0 / "tiers_e0.csv"), "exclusions.json": sha256_file(e0 / "exclusions.json")},
                "tables": {c: {"n_rows": int(len(tb["y"])), "n_features": int(tb["x"].shape[1]), "log_target": bool(tb["log"]), "sources": tb["sources"]} for c, tb in tables.items()}}
    environment = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
                   "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__}
    for name, payload in (("run_manifest.json", manifest), ("environment.json", environment)):
        (out / name).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    hashes = {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.name != "artifact_hashes.json"}
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"n_tiers": len(tiers), "n_loops": int(len(loops)), "n_family_fits": int(len(fams)), "n_cv_fits": int(len(cv))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
