"""Offline closed-loop benchmark runner — E0_E1_E2_CLOSED_LOOP_PROTOCOL_v1.

    python -I scaleup/w7/closed_loop.py --stage e0   --work W --data D --out W/e0_run1   [--jobs 8] [--smoke]
    python -I scaleup/w7/closed_loop.py --stage e1e2 --work W --data D --out W/e1e2_run1 --e0 W/e0_run1 [--jobs 8] [--smoke]

Tiers and source constructors come from the frozen H-J runner (`w6/h_fewshot_rule.py`) plus the D3
laboratory-B tier; the few-shot method definitions come from the frozen round-3/4 scripts.  Every
random choice is seeded; per-tier work is independent, so the multiprocessing order does not affect
any value.  The run directory must not exist.  Output files never contain absolute paths.
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
import math  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import xgboost  # noqa: E402
from sklearn.model_selection import GroupKFold, KFold  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

SCALEUP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCALEUP))
from w3.posthoc_round2 import d3_design, locked_regressor  # noqa: E402
from w3.posthoc_round3 import FLOOR, farthest_point_rows, kmeans_rows, mape, standardise  # noqa: E402
from w3.posthoc_round4_warmstart import WARM_PARAMS  # noqa: E402
from w3.uq_models import QuantileForest  # noqa: E402

PROTOCOL = SCALEUP / "validation_plan" / "E0_E1_E2_CLOSED_LOOP_PROTOCOL_v1.md"
TAU = 0.10
BUDGET = 20
LAMBDA = 0.1
QRF_SEED = 42
XGB_SEED = 42
OOF_SPLITS = 5
TARGET_QUANTILES = (0.20, 0.50, 0.80)
MAIN_SEEDS = tuple(range(10))
DRIFT_SEEDS = tuple(range(3))
DRIFT_PRE = (-0.2, 0.2)
DRIFT_DESC = (-0.5, 0.5)
DRIFT_DESC_SEEDS = tuple(range(2))
DRIFT_DESC_POLICIES = ("P0", "P1", "P3", "P6")
SHUFFLE_SEEDS = tuple(range(5))
SHUFFLE_POLICIES = ("P0", "P1")
PREVIEW_THRESHOLD = 0.63
E2_K = (1, 2, 3, 5, 8, 10, 15, 20)
E2_DRAWS = 100
E2_MIN_EVAL = 10
EPS_MULT = 1.25
V1_GAP_MAX = TAU / 2
MIN_LEAF_FOR_REPLICATES = 6  # significant digits for replicate detection

# policy = (k0, acquisition, adaptation, proposal)
POLICIES = {
    "P0": (0, "random", "none", "greedy"),
    "P1": (2, "random", "warm", "greedy"),
    "P2": (0, "random", "warm", "greedy"),
    "P3": (5, "random", "warm", "greedy"),
    "P4": (2, "farthest", "warm", "greedy"),
    "P5": (2, "kmeans", "warm", "greedy"),
    "P6": (2, "uncertainty", "warm", "greedy"),
    "P7": (2, "random", "affine", "greedy"),
    "P8": (5, "random", "scratch", "greedy"),
    "P9": (2, "random", "warm", "portfolio"),
    "P10": (5, "farthest", "warm", "portfolio"),
    "P11": (2, "uncertainty", "warm", "portfolio"),
    "P12": (2, "random", "preview", "greedy"),
    "CEIL": (0, "random", "ceiling", "greedy"),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


# ----------------------------------------------------------------------------- tiers
def build_all_tiers(work: Path, data: Path) -> tuple[dict, list[dict]]:
    hj = load_module("h_fewshot_rule", SCALEUP / "w6" / "h_fewshot_rule.py")
    tables = hj.load_tables(work, data)
    tiers = hj.build_tiers(tables)
    # D3 laboratory-B tier (posthoc_round3.load_t2 logic without the MAPE assertion)
    art = work / "d3_blind_artifacts"
    dev = pd.read_csv(art / "development_labeled.csv")
    feats = pd.read_csv(art / "sealed_primary_features.csv")
    scored = pd.read_csv(art / "reveal" / "sealed_primary_scored_rows.csv")
    dev_ok = dev[np.isfinite(dev["SIZE"].to_numpy(float)) & (dev["SIZE"].to_numpy(float) > 0)]
    joined = feats.merge(scored[["opaque_row_id", "actual_size_nm"]], on="opaque_row_id", how="inner")
    x_src = d3_design(dev_ok)
    y_src = dev_ok["SIZE"].to_numpy(float)
    x_tgt = d3_design(joined)
    y_tgt = joined["actual_size_nm"].to_numpy(float)
    tables["D3"] = {"x": np.vstack([x_src, x_tgt]), "y": np.concatenate([y_src, y_tgt]), "log": True,
                    "constructor_id": "D3", "oof_groups": np.array(["row"] * (len(y_src) + len(y_tgt))),
                    "sources": {"development_labeled.csv": sha256_file(art / "development_labeled.csv"),
                                "sealed_primary_features.csv": sha256_file(art / "sealed_primary_features.csv"),
                                "sealed_primary_scored_rows.csv": sha256_file(art / "reveal" / "sealed_primary_scored_rows.csv")}}
    tiers.append({"table": "D3", "tier": "D3-lab:B", "shift_type": "laboratory", "src_idx": np.arange(len(y_src)),
                  "tgt_idx": np.arange(len(y_src), len(y_src) + len(y_tgt)), "tgt_external": False})
    return tables, tiers


def make_model(constructor_id: str):
    if constructor_id == "D3":
        return locked_regressor()
    hj = sys.modules["h_fewshot_rule"]
    return hj.make_model(constructor_id)


def tier_payload(tables: dict, t: dict) -> dict:
    tb = tables[t["table"]]
    if t["tgt_external"]:
        x_tgt, y_tgt = tb["ext_x"][t["tgt_idx"]], tb["ext_y"][t["tgt_idx"]]
    else:
        x_tgt, y_tgt = tb["x"][t["tgt_idx"]], tb["y"][t["tgt_idx"]]
    return {"table": t["table"], "tier": t["tier"], "shift_type": t["shift_type"], "log": tb["log"],
            "constructor_id": tb["constructor_id"], "x_src": tb["x"][t["src_idx"]], "y_src": tb["y"][t["src_idx"]],
            "src_groups": tb["oof_groups"][t["src_idx"]], "x_tgt": x_tgt, "y_tgt": y_tgt}


# ----------------------------------------------------------------------------- per-tier precompute
def oof_folds(groups: np.ndarray, n: int):
    labels = sorted(set(groups.tolist()))
    if len(labels) >= 2:
        return list(GroupKFold(n_splits=min(OOF_SPLITS, len(labels))).split(np.zeros(n), groups=groups))
    return list(KFold(n_splits=OOF_SPLITS, shuffle=False).split(np.zeros(n)))


def fit_source(cid: str, x: np.ndarray, y: np.ndarray, log: bool):
    m = make_model(cid)
    m.fit(x, np.log(y) if log else y)
    return m


def predict(model, x: np.ndarray, log: bool) -> np.ndarray:
    p = np.asarray(model.predict(x), dtype=float)
    return np.exp(p) if log else p


def precompute(p: dict) -> dict:
    x_src, y_src, log, cid = p["x_src"], p["y_src"], p["log"], p["constructor_id"]
    x_tgt, y_tgt = p["x_tgt"], p["y_tgt"]
    model = fit_source(cid, x_src, y_src, log)
    f_src = predict(model, x_src, log)
    f_tgt = predict(model, x_tgt, log)
    f_oof = np.full(len(y_src), np.nan)
    for tr, te in oof_folds(p["src_groups"], len(y_src)):
        f_oof[te] = predict(fit_source(cid, x_src[tr], y_src[tr], log), x_src[te], log)
    dev_ref = mape(y_src, f_oof)
    q_src = float(np.quantile(np.abs(np.log(np.maximum(y_src, FLOOR)) - np.log(np.maximum(f_oof, FLOOR))), 0.90))
    qrf = QuantileForest(QRF_SEED).fit(x_src, np.log(np.maximum(y_src, FLOOR)))
    pr = qrf.predict(x_tgt)
    width = np.asarray(pr.upper, float) - np.asarray(pr.lower, float)
    z = standardise(x_src, x_tgt)
    return {**p, "f_src": f_src, "f_tgt": f_tgt, "dev_ref": dev_ref, "q_src": q_src, "qrf_width": width, "z": z,
            "lf_src": np.log(np.maximum(f_src, FLOOR)), "ly_src": np.log(np.maximum(y_src, FLOOR))}


# ----------------------------------------------------------------------------- E0
def e0_tier(p: dict) -> dict:
    z = p["z"]
    y = p["y_tgt"]
    n = len(y)
    nn = NearestNeighbors(n_neighbors=2).fit(z)
    dist, idx = nn.kneighbors(z)
    d = dist[:, 1]
    gap = np.abs(y[idx[:, 1]] - y) / y
    return {"table": p["table"], "tier": p["tier"], "shift_type": p["shift_type"], "n": n, "n_source": len(p["y_src"]),
            "v1_median_gap": float(np.median(gap)), "v1_p90_gap": float(np.quantile(gap, 0.90)),
            "rho_p95_distance": float(np.quantile(d, 0.95)), "eligible": 1, "dense": int(np.median(gap) <= V1_GAP_MAX),
            "dev_ref_mape": float(p["dev_ref"]), "q_src_log": float(p["q_src"]), "source_mape_uncal": float(mape(y, p["f_tgt"])),
            "log_target": int(p["log"])}


def noise_floor(tables: dict) -> dict:
    out = {}
    for code, tb in tables.items():
        x = tb["x"]
        y = tb["y"]
        keys = [tuple(float(f"{v:.{MIN_LEAF_FOR_REPLICATES}g}") for v in row) for row in x]
        groups: dict[tuple, list[int]] = {}
        for i, k in enumerate(keys):
            groups.setdefault(k, []).append(i)
        rel = []
        for members in groups.values():
            if len(members) >= 2:
                vals = y[members]
                rel.append(float(np.std(vals, ddof=1) / np.mean(vals)))
        out[code] = {"n_replicate_groups": int(len(rel)), "pooled_relative_sd": float(np.sqrt(np.mean(np.square(rel)))) if rel else 0.0}
    return out


# ----------------------------------------------------------------------------- adaptation and proposals
def adapt(p: dict, measured: list[int], y_meas: np.ndarray, method: str) -> np.ndarray:
    """Return predictions for every tier row under the adaptation method with the measured rows."""
    f, x = p["f_tgt"], p["x_tgt"]
    k = len(measured)
    if method == "none" or k == 0:
        return f
    idx = np.asarray(measured, int)
    lf = np.log(np.maximum(f, FLOOR))
    ly = np.log(np.maximum(y_meas, FLOOR))
    if method == "affine":
        if k >= 2 and np.ptp(lf[idx]) > 0:
            A = np.column_stack([lf[idx], np.ones(k)])
            coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
            return np.exp(coef[0] * lf + coef[1])
        return np.exp(lf + float(np.mean(ly - lf[idx])))
    if method == "warm":  # posthoc_round4_warmstart.warm_start, lambda = 0.1
        X = np.vstack([x[idx], p["x_src"]])
        Y = np.concatenate([ly, p["ly_src"]])
        BM = np.concatenate([lf[idx], p["lf_src"]])
        W = np.concatenate([np.ones(k), np.full(len(p["ly_src"]), LAMBDA)])
        reg = XGBRegressor(**WARM_PARAMS)
        reg.fit(X, Y, base_margin=BM, sample_weight=W)
        return np.exp(reg.predict(x, base_margin=lf).astype(float))
    if method == "scratch":
        if k < 3:
            return f
        reg = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.3, objective="reg:squarederror", random_state=XGB_SEED, n_jobs=1)
        if p["log"]:
            reg.fit(x[idx], ly)
            return np.exp(reg.predict(x).astype(float))
        reg.fit(x[idx], y_meas)
        return reg.predict(x).astype(float)
    raise ValueError(method)


def choose_phase_a(p: dict, rule: str, k0: int, rng: np.random.Generator) -> list[int]:
    n = len(p["y_tgt"])
    if k0 == 0:
        return []
    if rule == "random":
        return [int(i) for i in rng.choice(n, size=k0, replace=False)]
    if rule == "farthest":
        return [int(i) for i in farthest_point_rows(p["z"], np.zeros(p["z"].shape[1]), k0)]
    if rule == "kmeans":
        return [int(i) for i in kmeans_rows(p["z"], k0)]
    if rule == "uncertainty":
        order = np.lexsort((np.arange(n), -p["qrf_width"]))
        return [int(i) for i in order[:k0]]
    raise ValueError(rule)


def propose(f_adapted: np.ndarray, unmeasured: np.ndarray, y_star: float, rule: str, step: int, q: float) -> int:
    target = y_star
    if rule == "portfolio":
        s = (0, -1, 1)[step % 3]
        target = y_star * math.exp(s * q)
    cand = unmeasured[np.argsort(np.abs(f_adapted[unmeasured] - target), kind="stable")]
    return int(cand[0])


def run_loop(p: dict, policy: str, y_star: float, seed: int, delta: float, shuffled: np.ndarray | None, f_ceiling: np.ndarray | None) -> dict:
    k0, acq, method, prop = POLICIES[policy]
    y_true = p["y_tgt"] if shuffled is None else shuffled
    n = len(y_true)
    rng = np.random.default_rng(seed)
    measured: list[int] = []
    y_meas: list[float] = []
    regret = 0.0
    t_star = BUDGET + 1
    steps = 0

    def measure(i: int) -> bool:
        nonlocal regret, t_star, steps
        measured.append(i)
        y_obs = float(y_true[i] * (1.0 + delta))
        y_meas.append(y_obs)
        steps += 1
        err = abs(y_obs - y_star) / y_star
        if err <= TAU:
            t_star = steps
            return True
        regret += max(0.0, err - TAU)
        return False

    done = False
    for i in choose_phase_a(p, acq, k0, rng):
        if measure(i):
            done = True
            break
    chosen_method = method
    if method == "preview":
        lf = np.log(np.maximum(p["f_tgt"], FLOOR))
        r = np.log(np.maximum(np.asarray(y_meas), FLOOR)) - lf[np.asarray(measured, int)]
        sd = float(np.std(r, ddof=1)) if len(r) >= 2 else 0.0
        ratio = abs(float(np.mean(r))) / sd if sd > 0 else 10.0
        chosen_method = "warm" if ratio > PREVIEW_THRESHOLD else "none"
    while not done and steps < BUDGET:
        if method == "ceiling":
            f_ad = f_ceiling
        else:
            f_ad = adapt(p, measured, np.asarray(y_meas), chosen_method)
        k = len(measured)
        if prop == "portfolio" and k >= 10:
            res = np.sort(np.abs(np.log(np.maximum(np.asarray(y_meas), FLOOR)) - np.log(np.maximum(p["f_tgt"][np.asarray(measured, int)], FLOOR))))
            rank = min(math.ceil((k + 1) * 0.9), k)
            q = float(res[rank - 1])
        else:
            q = p["q_src"]
        unmeasured = np.setdiff1d(np.arange(n), np.asarray(measured, int), assume_unique=True)
        i = propose(f_ad, unmeasured, y_star, prop, steps - k0 if steps >= k0 else 0, q)
        done = measure(i)
    return {"t_star": int(t_star), "regret": float(regret), "steps": int(steps), "success": int(t_star <= BUDGET),
            "chosen_method": chosen_method}


# ----------------------------------------------------------------------------- E1/E2 per tier
def targets_for(y: np.ndarray) -> list[tuple[str, float]]:
    out = []
    for qn in TARGET_QUANTILES:
        qv = float(np.quantile(y, qn))
        i = int(np.argmin(np.abs(y - qv)))
        out.append((f"q{int(qn * 100)}", float(y[i])))
    return out


def e1e2_tier(p: dict, smoke: bool) -> dict:
    cid, log = p["constructor_id"], p["log"]
    f_ceiling = predict(fit_source(cid, p["x_tgt"], p["y_tgt"], log), p["x_tgt"], log)
    rows = []
    seeds = MAIN_SEEDS[:2] if smoke else MAIN_SEEDS
    drift_seeds = DRIFT_SEEDS[:1] if smoke else DRIFT_SEEDS
    targets = targets_for(p["y_tgt"])
    plan = []
    for policy in POLICIES:
        for seed in seeds:
            plan.append((policy, seed, 0.0, "main"))
        for delta in DRIFT_PRE:
            for seed in drift_seeds:
                plan.append((policy, seed, delta, "drift_pre"))
        if policy in DRIFT_DESC_POLICIES and not smoke:
            for delta in DRIFT_DESC:
                for seed in DRIFT_DESC_SEEDS:
                    plan.append((policy, seed, delta, "drift_desc"))
    shuffled = np.random.default_rng(0).permutation(p["y_tgt"])
    for policy in SHUFFLE_POLICIES:
        for seed in (SHUFFLE_SEEDS[:2] if smoke else SHUFFLE_SEEDS):
            plan.append((policy, seed, 0.0, "shuffled"))
    for policy, seed, delta, kind in plan:
        for tname, y_star in targets:
            if kind == "shuffled":
                y_star_eff = float(shuffled[int(np.argmin(np.abs(shuffled - y_star)))])
                r = run_loop(p, policy, y_star_eff, seed, 0.0, shuffled, f_ceiling)
            else:
                r = run_loop(p, policy, y_star * (1.0 + delta), seed, delta, None, f_ceiling)  # E-D2: target in the drifted scale
            rows.append({"table": p["table"], "tier": p["tier"], "shift_type": p["shift_type"], "policy": policy, "target": tname,
                         "y_star": y_star, "seed": seed, "delta": delta, "kind": kind, **r})
    # E2
    budget_rows = []
    n = len(p["y_tgt"])
    draws = 5 if smoke else E2_DRAWS
    for k in E2_K:
        if n - k < E2_MIN_EVAL:
            continue
        for seed in range(draws):
            idx = [int(i) for i in np.random.default_rng(seed).choice(n, size=k, replace=False)]
            mask = np.ones(n, bool)
            mask[idx] = False
            y_meas = p["y_tgt"][idx]
            for method in ("affine", "warm", "scratch"):
                f_ad = adapt(p, idx, y_meas, method)
                budget_rows.append({"table": p["table"], "tier": p["tier"], "shift_type": p["shift_type"], "k": k, "seed": seed,
                                    "method": method, "mape": float(mape(p["y_tgt"][mask], f_ad[mask])),
                                    "mape_uncal": float(mape(p["y_tgt"][mask], p["f_tgt"][mask]))})
    return {"loops": rows, "budget": budget_rows, "tier": p["tier"]}


def _worker(args):
    p, stage, smoke = args
    pc = precompute(p)
    if stage == "e0":
        return {"e0": e0_tier(pc)}
    return e1e2_tier(pc, smoke)


# ----------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("e0", "e1e2"), required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--e0", type=Path, default=None, help="E0 run directory (required for e1e2)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    tables, tiers = build_all_tiers(args.work.resolve(), args.data.resolve())
    if args.smoke:
        seen: dict[str, int] = {}
        keep = []
        for t in tiers:
            code = t["tier"].split(":")[0]
            if seen.get(code, 0) < 1:
                keep.append(t)
                seen[code] = seen.get(code, 0) + 1
        tiers = keep
    if args.stage == "e1e2":
        assert args.e0 is not None, "--e0 required"
        excl = json.loads((args.e0 / "exclusions.json").read_text(encoding="utf-8"))
        excluded = set(excl["excluded_tiers"])
        tiers = [t for t in tiers if f"{t['table']}|{t['tier']}" not in excluded]
    payloads = [(tier_payload(tables, t), args.stage, args.smoke) for t in tiers]
    with get_context("fork").Pool(processes=max(1, args.jobs)) as pool:
        results = pool.map(_worker, payloads, chunksize=1)
    manifest = {"schema_version": "closed-loop-run-v1.1", "stage": args.stage, "protocol": PROTOCOL.name,
                "protocol_sha256": sha256_file(PROTOCOL), "runner_sha256": sha256_file(Path(__file__).resolve()),
                "smoke": bool(args.smoke), "n_tiers": int(len(tiers)), "policies": {k: list(v) for k, v in POLICIES.items()},
                "tau": TAU, "budget": BUDGET, "lambda": LAMBDA, "warm_params": {k: v for k, v in WARM_PARAMS.items()},
                "target_quantiles": list(TARGET_QUANTILES), "main_seeds": list(MAIN_SEEDS), "drift_pre": list(DRIFT_PRE),
                "drift_seeds": list(DRIFT_SEEDS), "e2_k": list(E2_K), "e2_draws": E2_DRAWS, "eps_mult": EPS_MULT,
                "tables": {c: {"n_rows": int(len(tb["y"])), "n_features": int(tb["x"].shape[1]), "log_target": bool(tb["log"]),
                               "sources": tb["sources"]} for c, tb in tables.items()}}
    if args.stage == "e0":
        df = pd.DataFrame([r["e0"] for r in results]).sort_values(["table", "tier"], kind="stable").reset_index(drop=True)
        df.to_csv(out / "tiers_e0.csv", index=False, float_format="%.12g")
        floors = noise_floor(tables)
        excluded = [f"{r.table}|{r.tier}" for r in df.itertuples() if not r.eligible]
        (out / "exclusions.json").write_text(json.dumps({"rule": "revision 1.1: every tier with >= 12 rows is eligible; dense = median V1 relative gap <= 0.05", "excluded_tiers": excluded, "sparse_tiers": [f"{r.table}|{r.tier}" for r in df.itertuples() if not r.dense], "n_dense": int(df.dense.sum()),
                                                          "n_eligible": int(df.eligible.sum()), "n_total": int(len(df)),
                                                          "noise_floor": floors}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest["n_eligible"] = int(df.eligible.sum())
    else:
        loops = pd.DataFrame([row for r in results for row in r["loops"]]).sort_values(["table", "tier", "policy", "kind", "delta", "target", "seed"], kind="stable").reset_index(drop=True)
        budget = pd.DataFrame([row for r in results for row in r["budget"]]).sort_values(["table", "tier", "k", "seed", "method"], kind="stable").reset_index(drop=True)
        loops.to_csv(out / "loops.csv", index=False, float_format="%.12g")
        budget.to_csv(out / "budget.csv", index=False, float_format="%.12g")
        manifest["e0_sha256"] = {"tiers_e0.csv": sha256_file(args.e0 / "tiers_e0.csv"), "exclusions.json": sha256_file(args.e0 / "exclusions.json")}
        manifest["n_loops"] = int(len(loops))
        manifest["n_budget_rows"] = int(len(budget))
    environment = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
                   "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__}
    for name, payload in (("run_manifest.json", manifest), ("environment.json", environment)):
        (out / name).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    hashes = {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.name != "artifact_hashes.json"}
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k in ("stage", "n_tiers", "n_eligible", "n_loops", "n_budget_rows")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
