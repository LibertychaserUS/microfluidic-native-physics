"""H-J simulation runner — cross-tier meta-analysis of when few-shot adaptation helps.

Implements H_J_FEWSHOT_RULE_PROTOCOL_v1.md Sections 1-4 and 7 (the analysis of Section 5 is in
h_fewshot_rule_analyze.py).  Every few-shot method is imported from the frozen round-3/round-4
scripts (calibrate, warm_start); nothing is re-implemented here.

    python -I scaleup/w6/h_fewshot_rule.py --work <WORK> --data <OSF dir> --out <run dir> [--jobs 6]

The run directory must not exist.  --smoke writes a reduced run (not counted) to the given dir.
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
from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import GroupKFold, KFold  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

SCALEUP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCALEUP))
from w3.posthoc_round3 import FLOOR, calibrate, mape  # noqa: E402
from w3.posthoc_round4_warmstart import WARM_PARAMS, warm_start  # noqa: E402

PROTOCOL = SCALEUP / "validation_plan" / "H_J_FEWSHOT_RULE_PROTOCOL_v1.md"
K_VALUES = (2, 5, 10)
N_DRAWS = 100
MIN_TIER_ROWS = 12
MIN_EVAL_ROWS = 10
LAMBDA = 0.1
P1_CAP = 10.0
KNN = 5
KNN_PERCENTILE = 95.0
OOF_SPLITS = 5
QUANTILE_SEED = 42
XGB_SEED = 42


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def make_model(constructor_id: str):
    """Build the table's locked source constructor by id (picklable across forked workers)."""
    if constructor_id == "D1":
        import dafd_reproduction as DAFD  # noqa: N811

        return DAFD.build_model(XGB_SEED)
    if constructor_id == "D2":
        import native_physics_v2 as NP  # noqa: N811

        return NP.make_xgb(XGB_SEED)
    if constructor_id == "H":
        return sys.modules["h_run2"].make_point("xgboost", XGB_SEED)
    raise ValueError(constructor_id)


# ----------------------------------------------------------------------------- table loading
def load_tables(work: Path, data: Path) -> dict:
    sys.path.insert(0, str(work / "src"))
    import blind_custodian as BC  # noqa: N811
    import dafd_reproduction as DAFD  # noqa: N811
    import expanded_study as ES  # noqa: N811
    import native_physics_v2 as NP  # noqa: N811
    import study_core as SC  # noqa: N811

    tables: dict = {}
    # D1
    comp = DAFD.load_dataset(data / "Comprehensive_normalized.xlsx", require_rate=True)
    ext = DAFD.load_dataset(data / "Generalizability_data_normalized.xlsx", require_rate=False)
    tables["D1"] = {
        "x": comp[DAFD.FEATURES].to_numpy(float), "y": comp[DAFD.TARGET_NORMALIZED].to_numpy(float), "log": False,
        "constructor": "dafd_reproduction.build_model(42): 100 trees, depth 6, lr 0.3",
        "oof_groups": SC.comprehensive_experiment_clusters(comp).astype(str),
        "geometry_groups": SC.group_ids(comp, SC.GEOMETRY_FEATURES).astype(str),
        "event_types": np.asarray(ES.dafd_event_types(len(comp), "comprehensive")).astype(str),
        "ext_x": ext[DAFD.FEATURES].to_numpy(float), "ext_y": ext[DAFD.TARGET_NORMALIZED].to_numpy(float),
        "ext_ref": ext[DAFD.REFERENCE].astype(str).to_numpy(),
        "constructor_id": "D1",
        "sources": {"Comprehensive_normalized.xlsx": sha256_file(data / "Comprehensive_normalized.xlsx"),
                    "Generalizability_data_normalized.xlsx": sha256_file(data / "Generalizability_data_normalized.xlsx")},
    }
    # D2
    mats = [BC.extract_six_column_table(work / "blind_raw" / f) for f in ("matTrainingDataSet.mat", "matTestDataSet.mat")]
    d2 = pd.DataFrame(np.vstack(mats), columns=BC.RAW_INPUT_COLUMNS + [BC.TARGET_COLUMN])
    cmc = np.round(d2["cmc_mm"].to_numpy(float), 2)
    tables["D2"] = {
        "x": NP.b1_features(d2), "y": d2[NP.TARGET].to_numpy(float), "log": False,
        "constructor": "native_physics_v2.make_xgb(42): 100 trees, depth 3, lr 0.3 (B1, four inputs, raw mm)",
        "oof_groups": np.array([f"cmc:{v:g}" for v in cmc]), "cmc": cmc,
        "constructor_id": "D2",
        "sources": {f: sha256_file(work / "blind_raw" / f) for f in ("matTrainingDataSet.mat", "matTestDataSet.mat")},
    }
    # Stage-2 tables
    h_run2 = load_module("h_run2", SCALEUP / "w6" / "h_run2.py")
    binding_a1 = json.loads((SCALEUP / "validation_plan" / "H_BINDING_A1_v2.json").read_text(encoding="utf-8"))
    for code, target_col, label_col, native in (
        ("A1", "L_um", "L_um", list(binding_a1["columns"]["native"])),
        ("R2A1", "target", "target", None),
        ("R2A2", "target", "target", None),
    ):
        art = work / f"h_{code}_artifacts"
        manifest = json.loads((art / "custodian_manifest.json").read_text(encoding="utf-8"))
        devdiag = json.loads((art / f"h_{code}_devdiag.json").read_text(encoding="utf-8"))
        dev = pd.read_csv(art / "development_labeled.csv", keep_default_na=False, na_values=[])
        sealed = pd.read_csv(art / "sealed_features.csv", keep_default_na=False, na_values=[])
        labels = pd.read_csv(art / "sealed_labels.csv", keep_default_na=False, na_values=[])
        sealed = sealed.merge(labels[["row_id", label_col]], on="row_id", how="inner")
        if label_col != target_col:
            sealed = sealed.rename(columns={label_col: target_col})
        full = pd.concat([dev, sealed], ignore_index=True)
        y = full[target_col].to_numpy(float)
        keep = np.isfinite(y) & (y > 0)
        full = full.loc[keep].reset_index(drop=True)
        cols = native if native is not None else list(manifest["native_columns"])
        tables[code] = {
            "x": full[cols].to_numpy(float), "y": full[target_col].to_numpy(float), "log": devdiag["target_scale"] == "log",
            "constructor": "h_run2.make_point('xgboost', 42): 100 trees, depth 3, lr 0.3",
            "frame": full, "constructor_id": "H",
            "sources": {"development_labeled.csv": sha256_file(art / "development_labeled.csv"),
                        "sealed_features.csv": sha256_file(art / "sealed_features.csv"),
                        "sealed_labels.csv": sha256_file(art / "sealed_labels.csv")},
            "n_dropped_nonpositive": int((~keep).sum()),
        }
    tables["A1"]["oof_groups"] = tables["A1"]["frame"]["fluid_family"].astype(str).to_numpy()
    tables["R2A1"]["oof_groups"] = tables["R2A1"]["frame"]["s1_group"].astype(str).to_numpy()
    tables["R2A2"]["oof_groups"] = tables["R2A2"]["frame"]["s1_group"].astype(str).to_numpy()
    return tables


# ----------------------------------------------------------------------------- tier construction
def build_tiers(tables: dict) -> list[dict]:
    tiers: list[dict] = []

    def add(table: str, code: str, level: str, shift: str, src_idx: np.ndarray, tgt_idx: np.ndarray, tgt_external: bool = False):
        if len(tgt_idx) < MIN_TIER_ROWS:
            return
        tiers.append({"table": table, "tier": f"{code}:{level}", "shift_type": shift, "src_idx": np.asarray(src_idx, int),
                      "tgt_idx": np.asarray(tgt_idx, int), "tgt_external": tgt_external})

    d1 = tables["D1"]
    n1 = len(d1["y"])
    for ref in sorted(set(d1["ext_ref"])):
        add("D1", "D1-pool", ref, "laboratory", np.arange(n1), np.flatnonzero(d1["ext_ref"] == ref), tgt_external=True)
    for g in sorted(set(d1["geometry_groups"]), key=lambda s: int(s)):
        mask = d1["geometry_groups"] == g
        add("D1", "D1-geo", g, "device", np.flatnonzero(~mask), np.flatnonzero(mask))
    for t in ("SE", "DE"):
        mask = d1["event_types"] == t
        add("D1", "D1-type", t, "emulsion type", np.flatnonzero(~mask), np.flatnonzero(mask))
    d2 = tables["D2"]
    for level in (0.0, 2.0, 3.5, 4.33, 11.0, 20.0):
        mask = np.isclose(d2["cmc"], level)
        add("D2", "D2-chem", f"cmc{level:g}", "chemistry", np.flatnonzero(~mask), np.flatnonzero(mask))
    specs = (("A1", "fluid_family", "A1-fluid", "chemistry"), ("A1", "device", "A1-device", "device"),
             ("R2A1", "s2_group", "R2A1-fluid", "chemistry"), ("R2A1", "s3_group", "R2A1-device", "device"),
             ("R2A2", "s2_group", "R2A2-lipid", "composition"), ("R2A2", "s3_group", "R2A2-paper", "laboratory"),
             ("R2A2", "s4_group", "R2A2-prep", "preparation"))
    for table, col, code, shift in specs:
        values = tables[table]["frame"][col].astype(str).to_numpy()
        for level in sorted(set(values)):
            mask = values == level
            add(table, code, level, shift, np.flatnonzero(~mask), np.flatnonzero(mask))
    return tiers


# ----------------------------------------------------------------------------- per-tier computation
def oof_folds(groups: np.ndarray, n: int):
    labels = sorted(set(groups.tolist()))
    if len(labels) >= 2:
        return list(GroupKFold(n_splits=min(OOF_SPLITS, len(labels))).split(np.zeros(n), groups=groups)), f"GroupKFold({min(OOF_SPLITS, len(labels))})"
    return list(KFold(n_splits=OOF_SPLITS, shuffle=False).split(np.zeros(n))), "KFold(5) rows (single group)"


def fit_source(make, x: np.ndarray, y: np.ndarray, log: bool):
    model = make()
    model.fit(x, np.log(y) if log else y)
    return model


def predict_source(model, x: np.ndarray, log: bool) -> np.ndarray:
    p = np.asarray(model.predict(x), dtype=float)
    return np.exp(p) if log else p


def run_tier(payload: dict) -> dict:
    table, tier = payload["table"], payload["tier"]
    x_src, y_src, log = payload["x_src"], payload["y_src"], payload["log"]
    x_tgt, y_tgt = payload["x_tgt"], payload["y_tgt"]
    make = lambda: make_model(payload["constructor_id"])  # noqa: E731
    n_src, n = len(y_src), len(y_tgt)
    # source model and base margins
    model = fit_source(make, x_src, y_src, log)
    f_src = predict_source(model, x_src, log)
    f_tgt = predict_source(model, x_tgt, log)
    # E_src and P5 via OOF folds on the source rows
    folds, fold_kind = oof_folds(payload["src_groups"], n_src)
    f_oof = np.full(n_src, np.nan)
    width_oof = np.full(n_src, np.nan)
    ly_src = np.log(np.maximum(y_src, FLOOR))
    for tr, te in folds:
        m = fit_source(make, x_src[tr], y_src[tr], log)
        f_oof[te] = predict_source(m, x_src[te], log)
        q_lo = GradientBoostingRegressor(loss="quantile", alpha=0.05, n_estimators=300, max_depth=3, random_state=QUANTILE_SEED).fit(x_src[tr], ly_src[tr])
        q_hi = GradientBoostingRegressor(loss="quantile", alpha=0.95, n_estimators=300, max_depth=3, random_state=QUANTILE_SEED).fit(x_src[tr], ly_src[tr])
        width_oof[te] = q_hi.predict(x_src[te]) - q_lo.predict(x_src[te])
    e_src = float(np.mean(np.abs(ly_src - np.log(np.maximum(f_oof, FLOOR)))))
    q_lo = GradientBoostingRegressor(loss="quantile", alpha=0.05, n_estimators=300, max_depth=3, random_state=QUANTILE_SEED).fit(x_src, ly_src)
    q_hi = GradientBoostingRegressor(loss="quantile", alpha=0.95, n_estimators=300, max_depth=3, random_state=QUANTILE_SEED).fit(x_src, ly_src)
    width_tgt = q_hi.predict(x_tgt) - q_lo.predict(x_tgt)
    med_oof = float(np.median(width_oof))
    p5 = float(np.median(width_tgt) / med_oof) if med_oof > 0 else float("nan")
    # P4 support overlap
    mu, sd = x_src.mean(axis=0), x_src.std(axis=0)
    keep = sd > 0
    z_src = (x_src[:, keep] - mu[keep]) / sd[keep]
    z_tgt = (x_tgt[:, keep] - mu[keep]) / sd[keep]
    nn = NearestNeighbors(n_neighbors=min(KNN + 1, n_src)).fit(z_src)
    d_src = nn.kneighbors(z_src)[0][:, 1:].mean(axis=1)
    threshold = float(np.percentile(d_src, KNN_PERCENTILE))
    d_tgt = NearestNeighbors(n_neighbors=min(KNN, n_src)).fit(z_src).kneighbors(z_tgt)[0].mean(axis=1)
    p4 = float(np.mean(d_tgt <= threshold))
    # draws
    domain = {"y": y_tgt, "f": f_tgt, "x": x_tgt, "log_target_for_scratch": log}
    source = {"x": x_src, "ly": ly_src, "lf": np.log(np.maximum(f_src, FLOOR))}
    lf_tgt = np.log(np.maximum(f_tgt, FLOOR))
    ly_tgt = np.log(np.maximum(y_tgt, FLOOR))
    rows = []
    eligible = []
    for k in K_VALUES:
        if n - k < MIN_EVAL_ROWS:
            continue
        eligible.append(k)
        for seed in range(payload["n_draws"]):
            idx = np.random.default_rng(seed).choice(n, size=k, replace=False)
            cal = calibrate(domain, idx)
            warm = warm_start(domain, idx, LAMBDA, source)
            r = ly_tgt[idx] - lf_tgt[idx]
            sd_r = float(np.std(r, ddof=1))
            p1 = float(abs(np.mean(r)) / sd_r) if sd_r > 0 else P1_CAP
            p1 = min(p1, P1_CAP)
            p2 = float(max(int((r > 0).sum()), int((r < 0).sum())) / k)
            p3 = float(np.mean(np.abs(r)) / e_src) if e_src > 0 else float("nan")
            affine_fallback = cal["affine"] is None
            m_uncal = cal["baseline"]["mape"]
            m_affine = cal["scale"]["mape"] if affine_fallback else cal["affine"]["mape"]
            m_warm = warm["mape"]
            m_scratch = cal["scratch"]["mape"] if cal["scratch"] is not None else float("nan")
            available = [m_uncal, m_affine, m_warm] + ([m_scratch] if np.isfinite(m_scratch) else [])
            best = float(min(available))
            rows.append({"table": table, "tier": tier, "shift_type": payload["shift_type"], "k": k, "seed": seed, "n_eval": cal["n_eval"],
                         "mape_uncal": m_uncal, "mape_affine": m_affine, "affine_fallback": int(affine_fallback),
                         "mape_warm": m_warm, "mape_scratch": m_scratch,
                         "regret_uncal": m_uncal - best, "regret_affine": m_affine - best, "regret_warm": m_warm - best,
                         "regret_scratch": (m_scratch - best) if np.isfinite(m_scratch) else float("nan"),
                         "p1_bias_ratio": p1, "p2_sign_consistency": p2, "p3_magnitude_ratio": p3})
    tier_row = {"table": table, "tier": tier, "shift_type": payload["shift_type"], "n": n, "n_source": n_src,
                "source_mape_uncal": mape(y_tgt, f_tgt), "e_src_log": e_src, "oof_folds": fold_kind,
                "p4_support_overlap": p4, "p5_width_ratio": p5, "knn_threshold": threshold,
                "k_eligible": " ".join(str(k) for k in eligible), "log_target": int(log)}
    return {"tier_row": tier_row, "draw_rows": rows}


# ----------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--smoke", action="store_true", help="reduced run: 2 tiers per code, 3 draws; not counted")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)

    tables = load_tables(args.work.resolve(), args.data.resolve())
    tiers = build_tiers(tables)
    n_draws = 3 if args.smoke else N_DRAWS
    if args.smoke:
        seen: dict[str, int] = {}
        reduced = []
        for t in tiers:
            code = t["tier"].split(":")[0]
            if seen.get(code, 0) < 2:
                reduced.append(t)
                seen[code] = seen.get(code, 0) + 1
        tiers = reduced
    payloads = []
    for t in tiers:
        tb = tables[t["table"]]
        if t["tgt_external"]:
            x_tgt, y_tgt = tb["ext_x"][t["tgt_idx"]], tb["ext_y"][t["tgt_idx"]]
        else:
            x_tgt, y_tgt = tb["x"][t["tgt_idx"]], tb["y"][t["tgt_idx"]]
        payloads.append({"table": t["table"], "tier": t["tier"], "shift_type": t["shift_type"], "log": tb["log"], "constructor_id": tb["constructor_id"],
                         "x_src": tb["x"][t["src_idx"]], "y_src": tb["y"][t["src_idx"]], "src_groups": tb["oof_groups"][t["src_idx"]],
                         "x_tgt": x_tgt, "y_tgt": y_tgt, "n_draws": n_draws})
    if args.jobs > 1:
        with get_context("fork").Pool(processes=args.jobs) as pool:
            results = pool.map(run_tier, payloads, chunksize=1)
    else:
        results = [run_tier(p) for p in payloads]
    tier_rows = [r["tier_row"] for r in results]
    draw_rows = [row for r in results for row in r["draw_rows"]]
    tiers_df = pd.DataFrame(tier_rows).sort_values(["table", "tier"], kind="stable").reset_index(drop=True)
    draws_df = pd.DataFrame(draw_rows).sort_values(["table", "tier", "k", "seed"], kind="stable").reset_index(drop=True)
    tiers_df.to_csv(out / "tiers.csv", index=False, float_format="%.12g")
    draws_df.to_csv(out / "draws.csv", index=False, float_format="%.12g")
    manifest = {
        "schema_version": "h-fewshot-rule-run-v1", "protocol": PROTOCOL.name, "protocol_sha256": sha256_file(PROTOCOL),
        "runner_sha256": sha256_file(Path(__file__).resolve()), "smoke": bool(args.smoke), "n_draws": n_draws,
        "k_values": list(K_VALUES), "lambda": LAMBDA, "warm_params": {k: v for k, v in WARM_PARAMS.items()},
        "min_tier_rows": MIN_TIER_ROWS, "min_eval_rows": MIN_EVAL_ROWS, "n_tiers": int(len(tiers_df)),
        "tiers_per_code": {code: int(c) for code, c in tiers_df["tier"].str.split(":").str[0].value_counts().sort_index().items()},
        "tables": {code: {"n_rows": int(len(tb["y"])), "n_features": int(tb["x"].shape[1]), "log_target": bool(tb["log"]),
                          "constructor": tb["constructor"], "sources": tb["sources"], "n_dropped_nonpositive": tb.get("n_dropped_nonpositive", 0)}
                   for code, tb in tables.items()},
    }
    environment = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
                   "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__}
    for name, payload in (("run_manifest.json", manifest), ("environment.json", environment)):
        (out / name).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    hashes = {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.name != "artifact_hashes.json"}
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"n_tiers": int(len(tiers_df)), "n_draw_rows": int(len(draws_df)), "tiers_per_code": manifest["tiers_per_code"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
