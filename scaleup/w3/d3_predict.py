"""D3 frozen prediction step — D3_BLIND_PROTOCOL.md §5–§7 with D3_DEVIATION_REGISTER.md
(D3-A1 main-lipid strata, D3-C1 two-model interval construction).

Reads: development_labeled.csv (labels allowed), sealed_*_features.csv (inputs only).
Never opens a *_sealed_labels.csv file. Writes a hashed pre-reveal bundle.

Usage:
  python -m w3.d3_predict --artifacts-dir <d3_blind_artifacts> --protocol <D3_BLIND_PROTOCOL.md> \
      --binding <D3_BINDING.json> --register <D3_DEVIATION_REGISTER.md> --output-dir <d3_blind_artifacts/prereveal>
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import xgboost  # noqa: E402
from sklearn.covariance import OAS  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold, GroupShuffleSplit  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier, XGBRegressor  # noqa: E402

NUMERIC = ["ESM", "HSPC", "CHOL", "PEG", "TFR", "FRR"]
AQUEOUS_LEVELS = ("MQ", "PBS")  # development levels, fixed by the binding
SEED = 42
NOMINAL = 0.90
THRESHOLD_QUANTILE = 0.95
AD_RULES = ("knn_5", "mahalanobis_oas", "isolation_forest", "domain_classifier")
CONTROLS = ("random_score", "predicted_size_only")
REFERENCE_MULTIPLIERS = {"reproduction_success": 1.25, "strong_replication": 1.00}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sanitize(value):
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return [sanitize(v) for v in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if math.isnan(f):
            return None
        if math.isinf(f):
            raise ValueError("infinite value cannot enter a JSON artifact")
        return f
    return value


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(sanitize(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def finite_sample_quantile(scores: np.ndarray, coverage: float) -> float:
    values = np.sort(np.asarray(scores, dtype=float))
    rank = min(int(math.ceil((len(values) + 1) * coverage)), len(values))
    return float(values[rank - 1])


def design(frame: pd.DataFrame) -> tuple[np.ndarray, int]:
    """Locked design: six numeric inputs + one-hot AQUEOUS over development levels."""

    x = frame[NUMERIC].to_numpy(dtype=float)
    levels = frame["AQUEOUS"].astype(str).str.strip().to_numpy()
    onehot = np.column_stack([(levels == level).astype(float) for level in AQUEOUS_LEVELS])
    unseen = int((onehot.sum(axis=1) == 0).sum())
    return np.hstack([x, onehot]), unseen


def make_regressor(seed: int) -> XGBRegressor:
    return XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.3,
                        objective="reg:squarederror", random_state=seed, n_jobs=1)


def make_classifier(seed: int) -> XGBClassifier:
    return XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.3,
                         random_state=seed, n_jobs=1)


def main_lipid(frame: pd.DataFrame) -> np.ndarray:
    esm = frame["ESM"].to_numpy(dtype=float) > 0
    hspc = frame["HSPC"].to_numpy(dtype=float) > 0
    out = np.where(esm & ~hspc, "ESM", np.where(hspc & ~esm, "HSPC", "ambiguous"))
    return out.astype(object)


def knn_scores(reference: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    return NearestNeighbors(n_neighbors=min(k, len(reference))).fit(reference).kneighbors(query)[0][:, -1]


def mahalanobis_scores(reference: np.ndarray, *queries: np.ndarray) -> list[np.ndarray]:
    cov = OAS().fit(reference)
    return [np.sqrt(np.einsum("ij,jk,ik->i", q - cov.location_, cov.precision_, q - cov.location_)) for q in queries]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--register", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    art = args.artifacts_dir
    manifest = json.loads((art / "custodian_manifest.json").read_text(encoding="utf-8"))
    binding = json.loads(args.binding.read_text(encoding="utf-8"))
    if sha256_file(args.protocol) != binding["protocol_sha256"]:
        raise ValueError("protocol digest mismatch against binding")
    if sha256_file(args.binding) != manifest["binding"]["sha256"]:
        raise ValueError("binding digest mismatch against custodian manifest")

    # Verify the files this step is allowed to read; sealed label files are never opened here.
    for key in ("development_labeled", "sealed_primary_features", "sealed_secondary_features"):
        entry = manifest["artifacts"][key]
        observed = sha256_file(art / entry["file"])
        if observed != entry["sha256"]:
            raise ValueError(f"{entry['file']}: digest mismatch against custodian manifest")

    dev = pd.read_csv(art / "development_labeled.csv")
    primary = pd.read_csv(art / "sealed_primary_features.csv")
    secondary = pd.read_csv(art / "sealed_secondary_features.csv")
    for frame in (primary, secondary):
        if any(c in frame.columns for c in ("SIZE", "PDI", "OUTPUT")):
            raise RuntimeError("a sealed feature table unexpectedly contains a target column")

    x_dev, unseen_dev = design(dev)
    y_size = dev["SIZE"].to_numpy(dtype=float)
    y_pdi = dev["PDI"].to_numpy(dtype=float)
    out_levels = sorted(dev["OUTPUT"].astype(str).str.strip().unique().tolist())
    formation_ok = out_levels == ["NO", "YES"]
    y_form = (dev["OUTPUT"].astype(str).str.strip() == "YES").astype(int).to_numpy()
    groups = dev["ID"].astype(str).to_numpy()

    # §7 development reference: 5-fold grouped CV MAPE of the locked model on SIZE.
    folds = []
    splitter = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (fit_idx, held_idx) in enumerate(splitter.split(x_dev, groups=groups)):
        model = make_regressor(SEED + fold).fit(x_dev[fit_idx], y_size[fit_idx])
        pred = model.predict(x_dev[held_idx])
        folds.append({"fold": fold, "n_held": int(len(held_idx)),
                      "mape_percent": float(100 * np.mean(np.abs(pred - y_size[held_idx]) / y_size[held_idx]))})
    reference = float(np.mean([f["mape_percent"] for f in folds]))
    reference_payload = {
        "schema_version": "d3-development-reference-v1",
        "estimand": "mean over 5 grouped CV folds (GroupKFold over ID, shuffle, seed 42) of SIZE MAPE on development rows",
        "n_development_rows": int(len(dev)),
        "reference_mape_percent": reference,
        "folds": folds,
        "criteria_thresholds_mape_percent": {k: reference * m for k, m in REFERENCE_MULTIPLIERS.items()},
        "multipliers": REFERENCE_MULTIPLIERS,
    }
    write_json(out / "D3_DEVELOPMENT_REFERENCE.json", reference_payload)

    # §5 point models on all development rows.
    size_model = make_regressor(SEED).fit(x_dev, y_size)
    pdi_model = make_regressor(SEED).fit(x_dev, y_pdi)
    form_model = make_classifier(SEED).fit(x_dev, y_form) if formation_ok else None

    # §6 / D3-C1: proper-training vs calibration partition for interval + applicability.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    proper_idx, cal_idx = next(gss.split(x_dev, groups=groups))
    interval_model = make_regressor(SEED).fit(x_dev[proper_idx], y_size[proper_idx])
    cal_residual = np.abs(y_size[cal_idx] - interval_model.predict(x_dev[cal_idx]))
    q = finite_sample_quantile(cal_residual, NOMINAL)

    scaler = StandardScaler().fit(x_dev[proper_idx])
    proper_s = scaler.transform(x_dev[proper_idx])
    cal_s = scaler.transform(x_dev[cal_idx])
    proper_pred = size_model.predict(x_dev[proper_idx])
    size_centre = float(np.median(proper_pred))
    size_scale = max(float(np.quantile(proper_pred, 0.75) - np.quantile(proper_pred, 0.25)), 1e-9)
    numeric_min = dev[NUMERIC].to_numpy(dtype=float)[proper_idx].min(axis=0)
    numeric_max = dev[NUMERIC].to_numpy(dtype=float)[proper_idx].max(axis=0)
    isolation = IsolationForest(n_estimators=400, contamination="auto", random_state=SEED, n_jobs=1).fit(proper_s)
    rng = np.random.default_rng(SEED + 9173)

    def score_set(frame: pd.DataFrame, label: str) -> tuple[pd.DataFrame, dict]:
        x, unseen = design(frame)
        xs = scaler.transform(x)
        cal_scores: dict[str, np.ndarray] = {}
        test_scores: dict[str, np.ndarray] = {}
        cal_scores["knn_5"], test_scores["knn_5"] = knn_scores(proper_s, cal_s, 5), knn_scores(proper_s, xs, 5)
        cal_scores["mahalanobis_oas"], test_scores["mahalanobis_oas"] = mahalanobis_scores(proper_s, cal_s, xs)
        cal_scores["isolation_forest"] = -isolation.score_samples(cal_s)
        test_scores["isolation_forest"] = -isolation.score_samples(xs)
        clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED).fit(
            np.vstack([proper_s, xs]), np.concatenate([np.zeros(len(proper_s), int), np.ones(len(xs), int)]))
        cal_scores["domain_classifier"] = clf.predict_proba(cal_s)[:, 1]
        test_scores["domain_classifier"] = clf.predict_proba(xs)[:, 1]
        cal_scores["random_score"] = rng.uniform(size=len(cal_idx))
        test_scores["random_score"] = rng.uniform(size=len(frame))
        cal_pred = size_model.predict(x_dev[cal_idx])
        point_pred = size_model.predict(x)
        cal_scores["predicted_size_only"] = np.abs(cal_pred - size_centre) / size_scale
        test_scores["predicted_size_only"] = np.abs(point_pred - size_centre) / size_scale
        numeric = frame[NUMERIC].to_numpy(dtype=float)
        inside = (numeric >= numeric_min) & (numeric <= numeric_max)
        strict = inside.all(axis=1) & (frame["AQUEOUS"].astype(str).str.strip().isin(AQUEOUS_LEVELS).to_numpy())
        records = pd.DataFrame({
            "opaque_row_id": frame["opaque_row_id"].astype(str),
            "stratum_aqueous": frame["AQUEOUS"].astype(str).str.strip(),
            "stratum_main_lipid": main_lipid(frame),
            "predicted_size_nm": point_pred,
            "predicted_pdi": pdi_model.predict(x),
            "predicted_formation_probability": form_model.predict_proba(x)[:, 1] if form_model is not None else np.nan,
            "interval_center_nm": interval_model.predict(x),
            "interval_lower_nm": interval_model.predict(x) - q,
            "interval_upper_nm": interval_model.predict(x) + q,
            "accepted__strict_minmax": strict,
        })
        thresholds = {}
        for key in AD_RULES + CONTROLS:
            thr = float(np.quantile(cal_scores[key], THRESHOLD_QUANTILE, method="higher"))
            thresholds[key] = thr
            records[f"score__{key}"] = test_scores[key]
            records[f"accepted__{key}__q0_95"] = test_scores[key] <= thr
        support = {col: float(inside[:, i].mean()) for i, col in enumerate(NUMERIC)}
        meta = {
            "set": label, "n_rows": int(len(frame)), "unseen_aqueous_levels": int(unseen),
            "thresholds_q0_95_from_calibration": thresholds,
            "strict_minmax_acceptance_fraction": float(strict.mean()),
            "marginal_support_fraction_by_feature": support,
            "strata_counts": {"aqueous": records["stratum_aqueous"].value_counts().to_dict(),
                              "main_lipid": records["stratum_main_lipid"].value_counts().to_dict()},
        }
        return records, meta

    primary_records, primary_meta = score_set(primary, "sealed_primary")
    secondary_records, secondary_meta = score_set(secondary, "sealed_secondary")
    primary_records.to_csv(out / "sealed_primary_predictions.csv", index=False, float_format="%.12g", lineterminator="\n")
    secondary_records.to_csv(out / "sealed_secondary_predictions.csv", index=False, float_format="%.12g", lineterminator="\n")

    here = Path(__file__).resolve()
    code_hashes = {
        "d3_predict.py": sha256_file(here),
        "d3_reveal.py": sha256_file(here.with_name("d3_reveal.py")),
        "d3_custodian.py": sha256_file(here.with_name("d3_custodian.py")),
        "evaluate.py": sha256_file(here.with_name("evaluate.py")),
        args.protocol.name: sha256_file(args.protocol),
        args.binding.name: sha256_file(args.binding),
        args.register.name: sha256_file(args.register),
        "custodian_manifest.json": sha256_file(art / "custodian_manifest.json"),
    }
    environment = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
                   "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__}
    bundle = {
        "schema_version": "d3-prereveal-bundle-v1",
        "locked_model": {"size": "XGBRegressor(100, depth 3, lr 0.3, squared error, seed 42) on all 246 development rows",
                          "pdi": "same configuration", "formation": "XGBClassifier(100, 3, 0.3, seed 42)" if formation_ok else None,
                          "design": NUMERIC + [f"AQUEOUS=={lv}" for lv in AQUEOUS_LEVELS],
                          "development_output_levels": out_levels, "unseen_aqueous_in_development": unseen_dev},
        "interval": {"method": "split conformal, absolute residuals", "nominal": NOMINAL, "n_proper": int(len(proper_idx)),
                      "n_calibration": int(len(cal_idx)), "adjustment_nm": q,
                      "finite_threshold_attainable": bool(int(math.ceil((len(cal_idx) + 1) * NOMINAL)) <= len(cal_idx)),
                      "centre": "model fitted on the proper-training partition (D3-C1)"},
        "applicability": {"rules": list(AD_RULES), "controls": list(CONTROLS), "threshold_quantile": THRESHOLD_QUANTILE,
                           "transductive": ["domain_classifier"]},
        "strata": {"protocol": "aqueous", "amendment_D3_A1": "main_lipid"},
        "development_reference": reference_payload,
        "sets": {"sealed_primary": primary_meta, "sealed_secondary": secondary_meta},
        "records": {"sealed_primary": primary_records.to_dict(orient="records"),
                    "sealed_secondary": secondary_records.to_dict(orient="records")},
        "code_and_lock_hashes": code_hashes,
        "environment": environment,
    }
    bundle_path = out / "blind_prediction_bundle.json"
    write_json(bundle_path, bundle)
    digest = sha256_file(bundle_path)
    (out / "blind_prediction_bundle.sha256").write_text(f"{digest}  blind_prediction_bundle.json\n", encoding="utf-8")
    write_json(out / "environment.json", environment)
    write_json(out / "code_and_lock_hashes.json", code_hashes)
    print(json.dumps(sanitize({"development_reference": reference_payload["reference_mape_percent"],
                               "criteria_thresholds": reference_payload["criteria_thresholds_mape_percent"],
                               "interval_adjustment_nm": q, "sets": {"primary": primary_meta, "secondary": secondary_meta},
                               "bundle_sha256": digest}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
