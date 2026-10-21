"""D3 one-time reveal — D3_BLIND_PROTOCOL.md §7–§8.

Verifies every frozen hash, consumes an exclusive gate file, joins sealed labels to the
hashed pre-reveal bundle by opaque id, scores every pre-declared endpoint, and saves the
result regardless of sign through a strict JSON writer (NaN → null, infinities rejected).

Usage:
  python -m w3.d3_reveal --artifacts-dir <d3_blind_artifacts> --prereveal-dir <…/prereveal> \
      --output-dir <…/reveal> --protocol <D3_BLIND_PROTOCOL.md> --binding <D3_BINDING.json> \
      --register <D3_DEVIATION_REGISTER.md>
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .evaluate import aurc_fast, benjamini_hochberg  # noqa: E402

AD_RULES = ("knn_5", "mahalanobis_oas", "isolation_forest", "domain_classifier")
CONTROLS = ("random_score", "predicted_size_only")
RELIABLE_RELATIVE = 0.10
MATCHED_BUDGET_REPETITIONS = 10000
STRATUM_PERMUTATIONS = 2000
MIN_STRATUM_ROWS = 4


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


def _r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    return float(1 - np.sum((actual - predicted) ** 2) / denominator) if denominator > 0 else float("nan")


def point_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    rel = (predicted - actual) / actual
    return {
        "n": int(len(actual)),
        "mape_percent": float(100 * np.mean(np.abs(rel))),
        "mdape_percent_descriptive": float(100 * np.median(np.abs(rel))),  # D3-C3: descriptive only
        "mae_nm": float(np.mean(np.abs(predicted - actual))),
        "rmse_nm": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "median_absolute_error_nm": float(np.median(np.abs(predicted - actual))),
        "r2": _r2(actual, predicted),
        "r2_unstable": bool(len(actual) < 15),
        "fraction_within_10pct_relative": float(np.mean(np.abs(rel) <= RELIABLE_RELATIVE)),
        "signed_relative_bias": float(np.mean(rel)),
        "overprediction_fraction": float(np.mean(predicted > actual)),
    }


def matched_budget_random(rel_err: np.ndarray, accepted: np.ndarray, seed: int) -> dict:
    k, n = int(accepted.sum()), len(rel_err)
    observed = float(rel_err[accepted].mean()) if k else float("nan")
    if k == 0 or k == n:
        return {"retained_count": k, "observed_relative_risk": observed, "random_mean_risk": observed,
                "one_sided_p_random_risk_le_observed": 1.0, "survives_at_0_05": False,
                "degenerate_null": True, "repetitions": 0}
    rng = np.random.default_rng(seed)
    risks = np.array([rel_err[rng.choice(n, size=k, replace=False)].mean() for _ in range(MATCHED_BUDGET_REPETITIONS)])
    p = float((1 + np.sum(risks <= observed + 1e-12)) / (MATCHED_BUDGET_REPETITIONS + 1))
    return {"retained_count": k, "observed_relative_risk": observed, "random_mean_risk": float(risks.mean()),
            "random_risk_95_interval": [float(np.quantile(risks, 0.025)), float(np.quantile(risks, 0.975))],
            "one_sided_p_random_risk_le_observed": p, "survives_at_0_05": bool(p < 0.05),
            "degenerate_null": False, "repetitions": MATCHED_BUDGET_REPETITIONS}


def applicability_block(records: pd.DataFrame, rel_err: np.ndarray) -> dict:
    full_risk = float(rel_err.mean())
    reliable = np.abs(rel_err) <= RELIABLE_RELATIVE
    strict = records["accepted__strict_minmax"].to_numpy(dtype=bool)
    strict_risk = float(rel_err[strict].mean()) if strict.any() else float("nan")
    strict_retention = float(strict.mean())
    rules: dict[str, dict] = {}
    p_values, names = [], []
    for i, key in enumerate(AD_RULES + CONTROLS + ("strict_minmax",)):
        accepted = strict if key == "strict_minmax" else records[f"accepted__{key}__q0_95"].to_numpy(dtype=bool)
        k = int(accepted.sum())
        retained_risk = float(rel_err[accepted].mean()) if k else float("nan")
        comparable = abs(float(accepted.mean()) - strict_retention) <= 0.10 if k else False
        block = {
            "retained_count": k, "retention_fraction": float(accepted.mean()),
            "candidate_reduction_fraction": float(1 - accepted.mean()),
            "retained_relative_risk": retained_risk, "full_pool_relative_risk": full_risk,
            "risk_lower_than_full_pool": bool(k and retained_risk < full_risk),
            "reliable_candidate_precision": float(reliable[accepted].mean()) if k else float("nan"),
            "reliable_candidate_recall": float((reliable & accepted).sum() / reliable.sum()) if reliable.any() else float("nan"),
            "strict_minmax_retention_fraction": strict_retention, "strict_minmax_relative_risk": strict_risk,
            "retention_comparable_to_strict_minmax_within_10pct_pool": bool(comparable),
            "not_weaker_than_strict_minmax_when_comparable": bool((not comparable) or (k and retained_risk <= strict_risk + 1e-12)),
            "matched_budget_random_retention": matched_budget_random(rel_err, accepted, seed=42 + i),
        }
        block["beneficial_under_all_locked_criteria"] = bool(
            block["risk_lower_than_full_pool"] and block["not_weaker_than_strict_minmax_when_comparable"]
            and block["matched_budget_random_retention"]["survives_at_0_05"] and key in AD_RULES)
        rules[key] = block
        if key in AD_RULES:
            names.append(key)
            p_values.append(block["matched_budget_random_retention"]["one_sided_p_random_risk_le_observed"])
    adjusted = benjamini_hochberg(p_values)
    for name, bh in zip(names, adjusted):
        rules[name]["bh_adjusted_p_across_four_rules"] = bh
        rules[name]["bh_significant_at_0_05"] = bool(np.isfinite(bh) and bh < 0.05)
    return {"full_pool_relative_risk": full_risk, "n_reliable_full_pool": int(reliable.sum()),
            "reliable_definition": f"absolute relative error <= {RELIABLE_RELATIVE}", "rules": rules}


def candidate_level_block(records: pd.DataFrame, rel_err: np.ndarray, stratum_col: str, seed: int) -> dict:
    strata = records[stratum_col].astype(str).to_numpy()
    levels = [s for s in np.unique(strata) if (strata == s).sum() >= MIN_STRATUM_ROWS]
    if len(levels) < 2:
        return {"stratum_column": stratum_col, "evaluable": False,
                "reason": "single_level" if len(np.unique(strata)) < 2 else "fewer_than_two_levels_with_4_rows",
                "levels": {str(s): int((strata == s).sum()) for s in np.unique(strata)}}
    positions = [np.flatnonzero(strata == s) for s in levels]
    abs_rel = np.abs(rel_err)

    def within(score: np.ndarray) -> float:
        return float(np.mean([aurc_fast(abs_rel[p], score[p]) for p in positions]))

    controls = {key: within(records[f"score__{key}"].to_numpy(dtype=float)) for key in CONTROLS}
    rng = np.random.default_rng(seed)
    results, p_values = {}, []
    for key in AD_RULES:
        score = records[f"score__{key}"].to_numpy(dtype=float)
        observed = within(score)
        null = np.empty(STRATUM_PERMUTATIONS)
        for r in range(STRATUM_PERMUTATIONS):
            permuted = score.copy()
            for p in positions:
                permuted[p] = score[rng.permutation(p)]
            null[r] = within(permuted)
        p = float((1 + np.sum(null <= observed + 1e-12)) / (STRATUM_PERMUTATIONS + 1))
        results[key] = {"within_stratum_aurc": observed, "beats_random": bool(observed < controls["random_score"]),
                        "beats_size_only": bool(observed < controls["predicted_size_only"]), "permutation_p": p}
        p_values.append(p)
    for key, bh in zip(AD_RULES, benjamini_hochberg(p_values)):
        r = results[key]
        r["bh_adjusted_p"] = bh
        r["verdict"] = ("candidate_level_benefit" if (r["beats_random"] and r["beats_size_only"] and r["permutation_p"] < 0.05 and bh < 0.05)
                        else "no_candidate_level_benefit")
    return {"stratum_column": stratum_col, "evaluable": True, "levels": {str(s): int((strata == s).sum()) for s in levels},
            "control_within_stratum_aurc": controls, "rules": results, "permutations": STRATUM_PERMUTATIONS}


def score_set(records: pd.DataFrame, labels: pd.DataFrame, reference: dict | None, label: str) -> tuple[dict, pd.DataFrame]:
    if set(records["opaque_row_id"]) != set(labels["opaque_row_id"]):
        raise ValueError(f"{label}: opaque id sets differ between bundle and sealed labels")
    joined = records.merge(labels, on="opaque_row_id", how="inner", validate="one_to_one")
    size = pd.to_numeric(joined["SIZE"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(size) & (size > 0)
    result: dict = {"set": label, "n_rows": int(len(joined)), "n_rows_with_finite_size": int(finite.sum()),
                    "n_rows_missing_size": int((~finite).sum())}
    j = joined.loc[finite].reset_index(drop=True)
    actual = size[finite]
    pred = j["predicted_size_nm"].to_numpy(dtype=float)
    result["point_prediction"] = point_metrics(actual, pred)
    if reference is not None:
        mape = result["point_prediction"]["mape_percent"]
        result["pre_declared_criteria"] = {
            "development_reference_mape_percent": reference["reference_mape_percent"],
            **{name: {"threshold_mape_percent": thr, "met": bool(mape <= thr)}
               for name, thr in reference["criteria_thresholds_mape_percent"].items()},
        }
    lower, upper = j["interval_lower_nm"].to_numpy(float), j["interval_upper_nm"].to_numpy(float)
    covered = (actual >= lower) & (actual <= upper)
    result["interval_90"] = {"observed_coverage": float(covered.mean()), "mean_width_nm": float(np.mean(upper - lower)),
                             "guarantee_across_laboratories_asserted": False}
    rel_err = (pred - actual) / actual
    result["applicability"] = applicability_block(j, rel_err)
    result["candidate_level"] = {
        "aqueous": candidate_level_block(j, rel_err, "stratum_aqueous", seed=101),
        "main_lipid_D3_A1": candidate_level_block(j, rel_err, "stratum_main_lipid", seed=202),
    }
    pdi = pd.to_numeric(joined["PDI"], errors="coerce").to_numpy(dtype=float)
    pdi_ok = np.isfinite(pdi) & (pdi > 0)
    if pdi_ok.any():
        result["secondary_pdi"] = point_metrics(pdi[pdi_ok], joined.loc[pdi_ok, "predicted_pdi"].to_numpy(float))
    form = joined["OUTPUT"].astype(str).str.strip().to_numpy()
    prob = joined["predicted_formation_probability"].to_numpy(float)
    if set(np.unique(form)) <= {"YES", "NO"} and np.isfinite(prob).all():
        truth = (form == "YES").astype(int)
        predicted = (prob >= 0.5).astype(int)
        tp = int(((predicted == 1) & (truth == 1)).sum()); fp = int(((predicted == 1) & (truth == 0)).sum())
        fn = int(((predicted == 0) & (truth == 1)).sum())
        result["secondary_formation"] = {"n": int(len(truth)), "positives": int(truth.sum()),
                                         "accuracy": float((predicted == truth).mean()),
                                         "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else float("nan"),
                                         "both_classes_present": bool(0 < truth.sum() < len(truth))}
    j["actual_size_nm"] = actual
    j["absolute_relative_error"] = np.abs(rel_err)
    j["covered_90"] = covered
    return result, j


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--prereveal-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--register", type=Path, required=True)
    args = parser.parse_args(argv)

    art, pre, out = args.artifacts_dir, args.prereveal_dir, args.output_dir
    manifest = json.loads((art / "custodian_manifest.json").read_text(encoding="utf-8"))
    bundle_path = pre / "blind_prediction_bundle.json"
    bundle_digest = sha256_file(bundle_path)
    recorded = (pre / "blind_prediction_bundle.sha256").read_text(encoding="utf-8").split()[0]
    if bundle_digest != recorded:
        raise ValueError("prediction bundle digest does not match its sidecar")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    locks = bundle["code_and_lock_hashes"]
    here = Path(__file__).resolve()
    checks = {
        "protocol": sha256_file(args.protocol) == locks[args.protocol.name],
        "binding": sha256_file(args.binding) == locks[args.binding.name],
        "register": sha256_file(args.register) == locks[args.register.name],
        "custodian_manifest": sha256_file(art / "custodian_manifest.json") == locks["custodian_manifest.json"],
        "d3_predict.py": sha256_file(here.with_name("d3_predict.py")) == locks["d3_predict.py"],
        "d3_reveal.py": sha256_file(here) == locks["d3_reveal.py"],
        "evaluate.py": sha256_file(here.with_name("evaluate.py")) == locks["evaluate.py"],
    }
    for key in ("sealed_primary_sealed_labels", "sealed_secondary_sealed_labels", "sealed_primary_features", "sealed_secondary_features"):
        entry = manifest["artifacts"][key]
        checks[key] = sha256_file(art / entry["file"]) == entry["sha256"]
    if not all(checks.values()):
        raise ValueError(f"frozen-hash verification failed: {checks}")

    out.mkdir(parents=True, exist_ok=True)
    gate = out / "D3_REVEAL_CONSUMED.json"
    fd = os.open(gate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)  # refuses a second reveal
    reveal_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump({"prediction_bundle_sha256": bundle_digest, "reveal_utc": reveal_utc, "status": "reveal gate consumed"}, fh, indent=2)
        fh.write("\n")

    primary_labels = pd.read_csv(art / manifest["artifacts"]["sealed_primary_sealed_labels"]["file"])
    secondary_labels = pd.read_csv(art / manifest["artifacts"]["sealed_secondary_sealed_labels"]["file"])
    primary_records = pd.DataFrame(bundle["records"]["sealed_primary"])
    secondary_records = pd.DataFrame(bundle["records"]["sealed_secondary"])
    reference = bundle["development_reference"]

    primary_result, primary_rows = score_set(primary_records, primary_labels, reference, "sealed_primary_laboratory_B")
    secondary_result, secondary_rows = score_set(secondary_records, secondary_labels, None, "sealed_secondary_operator_B")
    secondary_result["status"] = "secondary set outside the frozen primary estimand; descriptive only"

    result = {
        "schema_version": "d3-reveal-v1",
        "reveal_utc": reveal_utc,
        "prediction_bundle_sha256": bundle_digest,
        "frozen_hash_checks": checks,
        "development_reference": reference,
        "sealed_primary": primary_result,
        "sealed_secondary": secondary_result,
        "interpretation_boundaries": [
            "Sealed laboratory-B rows all use PBS while development has 15 PBS rows (D3-B3): laboratory shift is confounded with buffer shift.",
            "Strict min-max acceptance fraction and marginal support fractions are in the pre-reveal bundle metadata.",
            "No guarantee is asserted for the 90% interval across laboratories.",
            "Secondary set (operator B) is descriptive only.",
        ],
    }
    write_json(out / "blind_reveal_results.json", result)
    primary_rows.to_csv(out / "sealed_primary_scored_rows.csv", index=False, float_format="%.12g", lineterminator="\n")
    secondary_rows.to_csv(out / "sealed_secondary_scored_rows.csv", index=False, float_format="%.12g", lineterminator="\n")
    summary = {"reveal_utc": reveal_utc, "primary_mape_percent": primary_result["point_prediction"]["mape_percent"],
               "criteria": primary_result.get("pre_declared_criteria"), "coverage_90": primary_result["interval_90"]["observed_coverage"]}
    print(json.dumps(sanitize(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
