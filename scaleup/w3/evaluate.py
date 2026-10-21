"""Endpoints, controls and the pre-registered candidate-level test — design D5.

This is the only module that receives test labels (`actual_um`), matching the
dashed boundary in W3_DESIGN.md §2.9.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .applicability import AD_METHODS, ALL_SCORES, CONTROL_METHODS

MIN_GROUP_ROWS = 4
R2_UNSTABLE_BELOW = 15
RELATIVE_TOLERANCE = 0.10
ABSOLUTE_TOLERANCE_UM = 10.0


def aurc_fast(errors: np.ndarray, scores: np.ndarray) -> float:
    """Tie-aware discrete AURC identical to study_core.risk_curve's scalar.

    Rows enter in increasing score; within a tied block the expected cumulative
    error after j of b tied rows is j * block_sum / b. Vectorised so that the
    permutation null can be evaluated thousands of times per cell.
    """

    errors = np.asarray(errors, dtype=float)
    scores = np.asarray(scores, dtype=float)
    n = len(errors)
    if n == 0 or len(scores) != n:
        raise ValueError("errors and scores must be non-empty and equal in length")
    order = np.argsort(scores, kind="mergesort")
    s, e = scores[order], errors[order]
    new_block = np.r_[True, s[1:] != s[:-1]]
    block = np.cumsum(new_block) - 1
    block_size = np.bincount(block).astype(float)
    block_sum = np.bincount(block, weights=e)
    before = np.concatenate([[0.0], np.cumsum(block_sum)[:-1]])
    start = np.flatnonzero(new_block)
    within = np.arange(n) - start[block] + 1
    expected = before[block] + within * (block_sum / block_size)[block]
    return float(np.mean(expected / np.arange(1, n + 1)))


def _r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    if denominator <= 0:
        return float("nan")
    return float(1.0 - np.sum((actual - predicted) ** 2) / denominator)


def point_rows(base: dict, actual_um: np.ndarray, pred_um: np.ndarray, groups: np.ndarray) -> list[dict]:
    """D5 point endpoints: overall row plus one row per test group and a macro row."""

    rel = np.abs(pred_um - actual_um) / actual_um
    abs_um = np.abs(pred_um - actual_um)

    def record(scope: str, mask: np.ndarray) -> dict:
        n = int(mask.sum())
        return {
            **base,
            "scope": scope,
            "n": n,
            "mape_percent": float(100.0 * rel[mask].mean()),
            "mae_um": float(abs_um[mask].mean()),
            "rmse_um": float(np.sqrt(np.mean(abs_um[mask] ** 2))),
            "r2": _r2(actual_um[mask], pred_um[mask]),
            "r2_unstable": bool(n < R2_UNSTABLE_BELOW),
            "signed_relative_bias": float(np.mean((pred_um[mask] - actual_um[mask]) / actual_um[mask])),
        }

    rows = [record("overall", np.ones(len(rel), dtype=bool))]
    group_mapes = []
    for group in np.unique(groups):
        mask = groups == group
        rows.append(record(f"group:{group}", mask))
        group_mapes.append(rows[-1]["mape_percent"])
    rows.append({**base, "scope": "group_macro", "n": int(len(rel)),
                 "mape_percent": float(np.mean(group_mapes)), "mae_um": float("nan"),
                 "rmse_um": float("nan"), "r2": float("nan"), "r2_unstable": True,
                 "signed_relative_bias": float("nan"), "n_groups": int(len(group_mapes))})
    return rows


def interval_row(base: dict, actual_um: np.ndarray, lower_um: np.ndarray, upper_um: np.ndarray,
                 groups: np.ndarray, meta: dict, nominal: float) -> dict:
    covered = (actual_um >= lower_um) & (actual_um <= upper_um)
    group_cov = [bool(covered[groups == g].all()) for g in np.unique(groups)]
    return {
        **base,
        "nominal_coverage": nominal,
        "n": int(len(actual_um)),
        "empirical_coverage": float(covered.mean()),
        "coverage_gap": float(covered.mean() - nominal),
        "mean_width_um": float(np.mean(upper_um - lower_um)),
        "median_width_um": float(np.median(upper_um - lower_um)),
        "mean_relative_width": float(np.mean((upper_um - lower_um) / actual_um)),
        "n_groups": int(len(group_cov)),
        "simultaneous_group_coverage": float(np.mean(group_cov)),
        **{k: meta[k] for k in ("calibration_unit", "n_calibration_units",
                                "nominal_finite_threshold_attainable",
                                "finite_threshold_coverage_ceiling")},
    }


def _within_group_mean(fn, errors: np.ndarray, scores: np.ndarray, groups: np.ndarray):
    values = []
    for group in np.unique(groups):
        mask = groups == group
        if mask.sum() >= MIN_GROUP_ROWS:
            values.append(fn(errors[mask], scores[mask]))
    finite = [v for v in values if np.isfinite(v)]
    return (float(np.mean(finite)), len(finite)) if finite else (float("nan"), 0)


def _spearman(errors: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(scores)) < 2 or len(np.unique(errors)) < 2:
        return float("nan")
    return float(spearmanr(scores, errors).statistic)


def _rule_metrics(prefix: str, accepted: np.ndarray, rel: np.ndarray, abs_um: np.ndarray) -> dict:
    out = {f"{prefix}_retention": float(accepted.mean()), f"{prefix}_n_retained": int(accepted.sum())}
    for label, reliable in (("rel10", rel <= RELATIVE_TOLERANCE), ("abs10um", abs_um <= ABSOLUTE_TOLERANCE_UM)):
        out[f"{prefix}_{label}_precision"] = float(reliable[accepted].mean()) if accepted.any() else float("nan")
        out[f"{prefix}_{label}_recall"] = float((reliable & accepted).sum() / reliable.sum()) if reliable.any() else float("nan")
        out[f"{prefix}_{label}_full_pool_rate"] = float(reliable.mean())
    return out


def ad_rows(base: dict, actual_um: np.ndarray, pred_um: np.ndarray, groups: np.ndarray,
            scores: dict[str, np.ndarray], global_masks: dict[str, np.ndarray],
            within_masks: dict[str, np.ndarray]) -> list[dict]:
    """D5 applicability endpoints for one (cell, repeat, model)."""

    rel = np.abs(pred_um - actual_um) / actual_um
    abs_um = np.abs(pred_um - actual_um)
    n_groups = len(np.unique(groups))
    rows = []
    for method in ALL_SCORES:
        score = scores[method]
        wg_rho, wg_rho_n = _within_group_mean(_spearman, rel, score, groups)
        wg_aurc, wg_aurc_n = _within_group_mean(aurc_fast, rel, score, groups)
        row = {
            **base,
            "ad_method": method,
            "is_control": method in CONTROL_METHODS,
            "n": int(len(rel)),
            "n_test_groups": int(n_groups),
            "spearman_rho_relative": _spearman(rel, score),
            "aurc_relative": aurc_fast(rel, score),
            "full_pool_relative_risk": float(rel.mean()),
            "within_group_spearman_mean": wg_rho,
            "within_group_spearman_groups": wg_rho_n,
            "within_group_aurc_mean": wg_aurc,
            "within_group_aurc_groups": wg_aurc_n,
            "within_group_reason": None if n_groups > 1 else "single_test_group",
        }
        row.update(_rule_metrics("global_q95", global_masks[method], rel, abs_um))
        row.update(_rule_metrics("within_group_median", within_masks[method], rel, abs_um))
        rows.append(row)
    return rows


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(values), np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return adjusted.tolist()
    order = finite[np.argsort(values[finite], kind="mergesort")]
    m = len(order)
    ranked = values[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted.tolist()


def candidate_level_rows(base: dict, actual_um: np.ndarray, pred_um: np.ndarray, groups: np.ndarray,
                         scores: dict[str, np.ndarray], permutations: int, seed: int) -> list[dict]:
    """D5 / protocol §7: within-group permutation test and the three-part verdict."""

    rel = np.abs(pred_um - actual_um) / actual_um
    unique_groups = np.unique(groups)
    eligible = [g for g in unique_groups if (groups == g).sum() >= MIN_GROUP_ROWS]
    rows = []
    if len(unique_groups) < 2 or len(eligible) < 2:
        reason = "single_test_group" if len(unique_groups) < 2 else "fewer_than_two_groups_with_4_rows"
        for method in AD_METHODS:
            rows.append({**base, "ad_method": method, "within_group_aurc": None,
                         "random_within_group_aurc": None, "size_only_within_group_aurc": None,
                         "beats_random": None, "beats_size_only": None, "permutation_p": None,
                         "bh_adjusted_p": None, "permutations": 0, "n_groups_used": 0,
                         "verdict": "not_evaluable", "reason": reason})
        return rows

    def within(score_vector: np.ndarray) -> float:
        return _within_group_mean(aurc_fast, rel, score_vector, groups)[0]

    random_wg = within(scores["random_score"])
    size_wg = within(scores["size_only"])
    rng = np.random.default_rng(seed + 31_337)
    group_positions = [np.flatnonzero(groups == g) for g in eligible]
    p_values: list[float] = []
    observed: list[float] = []
    for method in AD_METHODS:
        score = scores[method]
        obs = within(score)
        null = np.empty(permutations, dtype=float)
        for repeat in range(permutations):
            permuted = score.copy()
            for positions in group_positions:
                permuted[positions] = score[rng.permutation(positions)]
            null[repeat] = within(permuted)
        tolerance = 1e-12 * max(1.0, abs(obs))
        p = float((1 + np.sum(null <= obs + tolerance)) / (permutations + 1))
        observed.append(obs)
        p_values.append(p)
    adjusted = benjamini_hochberg(p_values)
    for method, obs, p, bh in zip(AD_METHODS, observed, p_values, adjusted):
        beats_random = bool(obs < random_wg)
        beats_size = bool(obs < size_wg)
        verdict = "candidate_level_benefit" if (beats_random and beats_size and p < 0.05 and bh < 0.05) else "no_candidate_level_benefit"
        rows.append({**base, "ad_method": method, "within_group_aurc": obs,
                     "random_within_group_aurc": random_wg, "size_only_within_group_aurc": size_wg,
                     "beats_random": beats_random, "beats_size_only": beats_size,
                     "permutation_p": p, "bh_adjusted_p": bh, "permutations": int(permutations),
                     "n_groups_used": int(len(eligible)), "verdict": verdict, "reason": None})
    return rows
