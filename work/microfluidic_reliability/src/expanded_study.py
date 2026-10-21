"""Run the frozen expanded reliability and retrospective-search study.

This script is intentionally exhaustive rather than optimized for a single
positive result. It releases split manifests, per-row predictions, continuous
risk curves, every locked threshold, negative controls, and replay selections.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn
import xgboost
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from study_core import (
    ABSOLUTE_TOLERANCES_UM,
    DAFD,
    DEVICE_FLUID_FEATURES,
    GEOMETRY_FEATURES,
    RELATIVE_TOLERANCES,
    RISK_QUANTILES,
    build_regressor,
    comprehensive_experiment_clusters,
    dependency_group_ids,
    external_experiment_clusters,
    fit_risk_scores,
    group_ids,
    prediction_frame,
    rank_candidates,
    risk_curve,
    summarize_prediction_rows,
    threshold_from_training,
)
from strict_json import write_strict_json as write_json


MODEL_NAMES = ("xgboost", "random_forest", "hist_gradient_boosting")
TARGET_GRID_UM = np.arange(30.0, 231.0, 10.0)
TOP_K_VALUES = (1, 3, 5)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_manifest() -> dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "matplotlib": importlib.import_module("matplotlib").__version__,
        "seaborn": sns.__version__,
    }


def reproducibility_manifest(
    project_dir: Path,
    test_evidence_path: Path,
) -> dict[str, object]:
    """Hash code, governance documents, and explicit test-evidence state."""

    governed_paths = [
        project_dir / "src" / "expanded_study.py",
        project_dir / "src" / "interval_experiments.py",
        project_dir / "src" / "study_core.py",
        project_dir / "src" / "strict_json.py",
        project_dir / "src" / "dafd_reproduction.py",
        project_dir / "EXPERIMENT_PROTOCOL.md",
        project_dir / "PROTOCOL_DEVIATION_REGISTER.md",
        test_evidence_path,
    ] + sorted((project_dir / "tests").glob("test_*.py"))
    missing = [str(path) for path in governed_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"governed manifest inputs are missing: {missing}")
    governed_files: list[dict[str, str]] = []
    for path in governed_paths:
        if path == test_evidence_path:
            # The evidence file lives inside the caller-selected output
            # directory.  Recording that directory name would make two
            # otherwise identical runs hash differently merely because the
            # protocol requires separate run directories.
            recorded_path = "test_evidence.json"
        elif path.is_relative_to(project_dir):
            recorded_path = str(path.relative_to(project_dir))
        else:
            recorded_path = str(path)
        governed_files.append(
            {
                "path": recorded_path,
                "sha256": sha256(path),
            }
        )
    return {
        "schema_version": "expanded-study-reproducibility-manifest-v1",
        "test_evidence_status": "not_executed_by_analysis_script",
        "test_evidence_note": (
            "The analysis entry point does not claim to run its own tests. "
            "The command and an explicit unverified status are hashed below."
        ),
        "governed_files": governed_files,
    }


def source_context_ids(frame: pd.DataFrame) -> np.ndarray:
    columns = [DAFD.REFERENCE] + DEVICE_FLUID_FEATURES
    identifiers = group_ids(frame, columns)
    _, compact = np.unique(identifiers, return_inverse=True)
    return np.array([f"context_{value:02d}" for value in compact])


def dafd_event_types(n_rows: int, dataset: str) -> np.ndarray:
    """Paper-defined single- versus double-emulsion row labels."""

    event = np.full(n_rows, "SE", dtype=object)
    if dataset == "comprehensive" and n_rows == 868:
        event[:197] = "DE"
        event[671:868] = "DE"
    elif dataset == "external" and n_rows == 64:
        event[44:64] = "DE"
    return event


def _partition_composition_row(
    partition: str,
    positions: np.ndarray,
    event_types: np.ndarray,
    physical_clusters: np.ndarray,
    dependency_groups: np.ndarray,
) -> dict[str, object]:
    positions = np.asarray(positions, dtype=int)
    de = event_types[positions] == "DE"
    return {
        "partition": partition,
        "n_rows": int(len(positions)),
        "n_se_rows": int((~de).sum()),
        "n_de_rows": int(de.sum()),
        "n_dependency_groups": int(
            len(np.unique(dependency_groups[positions]))
        ),
        "n_de_physical_experiments": int(
            len(np.unique(physical_clusters[positions][de])) if de.any() else 0
        ),
    }


def data_audit(
    comprehensive: pd.DataFrame,
    external: pd.DataFrame,
    data_paths: Iterable[Path],
) -> dict[str, object]:
    exact = group_ids(comprehensive, DAFD.FEATURES)
    geometry = group_ids(comprehensive, GEOMETRY_FEATURES)
    device_fluid = group_ids(comprehensive, DEVICE_FLUID_FEATURES)
    physical = comprehensive_experiment_clusters(comprehensive)
    external_physical = external_experiment_clusters(external)
    exact_sizes = pd.Series(exact).value_counts()
    exact_target_variation = (
        comprehensive.assign(_exact=exact)
        .groupby("_exact")[DAFD.TARGET_UM]
        .agg(["size", "nunique"])
    )
    contexts = source_context_ids(external)
    return {
        "n_comprehensive_rows": int(len(comprehensive)),
        "n_external_rows": int(len(external)),
        "n_exact_input_groups": int(pd.Series(exact).nunique()),
        "n_geometry_groups": int(pd.Series(geometry).nunique()),
        "n_device_fluid_groups": int(pd.Series(device_fluid).nunique()),
        "n_physical_experiment_clusters": int(pd.Series(physical).nunique()),
        "n_external_physical_clusters": int(
            pd.Series(external_physical).nunique()
        ),
        "n_external_sources": int(external[DAFD.REFERENCE].nunique()),
        "n_external_source_contexts": int(pd.Series(contexts).nunique()),
        "n_rows_in_repeated_exact_inputs": int(exact_sizes[exact_sizes > 1].sum()),
        "n_repeated_exact_input_groups": int((exact_sizes > 1).sum()),
        "n_repeated_input_groups_with_varying_target": int(
            (
                (exact_target_variation["size"] > 1)
                & (exact_target_variation["nunique"] > 1)
            ).sum()
        ),
        "n_missing_required_comprehensive": int(
            comprehensive[DAFD.FEATURES + [DAFD.TARGET_NORMALIZED]].isna().sum().sum()
        ),
        "n_missing_required_external": int(
            external[DAFD.FEATURES + [DAFD.TARGET_NORMALIZED]].isna().sum().sum()
        ),
        "external_source_rows": {
            str(key): int(value)
            for key, value in external[DAFD.REFERENCE].value_counts().items()
        },
        "data_sha256": {path.name: sha256(path) for path in data_paths},
    }


@dataclass(frozen=True)
class SplitProtocol:
    name: str
    repeats: int
    group_columns: tuple[str, ...] | None
    dependency_aware: bool


def split_protocols(fast: bool = False) -> list[SplitProtocol]:
    multiplier = 1 if not fast else 0
    return [
        SplitProtocol("row_random_reproduction", 15 if multiplier else 2, None, False),
        SplitProtocol(
            "exact_input_dependency_group",
            15 if multiplier else 2,
            tuple(DAFD.FEATURES),
            True,
        ),
        SplitProtocol(
            "geometry_dependency_group",
            30 if multiplier else 3,
            tuple(GEOMETRY_FEATURES),
            True,
        ),
        SplitProtocol(
            "device_fluid_dependency_group",
            30 if multiplier else 3,
            tuple(DEVICE_FLUID_FEATURES),
            True,
        ),
    ]


def _paired_experiment_leakage(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    experiment_clusters: np.ndarray,
) -> float:
    train_clusters = set(experiment_clusters[train_indices].tolist())
    test_de = [index for index in test_indices if experiment_clusters[index] < 197]
    if not test_de:
        return 0.0
    leaked = sum(experiment_clusters[index] in train_clusters for index in test_de)
    return float(leaked / len(test_de))


def _overlap_fraction(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    identifiers: np.ndarray,
) -> float:
    train_groups = set(identifiers[train_indices].tolist())
    return float(
        np.mean([identifier in train_groups for identifier in identifiers[test_indices]])
    )


def make_split(
    frame: pd.DataFrame,
    protocol: SplitProtocol,
    seed: int,
    experiment_clusters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    indices = np.arange(len(frame))
    if protocol.group_columns is None:
        train_indices, test_indices = train_test_split(
            indices, test_size=0.2, random_state=seed, shuffle=True
        )
        return train_indices, test_indices, None
    if protocol.dependency_aware:
        groups = dependency_group_ids(
            frame, protocol.group_columns, experiment_clusters
        )
    else:
        groups = group_ids(frame, protocol.group_columns)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_indices, test_indices = next(splitter.split(frame, groups=groups))
    if set(groups[train_indices]).intersection(groups[test_indices]):
        raise RuntimeError(f"Group leakage in {protocol.name}, seed {seed}")
    return train_indices, test_indices, groups


def evaluate_split_protocols(
    comprehensive: pd.DataFrame,
    protocols: list[SplitProtocol],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []
    physical = comprehensive_experiment_clusters(comprehensive)
    exact = group_ids(comprehensive, DAFD.FEATURES)
    geometry = group_ids(comprehensive, GEOMETRY_FEATURES)

    for protocol in protocols:
        for seed in range(protocol.repeats):
            train_indices, test_indices, groups = make_split(
                comprehensive, protocol, seed, physical
            )
            split_id = f"{protocol.name}_seed_{seed:02d}"
            leakage = {
                "exact_input_overlap_fraction": _overlap_fraction(
                    train_indices, test_indices, exact
                ),
                "geometry_overlap_fraction": _overlap_fraction(
                    train_indices, test_indices, geometry
                ),
                "de_pair_leakage_fraction": _paired_experiment_leakage(
                    train_indices, test_indices, physical
                ),
            }
            for index in train_indices:
                manifest_rows.append(
                    {
                        "split_id": split_id,
                        "row_index": int(index),
                        "partition": "train",
                        "physical_cluster": int(physical[index]),
                        "dependency_group": (
                            int(groups[index]) if groups is not None else None
                        ),
                    }
                )
            for index in test_indices:
                manifest_rows.append(
                    {
                        "split_id": split_id,
                        "row_index": int(index),
                        "partition": "test",
                        "physical_cluster": int(physical[index]),
                        "dependency_group": (
                            int(groups[index]) if groups is not None else None
                        ),
                    }
                )

            for model_name in MODEL_NAMES:
                model = build_regressor(model_name, seed)
                model.fit(
                    comprehensive.iloc[train_indices][DAFD.FEATURES],
                    comprehensive.iloc[train_indices][DAFD.TARGET_NORMALIZED],
                )
                predicted = model.predict(
                    comprehensive.iloc[test_indices][DAFD.FEATURES]
                )
                per_row = prediction_frame(
                    comprehensive.iloc[test_indices], predicted
                )
                summary = summarize_prediction_rows(per_row)
                metrics_rows.append(
                    {
                        "split_id": split_id,
                        "protocol": protocol.name,
                        "seed": seed,
                        "model": model_name,
                        "n_train": int(len(train_indices)),
                        "n_test": int(len(test_indices)),
                        **leakage,
                        **summary,
                    }
                )
                per_row.insert(0, "model", model_name)
                per_row.insert(0, "seed", seed)
                per_row.insert(0, "protocol", protocol.name)
                per_row.insert(0, "split_id", split_id)
                prediction_rows.append(per_row)
    return (
        pd.DataFrame(metrics_rows),
        pd.concat(prediction_rows, ignore_index=True),
        pd.DataFrame(manifest_rows),
    )


def evaluate_external_models(
    comprehensive: pd.DataFrame, external: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, object]] = []
    rows: list[pd.DataFrame] = []
    for model_name in MODEL_NAMES:
        model = build_regressor(model_name, 2024)
        model.fit(
            comprehensive[DAFD.FEATURES], comprehensive[DAFD.TARGET_NORMALIZED]
        )
        per_row = prediction_frame(
            external, model.predict(external[DAFD.FEATURES])
        )
        per_row.insert(0, "model", model_name)
        rows.append(per_row)
        summaries.append(
            {
                "model": model_name,
                "scope": "row_micro",
                "source": "ALL",
                **summarize_prediction_rows(per_row),
            }
        )
        per_source: list[dict[str, object]] = []
        for source, group in per_row.groupby("source"):
            source_summary = {
                "model": model_name,
                "scope": "source",
                "source": source,
                **summarize_prediction_rows(group),
            }
            summaries.append(source_summary)
            per_source.append(source_summary)
        metric_names = [
            "mean_absolute_normalized_error",
            "mean_absolute_relative_error",
            "mape_percent",
            "mae_um",
            "median_absolute_error_um",
            "rmse_um",
            "r2",
        ]
        summaries.append(
            {
                "model": model_name,
                "scope": "source_macro",
                "source": "ALL_EQUAL_WEIGHT",
                "n_rows": int(len(per_row)),
                "n_sources": int(len(per_source)),
                **{
                    metric: float(np.nanmean([row[metric] for row in per_source]))
                    for metric in metric_names
                },
            }
        )
    return pd.DataFrame(summaries), pd.concat(rows, ignore_index=True)


def shuffled_label_control(
    comprehensive: pd.DataFrame,
    external: pd.DataFrame,
    fast: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the focal model to permuted training outcomes as a negative control."""

    repeats = 3 if fast else 30
    summaries: list[dict[str, object]] = []
    rows: list[pd.DataFrame] = []
    original_target = comprehensive[DAFD.TARGET_NORMALIZED].to_numpy(dtype=float)
    for repeat in range(repeats):
        seed = 7200 + repeat
        shuffled = np.random.default_rng(seed).permutation(original_target)
        model = DAFD.build_model(seed)
        model.fit(comprehensive[DAFD.FEATURES], shuffled)
        per_row = prediction_frame(
            external,
            model.predict(external[DAFD.FEATURES]),
        )
        per_row.insert(0, "seed", seed)
        rows.append(per_row)
        summaries.append(
            {
                "seed": seed,
                "control": "permuted_training_targets",
                **summarize_prediction_rows(per_row),
            }
        )
    return pd.DataFrame(summaries), pd.concat(rows, ignore_index=True)


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _binary_screen_summary(
    scored: pd.DataFrame,
    accepted: np.ndarray,
    method: str,
    quantile: float | str,
    threshold: float,
) -> dict[str, object]:
    accepted = np.asarray(accepted, dtype=bool)
    retained = scored.loc[accepted]
    result: dict[str, object] = {
        "method": method,
        "training_quantile": quantile,
        "threshold": threshold,
        "n_candidates": int(len(scored)),
        "n_retained": int(accepted.sum()),
        "retention_rate": float(accepted.mean()),
        "candidate_reduction_factor": _safe_rate(len(scored), accepted.sum()),
        "mean_absolute_relative_error": (
            float(retained["absolute_relative_error"].mean())
            if len(retained)
            else float("nan")
        ),
        "mean_absolute_normalized_error": (
            float(retained["absolute_normalized_error"].mean())
            if len(retained)
            else float("nan")
        ),
        "mae_um": (
            float(retained["absolute_error_um"].mean())
            if len(retained)
            else float("nan")
        ),
    }
    relative_error = scored["absolute_relative_error"].to_numpy(dtype=float)
    absolute_error = scored["absolute_error_um"].to_numpy(dtype=float)
    for tolerance in RELATIVE_TOLERANCES:
        reliable = relative_error <= tolerance
        label = f"relative_{int(round(tolerance * 100))}pct"
        result[f"{label}_precision"] = (
            float(reliable[accepted].mean()) if accepted.any() else float("nan")
        )
        result[f"{label}_recall"] = _safe_rate(
            int((reliable & accepted).sum()), int(reliable.sum())
        )
    for tolerance in ABSOLUTE_TOLERANCES_UM:
        reliable = absolute_error <= tolerance
        label = f"absolute_{int(round(tolerance))}um"
        result[f"{label}_precision"] = (
            float(reliable[accepted].mean()) if accepted.any() else float("nan")
        )
        result[f"{label}_recall"] = _safe_rate(
            int((reliable & accepted).sum()), int(reliable.sum())
        )
    return result


def _risk_curves_for_scope(
    scored: pd.DataFrame,
    score: np.ndarray,
    method: str,
    scope: str,
    scope_id: str,
) -> tuple[list[pd.DataFrame], list[dict[str, object]]]:
    curve_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    outcomes = {
        "relative": "absolute_relative_error",
        "normalized": "absolute_normalized_error",
        "absolute_um": "absolute_error_um",
    }
    for outcome, column in outcomes.items():
        curve, aurc = risk_curve(
            scored[column].to_numpy(dtype=float), score, minimum_count=min(5, len(scored))
        )
        curve.insert(0, "outcome", outcome)
        curve.insert(0, "scope_id", scope_id)
        curve.insert(0, "scope", scope)
        curve.insert(0, "method", method)
        curve_frames.append(curve)
        summaries.append(
            {
                "method": method,
                "scope": scope,
                "scope_id": scope_id,
                "outcome": outcome,
                "n": int(len(scored)),
                "aurc": aurc,
                "full_coverage_risk": float(scored[column].mean()),
            }
        )
    return curve_frames, summaries


def _matched_random_control(
    scored: pd.DataFrame,
    accepted: np.ndarray,
    observed_relative_risk: float,
    seed: int,
    permutations: int,
    preserve_source_counts: bool,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    accepted = np.asarray(accepted, dtype=bool)
    errors = scored["absolute_relative_error"].to_numpy(dtype=float)
    sources = scored["source"].to_numpy()
    retained_count = int(accepted.sum())
    if retained_count == 0:
        return {
            "random_mean_risk": float("nan"),
            "random_ci_low": float("nan"),
            "random_ci_high": float("nan"),
            "one_sided_fraction_as_good_or_better": float("nan"),
            "permutations": 0,
        }
    if retained_count == len(scored):
        return {
            "random_mean_risk": observed_relative_risk,
            "random_ci_low": observed_relative_risk,
            "random_ci_high": observed_relative_risk,
            "one_sided_fraction_as_good_or_better": 1.0,
            "permutations": permutations,
        }
    risks: list[float] = []
    for _ in range(permutations):
        chosen: list[int] = []
        if preserve_source_counts:
            for source in np.unique(sources):
                indices = np.flatnonzero(sources == source)
                budget = int(accepted[indices].sum())
                if budget:
                    chosen.extend(rng.choice(indices, size=budget, replace=False).tolist())
        else:
            budget = retained_count
            if budget:
                chosen = rng.choice(len(scored), size=budget, replace=False).tolist()
        risks.append(float(np.mean(errors[chosen])) if chosen else float("nan"))
    finite = np.asarray([risk for risk in risks if np.isfinite(risk)], dtype=float)
    if not len(finite):
        return {
            "random_mean_risk": float("nan"),
            "random_ci_low": float("nan"),
            "random_ci_high": float("nan"),
            "one_sided_fraction_as_good_or_better": float("nan"),
            "permutations": permutations,
        }
    return {
        "random_mean_risk": float(finite.mean()),
        "random_ci_low": float(np.quantile(finite, 0.025)),
        "random_ci_high": float(np.quantile(finite, 0.975)),
        "one_sided_fraction_as_good_or_better": float(
            (
                1
                + np.sum(
                    finite
                    <= observed_relative_risk
                    + 1e-12 * max(1.0, abs(observed_relative_risk))
                )
            )
            / (len(finite) + 1)
        ),
        "permutations": int(len(finite)),
    }


def _cluster_blocks(
    scored: pd.DataFrame,
    include_context: bool,
) -> list[list[np.ndarray]]:
    """Return intact physical-cluster strata for the requested null.

    Physical experiments are formed within source *before* any context
    conditioning.  This is essential for the paired DE rows whose inner and
    outer events can have different source-context labels.  The context-aware
    null uses the sorted multiset of member contexts as a block signature; it
    never divides a physical experiment into separate rows.
    """

    sources = scored["source"].astype(str).to_numpy()
    clusters = scored["physical_cluster"].to_numpy()
    contexts = (
        scored["source_context"].astype(str).to_numpy()
        if include_context and "source_context" in scored.columns
        else None
    )
    by_stratum: dict[tuple[object, ...], list[np.ndarray]] = {}
    for source in sorted(np.unique(sources)):
        source_positions = np.flatnonzero(sources == source)
        for cluster in np.unique(clusters[source_positions]):
            block = source_positions[clusters[source_positions] == cluster]
            signature: tuple[object, ...] = (source, len(block))
            if contexts is not None:
                # Align source and destination vectors by their member-context
                # labels, not by incidental row order inside the table.
                block = block[
                    np.argsort(contexts[block], kind="stable")
                ]
                signature = signature + (
                    tuple(contexts[block].tolist()),
                )
            by_stratum.setdefault(signature, []).append(block.astype(int))
    return list(by_stratum.values())


def _permute_cluster_blocks(
    values: np.ndarray,
    strata: list[list[np.ndarray]],
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(values)
    permuted = np.empty_like(values)
    for blocks in strata:
        order = rng.permutation(len(blocks))
        for destination, source_position in zip(blocks, order):
            permuted[destination] = values[blocks[int(source_position)]]
    return permuted


def _cluster_permutation_binary_control(
    scored: pd.DataFrame,
    accepted: np.ndarray,
    observed_relative_risk: float,
    seed: int,
    permutations: int,
    include_context: bool,
) -> dict[str, float | int]:
    accepted = np.asarray(accepted, dtype=bool)
    retained_count = int(accepted.sum())
    if retained_count == 0:
        return {
            "random_mean_risk": float("nan"),
            "random_ci_low": float("nan"),
            "random_ci_high": float("nan"),
            "one_sided_fraction_as_good_or_better": float("nan"),
            "permutations": 0,
        }
    if retained_count == len(scored):
        return {
            "random_mean_risk": observed_relative_risk,
            "random_ci_low": observed_relative_risk,
            "random_ci_high": observed_relative_risk,
            "one_sided_fraction_as_good_or_better": 1.0,
            "permutations": permutations,
        }
    errors = scored["absolute_relative_error"].to_numpy(dtype=float)
    strata = _cluster_blocks(scored, include_context=include_context)
    rng = np.random.default_rng(seed)
    risks = np.empty(permutations, dtype=float)
    for repeat in range(permutations):
        permuted = _permute_cluster_blocks(accepted, strata, rng).astype(bool)
        risks[repeat] = float(np.mean(errors[permuted]))
    tolerance = 1e-12 * max(1.0, abs(observed_relative_risk))
    return {
        "random_mean_risk": float(risks.mean()),
        "random_ci_low": float(np.quantile(risks, 0.025)),
        "random_ci_high": float(np.quantile(risks, 0.975)),
        "one_sided_fraction_as_good_or_better": float(
            (1 + np.sum(risks <= observed_relative_risk + tolerance))
            / (permutations + 1)
        ),
        "permutations": permutations,
    }


def _cluster_permutation_aurc_control(
    scored: pd.DataFrame,
    scores: np.ndarray,
    error_column: str,
    observed_aurc: float,
    seed: int,
    permutations: int,
    include_context: bool,
) -> dict[str, float | int]:
    errors = scored[error_column].to_numpy(dtype=float)
    strata = _cluster_blocks(scored, include_context=include_context)
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=float)
    for repeat in range(permutations):
        permuted = _permute_cluster_blocks(scores, strata, rng)
        _, null[repeat] = risk_curve(errors, permuted)
    tolerance = 1e-12 * max(1.0, abs(observed_aurc))
    return {
        "random_mean_risk": float(null.mean()),
        "random_ci_low": float(np.quantile(null, 0.025)),
        "random_ci_high": float(np.quantile(null, 0.975)),
        "one_sided_fraction_as_good_or_better": float(
            (1 + np.sum(null <= observed_aurc + tolerance)) / (permutations + 1)
        ),
        "permutations": permutations,
    }


def _benjamini_hochberg(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    finite = values[np.isfinite(values.to_numpy(dtype=float))]
    if not len(finite):
        return result
    ordered = finite.sort_values()
    count = len(ordered)
    adjusted = ordered.to_numpy(dtype=float) * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[ordered.index] = np.minimum(adjusted, 1.0)
    return result


def hierarchical_bootstrap_external(
    scored: pd.DataFrame,
    fast: bool,
) -> pd.DataFrame:
    """Descriptive source-then-physical-cluster bootstrap intervals."""

    repetitions = 500 if fast else 5000
    rng = np.random.default_rng(8042)
    sources = np.asarray(sorted(scored["source"].unique()), dtype=object)
    source_clusters = {
        source: [
            np.asarray(list(indices), dtype=int)
            for _, indices in scored[scored["source"] == source]
            .groupby("physical_cluster")
            .groups.items()
        ]
        for source in sources
    }
    bootstrap_indices: list[np.ndarray] = []
    for _ in range(repetitions):
        selected: list[int] = []
        for source in rng.choice(sources, size=len(sources), replace=True):
            clusters = source_clusters[source]
            for position in rng.integers(0, len(clusters), size=len(clusters)):
                selected.extend(clusters[int(position)].tolist())
        bootstrap_indices.append(np.asarray(selected, dtype=int))

    rows: list[dict[str, object]] = []
    for outcome, column in (
        ("relative", "absolute_relative_error"),
        ("normalized", "absolute_normalized_error"),
        ("absolute_um", "absolute_error_um"),
    ):
        values = scored[column].to_numpy(dtype=float)
        estimates = np.asarray(
            [float(np.mean(values[indices])) for indices in bootstrap_indices]
        )
        rows.append(
            {
                "method": "full_pool",
                "outcome": outcome,
                "estimand": "mean_risk",
                "estimate": float(values.mean()),
                "ci_low": float(np.quantile(estimates, 0.025)),
                "ci_high": float(np.quantile(estimates, 0.975)),
                "repetitions": repetitions,
                "repetitions_requested": repetitions,
                "repetitions_valid": repetitions,
                "valid_repetition_fraction": 1.0,
                "conditional_on_nonempty_retention": False,
                "resampling_unit": "source_then_physical_experiment_cluster",
            }
        )

    mask_columns = {"strict_minmax": "accepted__strict_minmax"}
    for column in scored.columns:
        if column.startswith("accepted__") and column.endswith("__q0_95"):
            mask_columns[column.removeprefix("accepted__")] = column
    relative = scored["absolute_relative_error"].to_numpy(dtype=float)
    full_risk = float(relative.mean())
    for method, column in mask_columns.items():
        mask = scored[column].to_numpy(dtype=bool)
        if not mask.any():
            continue
        point = float(relative[mask].mean() - full_risk)
        differences: list[float] = []
        for indices in bootstrap_indices:
            sampled_mask = mask[indices]
            if sampled_mask.any():
                differences.append(
                    float(
                        relative[indices][sampled_mask].mean()
                        - relative[indices].mean()
                    )
                )
        estimates = np.asarray(differences, dtype=float)
        ci_low = (
            float(np.quantile(estimates, 0.025))
            if len(estimates)
            else float("nan")
        )
        ci_high = (
            float(np.quantile(estimates, 0.975))
            if len(estimates)
            else float("nan")
        )
        rows.append(
            {
                "method": method,
                "outcome": "relative",
                "estimand": (
                    "conditional_retained_minus_full_mean_risk_"
                    "given_nonempty_retention"
                ),
                "estimate": point,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "repetitions": int(len(estimates)),
                "repetitions_requested": repetitions,
                "repetitions_valid": int(len(estimates)),
                "valid_repetition_fraction": float(
                    len(estimates) / repetitions
                ),
                "conditional_on_nonempty_retention": True,
                "point_retained_rows": int(mask.sum()),
                "resampling_unit": "source_then_physical_experiment_cluster",
            }
        )
    return pd.DataFrame(rows)


def external_applicability_analysis(
    comprehensive: pd.DataFrame,
    external: pd.DataFrame,
    external_xgb_rows: pd.DataFrame,
    fast: bool,
) -> dict[str, pd.DataFrame]:
    scored = external_xgb_rows.copy().reset_index(drop=True)
    scored["source_context"] = source_context_ids(external)
    scored["physical_cluster"] = external_experiment_clusters(external)
    scored["event_type"] = dafd_event_types(len(external), "external")
    scored["observed_diameter_um"] = external[DAFD.TARGET_UM].to_numpy()
    scored["hydraulic_diameter_um"] = external[
        DAFD.HYDRAULIC_DIAMETER
    ].to_numpy()
    bundle = fit_risk_scores(
        comprehensive,
        external,
        seed=2024,
        group_columns=GEOMETRY_FEATURES,
        experiment_clusters=comprehensive_experiment_clusters(comprehensive),
        include_expensive=True,
    )
    for method, values in bundle.candidate.items():
        scored[f"risk__{method}"] = values
    scored["accepted__strict_minmax"] = bundle.strict_minmax_candidate

    metadata = bundle.metadata or {}
    fit_positions = np.asarray(
        metadata.get("fit_row_positions", []), dtype=int
    )
    calibration_positions = np.asarray(
        metadata.get("calibration_row_positions", []), dtype=int
    )
    comprehensive_physical = comprehensive_experiment_clusters(comprehensive)
    comprehensive_dependency = dependency_group_ids(
        comprehensive,
        GEOMETRY_FEATURES,
        comprehensive_physical,
    )
    comprehensive_event = dafd_event_types(len(comprehensive), "comprehensive")
    external_dependency = dependency_group_ids(
        external,
        GEOMETRY_FEATURES,
        external_experiment_clusters(external),
    )
    external_event = dafd_event_types(len(external), "external")
    partition_composition = pd.DataFrame(
        [
            _partition_composition_row(
                "proper_fit",
                fit_positions,
                comprehensive_event,
                comprehensive_physical,
                comprehensive_dependency,
            ),
            _partition_composition_row(
                "threshold_calibration",
                calibration_positions,
                comprehensive_event,
                comprehensive_physical,
                comprehensive_dependency,
            ),
            _partition_composition_row(
                "external_candidate",
                np.arange(len(external), dtype=int),
                external_event,
                external_experiment_clusters(external),
                external_dependency,
            ),
        ]
    )
    partition_composition["threshold_scope_note"] = (
        "If threshold_calibration contains no DE rows, applicability thresholds "
        "are SE-calibrated and cannot establish general DE applicability."
    )
    calibration_score_tables: list[pd.DataFrame] = []
    for method, values in bundle.training.items():
        if len(values) != len(calibration_positions):
            raise RuntimeError(
                f"applicability calibration audit length mismatch: {method}"
            )
        calibration_score_tables.append(
            pd.DataFrame(
                {
                    "method": method,
                    "calibration_row_order": np.arange(
                        len(calibration_positions), dtype=int
                    ),
                    "development_row_position": calibration_positions,
                    "development_row_index": comprehensive.index.to_numpy()[
                        calibration_positions
                    ],
                    "dependency_group": comprehensive_dependency[
                        calibration_positions
                    ],
                    "physical_experiment_cluster": comprehensive_physical[
                        calibration_positions
                    ],
                    "event_type": comprehensive_event[calibration_positions],
                    "calibration_score": np.asarray(values, dtype=float),
                }
            )
        )
    calibration_scores = pd.concat(calibration_score_tables, ignore_index=True)

    threshold_rows: list[dict[str, object]] = []
    binary_rows: list[dict[str, object]] = []
    source_binary_rows: list[dict[str, object]] = []
    risk_frames: list[pd.DataFrame] = []
    aurc_rows: list[dict[str, object]] = []
    correlation_rows: list[dict[str, object]] = []
    random_rows: list[dict[str, object]] = []
    permutations = 250 if fast else 3000

    strict_summary = _binary_screen_summary(
        scored,
        bundle.strict_minmax_candidate,
        "strict_minmax",
        "fixed",
        1.0,
    )
    binary_rows.append(strict_summary)
    scored["accepted__strict_minmax"] = bundle.strict_minmax_candidate
    strict_observed_risk = float(
        scored.loc[
            bundle.strict_minmax_candidate, "absolute_relative_error"
        ].mean()
    ) if bundle.strict_minmax_candidate.any() else float("nan")
    for preserve in (False, True):
        random_rows.append(
            {
                "method": "strict_minmax",
                "training_quantile": "fixed",
                "control_kind": (
                    "row_budget_source_stratified"
                    if preserve
                    else "row_budget_global"
                ),
                "statistic": "retained_relative_risk",
                "observed_value": strict_observed_risk,
                "preserve_source_counts": preserve,
                "n_retained": int(bundle.strict_minmax_candidate.sum()),
                "observed_relative_risk": strict_observed_risk,
                **_matched_random_control(
                    scored,
                    bundle.strict_minmax_candidate,
                    strict_observed_risk,
                    seed=8500 + int(preserve),
                    permutations=permutations,
                    preserve_source_counts=preserve,
                ),
            }
        )
    for include_context in (False, True):
        random_rows.append(
            {
                "method": "strict_minmax",
                "training_quantile": "fixed",
                "control_kind": (
                    "within_source_context_physical_cluster_mask_permutation"
                    if include_context
                    else "within_source_physical_cluster_mask_permutation"
                ),
                "statistic": "retained_relative_risk",
                "observed_value": strict_observed_risk,
                "preserve_source_counts": True,
                "n_retained": int(bundle.strict_minmax_candidate.sum()),
                "observed_relative_risk": strict_observed_risk,
                **_cluster_permutation_binary_control(
                    scored,
                    bundle.strict_minmax_candidate,
                    strict_observed_risk,
                    seed=8600 + int(include_context),
                    permutations=permutations,
                    include_context=include_context,
                ),
            }
        )

    for method, candidate_scores in bundle.candidate.items():
        training_scores = bundle.training[method]
        scored[f"risk__{method}"] = candidate_scores
        overall_curves, overall_summaries = _risk_curves_for_scope(
            scored, candidate_scores, method, "row_micro", "ALL"
        )
        risk_frames.extend(overall_curves)
        aurc_rows.extend(overall_summaries)
        outcome_columns = {
            "relative": "absolute_relative_error",
            "normalized": "absolute_normalized_error",
            "absolute_um": "absolute_error_um",
        }
        for offset, overall_summary in enumerate(overall_summaries):
            outcome = str(overall_summary["outcome"])
            observed_aurc = float(overall_summary["aurc"])
            for include_context in (False, True):
                random_rows.append(
                    {
                        "method": method,
                        "training_quantile": float("nan"),
                        "control_kind": (
                            "within_source_context_physical_cluster_score_permutation"
                            if include_context
                            else "within_source_physical_cluster_score_permutation"
                        ),
                        "statistic": f"{outcome}_aurc",
                        "observed_value": observed_aurc,
                        "n_retained": int(len(scored)),
                        "observed_relative_risk": float("nan"),
                        "preserve_source_counts": True,
                        **_cluster_permutation_aurc_control(
                            scored,
                            candidate_scores,
                            outcome_columns[outcome],
                            observed_aurc,
                            seed=(
                                9100
                                + 97 * offset
                                + len(random_rows)
                                + int(include_context)
                            ),
                            permutations=permutations,
                            include_context=include_context,
                        ),
                    }
                )
        for source, indices in scored.groupby("source").groups.items():
            subset = scored.loc[indices].reset_index(drop=True)
            sub_scores = candidate_scores[np.asarray(list(indices), dtype=int)]
            curves, summaries = _risk_curves_for_scope(
                subset, sub_scores, method, "source", str(source)
            )
            risk_frames.extend(curves)
            aurc_rows.extend(summaries)
        for context, indices in scored.groupby("source_context").groups.items():
            subset = scored.loc[indices].reset_index(drop=True)
            sub_scores = candidate_scores[np.asarray(list(indices), dtype=int)]
            curves, summaries = _risk_curves_for_scope(
                subset, sub_scores, method, "source_context", str(context)
            )
            risk_frames.extend(curves)
            aurc_rows.extend(summaries)

        for scope, indices in [("ALL", np.arange(len(scored)))] + [
            (str(source), np.asarray(list(group_indices), dtype=int))
            for source, group_indices in scored.groupby("source").groups.items()
        ]:
            for outcome in (
                "absolute_relative_error",
                "absolute_normalized_error",
                "absolute_error_um",
                "observed_diameter_um",
                "hydraulic_diameter_um",
            ):
                statistic = spearmanr(
                    candidate_scores[indices],
                    scored.iloc[indices][outcome].to_numpy(dtype=float),
                )
                correlation_rows.append(
                    {
                        "method": method,
                        "scope": scope,
                        "outcome": outcome,
                        "n": int(len(indices)),
                        "spearman_rho": float(statistic.statistic),
                        "row_naive_p_value": float(statistic.pvalue),
                    }
                )

        for quantile in RISK_QUANTILES:
            threshold = threshold_from_training(training_scores, quantile)
            accepted = candidate_scores <= threshold
            accepted_column = f"accepted__{method}__q{str(quantile).replace('.', '_')}"
            scored[accepted_column] = accepted
            threshold_rows.append(
                {
                    "method": method,
                    "training_quantile": quantile,
                    "threshold": threshold,
                }
            )
            summary = _binary_screen_summary(
                scored, accepted, method, quantile, threshold
            )
            binary_rows.append(summary)
            for source, indices in scored.groupby("source").groups.items():
                index_array = np.asarray(list(indices), dtype=int)
                source_summary = _binary_screen_summary(
                    scored.iloc[index_array].reset_index(drop=True),
                    accepted[index_array],
                    method,
                    quantile,
                    threshold,
                )
                source_summary["source"] = source
                source_binary_rows.append(source_summary)
            observed_risk = float(
                scored.loc[accepted, "absolute_relative_error"].mean()
            ) if accepted.any() else float("nan")
            for preserve in (False, True):
                random_rows.append(
                    {
                        "method": method,
                        "training_quantile": quantile,
                        "control_kind": (
                            "row_budget_source_stratified"
                            if preserve
                            else "row_budget_global"
                        ),
                        "statistic": "retained_relative_risk",
                        "observed_value": observed_risk,
                        "preserve_source_counts": preserve,
                        "n_retained": int(accepted.sum()),
                        "observed_relative_risk": observed_risk,
                        **_matched_random_control(
                            scored,
                            accepted,
                            observed_risk,
                            seed=2024 + int(quantile * 1000) + int(preserve),
                            permutations=permutations,
                            preserve_source_counts=preserve,
                        ),
                    }
                )
            for include_context in (False, True):
                random_rows.append(
                    {
                        "method": method,
                        "training_quantile": quantile,
                        "control_kind": (
                            "within_source_context_physical_cluster_mask_permutation"
                            if include_context
                            else "within_source_physical_cluster_mask_permutation"
                        ),
                        "statistic": "retained_relative_risk",
                        "observed_value": observed_risk,
                        "preserve_source_counts": True,
                        "n_retained": int(accepted.sum()),
                        "observed_relative_risk": observed_risk,
                        **_cluster_permutation_binary_control(
                            scored,
                            accepted,
                            observed_risk,
                            seed=(
                                10100
                                + int(quantile * 1000)
                                + len(random_rows)
                                + int(include_context)
                            ),
                            permutations=permutations,
                            include_context=include_context,
                        ),
                    }
                )

    for source, indices in scored.groupby("source").groups.items():
        index_array = np.asarray(list(indices), dtype=int)
        source_summary = _binary_screen_summary(
            scored.iloc[index_array].reset_index(drop=True),
            bundle.strict_minmax_candidate[index_array],
            "strict_minmax",
            "fixed",
            1.0,
        )
        source_summary["source"] = source
        source_binary_rows.append(source_summary)

    random_controls = pd.DataFrame(random_rows)
    random_controls["bh_adjusted_p_within_family"] = np.nan
    for _, indices in random_controls.groupby(
        ["control_kind", "statistic"], dropna=False
    ).groups.items():
        index_array = list(indices)
        adjusted = _benjamini_hochberg(
            random_controls.loc[
                index_array, "one_sided_fraction_as_good_or_better"
            ]
        )
        random_controls.loc[index_array, "bh_adjusted_p_within_family"] = adjusted

    metadata_record = {
        key: (
            json.dumps(value, sort_keys=True)
            if isinstance(value, (dict, list))
            else value
        )
        for key, value in (bundle.metadata or {}).items()
    }
    metadata_record["threshold_calibration_event_scope"] = (
        "SE_only"
        if int(
            partition_composition.loc[
                partition_composition["partition"] == "threshold_calibration",
                "n_de_rows",
            ].iloc[0]
        )
        == 0
        else "mixed_SE_DE"
    )
    metadata_record["de_applicability_claim_boundary"] = (
        "SE-only threshold calibration cannot support a general DE applicability claim."
    )

    return {
        "scored": scored,
        "calibration_metadata": pd.DataFrame([metadata_record]),
        "calibration_scores": calibration_scores,
        "partition_composition": partition_composition,
        "thresholds": pd.DataFrame(threshold_rows),
        "binary_summary": pd.DataFrame(binary_rows),
        "source_binary_summary": pd.DataFrame(source_binary_rows),
        "risk_curves": pd.concat(risk_frames, ignore_index=True),
        "aurc_summary": pd.DataFrame(aurc_rows),
        "correlations": pd.DataFrame(correlation_rows),
        "random_controls": random_controls,
    }


def internal_group_risk_validation(
    comprehensive: pd.DataFrame, fast: bool
) -> pd.DataFrame:
    physical = comprehensive_experiment_clusters(comprehensive)
    groups = dependency_group_ids(
        comprehensive, GEOMETRY_FEATURES, physical
    )
    repeats = 2 if fast else 10
    rows: list[dict[str, object]] = []
    for seed in range(repeats):
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=3000 + seed)
        train_indices, test_indices = next(
            splitter.split(comprehensive, groups=groups)
        )
        train = comprehensive.iloc[train_indices].reset_index(drop=True)
        test = comprehensive.iloc[test_indices].reset_index(drop=True)
        model = DAFD.build_model(3000 + seed)
        model.fit(train[DAFD.FEATURES], train[DAFD.TARGET_NORMALIZED])
        test_rows = prediction_frame(test, model.predict(test[DAFD.FEATURES]))
        bundle = fit_risk_scores(
            train,
            test,
            seed=3000 + seed,
            group_columns=GEOMETRY_FEATURES,
            experiment_clusters=physical[train_indices],
            include_expensive=True,
        )
        for method, score in bundle.candidate.items():
            for outcome, error_column in (
                ("relative", "absolute_relative_error"),
                ("normalized", "absolute_normalized_error"),
                ("absolute_um", "absolute_error_um"),
            ):
                _, aurc = risk_curve(
                    test_rows[error_column].to_numpy(dtype=float), score
                )
                rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "outcome": outcome,
                        "n_train": int(len(train)),
                        "n_test": int(len(test)),
                        "n_train_dependency_groups": int(
                            pd.Series(groups[train_indices]).nunique()
                        ),
                        "n_test_dependency_groups": int(
                            pd.Series(groups[test_indices]).nunique()
                        ),
                        "metric_kind": "continuous_selective_aurc",
                        "aurc": aurc,
                        "conditional_risk": float("nan"),
                        "full_risk": float(test_rows[error_column].mean()),
                    }
                )
        accepted = bundle.strict_minmax_candidate
        for outcome, error_column in (
            ("relative", "absolute_relative_error"),
            ("normalized", "absolute_normalized_error"),
            ("absolute_um", "absolute_error_um"),
        ):
            rows.append(
                {
                    "seed": seed,
                    "method": "strict_minmax_binary",
                    "outcome": outcome,
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "n_train_dependency_groups": int(
                        pd.Series(groups[train_indices]).nunique()
                    ),
                    "n_test_dependency_groups": int(
                        pd.Series(groups[test_indices]).nunique()
                    ),
                    "metric_kind": "binary_conditional_risk",
                    "aurc": float("nan"),
                    "conditional_risk": float(
                        test_rows.loc[accepted, error_column].mean()
                    ) if accepted.any() else float("nan"),
                    "full_risk": float(test_rows[error_column].mean()),
                    "retention_rate": float(accepted.mean()),
                }
            )
    return pd.DataFrame(rows)


def build_replay_masks(scored: pd.DataFrame) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {
        "point_only": np.ones(len(scored), dtype=bool),
        "strict_minmax": scored["accepted__strict_minmax"].to_numpy(dtype=bool),
    }
    for column in scored.columns:
        if column.startswith("accepted__") and column != "accepted__strict_minmax":
            masks[column.removeprefix("accepted__")] = scored[column].to_numpy(
                dtype=bool
            )
    return masks


def build_replay_selections(
    scored_inputs: pd.DataFrame,
    masks: dict[str, np.ndarray],
    interval_lower_um: np.ndarray | None = None,
    interval_upper_um: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build ranked candidate selections without reading observed outcomes."""

    predicted = scored_inputs["predicted_um"].to_numpy(dtype=float)
    contexts = scored_inputs["source_context"].astype(str).to_numpy()
    candidate_ids = scored_inputs["row_index"].to_numpy(dtype=int)
    rows: list[dict[str, object]] = []
    strategy_masks = dict(masks)
    if interval_lower_um is not None and interval_upper_um is not None:
        lower = np.asarray(interval_lower_um, dtype=float)
        upper = np.asarray(interval_upper_um, dtype=float)
    else:
        lower = upper = None

    for context in np.unique(contexts):
        context_indices = np.flatnonzero(contexts == context)
        for target_um in TARGET_GRID_UM:
            for tolerance_kind, tolerance_um in (
                ("relative_10pct", 0.10 * target_um),
                ("absolute_10um", 10.0),
            ):
                target_masks = dict(strategy_masks)
                if lower is not None and upper is not None:
                    interval_contains = (lower <= target_um) & (upper >= target_um)
                    target_masks["conformal_target_inclusion"] = interval_contains
                    for base in (
                        "strict_minmax",
                        "knn_5__q0_95",
                        "mahalanobis_oas__q0_95",
                        "isolation_forest__q0_95",
                    ):
                        if base in masks:
                            target_masks[f"{base}+conformal"] = (
                                masks[base] & interval_contains
                            )
                for strategy, global_mask in target_masks.items():
                    allowed = np.zeros(len(scored_inputs), dtype=bool)
                    allowed[context_indices] = global_mask[context_indices]
                    for top_k in TOP_K_VALUES:
                        selected_positions = rank_candidates(
                            predicted,
                            target_um,
                            allowed,
                            tolerance_um,
                            top_k,
                        )
                        rows.append(
                            {
                                "context": context,
                                "target_um": target_um,
                                "tolerance_kind": tolerance_kind,
                                "tolerance_um": tolerance_um,
                                "strategy": strategy,
                                "top_k": top_k,
                                "context_candidate_count": int(len(context_indices)),
                                "allowed_candidate_count": int(
                                    global_mask[context_indices].sum()
                                ),
                                "eligible_candidate_count": int(
                                    np.sum(
                                        allowed
                                        & (np.abs(predicted - target_um) <= tolerance_um)
                                    )
                                ),
                                "answered": bool(len(selected_positions)),
                                "selected_row_ids": ";".join(
                                    str(value)
                                    for value in candidate_ids[selected_positions]
                                ),
                            }
                        )
    return pd.DataFrame(rows)


def score_replay_selections(
    selections: pd.DataFrame,
    scored_outcomes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reveal historical outcomes only after candidate selections are fixed."""

    actual_by_id = dict(
        zip(
            scored_outcomes["row_index"].astype(int),
            scored_outcomes["actual_um"].astype(float),
        )
    )
    contexts_by_id = dict(
        zip(
            scored_outcomes["row_index"].astype(int),
            scored_outcomes["source_context"].astype(str),
        )
    )
    context_actual: dict[str, np.ndarray] = {}
    for context in scored_outcomes["source_context"].astype(str).unique():
        ids = [
            row_id for row_id, value in contexts_by_id.items() if value == context
        ]
        context_actual[context] = np.asarray(
            [actual_by_id[row_id] for row_id in ids], dtype=float
        )

    scored_rows: list[dict[str, object]] = []
    for record in selections.to_dict(orient="records"):
        target = float(record["target_um"])
        tolerance = float(record["tolerance_um"])
        oracle_error = float(
            np.min(np.abs(context_actual[str(record["context"])] - target))
        )
        selected_ids = [
            int(value)
            for value in str(record["selected_row_ids"]).split(";")
            if value
        ]
        if selected_ids:
            achieved_error = float(
                np.min([abs(actual_by_id[row_id] - target) for row_id in selected_ids])
            )
            success = achieved_error <= tolerance
            regret = achieved_error - oracle_error
        else:
            achieved_error = float("nan")
            success = False
            regret = float("nan")
        scored_rows.append(
            {
                **record,
                "oracle_error_um": oracle_error,
                "oracle_achievable": bool(oracle_error <= tolerance),
                "achieved_error_um": achieved_error,
                "success": bool(success),
                "oracle_regret_um": regret,
            }
        )
    scored = pd.DataFrame(scored_rows)
    summaries: list[dict[str, object]] = []
    group_columns = ["tolerance_kind", "strategy", "top_k"]
    for keys, group in scored.groupby(group_columns, dropna=False):
        tolerance_kind, strategy, top_k = keys
        answered = group["answered"].astype(bool)
        successes = group["success"].astype(bool)
        oracle = group["oracle_achievable"].astype(bool)
        summaries.append(
            {
                "scope": "context_macro_queries",
                "tolerance_kind": tolerance_kind,
                "strategy": strategy,
                "top_k": int(top_k),
                "n_context_target_queries": int(len(group)),
                "n_oracle_achievable": int(oracle.sum()),
                "query_coverage": float(answered.mean()),
                "abstention_rate": float((~answered).mean()),
                "success_precision_answered": (
                    float(successes[answered].mean())
                    if answered.any()
                    else float("nan")
                ),
                "successful_target_recall": _safe_rate(
                    int(successes.sum()), int(oracle.sum())
                ),
                "mean_oracle_regret_um_answered": (
                    float(group.loc[answered, "oracle_regret_um"].mean())
                    if answered.any()
                    else float("nan")
                ),
                "median_allowed_candidates": float(
                    group["allowed_candidate_count"].median()
                ),
                "median_context_candidates": float(
                    group["context_candidate_count"].median()
                ),
            }
        )
    return scored, pd.DataFrame(summaries)


def make_figures(
    split_metrics: pd.DataFrame,
    interval_summary: pd.DataFrame,
    applicability: dict[str, pd.DataFrame],
    replay_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    np.random.seed(2024)
    sns.set_theme(style="whitegrid", context="paper")
    palette = {
        "row_random_reproduction": "#4C78A8",
        "exact_input_dependency_group": "#72B7B2",
        "geometry_dependency_group": "#F58518",
        "device_fluid_dependency_group": "#E45756",
    }

    subset = split_metrics[split_metrics["model"] == "xgboost"].copy()
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    order = list(palette)
    sns.boxplot(
        data=subset,
        x="protocol",
        y="mape_percent",
        order=order,
        palette=palette,
        hue="protocol",
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=subset,
        x="protocol",
        y="mape_percent",
        order=order,
        color="#222222",
        alpha=0.5,
        size=3,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Diameter MAPE (%)")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(
        ["Row random\n(reproduction)", "Exact-input +\nDE pair", "Geometry +\nDE pair", "Geometry/fluid +\nDE pair"]
    )
    fig.tight_layout()
    fig.savefig(output_dir / "figure_1_split_generalization.png", dpi=240)
    plt.close(fig)

    if len(interval_summary):
        interval_plot = interval_summary[
            (interval_summary["nominal"] == 0.90)
            & (interval_summary["source"] == "__overall__")
        ].copy()
        protocols = ["row_random", "geometry_group"]
        fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), sharex="col")
        for row_index, protocol in enumerate(protocols):
            protocol_rows = interval_plot[
                interval_plot["protocol"] == protocol
            ]
            coverage_column = (
                "empirical_coverage"
                if protocol == "row_random"
                else "empirical_simultaneous_group_coverage"
            )
            coverage_label = (
                "Row coverage"
                if protocol == "row_random"
                else "Simultaneous dependency-group coverage"
            )
            sns.pointplot(
                data=protocol_rows,
                x="method",
                y=coverage_column,
                hue="dataset",
                errorbar="sd",
                dodge=0.25,
                ax=axes[row_index, 0],
            )
            axes[row_index, 0].axhline(
                0.90, color="#222222", linestyle="--", linewidth=1
            )
            axes[row_index, 0].set_ylim(0, 1.02)
            axes[row_index, 0].set_xlabel("")
            axes[row_index, 0].set_ylabel(
                f"{protocol}\n{coverage_label} (90% nominal)"
            )
            axes[row_index, 0].tick_params(axis="x", rotation=20)
            sns.pointplot(
                data=protocol_rows,
                x="method",
                y="mean_width_um",
                hue="dataset",
                errorbar="sd",
                dodge=0.25,
                ax=axes[row_index, 1],
            )
            axes[row_index, 1].set_xlabel("")
            axes[row_index, 1].set_ylabel(
                f"{protocol}\nMean interval width (µm)"
            )
            axes[row_index, 1].tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(output_dir / "figure_2_interval_transfer.png", dpi=240)
        plt.close(fig)

    curves = applicability["risk_curves"]
    core_methods = [
        "axis_aligned",
        "knn_5",
        "mahalanobis_oas",
        "isolation_forest",
        "bootstrap_disagreement",
        "oof_residual_risk",
        "hydraulic_diameter_only",
        "predicted_size_only",
    ]
    risk_plot = curves[
        (curves["scope"] == "row_micro")
        & (curves["outcome"] == "relative")
        & (curves["method"].isin(core_methods))
    ]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    sns.lineplot(
        data=risk_plot,
        x="coverage",
        y="risk",
        hue="method",
        estimator=None,
        linewidth=1.8,
        ax=ax,
    )
    ax.set_xlabel("Retained candidate coverage")
    ax.set_ylabel("Mean absolute relative error")
    ax.set_xlim(0.05, 1.0)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_3_relative_risk_coverage.png", dpi=240)
    plt.close(fig)

    source_binary = applicability["source_binary_summary"].copy()
    selected = source_binary[
        (
            (source_binary["training_quantile"] == 0.95)
            & source_binary["method"].isin(
                ["axis_aligned", "knn_5", "mahalanobis_oas", "isolation_forest"]
            )
        )
        | (source_binary["method"] == "strict_minmax")
    ]
    if len(selected):
        heat = selected.pivot_table(
            index="method", columns="source", values="retention_rate", aggfunc="first"
        )
        fig, ax = plt.subplots(figsize=(7.8, 3.8))
        sns.heatmap(
            heat,
            annot=True,
            fmt=".0%",
            cmap="Blues",
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Retention rate"},
            ax=ax,
        )
        ax.set_xlabel("External source")
        ax.set_ylabel("")
        fig.tight_layout()
        fig.savefig(output_dir / "figure_4_source_retention.png", dpi=240)
        plt.close(fig)

    if len(replay_summary):
        replay = replay_summary[
            (replay_summary["tolerance_kind"] == "relative_10pct")
            & (replay_summary["top_k"] == 1)
        ].copy()
        if "replay_scope" in replay.columns:
            replay = replay[
                replay["replay_scope"] == "source_pool_primary"
            ]
        preferred = [
            "point_only",
            "strict_minmax",
            "knn_5__q0_95",
            "mahalanobis_oas__q0_95",
            "isolation_forest__q0_95",
            "conformal_target_inclusion",
        ]
        replay = replay[replay["strategy"].isin(preferred)]
        if len(replay):
            fig, ax = plt.subplots(figsize=(7.4, 4.8))
            sns.scatterplot(
                data=replay,
                x="query_coverage",
                y="success_precision_answered",
                size="successful_target_recall",
                hue="strategy",
                sizes=(70, 260),
                ax=ax,
            )
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("Answered-query coverage")
            ax.set_ylabel("Success among answered queries")
            ax.legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
            fig.tight_layout()
            fig.savefig(output_dir / "figure_5_inverse_replay.png", dpi=240)
            plt.close(fig)

    binary = applicability["binary_summary"].copy()
    sensitivity = binary[
        binary["method"].isin(["knn_3", "knn_5", "knn_10", "knn_20"])
        & binary["training_quantile"].apply(lambda value: isinstance(value, float))
    ]
    if len(sensitivity):
        heat = sensitivity.pivot_table(
            index="method",
            columns="training_quantile",
            values="mean_absolute_relative_error",
            aggfunc="first",
        )
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        sns.heatmap(heat, annot=True, fmt=".3f", cmap="mako_r", ax=ax)
        ax.set_xlabel("Training-score quantile")
        ax.set_ylabel("")
        fig.tight_layout()
        fig.savefig(output_dir / "figure_6_knn_sensitivity.png", dpi=240)
    plt.close(fig)


INTERVAL_SOURCE_MACRO_ENDPOINT_FIELDS = (
    "empirical_coverage",
    "coverage_gap",
    "mean_width_normalized",
    "median_width_normalized",
    "mean_width_um",
    "median_width_um",
    "mean_relative_width",
    "median_relative_width",
    "max_width_normalized",
    "max_width_um",
    "minimum_lower_normalized",
    "maximum_upper_normalized",
    "fraction_negative_lower_endpoints",
    "fraction_upper_endpoints_above_hydraulic_diameter",
    "empirical_simultaneous_group_coverage",
    "simultaneous_group_coverage_gap",
)

INTERVAL_INVARIANT_METADATA_FIELDS = (
    "calibration_unit",
    "n_calibration_units",
    "calibration_score_aggregation",
    "calibration_requested_rank",
    "calibration_used_rank",
    "nominal_finite_threshold_attainable",
    "finite_threshold_coverage_ceiling",
    "scale_residual_protocol",
    "scale_oof_folds",
    "n_scale_training_groups",
    "n_train_rows",
    "n_calibration_rows",
    "n_internal_test_rows",
    "train_n_rows",
    "train_n_dependency_groups",
    "train_n_se_rows",
    "train_n_de_rows",
    "train_n_de_physical_experiments",
    "calibration_n_rows",
    "calibration_n_dependency_groups",
    "calibration_n_se_rows",
    "calibration_n_de_rows",
    "calibration_n_de_physical_experiments",
    "internal_test_n_rows",
    "internal_test_n_dependency_groups",
    "internal_test_n_se_rows",
    "internal_test_n_de_rows",
    "internal_test_n_de_physical_experiments",
)


def source_macro_interval_summary(source_rows: pd.DataFrame) -> pd.DataFrame:
    """Compute source-balanced endpoint macros without averaging metadata.

    Endpoint statistics receive an unweighted mean over sources. Evaluation
    counts are exported with explicit ``sum_source_`` names, while split and
    calibration metadata must be invariant within each analysis cell and are
    copied exactly rather than numerically averaged.
    """

    group_keys = ["method", "protocol", "nominal", "seed", "dataset"]
    records: list[dict[str, object]] = []
    for key, group in source_rows.groupby(group_keys, sort=True, dropna=False):
        record = dict(zip(group_keys, key))
        record["source"] = "__source_macro__"
        record["n_sources"] = int(group["source"].nunique())
        for field in INTERVAL_SOURCE_MACRO_ENDPOINT_FIELDS:
            if field in group.columns:
                values = pd.to_numeric(group[field], errors="coerce")
                record[field] = float(values.mean())
                record[f"n_sources_with_{field}"] = int(values.notna().sum())
        for field in (
            "n_rows",
            "n_dependency_groups",
            "n_negative_lower_endpoints",
            "n_upper_endpoints_above_hydraulic_diameter",
        ):
            if field in group.columns:
                record[f"sum_source_{field}"] = int(
                    pd.to_numeric(group[field], errors="raise").sum()
                )
        for field in INTERVAL_INVARIANT_METADATA_FIELDS:
            if field not in group.columns:
                continue
            values = group[field]
            if values.nunique(dropna=False) != 1:
                raise ValueError(
                    f"interval metadata {field!r} varies within a source-macro cell"
                )
            record[field] = values.iloc[0]
        records.append(record)
    return pd.DataFrame.from_records(records)


def summarize_key_results(
    split_metrics: pd.DataFrame,
    external_summary: pd.DataFrame,
    applicability: dict[str, pd.DataFrame],
    interval_summary: pd.DataFrame,
    replay_summary: pd.DataFrame,
) -> dict[str, object]:
    xgb_split = split_metrics[split_metrics["model"] == "xgboost"]
    split_summary = (
        xgb_split.groupby("protocol")[
            [
                "mape_percent",
                "mae_um",
                "mean_absolute_normalized_error",
                "exact_input_overlap_fraction",
                "de_pair_leakage_fraction",
            ]
        ]
        .agg(["mean", "std"])
    )
    result: dict[str, object] = {
        "split_protocol_xgboost": {
            protocol: {
                f"{metric}_{statistic}": float(value)
                for (metric, statistic), value in row.items()
            }
            for protocol, row in split_summary.iterrows()
        },
        "external_model_summary": external_summary.to_dict(orient="records"),
        "applicability_binary_summary": applicability[
            "binary_summary"
        ].to_dict(orient="records"),
        "claim_boundary": (
            "All external and replay outcomes are retrospective. The 64-row DAFD "
            "stress set was inspected during pilot work, so expanded analyses are "
            "exploratory rather than confirmatory. No result proves wet-lab feasibility."
        ),
    }
    if len(interval_summary):
        keys = ["dataset", "source", "method", "protocol", "nominal"]
        records: list[dict[str, object]] = []
        for key, group in interval_summary.groupby(
            keys, sort=True, dropna=False
        ):
            record: dict[str, object] = dict(zip(keys, key))
            record["n_seeds"] = int(group["seed"].nunique())
            for field in INTERVAL_SOURCE_MACRO_ENDPOINT_FIELDS:
                if field in group.columns:
                    values = pd.to_numeric(group[field], errors="coerce")
                    record[f"{field}_mean_over_seeds"] = float(values.mean())
                    record[f"{field}_valid_seed_count"] = int(values.notna().sum())
            records.append(record)
        result["interval_summary_mean_over_seeds"] = records
    if len(replay_summary):
        result["inverse_replay_summary"] = replay_summary.to_dict(orient="records")
    return result


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=workspace / "work" / "data" / "osf_938rs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workspace / "work" / "microfluidic_reliability" / "expanded_artifacts",
    )
    parser.add_argument("--fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    project_dir = Path(__file__).resolve().parents[1]
    comprehensive_path = args.data_dir / "Comprehensive_normalized.xlsx"
    external_path = args.data_dir / "Generalizability_data_normalized.xlsx"
    comprehensive = DAFD.load_dataset(comprehensive_path, require_rate=True)
    external = DAFD.load_dataset(external_path, require_rate=False)

    audit = data_audit(
        comprehensive,
        external,
        [
            comprehensive_path,
            external_path,
            args.data_dir / "Final_Comprehensive_plus_Generalizability_data_normalized.xlsx",
            args.data_dir / "Raw_SE_dataset.xlsx",
        ],
    )
    write_json(args.output_dir / "data_audit.json", audit)
    write_json(args.output_dir / "environment.json", environment_manifest())
    test_evidence_path = args.output_dir / "test_evidence.json"
    write_json(
        test_evidence_path,
        {
            "status": "not_executed_by_analysis_script",
            "recommended_command": (
                "python -m unittest discover -s "
                "work/microfluidic_reliability/tests -p 'test_*.py'"
            ),
            "claim_boundary": (
                "This placeholder is not evidence that tests passed; verification "
                "must be recorded by the release process after execution."
            ),
        },
    )
    write_json(
        args.output_dir / "reproducibility_manifest.json",
        reproducibility_manifest(project_dir, test_evidence_path),
    )
    write_json(
        args.output_dir / "experiment_registry.json",
        {
            "split_protocols": [asdict(value) for value in split_protocols(args.fast)],
            "models": MODEL_NAMES,
            "risk_quantiles": RISK_QUANTILES,
            "relative_tolerances": RELATIVE_TOLERANCES,
            "absolute_tolerances_um": ABSOLUTE_TOLERANCES_UM,
            "target_grid_um": TARGET_GRID_UM,
            "top_k": TOP_K_VALUES,
            "fast_mode": bool(args.fast),
        },
    )

    split_metrics, split_predictions, split_manifest = evaluate_split_protocols(
        comprehensive, split_protocols(args.fast)
    )
    split_metrics.to_csv(args.output_dir / "split_metrics.csv", index=False)
    split_predictions.to_csv(args.output_dir / "split_predictions.csv", index=False)
    split_manifest.to_csv(args.output_dir / "split_manifest.csv", index=False)

    external_summary, external_rows = evaluate_external_models(
        comprehensive, external
    )
    external_summary.to_csv(args.output_dir / "external_model_summary.csv", index=False)
    external_rows.to_csv(args.output_dir / "external_model_predictions.csv", index=False)
    shuffled_summary, shuffled_rows = shuffled_label_control(
        comprehensive, external, fast=args.fast
    )
    shuffled_summary.to_csv(
        args.output_dir / "shuffled_label_control_summary.csv", index=False
    )
    shuffled_rows.to_csv(
        args.output_dir / "shuffled_label_control_predictions.csv", index=False
    )
    external_xgb = external_rows[external_rows["model"] == "xgboost"].reset_index(
        drop=True
    )

    applicability = external_applicability_analysis(
        comprehensive, external, external_xgb, fast=args.fast
    )
    for name, frame in applicability.items():
        frame.to_csv(args.output_dir / f"applicability_{name}.csv", index=False)
    bootstrap_summary = hierarchical_bootstrap_external(
        applicability["scored"], fast=args.fast
    )
    bootstrap_summary.to_csv(
        args.output_dir / "hierarchical_bootstrap_summary.csv", index=False
    )

    group_risk = internal_group_risk_validation(comprehensive, fast=args.fast)
    group_risk.to_csv(args.output_dir / "internal_group_risk_validation.csv", index=False)

    from interval_experiments import run_interval_suite

    intervals = run_interval_suite(comprehensive, external, fast=args.fast)
    interval_summary_rows = intervals["summary"]
    source_rows = interval_summary_rows[
        (interval_summary_rows["dataset"] == "external")
        & (interval_summary_rows["source"] != "__overall__")
    ]
    if len(source_rows):
        macro = source_macro_interval_summary(source_rows)
        intervals["summary"] = pd.concat(
            [interval_summary_rows, macro], ignore_index=True, sort=False
        )
    for name, frame in intervals.items():
        frame.to_csv(args.output_dir / f"interval_{name}.csv", index=False)

    external_interval_predictions = intervals["external_predictions"]
    representative = external_interval_predictions[
        (external_interval_predictions["method"] == "cqr")
        & (external_interval_predictions["protocol"] == "geometry_group")
        & (external_interval_predictions["nominal"] == 0.90)
        & (external_interval_predictions["seed"] == 0)
    ]
    if len(representative):
        interval_for_replay = (
            representative[["row_index", "lower_um", "upper_um"]]
            .set_index("row_index")
            .reindex(applicability["scored"]["row_index"].astype(int))
        )
        lower_um = interval_for_replay["lower_um"].to_numpy(dtype=float)
        upper_um = interval_for_replay["upper_um"].to_numpy(dtype=float)
    else:
        lower_um = upper_um = None

    replay_masks = build_replay_masks(applicability["scored"])
    replay_selection_tables: list[pd.DataFrame] = []
    replay_scored_tables: list[pd.DataFrame] = []
    replay_summary_tables: list[pd.DataFrame] = []
    replay_outcomes_by_scope: dict[str, pd.DataFrame] = {}
    selection_columns = [
        "row_index",
        "predicted_um",
        "source",
        "source_context",
    ] + [
        column
        for column in applicability["scored"].columns
        if column.startswith("accepted__")
    ]
    for replay_scope, pool_column in (
        ("source_pool_primary", "source"),
        ("source_context_sensitivity", "source_context"),
        ("pooled_sensitivity", None),
    ):
        replay_inputs = applicability["scored"][selection_columns].copy()
        replay_inputs["source_context"] = (
            "ALL"
            if pool_column is None
            else replay_inputs[pool_column].astype(str)
        )
        replay_outcomes = applicability["scored"].copy()
        replay_outcomes["source_context"] = replay_inputs["source_context"]
        replay_outcomes_by_scope[replay_scope] = replay_outcomes
        selections = build_replay_selections(
            replay_inputs, replay_masks, lower_um, upper_um
        )
        selections.insert(0, "replay_scope", replay_scope)
        replay_selection_tables.append(selections)
    replay_selections = pd.concat(replay_selection_tables, ignore_index=True)
    replay_selection_path = (
        args.output_dir / "inverse_replay_selections_label_free.csv"
    )
    replay_selections.to_csv(replay_selection_path, index=False)
    write_json(
        args.output_dir / "inverse_replay_selection_freeze.json",
        {
            "selection_sha256": sha256(replay_selection_path),
            "selection_rows": int(len(replay_selections)),
            "selection_columns": list(replay_selections.columns),
            "observed_outcome_columns_present": False,
            "claim_boundary": (
                "Retrospective label-free ranking of already observed historical "
                "candidates; not a prospective blind search or feasibility validation."
            ),
        },
    )
    for replay_scope, outcomes in replay_outcomes_by_scope.items():
        scope_selections = replay_selections[
            replay_selections["replay_scope"] == replay_scope
        ].drop(columns="replay_scope")
        scored_table, summary_table = score_replay_selections(
            scope_selections, outcomes
        )
        scored_table.insert(0, "replay_scope", replay_scope)
        summary_table.insert(0, "replay_scope", replay_scope)
        replay_scored_tables.append(scored_table)
        replay_summary_tables.append(summary_table)
    replay_scored = pd.concat(replay_scored_tables, ignore_index=True)
    replay_summary = pd.concat(replay_summary_tables, ignore_index=True)
    replay_scored.to_csv(args.output_dir / "inverse_replay_scored.csv", index=False)
    replay_summary.to_csv(args.output_dir / "inverse_replay_summary.csv", index=False)

    make_figures(
        split_metrics,
        intervals["summary"],
        applicability,
        replay_summary,
        args.output_dir,
    )
    summary = summarize_key_results(
        split_metrics,
        external_summary,
        applicability,
        intervals["summary"],
        replay_summary,
    )
    write_json(args.output_dir / "expanded_results_summary.json", summary)

    hashes = {
        path.name: sha256(path)
        for path in sorted(args.output_dir.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    write_json(args.output_dir / "artifact_hashes.json", hashes)
    print(json.dumps({"output_dir": str(args.output_dir), "files": len(hashes)}, indent=2))


if __name__ == "__main__":
    main()
