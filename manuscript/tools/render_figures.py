#!/usr/bin/env python3
"""Re-render five manuscript figures from the frozen artifact data (v0.3).

The frozen PNGs in the workspace are left untouched; this script reads the same
artifact files they were drawn from and writes corrected renderings into
manuscript/figures/: Figure 1 (CQR coverage shift) with labels taken from the
JSON and placed above the error bars, Figure 2 (distance versus error) and
Figure 9 (source retention) with "et al." spelled with its full stop, and
Figures 8/9 with readable method names instead of code identifiers, and Figure 4 (V1 blind diagnostics) with more space between its two panels (v0.3.2).

Usage: python render_figures.py --work <work/microfluidic_reliability> --out <manuscript/figures>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

METHOD_LABELS = {
    "axis_aligned": "Per-feature min–max",
    "strict_minmax": "Strict min–max",
    "knn_5": "5-NN distance",
    "mahalanobis_oas": "OAS Mahalanobis",
    "isolation_forest": "Isolation Forest",
    "bootstrap_disagreement": "Bootstrap disagreement",
    "oof_residual_risk": "Out-of-fold residual risk",
    "hydraulic_diameter_only": "Hydraulic diameter only (control)",
    "predicted_size_only": "Predicted size only (control)",
    "random_score": "Random score (control)",
}
RISK_METHODS = ["axis_aligned", "knn_5", "mahalanobis_oas", "isolation_forest",
                "bootstrap_disagreement", "oof_residual_risk", "hydraulic_diameter_only", "predicted_size_only"]
RETENTION_METHODS = ["strict_minmax", "axis_aligned", "knn_5", "mahalanobis_oas", "isolation_forest"]


def source_label(name: str) -> str:
    name = str(name).strip()
    return name + "." if name.endswith("et al") else name


def figure_cqr(work: Path, out: Path) -> None:
    metrics = json.loads((work / "artifacts" / "reliability_metrics.json").read_text())
    cqr = metrics["cqr_distribution_shift"]
    internal, external = cqr["internal_random_split"], cqr["external_literature_shift"]
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    labels = ["Random split", "External literature"]
    means = [internal["coverage"]["mean"], external["coverage"]["mean"]]
    errors = [internal["coverage"]["std"], external["coverage"]["std"]]
    bars = ax.bar(labels, means, yerr=errors, capsize=5, color=["#3977B9", "#D45A45"])
    ax.axhline(cqr["nominal_coverage"], color="#222222", linestyle="--", linewidth=1.2, label="Nominal 90%")
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("Empirical interval coverage")
    ax.legend(frameon=False, loc="upper right")
    for bar, value, err in zip(bars, means, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, value + err + 0.02, f"{value:.1%}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out / "figure_cqr_coverage_shift.png", dpi=220)
    plt.close(fig)


def figure_distance(work: Path, out: Path) -> None:
    frame = pd.read_csv(work / "artifacts" / "external_reliability_predictions.csv")
    metrics = json.loads((work / "artifacts" / "reliability_metrics.json").read_text())
    threshold = float(metrics["label_free_applicability_pruning"]["threshold"])
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for reference, group in frame.groupby("Ref"):
        ax.scatter(group["Applicability distance"], group["Absolute error (um)"],
                   label=source_label(reference), s=42, alpha=0.85)
    ax.axvline(threshold, color="#222222", linestyle="--", linewidth=1.2, label="Training trust-region threshold")
    ax.set_xlabel("5-nearest-neighbour applicability distance")
    ax.set_ylabel("Absolute diameter error (µm)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figure_distance_vs_error.png", dpi=220)
    plt.close(fig)


def figure_risk_coverage(work: Path, out: Path) -> None:
    curves = pd.read_csv(work / "expanded_artifacts_canonical_run2" / "applicability_risk_curves.csv")
    plot = curves[(curves["scope"] == "row_micro") & (curves["outcome"] == "relative") & curves["method"].isin(RISK_METHODS)].copy()
    plot["Method"] = pd.Categorical(plot["method"].map(METHOD_LABELS), [METHOD_LABELS[m] for m in RISK_METHODS])
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    sns.lineplot(data=plot.sort_values(["Method", "coverage"]), x="coverage", y="risk", hue="Method", estimator=None, linewidth=1.8, ax=ax)
    ax.set_xlabel("Retained candidate coverage")
    ax.set_ylabel("Mean absolute relative error")
    ax.set_xlim(0.05, 1.0)
    ax.legend(frameon=False, fontsize=8, ncol=2, title=None)
    fig.tight_layout()
    fig.savefig(out / "expanded_figure_3_relative_risk_coverage.png", dpi=240)
    plt.close(fig)


def figure_source_retention(work: Path, out: Path) -> None:
    summary = pd.read_csv(work / "expanded_artifacts_canonical_run2" / "applicability_source_binary_summary.csv", dtype={"training_quantile": str})
    selected = summary[((summary["training_quantile"] == "0.95") & summary["method"].isin(RETENTION_METHODS)) | (summary["method"] == "strict_minmax")].copy()
    heat = selected.pivot_table(index="method", columns="source", values="retention_rate", aggfunc="first")
    heat = heat.loc[[m for m in RETENTION_METHODS if m in heat.index]]
    heat.index = [METHOD_LABELS[m] + ("" if m == "strict_minmax" else " (q95)") for m in heat.index]
    heat.columns = [source_label(c) for c in heat.columns]
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    sns.heatmap(heat, annot=True, fmt=".0%", cmap="Blues", vmin=0, vmax=1, cbar_kws={"label": "Retention rate"}, ax=ax)
    ax.set_xlabel("External source")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out / "expanded_figure_4_source_retention.png", dpi=240)
    plt.close(fig)


def figure_v1_diagnostics(work: Path, out: Path) -> None:
    """Figure 4: V1 blind diagnostics with more room between the two panels."""
    merged = pd.read_csv(work / "blind_artifacts" / "posthoc_analysis_corrected" / "blind_tx100_merged_diagnostics.csv")
    actual = merged["target_diameter_mm"].to_numpy(dtype=float)
    predicted = merged["predicted_diameter_mm"].to_numpy(dtype=float)
    signed = merged["signed_error_mm"].to_numpy(dtype=float)
    lower, upper = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    plt.rcdefaults()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    scatter = axes[0].scatter(actual, predicted, c=merged["absolute_relative_error"], cmap="viridis", s=34, edgecolor="white", linewidth=0.35)
    axes[0].plot([lower, upper], [lower, upper], "--", color="#b91c1c", lw=1.3)
    axes[0].set_xlabel("Observed TX100 diameter (mm)")
    axes[0].set_ylabel("Frozen prediction (mm)")
    axes[0].set_title("Blind point predictions")
    colorbar = fig.colorbar(scatter, ax=axes[0], fraction=0.046, pad=0.04)
    colorbar.set_label("Absolute relative error")
    axes[1].scatter(predicted, signed, color="#2563eb", alpha=0.78, s=32)
    axes[1].axhline(0.0, color="#111827", lw=1.1)
    axes[1].axhline(float(signed.mean()), color="#b91c1c", ls="--", lw=1.3)
    axes[1].set_xlabel("Frozen prediction (mm)")
    axes[1].set_ylabel("Prediction \u2212 observation (mm)")
    axes[1].set_title("Signed residuals")
    fig.set_constrained_layout_pads(wspace=0.08)
    fig.savefig(out / "v1_figure_1_blind_prediction_diagnostics.png", dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    figure_cqr(args.work, args.out)
    figure_distance(args.work, args.out)
    figure_risk_coverage(args.work, args.out)
    figure_source_retention(args.work, args.out)
    figure_v1_diagnostics(args.work, args.out)
    print("rendered 5 figures into", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
