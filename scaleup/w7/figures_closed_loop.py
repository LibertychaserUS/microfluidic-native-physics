"""Figures for the closed-loop results (manuscript v0.5.2+). Deterministic; every plotted value is also written to values.json.

    python -I scaleup/w7/figures_closed_loop.py --work W --figdir microfluidic/manuscript/figures --out W/figures_closed_loop

Inputs (all frozen run outputs under W): e1e2_run1 (loops, budget), e0_run1/tiers_e0.csv, e1_reanalysis, e1_dense_sensitivity,
e1b_run1 cells, e4_mac_run1/results.json, e6_run1 (results, chance), e7_run1 (loops, units, results).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# palette (validated reference instance, light mode): slots 1-3 + neutrals
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                     "svg.fonttype": "none", "savefig.dpi": 300})
VALUES: dict = {}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.5); ax.set_axisbelow(True)


# ------------------------------------------------------------------ Fig. budget: share of tiers usable by k (E2)
def fig_budget(W: Path, figdir: Path):
    bt = pd.read_csv(W / "e1e2_run1/budget_table.csv")  # per tier and method: k_star (21 = not reached within 20)
    ks = np.arange(1, 21)
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.45), sharey=True)
    VALUES["budget"] = {"by_shift_type_warm": {}, "by_method_all": {}}
    types = [("chemistry", BLUE, "-", "o"), ("laboratory", ORANGE, "--", "s"), ("composition", AQUA, ":", "^"), ("device", INK2, "-.", "D")]
    ax = axes[0]
    ends = []
    for st, col, ls, mk in types:
        g = bt[(bt.method == "warm") & (bt.shift_type == st)]
        y = np.array([100.0 * np.mean(g.k_star <= k) for k in ks])
        ax.step(ks, y, where="post", color=col, linestyle=ls, linewidth=1.6)
        marks = [1, 2, 3, 5, 8, 10, 15, 20]
        ax.plot(marks, y[np.array(marks) - 1], color=col, marker=mk, markersize=3.6, linestyle="none")
        VALUES["budget"]["by_shift_type_warm"][st] = {"n_tiers": int(len(g)), "share_reached_by_k": dict(zip(map(int, ks), np.round(y, 2).tolist()))}
        ends.append((y[-1], st, col, len(g)))
    placed = []
    for yend, st, col, ntier in sorted(ends, reverse=True):
        yy = yend if not placed else min(yend, placed[-1] - 6.5)
        placed.append(yy)
        ax.text(21.0, yy, f"{st} ({ntier} tiers)  {yend:.0f}%", color=col, fontsize=7, va="center")
    ax.set_title("warm-start, by shift type", color=INK)
    ax = axes[1]
    for m, lab, col, ls, mk in (("warm", "warm-start", BLUE, "-", "o"), ("affine", "log-affine", ORANGE, "--", "s"), ("scratch", "scratch", AQUA, ":", "^")):
        g = bt[bt.method == m]
        y = np.array([100.0 * np.mean(g.k_star <= k) for k in ks])
        ax.step(ks, y, where="post", color=col, linestyle=ls, linewidth=1.6)
        marks = [1, 2, 3, 5, 8, 10, 15, 20]
        ax.plot(marks, y[np.array(marks) - 1], color=col, marker=mk, markersize=3.6, linestyle="none")
        ax.text(21.0, y[-1] + (2.6 if m == "warm" else (-2.6 if m == "affine" else 0.0)), f"{lab}  {y[-1]:.0f}%", color=col, fontsize=7, va="center")
        VALUES["budget"]["by_method_all"][m] = {"n_tiers": int(len(g)), "share_reached_by_k": dict(zip(map(int, ks), np.round(y, 2).tolist()))}
    ax.set_title("all 107 tiers, by adaptation method", color=INK)
    for ax in axes:
        ax.set_xlim(0.5, 20.5); ax.set_ylim(0, 100)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_xlabel("target rows measured, $k$")
        grid(ax, "y")
    axes[0].set_ylabel("tiers usable by $k$ (%)")
    fig.tight_layout(w_pad=7.5)
    fig.savefig(figdir / "fig_e2_budget_curves.pdf", bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ Fig. E4 slope chart
def fig_e4(W: Path, figdir: Path):
    r = json.loads((W / "e4_mac_run1/results.json").read_text())["E4_family_effect"]
    fams = [("xgboost", "XGBoost"), ("random_forest", "random forest"), ("qrf", "quantile forest"), ("mlp", "perceptron"), ("deep_ensemble", "deep ensemble"), ("gp", "Gaussian process")]
    pols = [("P0", "no adaptation\n(P0)"), ("P1", "two rows, then\nwarm-start (P1)"), ("P2", "warm-start\nat once (P2)")]
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    VALUES["e4"] = {}
    marks = {"xgboost": "o", "random_forest": "s", "qrf": "D", "mlp": "^", "deep_ensemble": "v", "gp": "P"}
    for f, lab in fams:
        ys = [r[p]["levels_mean_t_star"][f] for p, _ in pols]
        VALUES["e4"][f] = dict(zip([p for p, _ in pols], [round(y, 4) for y in ys]))
        col = BLUE if f == "xgboost" else (ORANGE if f == "gp" else MUTED)
        lw = 1.8 if f in ("xgboost", "gp") else 1.1
        ax.plot([0, 1, 2], ys, color=col, linewidth=lw, marker=marks[f], markersize=4.5, zorder=3 if f in ("xgboost", "gp") else 2)
    # direct labels, de-overlapped at each end
    for side, idx, ha, dx in (("left", 0, "right", -0.06), ("right", 2, "left", 0.06)):
        pts = sorted(((r[pols[idx][0]]["levels_mean_t_star"][f], f, lab) for f, lab in fams))
        placed = []
        for y, f, lab in pts:
            yy = y if not placed else max(y, placed[-1] + 0.10)
            placed.append(yy)
            col = BLUE if f == "xgboost" else (ORANGE if f == "gp" else INK2)
            ax.text(idx + dx, yy, f"{lab}  {y:.2f}" if side == "right" else f"{lab}  {y:.2f}", ha=ha, va="center", fontsize=7, color=col)
    sp = {p: max(r[p]["levels_mean_t_star"].values()) - min(r[p]["levels_mean_t_star"].values()) for p, _ in pols}
    VALUES["e4"]["spread"] = {p: round(v, 4) for p, v in sp.items()}
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels([l for _, l in pols])
    ax.set_xlim(-1.25, 3.05)
    ax.set_ylabel("mean time-to-target $T^{*}$ (measurements)")
    grid(ax)
    fig.tight_layout(); fig.savefig(figdir / "fig_e4_family_slopes.pdf"); plt.close(fig)


# ------------------------------------------------------------------ Fig. robustness forest (E1, E1b, E7 with E6 controls)
def fig_forest(W: Path, figdir: Path):
    e6 = json.loads((W / "e6_run1/results.json").read_text())["shuffled"]["delta_vs_P1"]
    settings = [("E1, all tiers ($\\tau$=0.10, B=20)", json.loads((W / "e1_reanalysis/results.json").read_text())["E1"]["delta_vs_reference"], "e6"),
                ("E1b ($\\tau$=0.05, B=20)", json.loads((W / "e1b_run1/cell_tau0.05_B20/results.json").read_text())["E1"]["delta_vs_reference"], "e6"),
                ("E1b ($\\tau$=0.10, B=10)", json.loads((W / "e1b_run1/cell_tau0.10_B10/results.json").read_text())["E1"]["delta_vs_reference"], "e6"),
                ("E1b ($\\tau$=0.05, B=10)", json.loads((W / "e1b_run1/cell_tau0.05_B10/results.json").read_text())["E1"]["delta_vs_reference"], "e6"),
                ("E1, dense tiers only", json.loads((W / "e1_dense_sensitivity/results_dense.json").read_text())["E1"]["delta_vs_reference"], "e6"),
                ("E7, hard tail targets ($\\tau$=0.03)", json.loads((W / "e7_run1/results.json").read_text())["blocks"]["hard"]["delta_vs_P1"], "e7")]
    pols = [("P2", "P2: warm-start\nat once"), ("P5", "P5: $k$-means\nseeding"), ("P7", "P7: log-affine\nadaptation"), ("P8", "P8: scratch\ntraining")]
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 2.9), sharey=True)
    VALUES["forest"] = {}
    used = set()
    ys = np.arange(len(settings))[::-1]
    for ax, (p, ptitle) in zip(axes, pols):
        VALUES["forest"][p] = {}
        for y, (lab, d, src) in zip(ys, settings):
            e = d[p]; lo, hi = e["ci_table_block"]; v = e.get("verdict_bh")
            shuffled_better = (e6[p]["verdict_bh"] == "better") if src == "e6" else bool(e.get("artefact"))
            frag = bool(e.get("fragile_under_drift_0p2"))
            if v == "better":
                status = "artefact" if shuffled_better else ("fragile" if frag else "confirmed")
            else:
                status = {"worse": "worse", "not distinguishable": "n.d."}[v]
            col, mk, fill = {"confirmed": (BLUE, "o", True), "fragile": (ORANGE, "o", False), "artefact": (ORANGE, "X", True),
                             "n.d.": (MUTED, "o", False), "worse": (INK2, "s", True)}[status]
            ax.plot([lo, hi], [y, y], color=col, linewidth=1.4, solid_capstyle="round")
            ax.plot([e["mean"]], [y], marker=mk, markersize=5.2, color=col, markerfacecolor=col if fill else "white", markeredgewidth=1.2, linestyle="none")
            VALUES["forest"][p][lab] = {"mean": round(e["mean"], 4), "ci": [round(lo, 4), round(hi, 4)], "status": status}
            used.add(status)
        ax.axvline(0, color=INK2, linewidth=0.7)
        ax.set_title(ptitle, color=INK)
        grid(ax, "x")
    axes[0].set_yticks(ys); axes[0].set_yticklabels([s[0] for s in settings])
    from matplotlib.lines import Line2D
    handles = [(st, Line2D([], [], color=c, marker=m, markerfacecolor=fc, linestyle="-", label=lab)) for st, c, m, fc, lab in (
        ("confirmed", BLUE, "o", BLUE, "better, confirmed"),
        ("fragile", ORANGE, "o", "white", "better, fragile under drift"),
        ("artefact", ORANGE, "X", ORANGE, "better, also under shuffled labels"),
        ("n.d.", MUTED, "o", "white", "not distinguishable"),
        ("worse", INK2, "s", INK2, "worse"))]
    handles = [h for st, h in handles if st in used]  # legend lists only the statuses that occur
    VALUES["forest_statuses"] = sorted(used)
    fig.supxlabel("difference in mean $T^{*}$ to the reference P1 (measurements; negative = fewer)", fontsize=8, y=0.17)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.56, 0.02), handlelength=1.6, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=0.6)
    fig.savefig(figdir / "fig_e6_robustness_forest.pdf"); plt.close(fig)


# ------------------------------------------------------------------ Fig. difficulty: loop vs chance (E1 central targets + E7 tail targets)
def fig_difficulty(W: Path, figdir: Path):
    l1 = pd.read_csv(W / "e1e2_run1/loops.csv"); ch1 = pd.read_csv(W / "e6_run1/chance.csv")
    m1 = l1[(l1.kind == "main") & l1.policy.isin(["P0", "P1", "P2", "CEIL"])]
    u1 = m1.groupby(["table", "tier", "target", "policy"]).t_star.mean().unstack().reset_index().merge(ch1[["table", "tier", "target", "chance_expected_t_star"]], on=["table", "tier", "target"])
    u1["task"] = "E1"
    l7 = pd.read_csv(W / "e7_run1/loops.csv"); un7 = pd.read_csv(W / "e7_run1/units.csv")
    m7 = l7[(l7.kind == "main") & l7.policy.isin(["P0", "P1", "P2", "CEIL"])]
    u7 = m7.groupby(["table", "tier", "target", "policy"]).t_star.mean().unstack().reset_index().merge(un7[["table", "tier", "target", "chance_expected_t_star"]], on=["table", "tier", "target"])
    u7["task"] = "E7"
    u = pd.concat([u1, u7], ignore_index=True)
    edges = [0, 2, 3, 4, 6, 9, 13, 17, 21.01]
    u["bin"] = pd.cut(u.chance_expected_t_star, edges, right=False)
    g = u.groupby("bin", observed=True)
    x = g.chance_expected_t_star.mean().to_numpy(); nunits = g.size().to_numpy()
    fig, ax = plt.subplots(figsize=(4.3, 3.0))
    ax.plot([1, 21], [1, 21], color=MUTED, linewidth=1.0, linestyle=(0, (4, 2)))
    ax.text(15.6, 17.4, "random proposals", color=INK2, fontsize=7, rotation=33)
    series = [("P0", "no adaptation (P0)", AQUA, ":", "^"), ("P1", "two rows, then warm-start (P1)", ORANGE, "--", "s"), ("P2", "warm-start at once (P2)", BLUE, "-", "o"),
              ("CEIL", "ceiling: refit on all tier labels", INK2, "-.", "D")]
    VALUES["difficulty"] = {"bins": [str(b) for b in g.groups.keys()], "x_mean_chance": np.round(x, 4).tolist(), "n_units": nunits.tolist(),
                            "n_units_E1": int((u.task == "E1").sum()), "n_units_E7": int((u.task == "E7").sum())}
    for p, lab, col, ls, mk in series:
        y = g[p].mean().to_numpy()
        ax.plot(x, y, color=col, linestyle=ls, linewidth=1.6, marker=mk, markersize=3.8, label=lab)
        VALUES["difficulty"][p] = np.round(y, 4).tolist()
    ax.set_xlabel("difficulty: expected measurements for random proposals")
    ax.set_ylabel("mean time-to-target $T^{*}$ (measurements)")
    ax.set_xlim(1, 21); ax.set_ylim(0.8, 21.5)
    grid(ax, "both")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(figdir / "fig_e7_difficulty.pdf"); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--figdir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    W = a.work.resolve(); figdir = a.figdir.resolve(); out = a.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    fig_budget(W, figdir); fig_e4(W, figdir); fig_forest(W, figdir); fig_difficulty(W, figdir)
    VALUES["script_sha256"] = sha(Path(__file__).resolve())
    (out / "values.json").write_text(json.dumps(VALUES, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"e4_spread": VALUES["e4"]["spread"], "difficulty_n": [VALUES["difficulty"]["n_units_E1"], VALUES["difficulty"]["n_units_E7"]], "bins": VALUES["difficulty"]["n_units"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
