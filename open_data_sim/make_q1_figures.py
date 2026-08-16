"""
Q1 multi-panel figures + Excel / SI table export (Results Round-1 redesign).

Frozen Methods only. No model retuning.
Fig.2: 4-panel baseline stress (convergence → SI).
Fig.3: 4-panel RQ1 performance.
Fig.4: 6-panel equivalence + structural robustness (no gamma-tilt main panel).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs_revision"
FIG = Path(__file__).resolve().parent / "figures_revision"
FIG.mkdir(parents=True, exist_ok=True)
AUDIT = OUT / "RESULTS_ROUND1_AUDIT"
_root = Path(__file__).resolve().parents[1]
_ms_xlsx = _root / "MANUSCRIPT_OPEN_DATA" / "tables_excel"
XLSX_DIR = _ms_xlsx if (_root / "MANUSCRIPT_OPEN_DATA").exists() else (OUT / "tables_excel")
XLSX_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = ["S0", "S1", "S2", "S3", "S_OPT"]
SCENARIO_LABELS = {
    "S0": "S0",
    "S1": "S1",
    "S2": "S2",
    "S3": "S3",
    "S_OPT": "S_OPT",
    "S3s-H": "S3s-H",
    "S3-H": "S3-H",
}
COLORS = {
    "S0": "#8B1E1E",
    "S1": "#1B4F72",
    "S2": "#2874A6",
    "S3": "#148F77",
    "S_OPT": "#1D8348",
    "S3s-H": "#B9770E",
    "S3-H": "#148F77",
    "low": "#5D6D7E",
    "normal": "#2E86AB",
    "high": "#C0392B",
}

PANEL_NAMES = {
    2: {
        "a": "2024 intensity path and regimes",
        "b": "S0 peak queue by regime",
        "c": "S0 p90 workload proxy by regime",
        "d": "Low-to-high regime shift",
    },
    3: {
        "a": "Peak residual queue",
        "b": "p90 workload proxy",
        "c": "Mean workload proxy",
        "d": "Relative peak improvement vs S0",
    },
    4: {
        "a": "Equivalence vs S1 (CBB 90% CI)",
        "b": "Peak queue by effective capacity",
        "c": "Mean workload by effective capacity",
        "d": "Primary G: S2−S1 by φ × capacity",
        "e": "S0 peak-shape robustness",
        "f": "Hard-yard S3s-H vs S3-H",
    },
}


def apply_rc() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def finish_panel(ax, letter: str, name: str) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.text(
        0.5,
        -0.28,
        f"({letter}) {name}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        fontweight="normal",
        fontfamily="Times New Roman",
    )


def raincloud(ax, data_by_key, order, colors, ylabel, letter, name, logy=False):
    positions = np.arange(len(order))
    parts = ax.violinplot(
        [np.asarray(data_by_key[k], dtype=float) for k in order],
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.75,
    )
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[order[i]])
        pc.set_alpha(0.25)
        pc.set_edgecolor("none")
    for i, k in enumerate(order):
        y = np.asarray(data_by_key[k], dtype=float)
        rng = np.random.default_rng(i + 7)
        y_s = rng.choice(y, 400, replace=False) if len(y) > 400 else y
        x = i + rng.uniform(-0.12, 0.12, size=len(y_s))
        ax.scatter(x, y_s, s=4, alpha=0.18, color=colors[k], linewidths=0)
        q1, med, q3 = np.percentile(y, [25, 50, 75])
        ax.plot([i - 0.18, i + 0.18], [med, med], color="#111", lw=1.6, zorder=5)
        ax.vlines(i, q1, q3, color="#111", lw=1.1, zorder=4)
    ax.set_xticks(positions)
    ax.set_xticklabels(order, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    if logy:
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(bottom=0)
    finish_panel(ax, letter, name)


def filter_s1(res: pd.DataFrame) -> pd.DataFrame:
    if "n_servers" in res.columns:
        return res[res["n_servers"] == 1].copy()
    return res.copy()


def fig_a_stress(res: pd.DataFrame, daily: pd.DataFrame) -> None:
    """Figure 2: baseline stress envelope (no MC convergence)."""
    names = PANEL_NAMES[2]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6))
    fig.subplots_adjust(hspace=0.50, wspace=0.30, left=0.08, right=0.98, top=0.94, bottom=0.11)

    # (a) intensity path
    ax = axes[0, 0]
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"], utc=True)
    ax.fill_between(d["date"], d["I"], color="#D6EAF8", lw=0)
    ax.plot(d["date"], d["I"], color="#1B4F72", lw=1.0)
    for reg, col in [("high", COLORS["high"]), ("low", COLORS["low"])]:
        sample = res[(res.regime == reg) & (res.scenario == "S0")].groupby("date")["I"].first()
        ax.scatter(
            pd.to_datetime(sample.index),
            sample.values,
            s=22,
            color=col,
            zorder=3,
            label=f"{reg}-week days",
        )
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_ylabel("Seaside intensity $I_t$", fontsize=9)
    finish_panel(ax, "a", names["a"])

    # (b) peak by regime
    ax = axes[0, 1]
    s0 = res[res.scenario == "S0"]
    day_peak = s0.groupby(["date", "regime"])["peak_queue"].median().reset_index()
    data_peak = {reg: day_peak.loc[day_peak.regime == reg, "peak_queue"].to_numpy() for reg in ["low", "normal", "high"]}
    raincloud(ax, data_peak, ["low", "normal", "high"], COLORS, "Peak residual queue (trucks)", "b", names["b"], logy=False)

    # (c) p90 wait by regime
    ax = axes[1, 0]
    day_p90 = s0.groupby(["date", "regime"])["p90_wait_min"].median().reset_index()
    data_p90 = {reg: day_p90.loc[day_p90.regime == reg, "p90_wait_min"].to_numpy() for reg in ["low", "normal", "high"]}
    raincloud(
        ax, data_p90, ["low", "normal", "high"], COLORS, "p90 workload proxy (min)", "c", names["c"], logy=False
    )

    # (d) low-to-high stress change on a shared dimensionless scale.
    ax = axes[1, 1]
    day = (
        s0.groupby(["date", "regime"])[["peak_queue", "p90_wait_min"]]
        .median()
        .reset_index()
    )
    low_peak = day.loc[day.regime == "low", "peak_queue"].median()
    high_peak = day.loc[day.regime == "high", "peak_queue"].median()
    low_p90 = day.loc[day.regime == "low", "p90_wait_min"].median()
    high_p90 = day.loc[day.regime == "high", "p90_wait_min"].median()
    folds = [high_peak / low_peak, high_p90 / low_p90]
    labels = ["Peak queue", "p90 workload proxy"]
    y = np.arange(len(labels))
    ax.hlines(y, 1.0, folds, color="#888", lw=1.5)
    ax.scatter([1.0, 1.0], y, color=COLORS["low"], s=55, zorder=3)
    ax.scatter(folds, y, color=COLORS["high"], s=55, zorder=3)
    for i, (fold, low, high, unit) in enumerate(
        zip(folds, [low_peak, low_p90], [high_peak, high_p90], ["trucks", "min"])
    ):
        ax.annotate(
            f"{fold:.1f}×  ({low:.0f}→{high:.0f} {unit})",
            (fold, i),
            textcoords="offset points",
            xytext=(6, 0),
            va="center",
            fontsize=8,
        )
    ax.axvline(1.0, color="#999", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("High-to-low day-median ratio (×)", fontsize=9)
    ax.set_xlim(0.8, max(folds) + 0.8)
    ax.text(1.0, -0.22, "low-week baseline", ha="center", va="top", fontsize=7)
    finish_panel(ax, "d", names["d"])

    fig.savefig(FIG / "FigA_stress_envelope.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "FigA_stress_envelope.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_si_convergence(conv: pd.DataFrame, conv_multi: pd.DataFrame | None) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(conv["reps_used"], conv["median"], color="#1B4F72", lw=1.8, label="S0 peak median")
    ax.fill_between(conv["reps_used"], conv["p10"], conv["p90"], color="#1B4F72", alpha=0.18, label="p10–p90")
    ax.set_xlabel("Replications used", fontsize=9)
    ax.set_ylabel("S0 peak residual queue (trucks)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(FIG / "FigS_mc_convergence.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "FigS_mc_convergence.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_b_policy(res: pd.DataFrame, improv: pd.DataFrame, cbb: pd.DataFrame | None) -> None:
    """Figure 3: four-panel RQ1 performance under S=1 soft storage."""
    names = PANEL_NAMES[3]
    fig = plt.figure(figsize=(11.0, 8.2))
    gs = fig.add_gridspec(2, 2, hspace=0.48, wspace=0.30, left=0.08, right=0.98, top=0.94, bottom=0.11)

    day = res.groupby(["date", "scenario"])[["peak_queue", "p90_wait_min", "mean_wait_min"]].median().reset_index()
    data_peak = {s: day.loc[day.scenario == s, "peak_queue"].to_numpy() for s in SCENARIOS}
    data_p90 = {s: day.loc[day.scenario == s, "p90_wait_min"].to_numpy() for s in SCENARIOS}
    data_wait = {s: day.loc[day.scenario == s, "mean_wait_min"].to_numpy() for s in SCENARIOS}

    raincloud(
        fig.add_subplot(gs[0, 0]),
        data_peak,
        SCENARIOS,
        COLORS,
        "Peak residual queue (trucks)",
        "a",
        names["a"],
        logy=True,
    )
    raincloud(
        fig.add_subplot(gs[0, 1]),
        data_p90,
        SCENARIOS,
        COLORS,
        "p90 workload proxy (min)",
        "b",
        names["b"],
        logy=True,
    )
    raincloud(
        fig.add_subplot(gs[1, 0]),
        data_wait,
        SCENARIOS,
        COLORS,
        "Mean workload proxy (min)",
        "c",
        names["c"],
        logy=True,
    )

    ax = fig.add_subplot(gs[1, 1])
    sub = improv[improv.metric == "peak_queue"].set_index("scenario").reindex([s for s in SCENARIOS if s != "S0"])
    y = np.arange(len(sub))
    # Day-level p10–p90 of relative improvements
    ax.hlines(y, 100 * sub["p10_improvement_vs_S0"], 100 * sub["p90_improvement_vs_S0"], color="#888", lw=1.2)
    ax.scatter(100 * sub["median_improvement_vs_S0"], y, c=[COLORS[s] for s in sub.index], s=55, zorder=3)
    ax.axvline(0, color="#999", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([SCENARIO_LABELS[s] for s in sub.index], fontsize=8)
    ax.set_xlabel("Relative peak-queue improvement vs S0 (%)", fontsize=9)
    finish_panel(ax, "d", names["d"])

    fig.savefig(FIG / "FigB_policy_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "FigB_policy_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def table_r1_multiobjective(res: pd.DataFrame) -> pd.DataFrame:
    """Table 4 from day-level median-across-replication summaries."""
    day = (
        res.groupby(["date", "scenario"])[
            [
                "mean_wait_min",
                "p90_wait_min",
                "corridor_delay_index",
                "throughput",
                "storage_pressure",
            ]
        ]
        .median()
        .reset_index()
    )
    rows = []
    for s in SCENARIOS:
        g = day[day.scenario == s]
        rows.append(
            {
                "Scenario": s,
                "Mean workload proxy (min)": round(float(g["mean_wait_min"].median()), 1),
                "p90 workload proxy (min)": round(float(g["p90_wait_min"].median()), 1),
                "BPR-style congestion index": round(float(g["corridor_delay_index"].median()), 1),
                "Throughput": round(float(g["throughput"].median()), 1),
                "Mean notional yard-load ratio": round(float(g["storage_pressure"].median()), 3),
            }
        )
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "Table_R1_waiting_storage_deferred.csv", index=False)
    tab.to_csv(OUT / "Table4_soft_primary_redesign.csv", index=False)
    return tab


def fig_c_ablation(
    tost: pd.DataFrame,
    sens_s: pd.DataFrame,
    structural: pd.DataFrame,
    hard_contrasts: pd.DataFrame | None,
    cbb: pd.DataFrame | None,
) -> None:
    """Figure 4: CBB equivalence + structural robustness (no gamma-tilt panel)."""
    names = PANEL_NAMES[4]
    fig = plt.figure(figsize=(12.4, 8.8))
    gs = fig.add_gridspec(2, 3, hspace=0.52, wspace=0.34, left=0.08, right=0.98, top=0.94, bottom=0.11)

    # (a) forest plot with CBB CIs
    ax = fig.add_subplot(gs[0, 0])
    labels = ["S2−S1", "S3−S1", "S_OPT−S1"]
    contrasts = ["G_S2_minus_S1_rel", "S3_minus_S1_rel", "S_OPT_minus_S1_rel"]
    # fallback to tost file naming
    if cbb is None or not all((cbb.contrast == c).any() for c in contrasts):
        order = ["S2_vs_S1_rel_improvement", "S3_vs_S1_rel_improvement", "S_OPT_vs_S1_rel_improvement"]
        t5 = tost[(tost.metric == "peak_queue") & (tost.margin == 0.05)].set_index("compare").reindex(order)
        means = 100 * t5["mean_diff"].to_numpy()
        los = 100 * t5["ci_low"].to_numpy()
        his = 100 * t5["ci_high"].to_numpy()
    else:
        means, los, his = [], [], []
        for c in contrasts:
            row = cbb[cbb.contrast == c].iloc[0]
            means.append(float(row.mean_pp))
            los.append(float(row.ci_low_pp))
            his.append(float(row.ci_high_pp))
        means, los, his = np.array(means), np.array(los), np.array(his)
    y = np.arange(len(labels))
    ax.axvspan(-5, 5, color="#D5F5E3", alpha=0.75, label="±5 pp primary")
    ax.axvline(0, color="#666", lw=0.8)
    for i in range(len(labels)):
        ax.plot([los[i], his[i]], [y[i], y[i]], color="#1B4F72", lw=2.2, solid_capstyle="butt")
        ax.plot(means[i], y[i], "o", color="#1B4F72", markersize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean difference in relative peak improvement (pp)", fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    finish_panel(ax, "a", names["a"])

    # (b) peak by S
    ax = fig.add_subplot(gs[0, 1])
    hs = (
        sens_s[sens_s.scenario.isin(SCENARIOS)]
        .pivot(index="scenario", columns="n_servers", values="median_peak_queue")
        .reindex(SCENARIOS)
    )
    im = ax.imshow(hs.to_numpy(), aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(hs.shape[1]))
    ax.set_xticklabels([f"$S$={c}" for c in hs.columns], fontsize=8)
    ax.set_yticks(range(len(SCENARIOS)))
    ax.set_yticklabels(SCENARIOS, fontsize=8)
    for i in range(hs.shape[0]):
        for j in range(hs.shape[1]):
            ax.text(j, i, f"{hs.to_numpy()[i, j]:.0f}", ha="center", va="center", fontsize=7, color="#111")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    finish_panel(ax, "b", names["b"])

    # (c) mean wait by S
    ax = fig.add_subplot(gs[0, 2])
    wait_col = "median_mean_wait_min" if "median_mean_wait_min" in sens_s.columns else "median_waiting_proxy"
    hw = (
        sens_s[sens_s.scenario.isin(SCENARIOS)]
        .pivot(index="scenario", columns="n_servers", values=wait_col)
        .reindex(SCENARIOS)
    )
    im = ax.imshow(hw.to_numpy(), aspect="auto", cmap="PuBuGn")
    ax.set_xticks(range(hw.shape[1]))
    ax.set_xticklabels([f"$S$={c}" for c in hw.columns], fontsize=8)
    ax.set_yticks(range(len(SCENARIOS)))
    ax.set_yticklabels(SCENARIOS, fontsize=8)
    for i in range(hw.shape[0]):
        for j in range(hw.shape[1]):
            ax.text(j, i, f"{hw.to_numpy()[i, j]:.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    finish_panel(ax, "c", names["c"])

    # (d) phi × capacity: S2−S1 relative gap heatmap / slope
    ax = fig.add_subplot(gs[1, 0])
    grid = structural[structural.factor == "smooth_x_capacity"].copy()
    pivot = grid.pivot(index="static_smooth", columns="capacity_kind", values="S2_minus_S1_rel")
    pivot = pivot.reindex(columns=["flat", "mild", "strong"])
    im = ax.imshow(100 * pivot.to_numpy(), aspect="auto", cmap="Blues", vmin=0)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"$\\phi$={v:g}" for v in pivot.index], fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{100 * pivot.to_numpy()[i, j]:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    finish_panel(ax, "d", names["d"])

    # (e) S0 peak-shape robustness: median peaks by scenario
    ax = fig.add_subplot(gs[1, 1])
    shapes = structural[structural.factor == "s0_peak_shape"].copy()
    x = np.arange(3)
    width = 0.15
    shape_order = ["two_hump", "morning", "flat_uncoord"]
    scen_plot = ["S0", "S1", "S2", "S3", "S_OPT"]
    for i, s in enumerate(scen_plot):
        vals = []
        for sh in shape_order:
            row = shapes[shapes.peak_shape == sh]
            col = f"median_peak_{s}"
            vals.append(float(row.iloc[0][col]) if len(row) and col in row.columns else np.nan)
        ax.plot(x + i * 0.02, vals, marker="o", color=COLORS[s], label=s, lw=1.4)
    ax.set_xticks(x)
    ax.set_xticklabels(["two-hump", "morning", "flat"], fontsize=8)
    ax.set_ylabel("Median peak queue", fontsize=9)
    ax.set_yscale("symlog", linthresh=10)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    finish_panel(ax, "e", names["e"])

    # (f) hard-yard S3s-H vs S3-H
    ax = fig.add_subplot(gs[1, 2])
    metrics = ["peak_queue", "throughput", "deferred"]
    labels_m = ["Peak queue", "Throughput", "Proactive deferred"]
    means, los, his = [], [], []
    if hard_contrasts is not None:
        for m in metrics:
            key = f"S3sH_minus_S3H_{m}"
            row = hard_contrasts[hard_contrasts.contrast == key]
            if len(row) == 0 and m == "deferred":
                # Frozen export uses the short deferred label.
                row = hard_contrasts[hard_contrasts.contrast == "S3sH_minus_S3H_deferred"]
            if len(row):
                means.append(float(row.iloc[0].mean_diff))
                los.append(float(row.iloc[0].ci_low))
                his.append(float(row.iloc[0].ci_high))
            else:
                means.append(0.0)
                los.append(0.0)
                his.append(0.0)
    else:
        means, los, his = [0, 0, 0], [0, 0, 0], [0, 0, 0]
    y = np.arange(len(labels_m))
    ax.axvline(0, color="#666", lw=0.8)
    for i in range(len(labels_m)):
        ax.plot([los[i], his[i]], [y[i], y[i]], color="#B9770E", lw=2.2)
        ax.plot(means[i], y[i], "o", color="#B9770E", markersize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels_m, fontsize=8)
    ax.set_xlabel("S3s-H − S3-H (paired day effect)", fontsize=8)
    finish_panel(ax, "f", names["f"])

    fig.savefig(FIG / "FigC_ablation_robustness.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "FigC_ablation_robustness.pdf", bbox_inches="tight")
    plt.close(fig)


def write_si_tables(res: pd.DataFrame, improv: pd.DataFrame, wilc: pd.DataFrame, tost: pd.DataFrame, regime: pd.DataFrame) -> None:
    r = regime[regime.scenario == "S0"][["regime", "peak_queue", "mean_wait_min", "p90_wait_min", "waiting_proxy"]].copy()
    r.columns = [
        "Intensity regime",
        "Median peak queue",
        "Median mean wait (min)",
        "Median p90 wait (min)",
        "Median waiting proxy",
    ]
    r.to_csv(OUT / "SI_Table_regime_S0.csv", index=False)

    rows = []
    for s in SCENARIOS:
        g = res[res.scenario == s]
        rows.append(
            {
                "Scenario": s,
                "Median peak queue": g["peak_queue"].median(),
                "p10 peak": g["peak_queue"].quantile(0.1),
                "p90 peak": g["peak_queue"].quantile(0.9),
                "Median p90 wait": g["p90_wait_min"].median(),
                "Median waiting proxy": g["waiting_proxy"].median(),
                "Median BPR delay": g["corridor_delay_index"].median(),
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "SI_Table_KPI_by_scenario.csv", index=False)
    improv.to_csv(OUT / "SI_Table_improvement_vs_S0.csv", index=False)
    wilc.to_csv(OUT / "SI_Table_wilcoxon_holm_effectsize.csv", index=False)
    tost.to_csv(OUT / "SI_Table_TOST_3_5_7pp.csv", index=False)

    # Composite P SI-only table
    p_rows = []
    for s in SCENARIOS:
        g = res[res.scenario == s]
        p_rows.append(
            {
                "Scenario": s,
                "Median composite P": round(float(g.waiting_proxy.median()), 1),
                "p10": round(float(g.waiting_proxy.quantile(0.1)), 1),
                "p90": round(float(g.waiting_proxy.quantile(0.9)), 1),
            }
        )
    pd.DataFrame(p_rows).to_csv(OUT / "SI_Table_composite_P.csv", index=False)


def _capacity_sheet() -> pd.DataFrame:
    sens = pd.read_csv(OUT / "sensitivity_n_servers.csv")
    peak = sens.pivot(index="scenario", columns="n_servers", values="median_peak_queue")
    proxy_col = "median_mean_wait_min" if "median_mean_wait_min" in sens.columns else "median_waiting_proxy"
    proxy = sens.pivot(index="scenario", columns="n_servers", values=proxy_col)
    rows = []
    for s in SCENARIOS:
        if s not in peak.index:
            continue
        rows.append(
            {
                "Scenario": s,
                "Peak S=1": round(float(peak.loc[s, 1]), 0),
                "Peak S=2": round(float(peak.loc[s, 2]), 0),
                "Peak S=3": round(float(peak.loc[s, 3]), 0),
                "Mean wait S=1": round(float(proxy.loc[s, 1]), 0),
                "Mean wait S=2": round(float(proxy.loc[s, 2]), 0),
                "Mean wait S=3": round(float(proxy.loc[s, 3]), 0),
            }
        )
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "SI_Table_servers_S1_S2_S3.csv", index=False)
    return tab


def write_excel_workbook() -> Path:
    path = XLSX_DIR / "Tema_TAS_tables_for_crosscheck.xlsx"
    table4 = pd.read_csv(OUT / "Table4_soft_primary_redesign.csv")
    s1 = pd.read_csv(OUT / "SI_Table_regime_S0.csv")
    s2 = _capacity_sheet()
    wilc_path = OUT / "stats_wilcoxon_W_ties_table.csv"
    s3 = pd.read_csv(wilc_path) if wilc_path.exists() else pd.read_csv(OUT / "SI_Table_wilcoxon_holm_effectsize.csv")
    s4 = pd.read_csv(OUT / "stats_tost_equivalence.csv")
    structural = pd.read_csv(OUT / "robustness_structural_full_year.csv")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        table4.to_excel(writer, sheet_name="Table4_soft_primary", index=False)
        if (OUT / "Table4_panelB_hard_storage.csv").exists():
            pd.read_csv(OUT / "Table4_panelB_hard_storage.csv").to_excel(
                writer, sheet_name="Table4_panelB_hard", index=False
            )
        s1.to_excel(writer, sheet_name="TableS1_regime_S0", index=False)
        s2.to_excel(writer, sheet_name="TableS2_capacity_S123", index=False)
        s3.to_excel(writer, sheet_name="TableS3_wilcoxon", index=False)
        s4.to_excel(writer, sheet_name="TableS4_TOST", index=False)
        structural.to_excel(writer, sheet_name="TableS7_structural", index=False)
        if (AUDIT / "cbb_effects_and_cis.csv").exists():
            pd.read_csv(AUDIT / "cbb_effects_and_cis.csv").to_excel(
                writer, sheet_name="CBB_effects_CIs", index=False
            )
        if (OUT / "SI_Table_composite_P.csv").exists():
            pd.read_csv(OUT / "SI_Table_composite_P.csv").to_excel(
                writer, sheet_name="SI_composite_P", index=False
            )
    return path


def main() -> None:
    apply_rc()
    res = filter_s1(pd.read_csv(OUT / "scenario_day_replications.csv"))
    conv = pd.read_csv(OUT / "mc_convergence.csv")
    improv = pd.read_csv(OUT / "improvement_vs_S0.csv")
    tost = pd.read_csv(OUT / "stats_tost_equivalence.csv")
    wilc = pd.read_csv(OUT / "stats_wilcoxon_vs_S0.csv")
    regime = pd.read_csv(OUT / "summary_by_regime.csv")
    sens_s = pd.read_csv(OUT / "sensitivity_n_servers.csv")
    structural = pd.read_csv(OUT / "robustness_structural_full_year.csv")
    hard_path = OUT / "stats_hard_env_contrasts.csv"
    hard = pd.read_csv(hard_path) if hard_path.exists() else None
    cbb_path = AUDIT / "cbb_effects_and_cis.csv"
    if not cbb_path.exists():
        cbb_path = OUT / "stats_cbb_effects_vs_S0_and_confirmatory.csv"
    cbb = pd.read_csv(cbb_path) if cbb_path.exists() else None

    root = Path(__file__).resolve().parents[1]
    daily = pd.read_csv(root / "external_data" / "tema_portwatch_daily_2019_2026.csv")
    daily["date"] = pd.to_datetime(daily["date"], utc=True)
    daily = daily[daily["date"].dt.year == 2024].copy()
    # Use same historical anchors as the engine when available via scenario file I
    # For the path panel, reconstruct I from 2024 within-year min-max for display continuity
    # Prefer joining realised I from replications (already Methods-scaled).
    i_from_res = res[res.scenario == "S0"].groupby("date")["I"].first()
    daily["date_str"] = daily["date"].dt.strftime("%Y-%m-%d")
    daily = daily.merge(i_from_res.rename("I"), left_on="date_str", right_index=True, how="left")
    if daily["I"].isna().any():
        eps = 1e-9
        c = daily["portcalls_container"].astype(float)
        v = (daily["import_container"] + daily["export_container"]).astype(float)
        daily["I"] = 0.6 * (c - c.min()) / (c.max() - c.min() + eps) + 0.4 * (v - v.min()) / (
            v.max() - v.min() + eps
        )

    print("Building Figure 2 (stress envelope)...")
    fig_a_stress(res, daily)
    print("Building SI convergence...")
    fig_si_convergence(conv, None)
    print("Building Figure 3 (RQ1)...")
    fig_b_policy(res, improv, cbb)
    print("Building Table 4 source...")
    table_r1_multiobjective(res)
    print("Building Figure 4 (robustness)...")
    fig_c_ablation(tost, sens_s, structural, hard, cbb)
    print("Writing SI CSVs...")
    write_si_tables(res, improv, wilc, tost, regime)
    xlsx = write_excel_workbook()
    print("Excel ->", xlsx)
    print("Done ->", FIG)


if __name__ == "__main__":
    main()
