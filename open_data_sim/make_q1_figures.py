"""
Multi-panel figures and summary tables from revision experiment outputs.

Writes FigA/FigB/FigC PNGs and selected CSV tables under outputs_revision/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / "outputs_revision"
FIG = Path(__file__).resolve().parent / "figures_revision"
FIG.mkdir(parents=True, exist_ok=True)

SCENARIOS = ["S0", "S1", "S2", "S3", "S3s", "S_OPT"]
SCENARIO_LABELS = {
    "S0": "S0 Uncoordinated",
    "S1": "S1 Static flatten",
    "S2": "S2 Forecast",
    "S3": "S3 Residual realloc.",
    "S3s": "S3s + storage",
    "S_OPT": "S_OPT LP+compliance",
}
COLORS = {
    "S0": "#8B1E1E",
    "S1": "#1B4F72",
    "S2": "#2874A6",
    "S3": "#148F77",
    "S3s": "#B9770E",
    "S_OPT": "#1D8348",
    "low": "#5D6D7E",
    "normal": "#2E86AB",
    "high": "#C0392B",
}


def style_axes(ax, title=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    if title:
        ax.set_title(title, fontsize=9, loc="left", fontweight="bold", pad=4)


def raincloud(ax, data_by_key, order, colors, ylabel, title, logy=False):
    """Violin + box + jittered points."""
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
        if len(y) > 400:
            y_s = rng.choice(y, 400, replace=False)
        else:
            y_s = y
        x = i + rng.uniform(-0.12, 0.12, size=len(y_s))
        ax.scatter(x, y_s, s=3, alpha=0.15, color=colors[k], linewidths=0)
        q1, med, q3 = np.percentile(y, [25, 50, 75])
        ax.plot([i - 0.18, i + 0.18], [med, med], color="#111", lw=1.6, zorder=5)
        ax.vlines(i, q1, q3, color="#111", lw=1.1, zorder=4)
    ax.set_xticks(positions)
    ax.set_xticklabels([SCENARIO_LABELS.get(k, k).replace(" ", "\n") for k in order], fontsize=6.5)
    ax.set_ylabel(ylabel, fontsize=8)
    if logy:
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(bottom=0)
    style_axes(ax, title)


def fig_a_stress(res: pd.DataFrame, conv: pd.DataFrame, daily: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), constrained_layout=True)
    # A1 ridgeline-like densities for S0 peak by regime
    ax = axes[0, 0]
    s0 = res[res.scenario == "S0"]
    for i, reg in enumerate(["low", "normal", "high"]):
        y = s0.loc[s0.regime == reg, "peak_queue"].to_numpy()
        if len(y) < 5:
            continue
        # kernel-ish histogram smoothed
        hist, bins = np.histogram(y, bins=40, density=True)
        centers = 0.5 * (bins[:-1] + bins[1:])
        offset = i * (hist.max() * 1.15 if hist.max() > 0 else 1)
        ax.fill_between(centers, offset, offset + hist, color=COLORS[reg], alpha=0.55, lw=0)
        ax.plot(centers, offset + hist, color="#222", lw=0.6)
        ax.text(centers[np.argmax(hist)], offset + hist.max() * 0.15, reg, fontsize=8, color="#222")
    ax.set_xlabel("Peak residual queue (trucks)", fontsize=8)
    ax.set_yticks([])
    style_axes(ax, "A  Density of peak queue under S0 by intensity regime")

    # A2 dumbbell low vs high for peak and waiting proxy
    ax = axes[0, 1]
    day = res[res.scenario == "S0"].groupby(["date", "regime"])[["peak_queue", "waiting_proxy"]].median().reset_index()
    for metric, ypos, label in [("peak_queue", 1.0, "Peak queue"), ("waiting_proxy", 0.0, "Waiting proxy")]:
        low = day.loc[day.regime == "low", metric].median()
        high = day.loc[day.regime == "high", metric].median()
        # normalise waiting proxy to similar visual scale for dual axis effect via twin annotation
        if metric == "waiting_proxy":
            # plot on secondary scaled axis using right labels
            ax2 = ax.twiny() if False else ax
        ax.hlines(ypos, low if metric == "peak_queue" else low / 10, high if metric == "peak_queue" else high / 10, color="#888", lw=1.2)
        if metric == "peak_queue":
            ax.scatter([low, high], [ypos, ypos], c=[COLORS["low"], COLORS["high"]], s=55, zorder=3)
            ax.annotate(f"low {low:.0f}", (low, ypos), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)
            ax.annotate(f"high {high:.0f}", (high, ypos), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)
        else:
            ax.scatter([low / 10, high / 10], [ypos, ypos], c=[COLORS["low"], COLORS["high"]], s=55, zorder=3, marker="D")
            ax.annotate(f"low {low:.0f}", (low / 10, ypos), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=7)
            ax.annotate(f"high {high:.0f}", (high / 10, ypos), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=7)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Waiting proxy\n(÷10 scale)", "Peak queue"])
    ax.set_xlabel("Median level (S0 day medians)", fontsize=8)
    style_axes(ax, "B  Low→high intensity shift (S0)")

    # A3 intensity strip
    ax = axes[1, 0]
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"], utc=True)
    ax.fill_between(d["date"], d["I"], color="#D6EAF8", lw=0)
    ax.plot(d["date"], d["I"], color="#1B4F72", lw=0.9)
    for reg, col in [("high", COLORS["high"]), ("low", COLORS["low"])]:
        sub = res[res.regime == reg][["week_end"]].drop_duplicates()
        # mark sample days
        sample = res[(res.regime == reg) & (res.scenario == "S0")].groupby("date")["I"].first()
        ax.scatter(pd.to_datetime(sample.index), sample.values, s=18, color=col, zorder=3, label=f"{reg} sample days")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_ylabel("Seaside intensity $I_t$", fontsize=8)
    style_axes(ax, "C  PortWatch intensity and sampled stress days (2024)")

    # A4 MC convergence
    ax = axes[1, 1]
    ax.plot(conv["reps_used"], conv["median"], color="#1B4F72", lw=1.8, label="Median")
    ax.fill_between(conv["reps_used"], conv["p10"], conv["p90"], color="#1B4F72", alpha=0.18, label="p10–p90")
    ax.set_xlabel("Replications used", fontsize=8)
    ax.set_ylabel("S0 peak residual queue", fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    style_axes(ax, "D  Monte Carlo convergence of S0 peak queue")

    fig.suptitle("Figure 1. Seaside intensity stress envelope under uncoordinated arrivals (S0)", fontsize=11, fontweight="bold", y=1.01)
    fig.savefig(FIG / "FigA_stress_envelope.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "FigA_stress_envelope.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_b_policy(res: pd.DataFrame, improv: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(12.0, 8.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # day medians for cleaner rainclouds
    day = res.groupby(["date", "scenario"])[["peak_queue", "p90_wait_min", "waiting_proxy"]].median().reset_index()
    data_peak = {s: day.loc[day.scenario == s, "peak_queue"].to_numpy() for s in SCENARIOS}
    data_p90 = {s: day.loc[day.scenario == s, "p90_wait_min"].to_numpy() for s in SCENARIOS}
    data_proxy = {s: day.loc[day.scenario == s, "waiting_proxy"].to_numpy() for s in SCENARIOS}

    ax = fig.add_subplot(gs[0, 0])
    raincloud(ax, data_peak, SCENARIOS, COLORS, "Peak residual queue", "A  Peak residual queue by scenario", logy=True)
    ax = fig.add_subplot(gs[0, 1])
    raincloud(ax, data_p90, SCENARIOS, COLORS, "p90 wait (min)", "B  Upper-tail wait by scenario", logy=True)
    ax = fig.add_subplot(gs[0, 2])
    raincloud(ax, data_proxy, SCENARIOS, COLORS, "Waiting-time proxy", "C  Congestion waiting proxy by scenario")

    # D forest plot of relative improvement
    ax = fig.add_subplot(gs[1, 0])
    sub = improv[improv.metric == "peak_queue"].set_index("scenario").reindex([s for s in SCENARIOS if s != "S0"])
    y = np.arange(len(sub))
    ax.hlines(y, sub["p10_improvement_vs_S0"], sub["p90_improvement_vs_S0"], color="#555", lw=1.2)
    ax.scatter(sub["median_improvement_vs_S0"], y, c=[COLORS[s] for s in sub.index], s=48, zorder=3)
    ax.axvline(0, color="#999", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([SCENARIO_LABELS[s] for s in sub.index], fontsize=7)
    ax.set_xlabel("Relative improvement vs S0", fontsize=8)
    style_axes(ax, "D  Forest plot: peak-queue improvement vs S0")

    # E paired slopes S0 -> S1 and S0 -> S_OPT
    ax = fig.add_subplot(gs[1, 1])
    wide = day.pivot(index="date", columns="scenario", values="peak_queue")
    # sample up to 40 days for readability
    idx = wide.dropna(subset=["S0", "S1", "S_OPT"]).index
    if len(idx) > 40:
        idx = np.random.default_rng(1).choice(idx, 40, replace=False)
    for d0 in idx:
        ax.plot([0, 1], [wide.loc[d0, "S0"], wide.loc[d0, "S1"]], color="#AED6F1", lw=0.7, alpha=0.7)
        ax.plot([2, 3], [wide.loc[d0, "S0"], wide.loc[d0, "S_OPT"]], color="#ABEBC6", lw=0.7, alpha=0.7)
    ax.scatter([0, 1], [wide.loc[idx, "S0"].median(), wide.loc[idx, "S1"].median()], color=[COLORS["S0"], COLORS["S1"]], s=60, zorder=4)
    ax.scatter([2, 3], [wide.loc[idx, "S0"].median(), wide.loc[idx, "S_OPT"].median()], color=[COLORS["S0"], COLORS["S_OPT"]], s=60, zorder=4)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["S0", "S1", "S0", "S_OPT"], fontsize=8)
    ax.set_ylabel("Day-median peak queue", fontsize=8)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(bottom=0)
    style_axes(ax, "E  Paired day slopes: S0→S1 and S0→S_OPT")

    # F annotation of LP gap
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    s1 = float(sub.loc["S1", "median_improvement_vs_S0"])
    sopt = float(sub.loc["S_OPT", "median_improvement_vs_S0"])
    txt = (
        "F  Compliance-matched comparison\n\n"
        f"Separate median rel. imp. vs S0:\n"
        f"  S1 {s1:.2f}   S_OPT {sopt:.2f}\n"
        f"Headline gap: {sopt - s1:+.2f} (~{(sopt - s1) * 100:.0f} pp)\n\n"
        "Paired TOST on relative peak-queue\n"
        "improvement: equivalent within 3 pp.\n"
        "LP objective = peak queue only;\n"
        "waiting-proxy p90 need not dominate.\n"
        "Perfect-compliance diagnostic: SI only."
    )
    ax.text(0.02, 0.95, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8.5, family="DejaVu Sans",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9F9", edgecolor="#BFC9CA"))

    fig.suptitle("Figure 2. Comparative performance of appointment-style regimes", fontsize=11, fontweight="bold")
    fig.savefig(FIG / "FigB_policy_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "FigB_policy_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def table_r1_multiobjective(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in SCENARIOS:
        g = res[res.scenario == s]
        rows.append(
            {
                "Scenario": SCENARIO_LABELS[s],
                "Median waiting proxy": round(g["waiting_proxy"].median(), 1),
                "p10–p90 waiting proxy": f"[{g['waiting_proxy'].quantile(0.1):.0f}, {g['waiting_proxy'].quantile(0.9):.0f}]",
                "Median storage pressure": round(g["storage_pressure"].median(), 3),
                "Median deferred": round(g["deferred"].median(), 2),
                "Mean deferred": round(g["deferred"].mean(), 2),
            }
        )
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "Table_R1_waiting_storage_deferred.csv", index=False)
    return tab


def fig_c_ablation(res: pd.DataFrame, tost: pd.DataFrame, sens_s: pd.DataFrame, sens_l: pd.DataFrame, sens_t: pd.DataFrame, sens_w: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(12.2, 8.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # C1 TOST equivalence plot for peak_queue at 5pp (show all margins as bands)
    ax = fig.add_subplot(gs[0, 0])
    t5 = tost[(tost.metric == "peak_queue") & (tost.margin == 0.05)].copy()
    # order
    order = ["S2_vs_S1_rel_improvement", "S3_vs_S1_rel_improvement", "S3s_vs_S1_rel_improvement", "S_OPT_vs_S1_rel_improvement"]
    labels = ["S2−S1", "S3−S1", "S3s−S1", "S_OPT−S1"]
    t5 = t5.set_index("compare").reindex(order)
    y = np.arange(len(t5))
    ax.axvspan(-0.05, 0.05, color="#D5F5E3", alpha=0.7, label="±5 pp primary")
    ax.axvline(-0.03, color="#F5B041", ls="--", lw=0.9)
    ax.axvline(0.03, color="#F5B041", ls="--", lw=0.9, label="±3 pp")
    ax.axvline(-0.07, color="#AF7AC5", ls=":", lw=0.9)
    ax.axvline(0.07, color="#AF7AC5", ls=":", lw=0.9, label="±7 pp")
    ax.scatter(t5["mean_diff"], y, c="#1B4F72", s=55, zorder=3)
    ax.axvline(0, color="#666", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean difference in relative improvement", fontsize=8)
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    style_axes(ax, "A  TOST equivalence vs S1 (peak-queue improvement)")

    # C2 heatmap scenario x servers for peak
    ax = fig.add_subplot(gs[0, 1])
    hs = sens_s[sens_s.scenario.isin(SCENARIOS)].pivot(index="scenario", columns="n_servers", values="median_peak_queue").reindex(SCENARIOS)
    im = ax.imshow(hs.to_numpy(), aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(hs.shape[1]))
    ax.set_xticklabels([f"S={c}" for c in hs.columns], fontsize=8)
    ax.set_yticks(range(len(SCENARIOS)))
    ax.set_yticklabels(SCENARIOS, fontsize=8)
    for i in range(hs.shape[0]):
        for j in range(hs.shape[1]):
            val = hs.to_numpy()[i, j]
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7, color="#111")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    style_axes(ax, "B  Median peak queue heatmap (servers × scenario)")

    # C3 heatmap waiting proxy x servers
    ax = fig.add_subplot(gs[0, 2])
    hw = sens_s[sens_s.scenario.isin(SCENARIOS)].pivot(index="scenario", columns="n_servers", values="median_waiting_proxy").reindex(SCENARIOS)
    im = ax.imshow(hw.to_numpy(), aspect="auto", cmap="PuBuGn")
    ax.set_xticks(range(hw.shape[1]))
    ax.set_xticklabels([f"S={c}" for c in hw.columns], fontsize=8)
    ax.set_yticks(range(len(SCENARIOS)))
    ax.set_yticklabels(SCENARIOS, fontsize=8)
    for i in range(hw.shape[0]):
        for j in range(hw.shape[1]):
            ax.text(j, i, f"{hw.to_numpy()[i, j]:.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    style_axes(ax, "C  Median waiting-proxy heatmap (servers × scenario)")

    # C4 lambda0 curves
    ax = fig.add_subplot(gs[1, 0])
    for s in ["S0", "S1", "S_OPT"]:
        sub = sens_l[sens_l.scenario == s].sort_values("lambda0")
        ax.plot(sub["lambda0"], sub["median_peak_queue"], marker="o", color=COLORS[s], label=s, lw=1.5)
    ax.set_xlabel(r"Demand scale $\lambda_0$", fontsize=8)
    ax.set_ylabel("Median peak queue", fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    style_axes(ax, r"D  Sensitivity to $\lambda_0$")

    # C5 tilt
    ax = fig.add_subplot(gs[1, 1])
    for s in ["S1", "S2", "S3"]:
        sub = sens_t[sens_t.scenario == s].sort_values("tilt")
        ax.plot(sub["tilt"], sub["median_rel_improvement_peak_queue"], marker="o", color=COLORS[s], label=s, lw=1.5)
    ax.set_xlabel(r"Forecast tilt $\gamma$", fontsize=8)
    ax.set_ylabel("Median rel. improvement vs S0", fontsize=8)
    ax.set_ylim(0.5, 1.0)
    ax.legend(frameon=False, fontsize=7)
    style_axes(ax, r"E  Forecast-tilt sensitivity")

    # C6 intensity weights
    ax = fig.add_subplot(gs[1, 2])
    ax.bar(sens_w["weights"], sens_w["median_S1_improvement_vs_S0"], color="#1B4F72", alpha=0.85)
    ax.plot(sens_w["weights"], sens_w["median_S1_improvement_vs_S0"], color="#922B21", marker="D", lw=0)
    for i, row in sens_w.iterrows():
        ax.text(i, row["median_S1_improvement_vs_S0"] + 0.01, f"{row['median_S1_improvement_vs_S0']:.2f}", ha="center", fontsize=7)
    ax.set_ylabel("S1 rel. improvement vs S0", fontsize=8)
    ax.set_ylim(0, 1.05)
    style_axes(ax, "F  Intensity-weight robustness")

    fig.suptitle("Figure 3. Ablation, equivalence, and robustness", fontsize=11, fontweight="bold")
    fig.savefig(FIG / "FigC_ablation_robustness.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "FigC_ablation_robustness.pdf", bbox_inches="tight")
    plt.close(fig)


def write_si_tables(res: pd.DataFrame, improv: pd.DataFrame, wilc: pd.DataFrame, tost: pd.DataFrame, regime: pd.DataFrame) -> None:
    # SI Table: regime numeric for 5.1
    r = regime[regime.scenario == "S0"][["regime", "peak_queue", "mean_wait_min", "waiting_proxy"]].copy()
    r.columns = ["Intensity regime", "Median peak queue", "Median mean wait (min)", "Median waiting proxy"]
    r.to_csv(OUT / "SI_Table_regime_S0.csv", index=False)

    # SI Table: full KPI summary for 5.2
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


def main() -> None:
    res = pd.read_csv(OUT / "scenario_day_replications.csv")
    conv = pd.read_csv(OUT / "mc_convergence.csv")
    improv = pd.read_csv(OUT / "improvement_vs_S0.csv")
    tost = pd.read_csv(OUT / "stats_tost_equivalence.csv")
    wilc = pd.read_csv(OUT / "stats_wilcoxon_vs_S0.csv")
    regime = pd.read_csv(OUT / "summary_by_regime.csv")
    sens_s = pd.read_csv(OUT / "sensitivity_n_servers.csv")
    sens_l = pd.read_csv(OUT / "sensitivity_lambda0.csv")
    sens_t = pd.read_csv(OUT / "sensitivity_forecast_tilt.csv")
    sens_w = pd.read_csv(OUT / "sensitivity_intensity_weights.csv")

    # daily intensity for panel C
    root = Path(__file__).resolve().parents[1]
    daily = pd.read_csv(root / "external_data" / "tema_portwatch_daily_2019_2026.csv")
    daily["date"] = pd.to_datetime(daily["date"], utc=True)
    daily = daily[daily["date"].dt.year == 2024].copy()
    # rebuild I consistently with EPS
    eps = 1e-9
    c = daily["portcalls_container"].astype(float)
    v = (daily["import_container"] + daily["export_container"]).astype(float)
    daily["I"] = 0.6 * (c - c.min()) / (c.max() - c.min() + eps) + 0.4 * (v - v.min()) / (v.max() - v.min() + eps)

    print("Building Figure A...")
    fig_a_stress(res, conv, daily)
    print("Building Figure B...")
    fig_b_policy(res, improv)
    print("Building Table R1...")
    table_r1_multiobjective(res)
    print("Building Figure C...")
    fig_c_ablation(res, tost, sens_s, sens_l, sens_t, sens_w)
    print("Writing SI tables...")
    write_si_tables(res, improv, wilc, tost, regime)
    print("Done ->", FIG)


if __name__ == "__main__":
    main()
