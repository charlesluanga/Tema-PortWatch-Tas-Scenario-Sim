"""Generate publication figures from existing revision outputs (no full re-sim)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs_revision"
FIG = Path(__file__).resolve().parent / "figures_revision"
FIG.mkdir(parents=True, exist_ok=True)

SCENARIOS = ["S0", "S1", "S2", "S3", "S3s", "S_OPT"]
COLORS = {
    "S0": "#7a1f1f",
    "S1": "#1f4e79",
    "S2": "#2e75b6",
    "S3": "#5b9bd5",
    "S3s": "#9dc3e6",
    "S_OPT": "#548235",
}


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig1_peak():
    res = pd.read_csv(OUT / "scenario_day_replications.csv")
    summary = (
        res.groupby("scenario")["peak_queue"]
        .agg(median="median", p10=lambda x: x.quantile(0.10), p90=lambda x: x.quantile(0.90))
        .reindex(SCENARIOS)
    )
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = np.arange(len(SCENARIOS))
    ax.bar(x, summary["median"], color=[COLORS[s] for s in SCENARIOS], alpha=0.9)
    ax.errorbar(
        x,
        summary["median"],
        yerr=[summary["median"] - summary["p10"], summary["p90"] - summary["median"]],
        fmt="none",
        ecolor="#222",
        capsize=4,
        lw=1.0,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIOS)
    ax.set_ylabel("Peak post-service queue (trucks)")
    ax.set_title("Peak residual queue by scenario (median, p10–p90)")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(FIG / "Fig1_peak_queue_by_scenario.png", dpi=220)
    fig.savefig(FIG / "Fig1_peak_queue_by_scenario.pdf")
    plt.close(fig)


def fig2_improvement():
    imp = pd.read_csv(OUT / "improvement_vs_S0.csv")
    metrics = ["peak_queue", "p90_wait_min", "waiting_proxy"]
    labels = ["Peak queue", "p90 wait", "Waiting proxy"]
    scenarios = ["S1", "S2", "S3", "S3s", "S_OPT"]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    x = np.arange(len(metrics))
    width = 0.15
    for i, s in enumerate(scenarios):
        vals = []
        for m in metrics:
            row = imp[(imp.scenario == s) & (imp.metric == m)]
            vals.append(float(row["median_improvement_vs_S0"].iloc[0]) if len(row) else np.nan)
        ax.bar(x + (i - 2) * width, vals, width=width, label=s, color=COLORS[s], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Median relative improvement vs S0")
    ax.set_ylim(0, 1.05)
    ax.set_title("Day-matched improvements versus uncoordinated arrivals")
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(FIG / "Fig2_improvement_vs_S0.png", dpi=220)
    fig.savefig(FIG / "Fig2_improvement_vs_S0.pdf")
    plt.close(fig)


def fig3_regime():
    reg = pd.read_csv(OUT / "summary_by_regime.csv")
    order = ["low", "normal", "high"]
    scenarios = ["S0", "S1", "S2", "S3", "S3s", "S_OPT"]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(order))
    width = 0.13
    for i, s in enumerate(scenarios):
        vals = []
        for r in order:
            row = reg[(reg.scenario == s) & (reg.regime == r)]
            vals.append(float(row["peak_queue"].iloc[0]) if len(row) else np.nan)
        ax.bar(x + (i - 2.5) * width, vals, width=width, label=s, color=COLORS[s], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(["Low-intensity weeks", "Normal days", "High-intensity weeks"])
    ax.set_ylabel("Median peak post-service queue (trucks)")
    ax.set_title("Intensity-regime stress contrast by scenario")
    ax.legend(frameon=False, ncol=6, loc="upper left", fontsize=8)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(FIG / "Fig3_regime_peak_queue.png", dpi=220)
    fig.savefig(FIG / "Fig3_regime_peak_queue.pdf")
    plt.close(fig)


def fig_mc():
    conv = pd.read_csv(OUT / "mc_convergence.csv")
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(conv["reps_used"], conv["median"], label="median", color="#1f4e79")
    ax.fill_between(conv["reps_used"], conv["p10"], conv["p90"], color="#1f4e79", alpha=0.15, label="p10–p90")
    ax.set_xlabel("Replications used")
    ax.set_ylabel("Peak queue (S0 pooled)")
    ax.set_title("Monte Carlo convergence diagnostic")
    ax.legend(frameon=False)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(FIG / "Fig_MC_convergence.png", dpi=220)
    plt.close(fig)


def main():
    fig1_peak()
    fig2_improvement()
    fig3_regime()
    fig_mc()
    print("Wrote figures to", FIG)


if __name__ == "__main__":
    main()
