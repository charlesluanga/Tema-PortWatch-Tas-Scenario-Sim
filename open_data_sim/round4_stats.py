#!/usr/bin/env python3
"""Round-4 post-hoc stats from existing outputs (no full re-simulation)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

OUT = Path(__file__).resolve().parent / "outputs_revision"
SCENARIOS = ["S1", "S2", "S3", "S3s", "S_OPT"]


def rank_biserial(diffs: np.ndarray) -> tuple[float, int, int, int]:
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    n_pos = int(np.sum(d > 0))
    n_neg = int(np.sum(d < 0))
    n_tie = int(np.sum(d == 0))
    n_nt = n_pos + n_neg
    r = (n_pos - n_neg) / n_nt if n_nt else float("nan")
    return r, n_pos, n_neg, n_tie


def tost_wilcoxon(diffs: np.ndarray, margin: float) -> float:
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    # exclude exact zeros for wilcoxon stability when many ties at 0
    _, p_lo = wilcoxon(d - (-margin), alternative="greater", zero_method="wilcox")
    _, p_hi = wilcoxon(d - margin, alternative="less", zero_method="wilcox")
    return float(max(p_lo, p_hi))


def main() -> None:
    res = pd.read_csv(OUT / "scenario_day_replications.csv")
    day_peak = res.groupby(["date", "scenario"])["peak_queue"].median().unstack()
    day_proxy = res.groupby(["date", "scenario"])["waiting_proxy"].median().unstack()

    # --- D4: tie counts behind r=1.0 ---
    tie_rows = []
    for s in SCENARIOS:
        diffs = (day_peak["S0"] - day_peak[s]).to_numpy()
        r, n_pos, n_neg, n_tie = rank_biserial(diffs)
        tie_rows.append(
            {
                "level": "day",
                "metric": "peak_queue",
                "scenario": s,
                "n_pairs": len(diffs),
                "n_pos_S0_higher": n_pos,
                "n_neg": n_neg,
                "n_tied": n_tie,
                "rank_biserial_r_nontied": r,
            }
        )
    pd.DataFrame(tie_rows).to_csv(OUT / "stats_rank_biserial_ties.csv", index=False)

    # --- D3: S1 vs S_OPT ---
    mask = day_peak["S0"] > 0
    imp1 = (day_peak.loc[mask, "S0"] - day_peak.loc[mask, "S1"]) / day_peak.loc[mask, "S0"]
    impo = (day_peak.loc[mask, "S0"] - day_peak.loc[mask, "S_OPT"]) / day_peak.loc[mask, "S0"]
    gap = (impo - imp1).dropna()
    peak_diff = (day_peak["S1"] - day_peak["S_OPT"]).dropna()
    proxy_gap_rel = (
        (day_proxy["S0"] - day_proxy["S_OPT"]) / day_proxy["S0"]
        - (day_proxy["S0"] - day_proxy["S1"]) / day_proxy["S0"]
    ).dropna()

    _, p_gap = wilcoxon(gap.to_numpy(), alternative="two-sided", zero_method="wilcox")
    _, p_peak = wilcoxon(peak_diff.to_numpy(), alternative="two-sided", zero_method="wilcox")

    s1_opt = [
        {
            "contrast": "relative_peak_improvement_S_OPT_minus_S1",
            "n_days": int(len(gap)),
            "median_S1_rel_imp": float(imp1.median()),
            "median_S_OPT_rel_imp": float(impo.median()),
            "separate_median_gap_pp": float((impo.median() - imp1.median()) * 100),
            "paired_mean_gap": float(gap.mean()),
            "paired_median_gap": float(gap.median()),
            "wilcoxon_two_sided_p": float(p_gap),
            "tost_p_3pp": tost_wilcoxon(gap.to_numpy(), 0.03),
            "tost_p_5pp": tost_wilcoxon(gap.to_numpy(), 0.05),
            "tost_equivalent_3pp": tost_wilcoxon(gap.to_numpy(), 0.03) < 0.05,
            "tost_equivalent_5pp": tost_wilcoxon(gap.to_numpy(), 0.05) < 0.05,
        },
        {
            "contrast": "day_median_peak_S1_minus_S_OPT",
            "n_days": int(len(peak_diff)),
            "paired_mean": float(peak_diff.mean()),
            "paired_median": float(peak_diff.median()),
            "frac_S_OPT_lower_peak": float((peak_diff > 0).mean()),
            "frac_equal": float((peak_diff == 0).mean()),
            "frac_S_OPT_higher_peak": float((peak_diff < 0).mean()),
            "wilcoxon_two_sided_p": float(p_peak),
        },
        {
            "contrast": "relative_waiting_proxy_improvement_S_OPT_minus_S1",
            "n_days": int(len(proxy_gap_rel)),
            "paired_mean_gap": float(proxy_gap_rel.mean()),
            "paired_median_gap": float(proxy_gap_rel.median()),
            "tost_p_5pp": tost_wilcoxon(proxy_gap_rel.to_numpy(), 0.05),
            "tost_equivalent_5pp": tost_wilcoxon(proxy_gap_rel.to_numpy(), 0.05) < 0.05,
        },
    ]
    pd.DataFrame(s1_opt).to_csv(OUT / "stats_s1_vs_sopt.csv", index=False)

    # --- D2: waiting-proxy anomaly diagnostics ---
    wide = res.pivot_table(index=["date", "rep"], columns="scenario", values="waiting_proxy")
    cmp = wide[["S0", "S1", "S_OPT"]].dropna()
    anom = pd.DataFrame(
        [
            {
                "frac_S_OPT_waiting_gt_S0": float((cmp["S_OPT"] > cmp["S0"]).mean()),
                "frac_S1_waiting_gt_S0": float((cmp["S1"] > cmp["S0"]).mean()),
                "frac_S_OPT_waiting_gt_S1": float((cmp["S_OPT"] > cmp["S1"]).mean()),
                "S0_p90": float(res.loc[res.scenario == "S0", "waiting_proxy"].quantile(0.9)),
                "S1_p90": float(res.loc[res.scenario == "S1", "waiting_proxy"].quantile(0.9)),
                "S_OPT_p90": float(res.loc[res.scenario == "S_OPT", "waiting_proxy"].quantile(0.9)),
                "S0_median": float(res.loc[res.scenario == "S0", "waiting_proxy"].median()),
                "S_OPT_median": float(res.loc[res.scenario == "S_OPT", "waiting_proxy"].median()),
            }
        ]
    )
    anom.to_csv(OUT / "stats_sopt_waiting_proxy_anomaly.csv", index=False)

    # --- D10: S-server table already exists; write SI-friendly wide form ---
    sens = pd.read_csv(OUT / "sensitivity_n_servers.csv")
    peak_w = sens.pivot(index="scenario", columns="n_servers", values="median_peak_queue")
    proxy_w = sens.pivot(index="scenario", columns="n_servers", values="median_waiting_proxy")
    peak_w.columns = [f"median_peak_S{c}" for c in peak_w.columns]
    proxy_w.columns = [f"median_proxy_S{c}" for c in proxy_w.columns]
    out = peak_w.join(proxy_w).reset_index()
    out.to_csv(OUT / "SI_Table_servers_S1_S2_S3.csv", index=False)

    print("Wrote Round-4 stats to", OUT)


if __name__ == "__main__":
    main()
