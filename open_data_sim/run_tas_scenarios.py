"""
Open-data TAS scenario simulation for Tema Port (S0–S3s).

PortWatch daily intensity drives demand. Proprietary gate/GPS data are not used.
Monte Carlo uncertainty over compliance, service rate, forecast error, storage.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "external_data"
OUT = Path(__file__).resolve().parent / "outputs"
FIG = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

N_HOURS = 16
N_SERVERS = 3
LAMBDA0 = 1400
REPS = 40
RNG_MASTER = 42
SCENARIOS = ["S0", "S1", "S2", "S3", "S3s"]


def load_intensity() -> pd.DataFrame:
    d = pd.read_csv(DATA / "tema_portwatch_daily_2019_2026.csv")
    d["date"] = pd.to_datetime(d["date"], utc=True)
    d = d[d["date"].dt.year == 2024].copy().sort_values("date")
    c = d["portcalls_container"].astype(float)
    v = (d["import_container"] + d["export_container"]).astype(float)
    c_n = (c - c.min()) / (c.max() - c.min() + 1e-9)
    v_n = (v - v.min()) / (v.max() - v.min() + 1e-9)
    d["I"] = 0.6 * c_n + 0.4 * v_n
    d["N"] = np.maximum(200, np.round(LAMBDA0 * d["I"] / d["I"].mean())).astype(int)
    days_to_sunday = (6 - d["date"].dt.dayofweek) % 7
    d["week_end"] = (d["date"] + pd.to_timedelta(days_to_sunday, unit="D")).dt.strftime("%Y-%m-%d")
    d["I_hat_base"] = d["I"].shift(1).rolling(7, min_periods=1).mean().fillna(d["I"].mean())
    return d.reset_index(drop=True)


def peak_weights(rng: np.random.Generator) -> np.ndarray:
    x = np.linspace(0, 1, N_HOURS)
    w = 0.62 * np.exp(-0.5 * ((x - 0.22) / 0.09) ** 2) + 0.38 * np.exp(
        -0.5 * ((x - 0.70) / 0.10) ** 2
    )
    w = np.clip(w, 1e-6, None)
    w = w * rng.lognormal(0, 0.05, size=N_HOURS)
    return w / w.sum()


def uniform_weights(rng: np.random.Generator | None = None) -> np.ndarray:
    w = np.ones(N_HOURS)
    if rng is not None:
        w = w * rng.lognormal(0, 0.03, size=N_HOURS)
    return w / w.sum()


def forecast_weights(I_hat: float, I_mean: float, rng: np.random.Generator) -> np.ndarray:
    # Mild bowl shape: slightly more midday capacity when intensity high
    base = uniform_weights(rng)
    intensity_factor = np.clip(I_hat / (I_mean + 1e-9), 0.6, 1.8)
    x = np.linspace(-1, 1, N_HOURS)
    shape = 1.0 + 0.35 * (intensity_factor - 1.0) * (1.0 - x**2)
    w = base * np.clip(shape, 0.4, None)
    return w / w.sum()


def allocate(N: int, weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.multinomial(int(N), weights)


def mix_compliance(
    planned: np.ndarray, peak: np.ndarray, compliance: float
) -> np.ndarray:
    out = np.round(compliance * planned + (1.0 - compliance) * peak).astype(int)
    diff = int(planned.sum() - out.sum())
    out[N_HOURS // 2] += diff
    return np.maximum(out, 0)


def simulate_day(
    arrivals: np.ndarray,
    mu_per_server: float,
    storage_cap: float,
    hard_storage: bool,
) -> dict:
    """Hourly discrete-event queue with carryover."""
    service_per_hour = mu_per_server * N_SERVERS * 60.0  # trucks/hour
    queue = 0.0
    storage = 0.40 * storage_cap
    waits = []
    queues = []
    served_total = 0.0
    deferred = 0.0
    corridor_acc = 0.0
    emit_acc = 0.0
    storage_acc = 0.0
    overloaded_hours = 0

    for a in arrivals.astype(float):
        a_eff = a
        if hard_storage:
            free = max(0.0, storage_cap - storage)
            if a_eff > free:
                deferred += a_eff - free
                a_eff = free

        queue += a_eff
        # Approximate waiting time for trucks arriving this hour
        wait_min = 60.0 * queue / max(service_per_hour, 1e-6)
        waits.append(wait_min)
        queues.append(queue)
        if a_eff > service_per_hour:
            overloaded_hours += 1

        served = min(queue, service_per_hour)
        queue -= served
        served_total += served

        storage += 0.60 * served
        storage -= 0.52 * served
        storage = float(np.clip(storage, 0.0, 1.8 * storage_cap))
        storage_acc += storage / storage_cap

        util = (served + queue) / max(service_per_hour, 1e-6)
        corridor = 1.0 + 2.2 * max(util - 0.7, 0.0) ** 1.25
        corridor_acc += corridor
        emit_acc += served * (0.35 + 0.65 * min(wait_min, 90.0) / 45.0) + queue * 0.2

    return {
        "mean_wait_min": float(np.mean(waits)),
        "p90_wait_min": float(np.quantile(waits, 0.90)),
        "peak_queue": float(np.max(queues)),
        "corridor_delay_index": float(corridor_acc / N_HOURS),
        "throughput": float(served_total),
        "emissions_proxy": float(emit_acc),
        "storage_pressure": float(storage_acc / N_HOURS),
        "deferred": float(deferred),
        "overloaded_hours": float(overloaded_hours),
        "final_queue": float(queue),
    }


def mpc_plan(N: int, I_hat: float, I_mean: float, rng: np.random.Generator) -> np.ndarray:
    """Two-stage plan: forecast weights, then flatten residual after mid-day queue signal."""
    w = forecast_weights(I_hat, I_mean, rng)
    first_plan = allocate(N, w, rng)
    # Simulate first half under nominal mu to get queue pressure
    mu_nom = 1.55
    service = mu_nom * N_SERVERS * 60.0
    q = 0.0
    mid = N_HOURS // 2
    for a in first_plan[:mid]:
        q += a
        q = max(0.0, q - service)
    residual = int(first_plan[mid:].sum())
    # Reallocate residual more uniformly if queue already high; else keep forecast shape
    if q > 0.35 * service:
        w2 = uniform_weights(rng)
    else:
        w2 = forecast_weights(I_hat, I_mean, rng)
    w2 = w2[: N_HOURS - mid]
    w2 = w2 / w2.sum()
    second = rng.multinomial(residual, w2)
    return np.concatenate([first_plan[:mid], second])


def run_one(
    scenario: str,
    N: int,
    I: float,
    I_mean: float,
    I_hat: float,
    rng: np.random.Generator,
) -> dict:
    compliance = float(rng.uniform(0.75, 0.95))
    # Service deliberately in congestible range relative to peaked arrivals
    mu = float(np.clip(rng.normal(1.55, 0.18), 1.15, 2.10))
    storage_cap = 1000.0 * float(rng.choice([0.80, 1.00, 1.20]))

    peak = allocate(N, peak_weights(rng), rng)

    if scenario == "S0":
        arrivals = peak
        hard = False
    elif scenario == "S1":
        planned = allocate(N, uniform_weights(rng), rng)
        arrivals = mix_compliance(planned, peak, compliance)
        hard = False
    elif scenario == "S2":
        planned = allocate(N, forecast_weights(I_hat, I_mean, rng), rng)
        arrivals = mix_compliance(planned, peak, compliance)
        hard = False
    elif scenario == "S3":
        planned = mpc_plan(N, I_hat, I_mean, rng)
        arrivals = mix_compliance(planned, peak, compliance)
        hard = False
    elif scenario == "S3s":
        planned = mpc_plan(N, I_hat, I_mean, rng)
        arrivals = mix_compliance(planned, peak, compliance)
        hard = True
    else:
        raise ValueError(scenario)

    metrics = simulate_day(arrivals, mu, storage_cap, hard)
    metrics.update(
        {
            "scenario": scenario,
            "compliance": compliance,
            "mu": mu,
            "storage_cap": storage_cap,
            "N": N,
            "I": I,
            "I_hat": I_hat,
        }
    )
    return metrics


def summarize(res: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = [
        "mean_wait_min",
        "p90_wait_min",
        "peak_queue",
        "corridor_delay_index",
        "throughput",
        "emissions_proxy",
        "storage_pressure",
        "deferred",
        "overloaded_hours",
    ]
    summary_rows = []
    for s, g in res.groupby("scenario"):
        for col in metrics:
            summary_rows.append(
                {
                    "scenario": s,
                    "metric": col,
                    "median": g[col].median(),
                    "p10": g[col].quantile(0.10),
                    "p90": g[col].quantile(0.90),
                    "mean": g[col].mean(),
                }
            )
    summary = pd.DataFrame(summary_rows)

    base = (
        res[res.scenario == "S0"].groupby("date")[metrics].median().add_prefix("S0_")
    )
    improv_rows = []
    for s in ["S1", "S2", "S3", "S3s"]:
        cur = res[res.scenario == s].groupby("date")[metrics].median()
        merged = cur.join(base, how="inner")
        for col in ["mean_wait_min", "p90_wait_min", "peak_queue", "corridor_delay_index", "emissions_proxy"]:
            rel = (merged[f"S0_{col}"] - merged[col]) / merged[f"S0_{col}"].replace(0, np.nan)
            improv_rows.append(
                {
                    "scenario": s,
                    "metric": col,
                    "median_improvement_vs_S0": float(rel.median()),
                    "p10_improvement_vs_S0": float(rel.quantile(0.10)),
                    "p90_improvement_vs_S0": float(rel.quantile(0.90)),
                }
            )
    improv = pd.DataFrame(improv_rows)
    regime = res.groupby(["scenario", "regime"])[metrics].median().reset_index()
    return summary, improv, regime


def make_figures(summary: pd.DataFrame, improv: pd.DataFrame, res: pd.DataFrame) -> None:
    # Figure 1: median peak queue by scenario with p10-p90
    metric = "peak_queue"
    sub = summary[summary.metric == metric].set_index("scenario").reindex(SCENARIOS)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(SCENARIOS))
    ax.bar(x, sub["median"], color="#1f4e79", alpha=0.85)
    ax.errorbar(
        x,
        sub["median"],
        yerr=[sub["median"] - sub["p10"], sub["p90"] - sub["median"]],
        fmt="none",
        ecolor="#333333",
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIOS)
    ax.set_ylabel("Peak gate queue (trucks)")
    ax.set_title("Tema open-data TAS scenarios: peak queue (median, p10–p90)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "Fig_peak_queue_by_scenario.png", dpi=200)
    fig.savefig(FIG / "Fig_peak_queue_by_scenario.pdf")
    plt.close(fig)

    # Figure 2: improvement vs S0
    focus = improv[improv.metric.isin(["peak_queue", "mean_wait_min", "emissions_proxy"])]
    pivot = focus.pivot(index="scenario", columns="metric", values="median_improvement_vs_S0").reindex(
        ["S1", "S2", "S3", "S3s"]
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    pivot.plot(kind="bar", ax=ax, color=["#1f4e79", "#9c2a2a", "#2f6f4e"])
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_ylabel("Median relative improvement vs S0")
    ax.set_title("Ablation gains relative to uncoordinated arrivals")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "Fig_improvement_vs_S0.png", dpi=200)
    fig.savefig(FIG / "Fig_improvement_vs_S0.pdf")
    plt.close(fig)

    # Figure 3: high vs low regime peak queue
    reg = (
        res[res.regime.isin(["high", "low"])]
        .groupby(["scenario", "regime"])["peak_queue"]
        .median()
        .unstack("regime")
        .reindex(SCENARIOS)
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    reg[["low", "high"]].plot(kind="bar", ax=ax, color=["#6f8f6f", "#8b3a3a"])
    ax.set_ylabel("Median peak queue (trucks)")
    ax.set_title("Stress-test weeks: low vs high PortWatch intensity")
    ax.legend(title="Regime", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "Fig_regime_peak_queue.png", dpi=200)
    fig.savefig(FIG / "Fig_regime_peak_queue.pdf")
    plt.close(fig)


def main() -> None:
    df = load_intensity()
    I_mean = float(df["I"].mean())

    high_ends = {"2024-10-27", "2024-08-11", "2024-09-01", "2024-12-15", "2024-07-21"}
    low_ends = {"2024-02-11", "2024-03-17", "2024-02-04", "2024-01-21"}

    sample_idx: list[int] = []
    for _, g in df.groupby(df["date"].dt.month):
        sample_idx.extend(list(g.index[np.linspace(0, len(g) - 1, 3).astype(int)]))
    stress_idx = df.index[df["week_end"].isin(high_ends | low_ends)].tolist()
    use_idx = sorted(set(sample_idx) | set(stress_idx))
    days = df.loc[use_idx].reset_index(drop=True)

    rows = []
    for i, row in days.iterrows():
        regime = (
            "high"
            if row["week_end"] in high_ends
            else ("low" if row["week_end"] in low_ends else "normal")
        )
        for s in SCENARIOS:
            for r in range(REPS):
                rng = np.random.default_rng(
                    RNG_MASTER + int(i) * 100_000 + SCENARIOS.index(s) * 1_000 + r
                )
                I_hat = float(row["I_hat_base"] * rng.lognormal(0, 0.15))
                m = run_one(s, int(row["N"]), float(row["I"]), I_mean, I_hat, rng)
                m.update(
                    {
                        "date": row["date"].strftime("%Y-%m-%d"),
                        "month": int(row["date"].month),
                        "rep": r,
                        "week_end": row["week_end"],
                        "regime": regime,
                    }
                )
                rows.append(m)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "scenario_day_replications.csv", index=False)
    summary, improv, regime = summarize(res)
    summary.to_csv(OUT / "summary_by_scenario.csv", index=False)
    improv.to_csv(OUT / "improvement_vs_S0.csv", index=False)
    regime.to_csv(OUT / "summary_by_regime.csv", index=False)

    wide = summary.pivot(index="metric", columns="scenario", values="median")
    wide = wide.reindex(columns=SCENARIOS)
    wide.to_csv(OUT / "Table_Results_Median_by_Scenario.csv")

    # bands table for manuscript
    band_rows = []
    for metric in ["mean_wait_min", "peak_queue", "corridor_delay_index", "emissions_proxy"]:
        for s in SCENARIOS:
            g = summary[(summary.metric == metric) & (summary.scenario == s)].iloc[0]
            band_rows.append(
                {
                    "metric": metric,
                    "scenario": s,
                    "median": round(g["median"], 3),
                    "p10": round(g["p10"], 3),
                    "p90": round(g["p90"], 3),
                }
            )
    pd.DataFrame(band_rows).to_csv(OUT / "Table_Results_Uncertainty_Bands.csv", index=False)

    make_figures(summary, improv, res)

    meta = {
        "lambda0": LAMBDA0,
        "reps": REPS,
        "n_days": int(days.shape[0]),
        "n_rows": int(len(res)),
        "scenarios": SCENARIOS,
        "seed": RNG_MASTER,
    }
    (OUT / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Days:", days.shape[0], "Rows:", len(res))
    print("\nMedian peak_queue:")
    print(res.groupby("scenario")["peak_queue"].median().reindex(SCENARIOS).round(1).to_string())
    print("\nMedian mean_wait_min:")
    print(res.groupby("scenario")["mean_wait_min"].median().reindex(SCENARIOS).round(2).to_string())
    print("\nImprovement vs S0 (peak_queue):")
    print(
        improv[improv.metric == "peak_queue"][
            ["scenario", "median_improvement_vs_S0", "p10_improvement_vs_S0", "p90_improvement_vs_S0"]
        ].to_string(index=False)
    )
    print("Wrote", OUT)
    print("Figures", FIG)


if __name__ == "__main__":
    main()
