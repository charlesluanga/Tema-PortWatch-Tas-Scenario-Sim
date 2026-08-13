"""
Open-data truck appointment scenario experiments.

- Distinct documented defaults for θ, γ, δ
- S_OPT LP planned on expected capacity E[C]=60·μ̄·S (no foresight of μ draw)
- Co-primary effective capacity levels S ∈ {1,2,3}
- Compliance-matched S_OPT; Wilcoxon/Holm/TOST; shared Monte Carlo shocks
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "external_data"
OUT = Path(__file__).resolve().parent / "outputs_revision"
FIG = Path(__file__).resolve().parent / "figures_revision"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# Documented parameters — distinct defaults (E1)
EPS = 1e-9
THETA = 0.50  # residual reallocation when mid-queue > 50% of hourly capacity
GAMMA_TILT = 0.20  # mild forecast tilt (20% relative reweighting scale)
DELTA_CLEAR = 0.25  # yard clears 25% of occupancy per hour
ALPHA_IN = 0.55  # yard intake (stated assumption)
BPR_ALPHA = 0.15  # textbook BPR
BPR_BETA = 4.0
MU_MEAN = 1.55  # planning mean for LP (E2: no foresight of realised μ)
N_HOURS = 16
N_SERVERS = 1  # scarce-capacity stress calibration; co-primary S∈{1,2,3}
LAMBDA0_DEFAULT = 1400
REPS = 40
RNG_MASTER = 42
SCENARIOS = ["S0", "S1", "S2", "S3", "S3s", "S_OPT"]


def load_intensity(year: int = 2024) -> pd.DataFrame:
    d = pd.read_csv(DATA / "tema_portwatch_daily_2019_2026.csv")
    d["date"] = pd.to_datetime(d["date"], utc=True)
    d = d[d["date"].dt.year == year].copy().sort_values("date")
    c = d["portcalls_container"].astype(float)
    v = (d["import_container"] + d["export_container"]).astype(float)
    c_n = (c - c.min()) / (c.max() - c.min() + EPS)
    v_n = (v - v.min()) / (v.max() - v.min() + EPS)
    d["I"] = 0.6 * c_n + 0.4 * v_n
    days_to_sunday = (6 - d["date"].dt.dayofweek) % 7
    d["week_end"] = (d["date"] + pd.to_timedelta(days_to_sunday, unit="D")).dt.strftime("%Y-%m-%d")
    d["I_hat_base"] = d["I"].shift(1).rolling(7, min_periods=1).mean().fillna(d["I"].mean())
    return d.reset_index(drop=True)


def intensity_with_weights(d: pd.DataFrame, w_c: float, w_v: float) -> pd.Series:
    c = d["portcalls_container"].astype(float)
    v = (d["import_container"] + d["export_container"]).astype(float)
    c_n = (c - c.min()) / (c.max() - c.min() + EPS)
    v_n = (v - v.min()) / (v.max() - v.min() + EPS)
    return w_c * c_n + w_v * v_n


def peak_weights(rng: np.random.Generator) -> np.ndarray:
    x = np.linspace(0, 1, N_HOURS)
    w = 0.62 * np.exp(-0.5 * ((x - 0.22) / 0.09) ** 2) + 0.38 * np.exp(
        -0.5 * ((x - 0.70) / 0.10) ** 2
    )
    w = np.clip(w, 1e-6, None) * rng.lognormal(0, 0.05, size=N_HOURS)
    return w / w.sum()


def uniform_weights(rng: np.random.Generator | None = None) -> np.ndarray:
    w = np.ones(N_HOURS)
    if rng is not None:
        w = w * rng.lognormal(0, 0.03, size=N_HOURS)
    return w / w.sum()


def forecast_weights(I_hat: float, I_mean: float, rng: np.random.Generator, tilt: float = GAMMA_TILT) -> np.ndarray:
    base = uniform_weights(rng)
    intensity_factor = np.clip(I_hat / (I_mean + EPS), 0.6, 1.8)
    x = np.linspace(-1, 1, N_HOURS)
    shape = 1.0 + tilt * (intensity_factor - 1.0) * (1.0 - x**2)
    w = base * np.clip(shape, 0.4, None)
    return w / w.sum()


def allocate(N: int, weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.multinomial(int(N), weights)


def mix_compliance(planned: np.ndarray, peak: np.ndarray, compliance: float) -> np.ndarray:
    out = np.round(compliance * planned + (1.0 - compliance) * peak).astype(int)
    diff = int(planned.sum() - out.sum())
    out[N_HOURS // 2] += diff
    return np.maximum(out, 0)


def _uniform_int_arrivals(N: int) -> np.ndarray:
    base = np.full(N_HOURS, N // N_HOURS)
    base[: N % N_HOURS] += 1
    return base.astype(int)


def lp_optimal_arrivals(N: int, mu_plan: float, n_servers: int) -> np.ndarray:
    """LP plan under planning service rate mu_plan (use MU_MEAN for no foresight)."""
    C = 60.0 * mu_plan * n_servers
    H = N_HOURS
    n = 2 * H + 1
    c = np.zeros(n)
    c[-1] = 1.0
    A_eq = np.zeros((1, n))
    A_eq[0, :H] = 1.0
    b_eq = np.array([float(N)])
    rows, rhs = [], []
    for h in range(H):
        r = np.zeros(n)
        r[h] = -1.0
        r[H + h] = 1.0
        if h > 0:
            r[H + h - 1] = -1.0
        rows.append(r)
        rhs.append(-C)
        r2 = np.zeros(n)
        r2[H + h] = 1.0
        r2[-1] = -1.0
        rows.append(r2)
        rhs.append(0.0)
    res = linprog(
        c,
        A_ub=np.vstack(rows),
        b_ub=np.array(rhs, dtype=float),
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(0, float(N))] * H + [(0, None)] * H + [(0, None)],
        method="highs",
        options={"presolve": True},
    )
    if not res.success or res.x is None or not np.isfinite(res.x[:H]).all():
        return _uniform_int_arrivals(N)
    a = np.maximum(res.x[:H], 0.0)
    a_int = np.floor(a).astype(int)
    rem = int(N - a_int.sum())
    if rem < 0:
        return _uniform_int_arrivals(N)
    frac_order = np.argsort(-(a - a_int))
    for i in range(rem):
        a_int[frac_order[i % H]] += 1
    return a_int


def simulate_day(arrivals: np.ndarray, mu: float, storage_cap: float, hard_storage: bool, n_servers: int) -> dict:
    C = 60.0 * mu * n_servers
    queue = 0.0
    storage = 0.40 * storage_cap
    waits, post_queues = [], []
    served_total = deferred = corridor_acc = proxy_acc = storage_acc = 0.0

    for a in arrivals.astype(float):
        a_eff = a
        if hard_storage:
            free = max(0.0, storage_cap - storage)
            if a_eff > free:
                deferred += a_eff - free
                a_eff = free
        load = queue + a_eff
        wait_min = 60.0 * load / max(C, EPS)
        waits.append(wait_min)
        served = min(load, C)
        queue = max(0.0, load - C)
        post_queues.append(queue)
        served_total += served
        storage = (1.0 - DELTA_CLEAR) * storage + ALPHA_IN * served
        storage = float(np.clip(storage, 0.0, 1.8 * storage_cap))
        storage_acc += storage / storage_cap
        voc = load / max(C, EPS)
        corridor_acc += 1.0 + BPR_ALPHA * (voc**BPR_BETA)
        proxy_acc += served * (0.35 + 0.65 * min(wait_min, 90.0) / 45.0) + 0.2 * queue

    return {
        "mean_wait_min": float(np.mean(waits)),
        "p90_wait_min": float(np.quantile(waits, 0.90)),
        "peak_queue": float(np.max(post_queues)) if post_queues else 0.0,
        "corridor_delay_index": float(corridor_acc / N_HOURS),
        "throughput": float(served_total),
        "waiting_proxy": float(proxy_acc),
        "storage_pressure": float(storage_acc / N_HOURS),
        "deferred": float(deferred),
        "final_queue": float(queue),
        "hourly_capacity": float(C),
    }


def residual_reallocation(
    first: np.ndarray,
    residual_N: int,
    queue_after_first: float,
    mu: float,
    I_hat: float,
    I_mean: float,
    rng: np.random.Generator,
    n_servers: int,
    tilt: float,
) -> np.ndarray:
    C = 60.0 * mu * n_servers
    rem_h = N_HOURS - len(first)
    if rem_h <= 0:
        return first
    if queue_after_first > THETA * C:
        w = uniform_weights(rng)[:rem_h]
    else:
        w = forecast_weights(I_hat, I_mean, rng, tilt=tilt)[:rem_h]
    w = w / w.sum()
    return np.concatenate([first, rng.multinomial(int(residual_N), w)])


def run_one(
    scenario: str,
    N: int,
    I: float,
    I_mean: float,
    I_hat: float,
    rng: np.random.Generator,
    compliance: float,
    mu: float,
    storage_cap: float,
    peak: np.ndarray,
    tilt: float = GAMMA_TILT,
    n_servers: int = 1,
) -> dict:
    hard = False
    peak_perfect = np.nan
    peak_foresight = np.nan

    if scenario == "S0":
        arrivals = peak
    elif scenario == "S1":
        planned = allocate(N, uniform_weights(rng), rng)
        arrivals = mix_compliance(planned, peak, compliance)
    elif scenario == "S2":
        planned = allocate(N, forecast_weights(I_hat, I_mean, rng, tilt=tilt), rng)
        arrivals = mix_compliance(planned, peak, compliance)
    elif scenario in ("S3", "S3s"):
        planned = allocate(N, forecast_weights(I_hat, I_mean, rng, tilt=tilt), rng)
        mixed = mix_compliance(planned, peak, compliance)
        mid = N_HOURS // 2
        first = mixed[:mid].copy()
        C = 60.0 * mu * n_servers
        q = 0.0
        for a in first:
            q = max(0.0, q + a - C)
        arrivals = residual_reallocation(first, int(mixed[mid:].sum()), q, mu, I_hat, I_mean, rng, n_servers, tilt)
        hard = scenario == "S3s"
    elif scenario == "S_OPT":
        # E2: plan on expected capacity (MU_MEAN), not realised μ foresight
        planned = lp_optimal_arrivals(N, MU_MEAN, n_servers)
        # Diagnostic only: foresight LP using realised μ (not used in headlines)
        planned_foresight = lp_optimal_arrivals(N, mu, n_servers)
        peak_perfect = simulate_day(planned, mu, storage_cap, False, n_servers)["peak_queue"]
        peak_foresight = simulate_day(
            mix_compliance(planned_foresight, peak, compliance), mu, storage_cap, False, n_servers
        )["peak_queue"]
        arrivals = mix_compliance(planned, peak, compliance)
    else:
        raise ValueError(scenario)

    # Headline KPIs always from discrete recursion on realized (integer) arrivals
    metrics = simulate_day(arrivals, mu, storage_cap, hard, n_servers)
    metrics.update(
        {
            "scenario": scenario,
            "compliance": compliance,
            "mu": mu,
            "mu_plan_lp": MU_MEAN if scenario == "S_OPT" else np.nan,
            "storage_cap": storage_cap,
            "N": N,
            "I": I,
            "I_hat": I_hat,
            "theta": THETA,
            "eps": EPS,
            "gamma": tilt,
            "delta_clear": DELTA_CLEAR,
            "bpr_alpha": BPR_ALPHA,
            "bpr_beta": BPR_BETA,
            "n_servers": n_servers,
            "peak_queue_perfect_compliance": float(peak_perfect) if scenario == "S_OPT" else np.nan,
            "peak_queue_foresight_lp_diagnostic": float(peak_foresight) if scenario == "S_OPT" else np.nan,
        }
    )
    return metrics


def select_days(df: pd.DataFrame) -> pd.DataFrame:
    high_ends = {"2024-10-27", "2024-08-11", "2024-09-01", "2024-12-15", "2024-07-21"}
    low_ends = {"2024-02-11", "2024-03-17", "2024-02-04", "2024-01-21"}
    sample_idx: list[int] = []
    for _, g in df.groupby(df["date"].dt.month):
        sample_idx.extend(list(g.index[np.linspace(0, len(g) - 1, 3).astype(int)]))
    stress_idx = df.index[df["week_end"].isin(high_ends | low_ends)].tolist()
    use_idx = sorted(set(sample_idx) | set(stress_idx))
    days = df.loc[use_idx].copy()
    days["regime"] = days["week_end"].map(
        lambda w: "high" if w in high_ends else ("low" if w in low_ends else "normal")
    )
    return days.reset_index(drop=True)


def run_main_experiments(
    lambda0: float = LAMBDA0_DEFAULT,
    tilt: float = GAMMA_TILT,
    n_servers: int = 1,
) -> pd.DataFrame:
    """Shared shocks per (day, rep) across scenarios for valid paired tests (C7)."""
    df = load_intensity()
    I_mean = float(df["I"].mean())
    days = select_days(df)
    rows = []
    for i, row in days.iterrows():
        N = int(max(200, round(lambda0 * float(row["I"]) / I_mean)))
        for r in range(REPS):
            rng_shock = np.random.default_rng(RNG_MASTER + int(i) * 100_000 + r)
            compliance = float(rng_shock.uniform(0.75, 0.95))
            mu = float(np.clip(rng_shock.normal(1.55, 0.18), 1.15, 2.10))
            storage_cap = 450.0 * float(rng_shock.choice([0.80, 1.00, 1.20]))
            peak = allocate(N, peak_weights(rng_shock), rng_shock)
            I_hat = float(row["I_hat_base"] * rng_shock.lognormal(0, 0.15))
            for s in SCENARIOS:
                rng_s = np.random.default_rng(RNG_MASTER + int(i) * 100_000 + r * 10 + SCENARIOS.index(s))
                m = run_one(
                    s,
                    N,
                    float(row["I"]),
                    I_mean,
                    I_hat,
                    rng_s,
                    compliance,
                    mu,
                    storage_cap,
                    peak,
                    tilt=tilt,
                    n_servers=n_servers,
                )
                m.update(
                    {
                        "date": row["date"].strftime("%Y-%m-%d"),
                        "month": int(row["date"].month),
                        "rep": r,
                        "week_end": row["week_end"],
                        "regime": row["regime"],
                        "lambda0": lambda0,
                        "tilt": tilt,
                    }
                )
                rows.append(m)
    return pd.DataFrame(rows)


def day_medians(res: pd.DataFrame, metric: str) -> pd.DataFrame:
    return res.groupby(["date", "scenario"])[metric].median().unstack("scenario")


def rank_biserial_from_wilcoxon(diff: pd.Series) -> float:
    """Matched-pairs rank-biserial correlation from signed ranks."""
    d = diff.to_numpy(dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    r_plus = ranks[d > 0].sum()
    r_minus = ranks[d < 0].sum()
    n = len(d)
    return float((r_plus - r_minus) / (n * (n + 1) / 2.0))


def holm_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.zeros(m)
    running = 0.0
    for i, idx in enumerate(order):
        candidate = (m - i) * pvals[idx]
        running = max(running, candidate)
        adj[idx] = min(1.0, running)
    return adj.tolist()


def wilcoxon_battery(res: pd.DataFrame, metric: str, level: str = "day") -> pd.DataFrame:
    rows = []
    if level == "day":
        wide = day_medians(res, metric)
        for s in [x for x in SCENARIOS if x != "S0"]:
            if s not in wide.columns:
                continue
            diff = (wide["S0"] - wide[s]).dropna()
            if len(diff) < 5:
                continue
            stat, p = stats.wilcoxon(diff, alternative="greater", zero_method="wilcox")
            rows.append(
                {
                    "level": level,
                    "metric": metric,
                    "scenario": s,
                    "n": len(diff),
                    "median_diff": float(diff.median()),
                    "wilcoxon_stat": float(stat),
                    "p_value": float(p),
                    "rank_biserial_r": rank_biserial_from_wilcoxon(diff),
                }
            )
    else:
        # replication-level paired by (date, rep) with shared shocks
        a = res[res.scenario == "S0"].set_index(["date", "rep"])[metric]
        for s in [x for x in SCENARIOS if x != "S0"]:
            b = res[res.scenario == s].set_index(["date", "rep"])[metric]
            diff = (a - b).dropna()
            if len(diff) < 5:
                continue
            # For large n, wilcoxon can be slow; use sample if needed
            if len(diff) > 5000:
                diff = diff.sample(5000, random_state=0)
            stat, p = stats.wilcoxon(diff, alternative="greater", zero_method="wilcox")
            rows.append(
                {
                    "level": level,
                    "metric": metric,
                    "scenario": s,
                    "n": len(diff),
                    "median_diff": float(diff.median()),
                    "wilcoxon_stat": float(stat),
                    "p_value": float(p),
                    "rank_biserial_r": rank_biserial_from_wilcoxon(diff),
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm_adjust(out["p_value"].tolist())
        out["significant_holm_0_05"] = out["p_holm"] < 0.05
    return out


def tost_equivalence(res: pd.DataFrame, metric: str, margin: float) -> pd.DataFrame:
    wide = day_medians(res, metric)
    rows = []
    base_imp = (wide["S0"] - wide["S1"]) / wide["S0"].replace(0, np.nan)
    for s in ["S2", "S3", "S3s", "S_OPT"]:
        if s not in wide.columns:
            continue
        imp_s = (wide["S0"] - wide[s]) / wide["S0"].replace(0, np.nan)
        d = (imp_s - base_imp).dropna()
        if len(d) < 5:
            continue
        _, p1 = stats.ttest_1samp(d, -margin, alternative="greater")
        _, p2 = stats.ttest_1samp(d, margin, alternative="less")
        p_tost = max(p1, p2)
        rows.append(
            {
                "metric": metric,
                "compare": f"{s}_vs_S1_rel_improvement",
                "margin": margin,
                "n_days": len(d),
                "mean_diff": float(d.mean()),
                "p_tost": float(p_tost),
                "equivalent_at_margin": bool(p_tost < 0.05),
            }
        )
    return pd.DataFrame(rows)


def mc_convergence(res: pd.DataFrame, metric: str = "peak_queue", scenario: str = "S0") -> pd.DataFrame:
    sub = res[res.scenario == scenario]
    vals = []
    for r in range(REPS):
        chunk = sub[sub.rep <= r][metric]
        vals.append(
            {
                "reps_used": r + 1,
                "median": float(chunk.median()),
                "p10": float(chunk.quantile(0.10)),
                "p90": float(chunk.quantile(0.90)),
            }
        )
    return pd.DataFrame(vals)


def make_figures(res: pd.DataFrame, conv: pd.DataFrame) -> None:
    summary = (
        res.groupby("scenario")["peak_queue"]
        .agg(median="median", p10=lambda x: x.quantile(0.10), p90=lambda x: x.quantile(0.90))
        .reindex(SCENARIOS)
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    x = np.arange(len(SCENARIOS))
    ax.bar(x, summary["median"], color="#1f4e79", alpha=0.85)
    ax.errorbar(
        x,
        summary["median"],
        yerr=[summary["median"] - summary["p10"], summary["p90"] - summary["median"]],
        fmt="none",
        ecolor="#333",
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIOS, rotation=15)
    ax.set_ylabel("Peak post-service queue (trucks)")
    ax.set_title("Peak queue by scenario (median, p10–p90)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "Fig1_peak_queue_by_scenario.png", dpi=200)
    fig.savefig(FIG / "Fig1_peak_queue_by_scenario.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(conv["reps_used"], conv["median"], label="median", color="#1f4e79")
    ax.fill_between(conv["reps_used"], conv["p10"], conv["p90"], color="#1f4e79", alpha=0.15, label="p10–p90")
    ax.set_xlabel("Replications used")
    ax.set_ylabel("Peak queue (S0 pooled)")
    ax.set_title("Monte Carlo convergence diagnostic")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "Fig_MC_convergence.png", dpi=200)
    plt.close(fig)


def run_sensitivities() -> None:
    global THETA, DELTA_CLEAR
    rows = []
    for lam in [900, 1200, 1500]:
        res = run_main_experiments(lambda0=lam)
        for s in SCENARIOS:
            g = res[res.scenario == s]
            rows.append(
                {
                    "lambda0": lam,
                    "scenario": s,
                    "median_peak_queue": g["peak_queue"].median(),
                    "median_waiting_proxy": g["waiting_proxy"].median(),
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "sensitivity_lambda0.csv", index=False)

    # S-server sensitivity (C3)
    srows = []
    for ns in [1, 2, 3]:
        res = run_main_experiments(n_servers=ns)
        for s in SCENARIOS:
            g = res[res.scenario == s]
            srows.append(
                {
                    "n_servers": ns,
                    "scenario": s,
                    "median_peak_queue": g["peak_queue"].median(),
                    "median_waiting_proxy": g["waiting_proxy"].median(),
                }
            )
        # ranking by median peak (lower better among controlled)
        peaks = res.groupby("scenario")["peak_queue"].median().reindex(SCENARIOS)
        print(f"S={ns} peak medians:", peaks.round(1).to_dict())
    pd.DataFrame(srows).to_csv(OUT / "sensitivity_n_servers.csv", index=False)

    rows = []
    for tilt in [0.0, 0.10, 0.20, 0.40]:
        res = run_main_experiments(tilt=tilt)
        wide = day_medians(res, "peak_queue")
        for s in ["S1", "S2", "S3"]:
            imp = ((wide["S0"] - wide[s]) / wide["S0"].replace(0, np.nan)).median()
            rows.append({"tilt": tilt, "scenario": s, "median_rel_improvement_peak_queue": float(imp)})
    pd.DataFrame(rows).to_csv(OUT / "sensitivity_forecast_tilt.csv", index=False)

    # E1: independent θ / δ perturbations (lighter: 15 reps, S0/S1/S3/S3s only)
    param_rows = []
    df = load_intensity()
    I_mean = float(df["I"].mean())
    days = select_days(df)
    for theta in [0.35, 0.50, 0.65]:
        THETA = theta
        local = []
        for i, row in days.iterrows():
            N = int(max(200, round(LAMBDA0_DEFAULT * float(row["I"]) / I_mean)))
            for r in range(15):
                rng_shock = np.random.default_rng(RNG_MASTER + int(i) * 100_000 + r)
                compliance = float(rng_shock.uniform(0.75, 0.95))
                mu = float(np.clip(rng_shock.normal(MU_MEAN, 0.18), 1.15, 2.10))
                storage_cap = 450.0 * float(rng_shock.choice([0.80, 1.00, 1.20]))
                peak = allocate(N, peak_weights(rng_shock), rng_shock)
                I_hat = float(row["I_hat_base"] * rng_shock.lognormal(0, 0.15))
                for s in ["S0", "S1", "S3"]:
                    rng_s = np.random.default_rng(RNG_MASTER + 17 + int(i) * 97 + r * 13 + hash(s) % 97)
                    m = run_one(s, N, float(row["I"]), I_mean, I_hat, rng_s, compliance, mu, storage_cap, peak)
                    local.append({"date": row["date"].strftime("%Y-%m-%d"), "scenario": s, "peak_queue": m["peak_queue"]})
        loc = pd.DataFrame(local)
        wide = loc.groupby(["date", "scenario"])["peak_queue"].median().unstack()
        for s in ["S1", "S3"]:
            imp = ((wide["S0"] - wide[s]) / wide["S0"].replace(0, np.nan)).median()
            param_rows.append(
                {
                    "param": "theta",
                    "value": theta,
                    "scenario": s,
                    "median_rel_improvement_peak_queue": float(imp),
                    "median_peak": float(wide[s].median()),
                }
            )
    THETA = 0.50
    for delta in [0.15, 0.25, 0.40]:
        DELTA_CLEAR = delta
        local = []
        for i, row in days.iterrows():
            N = int(max(200, round(LAMBDA0_DEFAULT * float(row["I"]) / I_mean)))
            for r in range(15):
                rng_shock = np.random.default_rng(RNG_MASTER + int(i) * 100_000 + r)
                compliance = float(rng_shock.uniform(0.75, 0.95))
                mu = float(np.clip(rng_shock.normal(MU_MEAN, 0.18), 1.15, 2.10))
                storage_cap = 450.0 * float(rng_shock.choice([0.80, 1.00, 1.20]))
                peak = allocate(N, peak_weights(rng_shock), rng_shock)
                I_hat = float(row["I_hat_base"] * rng_shock.lognormal(0, 0.15))
                rng_s = np.random.default_rng(RNG_MASTER + 19 + int(i) * 97 + r * 13)
                m = run_one("S3s", N, float(row["I"]), I_mean, I_hat, rng_s, compliance, mu, storage_cap, peak)
                local.append(m)
        g3 = pd.DataFrame(local)
        param_rows.append(
            {
                "param": "delta_clear",
                "value": delta,
                "scenario": "S3s",
                "median_rel_improvement_peak_queue": np.nan,
                "median_peak": float(g3["peak_queue"].median()),
                "median_storage_pressure": float(g3["storage_pressure"].median()),
                "mean_deferred": float(g3["deferred"].mean()),
            }
        )
    DELTA_CLEAR = 0.25
    pd.DataFrame(param_rows).to_csv(OUT / "sensitivity_theta_delta.csv", index=False)

    df = load_intensity()
    rows = []
    for w_c, w_v, label in [(0.5, 0.5, "0.5/0.5"), (0.6, 0.4, "0.6/0.4"), (0.7, 0.3, "0.7/0.3")]:
        d = df.copy()
        d["I"] = intensity_with_weights(d, w_c, w_v)
        I_mean = float(d["I"].mean())
        days = select_days(d)
        local_rows = []
        for i, row in days.iterrows():
            N = int(max(200, round(LAMBDA0_DEFAULT * float(row["I"]) / I_mean)))
            for r in range(15):
                rng_shock = np.random.default_rng(7 + i * 1000 + r + hash(label) % 1000)
                compliance = float(rng_shock.uniform(0.75, 0.95))
                mu = float(np.clip(rng_shock.normal(1.55, 0.18), 1.15, 2.10))
                storage_cap = 450.0 * float(rng_shock.choice([0.80, 1.00, 1.20]))
                peak = allocate(N, peak_weights(rng_shock), rng_shock)
                I_hat = float(row["I_hat_base"] * rng_shock.lognormal(0, 0.15))
                for s in ["S0", "S1"]:
                    rng_s = np.random.default_rng(11 + i + r + (0 if s == "S0" else 1))
                    m = run_one(
                        s,
                        N,
                        float(row["I"]),
                        I_mean,
                        I_hat,
                        rng_s,
                        compliance,
                        mu,
                        storage_cap,
                        peak,
                    )
                    local_rows.append({"scenario": s, "peak_queue": m["peak_queue"], "date": row["date"].strftime("%Y-%m-%d")})
        loc = pd.DataFrame(local_rows)
        wide = loc.groupby(["date", "scenario"])["peak_queue"].median().unstack()
        imp = ((wide["S0"] - wide["S1"]) / wide["S0"]).median()
        rows.append(
            {
                "weights": label,
                "median_S1_improvement_vs_S0": float(imp),
                "median_S0_peak": float(wide["S0"].median()),
                "median_S1_peak": float(wide["S1"].median()),
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "sensitivity_intensity_weights.csv", index=False)

    main = pd.read_csv(OUT / "scenario_day_replications.csv")
    main["cap_mult"] = (main["storage_cap"] / 450.0).round(2)
    stor = (
        main[main.scenario == "S3s"]
        .groupby("cap_mult")
        .agg(
            median_deferred=("deferred", "median"),
            mean_deferred=("deferred", "mean"),
            median_peak=("peak_queue", "median"),
            n=("deferred", "size"),
        )
        .reset_index()
    )
    stor.to_csv(OUT / "sensitivity_storage_multiplier.csv", index=False)


def main() -> None:
    print("Running open-data TAS experiments (E[C] LP, distinct theta/gamma/delta, S-sweep)...")
    res = run_main_experiments()
    res.to_csv(OUT / "scenario_day_replications.csv", index=False)

    metrics = [
        "peak_queue",
        "mean_wait_min",
        "p90_wait_min",
        "corridor_delay_index",
        "waiting_proxy",
        "storage_pressure",
        "deferred",
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
    pd.DataFrame(summary_rows).to_csv(OUT / "summary_by_scenario.csv", index=False)

    # S_OPT diagnostics (not used in headlines)
    opt = res[res.scenario == "S_OPT"]
    pd.DataFrame(
        [
            {
                "median_peak_headline_E_capacity_plus_compliance": opt["peak_queue"].median(),
                "median_peak_perfect_compliance_diagnostic": opt["peak_queue_perfect_compliance"].median(),
                "median_peak_foresight_lp_diagnostic": opt["peak_queue_foresight_lp_diagnostic"].median(),
                "note": "Headline S_OPT: LP on E[C]=60*MU_MEAN*S then compliance mixture; foresight diagnostic uses realised mu in LP",
            }
        ]
    ).to_csv(OUT / "s_opt_compliance_diagnostic.csv", index=False)

    improv_rows = []
    base = res[res.scenario == "S0"].groupby("date")[metrics].median().add_prefix("S0_")
    for s in [x for x in SCENARIOS if x != "S0"]:
        cur = res[res.scenario == s].groupby("date")[metrics].median()
        merged = cur.join(base, how="inner")
        for col in ["peak_queue", "p90_wait_min", "corridor_delay_index", "waiting_proxy"]:
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
    pd.DataFrame(improv_rows).to_csv(OUT / "improvement_vs_S0.csv", index=False)

    wilc_day = pd.concat([wilcoxon_battery(res, m, "day") for m in ["peak_queue", "p90_wait_min", "waiting_proxy"]], ignore_index=True)
    wilc_rep = pd.concat(
        [wilcoxon_battery(res, m, "replication") for m in ["peak_queue", "p90_wait_min", "waiting_proxy"]],
        ignore_index=True,
    )
    wilc_day.to_csv(OUT / "stats_wilcoxon_vs_S0.csv", index=False)
    wilc_rep.to_csv(OUT / "stats_wilcoxon_replication_level.csv", index=False)

    tost_all = []
    for margin in [0.03, 0.05, 0.07]:
        for m in ["peak_queue", "waiting_proxy"]:
            tost_all.append(tost_equivalence(res, m, margin))
    pd.concat(tost_all, ignore_index=True).to_csv(OUT / "stats_tost_equivalence.csv", index=False)

    conv = mc_convergence(res)
    conv.to_csv(OUT / "mc_convergence.csv", index=False)
    make_figures(res, conv)
    res.groupby(["scenario", "regime"])[metrics].median().reset_index().to_csv(OUT / "summary_by_regime.csv", index=False)

    print("Running sensitivities (incl. S=1,2,3)...")
    run_sensitivities()

    meta = {
        "eps": EPS,
        "theta": THETA,
        "gamma_tilt": GAMMA_TILT,
        "delta_clear": DELTA_CLEAR,
        "alpha_in": ALPHA_IN,
        "mu_mean_lp_plan": MU_MEAN,
        "bpr_alpha": BPR_ALPHA,
        "bpr_beta": BPR_BETA,
        "n_servers_default": 1,
        "n_servers_coprimary": [1, 2, 3],
        "storage_cap_base": 450.0,
        "lambda0_default": LAMBDA0_DEFAULT,
        "reps": REPS,
        "scenarios": SCENARIOS,
        "q1_major_revision": {
            "E1": "distinct theta=0.50, gamma=0.20, delta=0.25",
            "E2": "S_OPT LP on E[C] via MU_MEAN; foresight diagnostic only",
            "E3": "sensitivity_n_servers.csv co-primary S in {1,2,3}",
            "shared_shocks": "compliance/mu/storage/peak/I_hat shared across scenarios per (day,rep)",
        },
    }
    (OUT / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\nMedian peak_queue:")
    print(res.groupby("scenario")["peak_queue"].median().reindex(SCENARIOS).round(2))
    print("\nS_OPT perfect diagnostic median:", round(opt["peak_queue_perfect_compliance"].median(), 2))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
