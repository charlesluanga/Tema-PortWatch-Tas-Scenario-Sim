"""
Final storage-fairness green-light correction.

Information ladder: S1 = historical weekday baseline; S2/S3/S_OPT = genuine
trailing public forecast (no artificial forecast noise).
S2 = forecast-and-capacity-informed planning layer (bundled info+control).
Soft primary: S0/S1/S2/S3/S_OPT under non-binding soft yard.
Common hard: S0-H…S_OPT-H; S3s-H differs from S3-H only by proactive deferral.
Two-stage lexicographic LP; custom circular moving-block bootstrap inference.
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

# Documented parameters — distinct defaults
EPS = 1e-9  # normalisation / numerical floor (not LP tolerance)
LP_TOL = 1e-6  # optimiser / integerisation peak-feasibility tolerance
THETA = 0.50  # residual reallocation when mid-queue > 50% of mid-hour capacity
KAPPA_MAX_DEFAULT = 0.60  # max S2 blend toward planning-capacity weights
KAPPA_SCALE = 1.0  # κ = clip(KAPPA_SCALE * max(0, I_hat/I_ref - 1), 0, κ_max)
GAMMA_TILT = KAPPA_MAX_DEFAULT  # legacy alias: tilt argument = κ_max
DELTA_CLEAR = 0.25  # yard clears 25% of occupancy per hour
ALPHA_IN = 0.55  # yard intake (stated assumption)
ETA_STORAGE_WARNING = 0.85  # η: assumed mid-day yard-warning trigger for S3s-H
STORAGE_TIGHT_FRAC = ETA_STORAGE_WARNING  # backwards-compatible alias
INITIAL_YARD_OCCUPANCY_FRAC = 0.40  # assumed Y_{t,0}/C^Y; tested in secondary hard-yard sensitivity
RHO_PROACTIVE = 0.25  # ρ: assumed proactive-deferral fraction of residual second-half booked volume
PROACTIVE_HOLD_SHARE = RHO_PROACTIVE  # backwards-compatible alias
STORAGE_CAP_BASE = 450.0
STORAGE_CAP_MULTS = (0.80, 1.00, 1.20)
BPR_ALPHA = 0.15  # textbook BPR (corridor congestion *index*, not physical travel time)
BPR_BETA = 4.0
MU_MEAN = 1.55  # planning mean for LP (no foresight of realised μ)
N_HOURS = 16
N_SERVERS = 1  # scarce-capacity stress calibration; S∈{2,3} are sensitivities
LAMBDA0_DEFAULT = 1400
N_MIN = 200
ZERO_S0_PEAK_THRESHOLD = 1.0  # trucks; peak relative Δ only when S0 peak > this
ZERO_S0_WAIT_THRESHOLD = 1.0  # minutes; waiting-proxy relative Δ if used
ZERO_S0_THRESHOLD = ZERO_S0_PEAK_THRESHOLD  # backward-compatible alias
REPS = 40
RNG_MASTER = 42
# Soft primary (no S3s): shared soft non-binding yard
SCENARIOS_SOFT = ["S0", "S1", "S2", "S3", "S_OPT"]
# Common hard-yard: S3s-H differs from S3-H only by proactive deferral
SCENARIOS_HARD = ["S0-H", "S1-H", "S2-H", "S3-H", "S3s-H", "S_OPT-H"]
SCENARIOS = SCENARIOS_SOFT  # default primary soft labels
HIST_YEARS = (2019, 2020, 2021, 2022, 2023)
EVAL_YEAR = 2024
# Primary relative equivalence margin = 5 pp; 7/10 pp are sensitivity-only
TOST_PRIMARY_MARGIN = 0.05
TOST_SENS_MARGINS = (0.07, 0.10)
TOST_MARGINS = (TOST_PRIMARY_MARGIN,) + TOST_SENS_MARGINS
TOST_ABS_PEAK_MARGINS = (5.0, 10.0, 15.0)
SENSITIVITY_DAY_STRIDE = 6  # deterministic secondary sweeps (~61 days)
STATIC_SMOOTH_DEFAULT = 0.50  # blend of expected peak toward uniform
CAPACITY_PROFILE_DEFAULT = "mild"  # reference stylised within-day capacity
BLOCK_BOOT_REPS = 2000
BLOCK_BOOT_SEED = 123
BLOCK_LEN_DEFAULT = 7  # weekly blocks for maritime intensity dependence
BLOCK_LENS = (5, 7, 14)
MBB_CI_ALPHA = 0.10  # 90% percentile CI for TOST-style equivalence (two one-sided α=0.05)
# Metric-specific MC stability (R=30 → R=40)
MC_TOL_PEAK_TRUCKS = 1.0
MC_TOL_WAIT_MIN = 1.0
MC_TOL_REL_PP = 0.005  # 0.5 percentage points on relative contrasts when used
MC_TOL_RELATIVE = 0.01  # 1% relative change when |estimate| is large

# Module-level reference intensity from 2019–2023 (set by load_intensity)
I_REF: float = 1.0


def scenario_base(name: str) -> str:
    """Map S3-H / S3s-H labels to engine base codes S3 / S3s."""
    return name[:-2] if str(name).endswith("-H") else str(name)


def is_proactive_scenario(name: str) -> bool:
    return scenario_base(name) == "S3s"


def load_intensity(year: int = EVAL_YEAR) -> pd.DataFrame:
    """
    Build open intensity with a clean information boundary:
      - min–max normalisation and I_ref from HIST_YEARS (2019–2023);
      - evaluation year rows only returned for experiments;
      - I_hat_t = trailing 7-day mean of past I (through t−1), no artificial noise;
      - I_S1_base = weekday mean of historical I (main static baseline for S1);
      - I_S1_recent = expanding past-only weekday mean (freshness sensitivity).
    """
    global I_REF
    raw = pd.read_csv(DATA / "tema_portwatch_daily_2019_2026.csv")
    raw["date"] = pd.to_datetime(raw["date"], utc=True)
    raw = raw.sort_values("date").copy()
    raw["year"] = raw["date"].dt.year
    usable = raw["portcalls_container"].notna() & raw["import_container"].notna() & raw["export_container"].notna()
    d = raw.loc[usable].copy()
    hist = d[d["year"].isin(HIST_YEARS)]
    if len(hist) < 100:
        raise RuntimeError("Insufficient 2019–2023 PortWatch history for reference normalisation.")
    c = d["portcalls_container"].astype(float)
    v = (d["import_container"] + d["export_container"]).astype(float)
    c_lo, c_hi = float(hist["portcalls_container"].min()), float(hist["portcalls_container"].max())
    v_hist = (hist["import_container"] + hist["export_container"]).astype(float)
    v_lo, v_hi = float(v_hist.min()), float(v_hist.max())
    c_n = (c - c_lo) / (c_hi - c_lo + EPS)
    v_n = (v - v_lo) / (v_hi - v_lo + EPS)
    # Intentionally unclipped: evaluation-year values may fall outside [0, 1].
    d["I"] = 0.6 * c_n + 0.4 * v_n
    I_REF = float(d.loc[d["year"].isin(HIST_YEARS), "I"].mean())
    # Genuine one-day-ahead public forecast: trailing seven-day mean through t−1
    d["I_hat"] = d["I"].shift(1).rolling(7, min_periods=1).mean()
    d["I_hat"] = d["I_hat"].fillna(I_REF)
    # Recent static comparator: use only observations strictly preceding the date
    # and of the same weekday.  This is a sensitivity-only S1 baseline.
    d["weekday"] = d["date"].dt.weekday
    d["I_S1_recent"] = (
        d.groupby("weekday", group_keys=False)["I"]
        .apply(lambda x: x.shift(1).expanding(min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    d["I_S1_recent"] = d["I_S1_recent"].fillna(I_REF)
    d["I_hat_base"] = d["I_hat"]  # alias for downstream compatibility
    dow_mean = d.loc[d["year"].isin(HIST_YEARS)].groupby(d.loc[d["year"].isin(HIST_YEARS), "date"].dt.dayofweek)["I"].mean()
    d["I_S1_base"] = d["date"].dt.dayofweek.map(dow_mean).astype(float).fillna(I_REF)
    days_to_sunday = (6 - d["date"].dt.dayofweek) % 7
    d["week_end"] = (d["date"] + pd.to_timedelta(days_to_sunday, unit="D")).dt.strftime("%Y-%m-%d")
    out = d[d["year"] == int(year)].copy().reset_index(drop=True)
    out.attrs["I_ref"] = I_REF
    out.attrs["hist_years"] = HIST_YEARS
    return out


def intensity_with_weights(d: pd.DataFrame, w_c: float, w_v: float) -> pd.Series:
    """Reweight intensity using the same historical min–max anchors when present."""
    c = d["portcalls_container"].astype(float)
    v = (d["import_container"] + d["export_container"]).astype(float)
    # Prefer historical anchors if columns came from load_intensity panel; else within-sample
    c_n = (c - c.min()) / (c.max() - c.min() + EPS)
    v_n = (v - v.min()) / (v.max() - v.min() + EPS)
    return w_c * c_n + w_v * v_n


def expected_peak_weights(shape: str = "two_hump") -> np.ndarray:
    """Deterministic expected uncoordinated profile (no jitter); used to build stylised S1."""
    x = np.linspace(0, 1, N_HOURS)
    if shape == "morning":
        w = np.exp(-0.5 * ((x - 0.25) / 0.12) ** 2)
    elif shape == "flat_uncoord":
        w = 0.7 + 0.3 * np.sin(np.pi * x)
    else:
        w = 0.62 * np.exp(-0.5 * ((x - 0.22) / 0.09) ** 2) + 0.38 * np.exp(
            -0.5 * ((x - 0.70) / 0.10) ** 2
        )
    w = np.clip(w, 1e-6, None)
    return w / w.sum()


def capacity_shape(kind: str = CAPACITY_PROFILE_DEFAULT) -> np.ndarray:
    """
    Stylised within-day staffing/capacity profile s_h (mean 1).
    Not observed MPS capacity. Hours 0..15 = 06:00-22:00.
    Reference = mild; flat and strong are lower-/upper-heterogeneity regimes.
    """
    if kind == "flat":
        s = np.ones(N_HOURS, dtype=float)
    elif kind == "strong":
        # Upper-heterogeneity stress: lunch trough
        s = np.array(
            [
                0.75, 0.85,
                1.15, 1.20, 1.15,
                0.70, 0.65,
                1.20, 1.25, 1.20, 1.15,
                1.00, 0.95, 0.90,
                0.80, 0.70,
            ],
            dtype=float,
        )
    else:
        # Mild (reference): gentle midday dip
        s = np.array(
            [
                0.95, 1.00, 1.05, 1.08, 1.05, 0.92, 0.90,
                1.05, 1.08, 1.05, 1.02, 1.00, 0.98, 0.96, 0.94, 0.92,
            ],
            dtype=float,
        )
    assert len(s) == N_HOURS
    return s / s.mean()


def hourly_capacity(
    mu: float,
    n_servers: int,
    shape: np.ndarray | None = None,
    capacity_kind: str = CAPACITY_PROFILE_DEFAULT,
) -> np.ndarray:
    """Realised or planning capacity path C_h = 60*mu*S*s_h."""
    s = capacity_shape(capacity_kind) if shape is None else shape
    return 60.0 * float(mu) * int(n_servers) * s


def peak_weights(rng: np.random.Generator, shape: str = "two_hump") -> np.ndarray:
    w = expected_peak_weights(shape)
    w = np.clip(w, 1e-6, None) * rng.lognormal(0, 0.05, size=N_HOURS)
    return w / w.sum()


def uniform_weights(rng: np.random.Generator | None = None) -> np.ndarray:
    w = np.ones(N_HOURS)
    if rng is not None:
        w = w * rng.lognormal(0, 0.03, size=N_HOURS)
    return w / w.sum()


def static_appointment_weights(
    rng: np.random.Generator | None = None,
    smooth: float = STATIC_SMOOTH_DEFAULT,
    peak_shape: str = "two_hump",
) -> np.ndarray:
    """
    Stylised static appointment template: pre-specified smoothing of the expected
    uncoordinated profile toward uniform. Independent of the capacity path.
    smooth=0 → expected peak; smooth=1 → uniform. Default smooth=0.50.
    No slot jitter in the main specification (rng retained for API compatibility).
    """
    smooth = float(np.clip(smooth, 0.0, 1.0))
    w_peak = expected_peak_weights(peak_shape)
    w_uni = np.ones(N_HOURS, dtype=float) / N_HOURS
    w = (1.0 - smooth) * w_peak + smooth * w_uni
    return w / w.sum()


def capacity_plan_weights(capacity_kind: str = CAPACITY_PROFILE_DEFAULT) -> np.ndarray:
    """Pre-day planning-capacity appointment weights ∝ s_h (mean-one profile)."""
    s = capacity_shape(capacity_kind)
    return s / s.sum()


def forecast_kappa(I_hat: float, I_ref: float, kappa_max: float = KAPPA_MAX_DEFAULT) -> float:
    """Bounded monotone forecast-pressure blend weight for S2."""
    pressure = max(0.0, float(I_hat) / (float(I_ref) + EPS) - 1.0)
    return float(np.clip(KAPPA_SCALE * pressure, 0.0, float(kappa_max)))


def forecast_weights(
    I_hat: float,
    I_ref: float,
    rng: np.random.Generator,
    tilt: float = KAPPA_MAX_DEFAULT,
    smooth: float = STATIC_SMOOTH_DEFAULT,
    peak_shape: str = "two_hump",
    capacity_kind: str = CAPACITY_PROFILE_DEFAULT,
) -> np.ndarray:
    """
    S2: convex blend of the static template toward the pre-day planning-capacity profile.
    w_S2 = (1−κ) w_S1 + κ w_capacity, with κ increasing in forecast demand pressure.
    Does not observe realised μ or mid-day state. No midday mechanical tilt.
    """
    w_s1 = static_appointment_weights(rng, smooth=smooth, peak_shape=peak_shape)
    w_cap = capacity_plan_weights(capacity_kind)
    kappa = forecast_kappa(I_hat, I_ref, kappa_max=tilt)
    w = (1.0 - kappa) * w_s1 + kappa * w_cap
    return w / w.sum()


def allocate(N: int, weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.multinomial(int(N), weights)


def scale_day_total(I_val: float, lambda0: float, I_ref: float) -> int:
    return int(max(N_MIN, round(float(lambda0) * float(I_val) / (float(I_ref) + EPS))))


def mix_compliance(planned: np.ndarray, peak: np.ndarray, compliance: float, total: int | None = None) -> np.ndarray:
    """Realised compliance *share* (not truck-level Bernoulli): round(p·plan + (1-p)·peak)."""
    planned = np.asarray(planned, dtype=float)
    peak = np.asarray(peak, dtype=float)
    assert planned.shape == peak.shape
    target = int(planned.sum() if total is None else total)
    out = np.round(compliance * planned + (1.0 - compliance) * peak).astype(int)
    diff = int(target - out.sum())
    mid = len(out) // 2
    out[mid] += diff
    return np.maximum(out, 0)


def _uniform_int_arrivals(N: int, n_hours: int = N_HOURS) -> np.ndarray:
    base = np.full(n_hours, N // n_hours)
    base[: N % n_hours] += 1
    return base.astype(int)


def _integerise_arrivals(a: np.ndarray, N: int, prefer: np.ndarray | None = None) -> np.ndarray:
    """Floor + largest-remainder; tie-break prefers higher `prefer` weights (default: capacity shape)."""
    a = np.maximum(np.asarray(a, dtype=float), 0.0)
    n_hours = len(a)
    if not np.isfinite(a).all() or a.sum() <= 0:
        return _uniform_int_arrivals(N, n_hours)
    a = a * (N / a.sum())
    a_int = np.floor(a).astype(int)
    rem = int(N - a_int.sum())
    if rem < 0:
        return _uniform_int_arrivals(N, n_hours)
    frac = a - a_int
    pref = capacity_shape()[:n_hours] if prefer is None else np.asarray(prefer, dtype=float)
    order = np.lexsort((np.arange(n_hours), -pref, -frac))
    for i in range(rem):
        a_int[order[i % n_hours]] += 1
    return a_int


def realise_forecast_plan(
    planned_hat: np.ndarray,
    N: int,
    peak: np.ndarray,
    compliance: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Planning used forecast day-total N_hat = sum(planned_hat); realisation uses realised N.
    Integer repair always preserves realised N_t (not the planned day-total).
    If N > N_hat: booked plan + uncoordinated excess.
    If N < N_hat: proportionally thin the plan to N, then apply compliance share.
    """
    planned_hat = np.asarray(planned_hat, dtype=int)
    N_hat = int(planned_hat.sum())
    peak = np.asarray(peak, dtype=int)
    if N_hat <= 0:
        return peak.copy() if int(peak.sum()) == N else _integerise_arrivals(peak.astype(float), N)
    if N == N_hat:
        return mix_compliance(planned_hat, peak, compliance, total=N)
    if N < N_hat:
        planned = _integerise_arrivals(planned_hat.astype(float), N)
        peak_n = _integerise_arrivals(peak.astype(float), N)
        return mix_compliance(planned, peak_n, compliance, total=N)
    # N > N_hat: retain booked plan under compliance vs peak slice of size N_hat, then add excess
    peak_booked = _integerise_arrivals(peak.astype(float), N_hat)
    booked = mix_compliance(planned_hat, peak_booked, compliance, total=N_hat)
    excess = int(N - N_hat)
    w = np.maximum(peak.astype(float), EPS)
    w = w / w.sum()
    extra = rng.multinomial(excess, w)
    out = booked + extra
    # Safety: enforce sum == N
    if int(out.sum()) != N:
        out = _integerise_arrivals(out.astype(float), N)
    return out


def _lp_residuals(a: np.ndarray, C: np.ndarray) -> np.ndarray:
    q = np.zeros(len(a), dtype=float)
    for h in range(len(a)):
        prev = 0.0 if h == 0 else q[h - 1]
        q[h] = max(0.0, prev + float(a[h]) - float(C[h]))
    return q


def lp_optimal_arrivals(
    N: int,
    mu_plan: float,
    n_servers: int,
    C_plan: np.ndarray | None = None,
    capacity_kind: str = CAPACITY_PROFILE_DEFAULT,
) -> np.ndarray:
    """
    True two-stage lexicographic LP:
      Stage 1: minimise peak residual M.
      Stage 2: fix M <= M* + LP_TOL and minimise sum of residual queues.
    Plans on expected/planning capacity and forecast day-total only.
    """
    C = hourly_capacity(mu_plan, n_servers, capacity_kind=capacity_kind) if C_plan is None else np.asarray(C_plan, dtype=float)
    H = N_HOURS
    n = 2 * H + 1  # a_h, q_h, M

    def _constraints(m_ub: float | None = None):
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
            rhs.append(-float(C[h]))
            r2 = np.zeros(n)
            r2[H + h] = 1.0
            r2[-1] = -1.0
            rows.append(r2)
            rhs.append(0.0)
        if m_ub is not None:
            r3 = np.zeros(n)
            r3[-1] = 1.0
            rows.append(r3)
            rhs.append(float(m_ub))
        return A_eq, b_eq, np.vstack(rows), np.array(rhs, dtype=float)

    # Stage 1: min M
    c1 = np.zeros(n)
    c1[-1] = 1.0
    A_eq, b_eq, A_ub, b_ub = _constraints()
    res1 = linprog(
        c1,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(0, float(N))] * H + [(0, None)] * H + [(0, None)],
        method="highs",
        options={"presolve": True},
    )
    if not res1.success or res1.x is None or not np.isfinite(res1.x[:H]).all():
        w = np.maximum(C, EPS)
        w = w / w.sum()
        return _integerise_arrivals(w * N, N, prefer=C)

    M_star = float(res1.x[-1])

    # Stage 2: min sum q_h s.t. M <= M* + LP_TOL
    c2 = np.zeros(n)
    c2[H : 2 * H] = 1.0
    A_eq2, b_eq2, A_ub2, b_ub2 = _constraints(m_ub=M_star + LP_TOL)
    res2 = linprog(
        c2,
        A_ub=A_ub2,
        b_ub=b_ub2,
        A_eq=A_eq2,
        b_eq=b_eq2,
        bounds=[(0, float(N))] * H + [(0, None)] * H + [(0, M_star + LP_TOL)],
        method="highs",
        options={"presolve": True},
    )
    a_cont = res2.x[:H] if res2.success and res2.x is not None else res1.x[:H]
    a_int = _integerise_arrivals(a_cont, N, prefer=C)
    # Verify integerisation does not materially violate primary peak bound
    q_int = _lp_residuals(a_int, C)
    if float(q_int.max()) > M_star + max(1.0, 10 * LP_TOL):
        # Fall back: re-integerise preferring capacity more aggressively
        a_int = _integerise_arrivals(np.maximum(C, EPS), N, prefer=C)
    return a_int


def simulate_day(
    arrivals: np.ndarray,
    mu: float,
    storage_cap: float,
    hard_storage: bool,
    n_servers: int,
    C_path: np.ndarray | None = None,
    capacity_kind: str = CAPACITY_PROFILE_DEFAULT,
    proactive_defer: np.ndarray | None = None,
    initial_yard_frac: float = INITIAL_YARD_OCCUPANCY_FRAC,
) -> dict:
    """
    Waiting and corridor metrics are gate-capacity-normalised proxies, not physical
    travel/sojourn times. Hard storage constrains served throughput via post-clearance
    yard room; blocked trucks remain in the residual gate queue (not deferred).
    Soft storage is non-binding (no occupancy clip). Under soft storage, yard state Y is a
    notional yard-load relative to reference C^Y, not physically capacity-constrained occupancy.
    Proactive deferral is separate.
    Conservation: Q_prev + A_pres = X + Q_cur each hour; sum(A_pres)=sum(X)+Q_end.
    """
    C = hourly_capacity(mu, n_servers, capacity_kind=capacity_kind) if C_path is None else np.asarray(C_path, dtype=float)
    queue = 0.0
    storage = float(initial_yard_frac) * storage_cap
    waits, post_queues = [], []
    served_total = deferred_proactive = corridor_acc = proxy_acc = storage_acc = 0.0
    presented_total = 0.0
    booked_total = float(np.sum(arrivals.astype(float)))
    proactive = np.zeros(N_HOURS) if proactive_defer is None else np.asarray(proactive_defer, dtype=float)
    max_storage = float(storage)

    for h, a_raw in enumerate(arrivals.astype(float)):
        Ch = float(C[h])
        a_sched = max(0.0, a_raw - float(proactive[h]))
        deferred_proactive += float(proactive[h])
        presented_total += a_sched

        y_pre = (1.0 - DELTA_CLEAR) * storage
        if hard_storage:
            if ALPHA_IN <= EPS:
                yard_room = float("inf") if y_pre <= float(storage_cap) + EPS else 0.0
            else:
                yard_room = max(0.0, (float(storage_cap) - y_pre) / ALPHA_IN)
        else:
            yard_room = float("inf")

        q_prev = queue
        load = q_prev + a_sched
        # Gate-capacity-normalised workload proxy (not physical queueing time)
        wait_proxy_min = 60.0 * load / max(Ch, EPS)
        waits.append(wait_proxy_min)

        served = min(load, Ch, yard_room)
        queue = max(0.0, load - served)
        assert abs((q_prev + a_sched) - (served + queue)) < 1e-6
        post_queues.append(queue)
        served_total += served
        storage = y_pre + ALPHA_IN * served
        if hard_storage:
            assert storage <= float(storage_cap) + 1e-6, f"hard-storage violation: Y={storage} > C^Y={storage_cap}"
        max_storage = max(max_storage, float(storage))

        storage_acc += storage / max(storage_cap, EPS)
        voc = load / max(Ch, EPS)
        corridor_acc += 1.0 + BPR_ALPHA * (voc ** BPR_BETA)
        proxy_acc += served * (0.35 + 0.65 * min(wait_proxy_min, 90.0) / 45.0) + 0.2 * queue

    assert abs(presented_total - (served_total + queue)) < 1e-6
    assert abs(booked_total - (presented_total + deferred_proactive)) < 1e-6

    return {
        "mean_wait_min": float(np.mean(waits)),
        "p90_wait_min": float(np.quantile(waits, 0.90)),
        "peak_queue": float(np.max(post_queues)) if post_queues else 0.0,
        "corridor_delay_index": float(corridor_acc / N_HOURS),
        "throughput": float(served_total),
        "waiting_proxy": float(proxy_acc),
        "storage_pressure": float(storage_acc / N_HOURS),
        "deferred": float(deferred_proactive),
        "deferred_physical": 0.0,
        "deferred_proactive": float(deferred_proactive),
        "presented": float(presented_total),
        "booked": float(booked_total),
        "final_queue": float(queue),
        "final_storage": float(storage),
        "max_storage": float(max_storage),
        "hourly_capacity_mean": float(np.mean(C)),
        "capacity_shape_lunch_ratio": float(C[5] / (np.mean(C) + EPS)),
        "arrival_sum": float(np.sum(arrivals)),
        "flow_check_served_plus_final_q": float(served_total + queue),
        "flow_check_presented": float(presented_total),
        "flow_ok": True,
    }


def residual_second_half(
    N: int,
    peak: np.ndarray,
    mixed_first: np.ndarray,
    queue_after_first: float,
    storage_after_first: float,
    storage_cap: float,
    I_hat: float,
    I_ref: float,
    rng: np.random.Generator,
    compliance: float,
    tilt: float,
    C_plan_path: np.ndarray,
    hard_storage: bool,
    smooth: float = STATIC_SMOOTH_DEFAULT,
    peak_shape: str = "two_hump",
    capacity_kind: str = CAPACITY_PROFILE_DEFAULT,
    proactive: bool = False,
    eta_storage_warning: float = ETA_STORAGE_WARNING,
    rho_proactive: float = RHO_PROACTIVE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reallocate residual day volume with compliance share retained on the second half.
    Returns (full_day_booked_arrivals, proactive_defer_vector).

    Proactive deferral (S3s-H only), deterministic given state:
      Activate iff hard_storage and storage_after_first > η * C^Y.
      Total deferred volume D^pro = round(ρ * N_rem) from the realised second-half booked
      profile, peeled from highest booked-count hours first (ties: earlier hour index).
      Stops when exactly D^pro trucks are removed; presented arrivals stay non-negative.
      Deferred trucks are removed before gate service and recorded as deferred_proactive.
    """
    mid = len(mixed_first)
    N_rem = int(N - int(mixed_first.sum()))
    N_rem = max(N_rem, 0)
    # S3's trigger and residual allocation use planned capacity only.  Its sole
    # additional information is observed mid-day queue (and hard-yard state).
    mid_cap = float(C_plan_path[mid - 1]) if mid > 0 else float(C_plan_path[0])
    storage_tight = hard_storage and (storage_after_first > eta_storage_warning * storage_cap)
    if queue_after_first > THETA * mid_cap or storage_tight:
        w = np.maximum(C_plan_path[mid:], EPS)
    else:
        w = forecast_weights(
            I_hat, I_ref, rng, tilt=tilt, smooth=smooth, peak_shape=peak_shape, capacity_kind=capacity_kind
        )[mid:]
    w = w / w.sum()
    adaptive_plan = rng.multinomial(N_rem, w) if N_rem > 0 else np.zeros(N_HOURS - mid, dtype=int)
    peak_rem = (
        _integerise_arrivals(peak[mid:].astype(float), N_rem, prefer=w)
        if N_rem > 0
        else np.zeros(N_HOURS - mid, dtype=int)
    )
    second = mix_compliance(adaptive_plan, peak_rem, compliance, total=N_rem)

    proactive_vec = np.zeros(N_HOURS, dtype=float)
    if proactive and storage_tight and N_rem > 0:
        d_pro = int(round(float(rho_proactive) * N_rem))
        d_pro = max(0, min(d_pro, int(second.sum())))
        if d_pro > 0:
            # Highest booked count first; ties broken by earlier second-half hour index.
            order = np.lexsort((np.arange(len(second)), -second.astype(float)))
            remaining = d_pro
            for idx in order:
                take = min(int(second[idx]), remaining)
                proactive_vec[mid + idx] += take
                remaining -= take
                if remaining <= 0:
                    break
            assert remaining == 0
            assert float(proactive_vec.sum()) == float(d_pro)
            assert np.all(second - proactive_vec[mid:] >= -1e-9)
    return np.concatenate([mixed_first, second]), proactive_vec


def _simulate_first_half(
    first: np.ndarray,
    C_real: np.ndarray,
    storage_cap: float,
    hard: bool,
    initial_yard_frac: float = INITIAL_YARD_OCCUPANCY_FRAC,
) -> tuple[float, float]:
    """Advance queue/storage through first half with physical hard-yard rule."""
    q = 0.0
    storage = float(initial_yard_frac) * storage_cap
    for h, a in enumerate(first):
        y_pre = (1.0 - DELTA_CLEAR) * storage
        yard_room = max(0.0, (storage_cap - y_pre) / max(ALPHA_IN, EPS)) if hard else float("inf")
        load = q + float(a)
        Ch = float(C_real[h])
        served_gate = min(load, Ch)
        served = min(served_gate, yard_room)
        q = max(0.0, load - served)
        storage = y_pre + ALPHA_IN * served
        if hard:
            assert storage <= float(storage_cap) + 1e-6
    return q, storage


def run_one(
    scenario: str,
    N: int,
    I: float,
    I_ref: float,
    I_hat: float,
    rng: np.random.Generator,
    compliance: float,
    mu: float,
    storage_cap: float,
    peak: np.ndarray,
    tilt: float = KAPPA_MAX_DEFAULT,
    n_servers: int = 1,
    hard_storage_env: bool = False,
    peak_shape: str = "two_hump",
    N_hat: int | None = None,
    N_base: int | None = None,
    lambda0: float = LAMBDA0_DEFAULT,
    static_smooth: float = STATIC_SMOOTH_DEFAULT,
    capacity_kind: str = CAPACITY_PROFILE_DEFAULT,
    eta_storage_warning: float = ETA_STORAGE_WARNING,
    initial_yard_frac: float = INITIAL_YARD_OCCUPANCY_FRAC,
    rho_proactive: float = RHO_PROACTIVE,
) -> dict:
    """
    Information ladder (no controlled policy observes realised N or μ at planning):
      S0: uncontrolled realised process.
      S1: stylised static plan on historical weekday baseline day-total N_base.
      S2: forecast-informed plan on trailing public N_hat; blends toward planning capacity.
      S3: S2 + private mid-day state (soft or hard env).
      S3s-H: same as S3-H plus proactive deferral only under hard env.
      S_OPT: two-stage LP on planning C_h and the same N_hat (no realised-μ foresight).
    """
    C_real = hourly_capacity(mu, n_servers, capacity_kind=capacity_kind)
    C_plan = hourly_capacity(MU_MEAN, n_servers, capacity_kind=capacity_kind)
    base = scenario_base(scenario)
    hard = bool(hard_storage_env)
    proactive = is_proactive_scenario(scenario)
    peak_perfect = np.nan
    peak_foresight = np.nan
    proactive_vec = np.zeros(N_HOURS, dtype=float)
    if N_hat is None:
        N_hat = scale_day_total(I_hat, lambda0, I_ref)
    if N_base is None:
        N_base = scale_day_total(I_ref, lambda0, I_ref)
    kappa_used = forecast_kappa(I_hat, I_ref, kappa_max=tilt) if base in ("S2", "S3", "S3s") else 0.0

    if base == "S0":
        arrivals = peak
        plan_total = int(N)
    elif base == "S1":
        planned = allocate(N_base, static_appointment_weights(rng, smooth=static_smooth, peak_shape=peak_shape), rng)
        arrivals = realise_forecast_plan(planned, N, peak, compliance, rng)
        plan_total = int(N_base)
    elif base == "S2":
        planned = allocate(
            N_hat,
            forecast_weights(
                I_hat, I_ref, rng, tilt=tilt, smooth=static_smooth, peak_shape=peak_shape, capacity_kind=capacity_kind
            ),
            rng,
        )
        arrivals = realise_forecast_plan(planned, N, peak, compliance, rng)
        plan_total = int(N_hat)
    elif base in ("S3", "S3s"):
        planned = allocate(
            N_hat,
            forecast_weights(
                I_hat, I_ref, rng, tilt=tilt, smooth=static_smooth, peak_shape=peak_shape, capacity_kind=capacity_kind
            ),
            rng,
        )
        provisional = realise_forecast_plan(planned, N, peak, compliance, rng)
        mid = N_HOURS // 2
        first = provisional[:mid].copy()
        q, storage = _simulate_first_half(
            first, C_real, storage_cap, hard, initial_yard_frac=initial_yard_frac
        )
        arrivals, proactive_vec = residual_second_half(
            N, peak, first, q, storage, storage_cap, I_hat, I_ref, rng, compliance, tilt, C_plan, hard,
            smooth=static_smooth, peak_shape=peak_shape, capacity_kind=capacity_kind, proactive=proactive,
            eta_storage_warning=eta_storage_warning, rho_proactive=rho_proactive,
        )
        plan_total = int(N_hat)
    elif base == "S_OPT":
        planned_hat = lp_optimal_arrivals(N_hat, MU_MEAN, n_servers, C_plan=C_plan, capacity_kind=capacity_kind)
        planned_foresight = lp_optimal_arrivals(N, mu, n_servers, C_plan=C_real, capacity_kind=capacity_kind)
        peak_perfect = simulate_day(
            realise_forecast_plan(planned_hat, N, peak, 1.0, rng), mu, storage_cap, hard, n_servers,
            C_path=C_real, capacity_kind=capacity_kind,
        )["peak_queue"]
        peak_foresight = simulate_day(
            mix_compliance(planned_foresight, peak, compliance, total=N), mu, storage_cap, hard, n_servers,
            C_path=C_real, capacity_kind=capacity_kind,
        )["peak_queue"]
        arrivals = realise_forecast_plan(planned_hat, N, peak, compliance, rng)
        plan_total = int(N_hat)
    else:
        raise ValueError(scenario)

    metrics = simulate_day(
        arrivals, mu, storage_cap, hard, n_servers, C_path=C_real, capacity_kind=capacity_kind,
        proactive_defer=proactive_vec if proactive else None,
        initial_yard_frac=initial_yard_frac,
    )
    metrics.update(
        {
            "scenario": scenario,
            "compliance": compliance,
            "mu": mu,
            "mu_plan_lp": MU_MEAN,
            "storage_cap": storage_cap,
            "hard_storage": hard,
            "hard_storage_env": bool(hard_storage_env),
            "N": N,
            "N_hat": int(N_hat),
            "N_base": int(N_base),
            "plan_total": int(plan_total),
            "I": I,
            "I_hat": I_hat,
            "I_ref": float(I_ref),
            "kappa": float(kappa_used),
            "theta": THETA,
            "eta_storage_warning": float(eta_storage_warning),
            "initial_yard_frac": float(initial_yard_frac),
            "rho_proactive": float(rho_proactive),
            "eps": EPS,
            "lp_tol": LP_TOL,
            "gamma": tilt,
            "delta_clear": DELTA_CLEAR,
            "bpr_alpha": BPR_ALPHA,
            "bpr_beta": BPR_BETA,
            "n_servers": n_servers,
            "peak_shape": peak_shape,
            "static_smooth": static_smooth,
            "capacity_kind": capacity_kind,
            "peak_queue_perfect_compliance": float(peak_perfect) if base == "S_OPT" else np.nan,
            "peak_queue_foresight_lp_diagnostic": float(peak_foresight) if base == "S_OPT" else np.nan,
        }
    )
    return metrics



def select_days(df: pd.DataFrame, full_year: bool = True) -> pd.DataFrame:
    """All usable evaluation-year days; high/low weeks are symmetric regime tags only."""
    days = df.dropna(subset=["I", "portcalls_container"]).copy()
    week_I = days.groupby("week_end")["I"].mean()
    high_ends = set(week_I.nlargest(5).index.astype(str))
    low_ends = set(week_I.nsmallest(5).index.astype(str))  # symmetric with top-5 high weeks
    if not full_year:
        sample_idx: list[int] = []
        for _, g in days.groupby(days["date"].dt.month):
            sample_idx.extend(list(g.index[np.linspace(0, len(g) - 1, 3).astype(int)]))
        stress_idx = days.index[days["week_end"].isin(high_ends | low_ends)].tolist()
        days = days.loc[sorted(set(sample_idx) | set(stress_idx))].copy()
    days["regime"] = days["week_end"].map(
        lambda w: "high" if str(w) in high_ends else ("low" if str(w) in low_ends else "normal")
    )
    return days.reset_index(drop=True)


def sensitivity_day_index(n_days: int, stride: int = SENSITIVITY_DAY_STRIDE) -> np.ndarray:
    return np.arange(0, n_days, max(1, stride))


def run_main_experiments(
    lambda0: float = LAMBDA0_DEFAULT,
    tilt: float = KAPPA_MAX_DEFAULT,
    n_servers: int = 1,
    hard_storage_env: bool = False,
    peak_shape: str = "two_hump",
    full_year: bool = True,
    reps: int | None = None,
    day_indices: np.ndarray | None = None,
    static_smooth: float = STATIC_SMOOTH_DEFAULT,
    capacity_kind: str = CAPACITY_PROFILE_DEFAULT,
    scenarios: list[str] | None = None,
    eta_storage_warning: float = ETA_STORAGE_WARNING,
    initial_yard_frac: float = INITIAL_YARD_OCCUPANCY_FRAC,
    s1_baseline: str = "historical",
    storage_cap_mult: float | None = None,
    rho_proactive: float = RHO_PROACTIVE,
) -> pd.DataFrame:
    """Shared shocks per (day, rep) across scenarios for valid paired tests."""
    global I_REF
    df = load_intensity()
    I_ref = float(df.attrs.get("I_ref", I_REF))
    I_REF = I_ref
    days = select_days(df, full_year=full_year)
    if day_indices is not None:
        days = days.iloc[list(day_indices)].reset_index(drop=True)
    n_reps = REPS if reps is None else int(reps)
    scen_list = list(scenarios) if scenarios is not None else (
        list(SCENARIOS_HARD) if hard_storage_env else list(SCENARIOS_SOFT)
    )
    rows = []
    for i, row in days.iterrows():
        N = scale_day_total(float(row["I"]), lambda0, I_ref)
        N_hat = scale_day_total(float(row["I_hat"]), lambda0, I_ref)
        if s1_baseline not in {"historical", "recent"}:
            raise ValueError("s1_baseline must be 'historical' or 'recent'")
        baseline_col = "I_S1_base" if s1_baseline == "historical" else "I_S1_recent"
        N_base = scale_day_total(float(row[baseline_col]), lambda0, I_ref)
        for r in range(n_reps):
            rng_shock = np.random.default_rng(RNG_MASTER + int(i) * 100_000 + r)
            compliance = float(rng_shock.uniform(0.75, 0.95))
            mu = float(np.clip(rng_shock.normal(1.55, 0.18), 1.15, 2.10))
            if storage_cap_mult is None:
                storage_cap = STORAGE_CAP_BASE * float(rng_shock.choice(list(STORAGE_CAP_MULTS)))
            else:
                storage_cap = STORAGE_CAP_BASE * float(storage_cap_mult)
                _ = float(rng_shock.choice(list(STORAGE_CAP_MULTS)))  # keep RNG stream aligned
            peak = allocate(N, peak_weights(rng_shock, shape=peak_shape), rng_shock)
            I_hat = float(row["I_hat"])  # genuine trailing forecast; no artificial noise
            for s in scen_list:
                # S3s shares S3 RNG so proactive is the only intentional difference under hard env
                seed_base = scenario_base(s)
                if seed_base == "S3s":
                    seed_base = "S3"
                seed_idx = ["S0", "S1", "S2", "S3", "S_OPT"].index(seed_base)
                rng_s = np.random.default_rng(RNG_MASTER + int(i) * 100_000 + r * 10 + seed_idx)
                m = run_one(
                    s, N, float(row["I"]), I_ref, I_hat, rng_s, compliance, mu, storage_cap, peak,
                    tilt=tilt, n_servers=n_servers, hard_storage_env=hard_storage_env, peak_shape=peak_shape,
                    N_hat=N_hat, N_base=N_base, lambda0=lambda0, static_smooth=static_smooth,
                    capacity_kind=capacity_kind, eta_storage_warning=eta_storage_warning,
                    initial_yard_frac=initial_yard_frac, rho_proactive=rho_proactive,
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
                        "I_hat_base": float(row["I_hat_base"]),
                        "I_S1_base": float(row["I_S1_base"]),
                        "I_S1_recent": float(row["I_S1_recent"]),
                        "s1_baseline": s1_baseline,
                        "storage_cap_mult_fixed": (
                            None if storage_cap_mult is None else float(storage_cap_mult)
                        ),
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
        s0 = "S0-H" if "S0-H" in wide.columns else "S0"
        for s in [x for x in wide.columns if scenario_base(x) != "S0"]:
            # Absolute paired differences (Wilcoxon complementary); exclude near-zero S0 for peak only
            if metric == "peak_queue":
                mask = wide[s0] > ZERO_S0_PEAK_THRESHOLD
            else:
                mask = pd.Series(True, index=wide.index)
            diff = (wide.loc[mask, s0] - wide.loc[mask, s]).dropna()
            n_zero_s0 = int((~mask).sum()) if metric == "peak_queue" else 0
            if len(diff) < 5:
                continue
            # Exact Wilcoxon excludes zeros in diff via zero_method='wilcox'
            stat, p = stats.wilcoxon(diff, alternative="greater", zero_method="wilcox")
            rows.append(
                {
                    "level": level,
                    "metric": metric,
                    "scenario": s,
                    "n": len(diff),
                    "n_zero_s0_excluded": n_zero_s0,
                    "median_diff": float(diff.median()),
                    "wilcoxon_stat": float(stat),
                    "p_value": float(p),
                    "rank_biserial_r": rank_biserial_from_wilcoxon(diff),
                    "inference_role": "primary",
                }
            )
    else:
        # replication-level paired by (date, rep) — diagnostic only
        for s in [x for x in res["scenario"].unique() if scenario_base(x) != "S0"]:
            b = res[res.scenario == s].set_index(["date", "rep"])[metric]
            a = res[res.scenario == ("S0-H" if str(s).endswith("-H") else "S0")].set_index(["date", "rep"])[metric]
            diff = (a - b).dropna()
            if len(diff) < 5:
                continue
            if len(diff) > 5000:
                diff = diff.sample(5000, random_state=0)
            stat, p = stats.wilcoxon(diff, alternative="greater", zero_method="wilcox")
            rows.append(
                {
                    "level": level,
                    "metric": metric,
                    "scenario": s,
                    "n": len(diff),
                    "n_zero_s0_excluded": 0,
                    "median_diff": float(diff.median()),
                    "wilcoxon_stat": float(stat),
                    "p_value": float(p),
                    "rank_biserial_r": rank_biserial_from_wilcoxon(diff),
                    "inference_role": "diagnostic",
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm_adjust(out["p_value"].tolist())
        out["significant_holm_0_05"] = out["p_holm"] < 0.05
    return out


def acf_lag1(x: np.ndarray) -> float:
    """Lag-1 ACF using only adjacent calendar pairs that are both eligible.

    NaN ineligible days (e.g. zero-S0 exclusions) break adjacency rather than
    stitching non-adjacent eligible dates together.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan")
    left = x[:-1]
    right = x[1:]
    ok = np.isfinite(left) & np.isfinite(right)
    if int(ok.sum()) < 2:
        return float("nan")
    a = left[ok]
    b = right[ok]
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if denom <= EPS:
        return 0.0
    return float(np.dot(a, b) / denom)


def moving_block_bootstrap_mean_ci(
    series: np.ndarray,
    block_len: int = BLOCK_LEN_DEFAULT,
    n_boot: int = BLOCK_BOOT_REPS,
    alpha: float = 0.05,
    seed: int = BLOCK_BOOT_SEED,
) -> dict:
    """Percentile CI for a calendar-ordered mean under circular block bootstrap.

    Missing values are retained in their original calendar positions while blocks
    are resampled; each replicate averages its eligible observations only. This
    prevents zero-S0 exclusions from creating artificial temporal adjacency.
    """
    x = np.asarray(series, dtype=float)
    n_calendar = len(x)
    n_eligible = int(np.isfinite(x).sum())
    if n_eligible < max(5, block_len) or n_calendar == 0:
        return {
            "mean": float(np.nanmean(x)) if n_eligible else float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n": n_eligible,
            "n_calendar": n_calendar,
            "n_eligible": n_eligible,
            "block_len": block_len,
            "method": "insufficient_n",
        }
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n_calendar / block_len))
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        starts = rng.integers(0, n_calendar, size=n_blocks)
        sample = np.concatenate(
            [
                x[s : s + block_len]
                if s + block_len <= n_calendar
                else np.concatenate([x[s:], x[: (s + block_len) - n_calendar]])
                for s in starts
            ]
        )[:n_calendar]
        means[b] = np.nanmean(sample)
    return {
        "mean": float(np.nanmean(x)),
        "ci_low": float(np.quantile(means, alpha / 2)),
        "ci_high": float(np.quantile(means, 1 - alpha / 2)),
        "n": n_eligible,
        "n_calendar": n_calendar,
        "n_eligible": n_eligible,
        "block_len": block_len,
        "method": "circular_block_bootstrap",
    }


def tost_from_ci(mean: float, ci_low: float, ci_high: float, margin: float) -> tuple[bool, float]:
    """Equivalence if the (1-2α) CI for the mean difference lies inside (-margin, margin)."""
    if not np.isfinite(ci_low) or not np.isfinite(ci_high):
        return False, 1.0
    equiv = (ci_low > -margin) and (ci_high < margin)
    # Distance of CI from 0 as a crude p-proxy for logging (not a classical p-value)
    if equiv:
        p_proxy = 0.0
    else:
        # How far the CI protrudes outside the margin
        p_proxy = max(0.0, -margin - ci_low, ci_high - margin)
    return bool(equiv), float(p_proxy)



def _has(wide: pd.DataFrame, base: str) -> bool:
    return base in wide.columns or (base + "-H") in wide.columns


def _col_soft_hard(wide: pd.DataFrame, base: str) -> str:
    if base in wide.columns:
        return base
    if (base + "-H") in wide.columns:
        return base + "-H"
    raise KeyError(base)


def _wide_s0(wide: pd.DataFrame) -> pd.DataFrame:
    if "S0" in wide.columns:
        return wide
    if "S0-H" in wide.columns:
        return wide.rename(columns={"S0-H": "S0"})
    return wide

def relative_improvement_peak(wide: pd.DataFrame, scenario: str) -> pd.Series:
    """Peak-queue relative improvement vs S0 from day medians (descriptive / structural).

    Primary confirmatory contrasts use day_level_paired_relative_improvement
    (median of within-replication paired ratios), not this ratio-of-medians helper.
    """
    usable = wide["S0"] > ZERO_S0_PEAK_THRESHOLD
    return ((wide["S0"] - wide[scenario]) / wide["S0"]).where(usable)


def _resolve_scenario_name(available: set[str], name: str) -> str:
    if name in available:
        return name
    alt = name + "-H" if not str(name).endswith("-H") else str(name)[:-2]
    if alt in available:
        return alt
    return name


def day_level_paired_relative_improvement(
    res: pd.DataFrame, scenario: str, s0_name: str = "S0"
) -> pd.Series:
    """
    Paired Monte Carlo aggregation (primary relative path):
      Δ_{s,t,r} = 1 - Q_{s,t,r}/Q_{S0,t,r}  when Q_{S0,t,r} > 1,
      Δ_{s,t}   = median_r(Δ_{s,t,r}).
    """
    available = set(res["scenario"].astype(str))
    sc = _resolve_scenario_name(available, scenario)
    s0 = _resolve_scenario_name(available, s0_name)
    a = res.loc[res.scenario == s0, ["date", "rep", "peak_queue"]].rename(columns={"peak_queue": "q0"})
    b = res.loc[res.scenario == sc, ["date", "rep", "peak_queue"]].rename(columns={"peak_queue": "qs"})
    m = a.merge(b, on=["date", "rep"], how="inner")
    usable = m["q0"] > ZERO_S0_PEAK_THRESHOLD
    m["delta"] = np.where(usable, 1.0 - m["qs"] / m["q0"], np.nan)
    return m.groupby("date", sort=True)["delta"].median()


def day_level_paired_absolute_diff(
    res: pd.DataFrame, left: str, right: str, metric: str
) -> pd.Series:
    """Median across replications of within-rep paired absolute differences (CRN)."""
    available = set(res["scenario"].astype(str))
    lc = _resolve_scenario_name(available, left)
    rc = _resolve_scenario_name(available, right)
    a = res.loc[res.scenario == lc, ["date", "rep", metric]].rename(columns={metric: "left"})
    b = res.loc[res.scenario == rc, ["date", "rep", metric]].rename(columns={metric: "right"})
    m = a.merge(b, on=["date", "rep"], how="inner")
    m["diff"] = m["left"] - m["right"]
    return m.groupby("date", sort=True)["diff"].median()


def paired_day_contrast_series(
    res: pd.DataFrame, metric: str, left: str, right: str, relative_to_s0: bool = False
) -> pd.Series:
    """
    Day-level paired contrasts after within-day Monte Carlo aggregation.

    Relative (peak_queue only): median_r of paired ratios first, then
    g_t = Δ_{left,t} - Δ_{right,t}, with ineligible zero-S0 dates kept as NaN
    calendar positions for CBB.

    Absolute: median_r of (left_{t,r} - right_{t,r}).
    """
    available = set(res["scenario"].astype(str))
    left_c = _resolve_scenario_name(available, left)
    right_c = _resolve_scenario_name(available, right)
    s0_c = _resolve_scenario_name(available, "S0")
    all_dates = sorted(res.loc[res.scenario == s0_c, "date"].unique())
    if relative_to_s0:
        if metric != "peak_queue":
            raise ValueError(
                f"Relative-to-S0 contrasts are defined for peak_queue only (got {metric}); "
                "use absolute paired differences for other metrics."
            )
        d_l = day_level_paired_relative_improvement(res, left_c, s0_c)
        d_r = day_level_paired_relative_improvement(res, right_c, s0_c)
        return (d_l - d_r).reindex(all_dates)
    return day_level_paired_absolute_diff(res, left_c, right_c, metric)


def diagnose_and_infer_contrast(
    series: pd.Series,
    margin: float | None = None,
    label: str = "",
    primary: bool = False,
    block_len: int = BLOCK_LEN_DEFAULT,
    mbb_alpha: float = MBB_CI_ALPHA,
) -> dict:
    """
    Report ACF diagnostics and dependence-aware circular-block bootstrap CIs.
    Equivalence is a CI-based assessment of the mean paired effect.
    """
    d_calendar = series.to_numpy(dtype=float)
    d = d_calendar[np.isfinite(d_calendar)]
    acf1 = acf_lag1(d_calendar)
    n = len(d)
    mean = float(np.mean(d)) if n else float("nan")
    se = float(stats.sem(d)) if n > 1 else float("nan")
    if n > 1 and np.isfinite(se):
        tcrit = float(stats.t.ppf(0.975, n - 1))
        ordinary_ci = (mean - tcrit * se, mean + tcrit * se)
    else:
        ordinary_ci = (float("nan"), float("nan"))

    boot = moving_block_bootstrap_mean_ci(d_calendar, block_len=block_len, alpha=mbb_alpha)
    ci_low, ci_high = boot["ci_low"], boot["ci_high"]
    method = "circular_block_bootstrap"

    row = {
        "contrast": label,
        "n_days": n,
        "n_calendar_days": int(len(d_calendar)),
        "n_zero_s0_excluded": int(len(d_calendar) - n),
        "estimand": "mean across calendar-day paired effects",
        "mean_diff": mean,
        "median_diff": float(np.median(d)) if n else float("nan"),
        "acf_lag1": acf1,
        "dependence_flag": bool(np.isfinite(acf1) and abs(acf1) > 0.10),  # diagnostic only
        "ci_method": method,
        "ci_low": float(ci_low) if ci_low == ci_low else float("nan"),
        "ci_high": float(ci_high) if ci_high == ci_high else float("nan"),
        "ordinary_ci_low": float(ordinary_ci[0]) if ordinary_ci[0] == ordinary_ci[0] else float("nan"),
        "ordinary_ci_high": float(ordinary_ci[1]) if ordinary_ci[1] == ordinary_ci[1] else float("nan"),
        "block_len": boot.get("block_len"),
        "mbb_alpha": mbb_alpha,
        "primary": primary,
    }
    if margin is not None:
        equiv, _ = tost_from_ci(mean, ci_low, ci_high, margin)
        row.update(
            {
                "margin": margin,
                "equivalent_at_margin": bool(equiv),  # MBB CI decision
                "ci_tost_decision": bool(equiv),
            }
        )
    return row



def tost_equivalence(res: pd.DataFrame, metric: str, margin: float, primary: bool = False) -> pd.DataFrame:
    """Equivalence vs S1: relative peak improvement; absolute contrasts for other metrics."""
    rows = []
    use_relative = metric == "peak_queue"
    for s in [x for x in (SCENARIOS_SOFT + SCENARIOS_HARD) if scenario_base(x) in ("S2", "S3", "S3s", "S_OPT") and x in day_medians(res, metric).columns]:
        series = paired_day_contrast_series(res, metric, s, "S1", relative_to_s0=use_relative)
        label = f"{s}_vs_S1_rel_improvement" if use_relative else f"{s}_vs_S1_abs_{metric}"
        row = diagnose_and_infer_contrast(
            series,
            margin=margin if use_relative else None,
            label=label,
            primary=primary and margin == TOST_PRIMARY_MARGIN and metric == "peak_queue",
        )
        row.update(
            {
                "metric": metric,
                "compare": label,
                "n_zero_s0_excluded": int(series.isna().sum()) if use_relative else 0,
                "n_calendar_days": int(len(series)),
                "n_days": int(series.notna().sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def tost_absolute_peak(res: pd.DataFrame, margin: float) -> pd.DataFrame:
    """Absolute peak-queue TOST: S_OPT − S1 on day medians."""
    series = paired_day_contrast_series(res, "peak_queue", "S_OPT", "S1", relative_to_s0=False)
    row = diagnose_and_infer_contrast(series, margin=margin, label="S_OPT_minus_S1_absolute", primary=False)
    row.update(
        {
            "metric": "peak_queue",
            "compare": "S_OPT_minus_S1_absolute",
        }
    )
    return pd.DataFrame([row])


def apply_holm_to_primary_tost(tost_df: pd.DataFrame) -> pd.DataFrame:
    """Label the pre-specified CI-equivalence family; no t-test p-values are used."""
    out = tost_df.copy()
    out["tost_family"] = "sensitivity"
    mask = (
        (out["metric"] == "peak_queue")
        & (out["margin"] == TOST_PRIMARY_MARGIN)
        & (out["compare"].isin(["S2_vs_S1_rel_improvement", "S3_vs_S1_rel_improvement", "S3s_vs_S1_rel_improvement", "S_OPT_vs_S1_rel_improvement"]))
    )
    if mask.any():
        out.loc[mask, "tost_family"] = "primary_rel_peak_5pp"
        # Confirmatory contrast for adaptive vs S1 is S2; others labelled secondary within family
        out.loc[mask & (out["compare"] == "S2_vs_S1_rel_improvement"), "tost_role"] = "confirmatory"
        out.loc[mask & (out["compare"] != "S2_vs_S1_rel_improvement"), "tost_role"] = "secondary"
    return out


def mc_convergence_multi(res: pd.DataFrame) -> pd.DataFrame:
    """
    Convergence across primary outcomes and contrasts between R=30 and R=40.
    Criterion: relative change in day-pooled median < 1% (or absolute < 0.5 trucks for near-zero).
    """
    targets = [
        ("S0", "peak_queue", None),
        ("S0", "p90_wait_min", None),
        ("S1", "peak_queue", "S0"),  # S1 vs S0 gap via later contrast rows
    ]
    rows = []
    max_rep = int(res["rep"].max()) + 1

    def _day_median(df, scenario, metric):
        return df[df.scenario == scenario].groupby("date")[metric].median()

    for r_cut in range(1, max_rep + 1):
        sub = res[res.rep < r_cut]
        s0 = _day_median(sub, "S0", "peak_queue")
        s1 = _day_median(sub, "S1", "peak_queue")
        s2 = _day_median(sub, "S2", "peak_queue")
        sopt = _day_median(sub, "S_OPT", "peak_queue")
        s0_p90 = _day_median(sub, "S0", "p90_wait_min")
        rows.append(
            {
                "reps_used": r_cut,
                "S0_peak_median": float(s0.median()),
                "S0_p90_wait_median": float(s0_p90.median()),
                "S1_minus_S0_peak_median": float((s1 - s0).median()),
                "S2_minus_S1_peak_median": float((s2 - s1).median()),
                "S_OPT_minus_S1_peak_median": float((sopt - s1).median()),
                "S0_peak_p10": float(s0.quantile(0.10)),
                "S0_peak_p90": float(s0.quantile(0.90)),
            }
        )
    conv = pd.DataFrame(rows)
    # Stability flags R30 vs R40
    if len(conv) >= 40:
        a = conv[conv.reps_used == 30].iloc[0]
        b = conv[conv.reps_used == 40].iloc[0]
        stab = {}
        tol_map = {
            "S0_peak_median": ("trucks", MC_TOL_PEAK_TRUCKS),
            "S0_p90_wait_median": ("minutes", MC_TOL_WAIT_MIN),
            "S1_minus_S0_peak_median": ("trucks", MC_TOL_PEAK_TRUCKS),
            "S2_minus_S1_peak_median": ("trucks", MC_TOL_PEAK_TRUCKS),
            "S_OPT_minus_S1_peak_median": ("trucks", MC_TOL_PEAK_TRUCKS),
        }
        for col, (unit, abs_tol) in tol_map.items():
            base = abs(float(a[col]))
            delta = abs(float(b[col]) - float(a[col]))
            ok = (delta <= abs_tol) or (base > abs_tol and delta / base <= MC_TOL_RELATIVE)
            stab[f"stable_30_40_{col}"] = bool(ok)
            stab[f"abs_change_30_40_{col}"] = float(delta)
            stab[f"unit_{col}"] = unit
            stab[f"rel_change_30_40_{col}"] = float(delta / base) if base > EPS else float(delta)
        (OUT / "mc_convergence_stability.json").write_text(json.dumps(stab, indent=2), encoding="utf-8")
        conv.attrs["stability"] = stab
    return conv


def mc_convergence(res: pd.DataFrame, metric: str = "peak_queue", scenario: str = "S0") -> pd.DataFrame:
    """Legacy single-series convergence (kept for Fig. 2d compatibility)."""
    multi = mc_convergence_multi(res)
    return multi.rename(
        columns={
            "S0_peak_median": "median",
            "S0_peak_p10": "p10",
            "S0_peak_p90": "p90",
        }
    )[["reps_used", "median", "p10", "p90"]]


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


def run_sensitivities(*, structural_only: bool = False) -> None:
    """
    Structural sensitivities:
      - full-year decisive grid (static smooth × capacity profile) with reduced reps;
      - strided deterministic subset for secondary parameter sweeps (unless structural_only).
    Stride rule: day indices 0,6,12,... over the ordered full-year sample (pre-specified).
    """
    global THETA, DELTA_CLEAR
    import os

    structural_only = structural_only or (
        os.environ.get("STRUCTURAL_ONLY", "").strip() in {"1", "true", "TRUE", "yes"}
    )
    df_all = load_intensity()
    n_days = len(select_days(df_all, full_year=True))
    day_idx = sensitivity_day_index(n_days)
    sens_kwargs = dict(full_year=True, reps=15, day_indices=day_idx)

    def _rob_row(res, *, smooth, cap, peak_shape, hard_env, sample, reps, factor):
        peaks = res.groupby("scenario")["peak_queue"].median()
        wide = day_medians(res, "peak_queue")
        usable = _wide_s0(wide)["S0"] > ZERO_S0_PEAK_THRESHOLD
        order = peaks.sort_values().index.tolist()
        return {
            "factor": factor,
            "static_smooth": smooth,
            "capacity_kind": cap,
            "peak_shape": peak_shape,
            "hard_storage_env": hard_env,
            "sample": sample,
            "n_days": int(res["date"].nunique()),
            "reps": reps,
            "median_peak_S0": float(peaks.get("S0", peaks.get("S0-H", float("nan")))),
            "median_peak_S1": float(peaks.get("S1", peaks.get("S1-H", float("nan")))),
            "median_peak_S2": float(peaks.get("S2", peaks.get("S2-H", float("nan")))),
            "median_peak_S3": float(peaks.get("S3", peaks.get("S3-H", float("nan")))),
            "median_peak_S3s": float(peaks.get("S3s", peaks.get("S3s-H", float("nan")))) if ("S3s" in peaks.index or "S3s-H" in peaks.index) else float("nan"),
            "median_peak_S_OPT": float(peaks.get("S_OPT", peaks.get("S_OPT-H", float("nan")))),
            "rel_imp_S1": float(relative_improvement_peak(_wide_s0(wide), _col_soft_hard(wide, "S1")).median()),
            "rel_imp_S2": float(relative_improvement_peak(_wide_s0(wide), _col_soft_hard(wide, "S2")).median()),
            "rel_imp_S_OPT": float(relative_improvement_peak(_wide_s0(wide), _col_soft_hard(wide, "S_OPT")).median()),
            "S2_minus_S1_rel": float(
                (
                    relative_improvement_peak(_wide_s0(wide), _col_soft_hard(wide, "S2"))
                    - relative_improvement_peak(_wide_s0(wide), _col_soft_hard(wide, "S1"))
                ).median()
            ),
            "S3s_minus_S3_peak": (
                float((wide[_col_soft_hard(wide, "S3s")] - wide[_col_soft_hard(wide, "S3")]).median())
                if _has(wide, "S3s") and _has(wide, "S3")
                else float("nan")
            ),
            "S_OPT_lt_S1": bool(peaks.get("S_OPT", peaks.get("S_OPT-H", 0)) < peaks.get("S1", peaks.get("S1-H", 0))),
            "ranking": ">".join(order),
        }

    def _flush_rob(rows):
        pd.DataFrame(rows).to_csv(OUT / "robustness_static_capacity_grid.csv", index=False)
        pd.DataFrame(rows).to_csv(OUT / "robustness_structural_full_year.csv", index=False)

    # Decisive FULL-YEAR structural cells (can change headline ranking)
    STRUCT_REPS = 15
    rob_rows = []
    for smooth in [0.25, 0.50, 0.75, 1.00]:
        for cap in ["flat", "mild", "strong"]:
            print(f"Structural FULL YEAR: smooth={smooth}, capacity={cap}", flush=True)
            res = run_main_experiments(
                static_smooth=smooth, capacity_kind=cap, full_year=True, reps=STRUCT_REPS
            )
            rob_rows.append(
                _rob_row(
                    res, smooth=smooth, cap=cap, peak_shape="two_hump", hard_env=False,
                    sample="full_year", reps=STRUCT_REPS, factor="smooth_x_capacity",
                )
            )
            _flush_rob(rob_rows)
    for shape in ["two_hump", "morning", "flat_uncoord"]:
        print(f"Structural FULL YEAR: peak_shape={shape}", flush=True)
        res = run_main_experiments(peak_shape=shape, full_year=True, reps=STRUCT_REPS)
        rob_rows.append(
            _rob_row(
                res, smooth=STATIC_SMOOTH_DEFAULT, cap=CAPACITY_PROFILE_DEFAULT, peak_shape=shape,
                hard_env=False, sample="full_year", reps=STRUCT_REPS, factor="s0_peak_shape",
            )
        )
        _flush_rob(rob_rows)
    print("Structural FULL YEAR: common hard-storage environment", flush=True)
    res = run_main_experiments(hard_storage_env=True, scenarios=SCENARIOS_HARD, full_year=True, reps=STRUCT_REPS)
    rob_rows.append(
        _rob_row(
            res, smooth=STATIC_SMOOTH_DEFAULT, cap=CAPACITY_PROFILE_DEFAULT, peak_shape="two_hump",
            hard_env=True, sample="full_year", reps=STRUCT_REPS, factor="hard_storage_env",
        )
    )
    _flush_rob(rob_rows)
    print(f"Wrote structural full-year robustness ({len(rob_rows)} cells)", flush=True)

    if structural_only:
        print("STRUCTURAL_ONLY set — skipping secondary strided sweeps.", flush=True)
        return

    rows = []
    for lam in [900, 1200, 1400, 1500]:
        res = run_main_experiments(lambda0=lam, **sens_kwargs)
        for s in SCENARIOS:
            g = res[res.scenario == s]
            rows.append(
                {
                    "lambda0": lam,
                    "scenario": s,
                    "median_peak_queue": g["peak_queue"].median(),
                    "median_waiting_proxy": g["waiting_proxy"].median(),
                    "n_days_sensitivity": res["date"].nunique(),
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "sensitivity_lambda0.csv", index=False)

    srows = []
    for ns in [1, 2, 3]:
        res = run_main_experiments(n_servers=ns, **sens_kwargs)
        for s in SCENARIOS:
            g = res[res.scenario == s]
            srows.append(
                {
                    "n_servers": ns,
                    "scenario": s,
                    "median_peak_queue": g["peak_queue"].median(),
                    "median_mean_wait_min": g["mean_wait_min"].median(),
                    "median_p90_wait_min": g["p90_wait_min"].median(),
                    "median_waiting_proxy": g["waiting_proxy"].median(),
                    "median_throughput": g["throughput"].median(),
                }
            )
        peaks = res.groupby("scenario")["peak_queue"].median()
        print(f"S={ns} peak medians:", peaks.round(1).to_dict())
    pd.DataFrame(srows).to_csv(OUT / "sensitivity_n_servers.csv", index=False)

    rows = []
    for tilt in [0.0, 0.10, 0.20, 0.40]:
        res = run_main_experiments(tilt=tilt, **sens_kwargs)
        wide = day_medians(res, "peak_queue")
        for s in ["S1", "S2", "S3"]:
            imp = ((wide["S0"] - wide[s]) / wide["S0"].replace(0, np.nan)).median()
            rows.append({"tilt": tilt, "scenario": s, "median_rel_improvement_peak_queue": float(imp)})
    pd.DataFrame(rows).to_csv(OUT / "sensitivity_forecast_tilt.csv", index=False)

    # S0 peak-shape sensitivity (1A item 4)
    prows = []
    for shape in ["two_hump", "morning", "flat_uncoord"]:
        res = run_main_experiments(peak_shape=shape, **sens_kwargs)
        for s in SCENARIOS:
            g = res[res.scenario == s]
            prows.append(
                {
                    "peak_shape": shape,
                    "scenario": s,
                    "median_peak_queue": g["peak_queue"].median(),
                    "median_waiting_proxy": g["waiting_proxy"].median(),
                }
            )
    pd.DataFrame(prows).to_csv(OUT / "sensitivity_s0_peak_shape.csv", index=False)

    # Common hard-storage environment (all scenarios under hard binding)
    hrows = []
    for hard in [False, True]:
        scen = SCENARIOS_HARD if hard else SCENARIOS_SOFT
        res = run_main_experiments(hard_storage_env=hard, scenarios=scen, **sens_kwargs)
        for s in scen:
            g = res[res.scenario == s]
            hrows.append(
                {
                    "hard_storage_env": hard,
                    "scenario": s,
                    "median_peak_queue": g["peak_queue"].median(),
                    "median_deferred": g["deferred"].median(),
                    "median_storage_pressure": g["storage_pressure"].median(),
                    "median_waiting_proxy": g["waiting_proxy"].median(),
                }
            )
    pd.DataFrame(hrows).to_csv(OUT / "sensitivity_hard_storage_env.csv", index=False)

    param_rows = []
    df = load_intensity()
    I_mean = float(df["I"].mean())
    days = select_days(df, full_year=True).iloc[list(day_idx)].reset_index(drop=True)
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
                I_hat = float(row["I_hat_base"])
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
                I_hat = float(row["I_hat_base"])
                rng_s = np.random.default_rng(RNG_MASTER + 19 + int(i) * 97 + r * 13)
                m = run_one(
                    "S3s-H", N, float(row["I"]), I_mean, I_hat, rng_s, compliance, mu, storage_cap, peak,
                    hard_storage_env=True,
                )
                local.append(m)
        g3 = pd.DataFrame(local)
        param_rows.append(
            {
                "param": "delta_clear",
                "value": delta,
                "scenario": "S3s-H",
                "median_rel_improvement_peak_queue": np.nan,
                "median_peak": float(g3["peak_queue"].median()),
                "median_storage_pressure": float(g3["storage_pressure"].median()),
                "mean_deferred": float(g3["deferred"].mean()),
            }
        )
    DELTA_CLEAR = 0.25
    pd.DataFrame(param_rows).to_csv(OUT / "sensitivity_theta_delta.csv", index=False)

    rows = []
    for w_c, w_v, label in [(0.5, 0.5, "0.5/0.5"), (0.6, 0.4, "0.6/0.4"), (0.7, 0.3, "0.7/0.3")]:
        d = df_all.copy()
        d["I"] = intensity_with_weights(d, w_c, w_v)
        I_mean = float(d["I"].mean())
        days = select_days(d, full_year=True).iloc[list(day_idx)].reset_index(drop=True)
        local_rows = []
        for i, row in days.iterrows():
            N = int(max(200, round(LAMBDA0_DEFAULT * float(row["I"]) / I_mean)))
            for r in range(15):
                rng_shock = np.random.default_rng(7 + i * 1000 + r + hash(label) % 1000)
                compliance = float(rng_shock.uniform(0.75, 0.95))
                mu = float(np.clip(rng_shock.normal(1.55, 0.18), 1.15, 2.10))
                storage_cap = 450.0 * float(rng_shock.choice([0.80, 1.00, 1.20]))
                peak = allocate(N, peak_weights(rng_shock), rng_shock)
                I_hat = float(row["I_hat_base"])
                for s in ["S0", "S1"]:
                    rng_s = np.random.default_rng(11 + i + r + (0 if s == "S0" else 1))
                    m = run_one(s, N, float(row["I"]), I_mean, I_hat, rng_s, compliance, mu, storage_cap, peak)
                    local_rows.append(
                        {"scenario": s, "peak_queue": m["peak_queue"], "date": row["date"].strftime("%Y-%m-%d")}
                    )
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

    main = pd.read_csv(OUT / "scenario_day_replications_hard.csv")
    main["cap_mult"] = (main["storage_cap"] / 450.0).round(2)
    stor = (
        main[main.scenario == "S3s-H"]
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



def _write_run_meta(n_days: int) -> None:
    meta = {
        "redesign": "methods_storage_fairness_final",
        "eps": EPS,
        "lp_tol": LP_TOL,
        "theta": THETA,
        "gamma_tilt": GAMMA_TILT,
        "delta_clear": DELTA_CLEAR,
        "alpha_in": ALPHA_IN,
        "mu_mean_lp_plan": MU_MEAN,
        "bpr_alpha": BPR_ALPHA,
        "bpr_beta": BPR_BETA,
        "n_servers_default": 1,
        "n_servers_sensitivities": [1, 2, 3],
        "storage_cap_base": 450.0,
        "lambda0_default": LAMBDA0_DEFAULT,
        "reps": REPS,
        "n_days_main": n_days,
        "sample_rule": "all usable 2024 PortWatch Tema days; high/low weeks = top-5 / bottom-5 weekly mean I tags",
        "sensitivity_day_rule": f"deterministic stride {SENSITIVITY_DAY_STRIDE} on ordered full-year sample (secondary only)",
        "inference_primary": "calendar-ordered day-level paired effects; circular block bootstrap 90% percentile CI for the across-day mean S2–S1 gap; ACF/time diagnostics; Wilcoxon complementary",
        "inference_diagnostic": "replication-level Wilcoxon only",
        "tost_primary_margin": TOST_PRIMARY_MARGIN,
        "tost_sensitivity_margins": list(TOST_SENS_MARGINS),
        "tost_abs_peak_margins": list(TOST_ABS_PEAK_MARGINS),
        "tost_family": "primary relative peak-queue at 5pp via MBB CI; confirmatory contrast S2 vs S1",
        "static_smooth_default": STATIC_SMOOTH_DEFAULT,
        "capacity_profile_default": CAPACITY_PROFILE_DEFAULT,
        "hist_years": list(HIST_YEARS),
        "eval_year": EVAL_YEAR,
        "I_ref": float(I_REF),
        "zero_s0_threshold": ZERO_S0_THRESHOLD,
        "kappa_max_default": KAPPA_MAX_DEFAULT,
        "block_lens": list(BLOCK_LENS),
        "mbb_ci_alpha": MBB_CI_ALPHA,
        "forecast": "trailing 7-day mean of past I; no artificial lognormal noise",
        "s1_day_total": "main: weekday mean intensity from 2019–2023 scaled by lambda0/I_ref; sensitivity: expanding past-only weekday mean",
        "s2_rule": "w=(1-kappa)*w_S1 + kappa*w_capacity; kappa monotone in forecast pressure",
        "kpi_labels": {
            "mean_wait_min": "waiting-time proxy (minutes)",
            "p90_wait_min": "p90 waiting-time proxy (minutes)",
            "corridor_delay_index": "BPR corridor congestion index (not physical travel time)",
            "waiting_proxy": "assumption-weighted composite score (SI/secondary)",
            "throughput": "served trucks",
            "deferred": "proactive deferred appointments only",
        },
        "scenarios_soft": SCENARIOS_SOFT,
        "scenarios_hard": SCENARIOS_HARD,
        "scenarios": SCENARIOS_SOFT,
        "capacity": "stylised C_h = 60*mu*S*s_h; reference=mild; flat/strong = heterogeneity bounds",
        "s1": "stylised static appointment on historical weekday baseline N_base",
        "information_sets": {
            "S0": "uncontrolled realised N",
            "S1": "historical weekday baseline N_base + stylised static weights",
            "S2": "trailing public N_hat + capacity-responsive blend",
            "S3_S3s": "S2 + observed private mid-queue (and hard-yard occupancy); trigger and residual weights use planning capacity; S3s proactive deferral separate from yard binding",
            "S_OPT": "two-stage LP on planning C_h and N_hat; no realised-mu foresight",
        },
        "hard_storage": "common hard env S0-H…S_OPT-H; S3s-H proactive only; soft primary non-binding (no 1.8 clip)",
        "mbb": {
            "implementation": "custom circular block bootstrap (NumPy)",
            "n_boot": BLOCK_BOOT_REPS,
            "seed": BLOCK_BOOT_SEED,
            "ci": "percentile",
            "alpha": MBB_CI_ALPHA,
            "primary_block": BLOCK_LEN_DEFAULT,
            "sensitivity_blocks": list(BLOCK_LENS),
        },
        "s2_estimand": (
            "Within each day–replication: Delta_Q,s,t,r = 1 - Q_s,t,r / Q_S0,t,r when Q_S0>1; "
            "then Delta_Q,s,t = median_r(Delta_Q,s,t,r); g_t = Delta_Q,S2,t - Delta_Q,S1,t; "
            "G = mean_t(g_t); CBB retains ineligible zero-S0 dates as NaN calendar positions"
        ),
        "proactive_rule": {
            "eta_storage_warning": ETA_STORAGE_WARNING,
            "rho_proactive": RHO_PROACTIVE,
            "initial_yard_frac": INITIAL_YARD_OCCUPANCY_FRAC,
            "trigger": "hard_storage and Y_mid > eta * C^Y",
            "D_pro": "round(rho * N_rem)",
            "selection": (
                "deterministic peel from highest second-half booked truck counts first; "
                "ties broken by earlier hour index; stops at exactly D_pro; presented >= 0"
            ),
        },
        "storage_cap": {
            "base": STORAGE_CAP_BASE,
            "main_draw": "shared random multiplier in {0.80,1.00,1.20} per day-rep",
            "hard_sensitivity": "fixed multipliers 0.80/1.00/1.20 for S3s-H vs S3-H (full year preferred)",
        },
        "soft_yard_state": "notional yard-load relative to reference C^Y (not physically clipped)",
        "hist_minmax_clipping": False,
    }
    (OUT / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")



def _summarise_env(res: pd.DataFrame, suffix: str = "") -> None:
    """Write scenario summaries for soft (suffix="") or hard (suffix="_hard")."""
    metrics = [
        "peak_queue", "mean_wait_min", "p90_wait_min", "corridor_delay_index",
        "waiting_proxy", "storage_pressure", "deferred", "deferred_physical",
        "deferred_proactive", "throughput",
    ]
    rows = []
    for s in res["scenario"].unique():
        g = res[res.scenario == s]
        for m in metrics:
            if m not in g.columns:
                continue
            rows.append(
                {
                    "scenario": s,
                    "metric": m,
                    "median": float(g[m].median()),
                    "p10": float(g[m].quantile(0.10)),
                    "p90": float(g[m].quantile(0.90)),
                    "mean": float(g[m].mean()),
                }
            )
    pd.DataFrame(rows).to_csv(OUT / f"summary_by_scenario{suffix}.csv", index=False)
    # Peak relative improvements vs S0: median of paired within-rep ratios
    improv = []
    for s in sorted(res["scenario"].unique()):
        if scenario_base(s) == "S0":
            continue
        ser = day_level_paired_relative_improvement(res, s)
        improv.append(
            {
                "scenario": s,
                "median_rel_improvement_peak_queue": float(ser.median()),
                "mean_rel_improvement_peak_queue": float(ser.mean()),
                "n_days": int(ser.notna().sum()),
                "aggregation": "median_of_paired_replication_ratios",
            }
        )
    pd.DataFrame(improv).to_csv(OUT / f"improvement_vs_S0{suffix}.csv", index=False)


def final_audit_sensitivities(res: pd.DataFrame) -> None:
    """Secondary, fixed-seed final-audit checks; no headline scenario changes."""
    full_days = select_days(load_intensity(), full_year=True)
    idx = sensitivity_day_index(len(full_days))

    # Eta / initial-yard state trade-off, evaluated only in the common hard env.
    rows = []
    for eta in (0.75, 0.85, 0.95):
        for y0 in (0.20, 0.40, 0.60):
            hard = run_main_experiments(
                hard_storage_env=True,
                scenarios=["S3-H", "S3s-H"],
                day_indices=idx,
                reps=15,
                eta_storage_warning=eta,
                initial_yard_frac=y0,
            )
            wide = hard.groupby(["date", "scenario"])[
                ["peak_queue", "throughput", "deferred_proactive"]
            ].median().unstack("scenario")
            for metric, direction in [
                ("peak_queue", "S3s-H minus S3-H; lower is better"),
                ("throughput", "S3s-H minus S3-H; lower is service cost"),
                ("deferred_proactive", "S3s-H minus S3-H; proactive appointments"),
            ]:
                diff = wide[(metric, "S3s-H")] - wide[(metric, "S3-H")]
                rows.append(
                    {
                        "eta_storage_warning": eta,
                        "initial_yard_frac": y0,
                        "n_days": int(len(diff)),
                        "metric": metric,
                        "direction": direction,
                        "mean_paired_difference": float(diff.mean()),
                        "median_paired_difference": float(diff.median()),
                        "p10_paired_difference": float(diff.quantile(0.10)),
                        "p90_paired_difference": float(diff.quantile(0.90)),
                        "fraction_days_s3s_deferred": float(
                            (wide[("deferred_proactive", "S3s-H")] > 0).mean()
                        ),
                    }
                )
    pd.DataFrame(rows).to_csv(OUT / "sensitivity_eta_initial_yard.csv", index=False)

    # ρ proactive-deferral fraction (strided deterministic subset).
    rho_rows = []
    for rho in (0.10, 0.25, 0.40):
        hard = run_main_experiments(
            hard_storage_env=True,
            scenarios=["S3-H", "S3s-H"],
            day_indices=idx,
            reps=15,
            rho_proactive=rho,
            storage_cap_mult=1.0,
        )
        for metric in ["peak_queue", "throughput", "deferred_proactive"]:
            series = paired_day_contrast_series(hard, metric, "S3s-H", "S3-H", relative_to_s0=False)
            d = series.dropna()
            rho_rows.append(
                {
                    "rho_proactive": rho,
                    "n_days": int(len(d)),
                    "metric": metric,
                    "mean_S3sH_minus_S3H": float(d.mean()) if len(d) else float("nan"),
                    "median_S3sH_minus_S3H": float(d.median()) if len(d) else float("nan"),
                    "fraction_days_s3s_deferred": float(
                        (
                            hard.loc[hard.scenario == "S3s-H"]
                            .groupby("date")["deferred_proactive"]
                            .median()
                            > 0
                        ).mean()
                    ),
                }
            )
    pd.DataFrame(rho_rows).to_csv(OUT / "sensitivity_rho_proactive.csv", index=False)

    # Fixed C^Y multipliers for S3s-H vs S3-H (full-year preferred; R=20 for tractability).
    cy_rows = []
    for mult in STORAGE_CAP_MULTS:
        hard = run_main_experiments(
            hard_storage_env=True,
            scenarios=["S3-H", "S3s-H"],
            reps=20,
            storage_cap_mult=float(mult),
            rho_proactive=RHO_PROACTIVE,
        )
        for metric in ["peak_queue", "throughput", "deferred_proactive", "storage_pressure"]:
            series = paired_day_contrast_series(hard, metric, "S3s-H", "S3-H", relative_to_s0=False)
            d = series.dropna()
            max_y_over_c = float(
                (hard["max_storage"] / hard["storage_cap"]).max()
            )
            cy_rows.append(
                {
                    "storage_cap_mult": float(mult),
                    "C_Y": STORAGE_CAP_BASE * float(mult),
                    "n_days": int(hard["date"].nunique()),
                    "reps": 20,
                    "metric": metric,
                    "mean_S3sH_minus_S3H": float(d.mean()) if len(d) else float("nan"),
                    "median_S3sH_minus_S3H": float(d.median()) if len(d) else float("nan"),
                    "fraction_days_s3s_deferred": float(
                        (
                            hard.loc[hard.scenario == "S3s-H"]
                            .groupby("date")["deferred_proactive"]
                            .median()
                            > 0
                        ).mean()
                    ),
                    "max_Y_over_C_Y": max_y_over_c,
                }
            )
    pd.DataFrame(cy_rows).to_csv(OUT / "sensitivity_CY_hard_storage.csv", index=False)

    # Freshness robustness: main historical S1 versus a recent past-only weekday
    # static baseline. S2 remains unchanged and no new headline scenario is added.
    fresh = run_main_experiments(
        day_indices=idx, reps=15, scenarios=["S0", "S1", "S2"], s1_baseline="recent"
    )
    series = paired_day_contrast_series(fresh, "peak_queue", "S2", "S1", relative_to_s0=True)
    row = diagnose_and_infer_contrast(
        series,
        margin=TOST_PRIMARY_MARGIN,
        label="S2_vs_recent_static_S1_rel_improvement",
        primary=False,
    )
    row.update(
        {
            "baseline": "expanding past-only weekday mean",
            "n_days_sensitivity": int(len(series)),
            "reps": 15,
            "sample": f"deterministic stride {SENSITIVITY_DAY_STRIDE}",
        }
    )
    pd.DataFrame([row]).to_csv(OUT / "sensitivity_s1_baseline_freshness.csv", index=False)

    # Secondary absolute peak-queue gap S2−S1 (does not replace ±5-pp relative primary).
    abs_gap = paired_day_contrast_series(res, "peak_queue", "S2", "S1", relative_to_s0=False)
    abs_row = diagnose_and_infer_contrast(
        abs_gap, label="S2_minus_S1_absolute_peak_queue", primary=False
    )
    abs_row.update(
        {
            "role": "secondary robustness; primary remains relative ±5-pp CBB equivalence",
            "aggregation": "median_r of paired within-replication absolute differences",
        }
    )
    pd.DataFrame([abs_row]).to_csv(OUT / "stats_absolute_peak_gap_S2_S1.csv", index=False)

    # Zero-S0 eligibility (calendar-preserving relative metric).
    g = paired_day_contrast_series(res, "peak_queue", "S2", "S1", relative_to_s0=True)
    pd.DataFrame(
        [
            {
                "n_focal_calendar_days": int(len(g)),
                "n_eligible_Qmax_S0_gt_1": int(g.notna().sum()),
                "n_ineligible_calendar_placeholders": int(g.isna().sum()),
                "zero_s0_threshold_trucks": ZERO_S0_PEAK_THRESHOLD,
                "note": "Ineligible dates retained as NaN calendar positions for CBB",
            }
        ]
    ).to_csv(OUT / "zero_s0_eligibility.csv", index=False)

    # Calendar-ordered primary effect diagnostics: ACF, monthly summaries, and
    # a simple trend slope. The CBB is applied to this paired-effect series.
    effect = paired_day_contrast_series(res, "peak_queue", "S2", "S1", relative_to_s0=True)
    dates = pd.to_datetime(effect.index)
    diag = pd.DataFrame({"date": dates, "g_t": effect.to_numpy()})
    diag["month"] = diag["date"].dt.month
    x = np.arange(len(diag), dtype=float)
    ok = diag["g_t"].notna().to_numpy()
    slope = float(np.polyfit(x[ok], diag.loc[ok, "g_t"], 1)[0]) if ok.sum() > 2 else float("nan")
    pd.DataFrame(
        [
            {
                "series": "g_t = Delta_Q,S2,t - Delta_Q,S1,t (median of paired replication ratios)",
                "n_calendar_days": int(len(diag)),
                "n_eligible": int(ok.sum()),
                "n_excluded": int((~ok).sum()),
                "acf_lag1": acf_lag1(diag.loc[ok, "g_t"].to_numpy()),
                "linear_time_slope_per_day": slope,
                "stationarity_note": "descriptive diagnostic only; CBB resamples calendar-ordered paired effects",
            }
        ]
    ).to_csv(OUT / "stats_cbb_effect_stability.csv", index=False)
    diag.to_csv(OUT / "stats_cbb_effect_daily.csv", index=False)
    diag.groupby("month")["g_t"].agg(["count", "mean", "median", "std"]).reset_index().to_csv(
        OUT / "stats_cbb_effect_monthly.csv", index=False
    )


def main() -> None:
    print("Running storage-fairness experiments (soft primary + common hard; MBB)...")
    res = run_main_experiments()
    res.to_csv(OUT / "scenario_day_replications.csv", index=False)
    # Common hard-yard experiment (same shocks structure; S3s-H vs S3-H only)
    print("Running common hard-storage environment (S0-H…S_OPT-H)...")
    res_hard = run_main_experiments(hard_storage_env=True, scenarios=SCENARIOS_HARD)
    res_hard.to_csv(OUT / "scenario_day_replications_hard.csv", index=False)
    _summarise_env(res_hard, suffix="_hard")

    n_days = int(res["date"].nunique())
    print(f"Main sample: {n_days} days × {REPS} reps × {len(SCENARIOS)} scenarios")

    metrics = [
        "peak_queue",
        "mean_wait_min",
        "p90_wait_min",
        "corridor_delay_index",
        "waiting_proxy",
        "storage_pressure",
        "deferred",
        "deferred_physical",
        "deferred_proactive",
        "throughput",
    ]
    summary_rows = []
    for s, g in res.groupby("scenario"):
        for col in metrics:
            if col not in g.columns:
                continue
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

    opt = res[res.scenario == "S_OPT"]
    pd.DataFrame(
        [
            {
                "median_peak_headline_E_capacity_plus_compliance": opt["peak_queue"].median(),
                "median_peak_perfect_compliance_diagnostic": opt["peak_queue_perfect_compliance"].median(),
                "median_peak_foresight_lp_diagnostic": opt["peak_queue_foresight_lp_diagnostic"].median(),
                "note": "Headline S_OPT: two-stage LP on planning C_h and N_hat then compliance; foresight diagnostic uses realised mu",
            }
        ]
    ).to_csv(OUT / "s_opt_compliance_diagnostic.csv", index=False)

    improv_rows = []
    # Relative improvement vs S0: median of paired within-replication ratios (peak only).
    for s in [x for x in SCENARIOS_SOFT if x != "S0"]:
        ser = day_level_paired_relative_improvement(res, s)
        improv_rows.append(
            {
                "scenario": s,
                "metric": "peak_queue",
                "median_improvement_vs_S0": float(ser.median()),
                "p10_improvement_vs_S0": float(ser.quantile(0.10)),
                "p90_improvement_vs_S0": float(ser.quantile(0.90)),
                "contrast_type": "relative_lower_is_better",
                "aggregation": "median_of_paired_replication_ratios",
            }
        )
    # Absolute paired day differences for other outcomes (direction noted)
    for metric, higher_better in [
        ("p90_wait_min", False),
        ("mean_wait_min", False),
        ("corridor_delay_index", False),
        ("throughput", True),
        ("deferred", False),
        ("storage_pressure", False),
    ]:
        for s in [x for x in SCENARIOS_SOFT if x != "S0"]:
            if s not in set(res["scenario"]):
                continue
            if higher_better:
                diff = day_level_paired_absolute_diff(res, s, "S0", metric).dropna()
            else:
                diff = day_level_paired_absolute_diff(res, "S0", s, metric).dropna()
            improv_rows.append(
                {
                    "scenario": s,
                    "metric": metric,
                    "median_improvement_vs_S0": float(diff.median()),
                    "p10_improvement_vs_S0": float(diff.quantile(0.10)),
                    "p90_improvement_vs_S0": float(diff.quantile(0.90)),
                    "contrast_type": "absolute_higher_better" if higher_better else "absolute_lower_is_better",
                    "aggregation": "median_of_paired_replication_differences",
                }
            )
    pd.DataFrame(improv_rows).to_csv(OUT / "improvement_vs_S0.csv", index=False)

    # Hard-environment S3s-H vs S3-H (proactive-only contrast)
    hard_rows = []
    for metric in ["peak_queue", "mean_wait_min", "p90_wait_min", "throughput", "deferred", "storage_pressure"]:
        series = paired_day_contrast_series(res_hard, metric, "S3s-H", "S3-H", relative_to_s0=False)
        # series = S3s-H - S3-H; for lower-is-better peak, negative means S3s better
        row = diagnose_and_infer_contrast(series, label=f"S3sH_minus_S3H_{metric}", primary=(metric == "peak_queue"))
        hard_rows.append(row)
    # Also relative peak gap vs S0-H for hard ladder
    for s in ["S1-H", "S2-H", "S3-H", "S3s-H", "S_OPT-H"]:
        series = paired_day_contrast_series(res_hard, "peak_queue", s, "S1-H", relative_to_s0=True)
        hard_rows.append(
            diagnose_and_infer_contrast(series, margin=TOST_PRIMARY_MARGIN, label=f"{s}_vs_S1H_rel_peak", primary=False)
        )
    pd.DataFrame(hard_rows).to_csv(OUT / "stats_hard_env_contrasts.csv", index=False)

    wilc_day = pd.concat(
        [wilcoxon_battery(res, m, "day") for m in ["peak_queue", "p90_wait_min", "waiting_proxy"]],
        ignore_index=True,
    )
    wilc_rep = pd.concat(
        [wilcoxon_battery(res, m, "replication") for m in ["peak_queue", "p90_wait_min", "waiting_proxy"]],
        ignore_index=True,
    )
    wilc_day.to_csv(OUT / "stats_wilcoxon_vs_S0.csv", index=False)
    wilc_rep.to_csv(OUT / "stats_wilcoxon_replication_level.csv", index=False)

    dep_rows = []
    for label, series in [
        ("S0_minus_S1_peak", paired_day_contrast_series(res, "peak_queue", "S0", "S1", False)),
        ("S2_minus_S1_rel_imp", paired_day_contrast_series(res, "peak_queue", "S2", "S1", True)),
        ("S3_minus_S1_rel_imp", paired_day_contrast_series(res, "peak_queue", "S3", "S1", True)),
        ("S_OPT_minus_S1_peak", paired_day_contrast_series(res, "peak_queue", "S_OPT", "S1", False)),
        ("S_OPT_minus_S1_rel_imp", paired_day_contrast_series(res, "peak_queue", "S_OPT", "S1", True)),
    ]:
        dep_rows.append(diagnose_and_infer_contrast(series, label=label, primary=True))
    pd.DataFrame(dep_rows).to_csv(OUT / "stats_temporal_dependence.csv", index=False)

    tost_all = []
    for margin in TOST_MARGINS:
        for m in ["peak_queue", "waiting_proxy"]:
            tost_all.append(tost_equivalence(res, m, margin, primary=(margin == TOST_PRIMARY_MARGIN)))
    for margin in TOST_ABS_PEAK_MARGINS:
        tost_all.append(tost_absolute_peak(res, margin))
    tost_df = apply_holm_to_primary_tost(pd.concat(tost_all, ignore_index=True))
    tost_df.to_csv(OUT / "stats_tost_equivalence.csv", index=False)
    # Block-length robustness for confirmatory S2−S1 relative peak gap
    mbb_rows = []
    series_s2 = paired_day_contrast_series(res, "peak_queue", "S2", "S1", relative_to_s0=True)
    for bl in BLOCK_LENS:
        row = diagnose_and_infer_contrast(
            series_s2,
            margin=TOST_PRIMARY_MARGIN,
            label="S2_vs_S1_rel_improvement",
            primary=True,
            block_len=bl,
            mbb_alpha=MBB_CI_ALPHA,
        )
        row["block_len_sensitivity"] = bl
        mbb_rows.append(row)
    pd.DataFrame(mbb_rows).to_csv(OUT / "stats_mbb_block_length_sensitivity.csv", index=False)
    # Forecast accuracy: trailing-7 public forecast vs realised I (2024)
    dfc = load_intensity()
    fc = pd.DataFrame(
        {
            "date": dfc["date"].dt.strftime("%Y-%m-%d"),
            "I": dfc["I"].to_numpy(),
            "I_hat": dfc["I_hat"].to_numpy(),
        }
    )
    fc["error"] = fc["I"] - fc["I_hat"]
    fc["abs_error"] = fc["error"].abs()
    fc["sq_error"] = fc["error"] ** 2
    fc.to_csv(OUT / "forecast_accuracy_2024_daily.csv", index=False)
    pd.DataFrame(
        [
            {
                "n_days": int(len(fc)),
                "MAE": float(fc["abs_error"].mean()),
                "RMSE": float(np.sqrt(fc["sq_error"].mean())),
                "bias": float(fc["error"].mean()),
                "corr": float(fc["I"].corr(fc["I_hat"])),
                "I_ref": float(I_REF),
                "note": "I_hat = trailing 7-day mean through t-1; no artificial forecast noise",
            }
        ]
    ).to_csv(OUT / "forecast_accuracy_2024_summary.csv", index=False)

    conv_multi = mc_convergence_multi(res)
    conv_multi.to_csv(OUT / "mc_convergence_multi.csv", index=False)
    conv = mc_convergence(res)
    conv.to_csv(OUT / "mc_convergence.csv", index=False)
    make_figures(res, conv)
    res.groupby(["scenario", "regime"])[[c for c in metrics if c in res.columns]].median().reset_index().to_csv(
        OUT / "summary_by_regime.csv", index=False
    )

    lp = lp_optimal_arrivals(1400, MU_MEAN, 1)
    s1 = _integerise_arrivals(static_appointment_weights() * 1400, 1400)
    design_check = {
        "lp_vs_s1_l1": float(np.abs(lp - s1).sum()),
        "lp_vs_uniform_l1": float(np.abs(lp - _uniform_int_arrivals(1400)).sum()),
        "s1_vs_uniform_l1": float(np.abs(s1 - _uniform_int_arrivals(1400)).sum()),
        "capacity_shape_strong": capacity_shape("strong").tolist(),
        "capacity_shape_mild": capacity_shape("mild").tolist(),
        "static_smooth_default": STATIC_SMOOTH_DEFAULT,
        "lp_tol": LP_TOL,
    }
    (OUT / "design_check_lp_vs_s1.json").write_text(json.dumps(design_check, indent=2), encoding="utf-8")
    print("LP vs S1 L1 distance:", design_check["lp_vs_s1_l1"])
    _write_run_meta(n_days)

    import os

    skip = os.environ.get("SKIP_SENSITIVITIES", "").strip() in {"1", "true", "TRUE", "yes"}
    structural_only = os.environ.get("STRUCTURAL_ONLY", "").strip() in {"1", "true", "TRUE", "yes"}
    if skip and not structural_only:
        print("SKIP_SENSITIVITIES set — structural/secondary sweeps deferred.")
    else:
        # Final-audit η×y0 and recent-S1 checks are secondary sweeps; respect the skip flag.
        if not structural_only:
            final_audit_sensitivities(res)
        print("Running sensitivities (structural grid" + (" only)" if structural_only else " + strided secondary)..."))
        run_sensitivities(structural_only=structural_only)
        _write_run_meta(n_days)

    print("\nMedian peak_queue:")
    print(res.groupby("scenario")["peak_queue"].median().reindex(SCENARIOS).round(2))
    print("\nS_OPT perfect diagnostic median:", round(opt["peak_queue_perfect_compliance"].median(), 2))
    print("Wrote", OUT)


if __name__ == "__main__":
    import os as _os

    if _os.environ.get("RUN_STRUCTURAL_ONLY", "").strip() in {"1", "true", "TRUE", "yes"}:
        print("RUN_STRUCTURAL_ONLY: decisive full-year structural cells only (main outputs left intact).")
        run_sensitivities(structural_only=True)
        # Refresh meta stamp without re-running main
        n_days_meta = int(pd.read_csv(OUT / "scenario_day_replications.csv")["date"].nunique())
        _write_run_meta(n_days_meta)
    else:
        main()
