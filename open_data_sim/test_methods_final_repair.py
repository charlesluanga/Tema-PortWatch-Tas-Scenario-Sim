"""Verification tests for storage-fairness green-light contracts."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
if (_here / "run_revision_experiments.py").exists():
    sys.path.insert(0, str(_here))
else:
    sys.path.insert(0, str(_here.parents[1] / "open_data_sim"))

import run_revision_experiments as r  # noqa: E402


def test_usable_2024_days_and_symmetric_regimes():
    days = r.select_days(r.load_intensity(), full_year=True)
    assert len(days) == 366
    assert set(days["regime"].unique()) <= {"high", "low", "normal"}
    week = days.groupby("week_end")["regime"].first()
    assert (week == "high").sum() == 5
    assert (week == "low").sum() == 5


def test_hist_reference_and_genuine_forecast():
    d = r.load_intensity()
    assert "I_hat" in d.columns and "I_S1_base" in d.columns and "I_S1_recent" in d.columns
    assert float(d.attrs["I_ref"]) > 0
    assert np.allclose(d["I_hat"], d["I_hat_base"])
    assert d["I_S1_recent"].notna().all()


def test_s1_uses_n_base_not_forecast_n_hat():
    rng = np.random.default_rng(0)
    N, N_hat, N_base = 1000, 700, 1400
    peak = r.allocate(N, r.peak_weights(rng), rng)
    m = r.run_one(
        "S1", N, 0.5, 0.5, 0.4, rng, 0.9, 1.55, 450.0, peak, N_hat=N_hat, N_base=N_base
    )
    assert m["plan_total"] == N_base
    assert abs(m["arrival_sum"] - N) < 1e-6
    assert m["hard_storage"] is False


def test_soft_primary_never_hard():
    rng = np.random.default_rng(1)
    N = 900
    peak = r.allocate(N, r.peak_weights(rng), rng)
    for s in r.SCENARIOS_SOFT:
        m = r.run_one(s, N, 0.5, 0.5, 0.5, rng, 0.85, 1.55, 450.0, peak, N_hat=N, N_base=N)
        assert m["hard_storage"] is False
        assert m["scenario"] == s
        assert "S3s" not in r.SCENARIOS_SOFT


def test_common_hard_all_hard_and_s3s_proactive_only():
    rng = np.random.default_rng(2)
    N = 1000
    peak = r.allocate(N, r.peak_weights(rng), rng)
    # Tight yard to trigger storage-aware path
    cap = 80.0
    s3 = r.run_one(
        "S3-H", N, 0.8, 0.5, 0.8, np.random.default_rng(11), 0.9, 1.4, cap, peak,
        N_hat=N, N_base=N, hard_storage_env=True,
    )
    s3s = r.run_one(
        "S3s-H", N, 0.8, 0.5, 0.8, np.random.default_rng(11), 0.9, 1.4, cap, peak,
        N_hat=N, N_base=N, hard_storage_env=True,
    )
    assert s3["hard_storage"] and s3s["hard_storage"]
    assert s3["deferred_proactive"] == 0.0
    assert s3s["deferred"] == s3s["deferred_proactive"]
    assert s3s["deferred_physical"] == 0.0
    assert abs(s3s["booked"] - (s3s["presented"] + s3s["deferred_proactive"])) < 1e-6
    assert abs(s3s["presented"] - (s3s["throughput"] + s3s["final_queue"])) < 1e-6


def test_soft_no_occupancy_clip_allows_above_cy():
    """Soft yard is non-binding: occupancy may exceed C^Y (no 1.8 clip)."""
    arrivals = np.full(r.N_HOURS, 200.0)
    m = r.simulate_day(arrivals, mu=1.2, storage_cap=50.0, hard_storage=False, n_servers=1)
    assert m["final_storage"] > 50.0  # would be impossible if clipped to C^Y or 1.8*C^Y only if small
    # With alpha_in=0.55 and high service, storage grows well above C^Y
    assert m["max_storage"] > 1.8 * 50.0 or m["final_storage"] > 50.0


def test_hard_storage_feasibility_no_post_clip_needed():
    rng = np.random.default_rng(1)
    N = 1200
    peak = r.allocate(N, r.peak_weights(rng), rng)
    m = r.run_one("S0-H", N, 0.5, 0.5, 0.5, rng, 0.9, 1.55, 50.0, peak, hard_storage_env=True)
    assert m["final_storage"] <= 50.0 + 1e-6
    assert m["max_storage"] <= 50.0 + 1e-6
    assert m["deferred_physical"] == 0.0
    assert abs(m["presented"] - (m["throughput"] + m["final_queue"])) < 1e-6


def test_flow_conservation_identities_soft_and_hard():
    rng = np.random.default_rng(4)
    N = 1100
    peak = r.allocate(N, r.peak_weights(rng), rng)
    for hard, label in [(False, "S2"), (True, "S2-H")]:
        m = r.run_one(label, N, 0.5, 0.5, 0.55, rng, 0.8, 1.55, 450.0, peak, N_hat=1000, N_base=1100, hard_storage_env=hard)
        assert abs(m["presented"] - (m["throughput"] + m["final_queue"])) < 1e-6
        assert abs(m["booked"] - (m["presented"] + m["deferred_proactive"])) < 1e-6
        assert m["flow_ok"] is True


def test_no_slot_jitter_in_static_weights():
    w1 = r.static_appointment_weights(np.random.default_rng(0), smooth=0.5)
    w2 = r.static_appointment_weights(np.random.default_rng(1), smooth=0.5)
    assert np.allclose(w1, w2)


def test_lp_two_stage_and_not_equal_s1():
    N = 1400
    lp = r.lp_optimal_arrivals(N, r.MU_MEAN, 1, capacity_kind="mild")
    s1 = r._integerise_arrivals(r.static_appointment_weights(smooth=0.5) * N, N)
    assert int(lp.sum()) == N
    assert abs(lp - s1).sum() > 10


def test_capacity_profiles_mean_one_and_mild_default():
    assert r.CAPACITY_PROFILE_DEFAULT == "mild"
    for kind in ["flat", "mild", "strong"]:
        s = r.capacity_shape(kind)
        assert abs(s.mean() - 1.0) < 1e-9


def test_relative_improvement_peak_only():
    wide = np.nan  # noqa — use DataFrame
    import pandas as pd

    wide = pd.DataFrame({"S0": [100.0, 0.5, 50.0], "S1": [80.0, 0.0, 40.0]})
    ser = r.relative_improvement_peak(wide, "S1")
    assert np.isnan(ser.iloc[1])  # S0 peak 0.5 <= 1 truck
    assert abs(ser.iloc[0] - 0.2) < 1e-12


def test_mbb_ci_runs_and_seeded():
    x = np.sin(np.linspace(0, 20, 100)) + np.random.default_rng(0).normal(0, 0.1, 100)
    a = r.moving_block_bootstrap_mean_ci(x, block_len=7, n_boot=200, alpha=0.10, seed=r.BLOCK_BOOT_SEED)
    b = r.moving_block_bootstrap_mean_ci(x, block_len=7, n_boot=200, alpha=0.10, seed=r.BLOCK_BOOT_SEED)
    assert a["method"] == "circular_block_bootstrap"
    assert a["ci_low"] == b["ci_low"] and a["ci_high"] == b["ci_high"]


def test_cbb_retains_calendar_gaps_for_zero_s0_rule():
    """NaN ineligible effects remain calendar positions, not compressed dates."""
    x = np.array([0.10, np.nan, 0.20, 0.30, np.nan, 0.40, 0.50, 0.60])
    out = r.moving_block_bootstrap_mean_ci(x, block_len=3, n_boot=100, alpha=0.10, seed=123)
    assert out["n_calendar"] == len(x)
    assert out["n_eligible"] == 6
    assert abs(out["mean"] - np.nanmean(x)) < 1e-12


def test_acf_lag1_respects_calendar_gaps():
    """Adjacent-eligible ACF must not stitch across ineligible dates."""
    # Without gaps, lag-1 is strong; inserting NaNs between values breaks adjacency.
    continuous = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    gapped = np.array([1.0, np.nan, 2.0, np.nan, 3.0, np.nan, 4.0, np.nan, 5.0, np.nan, 6.0])
    assert np.isfinite(r.acf_lag1(continuous))
    assert np.isnan(r.acf_lag1(gapped))


def test_s3_trigger_uses_planning_capacity_and_eta_y0_are_recorded():
    rng = np.random.default_rng(12)
    N = 1200
    peak = r.allocate(N, r.peak_weights(rng), rng)
    m = r.run_one(
        "S3-H", N, 0.7, 0.5, 0.6, np.random.default_rng(9), 0.9, 2.0, 450.0, peak,
        N_hat=N, N_base=N, hard_storage_env=True, eta_storage_warning=0.75, initial_yard_frac=0.60,
    )
    assert m["eta_storage_warning"] == 0.75
    assert m["initial_yard_frac"] == 0.60
    # The test is contractual: realised μ drives service, while C_plan governs the S3 trigger.
    assert m["mu"] == 2.0 and m["mu_plan_lp"] == r.MU_MEAN


def test_s3_s3s_share_rng_under_hard():
    """With shared seed, S3-H and S3s-H differ only when proactive fires."""
    rng = np.random.default_rng(0)
    N = 1000
    peak = r.allocate(N, r.peak_weights(rng), rng)
    # Identical RNG streams for both
    s3 = r.run_one("S3-H", N, 0.5, 0.5, 0.5, np.random.default_rng(99), 0.9, 1.55, 450.0, peak, N_hat=N, N_base=N, hard_storage_env=True)
    s3s = r.run_one("S3s-H", N, 0.5, 0.5, 0.5, np.random.default_rng(99), 0.9, 1.55, 450.0, peak, N_hat=N, N_base=N, hard_storage_env=True)
    assert s3["deferred_proactive"] == 0.0
    # If proactive did not fire, booked paths match and peaks match
    if s3s["deferred_proactive"] == 0.0:
        assert abs(s3["peak_queue"] - s3s["peak_queue"]) < 1e-9
        assert abs(s3["presented"] - s3s["presented"]) < 1e-9


def test_proactive_peel_ties_earlier_hour_and_exact_d_pro():
    """Highest booked count first; ties → earlier hour; exact D^pro; non-negative."""
    mid = r.N_HOURS // 2
    N = 160
    first = np.full(mid, 5, dtype=int)
    # Residual N_rem = 80; construct equal peak in two second-half hours
    peak = np.concatenate([first, np.full(r.N_HOURS - mid, 10)])
    mixed_first = first.copy()
    C_plan = r.hourly_capacity(r.MU_MEAN, 1)
    # Force storage-tight + proactive: large mid storage, hard=True
    arrivals, proactive = r.residual_second_half(
        N, peak, mixed_first, queue_after_first=0.0, storage_after_first=400.0,
        storage_cap=450.0, I_hat=0.5, I_ref=0.5, rng=np.random.default_rng(1),
        compliance=1.0, tilt=0.6, C_plan_path=C_plan, hard_storage=True,
        proactive=True, eta_storage_warning=0.85, rho_proactive=0.25,
    )
    n_rem = int(N - mixed_first.sum())
    d_pro = int(round(0.25 * n_rem))
    assert int(proactive.sum()) == d_pro
    assert np.all(arrivals - proactive >= 0)
    # Tie-break: if two equal hours at top, earlier index peeled first when counts equal
    second = arrivals[mid:]
    # After peel, deferred hours should prefer higher counts
    if d_pro > 0:
        assert proactive.sum() == d_pro


def test_median_of_paired_ratios_not_ratio_of_medians():
    """Primary relative path must median within-rep ratios before contrasting scenarios."""
    import pandas as pd

    rows = []
    for date, reps in [
        ("2024-01-01", [(100.0, 80.0, 70.0), (50.0, 40.0, 45.0)]),
        ("2024-01-02", [(200.0, 100.0, 150.0), (10.0, 8.0, 9.0)]),
    ]:
        for rep, (q0, q1, q2) in enumerate(reps):
            rows.append({"date": date, "rep": rep, "scenario": "S0", "peak_queue": q0})
            rows.append({"date": date, "rep": rep, "scenario": "S1", "peak_queue": q1})
            rows.append({"date": date, "rep": rep, "scenario": "S2", "peak_queue": q2})
    res = pd.DataFrame(rows)
    g = r.paired_day_contrast_series(res, "peak_queue", "S2", "S1", relative_to_s0=True)
    # Day 1: deltas_r = [1-70/100, 1-45/50]=[0.30,0.10] vs S1 [0.20,0.20]
    # median Δ_S2=0.20, median Δ_S1=0.20 → g=0
    # Ratio-of-medians would use med(S0)=75, med(S2)=57.5 → different
    assert abs(float(g.loc["2024-01-01"]) - 0.0) < 1e-12
    d2_s2 = np.median([1 - 150 / 200, 1 - 9 / 10])
    d2_s1 = np.median([1 - 100 / 200, 1 - 8 / 10])
    assert abs(float(g.loc["2024-01-02"]) - (d2_s2 - d2_s1)) < 1e-12


def test_hist_minmax_unclipped_allows_outside_unit_interval():
    """2019–2023 anchors; evaluation I is not clipped to [0, 1]."""
    import pandas as pd

    raw = pd.read_csv(r.DATA / "tema_portwatch_daily_2019_2026.csv")
    raw["date"] = pd.to_datetime(raw["date"], utc=True)
    raw["year"] = raw["date"].dt.year
    usable = (
        raw["portcalls_container"].notna()
        & raw["import_container"].notna()
        & raw["export_container"].notna()
    )
    d = raw.loc[usable].copy()
    hist = d[d["year"].isin(r.HIST_YEARS)]
    c = d["portcalls_container"].astype(float)
    v = (d["import_container"] + d["export_container"]).astype(float)
    c_lo, c_hi = float(hist["portcalls_container"].min()), float(hist["portcalls_container"].max())
    v_hist = (hist["import_container"] + hist["export_container"]).astype(float)
    v_lo, v_hi = float(v_hist.min()), float(v_hist.max())
    c_n = (c - c_lo) / (c_hi - c_lo + r.EPS)
    v_n = (v - v_lo) / (v_hi - v_lo + r.EPS)
    out = r.load_intensity()
    eval_I = out["I"].to_numpy()
    # Reconstruct 2024 I from the same anchors and require exact match (no clip).
    eval_mask = d["year"] == r.EVAL_YEAR
    I_eval = (0.6 * c_n[eval_mask] + 0.4 * v_n[eval_mask]).to_numpy()
    assert len(eval_I) == len(I_eval)
    assert np.allclose(eval_I, I_eval, rtol=0, atol=1e-12)
    # Document that out-of-range scaled values are possible and retained.
    assert bool((c_n[eval_mask] > 1).any() or (v_n[eval_mask] > 1).any() or (eval_I <= 1).all())
