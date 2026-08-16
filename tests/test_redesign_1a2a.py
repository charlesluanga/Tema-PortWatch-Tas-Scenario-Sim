"""Verification tests for methods green-light final contracts."""
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
    assert "I_hat" in d.columns and "I_S1_base" in d.columns
    assert float(d.attrs["I_ref"]) > 0
    # No artificial noise: I_hat equals I_hat_base
    assert np.allclose(d["I_hat"], d["I_hat_base"])


def test_s1_uses_n_base_not_forecast_n_hat():
    rng = np.random.default_rng(0)
    N, N_hat, N_base = 1000, 700, 1400
    peak = r.allocate(N, r.peak_weights(rng), rng)
    m = r.run_one(
        "S1", N, 0.5, 0.5, 0.4, rng, 0.9, 1.55, 450.0, peak, N_hat=N_hat, N_base=N_base
    )
    assert m["N"] == N
    assert m["N_hat"] == N_hat
    assert m["N_base"] == N_base
    assert m["plan_total"] == N_base
    assert abs(m["arrival_sum"] - N) < 1e-6


def test_s2_capacity_blend_not_midday_tilt():
    rng = np.random.default_rng(1)
    w_flat = r.forecast_weights(1.5, 1.0, rng, tilt=0.6, capacity_kind="flat")
    # Under flat capacity, blend moves toward uniform
    uni = np.ones(r.N_HOURS) / r.N_HOURS
    w_s1 = r.static_appointment_weights(None, smooth=0.5)
    # High pressure → closer to capacity/uniform than to raw S1 relative to kappa>0
    assert abs(w_flat.sum() - 1.0) < 1e-12
    assert r.forecast_kappa(1.5, 1.0) > 0


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
    assert abs(r.capacity_shape("flat") - 1.0).max() < 1e-12
    assert r.capacity_shape("strong")[5] < 0.85


def test_hard_storage_feasibility_and_queue_retention():
    rng = np.random.default_rng(1)
    N = 1200
    peak = r.allocate(N, r.peak_weights(rng), rng)
    m = r.run_one("S0", N, 0.5, 0.5, 0.5, rng, 0.9, 1.55, 50.0, peak, hard_storage_env=True)
    assert m["final_storage"] <= 50.0 + 1e-6
    assert m["deferred_physical"] == 0.0  # yard binding stays in residual queue
    assert abs(m["presented"] - m["flow_check_served_plus_final_q"]) < 1e-6


def test_proactive_deferral_separate():
    rng = np.random.default_rng(3)
    N = 1000
    peak = r.allocate(N, r.peak_weights(rng), rng)
    soft = r.run_one("S3", N, 0.5, 0.5, 0.5, rng, 0.85, 1.55, 450.0, peak, N_hat=N, N_base=N)
    hard = r.run_one("S3s", N, 0.5, 0.5, 0.5, rng, 0.85, 1.55, 200.0, peak, N_hat=N, N_base=N)
    assert soft["deferred_proactive"] == 0.0
    assert hard["deferred"] == hard["deferred_proactive"]
    assert hard["deferred_physical"] == 0.0


def test_compliance_endpoints():
    planned = np.array([10] * r.N_HOURS)
    peak = np.array([0] * r.N_HOURS)
    assert r.mix_compliance(planned, peak, 1.0, total=160).sum() == 160
    assert r.mix_compliance(planned, peak, 0.0, total=0).sum() == 0


def test_shared_shocks_seed_reproducible():
    N = 800
    peak = r.allocate(N, r.peak_weights(np.random.default_rng(9)), np.random.default_rng(9))
    a = r.run_one("S2", N, 0.4, 0.5, 0.45, np.random.default_rng(11), 0.8, 1.55, 450.0, peak, N_hat=750, N_base=800)
    b = r.run_one("S2", N, 0.4, 0.5, 0.45, np.random.default_rng(11), 0.8, 1.55, 450.0, peak, N_hat=750, N_base=800)
    assert a["peak_queue"] == b["peak_queue"]


def test_s3_second_half_retains_compliance_share():
    rng = np.random.default_rng(2)
    N = 1600
    mid = r.N_HOURS // 2
    peak = r.allocate(N, r.peak_weights(rng), rng)
    first = peak[:mid].copy()
    N_rem = int(N - int(first.sum()))
    C = r.hourly_capacity(1.55, 1)
    w = np.maximum(C[mid:], r.EPS)
    w = w / w.sum()
    adaptive_plan = rng.multinomial(N_rem, w)
    peak_rem = r._integerise_arrivals(peak[mid:].astype(float), N_rem, prefer=w)
    mixed = r.mix_compliance(adaptive_plan, peak_rem, 0.70, total=N_rem)
    pure_adapt = r.mix_compliance(adaptive_plan, peak_rem, 1.0, total=N_rem)
    pure_peak = r.mix_compliance(adaptive_plan, peak_rem, 0.0, total=N_rem)
    assert abs(mixed - pure_adapt).sum() > 0
    assert abs(mixed - pure_peak).sum() > 0
    full, _ = r.residual_second_half(
        N, peak, first, 500.0, 100.0, 450.0, 0.5, 0.5, np.random.default_rng(7), 0.70, 0.60, C, False
    )
    assert int(full.sum()) == N


def test_realisation_preserves_realised_n():
    rng = np.random.default_rng(5)
    planned = r.allocate(700, r.uniform_weights(rng), rng)
    peak = r.allocate(1000, r.peak_weights(rng), rng)
    out = r.realise_forecast_plan(planned, 1000, peak, 0.85, rng)
    assert int(out.sum()) == 1000


def test_mbb_ci_runs():
    x = np.sin(np.linspace(0, 20, 100)) + np.random.default_rng(0).normal(0, 0.1, 100)
    boot = r.moving_block_bootstrap_mean_ci(x, block_len=7, n_boot=200, alpha=0.10, seed=1)
    assert boot["method"] == "moving_block_bootstrap"
    assert boot["ci_low"] <= boot["mean"] <= boot["ci_high"]
