"""Structural smoke tests for the replication package."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_portwatch_exists_and_has_required_columns():
    path = ROOT / "external_data" / "tema_portwatch_daily_2019_2026.csv"
    assert path.exists()
    df = pd.read_csv(path, nrows=5)
    for col in ("date", "portcalls_container", "import_container", "export_container"):
        assert col in df.columns


def test_core_scripts_exist():
    sim = ROOT / "open_data_sim"
    for name in (
        "run_revision_experiments.py",
        "make_q1_figures.py",
        "round4_stats.py",
        "run_all.py",
    ):
        # run_all is at package root
        if name == "run_all.py":
            assert (ROOT / name).exists()
        else:
            assert (sim / name).exists()


def test_reference_summary_present():
    path = ROOT / "open_data_sim" / "outputs_revision" / "summary_by_scenario.csv"
    assert path.exists()
    df = pd.read_csv(path)
    assert len(df) > 0


def test_load_intensity_runs():
    import sys

    sys.path.insert(0, str(ROOT / "open_data_sim"))
    import run_revision_experiments as rre

    d = rre.load_intensity(2024)
    assert len(d) > 200
    assert "I" in d.columns
