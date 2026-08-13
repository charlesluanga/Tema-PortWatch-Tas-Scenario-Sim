# Replication guide

## Environment

1. Install Python 3.12 (or 3.10+).
2. From the package root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Full reproduction

```powershell
python run_all.py
```

This runs:

1. `open_data_sim/run_revision_experiments.py` — Monte Carlo scenarios and sensitivities  
2. `open_data_sim/make_q1_figures.py` — multi-panel figures and display tables  
3. `open_data_sim/round4_stats.py` — post-hoc Wilcoxon / tie / TOST helpers from saved outputs  

Expected runtime for step 1 can be substantial on a laptop. Figures and tables are written locally under `open_data_sim/figures_revision/` and `open_data_sim/outputs_revision/` and are not part of the repository.

## Path contract

Scripts resolve the package root as the parent of `open_data_sim/` and read:

`external_data/tema_portwatch_daily_2019_2026.csv`
