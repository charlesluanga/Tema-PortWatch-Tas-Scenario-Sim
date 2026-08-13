# Open-data truck appointment scenario simulation (Tema PortWatch intensity)

This package reproduces Monte Carlo scenario experiments that map publicly observed seaside container intensity to landside truck-appointment control policies (uncoordinated arrivals, static flattening, forecast heuristic, residual reallocation, storage-constrained variant, and an expected-capacity LP comparator).

## Quick start

```powershell
cd public_github_tema_tas_opendata
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_all.py
```

Full re-simulation can take a long time (many days × replications × scenarios × capacity levels). For a quick check of package integrity, run:

```powershell
python -m pytest tests -q
```

Reference summary CSVs and figures under `open_data_sim/outputs_revision/` and `open_data_sim/figures_revision/` are included so results can be inspected without a full re-run. Regenerating day-level replications requires `run_revision_experiments.py`.

## What is reproduced

- Demand forcing from PortWatch daily Tema container intensity (`external_data/tema_portwatch_daily_2019_2026.csv`)
- Scenarios: S0, S1, S2, S3, S3s, S_OPT
- Co-primary effective capacity levels \(S \in \{1,2,3\}\)
- Documented defaults: \(\varepsilon=10^{-9}\), \(\theta=0.50\), \(\gamma=0.20\), \(\delta=0.25\), LP planning \(\bar{\mu}=1.55\)
- Wilcoxon / Holm / TOST helpers and multi-panel figures

## Repository layout

```
run_all.py                 One-command entry point
requirements.txt
LICENSE
open_data_sim/             Experiment code and reference outputs/figures
external_data/             Public/derived input panels
data/README_DATA.md        Data access and licence notes
docs/                      Technical replication notes
tests/                     Smoke tests
```

## Software

- Python 3.12 recommended (3.10+ should work)
- NumPy, pandas, SciPy, matplotlib, pytest (see `requirements.txt`)

## Data access notes

See `data/README_DATA.md`. Upstream PortWatch estimates are redistributed here for reproducibility; respect HDX / PortWatch terms for redistribution beyond this package. GPHA monthly ship reports are represented only as a derived monthly comparison CSV (not the original spreadsheets).

## Licence

Code: MIT (see `LICENSE`). Upstream data remain under their original terms.
