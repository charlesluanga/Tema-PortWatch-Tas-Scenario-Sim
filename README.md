# Open-data truck appointment scenario simulation (Tema PortWatch intensity)

This package provides the scripts and input data needed to reproduce Monte Carlo scenario experiments that map publicly observed seaside container intensity to landside truck-appointment control policies (uncoordinated arrivals, static flattening, forecast heuristic, residual reallocation, storage-constrained variant, and an expected-capacity LP comparator).

Finished figures and result tables are **not** stored in this repository. They are created locally when you run the scripts.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_all.py
```

Full re-simulation can take a long time. For a quick integrity check:

```powershell
python -m pytest tests -q
```

After a successful run, outputs appear under:

- `open_data_sim/outputs_revision/`
- `open_data_sim/figures_revision/`

## What is reproduced

- Demand forcing from PortWatch daily Tema container intensity (`external_data/tema_portwatch_daily_2019_2026.csv`)
- Scenarios: S0, S1, S2, S3, S3s, S_OPT
- Co-primary effective capacity levels \(S \in \{1,2,3\}\)
- Documented defaults: \(\varepsilon=10^{-9}\), \(\theta=0.50\), \(\gamma=0.20\), \(\delta=0.25\), LP planning \(\bar{\mu}=1.55\)
- Wilcoxon / Holm / TOST helpers and multi-panel figures (generated locally)

## Repository layout

```
run_all.py                 One-command entry point
requirements.txt
LICENSE
open_data_sim/             Experiment scripts only
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
