# Open-data Tema TAS scenario simulation

## Reproducibility

From the package root:

```powershell
python run_all.py
```

Or step-by-step:

```powershell
python open_data_sim/run_revision_experiments.py
python open_data_sim/make_q1_figures.py
python open_data_sim/round4_stats.py
```

**Inputs:** `external_data/tema_portwatch_daily_2019_2026.csv` (IMF/UNGP PortWatch via HDX)

**Outputs (generated locally; not shipped):**
- `open_data_sim/outputs_revision/` — day/replication KPIs, day-level inference, sensitivities, design checks
- `open_data_sim/figures_revision/` — multi-panel figures

## Design lock

- Time-varying within-day capacity; historical-baseline static S1; forecast-and-capacity-informed S2; private mid-queue S3; compliance-matched LP on planning capacity
- Soft-storage primary ladder; common hard-yard environment for proactive storage control
- Full usable 2024 PortWatch Tema days (`n=366`); day-level inference primary; `R=40`

## Documented defaults

| Symbol | Value | Notes |
|--------|-------|-------|
| ε | 1e-9 | Normalisation stabiliser |
| θ | 0.50 | Residual-reallocation threshold |
| κ_max | 0.60 | Forecast-pressure blend cap |
| δ | 0.25 | Yard clearance fraction |
| ω₀, ω₁ | 0.35, 0.65 | Waiting-proxy weights |
| α_in | 0.55 | Yard intake fraction |
| λ0 | 1400 | Scale assumption |
| S | 1 (focal) | Capacity sensitivities {1,2,3} |
| μ | N(1.55, 0.18) truncated [1.15, 2.10] | Service-rate assumption |
| p | U(0.75, 0.95) | Compliance; shared across scenarios per (day, rep) |
| C^Y base | 450 | Storage multipliers {0.8,1.0,1.2} |
| BPR α, β | 0.15, 4.0 | Corridor delay index defaults |

## Scenario labels

- **S0** Uncoordinated peaked arrivals
- **S1** Historical-baseline stylised static template
- **S2** Forecast-and-capacity-informed information-and-control layer
- **S3** Private mid-queue residual reallocation
- **S3s-H** Hard-yard proactive storage control (common hard environment)
- **S_OPT** Compliance-matched LP on planning capacity

## KPI labels

`mean_wait_min` / `p90_wait_min` = waiting-time **proxy** (minutes).  
`corridor_delay_index` = BPR index (not physical travel time).

## Licence / data

PortWatch data via HDX. Code: MIT (see package `LICENSE`). Upstream data remain under their original terms.
