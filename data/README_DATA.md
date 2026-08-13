# Data notes

## Primary forcing series

| File | Description | Role |
|------|-------------|------|
| `../external_data/tema_portwatch_daily_2019_2026.csv` | Daily Tema port activity / container estimates | Exogenous seaside intensity \(I_t\) |

Source family: IMF/UNGP PortWatch daily port activity estimates distributed via the Humanitarian Data Exchange (HDX). Redistribute and cite according to upstream terms.

## Derived triangulation panels

| File | Description | Role |
|------|-------------|------|
| `../external_data/face_validity_anchors/gpha_vs_portwatch_monthly_2024_2025.csv` | Monthly GPHA container vessel day-sums vs PortWatch container calls | Seaside co-movement check (definitions differ; levels not interchangeable) |
| `../external_data/face_validity_anchors/public_landside_oom_anchors.csv` | Curated public order-of-magnitude anchors | Existence / context checks only; not gate-queue calibration |

## Not included

- Proprietary gate, GPS, appointment-transaction, or truck TAT microdata
- Original GPHA monthly Excel workbooks (local only)
- Any vessel-operation panels that are not approved for public redistribution
