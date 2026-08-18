# Changelog

## 2026-08-18

- Updated package short title to match the locked study title (truck appointment control under data constraints; Port of Tema)
- Scripts-and-inputs-only package policy unchanged

## 2026-08-17

- Synced experiment scripts with the frozen primary-estimand Results architecture (`κ_max` forecast-pressure blend; day-level structural aggregation compatible with the confirmatory estimand)
- Updated package README and `open_data_sim/README.md` for current scenario labels and defaults
- Hardened `.gitignore` against manuscript/identity artefacts
- Retained scripts-and-inputs-only package policy (figures/tables generated locally)

## 2026-08-14

- Public package limited to scripts, input data, and technical docs
- Removed committed figures and result tables; regenerate locally with `python run_all.py`
- Distinct defaults θ=0.50, κ_max=0.60, δ=0.25; expected-capacity LP planning; focal S=1 with S∈{2,3} sensitivities
