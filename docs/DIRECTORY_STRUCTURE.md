# Directory structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── run_all.py
├── .gitignore
├── data/
│   └── README_DATA.md
├── docs/
│   ├── REPLICATION.md
│   ├── SOFTWARE.md
│   ├── DIRECTORY_STRUCTURE.md
│   └── CHANGELOG.md
├── external_data/
│   ├── tema_portwatch_daily_2019_2026.csv
│   └── face_validity_anchors/
├── open_data_sim/
│   ├── run_revision_experiments.py
│   ├── make_q1_figures.py
│   ├── make_revision_figures.py
│   ├── round4_stats.py
│   └── run_tas_scenarios.py
└── tests/
    └── test_smoke.py
```

Generated folders (created after `python run_all.py`, not tracked):

- `open_data_sim/outputs_revision/`
- `open_data_sim/figures_revision/`
