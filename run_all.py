#!/usr/bin/env python3
"""One-command reproduction entry point for this package."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SIM = ROOT / "open_data_sim"


def run(script: str) -> None:
    path = SIM / script
    print(f"\n=== Running {path.name} ===\n", flush=True)
    subprocess.check_call([sys.executable, str(path)], cwd=str(ROOT))


def main() -> None:
    run("run_revision_experiments.py")
    run("make_q1_figures.py")
    run("round4_stats.py")
    print("\nDone. Outputs in open_data_sim/outputs_revision/ and figures_revision/.\n")


if __name__ == "__main__":
    main()
