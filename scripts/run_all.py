"""Orchestrate all project pipelines to regenerate artifacts end-to-end."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "run_ltv_pipeline.py",
    "run_forecasting_pipeline.py",
    "run_causal_pipeline.py",
    "build_final_report.py",
]


def run_step(script: str) -> None:
    script_path = Path(__file__).resolve().with_name(script)
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script}")
    print(f"\n>>> Running {script}...")
    result = subprocess.run([sys.executable, str(script_path)], check=True)
    if result.returncode == 0:
        print(f"{script} completed successfully.")


def main() -> None:
    for script in SCRIPTS:
        run_step(script)
    print("\nAll artifacts refreshed. You're ready to demo!")


if __name__ == "__main__":
    main()
