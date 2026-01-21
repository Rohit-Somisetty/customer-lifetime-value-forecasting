from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CAUSAL_RESULTS = ROOT / "data" / "causal_results.csv"
BALANCE_TABLE = ROOT / "data" / "causal_balance_table.csv"


@pytest.mark.skipif(not CAUSAL_RESULTS.exists(), reason="Run scripts/run_all.py for causal metrics")
def test_causal_results_columns() -> None:
    df = pd.read_csv(CAUSAL_RESULTS)
    required = {"metric", "estimate", "ci_low", "ci_high"}
    missing = required - set(df.columns)
    assert not missing, f"Missing causal result columns: {missing}"


@pytest.mark.skipif(not BALANCE_TABLE.exists(), reason="Run scripts/run_all.py for causal metrics")
def test_balance_table_columns() -> None:
    df = pd.read_csv(BALANCE_TABLE)
    required = {"covariate", "mean_treatment", "mean_control", "smd", "stage"}
    missing = required - set(df.columns)
    assert not missing, f"Missing balance table columns: {missing}"
