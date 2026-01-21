from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
LTV_PATH = ROOT / "data" / "ltv_predictions.csv"


@pytest.mark.skipif(not LTV_PATH.exists(), reason="Run scripts/run_all.py to generate LTV outputs")
def test_ltv_predictions_schema_and_values() -> None:
    df = pd.read_csv(LTV_PATH)
    required = {
        "customer_id",
        "predicted_ltv_6m",
        "predicted_ltv_12m",
        "expected_avg_value",
    }
    missing = required - set(df.columns)
    assert not missing, f"Missing columns: {missing}"
    assert (df["predicted_ltv_6m"] >= 0).all(), "Negative 6M LTV detected"
    assert (df["predicted_ltv_12m"] >= 0).all(), "Negative 12M LTV detected"
