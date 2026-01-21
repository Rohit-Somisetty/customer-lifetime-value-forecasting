from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
FORECAST_FILES = {
    "overall": ROOT / "data" / "revenue_forecasts_overall.csv",
    "segment": ROOT / "data" / "revenue_forecasts_by_segment.csv",
    "channel": ROOT / "data" / "revenue_forecasts_by_channel.csv",
}
REQUIRED_COLUMNS = {
    "date",
    "y_pred",
    "lower_80",
    "upper_80",
    "lower_95",
    "upper_95",
    "model_name",
    "horizon_weeks",
}


@pytest.mark.parametrize("name,path", list(FORECAST_FILES.items()))
def test_forecast_files_exist(name: str, path: Path) -> None:
    if not path.exists():
        pytest.skip(f"Missing forecast file for {name}; run scripts/run_all.py")
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    assert not missing, f"{name} forecast missing columns: {missing}"
    assert len(df) > 0, f"{name} forecast is empty"
