"""Generate revenue forecasts by segment and channel for planning workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.forecasting import (
    backtest,
    build_timeseries,
    fit_baseline,
    fit_sarimax_or_ets,
    forecast_with_intervals,
)
from src.ltv_model import load_transactions

FORECAST_FREQ = "W-MON"
FORECAST_HORIZONS = (12, 26)
TEST_WEEKS = 12
SEASONAL_LAG = 52
SEASONAL_PERIODS = 52


def create_segments(
    ltv_path: Path, output_path: Path, channel_map: pd.DataFrame | None = None
) -> pd.DataFrame:
    ltv = pd.read_csv(ltv_path)
    required = {"customer_id", "predicted_ltv_12m"}
    if missing := required - set(ltv.columns):
        raise ValueError(f"LTV predictions missing columns: {missing}")
    low_cut = ltv["predicted_ltv_12m"].quantile(0.2)
    high_cut = ltv["predicted_ltv_12m"].quantile(0.8)

    def label(value: float) -> str:
        if value >= high_cut:
            return "High"
        if value <= low_cut:
            return "Low"
        return "Mid"

    segments = ltv[["customer_id", "predicted_ltv_12m"]].copy()
    segments["ltv_segment"] = segments["predicted_ltv_12m"].apply(label)
    segments = segments[["customer_id", "ltv_segment"]]
    if channel_map is not None:
        segments = segments.merge(channel_map, on="customer_id", how="left")
        segments["primary_channel"] = segments["primary_channel"].fillna("unknown")
    segments.to_csv(output_path, index=False)
    return segments


def prepare_weekly_series(
    weekly_df: pd.DataFrame, group_type: str, group_value: str
) -> pd.Series:
    subset = (
        weekly_df[(weekly_df["group_type"] == group_type) & (weekly_df["group_value"] == group_value)]
        .sort_values("date")
        .set_index("date")
    )
    series = subset["revenue"].asfreq(FORECAST_FREQ, fill_value=0.0)
    return series


def select_model(series: pd.Series) -> Tuple[Dict[str, Dict[str, float]], str, object]:
    try:
        metrics = backtest(
            series,
            test_size=TEST_WEEKS,
            seasonal_lag=SEASONAL_LAG,
            seasonal_periods=SEASONAL_PERIODS,
        )
    except ValueError:
        metrics = {
            "baseline": {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan")},
            "model": {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan")},
        }
        best_key = "baseline"
    else:
        best_key = "model" if metrics["model"]["rmse"] <= metrics["baseline"]["rmse"] else "baseline"

    if best_key == "model":
        fitted = fit_sarimax_or_ets(series, seasonal_periods=SEASONAL_PERIODS)
        model_name = "ETS"
    else:
        fitted = fit_baseline(series, seasonal_lag=SEASONAL_LAG)
        model_name = "SeasonalNaive"
    return metrics, model_name, fitted


def forecast_group(
    series: pd.Series, group_type: str, group_value: str
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], str]:
    metrics, model_name, fitted_model = select_model(series)
    last_date = series.index[-1]
    outputs: List[pd.DataFrame] = []
    for horizon in FORECAST_HORIZONS:
        fc = forecast_with_intervals(model_name, fitted_model, horizon, FORECAST_FREQ, last_date)
        fc["horizon_weeks"] = horizon
        fc["y_true"] = np.nan
        fc["group_type"] = group_type
        fc["group_value"] = group_value
        outputs.append(fc)
    combined = pd.concat(outputs, ignore_index=True)
    return combined, metrics, model_name


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    transactions_path = data_dir / "transactions.csv"
    ltv_path = data_dir / "ltv_predictions.csv"

    transactions = load_transactions(transactions_path)
    channel_pref = (
        transactions.groupby(["customer_id", "channel"])
        .size()
        .reset_index(name="cnt")
        .sort_values(["customer_id", "cnt"], ascending=[True, False])
    )
    channel_map = (
        channel_pref.drop_duplicates("customer_id")[["customer_id", "channel"]]
        .rename(columns={"channel": "primary_channel"})
    )
    segments = create_segments(
        ltv_path,
        data_dir / "customer_segments.csv",
        channel_map=channel_map,
    )
    daily_ts, weekly_ts = build_timeseries(transactions, segments)
    daily_ts.to_csv(data_dir / "revenue_daily.csv", index=False)
    weekly_ts.to_csv(data_dir / "revenue_weekly.csv", index=False)

    overall_series = prepare_weekly_series(weekly_ts, "overall", "all")
    overall_fc, overall_metrics, overall_model = forecast_group(
        overall_series, "overall", "all"
    )
    print("Overall forecast model:", overall_model)
    print("Overall metrics:", overall_metrics)

    unique_segments = segments["ltv_segment"].unique().tolist()
    preferred_order = ["High", "Mid", "Low"]
    segment_values = [seg for seg in preferred_order if seg in unique_segments]
    segment_values += [seg for seg in unique_segments if seg not in segment_values]
    segment_frames = []
    for segment in segment_values:
        series = prepare_weekly_series(weekly_ts, "segment", segment)
        fc, metrics, model_name = forecast_group(series, "segment", segment)
        print(f"Segment {segment} model: {model_name}")
        print(f"Segment {segment} metrics: {metrics}")
        fc["segment"] = segment
        segment_frames.append(fc)
    segments_fc = pd.concat(segment_frames, ignore_index=True)

    channel_values = (
        weekly_ts.loc[weekly_ts["group_type"] == "channel", "group_value"].dropna().unique()
    )
    channel_frames = []
    for channel in sorted(channel_values):
        series = prepare_weekly_series(weekly_ts, "channel", channel)
        fc, metrics, model_name = forecast_group(series, "channel", channel)
        print(f"Channel {channel} model: {model_name}")
        print(f"Channel {channel} metrics: {metrics}")
        fc["channel"] = channel
        channel_frames.append(fc)
    channels_fc = pd.concat(channel_frames, ignore_index=True)

    overall_out = overall_fc[[
        "date",
        "y_true",
        "y_pred",
        "lower_80",
        "upper_80",
        "lower_95",
        "upper_95",
        "model_name",
        "horizon_weeks",
    ]]
    segment_out = segments_fc[[
        "segment",
        "date",
        "y_true",
        "y_pred",
        "lower_80",
        "upper_80",
        "lower_95",
        "upper_95",
        "model_name",
        "horizon_weeks",
    ]]
    channel_out = channels_fc[[
        "channel",
        "date",
        "y_true",
        "y_pred",
        "lower_80",
        "upper_80",
        "lower_95",
        "upper_95",
        "model_name",
        "horizon_weeks",
    ]]

    overall_out.to_csv(data_dir / "revenue_forecasts_overall.csv", index=False)
    segment_out.to_csv(data_dir / "revenue_forecasts_by_segment.csv", index=False)
    channel_out.to_csv(data_dir / "revenue_forecasts_by_channel.csv", index=False)


if __name__ == "__main__":
    main()
