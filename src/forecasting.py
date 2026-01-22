"""Forecasting utilities for revenue planning and segmentation analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from statsmodels.tsa.holtwinters import ExponentialSmoothing


@dataclass
class SeasonalNaiveModel:
    """Simple seasonal-naive forecaster."""

    history: pd.Series
    seasonal_lag: int
    residual_std: float

    def forecast(self, steps: int) -> pd.Series:
        if steps <= 0:
            raise ValueError("steps must be positive")
        freq = self.history.index.freq or to_offset(pd.infer_freq(self.history.index))
        if freq is None:
            raise ValueError("Time series must have an inferable frequency")
        hist_values = self.history.to_numpy()
        lag = max(1, self.seasonal_lag)
        values = [hist_values[-lag + (i % lag)] for i in range(steps)]
        start = self.history.index[-1] + freq
        index = pd.date_range(start=start, periods=steps, freq=freq)
        return pd.Series(values, index=index)


@dataclass
class ETSModel:
    """Wrapper around statsmodels ExponentialSmoothing results."""

    results: object
    residual_std: float

    def forecast(self, steps: int) -> pd.Series:
        preds = self.results.forecast(steps)
        if not isinstance(preds, pd.Series):
            preds = pd.Series(preds)
        return preds


def build_timeseries(
    transactions: pd.DataFrame, segments: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build daily and weekly revenue datasets for overall/channel/segment views."""

    transactions = transactions.copy()
    transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])
    segments = segments[["customer_id", "ltv_segment"]].copy()
    merged = transactions.merge(segments, on="customer_id", how="left")
    merged["ltv_segment"] = merged["ltv_segment"].fillna("Unknown")

    daily = _aggregate_revenue(merged, "D")
    weekly = _aggregate_revenue(merged, "W-MON")
    return daily, weekly


def fit_baseline(series: pd.Series, seasonal_lag: int = 52) -> SeasonalNaiveModel:
    """Fit a seasonal-naive baseline model."""

    series = series.sort_index()
    if series.empty:
        raise ValueError("Series is empty")
    lag = min(seasonal_lag, max(1, len(series)))
    shifted = series.shift(lag)
    residuals = series - shifted
    residuals = residuals.dropna()
    resid_std = float(residuals.std(ddof=1)) if not residuals.empty else float(series.std(ddof=1))
    return SeasonalNaiveModel(history=series, seasonal_lag=lag, residual_std=resid_std or 0.0)


def fit_sarimax_or_ets(series: pd.Series, seasonal_periods: int = 52) -> ETSModel:
    """Fit an exponential smoothing model with optional seasonality."""

    series = series.sort_index()
    if series.empty:
        raise ValueError("Series is empty")
    usable_periods = min(seasonal_periods, max(2, len(series) // 2))
    if usable_periods < 2:
        usable_periods = None
    trend = "add" if len(series) >= 3 else None
    seasonal = "add" if usable_periods else None
    model = ExponentialSmoothing(
        series,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=usable_periods,
        initialization_method="estimated",
    )
    fitted = model.fit(optimized=True)
    resid = getattr(fitted, "resid", pd.Series(dtype=float))
    resid_std = float(resid.std(ddof=1)) if len(resid) > 1 else float(series.std(ddof=1))
    return ETSModel(results=fitted, residual_std=resid_std or 0.0)


def backtest(
    series: pd.Series,
    test_size: int = 12,
    seasonal_lag: int = 52,
    seasonal_periods: int = 52,
) -> dict[str, dict[str, float]]:
    """Time-based backtest comparing baseline vs ETS forecasts."""

    series = series.sort_index()
    if len(series) <= test_size:
        raise ValueError("Series too short for requested test window")
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]

    baseline_model = fit_baseline(train, seasonal_lag=seasonal_lag)
    ets_model = fit_sarimax_or_ets(train, seasonal_periods=seasonal_periods)

    baseline_preds = baseline_model.forecast(test_size).reindex(test.index)
    ets_preds = ets_model.forecast(test_size).reindex(test.index)

    metrics = {
        "baseline": _compute_metrics(test, baseline_preds),
        "model": _compute_metrics(test, ets_preds),
    }
    return metrics


def forecast_with_intervals(
    model_name: str,
    model,
    steps: int,
    freq: str,
    last_date: pd.Timestamp,
) -> pd.DataFrame:
    """Produce forecasts with 80% and 95% normal-theory intervals."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    preds = model.forecast(steps)
    if not isinstance(preds, pd.Series):
        preds = pd.Series(preds)
    if preds.index.dtype != "datetime64[ns]":
        offset = to_offset(freq)
        start = last_date + offset
        preds.index = pd.date_range(start=start, periods=len(preds), freq=offset)
    std = getattr(model, "residual_std", None)
    if std is None or np.isnan(std) or std == 0:
        std = float(preds.std(ddof=0)) if len(preds) > 1 else 0.0
    intervals = {}
    for label, z in {"80": 1.2816, "95": 1.96}.items():
        lower = (preds - z * std).clip(lower=0.0)
        upper = (preds + z * std).clip(lower=0.0)
        intervals[f"lower_{label}"] = lower
        intervals[f"upper_{label}"] = upper
    return pd.DataFrame(
        {
            "date": preds.index,
            "y_pred": preds.values,
            "lower_80": intervals["lower_80"].values,
            "upper_80": intervals["upper_80"].values,
            "lower_95": intervals["lower_95"].values,
            "upper_95": intervals["upper_95"].values,
            "model_name": model_name,
        }
    )


def _aggregate_revenue(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    frames = []
    overall = (
        df.groupby(pd.Grouper(key="transaction_date", freq=freq))
        .agg(revenue=("revenue", "sum"))
        .reset_index()
    )
    overall["group_type"] = "overall"
    overall["group_value"] = "all"
    frames.append(overall)

    by_channel = (
        df.groupby([pd.Grouper(key="transaction_date", freq=freq), "channel"])
        .agg(revenue=("revenue", "sum"))
        .reset_index()
        .rename(columns={"channel": "group_value"})
    )
    by_channel["group_type"] = "channel"
    frames.append(by_channel)

    by_segment = (
        df.groupby([pd.Grouper(key="transaction_date", freq=freq), "ltv_segment"])
        .agg(revenue=("revenue", "sum"))
        .reset_index()
        .rename(columns={"ltv_segment": "group_value"})
    )
    by_segment["group_type"] = "segment"
    frames.append(by_segment)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={"transaction_date": "date"})
    completed = _complete_time_index(combined, freq)
    completed.sort_values(["group_type", "group_value", "date"], inplace=True)
    return completed.reset_index(drop=True)


def _complete_time_index(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    output = []
    offset = to_offset(freq)
    for (group_type, group_value), group_df in df.groupby(["group_type", "group_value"]):
        group_df = group_df.set_index("date").sort_index()
        full_index = pd.date_range(group_df.index.min(), group_df.index.max(), freq=offset)
        group_df = group_df.reindex(full_index, fill_value=0.0)
        group_df = group_df.rename_axis("date").reset_index()
        group_df["group_type"] = group_type
        group_df["group_value"] = group_value
        output.append(group_df)
    return pd.concat(output, ignore_index=True)


def _compute_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    actual, predicted = actual.align(predicted, join="inner")
    errors = actual - predicted
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    with np.errstate(divide="ignore", invalid="ignore"):
        perc_errors = np.abs(errors / actual.replace(0, np.nan)) * 100
    perc_errors = perc_errors.replace([np.inf, -np.inf], np.nan).dropna()
    mape = float(perc_errors.mean()) if not perc_errors.empty else float("nan")
    return {"mae": mae, "rmse": rmse, "mape": mape}
