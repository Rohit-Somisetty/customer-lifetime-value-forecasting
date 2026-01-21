"""Reusable helpers for preparing BG/NBD + Gamma-Gamma LTV models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import (
    calibration_and_holdout_data,
    summary_data_from_transaction_data,
)

DEFAULT_FREQ = "D"
DEFAULT_HOLDOUT_DAYS = 90
DAYS_PER_MONTH = 30


@dataclass
class LTVArtifacts:
    """Container for commonly reused modeling artifacts."""

    summary: pd.DataFrame
    calibration_table: pd.DataFrame
    bgf: BetaGeoFitter
    ggf: GammaGammaFitter


def load_transactions(csv_path: Path) -> pd.DataFrame:
    """Load the raw transaction log with simple schema checks."""

    df = pd.read_csv(csv_path, parse_dates=["transaction_date"])
    required_cols = {"customer_id", "transaction_date", "revenue"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"transactions file missing columns: {missing}")
    if (df["revenue"] <= 0).any():
        raise ValueError("revenue values must be strictly positive for lifetimes models")
    return df


def summarize_customers(
    transactions: pd.DataFrame, freq: str = DEFAULT_FREQ
) -> pd.DataFrame:
    """Aggregate transaction history to frequency/recency/monetary features."""

    summary = summary_data_from_transaction_data(
        transactions,
        customer_id_col="customer_id",
        datetime_col="transaction_date",
        monetary_value_col="revenue",
        freq=freq,
    )
    summary = summary.reset_index().rename(columns={"index": "customer_id"})
    summary["monetary_value"] = summary["monetary_value"].fillna(0.0)
    return summary


def build_calibration_holdout(
    transactions: pd.DataFrame,
    holdout_days: int = DEFAULT_HOLDOUT_DAYS,
    freq: str = DEFAULT_FREQ,
) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """Create calibration/holdout splits using the last N days as holdout."""

    if holdout_days <= 0:
        raise ValueError("holdout_days must be positive")
    max_date = transactions["transaction_date"].max()
    cutoff = max_date - pd.Timedelta(days=holdout_days)
    cal_hold = calibration_and_holdout_data(
        transactions,
        customer_id_col="customer_id",
        datetime_col="transaction_date",
        monetary_value_col="revenue",
        freq=freq,
        calibration_period_end=cutoff,
    ).reset_index().rename(columns={"index": "customer_id"})
    return cal_hold, cutoff


def fit_bgf_model(calibration_table: pd.DataFrame) -> BetaGeoFitter:
    """Train a penalized BG/NBD model on calibration features."""

    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(
        calibration_table["frequency_cal"],
        calibration_table["recency_cal"],
        calibration_table["T_cal"],
    )
    return bgf


def fit_ggf_model(summary_table: pd.DataFrame) -> GammaGammaFitter:
    """Train a Gamma-Gamma model on customers with repeat purchases."""

    mask = (summary_table["frequency"] > 0) & (summary_table["monetary_value"] > 0)
    filtered = summary_table.loc[mask]
    if filtered.empty:
        raise ValueError("Need customers with repeat purchases to fit Gamma-Gamma model")
    ggf = GammaGammaFitter(penalizer_coef=0.001)
    ggf.fit(filtered["frequency"], filtered["monetary_value"])
    return ggf


def assess_model_inputs(summary_table: pd.DataFrame) -> Dict[str, float]:
    """Compute quick diagnostics that align with BG/NBD assumptions."""

    total = len(summary_table)
    freq_positive = (summary_table["frequency"] > 0).sum()
    monetary_positive = (summary_table["monetary_value"] > 0).sum()
    diagnostics = {
        "customers_total": int(total),
        "share_repeat_purchasers": float(freq_positive / total),
        "share_positive_monetary": float(monetary_positive / total),
        "avg_frequency": float(summary_table["frequency"].mean()),
        "avg_recency": float(summary_table["recency"].mean()),
        "avg_T": float(summary_table["T"].mean()),
    }
    return diagnostics


def evaluate_holdout_predictions(
    bgf: BetaGeoFitter, calibration_table: pd.DataFrame
) -> Dict[str, float]:
    """Compare predicted vs actual holdout transactions."""

    preds = bgf.predict(
        calibration_table["duration_holdout"],
        calibration_table["frequency_cal"],
        calibration_table["recency_cal"],
        calibration_table["T_cal"],
    )
    actuals = calibration_table["frequency_holdout"]
    diff = preds - actuals
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(np.square(diff)))
    corr = float(np.corrcoef(preds, actuals)[0, 1]) if actuals.std() else float("nan")
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "pred_mean": float(preds.mean()),
        "actual_mean": float(actuals.mean()),
        "corr": corr,
    }


def _expected_avg_value(
    ggf: GammaGammaFitter, summary_table: pd.DataFrame
) -> pd.Series:
    """Helper that applies Gamma-Gamma expectation with sensible fallbacks."""

    result = pd.Series(np.nan, index=summary_table.index)
    mask = (summary_table["frequency"] > 0) & (summary_table["monetary_value"] > 0)
    if mask.any():
        result.loc[mask] = ggf.conditional_expected_average_profit(
            summary_table.loc[mask, "frequency"],
            summary_table.loc[mask, "monetary_value"],
        )
    fallback = summary_table.loc[mask, "monetary_value"].mean()
    if np.isnan(fallback):
        fallback = summary_table["monetary_value"].mean()
    result = result.fillna(fallback if not np.isnan(fallback) else 0.0)
    return result


def generate_ltv_predictions(
    summary_table: pd.DataFrame,
    bgf: BetaGeoFitter,
    ggf: GammaGammaFitter,
    horizons_months: Iterable[int] = (6, 12),
    freq: str = DEFAULT_FREQ,
    discount_rate: float = 0.01,
) -> pd.DataFrame:
    """Return per-customer LTV predictions for each requested horizon."""

    predictions = summary_table[
        ["customer_id", "frequency", "recency", "T", "monetary_value"]
    ].copy()
    predictions["expected_avg_value"] = _expected_avg_value(ggf, summary_table)

    for months in horizons_months:
        periods = months * DAYS_PER_MONTH
        exp_purchases = bgf.conditional_expected_number_of_purchases_up_to_time(
            periods,
            predictions["frequency"],
            predictions["recency"],
            predictions["T"],
        )
        ltv = ggf.customer_lifetime_value(
            bgf,
            frequency=predictions["frequency"],
            recency=predictions["recency"],
            T=predictions["T"],
            monetary_value=predictions["monetary_value"],
            time=periods,
            discount_rate=discount_rate,
            freq=freq,
        )
        predictions[f"expected_purchases_{months}m"] = exp_purchases
        predictions[f"predicted_ltv_{months}m"] = ltv
    return predictions


def summarize_ltv_distribution(
    predictions: pd.DataFrame, horizon_column: str
) -> Dict[str, float]:
    """Compute descriptive stats for stakeholder communication."""

    horizon_values = predictions[horizon_column]
    top_decile_threshold = horizon_values.quantile(0.9)
    top_decile_value = horizon_values[horizon_values >= top_decile_threshold].sum()
    total_value = horizon_values.sum()
    return {
        "mean": float(horizon_values.mean()),
        "median": float(horizon_values.median()),
        "top_decile_share": float(top_decile_value / total_value) if total_value else 0.0,
    }


def save_predictions(predictions: pd.DataFrame, output_path: Path) -> None:
    """Persist LTV results for downstream dashboards."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
