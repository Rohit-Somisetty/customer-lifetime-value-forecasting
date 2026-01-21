"""Generate README-friendly preview images from project data."""

from __future__ import annotations

import argparse
from pathlib import Path
from shutil import copyfile

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")


def plot_ltv_distribution(data_dir: Path, output_dir: Path) -> None:
    ltv = pd.read_csv(data_dir / "ltv_predictions.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ltv["predicted_ltv_12m"], bins=30, color="#4a90e2", alpha=0.8)
    ax.set_title("Predicted 12M LTV Distribution")
    ax.set_xlabel("Predicted LTV (12M)")
    ax.set_ylabel("Customers")
    fig.tight_layout()
    fig.savefig(output_dir / "ltv_distribution.png", dpi=150)
    plt.close(fig)


def plot_overall_forecast(data_dir: Path, output_dir: Path, horizon_weeks: int) -> None:
    forecast = pd.read_csv(data_dir / "revenue_forecasts_overall.csv", parse_dates=["date"])
    horizon_slice = forecast[forecast["horizon_weeks"] == horizon_weeks]

    if horizon_slice.empty:
        raise ValueError(f"No overall forecast rows found for horizon {horizon_weeks} weeks")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(horizon_slice["date"], horizon_slice["y_pred"], color="#d17a22", label="Forecast")
    ax.fill_between(
        horizon_slice["date"],
        horizon_slice["lower_80"],
        horizon_slice["upper_80"],
        color="#d17a22",
        alpha=0.2,
        label="80% PI",
    )
    ax.fill_between(
        horizon_slice["date"],
        horizon_slice["lower_95"],
        horizon_slice["upper_95"],
        color="#d17a22",
        alpha=0.1,
        label="95% PI",
    )
    ax.set_title(f"Overall Weekly Revenue Forecast ({horizon_weeks}W)")
    ax.set_xlabel("Week")
    ax.set_ylabel("Revenue")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "forecast_overall.png", dpi=150)
    plt.close(fig)


def copy_balance_plot(report_dir: Path, output_dir: Path) -> None:
    balance_plot = report_dir / "figures" / "balance_plot.png"
    if balance_plot.exists():
        copyfile(balance_plot, output_dir / "causal_balance.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root containing data/ and reports/ directories",
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=26,
        help="Forecast horizon (in weeks) used for the preview plot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    data_dir = root / "data"
    report_dir = root / "reports"
    output_dir = root / "assets"
    output_dir.mkdir(exist_ok=True)

    plot_ltv_distribution(data_dir, output_dir)
    plot_overall_forecast(data_dir, output_dir, args.forecast_horizon)
    copy_balance_plot(report_dir, output_dir)


if __name__ == "__main__":
    main()
