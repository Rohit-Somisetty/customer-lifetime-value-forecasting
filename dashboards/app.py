from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

st.set_page_config(page_title="CLV & Incrementality Control Room", layout="wide")
st.title("Customer Lifetime Value Forecasting – Control Room")
st.caption("Single source of truth for LTV, revenue outlook, and marketing incrementality.")


def _csv_path(name: str) -> Path:
    path = DATA_DIR / name
    if not path.exists():
        st.error(f"Missing data file: {path.relative_to(PROJECT_ROOT)}. Run scripts/run_all.py first.")
        st.stop()
    return path


@st.cache_data(show_spinner=False)
def load_csv(name: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(_csv_path(name), parse_dates=parse_dates)


ltv_df = load_csv("ltv_predictions.csv")
segments_df = load_csv("customer_segments.csv")
overall_fc = load_csv("revenue_forecasts_overall.csv", parse_dates=["date"])
segment_fc = load_csv("revenue_forecasts_by_segment.csv", parse_dates=["date"])
channel_fc = load_csv("revenue_forecasts_by_channel.csv", parse_dates=["date"])
causal_results = load_csv("causal_results.csv")
balance_table = load_csv("causal_balance_table.csv")

figures = {
    "balance": REPORTS_DIR / "figures" / "balance_plot.png",
    "lift": REPORTS_DIR / "figures" / "lift_distribution.png",
}

for label, fig_path in figures.items():
    if not fig_path.exists():
        st.warning(
            f"Missing figure {fig_path.relative_to(PROJECT_ROOT)}. Run scripts/run_causal_pipeline.py to regenerate."
        )


# --- Helper metrics --------------------------------------------------------

def compute_ltv_kpis(df: pd.DataFrame) -> dict[str, float]:
    stats = {}
    stats["mean"] = float(df["predicted_ltv_12m"].mean())
    stats["median"] = float(df["predicted_ltv_12m"].median())
    q90 = df["predicted_ltv_12m"].quantile(0.9)
    top_decile = df.loc[df["predicted_ltv_12m"] >= q90, "predicted_ltv_12m"].sum()
    total = df["predicted_ltv_12m"].sum()
    stats["top_decile_share"] = float(top_decile / total) if total else 0.0
    return stats


def overall_revenue_headline(fc: pd.DataFrame, horizon: int = 12) -> float:
    subset = fc[fc["horizon_weeks"] == horizon]
    return float(subset["y_pred"].sum()) if not subset.empty else float("nan")


def causal_headline(df: pd.DataFrame) -> tuple[float, float, float]:
    naive = df.loc[df["metric"] == "naive_diff_in_means", "estimate"].iloc[0]
    att = df.loc[df["metric"] == "psm_att", "estimate"].iloc[0]
    ate = df.loc[df["metric"] == "ipw_ate", "estimate"].iloc[0]
    return float(naive), float(att), float(ate)


ltv_stats = compute_ltv_kpis(ltv_df)
revenue_12w = overall_revenue_headline(overall_fc, horizon=12)
naive_lift, att_lift, ate_lift = causal_headline(causal_results)


tab_exec, tab_segments, tab_forecasts, tab_incrementality = st.tabs(
    [
        "Executive Overview",
        "LTV Segments",
        "Forecasting",
        "Incrementality",
    ]
)

with tab_exec:
    st.subheader("Key KPIs")
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Mean 12M LTV", f"${ltv_stats['mean']:,.0f}")
    kpi_cols[1].metric("Median 12M LTV", f"${ltv_stats['median']:,.0f}")
    kpi_cols[2].metric("Top-Decile Share", f"{ltv_stats['top_decile_share']*100:,.1f}%")
    kpi_cols[3].metric("Forecasted Rev (Next 12w)", f"${revenue_12w:,.0f}")

    st.markdown("### Causal Lift Snapshot")
    lift_cols = st.columns(3)
    lift_cols[0].metric("Naive lift", f"${naive_lift:,.1f}/cust")
    lift_cols[1].metric("PSM ATT", f"${att_lift:,.1f}/cust")
    lift_cols[2].metric("IPW ATE", f"${ate_lift:,.1f}/cust")
    st.caption(
        "Naive lift shows targeting bias; propensity-adjusted metrics provide realistic incremental expectations."
    )

with tab_segments:
    st.subheader("LTV Distribution")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(ltv_df["predicted_ltv_12m"], bins=30, color="#4a90e2", alpha=0.8)
    ax.set_xlabel("Predicted LTV (12M)")
    ax.set_ylabel("Customers")
    st.pyplot(fig)

    st.markdown("### Top Customers by Predicted LTV")
    top_customers = (
        ltv_df.sort_values("predicted_ltv_12m", ascending=False)
        .head(10)[["customer_id", "predicted_ltv_12m", "expected_avg_value"]]
    )
    top_customers.columns = ["Customer", "Predicted LTV (12M)", "Avg Order Value"]
    st.dataframe(top_customers.style.format({"Predicted LTV (12M)": "${:,.0f}", "Avg Order Value": "${:,.0f}"}))

    st.markdown("### Segment Mix by Primary Channel")
    if "primary_channel" in segments_df.columns:
        mix = (
            segments_df.groupby(["primary_channel", "ltv_segment"]).size().reset_index(name="customers")
        )
        totals = mix.groupby("primary_channel")["customers"].transform("sum")
        mix["share"] = mix["customers"] / totals
        pivot = mix.pivot(index="primary_channel", columns="ltv_segment", values="share").fillna(0)
        st.dataframe(pivot.style.format("{:.0%}"))
    else:
        st.info("Primary channel metadata missing; rerun scripts/run_forecasting_pipeline.py to populate it.")

with tab_forecasts:
    st.subheader("Forecast Explorer")
    level = st.selectbox("View", ["Overall", "Segment", "Channel"], index=0)
    horizon = st.selectbox("Horizon (weeks)", [12, 26], index=0)

    def plot_forecast(df: pd.DataFrame, label: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["date"], df["y_pred"], label="Forecast", color="#d17a22")
        ax.fill_between(df["date"], df["lower_80"], df["upper_80"], color="#d17a22", alpha=0.2, label="80% PI")
        ax.fill_between(df["date"], df["lower_95"], df["upper_95"], color="#d17a22", alpha=0.1, label="95% PI")
        ax.set_title(f"{label} revenue forecast ({horizon}w horizon)")
        ax.set_ylabel("Revenue")
        ax.set_xlabel("Week")
        ax.legend(loc="upper left")
        st.pyplot(fig)

    if level == "Overall":
        filtered = overall_fc[overall_fc["horizon_weeks"] == horizon]
        plot_forecast(filtered, "Overall")
    elif level == "Segment":
        segment = st.selectbox("Segment", sorted(segment_fc["segment"].dropna().unique().tolist()))
        filtered = segment_fc[
            (segment_fc["segment"] == segment) & (segment_fc["horizon_weeks"] == horizon)
        ]
        plot_forecast(filtered, f"Segment: {segment}")
    else:
        channel = st.selectbox("Channel", sorted(channel_fc["channel"].dropna().unique().tolist()))
        filtered = channel_fc[
            (channel_fc["channel"] == channel) & (channel_fc["horizon_weeks"] == horizon)
        ]
        plot_forecast(filtered, f"Channel: {channel}")

with tab_incrementality:
    st.subheader("Naive vs Adjusted Lift")
    display_cols = ["metric", "estimate", "ci_low", "ci_high"]
    st.dataframe(
        causal_results[display_cols]
        .assign(
            estimate=lambda df: df["estimate"].map(lambda x: f"${x:,.1f}"),
            ci_low=lambda df: df["ci_low"].map(lambda x: f"${x:,.1f}"),
            ci_high=lambda df: df["ci_high"].map(lambda x: f"${x:,.1f}"),
        )
        .rename(columns={
            "metric": "Metric",
            "estimate": "Estimate",
            "ci_low": "CI Low",
            "ci_high": "CI High",
        })
    )

    st.markdown("### Balance Diagnostics")
    if figures["balance"].exists():
        st.image(str(figures["balance"]), caption="Standardized mean differences")
    else:
        st.info("Balance plot not found.")

    st.markdown("### Lift Distribution")
    if figures["lift"].exists():
        st.image(str(figures["lift"]), caption="Post minus pre revenue lift")

    st.markdown(
        "**What this means:** Naive comparisons (\~$75) greatly overstate true lift. After adjusting for engagement and value, incremental revenue falls to low double digits. Focus treatments on High-LTV cohorts and prioritize controlled tests before scaling to lower tiers."
    )
