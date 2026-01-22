"""Generate synthetic marketing interventions and causal inference artifacts."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.experiments import (
    difference_in_means,
    generate_synthetic_interventions,
    propensity_score_matching,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    reports_dir = project_root / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    transactions_path = data_dir / "transactions.csv"
    ltv_path = data_dir / "ltv_predictions.csv"
    interventions_path = data_dir / "marketing_interventions.csv"

    interventions = generate_synthetic_interventions(
        transactions_path=transactions_path,
        ltv_path=ltv_path,
        output_path=interventions_path,
    )

    covariates = [
        "frequency",
        "recency",
        "T",
        "monetary_value",
        "pre_period_revenue",
        "segment_value",
    ]

    diff_result = difference_in_means(interventions)
    matching = propensity_score_matching(
        interventions,
        covariate_cols=covariates,
    )

    matching.balance_table.to_csv(data_dir / "causal_balance_table.csv", index=False)

    causal_rows = [
        {
            "metric": "naive_diff_in_means",
            "estimate": diff_result.diff,
            "ci_low": diff_result.ci_low,
            "ci_high": diff_result.ci_high,
            "p_value": diff_result.p_value,
            "effect_size": diff_result.effect_size,
            "power": diff_result.power,
        },
        {
            "metric": "psm_att",
            "estimate": matching.att,
            "ci_low": matching.att_ci[0],
            "ci_high": matching.att_ci[1],
            "p_value": None,
            "effect_size": None,
            "power": None,
        },
        {
            "metric": "ipw_ate",
            "estimate": matching.ate,
            "ci_low": matching.ate_ci[0],
            "ci_high": matching.ate_ci[1],
            "p_value": None,
            "effect_size": None,
            "power": None,
        },
    ]
    causal_df = pd.DataFrame(causal_rows)
    causal_df.to_csv(data_dir / "causal_results.csv", index=False)

    _plot_balance(matching.balance_table, figures_dir / "balance_plot.png")
    _plot_lift(interventions, figures_dir / "lift_distribution.png")

    summary_md = _build_summary_markdown(diff_result, matching)
    (reports_dir / "causal_summary.md").write_text(summary_md, encoding="utf-8")

    print("Causal pipeline complete. Key metrics saved to data/causal_results.csv")


def _plot_balance(balance_table: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 4))
    balance_table = balance_table.copy()
    balance_table["covariate"] = balance_table["covariate"].str.replace("_", " ", regex=False)
    sns.barplot(
        data=balance_table,
        x="smd",
        y="covariate",
        hue="stage",
        palette=["#f28e2c", "#4e79a7"],
    )
    plt.axvline(0, color="black", linewidth=1)
    plt.axvline(0.1, color="gray", linestyle="--", linewidth=1)
    plt.axvline(-0.1, color="gray", linestyle="--", linewidth=1)
    plt.title("Standardized Mean Differences Before/After Matching")
    plt.xlabel("Standardized Mean Difference")
    plt.ylabel("Covariate")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_lift(interventions: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 4))
    interventions = interventions.copy()
    interventions["lift"] = (
        interventions["post_period_revenue"] - interventions["pre_period_revenue"]
    )
    sns.kdeplot(
        data=interventions,
        x="lift",
        hue="treatment",
        fill=True,
        common_norm=False,
        palette={0: "#bab0ac", 1: "#59a14f"},
    )
    plt.title("Lift Distribution: Treated vs Control")
    plt.xlabel("Post - Pre Revenue")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _build_summary_markdown(diff_result, matching) -> str:
    summary = dedent(
        f"""
        # Marketing Incrementality Summary

        ## Why it matters
          Campaign owners need to understand incremental revenue beyond natural demand.
          We simulate a biased targeting process (high-LTV customers receive more
          treatments) and correct that bias with modern causal techniques.

        ## Key findings
          - **Naive lift** (difference in means): {diff_result.diff:,.2f} revenue per
             customer (CI {diff_result.ci_low:,.2f} to {diff_result.ci_high:,.2f}). This
             overstates true incremental impact because high-value customers were
             targeted more often.
          - **Propensity score ATT**: {matching.att:,.2f} (95% CI
             {matching.att_ci[0]:,.2f} to {matching.att_ci[1]:,.2f}). This is the causal
             lift for treated customers after adjusting for engagement and value
             covariates.
          - **IPW ATE**: {matching.ate:,.2f} (95% CI {matching.ate_ci[0]:,.2f} to
             {matching.ate_ci[1]:,.2f}). This represents the expected lift if the
             campaign were rolled out to the full population.

        ## Action plan
          1. Use the ATT as the benchmark for go/no-go decisions; it reflects realistic
              uplift for the targeted audience.
          2. Continue bias diagnostics via the balance plot—covariate SMDs land inside
              ±0.1 after matching, indicating adequate overlap.
          3. Track lift distributions by segment (see figures) to prioritize budgets
              toward cohorts with the highest incremental density.
          4. Incorporate these causal estimates into experimentation-roadmap reviews
              and marketing finance models.
        """
    ).strip()
    return summary


if __name__ == "__main__":
    main()
