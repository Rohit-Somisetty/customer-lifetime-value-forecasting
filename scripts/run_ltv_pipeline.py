"""Command-line entry point to generate CLV predictions."""

from pathlib import Path

from src.ltv_model import (
    assess_model_inputs,
    build_calibration_holdout,
    evaluate_holdout_predictions,
    fit_bgf_model,
    fit_ggf_model,
    generate_ltv_predictions,
    load_transactions,
    save_predictions,
    summarize_customers,
    summarize_ltv_distribution,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "transactions.csv"
    output_path = project_root / "data" / "ltv_predictions.csv"

    transactions = load_transactions(data_path)
    summary = summarize_customers(transactions)
    diagnostics = assess_model_inputs(summary)
    cal_hold, cutoff = build_calibration_holdout(transactions)

    bgf = fit_bgf_model(cal_hold)
    ggf = fit_ggf_model(summary)
    holdout_eval = evaluate_holdout_predictions(bgf, cal_hold)

    predictions = generate_ltv_predictions(summary, bgf, ggf)
    save_predictions(predictions, output_path)

    stats_6m = summarize_ltv_distribution(predictions, "predicted_ltv_6m")
    stats_12m = summarize_ltv_distribution(predictions, "predicted_ltv_12m")

    print("Input diagnostics:", diagnostics)
    print("Holdout cutoff:", cutoff.strftime("%Y-%m-%d"))
    print("Holdout evaluation:", holdout_eval)
    print("6M LTV stats:", stats_6m)
    print("12M LTV stats:", stats_12m)


if __name__ == "__main__":
    main()
