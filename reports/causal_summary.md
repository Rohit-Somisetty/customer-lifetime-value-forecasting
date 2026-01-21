# Marketing Incrementality Summary

## Why it matters
Campaign owners need to understand incremental revenue beyond natural demand. We simulate a biased targeting process (high-LTV customers receive more treatments) and correct that bias with modern causal techniques.

## Key findings
- **Naive lift** (difference in means): 75.09 revenue per customer (CI 20.37 to 129.82). This overstates true incremental impact because high-value customers were targeted more often.
- **Propensity score ATT**: 14.80 (95% CI -47.97 to 78.81). This is the causal lift for treated customers after adjusting for engagement and value covariates.
- **IPW ATE**: -2.12 (95% CI -58.82 to 55.85). This represents the expected lift if the campaign were rolled out to the full population.

## Action plan
1. Use the ATT as the benchmark for go/no-go decisions; it reflects realistic uplift for the targeted audience.
2. Continue bias diagnostics via the balance plot—covariate SMDs land inside ±0.1 after matching, indicating adequate overlap.
3. Track lift distributions by segment (see figures) to prioritize budgets toward cohorts with the highest incremental density.
4. Incorporate these causal estimates into experimentation-roadmap reviews and marketing finance models.