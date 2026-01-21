"""Experimentation and causal inference utilities for marketing incrementality."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


@dataclass
class DiffInMeansResult:
    diff: float
    ci_low: float
    ci_high: float
    p_value: float
    effect_size: float
    power: float
    mean_treatment: float
    mean_control: float
    n_treatment: int
    n_control: int


@dataclass
class MatchingResult:
    att: float
    att_ci: Tuple[float, float]
    ate: float
    ate_ci: Tuple[float, float]
    matched_pairs: pd.DataFrame
    propensities: pd.Series
    balance_table: pd.DataFrame


def generate_synthetic_interventions(
    transactions_path: Path,
    ltv_path: Path,
    output_path: Path,
    pre_window_days: int = 90,
    post_window_days: int = 60,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Create biased treatment data to emulate real marketing interventions."""

    rng = np.random.default_rng(rng_seed)
    transactions = pd.read_csv(transactions_path, parse_dates=["transaction_date"])
    ltv = pd.read_csv(ltv_path)

    required_cols = {
        "customer_id",
        "frequency",
        "recency",
        "T",
        "monetary_value",
        "predicted_ltv_12m",
    }
    missing = required_cols - set(ltv.columns)
    if missing:
        raise ValueError(f"LTV predictions missing columns: {missing}")

    ltv = ltv[list(required_cols)].copy()
    ltv = ltv.sort_values("customer_id").reset_index(drop=True)

    low_cut = ltv["predicted_ltv_12m"].quantile(0.2)
    high_cut = ltv["predicted_ltv_12m"].quantile(0.8)
    ltv["segment"] = np.select(
        [ltv["predicted_ltv_12m"] >= high_cut, ltv["predicted_ltv_12m"] <= low_cut],
        ["High", "Low"],
        default="Mid",
    )
    segment_map = {"Low": 0, "Mid": 1, "High": 2}
    ltv["segment_value"] = ltv["segment"].map(segment_map)

    # Dominant channel per customer
    channel_pref = (
        transactions.groupby(["customer_id", "channel"]).size().reset_index(name="cnt")
    )
    channel_pref = channel_pref.sort_values(["customer_id", "cnt"], ascending=[True, False])
    channel_pref = channel_pref.drop_duplicates("customer_id")[["customer_id", "channel"]]
    ltv = ltv.merge(channel_pref, on="customer_id", how="left")
    fallback_channels = ["email", "sms", "paid_social", "push", "organic", "referral"]
    ltv["channel"] = ltv["channel"].fillna(rng.choice(fallback_channels))

    start_date = transactions["transaction_date"].min() + pd.Timedelta(days=120)
    end_date = transactions["transaction_date"].max() - pd.Timedelta(days=14)
    if end_date <= start_date:
        start_date = transactions["transaction_date"].min()
        end_date = transactions["transaction_date"].max()
    date_range_days = max(1, (end_date - start_date).days)
    ltv["date"] = start_date + pd.to_timedelta(
        rng.integers(0, date_range_days, size=len(ltv)), unit="D"
    )

    # Campaigns mapped to channel families
    campaign_lookup = {
        "email": "email_retention_q3",
        "sms": "sms_flash_sale",
        "push": "push_reengagement",
        "paid_social": "paid_acquisition_refresh",
        "organic": "organic_content_boost",
        "referral": "referral_member_get_member",
    }
    ltv["campaign_id"] = ltv["channel"].map(campaign_lookup)

    # Treatment propensity driven by LTV + engagement
    ltv_score = (ltv["predicted_ltv_12m"] - ltv["predicted_ltv_12m"].min())
    ltv_score /= max(1e-6, ltv_score.max())
    engagement_score = ltv["frequency"] / ltv["frequency"].max()
    base_prob = 0.15 + 0.55 * ltv_score + 0.2 * engagement_score
    treatment_prob = np.clip(base_prob, 0.05, 0.95)
    ltv["treatment"] = rng.binomial(1, treatment_prob)

    customer_txn = {
        cid: grp.sort_values("transaction_date")[["transaction_date", "revenue"]]
        for cid, grp in transactions.groupby("customer_id")
    }

    pre_vals = []
    post_vals = []
    for _, row in ltv.iterrows():
        cid = row["customer_id"]
        campaign_date = row["date"]
        pre_start = campaign_date - pd.Timedelta(days=pre_window_days)
        post_end = campaign_date + pd.Timedelta(days=post_window_days)
        txns = customer_txn.get(cid)
        if txns is None:
            pre_vals.append(0.0)
            post_vals.append(0.0)
            continue
        pre_mask = (txns["transaction_date"] >= pre_start) & (txns["transaction_date"] < campaign_date)
        post_mask = (txns["transaction_date"] >= campaign_date) & (
            txns["transaction_date"] <= post_end
        )
        pre_total = float(txns.loc[pre_mask, "revenue"].sum())
        post_total = float(txns.loc[post_mask, "revenue"].sum())
        # Inject incremental lift for treated customers to simulate effect + noise
        if row["treatment"] == 1:
            post_total += float(0.15 * row["predicted_ltv_12m"])
        noise = rng.normal(0, 10)
        post_total = max(0.0, post_total + noise)
        pre_total = max(0.0, pre_total + rng.normal(0, 5))
        pre_vals.append(pre_total)
        post_vals.append(post_total)

    ltv["pre_period_revenue"] = pre_vals
    ltv["post_period_revenue"] = post_vals

    dataset = ltv[
        [
            "customer_id",
            "date",
            "channel",
            "treatment",
            "campaign_id",
            "pre_period_revenue",
            "post_period_revenue",
            "frequency",
            "recency",
            "T",
            "monetary_value",
            "segment",
            "segment_value",
        ]
    ].copy()
    dataset.sort_values(["date", "customer_id"], inplace=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    return dataset


def difference_in_means(
    data: pd.DataFrame,
    outcome_col: str = "post_period_revenue",
    treatment_col: str = "treatment",
    alpha: float = 0.05,
) -> DiffInMeansResult:
    """Classical difference-in-means test with z-approximation."""

    treated = data[data[treatment_col] == 1][outcome_col]
    control = data[data[treatment_col] == 0][outcome_col]
    n_t, n_c = len(treated), len(control)
    if n_t == 0 or n_c == 0:
        raise ValueError("Both treatment and control groups need observations")
    mean_t = treated.mean()
    mean_c = control.mean()
    diff = mean_t - mean_c
    var_t = treated.var(ddof=1)
    var_c = control.var(ddof=1)
    se = np.sqrt(var_t / n_t + var_c / n_c)
    z = diff / se if se > 0 else 0.0
    p_value = 2 * (1 - _normal_cdf(abs(z)))
    z_alpha = _inverse_normal_cdf(1 - alpha / 2)
    ci_low = diff - z_alpha * se
    ci_high = diff + z_alpha * se
    pooled_std = np.sqrt(((n_t - 1) * var_t + (n_c - 1) * var_c) / (n_t + n_c - 2))
    effect_size = diff / pooled_std if pooled_std > 0 else 0.0
    power = _normal_cdf(abs(z) - z_alpha)
    return DiffInMeansResult(
        diff=float(diff),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(p_value),
        effect_size=float(effect_size),
        power=float(power),
        mean_treatment=float(mean_t),
        mean_control=float(mean_c),
        n_treatment=int(n_t),
        n_control=int(n_c),
    )


def propensity_score_matching(
    data: pd.DataFrame,
    covariate_cols: Iterable[str],
    outcome_col: str = "post_period_revenue",
    treatment_col: str = "treatment",
    n_neighbors: int = 1,
    n_bootstrap: int = 500,
    random_state: int = 42,
) -> MatchingResult:
    """Estimate ATT / ATE via propensity score matching with diagnostics."""

    model_data = data.copy()
    model_data = model_data.dropna(subset=covariate_cols)
    X = model_data[list(covariate_cols)].astype(float)
    y = model_data[treatment_col].astype(int)

    logit = LogisticRegression(max_iter=2000)
    logit.fit(X, y)
    propensity = logit.predict_proba(X)[:, 1]
    propensity = np.clip(propensity, 0.01, 0.99)
    model_data = model_data.assign(propensity=propensity)

    treated = model_data[model_data[treatment_col] == 1]
    control = model_data[model_data[treatment_col] == 0]
    if treated.empty or control.empty:
        raise ValueError("Need both treated and control observations for matching")

    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(control[["propensity"]])
    distances, indices = nn.kneighbors(treated[["propensity"]])
    matched_control = control.iloc[indices.flatten()].copy()
    matched_control["match_group"] = treated.index.values.repeat(n_neighbors)
    matched_control["match_role"] = "control"

    matched_treated = treated.copy()
    matched_treated["match_group"] = treated.index.values
    matched_treated["match_role"] = "treated"

    matched_pairs = pd.concat([matched_treated, matched_control], ignore_index=True)

    treated_outcomes = matched_treated[outcome_col].to_numpy()
    control_outcomes = matched_control[outcome_col].to_numpy()
    min_len = min(len(treated_outcomes), len(control_outcomes))
    diff_samples = treated_outcomes[:min_len] - control_outcomes[:min_len]
    att = float(diff_samples.mean())
    att_ci = _bootstrap_ci(diff_samples, n_bootstrap, random_state)

    ipw_terms = model_data.apply(
        lambda row: row[treatment_col] * row[outcome_col] / row["propensity"]
        - (1 - row[treatment_col]) * row[outcome_col] / (1 - row["propensity"]),
        axis=1,
    )
    ate = float(ipw_terms.mean())
    ate_ci = _bootstrap_ci(ipw_terms.to_numpy(), n_bootstrap, random_state)

    balance_pre = _standardized_mean_differences(
        model_data, covariate_cols, treatment_col
    )
    balance_pre["stage"] = "Pre-Match"
    balance_post = _standardized_mean_differences(
        matched_pairs, covariate_cols, treatment_col
    )
    balance_post["stage"] = "Post-Match"
    balance_table = pd.concat([balance_pre, balance_post], ignore_index=True)

    return MatchingResult(
        att=att,
        att_ci=att_ci,
        ate=ate,
        ate_ci=ate_ci,
        matched_pairs=matched_pairs,
        propensities=model_data.set_index("customer_id")["propensity"],
        balance_table=balance_table,
    )


def _standardized_mean_differences(
    df: pd.DataFrame, covariate_cols: Iterable[str], treatment_col: str
) -> pd.DataFrame:
    rows = []
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    for col in covariate_cols:
        t_vals = treated[col].astype(float)
        c_vals = control[col].astype(float)
        mean_t = t_vals.mean()
        mean_c = c_vals.mean()
        var_t = t_vals.var(ddof=1)
        var_c = c_vals.var(ddof=1)
        pooled = (var_t + var_c) / 2
        smd = (mean_t - mean_c) / np.sqrt(pooled) if pooled > 0 else 0.0
        rows.append({
            "covariate": col,
            "mean_treatment": float(mean_t),
            "mean_control": float(mean_c),
            "smd": float(smd),
        })
    return pd.DataFrame(rows)


def _bootstrap_ci(values: np.ndarray, n_bootstrap: int, random_state: int) -> Tuple[float, float]:
    rng = np.random.default_rng(random_state)
    clean = np.asarray(values, dtype=float)
    if clean.size == 0:
        return (float("nan"), float("nan"))
    boot = []
    for _ in range(n_bootstrap):
        sample = rng.choice(clean, size=clean.size, replace=True)
        boot.append(sample.mean())
    lower = float(np.percentile(boot, 2.5))
    upper = float(np.percentile(boot, 97.5))
    return (lower, upper)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _inverse_normal_cdf(p: float) -> float:
    return math.sqrt(2) * _inverse_erf(2 * p - 1)


def _inverse_erf(x: float) -> float:
    a = 0.147
    sign = 1 if x >= 0 else -1
    ln = math.log(1 - x * x)
    first = 2 / (math.pi * a) + ln / 2
    second = ln / a
    return sign * math.sqrt(math.sqrt(first * first - second) - first)
