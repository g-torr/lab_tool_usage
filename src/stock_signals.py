"""Explainable public-equity signals derived from equipment adoption mentions."""

import pandas as pd

SIGNAL_COLUMNS = [
    "parent_company", "ticker", "mentions", "market_share",
    "recent_change_pct", "models_mentioned", "signal", "rationale",
]


def _ticker_is_listed(ticker: object) -> bool:
    value = str(ticker).strip().upper()
    return bool(value) and value not in {
        "N/A", "NA", "PRIVATE", "PRIVATE (SUB)", "VARIOUS", "NAN", "NONE"
    }


def build_stock_signals(data: pd.DataFrame, lookback_days: int = 90) -> pd.DataFrame:
    """Rank listed parents using mention volume and recent momentum.

    This is a research signal only: it excludes valuation, prices, financial
    statements, and investor-specific circumstances.
    """
    required = {"day", "parent_company", "mention_count", "ticker"}
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"Stock signal data is missing columns: {', '.join(sorted(missing))}")

    work = data.copy()
    work["day"] = pd.to_datetime(work["day"], errors="coerce")
    work["mention_count"] = pd.to_numeric(work["mention_count"], errors="coerce").fillna(0)
    work["ticker"] = work["ticker"].fillna("").astype(str).str.strip().str.upper()
    work = work[
        work["day"].notna()
        & work["parent_company"].notna()
        & work["ticker"].map(_ticker_is_listed)
    ]
    if work.empty:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)

    end = work["day"].max()
    cutoff = end - pd.Timedelta(days=lookback_days)
    prior_start = cutoff - pd.Timedelta(days=lookback_days)
    # Compare two equal-length windows. Previously, ``prior`` included the
    # entire history, which made the table incomparable to a recent trend.
    recent = work[work["day"] > cutoff]
    prior = work[(work["day"] > prior_start) & (work["day"] <= cutoff)]
    total_mentions = recent["mention_count"].sum()

    group_cols = ["parent_company", "ticker"]
    grouped = recent.groupby(group_cols, as_index=False).agg(
        mentions=("mention_count", "sum"),
        models_mentioned=("resolved_machine", "nunique")
        if "resolved_machine" in recent.columns else ("parent_company", "size"),
    )
    grouped["market_share"] = grouped["mentions"] / total_mentions if total_mentions else 0.0

    prior_grouped = prior.groupby(group_cols, as_index=False)["mention_count"].sum()
    prior_grouped = prior_grouped.rename(columns={"mention_count": "prior_mentions"})
    grouped = grouped.merge(prior_grouped, on=group_cols, how="left").fillna({"prior_mentions": 0})
    grouped["recent_change_pct"] = grouped.apply(
        lambda row: ((row["mentions"] - row["prior_mentions"]) / row["prior_mentions"] * 100)
        if row["prior_mentions"] else (100.0 if row["mentions"] else 0.0), axis=1
    )

    def classify(row: pd.Series) -> str:
        if row["mentions"] >= 5 and row["recent_change_pct"] >= 10:
            return "Positive research signal"
        if row["mentions"] >= 3:
            return "Watch"
        return "Insufficient evidence"

    grouped["signal"] = grouped.apply(classify, axis=1)
    grouped["rationale"] = grouped.apply(
        lambda row: f"{int(row['mentions']):,} recent mentions across "
        f"{int(row['models_mentioned']):,} model(s); {row['recent_change_pct']:+.0f}% "
        "versus the prior period.", axis=1
    )
    return grouped.sort_values(["signal", "mentions"], ascending=[True, False])[SIGNAL_COLUMNS].reset_index(drop=True)
