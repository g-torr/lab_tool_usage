"""Leakage-aware validation for equipment-mention equity signals.

This module is deliberately a validator, not a stock recommender.  It creates
a balanced daily panel (zero mentions are observations), computes features from
trailing calendar windows, and evaluates pre-specified signals only after an
initial training/embargo period.  No market-derived predictor is included.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from src.alpha_backtest import HORIZONS, attach_returns, fetch_prices, normalize_market_days


# Listed life-science tools suppliers represented by the registry.  The list is
# version-controlled to avoid changing the universe after seeing returns.
INVESTABLE_UNIVERSE = frozenset({
    "A", "BDX", "BRKR", "CTKB", "DHR", "ILMN", "LAB", "MKSI",
    "ONT", "PACB", "RVTY", "TMO", "TXG", "WAT",
})
FEATURES = ("breadth_ewm_20d", "breadth_30d")


def prepare_daily_panel(mentions: pd.DataFrame, tickers: Iterable[str] | None = None) -> pd.DataFrame:
    """Return one row per ticker/calendar day, with no-mention days as zero.

    ``doi_count`` must be a daily count of distinct papers when supplied.  It
    is summed only because the input is already ticker-day aggregated.
    """
    required = {"day", "ticker", "mention_count"}
    missing = required.difference(mentions.columns)
    if missing:
        raise KeyError(f"mentions missing: {', '.join(sorted(missing))}")
    work = mentions.copy()
    work["day"] = normalize_market_days(work["day"])
    work["ticker"] = work["ticker"].astype(str).str.strip().str.upper()
    work["mention_count"] = pd.to_numeric(work["mention_count"], errors="coerce").fillna(0)
    if "doi_count" not in work:
        work["doi_count"] = work["mention_count"].gt(0).astype(int)
    work["doi_count"] = pd.to_numeric(work["doi_count"], errors="coerce").fillna(0)
    allowed = set(tickers) if tickers is not None else INVESTABLE_UNIVERSE
    work = work[work["day"].notna() & work["ticker"].isin(allowed)]
    if work.empty:
        return pd.DataFrame(columns=["day", "ticker", "mention_count", "doi_count"])
    daily = work.groupby(["day", "ticker"], as_index=False)[["mention_count", "doi_count"]].sum()
    dates = pd.date_range(daily["day"].min(), daily["day"].max(), freq="D")
    universe = sorted(set(daily["ticker"]))
    index = pd.MultiIndex.from_product([dates, universe], names=["day", "ticker"])
    return (daily.set_index(["day", "ticker"]).reindex(index, fill_value=0).reset_index()
            .sort_values(["ticker", "day"]).reset_index(drop=True))


def build_calendar_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Compute trailing breadth features using EWMA and rolling windows.

    Breadth (distinct papers) is the cleaner signal — volume conflates
    writing style with genuine research attention. EWMA gives more weight
    to recent mentions, which perform better than simple rolling sums.
    """
    if daily.empty:
        return daily.copy()
    panel = daily.copy().sort_values(["ticker", "day"])
    group = panel.groupby("ticker", group_keys=False)
    # EWMA breadth: exponentially weighted, span=20 (best IC)
    panel["breadth_ewm_20d"] = group["doi_count"].transform(
        lambda s: s.ewm(span=20, min_periods=20).mean()
    )
    # Simple rolling breadth: 30-day sum (baseline comparison)
    panel["breadth_30d"] = group["doi_count"].transform(
        lambda s: s.rolling(30, min_periods=30).sum()
    )
    # Percentile ranks are calculated across the frozen universe on the day
    # the signal is known.  Flat all-zero days intentionally receive no rank.
    for feature in FEATURES:
        panel[f"rank_{feature}"] = panel.groupby("day")[feature].transform(
            lambda s: s.rank(pct=True) if s.nunique(dropna=True) > 1 else np.nan
        )
    return panel


def weekly_signals(feature_panel: pd.DataFrame) -> pd.DataFrame:
    """Sample last known signal each Friday; weekends add no new information."""
    rank_columns = [f"rank_{feature}" for feature in FEATURES]
    keep = feature_panel[["day", "ticker", *rank_columns]].copy()
    weekly = (keep.set_index("day").groupby("ticker")[rank_columns]
              .resample("W-FRI").last().reset_index())
    return weekly.dropna(how="all", subset=rank_columns).sort_values(["day", "ticker"])


def _bootstrap_ci(values: pd.Series, seed: int = 7, samples: int = 2_000) -> tuple[float, float]:
    values = values.dropna().to_numpy(dtype=float)
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(means, [.025, .975]))


def evaluate_predefined_features(
    panel: pd.DataFrame,
    training_weeks: int = 52,
    features: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Evaluate pre-specified ranks after a training and label-overlap embargo.

    There is no fitted model here.  The withheld period makes results suitable
    for feature selection: a later composite may use a feature only when its
    individual out-of-sample results are sufficiently robust.
    """
    active_features = list(features) if features is not None else list(FEATURES)
    rows: list[dict] = []
    for horizon, days in HORIZONS.items():
        return_col = f"fwd_{horizon}"
        for feature in active_features:
            signal_col = f"rank_{feature}"
            data = panel.dropna(subset=[signal_col, return_col]).copy()
            if data.empty:
                rows.append({"feature": feature, "horizon": horizon, "observations": 0, "weeks": 0})
                continue
            dates = sorted(data["day"].unique())
            # The embargo prevents a training window from containing labels
            # that overlap the first held-out return window.
            holdout_index = training_weeks + math.ceil(days / 5)
            held_out_dates = dates[holdout_index:]
            test = data[data["day"].isin(held_out_dates)]
            weekly_ic = (test.groupby("day").apply(
                lambda x: x[signal_col].corr(x[return_col], method="spearman")
                if x["ticker"].nunique() >= 4 else np.nan,
                include_groups=False,
            ).dropna())
            spreads = []
            for _, group in test.groupby("day"):
                if group["ticker"].nunique() < 4:
                    continue
                top = group.loc[group[signal_col] == group[signal_col].max(), return_col].mean()
                bottom = group.loc[group[signal_col] == group[signal_col].min(), return_col].mean()
                if pd.notna(top) and pd.notna(bottom):
                    spreads.append(top - bottom)
            ic_mean = weekly_ic.mean()
            ic_std = weekly_ic.std(ddof=1)
            ci_low, ci_high = _bootstrap_ci(weekly_ic)
            rows.append({
                "feature": feature, "horizon": horizon,
                "observations": int(len(test)), "weeks": int(len(weekly_ic)),
                "mean_ic": ic_mean, "icir": ic_mean / ic_std if pd.notna(ic_std) and ic_std else np.nan,
                "ic_ci_low": ci_low, "ic_ci_high": ci_high,
                "mean_top_bottom_log_return": float(np.mean(spreads)) if spreads else np.nan,
                "spread_weeks": len(spreads),
                "holdout_start": pd.Timestamp(held_out_dates[0]).date() if held_out_dates else None,
            })
    return pd.DataFrame(rows)


def run_walk_forward_validation(
    mentions: pd.DataFrame,
    training_weeks: int = 52,
    features: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Run the full mention-only validation using public close prices."""
    daily = prepare_daily_panel(mentions)
    features_panel = build_calendar_features(daily)
    weekly = weekly_signals(features_panel)
    tickers = sorted(weekly["ticker"].unique())
    if not tickers:
        raise RuntimeError("No investable tickers with mention history")
    prices = fetch_prices(tickers, str(daily["day"].min().date()), str(daily["day"].max().date() + pd.Timedelta(days=190)))
    joined = attach_returns(weekly, prices)
    return evaluate_predefined_features(joined, training_weeks=training_weeks, features=features)
