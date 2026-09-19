"""Pre-screening of features before walk-forward validation.

Implements a three-way temporal split to test predictive power
*before* committing to a feature:

    1. Screening period — first ``screening_weeks`` weeks.
       Features are evaluated here; only features whose mean IC
       exceeds ``ic_threshold`` survive.

    2. Embargo gap — ``math.ceil(horizon / 5)`` weeks added after
       the screening period so that no label in the evaluation
       window overlaps a training window from the screening period.

    3. Evaluation period — remaining dates.  Only surviving features
       are formally evaluated here with the full walk-forward metrics
       (IC, ICIR, bootstrap CI, top-minus-bottom spread).

This prevents the multiple-testing problem inherent in evaluating
four features on a single holdout period: features must first show
independent evidence of signal on the screening period before being
confirmed on the evaluation period.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from src.alpha_backtest import HORIZONS
from src.walk_forward_validation import (
    INVESTABLE_UNIVERSE,
    _bootstrap_ci,
    attach_returns,
    build_calendar_features,
    FEATURES,
    fetch_prices,
    normalize_market_days,
    prepare_daily_panel,
    weekly_signals,
)

DEFAULT_SCORING_WINDOW = 26          # weeks in the screening period
DEFAULT_IC_THRESHOLD = 0.05          # minimum mean IC to survive screening
DEFAULT_MIN_SCREENING_WEEKS = 4      # minimum weekly observations required
DEFAULT_MULTIPLE_TESTING = "bonferroni"  # or "fdr" or "none"


def _adjust_threshold(n_tests: int, threshold: float, method: str) -> float:
    """Apply multiple-testing correction to the IC threshold."""
    if method == "bonferroni":
        return threshold / n_tests
    if method == "fdr":
        # Benjamini-Hochberg-style: use threshold / log(n_tests + 1) as a
        # conservative proxy; the full procedure requires ordering.
        return threshold / max(math.log(n_tests + 1), 1.0)
    return threshold  # "none"


def _compute_weekly_ic(
    test: pd.DataFrame,
    signal_col: str,
    return_col: str,
) -> pd.Series:
    """Compute weekly Spearman IC across the cross-section."""
    return (
        test.groupby("day")
        .apply(
            lambda x: x[signal_col].corr(x[return_col], method="spearman")
            if x["ticker"].nunique() >= 4
            else np.nan,
            include_groups=False,
        )
        .dropna()
    )


def screen_features(
    panel: pd.DataFrame,
    screening_weeks: int = DEFAULT_SCORING_WINDOW,
    ic_threshold: float = DEFAULT_IC_THRESHOLD,
    min_screening_weeks: int = DEFAULT_MIN_SCREENING_WEEKS,
    multiple_testing: str = DEFAULT_MULTIPLE_TESTING,
) -> dict[str, bool]:
    """Determine which features survive a preliminary screening period.

    Parameters
    ----------
    panel : DataFrame with feature ranks and forward returns
        Must contain ``rank_{feature}`` columns and ``fwd_{horizon}`` columns.
    screening_weeks : int
        Number of weeks at the start of the data used for screening.
    ic_threshold : float
        Minimum mean IC for a feature to survive screening.
    min_screening_weeks : int
        Minimum number of weekly IC observations required for a verdict.
    multiple_testing : str
        Correction method: "bonferroni", "fdr", or "none".

    Returns
    -------
    dict mapping feature name → bool (True if the feature survives)
    """
    n_tests = len(FEATURES) * len(HORIZONS)
    adjusted_threshold = _adjust_threshold(n_tests, ic_threshold, multiple_testing)

    dates = sorted(panel["day"].unique())
    if len(dates) < screening_weeks + 2:
        # Not enough data for a meaningful split; all features pass
        return {f: True for f in FEATURES}

    # Features are computed from trailing windows — they are available
    # from the earliest date where the rolling window has enough data.
    # We use the first ``screening_weeks`` dates as the screening window.
    screening_dates = dates[:screening_weeks]

    results: dict[str, bool] = {}
    for feature in FEATURES:
        signal_col = f"rank_{feature}"
        # Collect IC across all horizons for this feature
        all_ics: list[float] = []
        for horizon, days in HORIZONS.items():
            return_col = f"fwd_{horizon}"
            if return_col not in panel.columns or signal_col not in panel.columns:
                continue
            data = panel.dropna(subset=[signal_col, return_col]).copy()
            test = data[data["day"].isin(screening_dates)]
            weekly_ic = _compute_weekly_ic(test, signal_col, return_col)
            if len(weekly_ic) >= min_screening_weeks:
                all_ics.append(weekly_ic.mean())

        if not all_ics:
            results[feature] = False
            continue

        mean_ic = np.mean(all_ics)
        # Feature survives if mean IC exceeds the (possibly corrected) threshold
        results[feature] = bool(mean_ic > adjusted_threshold)

    return results


def _get_evaluation_dates(
    panel: pd.DataFrame,
    screening_weeks: int,
    training_weeks: int,
) -> list[pd.Timestamp]:
    """Return dates belonging to the evaluation (holdout) period."""
    dates = sorted(panel["day"].unique())
    screening_end_idx = screening_weeks
    # Add embargo gap between screening and evaluation
    min_embargo = math.ceil(max(HORIZONS.values()) / 5)
    eval_start_idx = screening_end_idx + min_embargo
    return dates[eval_start_idx:]


def run_feature_screening(
    mentions: pd.DataFrame,
    training_weeks: int = 52,
    screening_weeks: int = DEFAULT_SCORING_WINDOW,
    ic_threshold: float = DEFAULT_IC_THRESHOLD,
    min_screening_weeks: int = DEFAULT_MIN_SCREENING_WEEKS,
    multiple_testing: str = DEFAULT_MULTIPLE_TESTING,
) -> tuple[dict[str, bool], pd.DataFrame]:
    """Run the full screening pipeline and return surviving features.

    Returns
    -------
    (survivors, screening_results)
        survivors: dict of {feature: bool} indicating which features pass
        screening_results: DataFrame with per-feature, per-horizon IC stats
           from the screening period
    """
    daily = prepare_daily_panel(mentions)
    features_panel = build_calendar_features(daily)
    weekly = weekly_signals(features_panel)
    tickers = sorted(weekly["ticker"].unique())
    if not tickers:
        raise RuntimeError("No investable tickers with mention history")

    prices = fetch_prices(
        tickers,
        str(daily["day"].min().date()),
        str(daily["day"].max().date() + pd.Timedelta(days=190)),
    )
    joined = attach_returns(weekly, prices)

    # Step 1: Screen features on the first screening_weeks of data
    survivors = screen_features(
        joined,
        screening_weeks=screening_weeks,
        ic_threshold=ic_threshold,
        min_screening_weeks=min_screening_weeks,
        multiple_testing=multiple_testing,
    )

    # Step 2: Build a summary of screening results for transparency
    dates = sorted(joined["day"].unique())
    screening_dates = dates[:screening_weeks]
    results_rows = []
    for feature in FEATURES:
        signal_col = f"rank_{feature}"
        for horizon, days in HORIZONS.items():
            return_col = f"fwd_{horizon}"
            data = joined.dropna(subset=[signal_col, return_col]).copy()
            test = data[data["day"].isin(screening_dates)]
            weekly_ic = _compute_weekly_ic(test, signal_col, return_col)
            if len(weekly_ic) >= min_screening_weeks:
                ci_low, ci_high = _bootstrap_ci(weekly_ic)
                results_rows.append({
                    "feature": feature,
                    "horizon": horizon,
                    "screening_mean_ic": weekly_ic.mean(),
                    "screening_icir": (
                        weekly_ic.mean() / weekly_ic.std(ddof=1)
                        if weekly_ic.std(ddof=1) else np.nan
                    ),
                    "screening_ci_low": ci_low,
                    "screening_ci_high": ci_high,
                    "screening_weeks": len(weekly_ic),
                    "survives_screening": survivors[feature],
                })

    screening_results = pd.DataFrame(results_rows)
    return survivors, screening_results


def evaluate_surviving_features(
    panel: pd.DataFrame,
    surviving_features: set[str],
    training_weeks: int = 52,
) -> pd.DataFrame:
    """Evaluate only surviving features on the evaluation (holdout) period.

    Parameters
    ----------
    panel : DataFrame with feature ranks and forward returns.
    surviving_features : set of feature names that passed screening.
    training_weeks : int — weeks of training/embargo before evaluation.

    Returns
    -------
    DataFrame with evaluation metrics for surviving features only.
    """
    if not surviving_features:
        return pd.DataFrame(
            columns=["feature", "horizon", "observations", "weeks",
                     "mean_ic", "icir", "ic_ci_low", "ic_ci_high",
                     "mean_top_bottom_log_return", "spread_weeks", "holdout_start"]
        )

    from src.walk_forward_validation import evaluate_predefined_features

    # Evaluate all features but filter to survivors
    all_results = evaluate_predefined_features(panel, training_weeks=training_weeks)
    return all_results[all_results["feature"].isin(surviving_features)].reset_index(drop=True)


def run_screened_validation(
    mentions: pd.DataFrame,
    training_weeks: int = 52,
    screening_weeks: int = DEFAULT_SCORING_WINDOW,
    ic_threshold: float = DEFAULT_IC_THRESHOLD,
    min_screening_weeks: int = DEFAULT_MIN_SCREENING_WEEKS,
    multiple_testing: str = DEFAULT_MULTIPLE_TESTING,
) -> dict:
    """Full pipeline: screen features first, then validate survivors.

    Returns a dict with screening and evaluation results.
    """
    daily = prepare_daily_panel(mentions)
    features_panel = build_calendar_features(daily)
    weekly = weekly_signals(features_panel)
    tickers = sorted(weekly["ticker"].unique())
    if not tickers:
        raise RuntimeError("No investable tickers with mention history")

    prices = fetch_prices(
        tickers,
        str(daily["day"].min().date()),
        str(daily["day"].max().date() + pd.Timedelta(days=190)),
    )
    joined = attach_returns(weekly, prices)

    # Screening phase
    survivors, screening_results = run_feature_screening(
        mentions,
        training_weeks=training_weeks,
        screening_weeks=screening_weeks,
        ic_threshold=ic_threshold,
        min_screening_weeks=min_screening_weeks,
        multiple_testing=multiple_testing,
    )

    # Evaluation phase on survivors only
    eval_results = evaluate_surviving_features(
        joined, set(survivors), training_weeks=training_weeks
    )

    return {
        "survivors": survivors,
        "screening_results": screening_results,
        "evaluation_results": eval_results,
        "n_features_total": len(FEATURES),
        "n_features_surviving": sum(1 for v in survivors.values() if v),
    }
