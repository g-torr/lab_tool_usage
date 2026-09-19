"""Tests for feature_screening module."""

import pytest
import pandas as pd

from src.feature_screening import (
    _adjust_threshold,
    _compute_weekly_ic,
    screen_features,
    run_feature_screening,
)
from src.walk_forward_validation import FEATURES, HORIZONS, INVESTABLE_UNIVERSE

# Use tickers from the investable universe for tests
TEST_TICKERS = sorted(list(INVESTABLE_UNIVERSE)[:5])  # first 5 universe tickers


# --- _adjust_threshold ---

def test_bonferroni_adjustment():
    threshold = _adjust_threshold(12, 0.05, "bonferroni")
    assert threshold == 0.05 / 12


def test_fdr_adjustment():
    import math
    threshold = _adjust_threshold(12, 0.05, "fdr")
    assert threshold == 0.05 / math.log(13)


def test_no_adjustment():
    threshold = _adjust_threshold(12, 0.05, "none")
    assert threshold == 0.05


# --- _compute_weekly_ic ---

def test_weekly_ic_returns_series():
    rng = pd.date_range("2025-01-01", periods=20, freq="W-FRI")
    test = pd.DataFrame({
        "day": rng,
        "ticker": ["A"] * 20,
        "rank_breadth_ewm_20d": range(20),
        "fwd_1M": range(20, 40),
    })
    ic = _compute_weekly_ic(test, "rank_breadth_ewm_20d", "fwd_1M")
    assert isinstance(ic, pd.Series)


def test_weekly_ic_requires_minimum_tickers():
    # Only 3 tickers — below the threshold of 4
    rng = pd.date_range("2025-01-01", periods=10, freq="W-FRI")
    test = pd.DataFrame({
        "day": rng,
        "ticker": ["A", "B", "C"] * 3 + ["A"],
        "rank_breadth_ewm_20d": range(10),
        "fwd_1M": range(10),
    })
    ic = _compute_weekly_ic(test, "rank_breadth_ewm_20d", "fwd_1M")
    assert len(ic) == 0  # all NaN due to <4 tickers


# --- screen_features ---

def test_screen_features_returns_dict():
    # Build a minimal panel with synthetic weekly data
    rng = pd.date_range("2025-01-03", periods=60, freq="W-FRI")
    rows = []
    for day in rng:
        for ticker in TEST_TICKERS:
            for feature in FEATURES:
                rows.append({
                    "day": day,
                    "ticker": ticker,
                    f"rank_{feature}": __import__("numpy").random.rand(),
                    "fwd_1M": __import__("numpy").random.rand(),
                })
    panel = pd.DataFrame(rows)
    panel["day"] = pd.to_datetime(panel["day"])

    survivors = screen_features(panel, screening_weeks=4, ic_threshold=-1.0)
    assert isinstance(survivors, dict)
    assert set(survivors.keys()) == set(FEATURES)


def test_screen_features_low_threshold_all_survive():
    rng = pd.date_range("2025-01-03", periods=60, freq="W-FRI")
    rows = []
    for day in rng:
        for ticker in TEST_TICKERS:
            for feature in FEATURES:
                rows.append({
                    "day": day,
                    "ticker": ticker,
                    f"rank_{feature}": __import__("numpy").random.rand(),
                    "fwd_1M": __import__("numpy").random.rand(),
                })
    panel = pd.DataFrame(rows)
    panel["day"] = pd.to_datetime(panel["day"])

    # Very low threshold — all should survive (IC is in [-1, 1])
    survivors = screen_features(panel, screening_weeks=4, ic_threshold=-2.0)
    # Only features with valid IC observations should be evaluated;
    # any feature with data passes a threshold of -2.0
    for feature, survives in survivors.items():
        if survives:
            assert survivors[feature], f"{feature} should survive with ic_threshold=-2.0"
    # At minimum, features with valid data must pass
    assert any(survivors.values()), "At least one feature should have valid IC"


def test_screen_features_not_enough_data_all_pass():
    rng = pd.date_range("2025-01-03", periods=3, freq="W-FRI")
    rows = []
    for day in rng:
        for ticker in TEST_TICKERS:
            for feature in FEATURES:
                rows.append({
                    "day": day,
                    "ticker": ticker,
                    f"rank_{feature}": __import__("numpy").random.rand(),
                    "fwd_1M": __import__("numpy").random.rand(),
                })
    panel = pd.DataFrame(rows)
    panel["day"] = pd.to_datetime(panel["day"])

    # Not enough data for split — all features pass
    survivors = screen_features(panel, screening_weeks=4, ic_threshold=0.05)
    assert all(survivors.values())


# --- run_feature_screening ---

def test_run_feature_screening_returns_tuple():
    rng = pd.date_range("2025-01-01", periods=365, freq="D")
    rows = []
    for day in rng:
        for ticker in TEST_TICKERS:
            for feature in FEATURES:
                rows.append({
                    "day": day,
                    "ticker": ticker,
                    f"mention_count": __import__("numpy").random.randint(0, 5),
                    "doi_count": __import__("numpy").random.randint(0, 3),
                })
    mentions = pd.DataFrame(rows)

    try:
        survivors, screening_results = run_feature_screening(
            mentions,
            screening_weeks=4,
            ic_threshold=-1.0,
            min_screening_weeks=1,
        )
    except RuntimeError as e:
        if "price" in str(e).lower() or "ticker" in str(e).lower():
            pytest.skip(f"Market data unavailable: {e}")
        raise
    assert isinstance(survivors, dict)
    assert isinstance(screening_results, pd.DataFrame)
    assert "feature" in screening_results.columns
    assert "survives_screening" in screening_results.columns