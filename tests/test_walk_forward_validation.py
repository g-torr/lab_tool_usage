import pandas as pd

from src.walk_forward_validation import build_calendar_features, prepare_daily_panel, weekly_signals


def test_zero_filled_calendar_makes_30_day_window_a_real_calendar_window():
    mentions = pd.DataFrame([
        {"day": "2025-01-01", "ticker": "AAA", "mention_count": 2, "doi_count": 1},
        {"day": "2025-01-30", "ticker": "AAA", "mention_count": 3, "doi_count": 1},
        {"day": "2025-01-15", "ticker": "BBB", "mention_count": 1, "doi_count": 1},
    ])

    daily = prepare_daily_panel(mentions, tickers=["AAA", "BBB"])
    features = build_calendar_features(daily)
    jan_30 = features[(features.ticker == "AAA") & (features.day == pd.Timestamp("2025-01-30"))].iloc[0]

    assert len(daily) == 60  # 30 calendar days x two tickers
    # Breadth EWMA and rolling sum of distinct papers (doi_count)
    assert jan_30["breadth_ewm_20d"] > 0
    assert jan_30["breadth_30d"] == 2  # 1 (Jan 1) + 1 (Jan 30) = 2 distinct papers


def test_weekly_sampling_uses_a_common_friday_schedule():
    rows = []
    for ticker, mention_day in [("AAA", "2025-01-01"), ("BBB", "2025-01-02")]:
        rows.append({"day": mention_day, "ticker": ticker, "mention_count": 1, "doi_count": 1})
        rows.append({"day": "2025-01-31", "ticker": ticker, "mention_count": 0, "doi_count": 0})
    features = build_calendar_features(prepare_daily_panel(pd.DataFrame(rows), tickers=["AAA", "BBB"]))
    weekly = weekly_signals(features)

    assert weekly["day"].dt.dayofweek.eq(4).all()
    assert set(weekly["ticker"]) == {"AAA", "BBB"}
