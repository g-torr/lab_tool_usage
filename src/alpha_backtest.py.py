"""Offline backtest: does the mention signal predict returns? (research only)

Run from repo root:  DATABASE_URL=... python -m src.alpha_backtest
Needs: pip install yfinance statsmodels
"""
import os, logging
import numpy as np
import pandas as pd
import psycopg2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HORIZONS = {"1M": 21, "3M": 63, "6M": 126}          # trading days

# Revenue exposure to this market (manual, pre-registered). 1.0 = pure play.
PURITY = {"TXG": 1.0, "AKYA": 1.0, "ILMN": 0.6, "BRKR": 0.5,
          "RVTY": 0.4, "DHR": 0.15}

def normalize_market_days(values: pd.Series) -> pd.Series:
    """Normalize mixed naive/timezone-aware timestamps to local calendar days.

    yfinance may return America/New_York timestamps while the database panel
    contains timezone-naive dates. Removing the timezone preserves the market
    calendar date and gives merge_asof identical datetime dtypes.
    """
    days = pd.to_datetime(values, errors="coerce")
    if getattr(days.dt, "tz", None) is not None:
        days = days.dt.tz_localize(None)
    # pandas 2.0+ can preserve the source unit (s/us/ms/ns). merge_asof
    # requires the left and right keys to use the exact same resolution.
    return days.dt.normalize().astype("datetime64[ns]")

def fetch_mention_panel() -> pd.DataFrame:
    query = """
        SELECT c.date AS day,
               COALESCE(NULLIF(TRIM(r.ticker), ''), 'Unresolved') AS ticker,
               COUNT(*)              AS mention_count,
               COUNT(DISTINCT c.doi) AS doi_count        -- breadth
        FROM candidates c
        JOIN machine_mentions m ON c.doi = m.doi
        LEFT JOIN registry_machines r
               ON TRIM(r.canonical_name) = TRIM(m.resolved_machine)
        WHERE c.date IS NOT NULL
        GROUP BY c.date, r.ticker
    """
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
        return pd.DataFrame.from_records(rows, columns=columns)
    finally:
        conn.close()

def rolling_z(s, window=252, min_periods=60):
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std()
    return ((s - mu) / sd.replace(0, np.nan)).clip(-3, 3)   # winsorise in z-space

def build_signal_panel(mentions: pd.DataFrame) -> pd.DataFrame:
    m = mentions.copy()
    m["day"] = normalize_market_days(m["day"])
    m = m.sort_values(["ticker", "day"])
    g = m.groupby("ticker")

    m["S30"] = g["mention_count"].transform(lambda s: s.rolling(30, min_periods=7).sum())
    m["D30"] = g["doi_count"].transform(lambda s: s.rolling(30, min_periods=7).sum())
    # velocity: log-change of 30d mentions vs 4 weeks earlier
    m["vel"] = np.log1p(m["S30"]) - np.log1p(g["S30"].transform(lambda s: s.shift(30)))
    # share of the field + its 90-day shift (controls for field growth)
    m["share"] = m["S30"] / m.groupby("day")["S30"].transform("sum").replace(0, np.nan)
    m["dshare"] = m["share"] - g["share"].transform(lambda s: s.shift(90))
    # breadth-weighted velocity (downweights single-lab spam)
    m["bvel"] = m["vel"] * np.log1p(m["D30"])

    for col in ("vel", "dshare", "bvel"):
        m[f"z_{col}"] = g[col].transform(rolling_z)
    m["z"] = m[["z_vel", "z_dshare", "z_bvel"]].mean(axis=1)
    m["z"] = m["z"] * m["ticker"].map(PURITY).fillna(0.0)   # purity weighting
    return m

def to_weekly(panel: pd.DataFrame) -> pd.DataFrame:
    w = (panel.dropna(subset=["z"]).set_index("day")
         .groupby("ticker").resample("W-FRI")[["z"]].last()
         .reset_index().dropna(subset=["z"]))
    return w.sort_values("day")

def fetch_prices(tickers, start, end) -> pd.DataFrame:
    import yfinance as yf
    frames = []
    for t in tickers:
        try:  # yfinance usually returns history even for delisted tickers
            c = yf.Ticker(t).history(start=start, end=end, auto_adjust=True)["Close"]
            c = c.dropna().rename("close").reset_index().rename(columns={"Date": "day"})
            c["day"] = normalize_market_days(c["day"])
            c["ticker"] = t
            frames.append(c)
        except Exception as e:
            logger.warning("price download failed for %s: %s", t, e)
    if not frames:
        raise RuntimeError("no price data")
    return pd.concat(frames, ignore_index=True).sort_values("day")

def attach_returns(w: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    w = w.copy()
    p = prices.copy()
    w["day"] = normalize_market_days(w["day"])
    p["day"] = normalize_market_days(p["day"])
    w = w.dropna(subset=["day", "ticker"]).sort_values(["day", "ticker"])
    p = p.dropna(subset=["day", "ticker"]).sort_values(["day", "ticker"])
    g = p.groupby("ticker")
    # enter at NEXT close after the signal is known -> no look-ahead
    for name, h in HORIZONS.items():
        p[f"fwd_{name}"] = g["close"].transform(lambda c: np.log(c.shift(-1 - h) / c.shift(-1)))
    p["past_ret_12m"] = g["close"].transform(lambda c: np.log(c / c.shift(252)))
    return pd.merge_asof(w, p.drop(columns="close"), on="day", by="ticker",
                         direction="backward", tolerance=pd.Timedelta("4D"),
                         ).sort_values(["ticker", "day"])

def test1_panel_regression(d: pd.DataFrame, horizon: str):
    import statsmodels.formula.api as smf
    data = d.dropna(subset=[f"fwd_{horizon}", "z", "past_ret_12m"])
    if len(data) < 5 or data["ticker"].nunique() < 2:
        return np.nan, np.nan, len(data)

    fit = smf.ols(f"fwd_{horizon} ~ z + past_ret_12m + C(ticker)", data=data).fit()
    rob = fit.get_robustcov_results(cov_type="cluster", groups=data["ticker"])

    # statsmodels 0.14 returns ndarrays for robust results, while some older
    # versions return labelled Series. Use the model's coefficient ordering so
    # this works with either representation.
    z_index = fit.model.exog_names.index("z")
    beta = np.asarray(rob.params)[z_index]
    p_value = np.asarray(rob.pvalues)[z_index]
    return float(beta), float(p_value), len(data)

def test2_information_coefficient(d: pd.DataFrame, horizon: str):
    data = d.dropna(subset=[f"fwd_{horizon}", "z"])
    ic = (data.groupby("day")
          .apply(lambda x: x["z"].corr(x[f"fwd_{horizon}"], method="spearman")
                 if len(x) >= 4 else np.nan).dropna())
    mean_ic, sd = ic.mean(), ic.std()
    icir = mean_ic / sd if sd else np.nan
    t = mean_ic / (sd / np.sqrt(len(ic))) if sd else np.nan
    return mean_ic, icir, t, len(ic)

def main():
    weekly = to_weekly(build_signal_panel(fetch_mention_panel()))
    tickers = [t for t in weekly["ticker"].unique() if t in PURITY]
    prices = fetch_prices(tickers, "2023-12-01", "2026-08-15")
    panel = attach_returns(weekly, prices)

    eligible = panel.dropna(subset=["z"])
    cross_section = eligible.groupby("day")["ticker"].nunique()
    print(
        f"\nIC eligibility: {eligible['ticker'].nunique()} tickers, "
        f"{int((cross_section >= 4).sum())} dates with >=4 tickers "
        f"(maximum {int(cross_section.max()) if len(cross_section) else 0})"
    )
    print(f"\n{'hor':>4} {'beta':>7} {'p':>6} {'meanIC':>7} {'ICIR':>6} {'t':>5} {'n_reg':>6} {'n_ic':>5}")
    for name in HORIZONS:
        b, p, n1 = test1_panel_regression(panel, name)
        mic, icir, t, n2 = test2_information_coefficient(panel, name)
        print(f"{name:>4} {b:7.4f} {p:6.3f} {mic:7.3f} {icir:6.2f} {t:5.2f} {n1:6d} {n2:5d}")

if __name__ == "__main__":
    main()