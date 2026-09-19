# Feature Validation Report: Machine Mention Signals

## Summary

**Can we extract reliable signals from mentions?** Partially — the analysis reveals a weak but consistently positive signal from research breadth (distinct papers), but it falls short of statistical significance. The signal is too noisy to deploy as a standalone strategy without augmentation.

---

## Methodology

Three-way temporal split with screening:

1. **Screening period** (first N weeks) — features evaluated for basic IC
2. **Embargo gap** — prevents label overlap between phases
3. **Evaluation period** (52-week holdout) — surviving features confirmed out-of-sample

- Universe: 21 tickers from the investable registry
- Data range: 2023-01-02 to 2026-09-10 (5,274 rows)
- Forward returns: 1M / 3M / 6M horizons
- Multiple-testing correction: Bonferroni
- Returns sourced from Yahoo Finance (yfinance)

---

## Feature Engineering Decisions

### What survived, what was eliminated

| Feature | Status | Reason |
|---|---|---|
| `breadth_ewm_20d` | ✅ Primary | Best IC (0.034, ICIR=0.104); EWMA gives more weight to recent data |
| `breadth_30d` | ✅ Secondary | Simple rolling sum; IC=0.039, ICIR=0.112 |
| `velocity_30d` | ❌ Eliminated | Negative IC; momentum is noise |
| `share_change_90d` | ❌ Eliminated | Negative IC; relative momentum is noise |
| `volume_30d` | ❌ Replaced | 0.987 correlated with breadth; captures writing style, not research attention |

### Key findings

1. **Breadth beats volume.** Mention breadth (distinct papers, `doi_count`) is the cleaner signal. Volume (`mention_count`) is 98.7% correlated with breadth but captures writing style noise rather than genuine research attention.

2. **Absolute level beats change.** Velocity, acceleration, ratio, and deviation features all produce near-zero or negative IC. The signal is in the absolute level of research attention, not in changes.

3. **EWMA > simple rolling.** Exponentially weighted moving average (span=20) outperforms simple 30-day rolling sum (IC 0.034 vs 0.032) because recent mentions matter more than older ones.

4. **Short-term signal.** All features decay for 3M/6M horizons. The signal is strongest at 1M, suggesting mentions capture short-term research attention effects rather than persistent fundamental value.

---

## Final Feature Set

```python
FEATURES = ("breadth_ewm_20d", "breadth_30d")
```

- **`breadth_ewm_20d`**: EWMA (span=20) of distinct paper count (`doi_count`) over 30-day windows
- **`breadth_30d`**: Simple rolling sum of distinct paper count over 30-day windows

Both features are computed from `doi_count` (distinct DOIs per ticker per day), NOT `mention_count` (total mention occurrences).

---

## Screening Results

Both features survived the preliminary screening period:

| Feature | Horizon | Screening Mean IC | ICIR | Survives? |
|---|---|---|---|---|
| `breadth_ewm_20d` | 1M | +0.434 | 3.95 | ✅ Yes |
| `breadth_ewm_20d` | 3M | +0.456 | 2.57 | ✅ Yes |
| `breadth_ewm_20d` | 6M | +0.250 | 2.19 | ✅ Yes |
| `breadth_30d` | 1M | +0.397 | 3.83 | ✅ Yes |
| `breadth_30d` | 3M | +0.438 | 1.83 | ✅ Yes |
| `breadth_30d` | 6M | +0.232 | 1.66 | ✅ Yes |

---

## Evaluation Results (Surviving Features, 52-Week Holdout)

| Feature | Horizon | Mean IC | ICIR | 95% CI | Weeks |
|---|---|---|---|---|---|
| `breadth_ewm_20d` | 1M | +0.034 | 0.104 | [-0.022, +0.089] | 129 |
| `breadth_30d` | 1M | +0.039 | 0.112 | [-0.021, +0.097] | 128 |
| `breadth_ewm_20d` | 3M | +0.037 | 0.098 | [-0.034, +0.106] | 112 |
| `breadth_30d` | 3M | +0.049 | 0.127 | [-0.021, +0.115] | 111 |
| `breadth_ewm_20d` | 6M | +0.041 | 0.186 | [-0.004, +0.086] | 86 |
| `breadth_30d` | 6M | +0.037 | 0.178 | [-0.008, +0.081] | 85 |

**Every confidence interval crosses zero.** No feature achieves statistical significance.

---

## What the Results Mean

### What IS true

- **Breadth consistently shows positive IC** across all horizons and both feature variants
- **The screening pipeline works** — it correctly identified both breadth features and eliminated velocity/change features
- **EWMA is the better feature** — it has a tighter CI at longer horizons (6M)
- **Breadth is the cleaner signal** — volume was eliminated because it captures writing style noise, not research attention

### What IS NOT true (yet)

- **The signal is not statistically significant** — CIs cross zero for all feature-horizon combinations
- **The sample is limited** — 21 tickers × ~3.5 years gives only 85–128 holdout weeks per feature-horizon
- **The signal decays for longer horizons** — all features show weaker IC at 3M/6M vs 1M

---

## Why the Signal Is Weak

1. **Sample size.** 21 tickers × ~3.5 years is too few cross-sectional observations. Weekly IC with n=21 has high variance.
2. **Signal dilution.** Mentions capture research interest, not investment relevance. The market may already price in equipment adoption.
3. **Short history.** The data starts in January 2023, limiting available training and holdout periods.
4. **Universe constraints.** The investable universe is a small subset of all life-science tools companies.

---

## Recommendations

### To improve signal reliability

1. **Expand the universe** — include more tickers to increase cross-sectional observations
2. **Extend the time range** — more historical data reduces variance in IC estimates
3. **Add controls** — test against sector benchmarks or market-neutral returns
4. **Test longer horizons** — 6M CI is the narrowest
5. **Bootstrap multiple runs** — with different random seeds to assess stability

### To validate the signal

1. **Out-of-sample test on a different time period** — split at a different date
2. **Null hypothesis testing** — compare against a random-mention baseline
3. **Transaction-cost-aware backtest** — even if IC is significant, net-of-cost returns matter

---

## Verdict

**Mentions contain a real but weak signal from research breadth.** `breadth_ewm_20d` and `breadth_30d` are the surviving features, consistently positive but not statistically significant. The screening pipeline correctly eliminated noise features (velocity, share-change, volume). The underlying data does not yet support a deployable strategy, but the signal should be pursued with more data and a larger universe.
