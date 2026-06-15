"""
No-lookahead factor scoring for Lynch-style stock categories.

All factors at a given `as_of` date use only price/volume data with
index <= as_of. Cross-sectional z-scores combine into 4 category
scores; each stock is assigned to its best-fit category (argmax) for
that rebalance period.

Categories (price/volume proxies for Lynch's stock types):
  fast_grower  — strong 12-1 month momentum + steady uptrend
  stalwart     — low volatility, steady 200dma uptrend, large-cap
  turnaround   — deep drawdown from 52w high + recent positive reversal
  cyclical     — high beta to Nifty + sector rotation tailwind
"""
import numpy as np
import pandas as pd

CATEGORIES = ["fast_grower", "stalwart", "turnaround", "cyclical"]


def _zscore(s: pd.Series) -> pd.Series:
    std = s.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def _slice_to(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    return df.loc[df.index <= as_of]


def _stock_factors(close: pd.Series, volume: pd.Series, nifty_ret: pd.Series) -> dict:
    n = len(close)
    if n < 280:
        return None

    ret = close.pct_change()

    # Momentum (12-1 month): exclude most recent month
    momentum_12_1 = close.iloc[-21] / close.iloc[-252] - 1 if n >= 252 else np.nan

    # Trend steadiness (Stalwart)
    sma200 = close.rolling(200).mean()
    pct_above_sma200 = (close.tail(252) > sma200.tail(252)).mean() if n >= 252 else np.nan
    sma200_slope = sma200.iloc[-1] / sma200.iloc[-60] - 1 if n >= 260 and not np.isnan(sma200.iloc[-60]) else np.nan

    # Volatility (Stalwart, lower is better)
    vol_60d = ret.rolling(60).std().iloc[-1] * np.sqrt(252) if n >= 61 else np.nan

    # Drawdown from 52w high (Turnaround)
    high_52w = close.rolling(252).max()
    drawdown_52w = close.iloc[-1] / high_52w.iloc[-1] - 1 if n >= 252 else np.nan

    # Reversal score (Turnaround)
    if n >= 252:
        ret_3m = close.iloc[-1] / close.iloc[-63] - 1
        ret_prior_9m = close.iloc[-63] / close.iloc[-252] - 1
        reversal_score = ret_3m - ret_prior_9m
        turnaround_flag = (drawdown_52w <= -0.30) and (ret_3m > 0.05)
    else:
        ret_3m, reversal_score, turnaround_flag = np.nan, np.nan, False

    # Beta to Nifty (Cyclical), 252d rolling
    if n >= 252:
        aligned = pd.concat([ret.tail(252), nifty_ret.tail(252)], axis=1, join="inner").dropna()
        if len(aligned) > 30 and aligned.iloc[:, 1].var() > 0:
            beta = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1] / aligned.iloc[:, 1].var()
        else:
            beta = np.nan
    else:
        beta = np.nan

    # Liquidity
    avg_dollar_volume_60d = (close * volume).rolling(60).mean().iloc[-1] if n >= 60 else np.nan

    return {
        "momentum_12_1": momentum_12_1,
        "pct_above_sma200": pct_above_sma200,
        "sma200_slope": sma200_slope,
        "vol_60d": vol_60d,
        "drawdown_52w": drawdown_52w,
        "ret_3m": ret_3m,
        "reversal_score": reversal_score,
        "turnaround_flag": turnaround_flag,
        "beta": beta,
        "avg_dollar_volume_60d": avg_dollar_volume_60d,
    }


def compute_factor_snapshot(
    prices: dict[str, pd.DataFrame],
    nifty: pd.DataFrame,
    as_of: pd.Timestamp,
    sector_map: dict[str, str] = None,
    market_cap_tier: dict[str, str] = None,
    min_history_days: int = 280,
    min_avg_dollar_volume: float = 5e7,
) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by ticker with raw factors, category
    scores, and assigned_category — computed using only data with
    index <= as_of (no lookahead).
    """
    nifty_close = _slice_to(nifty, as_of)["Close"]
    nifty_ret = nifty_close.pct_change()

    rows = {}
    for ticker, df in prices.items():
        sliced = _slice_to(df, as_of)
        if len(sliced) < min_history_days:
            continue
        f = _stock_factors(sliced["Close"], sliced["Volume"], nifty_ret)
        if f is None:
            continue
        if f["avg_dollar_volume_60d"] is not None and not np.isnan(f["avg_dollar_volume_60d"]) \
                and f["avg_dollar_volume_60d"] < min_avg_dollar_volume:
            continue
        rows[ticker] = f

    if not rows:
        return pd.DataFrame()

    snap = pd.DataFrame(rows).T
    snap.index.name = "ticker"

    # Sector rotation score: sector's trailing 3m return minus Nifty's 3m return
    if sector_map and len(nifty_close) >= 64:
        nifty_3m = nifty_close.iloc[-1] / nifty_close.iloc[-63] - 1
        snap["sector"] = snap.index.map(lambda t: sector_map.get(t, "Unknown"))
        sector_ret = snap.groupby("sector")["ret_3m"].mean() - nifty_3m
        snap["sector_rs_3m"] = snap["sector"].map(sector_ret)
    else:
        snap["sector_rs_3m"] = 0.0
        if sector_map:
            snap["sector"] = snap.index.map(lambda t: sector_map.get(t, "Unknown"))

    if market_cap_tier:
        snap["market_cap_tier"] = snap.index.map(lambda t: market_cap_tier.get(t, "Mid"))
    else:
        snap["market_cap_tier"] = "Mid"

    # Category scores (cross-sectional z-scores)
    snap["score_fast_grower"] = (
        _zscore(snap["momentum_12_1"]) * 0.7 + _zscore(snap["pct_above_sma200"]) * 0.3
    )
    snap["score_stalwart"] = (
        _zscore(-snap["vol_60d"]) * 0.5
        + _zscore(snap["sma200_slope"]) * 0.3
        + (snap["market_cap_tier"] == "Large").astype(float) * 0.2
    )
    score_turn = (
        _zscore(-snap["drawdown_52w"]) * 0.4 + _zscore(snap["reversal_score"]) * 0.6
    )
    # Only candidates flagged as turnaround get a meaningful score; others penalized
    snap["score_turnaround"] = np.where(snap["turnaround_flag"], score_turn, score_turn - 5)
    snap["score_cyclical"] = (
        _zscore(snap["beta"]) * 0.5 + _zscore(snap["sector_rs_3m"]) * 0.5
    )

    score_cols = ["score_fast_grower", "score_stalwart", "score_turnaround", "score_cyclical"]
    snap[score_cols] = snap[score_cols].fillna(-999)
    snap["assigned_category"] = snap[score_cols].idxmax(axis=1).str.replace("score_", "")
    snap["composite_score"] = snap[score_cols].max(axis=1)

    return snap
