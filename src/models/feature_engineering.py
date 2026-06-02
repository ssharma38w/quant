"""
Feature engineering for the strategy predictor ML model.
Builds a weekly feature vector from daily market data.
"""
import numpy as np
import pandas as pd


def build_weekly_features(df: pd.DataFrame, regime_probs: np.ndarray = None) -> pd.DataFrame:
    """
    Construct features for each Monday entry from the enriched daily dataframe.
    df must have: Close, VIX, HV20, HV10, IV_HV_ratio, daily_return, vix_change_5d, nifty_return_5d

    Returns a DataFrame indexed by entry date (Mondays).
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # Rolling percentile rank (252-day window)
    df["vix_rank"] = df["VIX"].rolling(252, min_periods=60).rank(pct=True)
    df["iv_hv_rank"] = df["IV_HV_ratio"].rolling(252, min_periods=60).rank(pct=True)

    # Momentum features
    df["ret_1d"]  = df["Close"].pct_change(1)
    df["ret_5d"]  = df["Close"].pct_change(5)
    df["ret_20d"] = df["Close"].pct_change(20)
    df["ret_60d"] = df["Close"].pct_change(60)

    # VIX momentum
    df["vix_ret_1w"] = df["VIX"].pct_change(5)
    df["vix_ret_4w"] = df["VIX"].pct_change(20)

    # Realized vs implied spread
    df["vol_spread"] = df["IV"] - df["HV20"]   # positive = IV rich

    # ATM straddle value as % of spot (proxy for absolute premium richness)
    df["straddle_pct"] = (df["VIX"] / 100) / np.sqrt(52) * 2 * 0.798  # ≈ 2×σ√(T/π)

    # Day-of-month (options behave differently at month start/end)
    df["day_of_month"] = df.index.day
    df["week_of_month"] = ((df.index.day - 1) // 7) + 1

    # 52-week high/low distance
    df["pct_from_52w_high"] = df["Close"] / df["Close"].rolling(252, min_periods=60).max() - 1
    df["pct_from_52w_low"]  = df["Close"] / df["Close"].rolling(252, min_periods=60).min() - 1

    # Attach regime probabilities if provided
    if regime_probs is not None and len(regime_probs) == len(df):
        df["regime_p0"] = regime_probs[:, 0]
        df["regime_p1"] = regime_probs[:, 1]
        df["regime_p2"] = regime_probs[:, 2]
    else:
        df["regime_p0"] = 0.33
        df["regime_p1"] = 0.34
        df["regime_p2"] = 0.33

    feature_cols = [
        "VIX", "vix_rank", "HV20", "HV10", "IV_HV_ratio", "iv_hv_rank",
        "vol_spread", "straddle_pct",
        "ret_1d", "ret_5d", "ret_20d", "ret_60d",
        "vix_ret_1w", "vix_ret_4w",
        "pct_from_52w_high", "pct_from_52w_low",
        "day_of_month", "week_of_month",
        "regime_p0", "regime_p1", "regime_p2",
    ]

    feats = df[feature_cols].copy()
    # Only keep Mondays (entry days)
    feats = feats[feats.index.dayofweek == 0]
    return feats.dropna()


FEATURE_NAMES = [
    "VIX", "vix_rank", "HV20", "HV10", "IV_HV_ratio", "iv_hv_rank",
    "vol_spread", "straddle_pct",
    "ret_1d", "ret_5d", "ret_20d", "ret_60d",
    "vix_ret_1w", "vix_ret_4w",
    "pct_from_52w_high", "pct_from_52w_low",
    "day_of_month", "week_of_month",
    "regime_p0", "regime_p1", "regime_p2",
]
