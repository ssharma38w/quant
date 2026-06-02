import pandas as pd
import numpy as np


TRADING_WEEKS_PER_YEAR = 52


def expected_weekly_move(spot: float, vix: float) -> float:
    """1 SD expected weekly move from India VIX."""
    return (vix / 100) / np.sqrt(TRADING_WEEKS_PER_YEAR) * spot


def historical_volatility(closes: pd.Series, window: int = 20) -> pd.Series:
    """Annualized historical volatility from log returns."""
    log_ret = np.log(closes / closes.shift(1))
    hv = log_ret.rolling(window).std() * np.sqrt(252)
    return hv


def iv_hv_premium(vix: pd.Series, hv: pd.Series) -> pd.Series:
    """Ratio of implied vol (VIX/100) to realized HV. >1 = IV rich = good to sell."""
    return (vix / 100) / hv


def weekly_stats(df: pd.DataFrame, hv_window: int = 20) -> pd.DataFrame:
    """
    Enrich daily dataframe with vol features.
    Expects columns: Close, VIX
    """
    df = df.copy()
    df["HV20"] = historical_volatility(df["Close"], hv_window)
    df["HV10"] = historical_volatility(df["Close"], 10)
    df["IV"] = df["VIX"] / 100
    df["IV_HV_ratio"] = iv_hv_premium(df["VIX"], df["HV20"])
    df["weekly_move_1sd"] = expected_weekly_move(df["Close"], df["VIX"])
    df["daily_return"] = df["Close"].pct_change()
    df["vix_change_5d"] = df["VIX"].pct_change(5)
    df["nifty_return_5d"] = df["Close"].pct_change(5)
    return df.dropna()
