"""
Generates realistic synthetic Nifty + India VIX data for offline testing.
On a real machine use fetcher.py (yfinance) for actual historical data.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_sample_data(
    start: str = "2019-01-01",
    end: str = "2024-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate daily Nifty OHLCV + India VIX data with realistic regime shifts.
    Nifty: geometric Brownian motion with vol regime shifts
    VIX: mean-reverting process correlated with Nifty vol
    """
    np.random.seed(seed)
    dates = pd.bdate_range(start=start, end=end)  # business days only
    n = len(dates)

    # Regime params: (annual_vol, vix_mean, drift)
    regime_params = [
        (0.12, 13.0, 0.12),   # low vol bull
        (0.18, 16.0, 0.08),   # normal
        (0.30, 24.0, -0.10),  # high vol / crisis
        (0.18, 17.0, 0.10),   # normal recovery
    ]

    # Assign regimes in blocks with smooth transitions
    regime_blocks = np.array_split(np.arange(n), len(regime_params))
    annual_vol = np.zeros(n)
    vix_target = np.zeros(n)
    drift = np.zeros(n)
    for i, block in enumerate(regime_blocks):
        annual_vol[block] = regime_params[i][0]
        vix_target[block] = regime_params[i][1]
        drift[block] = regime_params[i][2]

    daily_vol = annual_vol / np.sqrt(252)
    daily_drift = drift / 252

    # Nifty price path
    returns = np.random.normal(daily_drift, daily_vol, n)
    close = np.zeros(n)
    close[0] = 11000.0  # approx Nifty Jan 2019
    for i in range(1, n):
        close[i] = close[i - 1] * np.exp(returns[i])

    # OHLC from close
    intraday_range = np.abs(np.random.normal(0, daily_vol * 0.8, n)) * close
    high = close + intraday_range * np.random.uniform(0.3, 0.7, n)
    low = close - intraday_range * np.random.uniform(0.3, 0.7, n)
    open_ = close * np.exp(np.random.normal(0, daily_vol * 0.3, n))
    volume = np.random.randint(100_000, 500_000, n)

    # VIX: mean-reverting + noise + negative correlation with Nifty
    vix = np.zeros(n)
    vix[0] = 15.0
    kappa = 0.08  # mean reversion speed
    vix_noise = np.random.normal(0, 1.2, n)
    nifty_shock = -np.where(returns < 0, returns * 200, 0)  # VIX spikes on down moves
    for i in range(1, n):
        vix[i] = (
            vix[i - 1]
            + kappa * (vix_target[i] - vix[i - 1])
            + vix_noise[i]
            + nifty_shock[i]
        )
        vix[i] = max(8.0, min(vix[i], 80.0))

    df = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
            "VIX": vix,
        },
        index=dates,
    )
    df.index.name = "Date"
    return df
