import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


def fetch_nifty(start: str, end: str = None) -> pd.DataFrame:
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    df = yf.download("^NSEI", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def fetch_vix(start: str, end: str = None) -> pd.Series:
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    df = yf.download("^INDIAVIX", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df["Close"].rename("VIX").dropna()


def fetch_combined(start: str = "2018-01-01", end: str = None) -> pd.DataFrame:
    nifty = fetch_nifty(start, end)
    vix = fetch_vix(start, end)
    df = nifty.join(vix, how="left")
    df["VIX"] = df["VIX"].ffill()
    df = df.dropna(subset=["Close", "VIX"])
    return df


def load_from_csv(nifty_csv: str, vix_csv: str = None) -> pd.DataFrame:
    """
    Load Nifty spot data from a local CSV file.
    Expected columns: Date, Open, High, Low, Close, Volume
    VIX CSV (optional): Date, VIX  (or a column named 'Close' / 'VIX')

    Use this when yfinance is unavailable. You can export data from:
    - NSE website (Nifty historical data)
    - TradingView CSV export
    - Any broker platform
    """
    nifty = pd.read_csv(nifty_csv, parse_dates=["Date"], index_col="Date")
    nifty.index.name = "Date"
    nifty.columns = [c.strip().title() for c in nifty.columns]
    nifty = nifty[["Open", "High", "Low", "Close", "Volume"]].dropna()

    if vix_csv:
        vix_df = pd.read_csv(vix_csv, parse_dates=["Date"], index_col="Date")
        vix_col = "VIX" if "VIX" in vix_df.columns else "Close"
        vix_series = vix_df[vix_col].rename("VIX")
        df = nifty.join(vix_series, how="left")
        df["VIX"] = df["VIX"].ffill()
    else:
        # No VIX file: estimate IV from realized vol (rough proxy)
        log_ret = np.log(nifty["Close"] / nifty["Close"].shift(1))
        hv20 = log_ret.rolling(20).std() * np.sqrt(252) * 100
        nifty["VIX"] = hv20 * 1.2  # apply typical IV premium
        df = nifty

    return df.dropna(subset=["Close", "VIX"])
