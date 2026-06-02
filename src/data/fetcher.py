import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime


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
