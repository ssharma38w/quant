"""
NSE F&O Bhavcopy downloader and parser.
Downloads actual daily options chain data from NSE archives.
Replaces Black-Scholes synthetic pricing with real market prices.

NSE bhavcopy URL:
  https://archives.nseindia.com/content/historical/DERIVATIVES/{YEAR}/{MON}/fo{DDMONYYYY}bhav.csv.zip
"""
import os
import io
import zipfile
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


CACHE_DIR = Path("data/nse_cache")
NSE_BHAVCOPY_URL = (
    "https://archives.nseindia.com/content/historical/DERIVATIVES"
    "/{year}/{mon}/fo{date}bhav.csv.zip"
)
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

MONTH_MAP = {
    1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
    7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC"
}


def _cache_path(date: datetime) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"nifty_options_{date.strftime('%Y%m%d')}.parquet"


def _bhavcopy_url(date: datetime) -> str:
    return NSE_BHAVCOPY_URL.format(
        year=date.year,
        mon=MONTH_MAP[date.month],
        date=date.strftime("%d") + MONTH_MAP[date.month] + str(date.year),
    )


def download_bhavcopy(date: datetime, retries: int = 3) -> pd.DataFrame | None:
    """Download and parse NSE F&O bhavcopy for a given date. Returns Nifty options only."""
    cache = _cache_path(date)
    if cache.exists():
        return pd.read_parquet(cache)

    url = _bhavcopy_url(date)
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 404:
                return None  # Holiday / no data
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                fname = z.namelist()[0]
                df = pd.read_csv(z.open(fname))
            break
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [bhavcopy] Failed {date.date()}: {e}")
                return None
            continue

    # Filter Nifty index options
    df.columns = df.columns.str.strip()
    mask = (df["INSTRUMENT"].str.strip() == "OPTIDX") & (df["SYMBOL"].str.strip() == "NIFTY")
    df = df[mask].copy()
    if df.empty:
        return None

    df["EXPIRY_DT"] = pd.to_datetime(df["EXPIRY_DT"], format="%d-%b-%Y", errors="coerce")
    df["STRIKE_PR"] = pd.to_numeric(df["STRIKE_PR"], errors="coerce")
    df["CLOSE"] = pd.to_numeric(df["CLOSE"], errors="coerce")
    df["SETTLE_PR"] = pd.to_numeric(df["SETTLE_PR"], errors="coerce")
    df["OPEN_INT"] = pd.to_numeric(df["OPEN_INT"], errors="coerce")
    df["OPTION_TYP"] = df["OPTION_TYP"].str.strip().str.upper()
    df["date"] = date

    cols = ["date", "EXPIRY_DT", "STRIKE_PR", "OPTION_TYP", "CLOSE", "SETTLE_PR", "OPEN_INT"]
    df = df[cols].dropna(subset=["EXPIRY_DT", "STRIKE_PR"])
    df.to_parquet(cache, index=False)
    return df


def get_weekly_expiry(date: datetime) -> datetime:
    """Get the nearest upcoming Thursday (Nifty weekly expiry)."""
    days_ahead = (3 - date.weekday()) % 7  # 3 = Thursday
    if days_ahead == 0:
        days_ahead = 7
    return date + timedelta(days=days_ahead)


def lookup_option_price(
    bhavcopy: pd.DataFrame,
    expiry: datetime,
    strike: int,
    option_type: str,  # "CE" or "PE"
    use_settle: bool = True,
) -> float | None:
    """Find closest available option price for a given strike/expiry."""
    if bhavcopy is None or bhavcopy.empty:
        return None

    mask = (
        (bhavcopy["EXPIRY_DT"].dt.date == expiry.date())
        & (bhavcopy["OPTION_TYP"] == option_type)
    )
    chain = bhavcopy[mask].copy()
    if chain.empty:
        return None

    # Find nearest available strike
    chain["strike_diff"] = (chain["STRIKE_PR"] - strike).abs()
    row = chain.nsmallest(1, "strike_diff").iloc[0]

    price_col = "SETTLE_PR" if use_settle else "CLOSE"
    price = row[price_col]
    return float(price) if price > 0 else None


def fetch_options_range(
    start: str, end: str, verbose: bool = True
) -> dict[str, pd.DataFrame]:
    """
    Download bhavcopy for all trading days in range.
    Returns dict: {date_str → DataFrame}
    """
    dates = pd.bdate_range(start=start, end=end)
    data = {}
    for date in dates:
        dt = date.to_pydatetime()
        if verbose:
            print(f"  Fetching {dt.date()}...", end="\r")
        df = download_bhavcopy(dt)
        if df is not None:
            data[str(dt.date())] = df
    if verbose:
        print(f"\n  Fetched {len(data)} trading days of options data.")
    return data
