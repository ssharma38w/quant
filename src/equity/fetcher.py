"""
Multi-stock price fetcher with local CSV caching.

Downloads OHLCV history for the equity universe via yfinance, batching
requests to avoid rate limits, and caches each ticker to
data/equity_cache/<ticker>.csv so repeated runs are fast.
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime

import pandas as pd
import yfinance as yf

from ..data.fetcher import fetch_nifty
from .universe import UNIVERSE, tickers as universe_tickers

CACHE_DIR = Path("data/equity_cache")
BATCH_SIZE = 25


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.replace('.', '_')}.csv"


def _load_cached(ticker: str) -> pd.DataFrame | None:
    path = _cache_path(ticker)
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df if not df.empty else None


def _save_cache(ticker: str, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(_cache_path(ticker), index_label="Date")


def _covers_range(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    return df.index.min() <= start and df.index.max() >= end - pd.Timedelta(days=5)


def _reshape_multi(raw: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Reshape yf.download(group_by='ticker') MultiIndex output into per-ticker frames."""
    out = {}
    for t in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                sub = raw[t]
            else:
                sub = raw  # single ticker, flat columns
            sub = sub[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if not sub.empty:
                out[t] = sub
        except (KeyError, IndexError):
            continue
    return out


def fetch_multiple_stocks(
    tickers: list[str],
    start: str,
    end: str = None,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Fetch daily OHLCV for a list of NSE tickers (e.g. 'RELIANCE.NS').
    Returns {ticker: DataFrame[Open,High,Low,Close,Volume]}.
    Caches per-ticker to data/equity_cache/<ticker>.csv.
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    result: dict[str, pd.DataFrame] = {}
    to_download: list[str] = []

    if use_cache:
        for t in tickers:
            cached = _load_cached(t)
            if cached is not None and _covers_range(cached, start_ts, end_ts):
                result[t] = cached.loc[(cached.index >= start_ts) & (cached.index <= end_ts)]
            else:
                to_download.append(t)
    else:
        to_download = list(tickers)

    for i in range(0, len(to_download), BATCH_SIZE):
        batch = to_download[i:i + BATCH_SIZE]
        try:
            raw = yf.download(batch, start=start, end=end, auto_adjust=True,
                               group_by="ticker", progress=False, threads=True)
        except Exception as e:
            print(f"  [equity fetcher] batch download failed: {e}")
            continue

        reshaped = _reshape_multi(raw, batch)
        for t in batch:
            df = reshaped.get(t)
            if df is None or df.empty:
                print(f"  [equity fetcher] skipping {t} (no data — delisted/renamed?)")
                continue
            _save_cache(t, df)
            result[t] = df

    return result


def load_universe_prices(
    start: str,
    end: str = None,
    use_cache: bool = True,
    universe: list = None,
) -> dict[str, pd.DataFrame]:
    """Fetch prices for the full (or given) equity universe."""
    tlist = universe_tickers(universe)
    print(f"\nLoading price history for {len(tlist)} stocks ({start} → {end or 'today'})...")
    prices = fetch_multiple_stocks(tlist, start, end, use_cache=use_cache)
    print(f"  Loaded {len(prices)}/{len(tlist)} tickers")
    return prices


def fetch_nifty_benchmark(start: str, end: str = None) -> pd.DataFrame:
    """Nifty 50 index series, used as the buy-and-hold benchmark."""
    return fetch_nifty(start, end)
