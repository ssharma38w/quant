"""
"Live picks today" — current-fundamentals stock ideas by Lynch category.

CAVEAT: this mode uses CURRENT yfinance fundamentals snapshot (PEG ratio,
debt/equity, earnings growth, etc.), which is NOT point-in-time historical
data. It was NOT used in the historical backtest (would be lookahead bias).
Use this only for "what looks interesting today" idea generation.
"""
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

from .universe import UNIVERSE, tickers, sector_map, universe_df
from .fetcher import load_universe_prices, fetch_nifty_benchmark
from .factors import compute_factor_snapshot

# Per-category fundamental sanity filters (current snapshot only)
CATEGORY_FILTERS = {
    "fast_grower": lambda f: (f.get("pegRatio") or 99) < 1.5 and (f.get("earningsQuarterlyGrowth") or 0) > 0.15,
    "stalwart": lambda f: (f.get("debtToEquity") or 999) < 100 and (f.get("marketCap") or 0) > 5e11,
    "turnaround": lambda f: True,  # reversal already captured by price factors
    "cyclical": lambda f: True,
}

FUNDAMENTAL_FIELDS = ["trailingPE", "pegRatio", "debtToEquity", "earningsQuarterlyGrowth", "revenueGrowth", "marketCap"]


def _get_fundamentals(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return {k: info.get(k) for k in FUNDAMENTAL_FIELDS}
    except Exception:
        return {k: None for k in FUNDAMENTAL_FIELDS}


def generate_live_picks(top_n_per_category: int = 5) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("  LIVE PICKS — TODAY'S LYNCH-STYLE IDEAS")
    print("=" * 60)
    print("  CAVEAT: uses CURRENT fundamentals (PEG, debt/equity, earnings")
    print("  growth) — NOT point-in-time historical data, and NOT used in")
    print("  the historical backtest. For idea generation only.")
    print("=" * 60)

    end = datetime.today().strftime("%Y-%m-%d")
    start = "2020-01-01"  # ~5yr lookback, more than enough for 252d factors

    prices = load_universe_prices(start, end)
    nifty = fetch_nifty_benchmark(start, end)
    udf = universe_df()

    snap = compute_factor_snapshot(
        prices, nifty, as_of=pd.Timestamp(nifty.index[-1]),
        sector_map=sector_map(), market_cap_tier=dict(zip(udf["ticker"], udf["market_cap_tier"])),
    )
    if snap.empty:
        print("  No eligible stocks (insufficient history).")
        return pd.DataFrame()

    print(f"\n  Fetching current fundamentals for {len(snap)} eligible stocks...")
    fundamentals = {t: _get_fundamentals(t) for t in snap.index}
    for field in FUNDAMENTAL_FIELDS:
        snap[field] = snap.index.map(lambda t: fundamentals[t].get(field))

    results = []
    for cat, filter_fn in CATEGORY_FILTERS.items():
        cat_df = snap[snap["assigned_category"] == cat].sort_values(f"score_{cat}", ascending=False)
        picks = []
        for ticker, row in cat_df.iterrows():
            if filter_fn(fundamentals[ticker]):
                picks.append(ticker)
            if len(picks) >= top_n_per_category:
                break
        for ticker in picks:
            row = snap.loc[ticker]
            results.append({
                "category": cat,
                "ticker": ticker,
                "sector": row.get("sector", "Unknown"),
                "score": round(row[f"score_{cat}"], 3),
                "momentum_12_1": round(row["momentum_12_1"], 3) if pd.notna(row["momentum_12_1"]) else None,
                "drawdown_52w": round(row["drawdown_52w"], 3) if pd.notna(row["drawdown_52w"]) else None,
                "pegRatio": fundamentals[ticker].get("pegRatio"),
                "trailingPE": fundamentals[ticker].get("trailingPE"),
                "debtToEquity": fundamentals[ticker].get("debtToEquity"),
                "earningsQuarterlyGrowth": fundamentals[ticker].get("earningsQuarterlyGrowth"),
            })

    out = pd.DataFrame(results)
    if out.empty:
        print("  No picks passed the fundamental filters.")
        return out

    from tabulate import tabulate
    for cat in CATEGORY_FILTERS:
        cat_out = out[out["category"] == cat]
        if cat_out.empty:
            continue
        print(f"\n  {cat.upper().replace('_', ' ')}:")
        print(tabulate(cat_out.drop(columns=["category"]), headers="keys", tablefmt="rounded_outline", showindex=False))

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"live_picks_{datetime.today().strftime('%Y%m%d')}.csv"
    out.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")
    print("=" * 60 + "\n")
    return out
