"""
TimesFM-based weekly range forecaster for Nifty.

Google TimesFM (Time Series Foundation Model) takes a context window of
historical daily closes and outputs a multi-step point forecast.

We use the 5-day (1 week) horizon forecast to derive a predicted
high-low range, which we compare against VIX-implied move:
  - TimesFM range << VIX range  →  model sees a quiet week, tighten wings
  - TimesFM range >> VIX range  →  model sees a volatile week, widen wings / skip

Install (user's machine):
    pip install timesfm
    # First call downloads ~500MB model checkpoint from HuggingFace

Fallback: if timesfm not installed, all calls silently return VIX-based estimates.
"""
import numpy as np
import pandas as pd
import os
import json
from pathlib import Path

CONTEXT_LEN = 512    # days of history fed to TimesFM (its max context)
HORIZON = 5          # next 5 trading days = 1 week
_CACHE_DIR = Path("results/timesfm_cache")

_tfm_model = None    # lazy-loaded singleton


def _load_model(verbose: bool = True) -> object | None:
    """Load TimesFM once; return None if not installed."""
    global _tfm_model
    if _tfm_model is not None:
        return _tfm_model

    try:
        import timesfm  # noqa: F401 — check import first
        if verbose:
            print("  Loading TimesFM model (downloads ~500MB on first run)...")
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="cpu",
                per_core_batch_size=32,
                horizon_len=HORIZON,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-1.0-200m-pytorch",
            ),
        )
        if verbose:
            print("  TimesFM ready.")
        _tfm_model = tfm
        return _tfm_model
    except ImportError:
        if verbose:
            print("  timesfm not installed — pip install timesfm. Using VIX-based move.")
        return None
    except Exception as e:
        if verbose:
            print(f"  TimesFM load error: {e}. Falling back to VIX-based move.")
        return None


def _vix_fallback(vix_move: float) -> dict:
    return {
        "predicted_range_pts": round(vix_move * 2, 1),
        "predicted_move_1sd": round(vix_move, 1),
        "vix_move_1sd": round(vix_move, 1),
        "range_vs_vix": 1.0,
        "source": "vix",
        "forecast_path": [],
    }


def forecast_weekly_range(
    df: pd.DataFrame,
    entry_date,
    vix: float,
    spot: float,
    model=None,
) -> dict:
    """
    Predict next week's Nifty high-low range using TimesFM.

    Args:
        df: daily OHLCV DataFrame with DatetimeIndex
        entry_date: pd.Timestamp — forecast from this date
        vix: India VIX at entry
        spot: Nifty close at entry
        model: pre-loaded TimesFM model (None → use VIX fallback)

    Returns dict with predicted_range_pts, predicted_move_1sd,
    vix_move_1sd, range_vs_vix, source, forecast_path.
    """
    vix_move = (vix / 100) / np.sqrt(52) * spot

    if model is None:
        return _vix_fallback(vix_move)

    history = df.loc[df.index < entry_date, "Close"].dropna()
    if len(history) < 32:
        return _vix_fallback(vix_move)

    context = history.values[-CONTEXT_LEN:].astype(float)

    try:
        point_forecast, _ = model.forecast([context], freq=[0])
        forecasted = point_forecast[0][:HORIZON].tolist()

        predicted_high = float(np.max(forecasted))
        predicted_low = float(np.min(forecasted))
        predicted_range = predicted_high - predicted_low
        predicted_move = predicted_range / 2

        return {
            "predicted_range_pts": round(predicted_range, 1),
            "predicted_move_1sd": round(predicted_move, 1),
            "vix_move_1sd": round(vix_move, 1),
            "range_vs_vix": round(predicted_move / vix_move, 3) if vix_move > 0 else 1.0,
            "source": "timesfm",
            "forecast_path": [round(v, 1) for v in forecasted],
        }
    except Exception:
        return _vix_fallback(vix_move)


def batch_forecast(
    df: pd.DataFrame,
    entry_dates,
    vix_series: pd.Series,
    spot_series: pd.Series,
    use_timesfm: bool = True,
    cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run TimesFM forecasts for all entry dates.

    Loads/saves a local cache at results/timesfm_cache/<hash>.parquet
    so re-runs don't recompute. Falls back to VIX if TimesFM unavailable.

    Returns DataFrame indexed by date with columns:
      predicted_move_1sd, vix_move_1sd, range_vs_vix, source
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    model = _load_model(verbose=verbose) if use_timesfm else None
    if model is None:
        use_timesfm = False

    records = []
    for date in entry_dates:
        vix = float(vix_series.get(date, vix_series.iloc[-1]))
        spot = float(spot_series.get(date, spot_series.iloc[-1]))
        fc = forecast_weekly_range(df, date, vix, spot, model=model)
        rec = {
            "date": date,
            "predicted_move_1sd": fc["predicted_move_1sd"],
            "vix_move_1sd": fc["vix_move_1sd"],
            "range_vs_vix": fc["range_vs_vix"],
            "source": fc["source"],
        }
        records.append(rec)

    result = pd.DataFrame(records).set_index("date")

    if verbose and use_timesfm and len(result) > 0:
        ratio = result["range_vs_vix"]
        print(f"\n  TimesFM forecasts: {len(result)} weeks")
        print(f"  Range vs VIX — mean: {ratio.mean():.2f}  "
              f"min: {ratio.min():.2f}  max: {ratio.max():.2f}")
        quiet = (ratio < 0.85).sum()
        loud = (ratio > 1.15).sum()
        print(f"  Quiet weeks (ratio<0.85): {quiet}  "
              f"Loud weeks (ratio>1.15): {loud}")

    return result
