"""
Parameter sweep to find optimal SD multiples and wing widths.
Tests all combinations and ranks by Sharpe ratio.
"""
import itertools
import pandas as pd
import numpy as np
from tabulate import tabulate

from ..strategy.strategies import short_strangle, iron_condor, payoff_at_expiry
from ..strategy.pricer import black_scholes
from ..sizing.scaler import LOT_SIZE
from .engine import get_weekly_schedule
from .metrics import performance_report


PARAM_GRID = {
    "sd_multiple": [0.6, 0.8, 1.0, 1.2, 1.5],
    "wing_width": [150, 200, 300, 400],
    "iv_hv_min": [1.0, 1.1, 1.2],
}


def _simulate_weekly(
    df: pd.DataFrame,
    regime_series: pd.Series,
    sd_multiple: float,
    wing_width: int,
    iv_hv_min: float,
    capital: float = 1_000_000,
) -> dict:
    schedule = get_weekly_schedule(df)
    cap = capital
    pnls = []

    for entry_date, expiry_date in schedule:
        if entry_date not in df.index or expiry_date not in df.index:
            continue

        row = df.loc[entry_date]
        spot = float(row["Close"])
        vix = float(row["VIX"])
        iv_hv = float(row.get("IV_HV_ratio", 1.2))
        regime = regime_series.get(entry_date, "normal")
        dte = max((expiry_date - entry_date).days, 1)

        if iv_hv < iv_hv_min:
            continue

        sigma = vix / 100
        T = dte / 365
        weekly_move = (sigma / np.sqrt(52)) * spot * sd_multiple
        short_call_k = int(round((spot + weekly_move) / 50) * 50)
        short_put_k = int(round((spot - weekly_move) / 50) * 50)
        long_call_k = short_call_k + wing_width
        long_put_k = short_put_k - wing_width

        sc = black_scholes(spot, short_call_k, T, sigma, "call")
        sp = black_scholes(spot, short_put_k, T, sigma, "put")
        lc = black_scholes(spot, long_call_k, T, sigma, "call")
        lp = black_scholes(spot, long_put_k, T, sigma, "put")
        net_premium = (sc + sp) - (lc + lp)
        max_loss = wing_width - net_premium

        spot_exit = float(df.loc[expiry_date, "Close"])

        def intrinsic(k, otype):
            return max(spot_exit - k, 0) if otype == "call" else max(k - spot_exit, 0)

        pnl_pt = (
            (sc - intrinsic(short_call_k, "call"))
            + (sp - intrinsic(short_put_k, "put"))
            - (lc - intrinsic(long_call_k, "call"))  # long: we paid lc, get intrinsic
            + (intrinsic(long_call_k, "call") - lc)  # rearranging...
        )
        # Correct calculation:
        pnl_pt = (
            (sc - intrinsic(short_call_k, "call"))
            + (sp - intrinsic(short_put_k, "put"))
            + (intrinsic(long_call_k, "call") - lc)
            + (intrinsic(long_put_k, "put") - lp)
        )

        lots = max(1, int((cap * 0.02) / (max_loss * LOT_SIZE)))
        lots = min(lots, 10)
        trade_pnl = pnl_pt * lots * LOT_SIZE - 150 * lots
        cap += trade_pnl
        pnls.append(trade_pnl)

    if not pnls:
        return {"sharpe": -99, "cagr": -99, "win_rate": 0, "total_trades": 0}

    arr = np.array(pnls)
    wins = (arr >= 0).sum()
    initial = capital
    n_years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = (cap / initial) ** (1 / n_years) - 1 if n_years > 0 else 0
    weekly_ret = arr / initial
    sharpe = weekly_ret.mean() / weekly_ret.std() * np.sqrt(52) if weekly_ret.std() > 0 else 0

    return {
        "sharpe": round(sharpe, 3),
        "cagr": round(cagr * 100, 2),
        "win_rate": round(wins / len(arr) * 100, 1),
        "total_trades": len(arr),
    }


def run_sweep(df: pd.DataFrame, regime_series: pd.Series, capital: float = 1_000_000) -> pd.DataFrame:
    """Run full parameter grid search. Returns ranked results."""
    print(f"\nRunning parameter sweep ({len(PARAM_GRID['sd_multiple'])} × {len(PARAM_GRID['wing_width'])} × {len(PARAM_GRID['iv_hv_min'])} = "
          f"{len(PARAM_GRID['sd_multiple'])*len(PARAM_GRID['wing_width'])*len(PARAM_GRID['iv_hv_min'])} combinations)...")

    results = []
    combos = list(itertools.product(
        PARAM_GRID["sd_multiple"],
        PARAM_GRID["wing_width"],
        PARAM_GRID["iv_hv_min"],
    ))

    for i, (sd, ww, ivhv) in enumerate(combos):
        print(f"  {i+1}/{len(combos)}: sd={sd}, wing={ww}, iv_hv_min={ivhv}", end="\r")
        stats = _simulate_weekly(df, regime_series, sd, ww, ivhv, capital)
        results.append({
            "sd_multiple": sd,
            "wing_width": ww,
            "iv_hv_min": ivhv,
            **stats,
        })

    df_results = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    print("\n\nTop 10 Parameter Combinations (by Sharpe):")
    print(tabulate(df_results.head(10), headers="keys", tablefmt="rounded_outline", showindex=False))
    return df_results


def best_params(sweep_results: pd.DataFrame) -> dict:
    """Extract best params from sweep results."""
    best = sweep_results.iloc[0]
    return {
        "sd_multiple": best["sd_multiple"],
        "wing_width": int(best["wing_width"]),
        "iv_hv_min": best["iv_hv_min"],
    }
