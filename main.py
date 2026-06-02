"""
Nifty Weekly Options Trading System
Entry point: run backtest + generate weekly signal
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt

from src.data.fetcher import fetch_combined
from src.volatility.analyzer import weekly_stats
from src.regime.state_model import RegimeModel
from src.strategy.selector import select_strategy
from src.backtest.engine import Backtester
from src.backtest.metrics import performance_report, print_report, regime_breakdown


def run_backtest(start: str = "2019-01-01", end: str = "2024-12-31", capital: float = 1_000_000, use_sample: bool = False):
    print(f"\nFetching Nifty + VIX data ({start} → {end})...")
    if use_sample:
        from src.data.sample_data import generate_sample_data
        raw = generate_sample_data(start, end)
        print("(Using synthetic sample data — run on your local machine for real yfinance data)")
    else:
        raw = fetch_combined(start, end)

    print("Computing volatility features...")
    df = weekly_stats(raw)

    print("Training regime model (HMM)...")
    model = RegimeModel(n_states=3)
    model.fit(df)
    regimes = model.predict(df)
    df["regime"] = regimes

    print("Running backtest...")
    bt = Backtester(df, regimes, capital=capital)
    equity_curve = bt.run()

    metrics = performance_report(equity_curve, bt.trades)
    print_report(metrics)

    rd = regime_breakdown(bt.trades)
    if not rd.empty:
        print("\nRegime Breakdown:")
        print(rd.to_string(index=False))

    _plot_equity(equity_curve, capital)
    return equity_curve, bt.trades, metrics


def weekly_signal(capital: float = 1_000_000):
    """Generate a live trading signal for the current week."""
    print("\nFetching latest data for weekly signal...")
    df = fetch_combined(start="2022-01-01")
    df = weekly_stats(df)

    model = RegimeModel(n_states=3)
    model.fit(df)
    state = model.current_regime(df)

    latest = df.iloc[-1]
    spot = float(latest["Close"])
    vix = float(latest["VIX"])
    iv_hv = float(latest["IV_HV_ratio"])
    weekly_move = float(latest["weekly_move_1sd"])

    print("\n" + "=" * 50)
    print("  WEEKLY SIGNAL")
    print("=" * 50)
    print(f"  Nifty Spot  : {spot:,.0f}")
    print(f"  India VIX   : {vix:.2f}")
    print(f"  HV20        : {latest['HV20']*100:.2f}%")
    print(f"  IV/HV Ratio : {iv_hv:.2f}  {'✓ SELL EDGE' if iv_hv > 1.0 else '✗ NO EDGE'}")
    print(f"  1SD Weekly  : ±{weekly_move:,.0f} pts  ({spot-weekly_move:,.0f} – {spot+weekly_move:,.0f})")
    print(f"  Regime      : {state['regime'].upper()}  (confidence: {state['confidence']:.0%})")

    from datetime import datetime, timedelta
    today = datetime.today()
    entry = today.strftime("%Y-%m-%d")
    # Next Thursday
    days_to_thu = (3 - today.weekday()) % 7 or 7
    expiry = (today + timedelta(days=days_to_thu)).strftime("%Y-%m-%d")

    setup = select_strategy(
        spot=spot, vix=vix, entry_date=entry, expiry_date=expiry,
        regime=state["regime"], confidence=state["confidence"],
        iv_hv_ratio=iv_hv,
    )

    if setup is None:
        print("\n  RECOMMENDATION: SKIP THIS WEEK (IV not rich enough)")
    else:
        print(f"\n  STRATEGY    : {setup.strategy.upper().replace('_', ' ')}")
        print(f"  Max Profit  : {setup.max_profit:,.0f} pts/lot")
        print(f"  Breakevens  : {setup.breakeven_lower:,.0f} – {setup.breakeven_upper:,.0f}")
        print("\n  Legs:")
        for strike, opt_type, action, premium in setup.legs:
            print(f"    {action.upper():4s} {strike:,} {opt_type.upper():4s}  @ {premium:.1f}")

    print("=" * 50)
    return setup


def _plot_equity(equity_curve: pd.DataFrame, initial_capital: float):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ec = equity_curve[equity_curve["trade"] != "skip"]

    ax1.plot(equity_curve.index, equity_curve["capital"] / initial_capital * 100 - 100,
             color="steelblue", linewidth=1.5)
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.fill_between(equity_curve.index,
                     equity_curve["capital"] / initial_capital * 100 - 100, 0,
                     where=(equity_curve["capital"] >= initial_capital),
                     alpha=0.2, color="green")
    ax1.fill_between(equity_curve.index,
                     equity_curve["capital"] / initial_capital * 100 - 100, 0,
                     where=(equity_curve["capital"] < initial_capital),
                     alpha=0.2, color="red")
    ax1.set_ylabel("Cumulative Return (%)")
    ax1.set_title("Nifty Weekly Options — Equity Curve")
    ax1.grid(True, alpha=0.3)

    colors = {"skip": "gray", "short_straddle": "green", "short_strangle": "blue", "iron_condor": "orange"}
    for strat, color in colors.items():
        mask = equity_curve["trade"] == strat
        ax2.bar(equity_curve.index[mask],
                equity_curve["pnl"][mask],
                color=color, alpha=0.7, width=3, label=strat)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Weekly P&L (INR)")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("backtest_results.png", dpi=150)
    print("\nEquity curve saved → backtest_results.png")
    plt.close()


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    use_sample = "--sample" in args
    if "signal" in args:
        weekly_signal()
    else:
        run_backtest(use_sample=use_sample)
