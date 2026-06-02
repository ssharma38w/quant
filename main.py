"""
Nifty Weekly Options Trading System
Usage:
  python main.py                         # backtest with real yfinance data
  python main.py --sample                # backtest with synthetic data (offline)
  python main.py --real-options          # backtest + download NSE bhavcopy options data
  python main.py --sweep                 # parameter sweep to find optimal settings
  python main.py signal                  # this week's live trade recommendation
  python main.py --csv nifty.csv         # backtest from your own CSV file
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import pandas as pd
import matplotlib.pyplot as plt

from src.data.fetcher import fetch_combined, load_from_csv
from src.volatility.analyzer import weekly_stats
from src.regime.state_model import RegimeModel
from src.strategy.selector import select_strategy
from src.backtest.engine import Backtester
from src.backtest.metrics import performance_report, print_report, regime_breakdown


def _load_data(args: list, start: str, end: str) -> pd.DataFrame:
    if "--sample" in args:
        from src.data.sample_data import generate_sample_data
        print("(Using synthetic sample data)")
        return generate_sample_data(start, end)
    elif "--csv" in args:
        idx = args.index("--csv")
        nifty_csv = args[idx + 1]
        vix_csv = args[idx + 2] if idx + 2 < len(args) and not args[idx + 2].startswith("--") else None
        print(f"Loading from CSV: {nifty_csv}")
        return load_from_csv(nifty_csv, vix_csv)
    else:
        return fetch_combined(start, end)


def run_backtest(
    args: list,
    start: str = "2019-01-01",
    end: str = "2025-05-30",
    capital: float = 1_000_000,
):
    print(f"\nFetching Nifty + VIX data ({start} → {end})...")
    raw = _load_data(args, start, end)

    print("Computing volatility features...")
    df = weekly_stats(raw)

    print("Training regime model (HMM)...")
    model = RegimeModel(n_states=3)
    model.fit(df)
    regimes = model.predict(df)
    df["regime"] = regimes

    # Parameter sweep
    if "--sweep" in args:
        from src.backtest.parameter_sweep import run_sweep, best_params
        sweep_results = run_sweep(df, regimes, capital)
        sweep_results.to_csv("sweep_results.csv", index=False)
        print("Full sweep saved → sweep_results.csv")
        params = best_params(sweep_results)
        print(f"\nBest params: {params}")
        return

    # Real NSE options data
    if "--real-options" in args:
        from src.data.nse_options import fetch_options_range
        from src.backtest.engine_real import RealDataBacktester
        print("\nDownloading NSE F&O bhavcopy data (this takes a few minutes)...")
        options_data = fetch_options_range(start, end)
        print("Running real-data backtest...")
        bt = RealDataBacktester(df, regimes, options_data, capital=capital)
        equity_curve = bt.run()
    else:
        print("Running backtest (Black-Scholes pricing)...")
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


def weekly_signal(args: list, capital: float = 1_000_000):
    """Generate a live trading signal for the current week."""
    from datetime import datetime, timedelta

    print("\nFetching latest data for weekly signal...")
    if "--sample" in args:
        from src.data.sample_data import generate_sample_data
        raw = generate_sample_data("2019-01-01")
    elif "--csv" in args:
        idx = args.index("--csv")
        raw = load_from_csv(args[idx + 1])
    else:
        raw = fetch_combined(start="2019-01-01")

    df = weekly_stats(raw)
    model = RegimeModel(n_states=3)
    model.fit(df)
    state = model.current_regime(df)

    latest = df.iloc[-1]
    spot = float(latest["Close"])
    vix = float(latest["VIX"])
    iv_hv = float(latest["IV_HV_ratio"])
    weekly_move = float(latest["weekly_move_1sd"])

    today = datetime.today()
    entry = today.strftime("%Y-%m-%d")
    days_to_thu = (3 - today.weekday()) % 7 or 7
    expiry = (today + timedelta(days=days_to_thu)).strftime("%Y-%m-%d")

    print("\n" + "=" * 55)
    print("  NIFTY WEEKLY OPTIONS — SIGNAL")
    print("=" * 55)
    print(f"  Nifty Spot   : {spot:>10,.0f}")
    print(f"  India VIX    : {vix:>10.2f}")
    print(f"  HV20         : {latest['HV20']*100:>9.2f}%")
    print(f"  IV/HV Ratio  : {iv_hv:>10.2f}  {'✓ SELL EDGE' if iv_hv > 1.0 else '✗ NO EDGE - SKIP'}")
    print(f"  1SD Range    :  {spot-weekly_move:>8,.0f} – {spot+weekly_move:,.0f}  (±{weekly_move:,.0f} pts)")
    print(f"  Regime       :  {state['regime'].upper():<12}  confidence: {state['confidence']:.0%}")
    print(f"  Entry        :  {entry}")
    print(f"  Expiry       :  {expiry}")
    print("-" * 55)

    setup = select_strategy(
        spot=spot, vix=vix, entry_date=entry, expiry_date=expiry,
        regime=state["regime"], confidence=state["confidence"],
        iv_hv_ratio=iv_hv,
    )

    if setup is None:
        print("  RECOMMENDATION: SKIP (IV not rich vs realized vol)")
    else:
        print(f"  STRATEGY     :  {setup.strategy.upper().replace('_', ' ')}")
        print(f"  Max Profit   : {setup.max_profit:>9,.1f} pts/lot  = ₹{setup.max_profit*75:,.0f}/lot")
        if setup.max_loss != float("inf"):
            print(f"  Max Loss     : {setup.max_loss:>9,.1f} pts/lot  = ₹{setup.max_loss*75:,.0f}/lot")
        print(f"  Breakevens   :  {setup.breakeven_lower:,.0f} – {setup.breakeven_upper:,.0f}")
        print("\n  Legs:")
        for strike, opt_type, action, premium in setup.legs:
            print(f"    {action.upper():4s}  {strike:>7,}  {opt_type.upper():4s}  @ {premium:>6.1f} pts")

    print("=" * 55)
    return setup


def _plot_equity(equity_curve: pd.DataFrame, initial_capital: float):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.plot(
        equity_curve.index,
        equity_curve["capital"] / initial_capital * 100 - 100,
        color="steelblue", linewidth=1.5,
    )
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.fill_between(
        equity_curve.index,
        equity_curve["capital"] / initial_capital * 100 - 100, 0,
        where=(equity_curve["capital"] >= initial_capital), alpha=0.2, color="green",
    )
    ax1.fill_between(
        equity_curve.index,
        equity_curve["capital"] / initial_capital * 100 - 100, 0,
        where=(equity_curve["capital"] < initial_capital), alpha=0.2, color="red",
    )
    ax1.set_ylabel("Cumulative Return (%)")
    ax1.set_title("Nifty Weekly Options — Equity Curve")
    ax1.grid(True, alpha=0.3)

    colors = {
        "skip": "lightgray", "short_straddle": "green",
        "short_strangle": "steelblue", "iron_condor": "darkorange",
    }
    for strat, color in colors.items():
        mask = equity_curve["trade"] == strat
        if mask.any():
            ax2.bar(
                equity_curve.index[mask], equity_curve["pnl"][mask],
                color=color, alpha=0.75, width=3, label=strat,
            )
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Weekly P&L (INR)")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("backtest_results.png", dpi=150)
    print("\nEquity curve saved → backtest_results.png")
    plt.close()


if __name__ == "__main__":
    args = sys.argv[1:]
    if "signal" in args:
        weekly_signal(args)
    else:
        run_backtest(args)
