"""
Nifty Weekly Options Trading System
Usage:
  python main.py                    # backtest with real yfinance data
  python main.py --sample           # offline test with synthetic data
  python main.py --ml               # backtest + train ML strategy predictor
  python main.py --sweep            # parameter sweep
  python main.py signal             # this week's live trade recommendation
  python main.py --csv nifty.csv    # backtest from your own CSV
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data.fetcher import fetch_combined, load_from_csv
from src.volatility.analyzer import weekly_stats
from src.regime.state_model import RegimeModel
from src.strategy.selector import select_strategy
from src.backtest.engine import Backtester
from src.backtest.metrics import performance_report, print_report, regime_breakdown


def _load_data(args, start, end):
    if "--sample" in args:
        from src.data.sample_data import generate_sample_data
        print("(Using synthetic sample data)")
        return generate_sample_data(start, end)
    if "--csv" in args:
        idx = args.index("--csv")
        nifty_csv = args[idx + 1]
        vix_csv = args[idx + 2] if idx + 2 < len(args) and not args[idx + 2].startswith("--") else None
        return load_from_csv(nifty_csv, vix_csv)
    return fetch_combined(start, end)


def run_backtest(args, start="2019-01-01", end="2025-05-30", capital=1_000_000):
    print(f"\nFetching Nifty + VIX data ({start} → {end})...")
    raw = _load_data(args, start, end)

    print("Computing volatility features...")
    df = weekly_stats(raw)

    print("Training regime model (HMM)...")
    model = RegimeModel(n_states=3)
    model.fit(df)
    regimes = model.predict(df)
    regime_probs = model.predict_proba(df)
    df["regime"] = regimes

    if "--sweep" in args:
        from src.backtest.parameter_sweep import run_sweep
        run_sweep(df, regimes, capital)
        return

    # ML predictor
    ml_predictor, ml_features = None, None
    if "--ml" in args:
        print("Training ML strategy predictor...")
        from src.models.feature_engineering import build_weekly_features
        from src.models.strategy_predictor import StrategyPredictor

        # First pass: run rule-based backtest to generate training labels
        bt0 = Backtester(df, regimes, capital=capital, stop_loss_multiple=2.0, use_calendar=False)
        bt0.run()

        ml_features = build_weekly_features(df, regime_probs)
        ml_predictor = StrategyPredictor()
        ml_predictor.fit(ml_features, bt0.trades, verbose=True)

        imp = ml_predictor.feature_importance()
        if not imp.empty:
            print("\n  Top predictive features:")
            for feat, score in imp.items():
                print(f"    {feat:<25} {score:.4f}")

    print("Running backtest...")
    bt = Backtester(
        df, regimes, capital=capital,
        stop_loss_multiple=2.0,
        ml_predictor=ml_predictor,
        ml_features=ml_features,
        use_calendar=True,
    )
    equity_curve = bt.run()

    metrics = performance_report(equity_curve, bt.trades)
    print_report(metrics)

    rd = regime_breakdown(bt.trades)
    if not rd.empty:
        print("\nRegime Breakdown:")
        print(rd.to_string(index=False))

    _plot_equity(equity_curve, capital)
    return equity_curve, bt.trades, metrics


def weekly_signal(args, capital=1_000_000):
    from datetime import datetime, timedelta
    print("\nFetching latest data for weekly signal...")
    raw = _load_data(args, "2019-01-01", None) if "--sample" not in args else \
          __import__("src.data.sample_data", fromlist=["generate_sample_data"]).generate_sample_data("2019-01-01")

    df = weekly_stats(raw)
    model = RegimeModel(n_states=3)
    model.fit(df)
    state = model.current_regime(df)

    latest = df.iloc[-1]
    spot = float(latest["Close"])
    vix  = float(latest["VIX"])
    iv_hv = float(latest["IV_HV_ratio"])
    weekly_move = float(latest["weekly_move_1sd"])

    today = datetime.today()
    days_to_thu = (3 - today.weekday()) % 7 or 7
    expiry = (today + timedelta(days=days_to_thu)).strftime("%Y-%m-%d")
    next_expiry = (today + timedelta(days=days_to_thu + 7)).strftime("%Y-%m-%d")

    print("\n" + "=" * 58)
    print("  NIFTY WEEKLY OPTIONS — SIGNAL")
    print("=" * 58)
    print(f"  Nifty Spot   : {spot:>10,.0f}")
    print(f"  India VIX    : {vix:>10.2f}")
    print(f"  HV20         : {latest['HV20']*100:>9.2f}%")
    print(f"  IV/HV Ratio  : {iv_hv:>10.2f}  {'✓ EDGE' if iv_hv > 1.05 else '✗ NO EDGE'}")
    print(f"  1SD Range    :  {spot-weekly_move:>8,.0f} – {spot+weekly_move:,.0f}  (±{weekly_move:,.0f} pts)")
    print(f"  Regime       :  {state['regime'].upper():<12}  confidence: {state['confidence']:.0%}")
    print(f"  Expiry       :  {expiry}")
    print("-" * 58)

    setup = select_strategy(
        spot=spot, vix=vix, entry_date=today.strftime("%Y-%m-%d"),
        expiry_date=expiry, regime=state["regime"], confidence=state["confidence"],
        iv_hv_ratio=iv_hv, next_expiry_date=next_expiry, use_calendar=True,
    )

    if setup is None:
        print("  RECOMMENDATION: SKIP (IV not rich or VIX too high)")
    else:
        print(f"  STRATEGY     :  {setup.strategy.upper().replace('_', ' ')}")
        print(f"  Max Profit   : {setup.max_profit:>9,.1f} pts  = ₹{setup.max_profit*75:,.0f}/lot")
        if setup.max_loss != float("inf"):
            print(f"  Max Loss     : {setup.max_loss:>9,.1f} pts  = ₹{setup.max_loss*75:,.0f}/lot")
        print(f"  Breakevens   :  {setup.breakeven_lower:,.0f} – {setup.breakeven_upper:,.0f}")
        print("\n  Legs:")
        for strike, opt_type, action, premium in setup.legs:
            print(f"    {action.upper():4s}  {strike:>7,}  {opt_type.upper():4s}  @ {premium:>6.1f} pts")
    print("=" * 58)
    return setup


def _plot_equity(equity_curve, initial_capital):
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    # Equity curve
    ret = equity_curve["capital"] / initial_capital * 100 - 100
    axes[0].plot(equity_curve.index, ret, color="steelblue", linewidth=1.5)
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].fill_between(equity_curve.index, ret, 0, where=(ret >= 0), alpha=0.2, color="green")
    axes[0].fill_between(equity_curve.index, ret, 0, where=(ret < 0), alpha=0.2, color="red")
    axes[0].set_ylabel("Cumulative Return (%)")
    axes[0].set_title("Nifty Weekly Options — Equity Curve")
    axes[0].grid(True, alpha=0.3)

    # Weekly P&L bars by strategy
    colors = {
        "short_straddle": "green", "iron_condor": "darkorange",
        "calendar_spread": "purple", "broken_wing_butterfly": "brown",
        "ratio_spread": "teal", "skip": "lightgray",
    }
    for strat, color in colors.items():
        mask = equity_curve["trade"] == strat
        if mask.any():
            axes[1].bar(equity_curve.index[mask], equity_curve["pnl"][mask],
                        color=color, alpha=0.75, width=3, label=strat)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Weekly P&L (INR)")
    axes[1].legend(loc="upper left", fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.3)

    # VIX overlay
    if "vix" in equity_curve.columns:
        axes[2].fill_between(equity_curve.index, equity_curve["vix"],
                             alpha=0.4, color="red", label="India VIX")
        axes[2].axhline(16, color="green",  linestyle="--", linewidth=0.8, label="VIX 16")
        axes[2].axhline(21, color="orange", linestyle="--", linewidth=0.8, label="VIX 21")
        axes[2].axhline(26, color="red",    linestyle="--", linewidth=0.8, label="VIX 26 (skip)")
        axes[2].set_ylabel("India VIX")
        axes[2].legend(loc="upper right", fontsize=7)
        axes[2].grid(True, alpha=0.3)

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
