import pandas as pd
import numpy as np
from tabulate import tabulate


def performance_report(equity_curve: pd.DataFrame, trades: list) -> dict:
    ec = equity_curve.copy()
    initial_capital = ec["capital"].iloc[0]
    final_capital = ec["capital"].iloc[-1]

    total_return = (final_capital / initial_capital - 1) * 100
    n_years = (ec.index[-1] - ec.index[0]).days / 365.25
    cagr = ((final_capital / initial_capital) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    trade_pnls = [t.pnl for t in trades if t.pnl is not None]
    wins = [p for p in trade_pnls if p >= 0]
    losses = [p for p in trade_pnls if p < 0]

    win_rate = len(wins) / len(trade_pnls) * 100 if trade_pnls else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    profit_factor = abs(sum(wins) / sum(losses)) if losses else float("inf")

    # Sharpe (weekly returns)
    weekly_pnl = ec["pnl"].dropna()
    weekly_returns = weekly_pnl / initial_capital
    sharpe = (weekly_returns.mean() / weekly_returns.std() * np.sqrt(52)) if weekly_returns.std() > 0 else 0

    # Max drawdown
    capital_series = ec["capital"]
    rolling_peak = capital_series.cummax()
    drawdown = (capital_series - rolling_peak) / rolling_peak * 100
    max_dd = drawdown.min()

    # Strategy breakdown
    strat_counts = {}
    for t in trades:
        strat_counts[t.strategy] = strat_counts.get(t.strategy, 0) + 1

    # Stop loss stats (if available in equity curve)
    stop_loss_exits = 0
    if "exit_type" in ec.columns:
        stop_loss_exits = (ec["exit_type"] == "stop_loss").sum()

    # Avg lots traded
    avg_lots = ec["lots"].mean() if "lots" in ec.columns else 1

    metrics = {
        "Total Return (%)": round(total_return, 2),
        "CAGR (%)": round(cagr, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Max Drawdown (%)": round(max_dd, 2),
        "Win Rate (%)": round(win_rate, 2),
        "Avg Win (INR)": round(avg_win, 0),
        "Avg Loss (INR)": round(avg_loss, 0),
        "Profit Factor": round(profit_factor, 2),
        "Total Trades": len(trade_pnls),
        "Winning Trades": len(wins),
        "Losing Trades": len(losses),
        "Stop Loss Exits": stop_loss_exits,
        "Avg Lots / Trade": round(avg_lots, 1),
        "Strategy Mix": strat_counts,
        "Final Capital (INR)": round(final_capital, 0),
        "Initial Capital (INR)": round(initial_capital, 0),
    }
    return metrics


def print_report(metrics: dict):
    rows = [(k, v) for k, v in metrics.items() if k not in ("Strategy Mix",)]
    print("\n" + "=" * 50)
    print("   NIFTY WEEKLY OPTIONS BACKTEST REPORT")
    print("=" * 50)
    print(tabulate(rows, headers=["Metric", "Value"], tablefmt="rounded_outline"))
    print("\nStrategy Mix:", metrics.get("Strategy Mix", {}))
    print("=" * 50 + "\n")

    # Quick performance grade
    cagr = metrics.get("CAGR (%)", 0)
    sharpe = metrics.get("Sharpe Ratio", 0)
    dd = abs(metrics.get("Max Drawdown (%)", 0))
    pf = metrics.get("Profit Factor", 0)
    grade = "🟢 EXCELLENT" if cagr >= 20 and sharpe >= 2.0 and dd <= 20 else \
            "🟡 GOOD"      if cagr >= 12 and sharpe >= 1.0 and dd <= 25 else \
            "🔴 NEEDS WORK"
    print(f"  Grade: {grade}  |  CAGR {cagr:.1f}%  Sharpe {sharpe:.2f}  DD -{dd:.1f}%  PF {pf:.2f}\n")


def regime_breakdown(trades: list) -> pd.DataFrame:
    records = []
    for t in trades:
        if t.pnl is not None:
            records.append({"regime": t.regime, "strategy": t.strategy, "pnl": t.pnl})
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    summary = df.groupby(["regime", "strategy"])["pnl"].agg(["count", "sum", "mean"]).reset_index()
    summary.columns = ["regime", "strategy", "trades", "total_pnl", "avg_pnl"]
    return summary
