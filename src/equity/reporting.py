"""
Reporting for the Lynch-style equity backtest: per-stock, per-sector,
and per-Lynch-category breakdowns ("where it worked / where it didn't"),
plus equity-curve plotting vs the Nifty benchmark.
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

from ..backtest.metrics import breakdown_by
from ..backtest.results_saver import save_run


def per_stock_breakdown(trades: list) -> pd.DataFrame:
    return breakdown_by(trades, "strategy").rename(columns={"strategy": "ticker"})


def per_sector_breakdown(trades: list) -> pd.DataFrame:
    return breakdown_by(trades, "sector")


def per_category_breakdown(trades: list) -> pd.DataFrame:
    return breakdown_by(trades, "category")


def category_equity_curves(trades: list, equity_curve: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate cumulative P&L contribution per Lynch category over time,
    built from each trade's pnl realized at its exit_date.
    """
    records = []
    for t in trades:
        if t.pnl is not None and t.exit_date:
            records.append({"date": pd.Timestamp(t.exit_date), "category": t.category, "pnl": t.pnl})
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    pivot = df.pivot_table(index="date", columns="category", values="pnl", aggfunc="sum")
    pivot = pivot.reindex(equity_curve.index, fill_value=0).fillna(0)
    return pivot.cumsum()


def save_equity_run(equity_curve: pd.DataFrame, trades: list, run_label: str, extra: dict = None) -> Path:
    """Persists standard run artifacts via save_run(), plus per-stock/sector/category CSVs."""
    from ..backtest.metrics import performance_report
    metrics = performance_report(equity_curve, trades, periods_per_year=252)

    run_dir = save_run(metrics, equity_curve, trades, run_label=run_label, extra=extra)

    per_stock_breakdown(trades).to_csv(run_dir / "per_stock.csv", index=False)
    per_sector_breakdown(trades).to_csv(run_dir / "per_sector.csv", index=False)
    per_category_breakdown(trades).to_csv(run_dir / "per_category.csv", index=False)

    return run_dir


def print_equity_report(metrics: dict, per_stock: pd.DataFrame, per_sector: pd.DataFrame, per_category: pd.DataFrame):
    print("\n" + "=" * 60)
    print("   LYNCH-STYLE EQUITY BACKTEST — RESULTS")
    print("=" * 60)

    rows = [(k, v) for k, v in metrics.items() if k not in ("Strategy Mix",)]
    print(tabulate(rows, headers=["Metric", "Value"], tablefmt="rounded_outline"))

    if not per_category.empty:
        print("\nPer-Category (Lynch type) — where it worked / didn't:")
        print(tabulate(per_category, headers="keys", tablefmt="rounded_outline", showindex=False))

    if not per_sector.empty:
        print("\nPer-Sector:")
        print(tabulate(per_sector, headers="keys", tablefmt="rounded_outline", showindex=False))

    if not per_stock.empty:
        print("\nTop 10 stocks by total P&L:")
        print(tabulate(per_stock.head(10), headers="keys", tablefmt="rounded_outline", showindex=False))
        print("\nBottom 10 stocks by total P&L:")
        print(tabulate(per_stock.tail(10), headers="keys", tablefmt="rounded_outline", showindex=False))

    print("=" * 60 + "\n")


def plot_equity_vs_benchmark(equity_curve: pd.DataFrame, trades: list, run_label: str = "equity_lynch"):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)

    initial = equity_curve["capital"].iloc[0]
    strat_ret = equity_curve["capital"] / initial * 100 - 100
    bench_ret = equity_curve["nifty_capital"] / equity_curve["nifty_capital"].iloc[0] * 100 - 100

    axes[0].plot(equity_curve.index, strat_ret, label="Lynch Strategy", color="steelblue", linewidth=1.5)
    axes[0].plot(equity_curve.index, bench_ret, label="Nifty 50 (Buy & Hold)", color="gray", linewidth=1.2, linestyle="--")
    axes[0].axhline(0, color="black", linewidth=0.6)
    axes[0].set_ylabel("Cumulative Return (%)")
    axes[0].set_title("Lynch-Style Equity Strategy vs Nifty 50")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    cat_curves = category_equity_curves(trades, equity_curve)
    if not cat_curves.empty:
        for col in cat_curves.columns:
            axes[1].plot(cat_curves.index, cat_curves[col], label=col)
        axes[1].axhline(0, color="black", linewidth=0.6)
        axes[1].set_ylabel("Cumulative P&L (INR)")
        axes[1].set_title("P&L Contribution by Lynch Category")
        axes[1].legend(loc="upper left")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"equity_results_{run_label}.png"
    plt.savefig(fname, dpi=150)
    print(f"  Equity chart saved → {fname}")
    plt.close()
