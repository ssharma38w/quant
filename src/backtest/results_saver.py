"""
Persist backtest results to results/ directory.

Each run saves:
  results/run_<timestamp>/
    metrics.json      — performance metrics dict
    equity_curve.csv  — weekly equity curve
    trades.csv        — all trade records
    summary.txt       — human-readable report
"""
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd


_RESULTS_DIR = Path("results")


def save_run(
    metrics: dict,
    equity_curve: pd.DataFrame,
    trades: list,
    run_label: str = "",
    extra: dict = None,
) -> Path:
    """
    Save metrics, equity curve, and trade log.
    Returns the run directory path.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = f"_{run_label}" if run_label else ""
    run_dir = _RESULTS_DIR / f"run_{ts}{label}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- metrics.json ---
    meta = {
        "timestamp": ts,
        "label": run_label,
        **(extra or {}),
    }
    out = {**meta, **metrics}
    # Convert non-serialisable types
    for k, v in out.items():
        if hasattr(v, "item"):       # numpy scalar
            out[k] = v.item()
        elif isinstance(v, dict):
            pass
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2))

    # --- equity_curve.csv ---
    ec = equity_curve.copy()
    ec.index.name = "date"
    ec.to_csv(run_dir / "equity_curve.csv")

    # --- trades.csv ---
    if trades:
        rows = []
        for t in trades:
            rows.append({
                "entry_date": t.entry_date,
                "exit_date": getattr(t, "exit_date", ""),
                "expiry_date": t.expiry_date,
                "strategy": t.strategy,
                "regime": t.regime,
                "spot_entry": t.spot_entry,
                "vix_entry": t.vix_entry,
                "lots": getattr(t, "lots", 1),
                "pnl": getattr(t, "pnl", None),
                "pnl_pct": getattr(t, "pnl_pct", None),
                "outcome": getattr(t, "outcome", ""),
                "max_profit": t.max_profit,
                "max_loss": t.max_loss if t.max_loss != float("inf") else None,
            })
        pd.DataFrame(rows).to_csv(run_dir / "trades.csv", index=False)

    # --- summary.txt ---
    lines = [
        f"Nifty Weekly Options Backtest — {ts}",
        f"Label: {run_label}" if run_label else "",
        "=" * 50,
    ]
    for k, v in metrics.items():
        if k != "Strategy Mix":
            lines.append(f"  {k:<30} {v}")
    lines.append("")
    lines.append(f"Strategy Mix: {metrics.get('Strategy Mix', {})}")
    lines.append("=" * 50)
    (run_dir / "summary.txt").write_text("\n".join(l for l in lines if l is not None))

    # Update rolling best-results index
    _update_index(run_dir, metrics, run_label)

    print(f"\n  Results saved → {run_dir}/")
    return run_dir


def _update_index(run_dir: Path, metrics: dict, label: str):
    """Append this run to results/index.csv for easy comparison."""
    index_path = _RESULTS_DIR / "index.csv"
    row = {
        "run_dir": str(run_dir.name),
        "label": label,
        "cagr": metrics.get("CAGR (%)", 0),
        "sharpe": metrics.get("Sharpe Ratio", 0),
        "max_dd": metrics.get("Max Drawdown (%)", 0),
        "profit_factor": metrics.get("Profit Factor", 0),
        "win_rate": metrics.get("Win Rate (%)", 0),
        "total_trades": metrics.get("Total Trades", 0),
        "final_capital": metrics.get("Final Capital (INR)", 0),
    }
    if index_path.exists():
        df = pd.read_csv(index_path)
    else:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(index_path, index=False)


def load_best_run(metric: str = "cagr") -> dict:
    """Load metrics from the best run by a given metric."""
    index_path = _RESULTS_DIR / "index.csv"
    if not index_path.exists():
        return {}
    df = pd.read_csv(index_path)
    if metric not in df.columns or df.empty:
        return {}
    best_row = df.loc[df[metric].idxmax()]
    run_dir = _RESULTS_DIR / best_row["run_dir"]
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    return {}
