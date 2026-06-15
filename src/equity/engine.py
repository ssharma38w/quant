"""
Lynch-style multi-stock equity backtest engine.

Monthly (or quarterly) rebalance: rank universe by category factor
scores (factors.compute_factor_snapshot), select top_n stocks per
category target weights, equal-weight (or score-weighted) portfolio,
daily mark-to-market between rebalances. Tracks per-trade records for
per-stock/sector/category reporting.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .factors import compute_factor_snapshot, CATEGORIES


@dataclass
class EquityTrade:
    """getattr-compatible with src.backtest.results_saver.save_run()"""
    entry_date: str
    exit_date: Optional[str] = None
    expiry_date: Optional[str] = None
    strategy: str = ""          # ticker
    regime: str = ""            # sector
    spot_entry: float = 0.0
    spot_exit: Optional[float] = None
    vix_entry: Optional[float] = None
    lots: int = 1
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    outcome: Optional[str] = None
    max_profit: float = float("inf")
    max_loss: float = float("-inf")
    sector: str = ""
    category: str = ""
    weight: float = 0.0


@dataclass
class EquityBacktestConfig:
    start: str = "2015-01-01"
    end: Optional[str] = None
    rebalance_freq: str = "M"       # "M" monthly, "Q" quarterly
    top_n: int = 20
    weighting: str = "category_balanced"  # "equal" | "score_weighted" | "category_balanced"
    category_targets: dict = field(default_factory=lambda: {
        "fast_grower": 0.35, "stalwart": 0.30, "turnaround": 0.20, "cyclical": 0.15,
    })
    transaction_cost_bps: float = 10.0
    initial_capital: float = 1_000_000
    min_history_days: int = 280
    min_avg_dollar_volume: float = 5e7


class EquityBacktester:
    def __init__(
        self,
        prices: dict[str, pd.DataFrame],
        nifty: pd.DataFrame,
        universe_df: pd.DataFrame,
        config: EquityBacktestConfig,
    ):
        self.prices = prices
        self.nifty = nifty.copy()
        self.nifty.index = pd.to_datetime(self.nifty.index)
        self.sector_map = dict(zip(universe_df["ticker"], universe_df["sector"]))
        self.cap_map = dict(zip(universe_df["ticker"], universe_df["market_cap_tier"]))
        self.config = config
        self.trades: list[EquityTrade] = []

    # ── helpers ────────────────────────────────────────────────────
    def get_rebalance_dates(self) -> list[pd.Timestamp]:
        all_dates = self.nifty.index
        cfg = self.config
        start, end = pd.Timestamp(cfg.start), pd.Timestamp(cfg.end or all_dates[-1])
        all_dates = all_dates[(all_dates >= start) & (all_dates <= end)]
        if len(all_dates) == 0:
            return []
        df = pd.Series(all_dates, index=all_dates)
        grouped = df.groupby(df.index.to_period(cfg.rebalance_freq))
        return [g.iloc[-1] for _, g in grouped]

    def _select_portfolio(self, snap: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        if snap.empty:
            return snap

        if cfg.weighting == "category_balanced":
            picks = []
            for cat, target_pct in cfg.category_targets.items():
                n = max(1, round(cfg.top_n * target_pct))
                col = f"score_{cat}"
                cat_df = snap[snap["assigned_category"] == cat].sort_values(col, ascending=False).head(n)
                picks.append(cat_df)
            selected = pd.concat(picks) if picks else snap.head(0)
            selected = selected[~selected.index.duplicated()]
        else:
            selected = snap.sort_values("composite_score", ascending=False).head(cfg.top_n)

        if selected.empty:
            return selected

        if cfg.weighting == "score_weighted":
            raw = (selected["composite_score"] - selected["composite_score"].min() + 0.1)
            w = raw / raw.sum()
            w = w.clip(upper=0.10)
            w = w / w.sum()
            selected = selected.copy()
            selected["weight"] = w
        else:
            selected = selected.copy()
            selected["weight"] = 1.0 / len(selected)

        return selected

    def _price_on_or_before(self, ticker: str, date: pd.Timestamp) -> Optional[float]:
        df = self.prices.get(ticker)
        if df is None:
            return None
        sub = df.loc[df.index <= date]
        if sub.empty:
            return None
        return float(sub["Close"].iloc[-1])

    # ── main loop ──────────────────────────────────────────────────
    def run(self) -> tuple[pd.DataFrame, list[EquityTrade]]:
        cfg = self.config
        rebalance_dates = self.get_rebalance_dates()
        if not rebalance_dates:
            return pd.DataFrame(), []

        capital = cfg.initial_capital
        nifty_capital = cfg.initial_capital
        nifty_close = self.nifty["Close"]
        nifty_ret = nifty_close.pct_change()

        # current holdings: ticker -> {shares, weight, entry_price, trade(EquityTrade)}
        holdings: dict[str, dict] = {}
        equity_records = []

        all_days = nifty_close.index
        for i, day in enumerate(all_days):
            if day < pd.Timestamp(cfg.start):
                continue
            if day > pd.Timestamp(cfg.end or all_days[-1]):
                break

            # Mark-to-market existing holdings
            if i > 0 and not np.isnan(nifty_ret.loc[day]):
                nifty_capital *= (1 + nifty_ret.loc[day])

            portfolio_value = 0.0
            for ticker, h in holdings.items():
                price = self._price_on_or_before(ticker, day)
                if price is not None:
                    portfolio_value += h["shares"] * price
            if holdings:
                # cash portion (if portfolio doesn't use full capital due to rounding)
                day_capital = portfolio_value + holdings.get("__cash__", 0.0)
            else:
                day_capital = capital

            # Rebalance
            if day in rebalance_dates:
                # value current holdings at today's prices to get current capital
                if holdings:
                    capital = sum(
                        h["shares"] * (self._price_on_or_before(t, day) or h["entry_price"])
                        for t, h in holdings.items()
                    )

                snap = compute_factor_snapshot(
                    self.prices, self.nifty, as_of=day,
                    sector_map=self.sector_map, market_cap_tier=self.cap_map,
                    min_history_days=cfg.min_history_days,
                    min_avg_dollar_volume=cfg.min_avg_dollar_volume,
                )
                selected = self._select_portfolio(snap)

                new_tickers = set(selected.index) if not selected.empty else set()
                old_tickers = set(holdings.keys())

                turnover_value = 0.0

                # Close positions no longer selected
                for ticker in old_tickers - new_tickers:
                    h = holdings.pop(ticker)
                    exit_price = self._price_on_or_before(ticker, day) or h["entry_price"]
                    trade = h["trade"]
                    trade.exit_date = str(day.date())
                    trade.spot_exit = exit_price
                    pnl = h["shares"] * (exit_price - h["entry_price"])
                    trade.pnl = round(pnl, 2)
                    trade.pnl_pct = round(pnl / cfg.initial_capital * 100, 4)
                    trade.outcome = "profit" if pnl >= 0 else "loss"
                    self.trades.append(trade)
                    turnover_value += h["shares"] * exit_price

                # Open / resize positions
                if not selected.empty:
                    for ticker, row in selected.iterrows():
                        price = self._price_on_or_before(ticker, day)
                        if price is None or price <= 0:
                            continue
                        target_value = capital * row["weight"]
                        shares = target_value / price

                        if ticker in holdings:
                            old_value = holdings[ticker]["shares"] * price
                            turnover_value += abs(target_value - old_value)
                            holdings[ticker]["shares"] = shares
                            holdings[ticker]["weight"] = row["weight"]
                        else:
                            turnover_value += target_value
                            holdings[ticker] = {
                                "shares": shares,
                                "weight": row["weight"],
                                "entry_price": price,
                                "trade": EquityTrade(
                                    entry_date=str(day.date()),
                                    strategy=ticker,
                                    regime=self.sector_map.get(ticker, "Unknown"),
                                    spot_entry=price,
                                    sector=self.sector_map.get(ticker, "Unknown"),
                                    category=row["assigned_category"],
                                    weight=row["weight"],
                                ),
                            }

                # Transaction costs reduce capital
                cost = turnover_value * (cfg.transaction_cost_bps / 10000.0)
                capital -= cost

                # Recompute day_capital after rebalance
                day_capital = sum(
                    h["shares"] * (self._price_on_or_before(t, day) or h["entry_price"])
                    for t, h in holdings.items()
                )

            pnl_today = day_capital - (equity_records[-1]["capital"] if equity_records else cfg.initial_capital)
            equity_records.append({
                "date": day,
                "capital": day_capital,
                "pnl": pnl_today,
                "nifty_capital": nifty_capital,
            })

        # Close any remaining open positions at the end
        if equity_records:
            last_day = equity_records[-1]["date"]
            for ticker, h in holdings.items():
                exit_price = self._price_on_or_before(ticker, last_day) or h["entry_price"]
                trade = h["trade"]
                trade.exit_date = str(last_day.date())
                trade.spot_exit = exit_price
                pnl = h["shares"] * (exit_price - h["entry_price"])
                trade.pnl = round(pnl, 2)
                trade.pnl_pct = round(pnl / cfg.initial_capital * 100, 4)
                trade.outcome = "profit" if pnl >= 0 else "loss"
                self.trades.append(trade)

        equity_curve = pd.DataFrame(equity_records).set_index("date")
        return equity_curve, self.trades
