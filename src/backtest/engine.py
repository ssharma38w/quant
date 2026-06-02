"""
Weekly options backtesting engine for Nifty.
Entry: Monday of each week
Exit: Thursday (weekly expiry)
Pricing: Black-Scholes using VIX as IV proxy
"""
import pandas as pd
import numpy as np
from typing import List, Optional

from ..strategy.selector import select_strategy
from ..strategy.strategies import payoff_at_expiry
from ..strategy.base import TradeSetup
from ..sizing.scaler import ScalerAgent, LOT_SIZE


def get_weekly_schedule(df: pd.DataFrame) -> List[tuple]:
    """
    Returns list of (entry_date, expiry_date) pairs.
    Entry: Monday (or first trading day of week)
    Expiry: Thursday (or last trading day before Friday)
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df["weekday"] = df.index.dayofweek  # 0=Mon, 3=Thu, 4=Fri

    weeks = []
    trading_days = df.index.tolist()

    for i, date in enumerate(trading_days):
        if date.dayofweek == 0:  # Monday
            # Find Thursday of same week
            expiry = None
            for j in range(i, min(i + 5, len(trading_days))):
                if trading_days[j].dayofweek == 3:
                    expiry = trading_days[j]
                    break
            if expiry is not None:
                weeks.append((date, expiry))

    return weeks


class Backtester:
    def __init__(
        self,
        df: pd.DataFrame,
        regime_series: pd.Series,
        capital: float = 1_000_000,
        risk_per_trade: float = 0.02,
        transaction_cost_per_lot: float = 150,
    ):
        """
        df: daily data with Close, VIX, IV_HV_ratio columns
        regime_series: daily regime labels from RegimeModel
        """
        self.df = df.copy()
        self.df.index = pd.to_datetime(self.df.index)
        self.regime_series = regime_series
        self.scaler = ScalerAgent(capital=capital, risk_per_trade=risk_per_trade)
        self.transaction_cost = transaction_cost_per_lot
        self.trades: List[TradeSetup] = []
        self.equity_curve: List[dict] = []

    def run(self) -> pd.DataFrame:
        schedule = get_weekly_schedule(self.df)
        capital = self.scaler.capital

        for entry_date, expiry_date in schedule:
            if entry_date not in self.df.index or expiry_date not in self.df.index:
                continue

            row = self.df.loc[entry_date]
            spot = float(row["Close"])
            vix = float(row["VIX"])
            iv_hv = float(row.get("IV_HV_ratio", 1.2))
            regime = self.regime_series.get(entry_date, "normal")
            confidence = 0.70  # default; use model proba if available

            dte = (expiry_date - entry_date).days

            setup = select_strategy(
                spot=spot, vix=vix,
                entry_date=str(entry_date.date()),
                expiry_date=str(expiry_date.date()),
                regime=regime, confidence=confidence,
                iv_hv_ratio=iv_hv, dte=max(dte, 1),
            )

            if setup is None:
                self.equity_curve.append({
                    "date": entry_date, "capital": capital,
                    "trade": "skip", "pnl": 0, "regime": regime
                })
                continue

            max_loss = setup.max_loss if setup.max_loss != float("inf") else spot * 0.03
            lots = self.scaler.size(max_loss, iv_hv, confidence, regime)
            if lots == 0:
                continue

            setup.lots = lots
            spot_exit = float(self.df.loc[expiry_date, "Close"])
            pnl_per_lot = payoff_at_expiry(setup, spot_exit)
            total_pnl = pnl_per_lot * lots * LOT_SIZE - self.transaction_cost * lots

            setup.exit_date = str(expiry_date.date())
            setup.spot_exit = spot_exit
            setup.pnl = round(total_pnl, 2)
            setup.pnl_pct = round(total_pnl / capital * 100, 4)
            setup.outcome = "profit" if total_pnl >= 0 else "loss"

            capital += total_pnl
            self.scaler.update_capital(capital)
            self.trades.append(setup)

            self.equity_curve.append({
                "date": entry_date, "capital": capital,
                "trade": setup.strategy, "pnl": total_pnl,
                "regime": regime, "lots": lots, "vix": vix,
                "iv_hv": iv_hv, "spot": spot, "spot_exit": spot_exit,
            })

        return pd.DataFrame(self.equity_curve).set_index("date")
