"""
Weekly options backtesting engine for Nifty.
Entry: Monday of each week
Exit: Thursday (expiry) OR mid-week stop loss (daily close check)
Pricing: Black-Scholes using VIX as IV proxy
"""
import pandas as pd
import numpy as np
from typing import List

from ..strategy.selector import select_strategy
from ..strategy.strategies import payoff_at_expiry
from ..strategy.pricer import black_scholes
from ..strategy.base import TradeSetup
from ..sizing.scaler import ScalerAgent, LOT_SIZE


def get_weekly_schedule(df: pd.DataFrame) -> List[tuple]:
    """
    Returns list of (entry_date, expiry_date) pairs.
    Entry: Monday (or first trading day of week)
    Expiry: Thursday
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    weeks = []
    trading_days = df.index.tolist()

    for i, date in enumerate(trading_days):
        if date.dayofweek == 0:  # Monday
            expiry = None
            for j in range(i, min(i + 5, len(trading_days))):
                if trading_days[j].dayofweek == 3:
                    expiry = trading_days[j]
                    break
            if expiry is not None:
                weeks.append((date, expiry))

    return weeks


def _position_pnl_at_date(
    setup: TradeSetup, spot: float, vix: float, days_remaining: int
) -> float:
    """Current unrealized P&L of the position using BS pricing."""
    T = max(days_remaining, 0) / 365
    sigma = vix / 100
    pnl = 0.0
    for strike, opt_type, action, entry_price in setup.legs:
        if T <= 0:
            current = max(spot - strike, 0) if opt_type == "call" else max(strike - spot, 0)
        else:
            current = black_scholes(spot, strike, T, sigma, opt_type)
        pnl += (entry_price - current) if action == "sell" else (current - entry_price)
    return round(pnl, 2)


def _run_with_stop_loss(
    setup: TradeSetup,
    df: pd.DataFrame,
    entry_date: pd.Timestamp,
    expiry_date: pd.Timestamp,
    stop_multiple: float,
) -> tuple:
    """
    Monitor daily closes between entry and expiry.
    Exit early if unrealized loss exceeds stop_multiple × premium collected.
    Returns: (pnl_per_lot, exit_date_str, exit_type)
    """
    stop_threshold = -stop_multiple * setup.max_profit  # e.g. -2 × premium

    days_in_trade = df.loc[
        (df.index > entry_date) & (df.index <= expiry_date)
    ].index

    for day in days_in_trade:
        spot = float(df.loc[day, "Close"])
        vix = float(df.loc[day, "VIX"])
        days_left = max((expiry_date - day).days, 0)
        pnl = _position_pnl_at_date(setup, spot, vix, days_left)

        if pnl <= stop_threshold:
            return pnl, str(day.date()), "stop_loss"

    # No stop triggered — settle at expiry intrinsic
    spot_expiry = float(df.loc[expiry_date, "Close"])
    pnl_expiry = payoff_at_expiry(setup, spot_expiry)
    return pnl_expiry, str(expiry_date.date()), "expiry"


class Backtester:
    def __init__(
        self,
        df: pd.DataFrame,
        regime_series: pd.Series,
        capital: float = 1_000_000,
        risk_per_trade: float = 0.06,
        transaction_cost_per_lot: float = 150,
        stop_loss_multiple: float = 2.0,
    ):
        """
        df: daily data with Close, VIX, IV_HV_ratio columns
        regime_series: daily regime labels from RegimeModel
        stop_loss_multiple: exit if loss > this × premium collected (0 = hold to expiry)
        """
        self.df = df.copy()
        self.df.index = pd.to_datetime(self.df.index)
        self.regime_series = regime_series
        self.scaler = ScalerAgent(capital=capital, risk_per_trade=risk_per_trade)
        self.transaction_cost = transaction_cost_per_lot
        self.stop_loss_multiple = stop_loss_multiple
        self.trades: List[TradeSetup] = []
        self.equity_curve: List[dict] = []
        self._stop_loss_count = 0
        self._expiry_count = 0

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
            confidence = 0.70
            dte = max((expiry_date - entry_date).days, 1)

            setup = select_strategy(
                spot=spot, vix=vix,
                entry_date=str(entry_date.date()),
                expiry_date=str(expiry_date.date()),
                regime=regime, confidence=confidence,
                iv_hv_ratio=iv_hv, dte=dte,
            )

            if setup is None:
                self.equity_curve.append({
                    "date": entry_date, "capital": capital,
                    "trade": "skip", "pnl": 0, "regime": regime,
                    "lots": 0, "vix": vix, "exit_type": "skip",
                })
                continue

            # Effective max loss for sizing includes stop loss
            if self.stop_loss_multiple > 0 and setup.max_profit > 0:
                # Stop fires at (stop_multiple × premium) loss
                effective_max_loss_pts = self.stop_loss_multiple * setup.max_profit
            elif setup.max_loss != float("inf"):
                effective_max_loss_pts = setup.max_loss
            else:
                effective_max_loss_pts = (vix / 100 / np.sqrt(52)) * spot * 2

            lots = self.scaler.size(effective_max_loss_pts, iv_hv, confidence, regime)
            if lots == 0:
                continue

            setup.lots = lots

            if self.stop_loss_multiple > 0:
                pnl_per_lot, exit_date, exit_type = _run_with_stop_loss(
                    setup, self.df, entry_date, expiry_date, self.stop_loss_multiple
                )
                if exit_type == "stop_loss":
                    self._stop_loss_count += 1
                else:
                    self._expiry_count += 1
            else:
                spot_expiry = float(self.df.loc[expiry_date, "Close"])
                pnl_per_lot = payoff_at_expiry(setup, spot_expiry)
                exit_date = str(expiry_date.date())
                exit_type = "expiry"

            total_pnl = pnl_per_lot * lots * LOT_SIZE - self.transaction_cost * lots

            setup.exit_date = exit_date
            setup.spot_exit = float(self.df.loc[expiry_date, "Close"])
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
                "iv_hv": iv_hv, "spot": spot,
                "spot_exit": setup.spot_exit, "exit_type": exit_type,
            })

        total = self._stop_loss_count + self._expiry_count
        if total > 0:
            print(f"  Stop losses triggered: {self._stop_loss_count}/{total} "
                  f"({self._stop_loss_count/total*100:.0f}% of trades)")
        return pd.DataFrame(self.equity_curve).set_index("date")
