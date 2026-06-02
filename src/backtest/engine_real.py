"""
Real-data backtester: uses actual NSE bhavcopy options prices.
Falls back to Black-Scholes if bhavcopy data is unavailable for a date.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List

from ..strategy.selector import select_strategy
from ..strategy.strategies import payoff_at_expiry, TradeSetup
from ..strategy.pricer import black_scholes, round_to_strike
from ..sizing.scaler import ScalerAgent, LOT_SIZE
from ..data.nse_options import (
    lookup_option_price,
    get_weekly_expiry,
    fetch_options_range,
)
from .engine import get_weekly_schedule


def _reprice_legs_with_real_data(
    setup: TradeSetup,
    entry_bhavcopy: pd.DataFrame,
    expiry_dt: datetime,
) -> TradeSetup:
    """
    Replace Black-Scholes prices in legs with actual NSE market prices.
    If a real price isn't found, keeps the BS price.
    """
    new_legs = []
    for strike, opt_type, action, bs_price in setup.legs:
        nse_type = "CE" if opt_type == "call" else "PE"
        real_price = lookup_option_price(entry_bhavcopy, expiry_dt, int(strike), nse_type)
        price = real_price if real_price is not None else bs_price
        new_legs.append((strike, opt_type, action, round(price, 2)))
    setup.legs = new_legs

    # Recalculate max_profit / max_loss from real prices
    net = sum(p if a == "sell" else -p for _, _, a, p in setup.legs)
    setup.max_profit = round(net, 2)

    wing_width = None
    strikes = [s for s, _, a, _ in setup.legs if a == "buy"]
    sell_strikes = [s for s, _, a, _ in setup.legs if a == "sell"]
    if strikes and sell_strikes:
        wing_width = abs(strikes[0] - sell_strikes[0]) if len(strikes) == len(sell_strikes) else None

    if wing_width and setup.strategy == "iron_condor":
        setup.max_loss = round(wing_width - net, 2)

    return setup


def _expiry_pnl_from_bhavcopy(
    setup: TradeSetup,
    expiry_bhavcopy: pd.DataFrame,
    expiry_dt: datetime,
    spot_exit: float,
) -> float:
    """
    Calculate P&L at expiry using settlement prices from bhavcopy.
    For expired options: use intrinsic value (they settle at intrinsic).
    For mid-week exit: use SETTLE_PR from bhavcopy.
    """
    pnl = 0.0
    for strike, opt_type, action, entry_price in setup.legs:
        # At expiry, settlement = intrinsic value
        intrinsic = max(spot_exit - strike, 0) if opt_type == "call" else max(strike - spot_exit, 0)

        if expiry_bhavcopy is not None:
            nse_type = "CE" if opt_type == "call" else "PE"
            settle = lookup_option_price(expiry_bhavcopy, expiry_dt, int(strike), nse_type, use_settle=True)
            exit_price = settle if settle is not None else intrinsic
        else:
            exit_price = intrinsic

        if action == "sell":
            pnl += entry_price - exit_price
        else:
            pnl += exit_price - entry_price

    return round(pnl, 2)


class RealDataBacktester:
    """
    Backtester using actual NSE F&O bhavcopy prices.
    Usage:
        options_data = fetch_options_range("2019-01-01", "2024-12-31")
        bt = RealDataBacktester(df, regimes, options_data)
        equity = bt.run()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        regime_series: pd.Series,
        options_data: dict,
        capital: float = 1_000_000,
        risk_per_trade: float = 0.02,
        transaction_cost_per_lot: float = 150,
    ):
        self.df = df.copy()
        self.df.index = pd.to_datetime(self.df.index)
        self.regime_series = regime_series
        self.options_data = options_data  # {date_str: DataFrame}
        self.scaler = ScalerAgent(capital=capital, risk_per_trade=risk_per_trade)
        self.txn_cost = transaction_cost_per_lot
        self.trades: List[TradeSetup] = []
        self.equity_curve: List[dict] = []
        self._bs_fallbacks = 0
        self._real_prices = 0

    def _get_bhavcopy(self, date: datetime) -> pd.DataFrame | None:
        return self.options_data.get(str(date.date()))

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
                    "data_source": "skip",
                })
                continue

            # Try to reprice with real NSE data
            entry_bhav = self._get_bhavcopy(entry_date)
            expiry_bhav = self._get_bhavcopy(expiry_date)

            if entry_bhav is not None:
                setup = _reprice_legs_with_real_data(setup, entry_bhav, expiry_date)
                self._real_prices += 1
                data_source = "nse_real"
            else:
                self._bs_fallbacks += 1
                data_source = "bs_synthetic"

            max_loss = setup.max_loss if setup.max_loss not in (float("inf"), 0) else spot * 0.03
            lots = self.scaler.size(max_loss, iv_hv, confidence, regime)
            if lots == 0:
                continue

            setup.lots = lots
            spot_exit = float(self.df.loc[expiry_date, "Close"])

            pnl_per_lot = _expiry_pnl_from_bhavcopy(setup, expiry_bhav, expiry_date, spot_exit)
            total_pnl = pnl_per_lot * lots * LOT_SIZE - self.txn_cost * lots

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
                "data_source": data_source,
            })

        ec = pd.DataFrame(self.equity_curve).set_index("date")
        real_pct = self._real_prices / max(self._real_prices + self._bs_fallbacks, 1) * 100
        print(f"  Real NSE prices: {self._real_prices} weeks ({real_pct:.0f}%),  BS fallback: {self._bs_fallbacks} weeks")
        return ec
