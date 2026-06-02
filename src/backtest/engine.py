"""
Weekly options backtesting engine for Nifty.
Entry: Monday | Exit: Thursday expiry OR mid-week stop loss
Supports: Iron Condor, Short Straddle, Calendar Spread, Advanced strategies
Optional: ML strategy predictor overrides rule-based selector
"""
import pandas as pd
import numpy as np
from typing import List, Optional

from ..strategy.selector import select_strategy
from ..strategy.strategies import payoff_at_expiry
from ..strategy.calendar_spread import calendar_payoff_at_near_expiry
from ..strategy.pricer import black_scholes
from ..strategy.base import TradeSetup
from ..sizing.scaler import ScalerAgent, LOT_SIZE


def get_weekly_schedule(df: pd.DataFrame) -> List[tuple]:
    """Entry: Monday, Expiry: Thursday. Returns (entry, expiry, next_expiry) triples."""
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    trading_days = df.index.tolist()
    weeks = []

    for i, date in enumerate(trading_days):
        if date.dayofweek != 0:
            continue
        expiry, next_expiry = None, None
        for j in range(i, min(i + 5, len(trading_days))):
            if trading_days[j].dayofweek == 3:
                expiry = trading_days[j]
                break
        if expiry is None:
            continue
        for j in range(trading_days.index(expiry) + 1, min(trading_days.index(expiry) + 6, len(trading_days))):
            if trading_days[j].dayofweek == 3:
                next_expiry = trading_days[j]
                break
        weeks.append((date, expiry, next_expiry))

    return weeks


def _position_pnl_at_date(setup: TradeSetup, spot: float, vix: float, days_remaining: int) -> float:
    """Unrealized P&L using BS mid-week repricing."""
    T = max(days_remaining, 0) / 365
    sigma = vix / 100
    pnl = 0.0
    for strike, opt_type, action, entry_price in setup.legs:
        current = (max(spot - strike, 0) if opt_type == "call" else max(strike - spot, 0)) if T <= 0 \
                   else black_scholes(spot, strike, T, sigma, opt_type)
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
    Check daily closes between entry and expiry.
    Exit early if unrealized loss exceeds stop_multiple × premium collected.
    """
    stop_threshold = -stop_multiple * setup.max_profit
    days_in_trade = df.loc[(df.index > entry_date) & (df.index <= expiry_date)].index

    for day in days_in_trade:
        spot = float(df.loc[day, "Close"])
        vix = float(df.loc[day, "VIX"])
        days_left = max((expiry_date - day).days, 0)
        pnl = _position_pnl_at_date(setup, spot, vix, days_left)
        if pnl <= stop_threshold:
            return pnl, str(day.date()), "stop_loss"

    # Held to expiry
    if setup.strategy == "calendar_spread":
        spot_expiry = float(df.loc[expiry_date, "Close"])
        vix_expiry = float(df.loc[expiry_date, "VIX"])
        pnl = calendar_payoff_at_near_expiry(setup, spot_expiry, vix_expiry)
    else:
        spot_expiry = float(df.loc[expiry_date, "Close"])
        pnl = payoff_at_expiry(setup, spot_expiry)
    return pnl, str(expiry_date.date()), "expiry"


class Backtester:
    def __init__(
        self,
        df: pd.DataFrame,
        regime_series: pd.Series,
        capital: float = 1_000_000,
        risk_per_trade: float = 0.06,
        transaction_cost_per_lot: float = 150,
        stop_loss_multiple: float = 2.0,
        ml_predictor=None,         # StrategyPredictor instance (optional)
        ml_features: pd.DataFrame = None,
        use_calendar: bool = True,
    ):
        self.df = df.copy()
        self.df.index = pd.to_datetime(self.df.index)
        self.regime_series = regime_series
        self.scaler = ScalerAgent(capital=capital, risk_per_trade=risk_per_trade)
        self.transaction_cost = transaction_cost_per_lot
        self.stop_loss_multiple = stop_loss_multiple
        self.ml_predictor = ml_predictor
        self.ml_features = ml_features
        self.use_calendar = use_calendar
        self.trades: List[TradeSetup] = []
        self.equity_curve: List[dict] = []
        self._stop_count = 0
        self._expiry_count = 0
        self._ml_used = 0

    def run(self) -> pd.DataFrame:
        schedule = get_weekly_schedule(self.df)
        capital = self.scaler.capital

        # Precompute ML predictions if predictor is available
        ml_predictions = {}
        if self.ml_predictor and self.ml_predictor.fitted and self.ml_features is not None:
            preds = self.ml_predictor.predict(self.ml_features)
            ml_predictions = preds.to_dict()

        for entry_date, expiry_date, next_expiry_date in schedule:
            if entry_date not in self.df.index or expiry_date not in self.df.index:
                continue

            row = self.df.loc[entry_date]
            spot = float(row["Close"])
            vix = float(row["VIX"])
            iv_hv = float(row.get("IV_HV_ratio", 1.2))
            regime = self.regime_series.get(entry_date, "normal")
            confidence = float(row.get("regime_confidence", 0.70)) if "regime_confidence" in self.df.columns else 0.70
            dte = max((expiry_date - entry_date).days, 1)

            ml_override = ml_predictions.get(entry_date)
            if ml_override:
                self._ml_used += 1

            next_expiry_str = str(next_expiry_date.date()) if next_expiry_date else None

            setup = select_strategy(
                spot=spot, vix=vix,
                entry_date=str(entry_date.date()),
                expiry_date=str(expiry_date.date()),
                regime=regime, confidence=confidence,
                iv_hv_ratio=iv_hv, dte=dte,
                next_expiry_date=next_expiry_str,
                use_calendar=self.use_calendar,
                ml_override=ml_override,
            )

            if setup is None:
                self.equity_curve.append({
                    "date": entry_date, "capital": capital, "trade": "skip",
                    "pnl": 0, "regime": regime, "lots": 0, "vix": vix,
                    "exit_type": "skip", "ml_used": bool(ml_override),
                })
                continue

            # Stop-aware max loss for sizing
            if self.stop_loss_multiple > 0 and setup.max_profit > 0:
                effective_max_loss_pts = self.stop_loss_multiple * setup.max_profit
            elif setup.max_loss not in (float("inf"), 0):
                effective_max_loss_pts = setup.max_loss
            else:
                effective_max_loss_pts = (vix / 100 / np.sqrt(52)) * spot * 2

            lots = self.scaler.size(effective_max_loss_pts, iv_hv, confidence, regime)
            if lots == 0:
                continue

            setup.lots = lots

            pnl_per_lot, exit_date, exit_type = _run_with_stop_loss(
                setup, self.df, entry_date, expiry_date, self.stop_loss_multiple
            ) if self.stop_loss_multiple > 0 else (
                payoff_at_expiry(setup, float(self.df.loc[expiry_date, "Close"])),
                str(expiry_date.date()), "expiry"
            )

            if exit_type == "stop_loss":
                self._stop_count += 1
            else:
                self._expiry_count += 1

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
                "ml_used": bool(ml_override),
            })

        total = self._stop_count + self._expiry_count
        if total > 0:
            print(f"  Stop losses: {self._stop_count}/{total} ({self._stop_count/total*100:.0f}%)"
                  + (f"  |  ML overrides: {self._ml_used}" if self._ml_used else ""))
        return pd.DataFrame(self.equity_curve).set_index("date")
