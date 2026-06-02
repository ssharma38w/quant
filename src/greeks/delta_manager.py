"""
Delta Neutral Manager — simulates dynamic delta hedging using Nifty Futures.

Logic:
  - Check position delta daily after close
  - If |position_delta × lots| > delta_threshold: execute hedge via futures
  - Hedge: buy/sell Nifty futures to bring net delta back to ~0
  - Futures P&L tracked separately and added to total trade P&L

Nifty Futures specs:
  - 1 futures contract = 75 units (same as options lot)
  - Margin: ~₹1.2L per lot (not modelled here — assume sufficient margin)
  - Transaction cost: 0.01% of notional per hedge trade
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List

from .calculator import position_greeks, _bs_greeks_single

LOT_SIZE = 75
FUTURES_TXN_COST_PCT = 0.0001  # 0.01% of notional per leg


@dataclass
class HedgeTrade:
    date: str
    action: str        # "buy" or "sell"
    quantity: int      # in lots (negative = short)
    price: float
    pnl: float = 0.0   # filled when closed


@dataclass
class DeltaHedgeResult:
    total_hedge_pnl: float = 0.0
    total_hedge_cost: float = 0.0
    hedge_trades: List[HedgeTrade] = field(default_factory=list)
    daily_delta_log: List[dict] = field(default_factory=list)


def run_delta_hedge(
    legs: list,
    df: pd.DataFrame,
    entry_date,
    expiry_date,
    lots: int,
    delta_threshold: float = 0.25,  # hedge when |net delta| > 0.25 per lot
    hedge_band: float = 0.10,       # re-hedge band (don't over-trade)
) -> DeltaHedgeResult:
    """
    Simulate delta hedging for a position over its life.

    delta_threshold: if |position_delta_per_lot| > this → hedge
    hedge_band: smaller adjustments within threshold are ignored (reduces over-hedging)

    Returns DeltaHedgeResult with cumulative hedge P&L and all hedge trades.
    """
    result = DeltaHedgeResult()
    open_hedges: List[HedgeTrade] = []  # open futures positions
    net_hedge_lots = 0  # current futures position in lots (+ = long, - = short)

    days_in_trade = df.loc[
        (df.index >= entry_date) & (df.index <= expiry_date)
    ]

    prev_spot = None

    for date, row in days_in_trade.iterrows():
        spot = float(row["Close"])
        vix  = float(row["VIX"])
        days_left = max((expiry_date - date).days, 0)

        # Update open hedge P&L
        if prev_spot is not None:
            spot_move = spot - prev_spot
            result.total_hedge_pnl += net_hedge_lots * LOT_SIZE * spot_move

        # Calculate current position delta
        g = position_greeks(legs, spot, vix, days_left)
        position_delta_net = g["delta"] * lots  # total delta across all lots

        result.daily_delta_log.append({
            "date": str(date.date()),
            "spot": spot,
            "position_delta": round(position_delta_net, 4),
            "hedge_lots": net_hedge_lots,
            "net_delta": round(position_delta_net + net_hedge_lots, 4),
        })

        # Hedge decision
        net_delta = position_delta_net + net_hedge_lots
        if abs(net_delta) > delta_threshold:
            # Hedge to bring net delta back to 0
            lots_to_hedge = -round(net_delta)  # negative = sell futures if delta positive

            if abs(lots_to_hedge) >= 1:
                action = "sell" if lots_to_hedge < 0 else "buy"
                notional = abs(lots_to_hedge) * LOT_SIZE * spot
                txn_cost = notional * FUTURES_TXN_COST_PCT

                hedge = HedgeTrade(
                    date=str(date.date()),
                    action=action,
                    quantity=lots_to_hedge,
                    price=spot,
                )
                open_hedges.append(hedge)
                result.hedge_trades.append(hedge)
                result.total_hedge_cost += txn_cost
                net_hedge_lots += lots_to_hedge

        prev_spot = spot

    # Close all open hedges at expiry
    if open_hedges and prev_spot is not None:
        expiry_spot = float(df.loc[expiry_date, "Close"]) if expiry_date in df.index else prev_spot
        for hedge in open_hedges:
            close_pnl = hedge.quantity * LOT_SIZE * (expiry_spot - hedge.price)
            hedge.pnl = close_pnl
            notional = abs(hedge.quantity) * LOT_SIZE * expiry_spot
            result.total_hedge_cost += notional * FUTURES_TXN_COST_PCT

    result.total_hedge_pnl -= result.total_hedge_cost
    return result


def delta_summary(result: DeltaHedgeResult) -> dict:
    n_hedges = len(result.hedge_trades)
    return {
        "hedge_trades": n_hedges,
        "hedge_pnl_inr": round(result.total_hedge_pnl, 2),
        "hedge_cost_inr": round(result.total_hedge_cost, 2),
        "max_delta_exposure": max(
            (abs(d["net_delta"]) for d in result.daily_delta_log), default=0
        ),
    }
