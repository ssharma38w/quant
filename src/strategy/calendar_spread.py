"""
Calendar Spread (Double Calendar) for Nifty weekly options.

Structure:
  - SELL near-term ATM straddle (current week, expires Thursday)
  - BUY  far-term ATM straddle  (next week, expires next Thursday)

Edge:
  - Near-term theta decays faster → we collect more decay per day on short leg
  - Net credit if near-term premium > far-term premium (unusual, needs VIX curve)
  - Net debit if far-term > near-term (standard) — profits if spot stays pinned to ATM

Payoff at near-term expiry:
  - Short near-term settled at intrinsic
  - Long far-term: still has ~7 days, valued at BS(spot, strike, 7/365, vix_expiry)
  - Close entire position at near expiry
"""
import numpy as np
from .pricer import black_scholes, round_to_strike
from .base import TradeSetup


def calendar_spread(
    spot: float,
    vix: float,
    entry_date: str,
    near_expiry: str,       # current week Thursday
    far_expiry: str,        # next week Thursday
    regime: str = "low_vol",
    confidence: float = 0.8,
    dte: int = 5,           # days to near expiry
) -> TradeSetup:
    """
    Build an ATM double calendar spread.
    Long vega, short theta (net debit) — profits from spot staying near ATM.
    """
    sigma = vix / 100
    T_near = max(dte, 1) / 365
    T_far = max(dte + 7, 8) / 365

    strike = round_to_strike(spot)

    # Near-term (SELL)
    near_call = black_scholes(spot, strike, T_near, sigma, "call")
    near_put = black_scholes(spot, strike, T_near, sigma, "put")

    # Far-term (BUY) — same strike, one week further out
    far_call = black_scholes(spot, strike, T_far, sigma, "call")
    far_put = black_scholes(spot, strike, T_far, sigma, "put")

    # Net position cost (debit = positive cost to enter)
    net_debit = (far_call + far_put) - (near_call + near_put)

    # Theoretical max profit: near expiry at exactly ATM, far still has full time value
    # Approx: far_value_at_expiry ≈ BS(spot, strike, 7/365, sigma)
    # Net P&L ≈ near_premium_collected - far_time_decay
    max_profit_approx = max(near_call + near_put - net_debit * 0.5, near_call + near_put * 0.3)
    max_loss = net_debit if net_debit > 0 else near_call + near_put  # max loss = debit paid

    setup = TradeSetup(
        strategy="calendar_spread",
        entry_date=entry_date,
        expiry_date=near_expiry,
        spot_entry=spot,
        vix_entry=vix,
        regime=regime,
        regime_confidence=confidence,
        legs=[
            # Near-term (sell)
            (strike, "call", "sell", round(near_call, 2)),
            (strike, "put",  "sell", round(near_put,  2)),
            # Far-term (buy)
            (strike, "call", "buy",  round(far_call,  2)),
            (strike, "put",  "buy",  round(far_put,   2)),
        ],
        max_profit=round(max_profit_approx, 2),
        max_loss=round(max_loss, 2),
        breakeven_lower=round(strike - max_profit_approx, 2),
        breakeven_upper=round(strike + max_profit_approx, 2),
    )
    return setup


def calendar_payoff_at_near_expiry(
    setup: TradeSetup,
    spot_expiry: float,
    vix_expiry: float,
    dte_far_remaining: int = 7,
) -> float:
    """
    P&L when near-term leg expires.
    Near-term: settles at intrinsic.
    Far-term: closed at BS value (still has 7 days).
    """
    sigma = vix_expiry / 100
    T_far = max(dte_far_remaining, 1) / 365
    pnl = 0.0

    for strike, opt_type, action, entry_price in setup.legs:
        is_near = action in ("sell",)  # near legs are sold
        is_far = action in ("buy",)    # far legs are bought

        intrinsic = max(spot_expiry - strike, 0) if opt_type == "call" else max(strike - spot_expiry, 0)

        if is_near:
            # Near sold: collect entry_price, pay intrinsic at expiry
            exit_price = intrinsic
        else:
            # Far bought: paid entry_price, close at current BS value
            exit_price = black_scholes(spot_expiry, strike, T_far, sigma, opt_type)

        pnl += (entry_price - exit_price) if action == "sell" else (exit_price - entry_price)

    return round(pnl, 2)
