"""
Theta-based early exit manager.

Instead of only using a stop loss, we also exit early when we've collected
a target percentage of the maximum possible theta/premium.

Rationale:
  - If we've captured 65-70% of max profit by Wednesday, the remaining 30-35%
    isn't worth the overnight gamma risk into Thursday expiry.
  - Exit early = lock in profit, redeploy capital next week.

Also tracks vega risk: if VIX spikes > spike_threshold points mid-week,
alert to consider closing (vega losses can wipe theta gains quickly).
"""
import numpy as np
from .calculator import position_greeks, _bs_greeks_single


def check_early_exit(
    legs: list,
    spot: float,
    vix: float,
    days_remaining: int,
    entry_premium: float,       # total premium collected at entry (per lot)
    theta_target_pct: float = 0.65,   # exit when unrealized P&L >= 65% of entry premium
    gamma_danger_dte: int = 1,        # flag if DTE ≤ this and position tested
    delta_threshold: float = 0.35,    # flag if |delta| > this
) -> dict:
    """
    Evaluate whether to exit a position early based on greeks.

    Returns:
      {
        "exit": True/False,
        "reason": "theta_target" | "delta_risk" | "gamma_danger" | None,
        "unrealized_pnl": float,
        "theta_captured_pct": float,
        "current_delta": float,
      }
    """
    T = max(days_remaining, 0) / 365
    sigma = vix / 100

    # Current position value (unrealized P&L)
    unreal = 0.0
    for strike, opt_type, action, entry_price in legs:
        g = _bs_greeks_single(spot, strike, T, sigma, opt_type, action)
        cur_price = g["price"]
        unreal += (entry_price - cur_price) if action == "sell" else (cur_price - entry_price)

    theta_captured_pct = (unreal / entry_premium * 100) if entry_premium > 0 else 0

    g_pos = position_greeks(legs, spot, vix, days_remaining)
    current_delta = g_pos["delta"]

    # Early exit conditions
    if theta_captured_pct >= theta_target_pct * 100:
        return {
            "exit": True,
            "reason": "theta_target",
            "unrealized_pnl": round(unreal, 2),
            "theta_captured_pct": round(theta_captured_pct, 1),
            "current_delta": round(current_delta, 4),
        }

    if days_remaining <= gamma_danger_dte and abs(current_delta) > delta_threshold:
        return {
            "exit": True,
            "reason": "gamma_danger",
            "unrealized_pnl": round(unreal, 2),
            "theta_captured_pct": round(theta_captured_pct, 1),
            "current_delta": round(current_delta, 4),
        }

    if abs(current_delta) > 0.45:
        return {
            "exit": True,
            "reason": "delta_breach",
            "unrealized_pnl": round(unreal, 2),
            "theta_captured_pct": round(theta_captured_pct, 1),
            "current_delta": round(current_delta, 4),
        }

    return {
        "exit": False,
        "reason": None,
        "unrealized_pnl": round(unreal, 2),
        "theta_captured_pct": round(theta_captured_pct, 1),
        "current_delta": round(current_delta, 4),
    }


def vega_risk_alert(
    vix_entry: float,
    vix_current: float,
    vega_inr_per_lot: float,
    lots: int,
    spike_threshold: float = 2.5,
) -> dict:
    """
    Alert when VIX has spiked significantly, threatening vega losses.
    vega_inr_per_lot: position vega in INR per 1% VIX move per lot
    """
    vix_move = vix_current - vix_entry
    vega_pnl_estimate = vega_inr_per_lot * lots * vix_move  # negative if short vega + VIX up

    alert = abs(vix_move) >= spike_threshold
    return {
        "alert": alert,
        "vix_move": round(vix_move, 2),
        "vega_pnl_estimate_inr": round(vega_pnl_estimate, 0),
        "action": "consider closing or adding vega hedge" if alert and vix_move > 0 else None,
    }
