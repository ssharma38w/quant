"""
Adaptive strategy selector — regime-first, VIX for parameter tuning.
Optionally uses TimesFM predicted move to scale IC wing width.

Lessons from real backtest (2019-2025):
  - normal/high_vol + straddle/strangle → consistent losses
  - iron_condor works across ALL regimes
  - straddle only profitable in confirmed low_vol + VIX ≤ 15
  - strangle never outperformed IC consistently → removed

Decision matrix:
  low_vol  + VIX ≤ 15 + IV/HV ≥ 1.1  → Short Straddle
  low_vol  + VIX 15-20                 → Iron Condor (300 wing)
  normal                               → Iron Condor (350 wing)
  high_vol + VIX 20-26                 → Iron Condor (450 wing)
  any      + VIX > 26                  → Skip (tail risk too high)
  any      + IV/HV < 1.05              → Skip (no premium edge)
  any      + calendar_mode             → Calendar Spread (low vol, range-bound)

TimesFM influence (when predicted_move_pts is provided):
  range_vs_vix < 0.75  → model predicts quiet week, tighten wings 15%
  range_vs_vix 0.75-1.25 → use VIX-based wings (no adjustment)
  range_vs_vix > 1.25  → model predicts wide move, widen wings 20%
  range_vs_vix > 1.60  → skip (TimesFM disagrees strongly with VIX)
"""
import numpy as np
from .strategies import short_straddle, iron_condor, TradeSetup
from .calendar_spread import calendar_spread


def select_strategy(
    spot: float,
    vix: float,
    entry_date: str,
    expiry_date: str,
    regime: str,
    confidence: float,
    iv_hv_ratio: float,
    dte: int = 5,
    next_expiry_date: str = None,
    use_calendar: bool = False,
    ml_override: str = None,        # strategy name from ML predictor
    predicted_move_pts: float = None,  # TimesFM predicted 1-SD move in points
) -> TradeSetup | None:

    if iv_hv_ratio < 1.05:
        return None
    if vix > 26:
        return None

    # Compute TimesFM range_vs_vix ratio for adjusting wing width
    vix_move = (vix / 100) / np.sqrt(52) * spot
    range_vs_vix = 1.0
    if predicted_move_pts is not None and vix_move > 0:
        range_vs_vix = predicted_move_pts / vix_move
        # TimesFM strongly disagrees with VIX → skip (high uncertainty)
        if range_vs_vix > 1.6:
            return None

    # ML predictor can override the rule-based selection
    if ml_override == "skip":
        return None
    if ml_override == "calendar" and next_expiry_date and use_calendar:
        return calendar_spread(
            spot=spot, vix=vix, entry_date=entry_date,
            near_expiry=expiry_date, far_expiry=next_expiry_date,
            regime=regime, confidence=confidence, dte=dte,
        )

    kwargs = dict(
        spot=spot, vix=vix, entry_date=entry_date, expiry_date=expiry_date,
        regime=regime, confidence=confidence, dte=dte,
    )

    # Calendar spread: low vol, high confidence, range-bound signal
    if use_calendar and next_expiry_date and regime == "low_vol" and vix <= 14 and confidence >= 0.80:
        return calendar_spread(
            spot=spot, vix=vix, entry_date=entry_date,
            near_expiry=expiry_date, far_expiry=next_expiry_date,
            regime=regime, confidence=confidence, dte=dte,
        )

    # ML override for named strategies
    if ml_override == "short_straddle":
        return short_straddle(**kwargs)
    if ml_override == "iron_condor":
        wing = _adjusted_wing(vix, regime, range_vs_vix)
        return iron_condor(**kwargs, short_sd=_ic_sd(vix), wing_width=wing)

    # Rule-based: straddle only in confirmed low-vol environment
    if regime == "low_vol" and vix <= 15 and iv_hv_ratio >= 1.1:
        # If TimesFM predicts a big move, fall back to IC instead of straddle
        if range_vs_vix > 1.25:
            wing = _adjusted_wing(vix, regime, range_vs_vix)
            return iron_condor(**kwargs, short_sd=_ic_sd(vix), wing_width=wing)
        return short_straddle(**kwargs)

    wing = _adjusted_wing(vix, regime, range_vs_vix)
    return iron_condor(**kwargs, short_sd=_ic_sd(vix, range_vs_vix), wing_width=wing)


def _ic_wing_width(vix: float, regime: str) -> int:
    """Base wing width from VIX / regime."""
    if regime == "high_vol" or vix > 22:
        return 450
    if regime == "normal" or vix > 17:
        return 350
    return 300


def _adjusted_wing(vix: float, regime: str, range_vs_vix: float) -> int:
    """Scale wing width by TimesFM range_vs_vix ratio."""
    base = _ic_wing_width(vix, regime)
    if range_vs_vix < 0.75:
        # Quiet week forecast → tighter wings collect more premium
        return max(int(base * 0.85), 150)
    if range_vs_vix > 1.25:
        # Wide move forecast → wider wings for safety
        return min(int(base * 1.20), 600)
    return base


def _ic_sd(vix: float, range_vs_vix: float = 1.0) -> float:
    """
    Place short strikes further OTM when VIX elevated or TimesFM warns of wide move.
    range_vs_vix > 1.15 nudges strikes slightly wider.
    """
    if vix > 22:
        base = 1.0
    elif vix > 17:
        base = 0.85
    else:
        base = 0.75
    # Nudge OTM when TimesFM forecasts a wide week
    if range_vs_vix > 1.15:
        base = min(base * 1.08, 1.2)
    return base
