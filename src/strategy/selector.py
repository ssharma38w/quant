"""
Adaptive strategy selector — regime-first, VIX for parameter tuning.

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
"""
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
    ml_override: str = None,       # strategy name from ML predictor
) -> TradeSetup | None:

    if iv_hv_ratio < 1.05:
        return None
    if vix > 26:
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
        wing = _ic_wing_width(vix, regime)
        return iron_condor(**kwargs, short_sd=_ic_sd(vix), wing_width=wing)

    # Rule-based fallback
    if regime == "low_vol" and vix <= 15 and iv_hv_ratio >= 1.1:
        return short_straddle(**kwargs)

    wing = _ic_wing_width(vix, regime)
    return iron_condor(**kwargs, short_sd=_ic_sd(vix), wing_width=wing)


def _ic_wing_width(vix: float, regime: str) -> int:
    if regime == "high_vol" or vix > 22:
        return 450
    if regime == "normal" or vix > 17:
        return 350
    return 300


def _ic_sd(vix: float) -> float:
    """Place short strikes wider when VIX is elevated."""
    if vix > 22:
        return 1.0
    if vix > 17:
        return 0.85
    return 0.75
