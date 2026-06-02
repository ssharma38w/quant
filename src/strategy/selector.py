"""
Adaptive strategy selector based on VIX zone + regime + IV/HV premium.

VIX zones (informed by real backtest results):
  < 12         → Short Straddle (ultra-low vol, max premium capture)
  12–16        → Short Straddle (low vol regime confirmed)
  16–21        → Short Strangle @ 0.8 SD (normal, some buffer)
  21–26        → Iron Condor, 400-pt wings (elevated, defined risk)
  > 26         → Skip (too much tail risk for premium selling)
"""
from .strategies import short_straddle, short_strangle, iron_condor, TradeSetup


def select_strategy(
    spot: float,
    vix: float,
    entry_date: str,
    expiry_date: str,
    regime: str,
    confidence: float,
    iv_hv_ratio: float,
    dte: int = 5,
) -> TradeSetup | None:
    """
    Returns a TradeSetup or None (skip this week).

    Skip conditions:
    - IV/HV < 1.05 → not enough premium over realized vol
    - VIX > 26 → tail risk too high for naked/semi-naked strategies
    """
    if iv_hv_ratio < 1.05:
        return None

    if vix > 26:
        return None

    kwargs = dict(
        spot=spot, vix=vix, entry_date=entry_date, expiry_date=expiry_date,
        regime=regime, confidence=confidence, dte=dte,
    )

    # VIX-zone based selection (overrides HMM regime when VIX is clear)
    if vix <= 16:
        return short_straddle(**kwargs)

    if vix <= 21:
        # Low confidence → widen to strangle for safety
        sd = 0.7 if confidence >= 0.70 else 0.9
        return short_strangle(**kwargs, sd_multiple=sd)

    # VIX 21–26: Iron Condor, wider wings for elevated vol
    wing = 400 if vix > 23 else 300
    return iron_condor(**kwargs, short_sd=0.8, wing_width=wing)
