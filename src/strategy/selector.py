"""
Adaptive strategy selector: picks the best strategy each week
based on regime, IV/HV premium, and VIX level.
"""
from .strategies import short_straddle, short_strangle, iron_condor, TradeSetup


STRATEGY_RULES = {
    # (regime, vix_level, iv_hv_ratio) → strategy, params
    # Low vol: aggressive premium selling
    "low_vol": {
        "strategy": "short_straddle",
        "sd_multiple": None,
        "wing_width": None,
    },
    # Normal: sell 1SD strangle
    "normal": {
        "strategy": "short_strangle",
        "sd_multiple": 1.0,
        "wing_width": None,
    },
    # High vol: iron condor (defined risk), wider strikes
    "high_vol": {
        "strategy": "iron_condor",
        "sd_multiple": 0.8,
        "wing_width": 300,
    },
}


def select_strategy(
    spot: float,
    vix: float,
    entry_date: str,
    expiry_date: str,
    regime: str,
    confidence: float,
    iv_hv_ratio: float,
    dte: int = 5,
) -> TradeSetup:
    """
    Select and build a strategy based on market conditions.

    Override rules:
    - If IV/HV < 1.0: IV is cheap → skip selling premium (return None)
    - If VIX > 25 and regime != high_vol: force iron_condor for safety
    - If confidence < 0.55: default to iron_condor (safer)
    """
    if iv_hv_ratio < 1.0:
        return None  # No edge — IV not rich enough to sell

    effective_regime = regime
    if vix > 25:
        effective_regime = "high_vol"
    if confidence < 0.55:
        effective_regime = "high_vol"  # Uncertain regime → go safe

    rule = STRATEGY_RULES.get(effective_regime, STRATEGY_RULES["normal"])
    strat = rule["strategy"]

    kwargs = dict(
        spot=spot, vix=vix, entry_date=entry_date, expiry_date=expiry_date,
        regime=effective_regime, confidence=confidence, dte=dte
    )

    if strat == "short_straddle":
        return short_straddle(**kwargs)
    elif strat == "short_strangle":
        return short_strangle(**kwargs, sd_multiple=rule["sd_multiple"])
    elif strat == "iron_condor":
        return iron_condor(**kwargs, short_sd=rule["sd_multiple"], wing_width=rule["wing_width"])
    return None
