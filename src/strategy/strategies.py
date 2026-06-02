"""
Iron Condor, Short Straddle, Short Strangle strategy builders.
All return a TradeSetup with legs and payoff parameters.
"""
import numpy as np
from .pricer import black_scholes, round_to_strike
from .base import TradeSetup


DAYS_TO_EXPIRY = 5  # weekly options: ~5 calendar days


def _dte_years(dte_days: int = DAYS_TO_EXPIRY) -> float:
    return dte_days / 365


def short_straddle(
    spot: float, vix: float, entry_date: str, expiry_date: str,
    regime: str = "normal", confidence: float = 0.7, dte: int = DAYS_TO_EXPIRY
) -> TradeSetup:
    T = _dte_years(dte)
    sigma = vix / 100
    atm = round_to_strike(spot)

    call_p = black_scholes(spot, atm, T, sigma, "call")
    put_p = black_scholes(spot, atm, T, sigma, "put")
    premium = call_p + put_p

    setup = TradeSetup(
        strategy="short_straddle",
        entry_date=entry_date,
        expiry_date=expiry_date,
        spot_entry=spot,
        vix_entry=vix,
        regime=regime,
        regime_confidence=confidence,
        legs=[
            (atm, "call", "sell", round(call_p, 2)),
            (atm, "put", "sell", round(put_p, 2)),
        ],
        max_profit=round(premium, 2),
        max_loss=float("inf"),
        breakeven_lower=round(atm - premium, 2),
        breakeven_upper=round(atm + premium, 2),
    )
    return setup


def short_strangle(
    spot: float, vix: float, entry_date: str, expiry_date: str,
    regime: str = "normal", confidence: float = 0.7,
    sd_multiple: float = 1.0, dte: int = DAYS_TO_EXPIRY
) -> TradeSetup:
    T = _dte_years(dte)
    sigma = vix / 100
    weekly_move = (sigma / np.sqrt(52)) * spot * sd_multiple

    call_strike = round_to_strike(spot + weekly_move)
    put_strike = round_to_strike(spot - weekly_move)

    call_p = black_scholes(spot, call_strike, T, sigma, "call")
    put_p = black_scholes(spot, put_strike, T, sigma, "put")
    premium = call_p + put_p

    setup = TradeSetup(
        strategy="short_strangle",
        entry_date=entry_date,
        expiry_date=expiry_date,
        spot_entry=spot,
        vix_entry=vix,
        regime=regime,
        regime_confidence=confidence,
        legs=[
            (call_strike, "call", "sell", round(call_p, 2)),
            (put_strike, "put", "sell", round(put_p, 2)),
        ],
        max_profit=round(premium, 2),
        max_loss=float("inf"),
        breakeven_lower=round(put_strike - premium, 2),
        breakeven_upper=round(call_strike + premium, 2),
    )
    return setup


def iron_condor(
    spot: float, vix: float, entry_date: str, expiry_date: str,
    regime: str = "normal", confidence: float = 0.7,
    short_sd: float = 1.0, wing_width: int = 200, dte: int = DAYS_TO_EXPIRY
) -> TradeSetup:
    T = _dte_years(dte)
    sigma = vix / 100
    weekly_move = (sigma / np.sqrt(52)) * spot * short_sd

    short_call = round_to_strike(spot + weekly_move)
    short_put = round_to_strike(spot - weekly_move)
    long_call = short_call + wing_width
    long_put = short_put - wing_width

    sc_p = black_scholes(spot, short_call, T, sigma, "call")
    sp_p = black_scholes(spot, short_put, T, sigma, "put")
    lc_p = black_scholes(spot, long_call, T, sigma, "call")
    lp_p = black_scholes(spot, long_put, T, sigma, "put")

    net_premium = (sc_p + sp_p) - (lc_p + lp_p)
    max_loss = wing_width - net_premium

    setup = TradeSetup(
        strategy="iron_condor",
        entry_date=entry_date,
        expiry_date=expiry_date,
        spot_entry=spot,
        vix_entry=vix,
        regime=regime,
        regime_confidence=confidence,
        legs=[
            (short_call, "call", "sell", round(sc_p, 2)),
            (long_call, "call", "buy", round(lc_p, 2)),
            (short_put, "put", "sell", round(sp_p, 2)),
            (long_put, "put", "buy", round(lp_p, 2)),
        ],
        max_profit=round(net_premium, 2),
        max_loss=round(max_loss, 2),
        breakeven_lower=round(short_put - net_premium, 2),
        breakeven_upper=round(short_call + net_premium, 2),
    )
    return setup


def payoff_at_expiry(setup: TradeSetup, spot_expiry: float) -> float:
    """Calculate actual P&L at expiry given final spot price."""
    pnl = 0.0
    for strike, opt_type, action, premium in setup.legs:
        if opt_type == "call":
            intrinsic = max(spot_expiry - strike, 0)
        else:
            intrinsic = max(strike - spot_expiry, 0)

        if action == "sell":
            pnl += premium - intrinsic
        else:
            pnl += intrinsic - premium
    return round(pnl, 2)
