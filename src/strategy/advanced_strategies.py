"""
Advanced Nifty options strategies:
  - Broken Wing Butterfly (BWB): asymmetric butterfly, net credit, skew capture
  - Ratio Spread: sell 2 OTM, buy 1 ATM — directional premium collection
  - Bull/Bear Spread: defined-risk directional plays when HMM detects trend
"""
import numpy as np
from .pricer import black_scholes, round_to_strike
from .base import TradeSetup


def broken_wing_butterfly(
    spot: float, vix: float, entry_date: str, expiry_date: str,
    regime: str = "normal", confidence: float = 0.7,
    bias: str = "neutral",   # "up", "down", "neutral"
    dte: int = 5,
) -> TradeSetup:
    """
    Broken Wing Butterfly — net credit strategy.
    Neutral/slight directional bias. Profits from spot staying near ATM.

    Structure (put side, bearish example):
      Buy  1× (ATM - 100) put
      Sell 2× (ATM - 200) put
      Buy  1× (ATM - 350) put   ← broken wing (wider than standard)
    Net credit due to asymmetric wing placement.
    """
    sigma = vix / 100
    T = max(dte, 1) / 365
    atm = round_to_strike(spot)

    if bias == "up":
        # Call-side BWB
        b1 = atm + 100
        b2 = atm + 200
        b3 = atm + 350
        p_b1 = black_scholes(spot, b1, T, sigma, "call")
        p_b2 = black_scholes(spot, b2, T, sigma, "call")
        p_b3 = black_scholes(spot, b3, T, sigma, "call")
        legs = [
            (b1, "call", "buy",  round(p_b1, 2)),
            (b2, "call", "sell", round(p_b2, 2)),
            (b2, "call", "sell", round(p_b2, 2)),
            (b3, "call", "buy",  round(p_b3, 2)),
        ]
    else:
        # Put-side BWB (default: neutral/down)
        b1 = atm - 100
        b2 = atm - 200
        b3 = atm - 350
        p_b1 = black_scholes(spot, b1, T, sigma, "put")
        p_b2 = black_scholes(spot, b2, T, sigma, "put")
        p_b3 = black_scholes(spot, b3, T, sigma, "put")
        legs = [
            (b1, "put", "buy",  round(p_b1, 2)),
            (b2, "put", "sell", round(p_b2, 2)),
            (b2, "put", "sell", round(p_b2, 2)),
            (b3, "put", "buy",  round(p_b3, 2)),
        ]

    net = sum(p if a == "sell" else -p for _, _, a, p in legs)
    max_profit = net + abs(b2 - b1)  # max profit at b2
    max_loss = abs(b3 - b2) - abs(b2 - b1) - net  # loss at b3

    return TradeSetup(
        strategy="broken_wing_butterfly",
        entry_date=entry_date, expiry_date=expiry_date,
        spot_entry=spot, vix_entry=vix,
        regime=regime, regime_confidence=confidence,
        legs=legs,
        max_profit=round(max_profit, 2),
        max_loss=round(max_loss, 2),
        breakeven_lower=round(atm - max_profit, 2),
        breakeven_upper=round(atm + max_profit, 2),
    )


def ratio_spread(
    spot: float, vix: float, entry_date: str, expiry_date: str,
    regime: str = "normal", confidence: float = 0.7,
    direction: str = "call",    # "call" = slightly bullish, "put" = slightly bearish
    sd_multiple: float = 0.8,
    dte: int = 5,
) -> TradeSetup:
    """
    1x2 Ratio Spread — buy 1 ATM, sell 2 OTM.
    Net credit. Profits if market moves moderately in the direction,
    but has uncapped risk beyond short strikes.
    Best used when HMM detects directional regime with moderate confidence.
    """
    sigma = vix / 100
    T = max(dte, 1) / 365
    weekly_move = (sigma / np.sqrt(52)) * spot * sd_multiple

    atm = round_to_strike(spot)
    if direction == "call":
        otm = round_to_strike(spot + weekly_move)
        p_atm = black_scholes(spot, atm, T, sigma, "call")
        p_otm = black_scholes(spot, otm, T, sigma, "call")
        legs = [
            (atm, "call", "buy",  round(p_atm, 2)),
            (otm, "call", "sell", round(p_otm, 2)),
            (otm, "call", "sell", round(p_otm, 2)),
        ]
    else:
        otm = round_to_strike(spot - weekly_move)
        p_atm = black_scholes(spot, atm, T, sigma, "put")
        p_otm = black_scholes(spot, otm, T, sigma, "put")
        legs = [
            (atm, "put", "buy",  round(p_atm, 2)),
            (otm, "put", "sell", round(p_otm, 2)),
            (otm, "put", "sell", round(p_otm, 2)),
        ]

    net = sum(p if a == "sell" else -p for _, _, a, p in legs)
    spread_width = abs(otm - atm)
    max_profit = net + spread_width

    return TradeSetup(
        strategy="ratio_spread",
        entry_date=entry_date, expiry_date=expiry_date,
        spot_entry=spot, vix_entry=vix,
        regime=regime, regime_confidence=confidence,
        legs=legs,
        max_profit=round(max_profit, 2),
        max_loss=float("inf"),
        breakeven_lower=round(atm - net, 2) if direction == "put" else spot,
        breakeven_upper=round(atm + net, 2) if direction == "call" else spot,
    )


def bull_call_spread(
    spot: float, vix: float, entry_date: str, expiry_date: str,
    regime: str = "normal", confidence: float = 0.7,
    sd_low: float = 0.2, sd_high: float = 0.8, dte: int = 5,
) -> TradeSetup:
    """
    Bull Call Spread — buy lower strike call, sell higher strike call.
    Defined risk + reward. Use when HMM shows uptrend signal.
    """
    sigma = vix / 100
    T = max(dte, 1) / 365
    weekly_move = (sigma / np.sqrt(52)) * spot
    buy_strike  = round_to_strike(spot + weekly_move * sd_low)
    sell_strike = round_to_strike(spot + weekly_move * sd_high)

    p_buy  = black_scholes(spot, buy_strike,  T, sigma, "call")
    p_sell = black_scholes(spot, sell_strike, T, sigma, "call")
    net_debit = p_buy - p_sell
    max_profit = abs(sell_strike - buy_strike) - net_debit

    return TradeSetup(
        strategy="bull_call_spread",
        entry_date=entry_date, expiry_date=expiry_date,
        spot_entry=spot, vix_entry=vix,
        regime=regime, regime_confidence=confidence,
        legs=[
            (buy_strike,  "call", "buy",  round(p_buy,  2)),
            (sell_strike, "call", "sell", round(p_sell, 2)),
        ],
        max_profit=round(max_profit, 2),
        max_loss=round(net_debit, 2),
        breakeven_lower=spot,
        breakeven_upper=round(buy_strike + net_debit, 2),
    )


def bear_put_spread(
    spot: float, vix: float, entry_date: str, expiry_date: str,
    regime: str = "normal", confidence: float = 0.7,
    sd_low: float = 0.2, sd_high: float = 0.8, dte: int = 5,
) -> TradeSetup:
    """
    Bear Put Spread — buy higher strike put, sell lower strike put.
    Defined risk. Use when HMM shows downtrend signal.
    """
    sigma = vix / 100
    T = max(dte, 1) / 365
    weekly_move = (sigma / np.sqrt(52)) * spot
    buy_strike  = round_to_strike(spot - weekly_move * sd_low)
    sell_strike = round_to_strike(spot - weekly_move * sd_high)

    p_buy  = black_scholes(spot, buy_strike,  T, sigma, "put")
    p_sell = black_scholes(spot, sell_strike, T, sigma, "put")
    net_debit = p_buy - p_sell
    max_profit = abs(buy_strike - sell_strike) - net_debit

    return TradeSetup(
        strategy="bear_put_spread",
        entry_date=entry_date, expiry_date=expiry_date,
        spot_entry=spot, vix_entry=vix,
        regime=regime, regime_confidence=confidence,
        legs=[
            (buy_strike,  "put", "buy",  round(p_buy,  2)),
            (sell_strike, "put", "sell", round(p_sell, 2)),
        ],
        max_profit=round(max_profit, 2),
        max_loss=round(net_debit, 2),
        breakeven_lower=round(buy_strike - net_debit, 2),
        breakeven_upper=spot,
    )
