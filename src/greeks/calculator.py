"""
Position-level Greeks calculator.
Computes delta, gamma, theta, vega for the full options position
and tracks them daily as spot/time/vol change.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import List

RISK_FREE_RATE = 0.065
LOT_SIZE = 75


@dataclass
class GreeksSnapshot:
    date: str
    spot: float
    vix: float
    days_remaining: int
    delta: float       # position delta (net, per lot)
    gamma: float       # position gamma (net, per lot)
    theta: float       # daily theta decay (INR per lot)
    vega: float        # vega per 1% VIX move (INR per lot)
    net_premium: float # cumulative premium collected so far
    unrealized_pnl: float


def _bs_greeks_single(S, K, T, sigma, option_type, action):
    """Greeks for one leg, adjusted for buy/sell direction."""
    if T <= 0:
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "price": intrinsic}

    r = RISK_FREE_RATE
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)

    price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)) if option_type == "call" \
            else (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))

    delta = norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1)
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    theta_raw = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    if option_type == "put":
        theta_raw = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    vega = S * pdf_d1 * np.sqrt(T) / 100  # per 1% change in vol

    sign = 1 if action == "sell" else -1
    return {
        "price": price,
        "delta": sign * delta,
        "gamma": sign * gamma,
        "theta": sign * theta_raw,   # positive theta = we collect
        "vega":  sign * vega,        # negative vega = short vol exposure
    }


def position_greeks(legs: list, spot: float, vix: float, days_remaining: int) -> dict:
    """
    Aggregate Greeks for all legs at given market conditions.

    legs: list of (strike, option_type, action, entry_price)
    Returns position-level Greeks per lot (per 75 shares).
    """
    T = max(days_remaining, 0) / 365
    sigma = vix / 100
    agg = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "price": 0.0}

    for strike, opt_type, action, entry_price in legs:
        g = _bs_greeks_single(spot, strike, T, sigma, opt_type, action)
        for k in agg:
            agg[k] += g[k]

    # Convert to INR per lot for theta and vega
    agg["theta_inr"] = agg["theta"] * LOT_SIZE     # daily theta in INR per lot
    agg["vega_inr"]  = agg["vega"]  * LOT_SIZE     # vega in INR per 1% vol move

    return agg


def track_greeks_daily(
    legs: list,
    df: pd.DataFrame,
    entry_date,
    expiry_date,
    entry_legs_prices: list = None,  # actual entry prices (may differ from BS)
) -> List[GreeksSnapshot]:
    """
    Compute Greeks snapshot for every trading day of the position.
    Returns list of GreeksSnapshot objects.
    """
    days_in_trade = df.loc[
        (df.index >= entry_date) & (df.index <= expiry_date)
    ]
    snapshots = []
    cumulative_theta_inr = 0.0

    for i, (date, row) in enumerate(days_in_trade.iterrows()):
        spot = float(row["Close"])
        vix  = float(row["VIX"])
        days_left = max((expiry_date - date).days, 0)

        g = position_greeks(legs, spot, vix, days_left)
        cumulative_theta_inr += g["theta_inr"]

        # Unrealized P&L vs entry
        unreal = 0.0
        sigma = vix / 100
        T = max(days_left, 0) / 365
        for j, (strike, opt_type, action, entry_price) in enumerate(legs):
            g2 = _bs_greeks_single(spot, strike, T, sigma, opt_type, action)
            cur_price = g2["price"]
            unreal += (entry_price - cur_price) if action == "sell" else (cur_price - entry_price)

        snapshots.append(GreeksSnapshot(
            date=str(date.date()),
            spot=round(spot, 2),
            vix=round(vix, 2),
            days_remaining=days_left,
            delta=round(g["delta"], 4),
            gamma=round(g["gamma"], 6),
            theta=round(g["theta_inr"], 2),
            vega=round(g["vega_inr"], 2),
            net_premium=round(cumulative_theta_inr, 2),
            unrealized_pnl=round(unreal, 2),
        ))

    return snapshots
