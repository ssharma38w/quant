import numpy as np
from scipy.stats import norm


RISK_FREE_RATE = 0.065  # India 10Y approx


def black_scholes(S: float, K: float, T: float, sigma: float, option_type: str = "call") -> float:
    """
    European option price via Black-Scholes.
    T: time to expiry in years
    sigma: annualized IV (e.g. 0.15 for 15%)
    """
    if T <= 0:
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)

    r = RISK_FREE_RATE
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(S: float, K: float, T: float, sigma: float, option_type: str = "call") -> dict:
    if T <= 0:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}

    r = RISK_FREE_RATE
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)

    delta = norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1)
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    if option_type == "put":
        theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    vega = S * pdf_d1 * np.sqrt(T) / 100

    return {"delta": round(delta, 4), "gamma": round(gamma, 6), "theta": round(theta, 2), "vega": round(vega, 2)}


def round_to_strike(value: float, step: int = 50) -> int:
    """Round to nearest Nifty strike (multiples of 50)."""
    return int(round(value / step) * step)
