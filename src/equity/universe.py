"""
NSE stock universe for the Lynch-style equity backtest.

~115 large/mid-cap NSE stocks spanning ~12 sectors, drawn from
Nifty 100 / Nifty Next 50 constituents. Used both for the historical
backtest and the "live picks" mode.

market_cap_tier is a rough static classification ("Large"/"Mid") used
as a tie-breaker proxy for the Stalwart category — not meant to be
precise or kept perfectly current.
"""
from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class StockInfo:
    ticker: str
    name: str
    sector: str
    market_cap_tier: str  # "Large" | "Mid"


UNIVERSE: list[StockInfo] = [
    # ── Banking / Financial Services ──────────────────────────────
    StockInfo("HDFCBANK.NS", "HDFC Bank", "Banking", "Large"),
    StockInfo("ICICIBANK.NS", "ICICI Bank", "Banking", "Large"),
    StockInfo("SBIN.NS", "State Bank of India", "Banking", "Large"),
    StockInfo("KOTAKBANK.NS", "Kotak Mahindra Bank", "Banking", "Large"),
    StockInfo("AXISBANK.NS", "Axis Bank", "Banking", "Large"),
    StockInfo("INDUSINDBK.NS", "IndusInd Bank", "Banking", "Mid"),
    StockInfo("BANKBARODA.NS", "Bank of Baroda", "Banking", "Mid"),
    StockInfo("PNB.NS", "Punjab National Bank", "Banking", "Mid"),
    StockInfo("FEDERALBNK.NS", "Federal Bank", "Banking", "Mid"),
    StockInfo("IDFCFIRSTB.NS", "IDFC First Bank", "Banking", "Mid"),
    StockInfo("BAJFINANCE.NS", "Bajaj Finance", "Financial Services", "Large"),
    StockInfo("BAJAJFINSV.NS", "Bajaj Finserv", "Financial Services", "Large"),
    StockInfo("HDFCLIFE.NS", "HDFC Life Insurance", "Financial Services", "Large"),
    StockInfo("SBILIFE.NS", "SBI Life Insurance", "Financial Services", "Large"),
    StockInfo("ICICIGI.NS", "ICICI Lombard", "Financial Services", "Mid"),
    StockInfo("ICICIPRULI.NS", "ICICI Prudential Life", "Financial Services", "Mid"),
    StockInfo("CHOLAFIN.NS", "Cholamandalam Investment", "Financial Services", "Mid"),
    StockInfo("MUTHOOTFIN.NS", "Muthoot Finance", "Financial Services", "Mid"),
    StockInfo("PFC.NS", "Power Finance Corp", "Financial Services", "Mid"),
    StockInfo("RECLTD.NS", "REC Limited", "Financial Services", "Mid"),

    # ── IT ─────────────────────────────────────────────────────────
    StockInfo("TCS.NS", "Tata Consultancy Services", "IT", "Large"),
    StockInfo("INFY.NS", "Infosys", "IT", "Large"),
    StockInfo("HCLTECH.NS", "HCL Technologies", "IT", "Large"),
    StockInfo("WIPRO.NS", "Wipro", "IT", "Large"),
    StockInfo("TECHM.NS", "Tech Mahindra", "IT", "Large"),
    StockInfo("LTIM.NS", "LTIMindtree", "IT", "Mid"),
    StockInfo("PERSISTENT.NS", "Persistent Systems", "IT", "Mid"),
    StockInfo("COFORGE.NS", "Coforge", "IT", "Mid"),
    StockInfo("MPHASIS.NS", "Mphasis", "IT", "Mid"),

    # ── Pharma ─────────────────────────────────────────────────────
    StockInfo("SUNPHARMA.NS", "Sun Pharmaceutical", "Pharma", "Large"),
    StockInfo("DRREDDY.NS", "Dr Reddy's Labs", "Pharma", "Large"),
    StockInfo("CIPLA.NS", "Cipla", "Pharma", "Large"),
    StockInfo("DIVISLAB.NS", "Divi's Laboratories", "Pharma", "Large"),
    StockInfo("APOLLOHOSP.NS", "Apollo Hospitals", "Pharma", "Large"),
    StockInfo("TORNTPHARM.NS", "Torrent Pharma", "Pharma", "Mid"),
    StockInfo("LUPIN.NS", "Lupin", "Pharma", "Mid"),
    StockInfo("AUROPHARMA.NS", "Aurobindo Pharma", "Pharma", "Mid"),
    StockInfo("ALKEM.NS", "Alkem Laboratories", "Pharma", "Mid"),
    StockInfo("BIOCON.NS", "Biocon", "Pharma", "Mid"),

    # ── FMCG ───────────────────────────────────────────────────────
    StockInfo("HINDUNILVR.NS", "Hindustan Unilever", "FMCG", "Large"),
    StockInfo("ITC.NS", "ITC", "FMCG", "Large"),
    StockInfo("NESTLEIND.NS", "Nestle India", "FMCG", "Large"),
    StockInfo("BRITANNIA.NS", "Britannia Industries", "FMCG", "Large"),
    StockInfo("TATACONSUM.NS", "Tata Consumer Products", "FMCG", "Mid"),
    StockInfo("DABUR.NS", "Dabur India", "FMCG", "Mid"),
    StockInfo("GODREJCP.NS", "Godrej Consumer Products", "FMCG", "Mid"),
    StockInfo("MARICO.NS", "Marico", "FMCG", "Mid"),
    StockInfo("COLPAL.NS", "Colgate-Palmolive India", "FMCG", "Mid"),
    StockInfo("VBL.NS", "Varun Beverages", "FMCG", "Mid"),

    # ── Auto & Auto Ancillaries ────────────────────────────────────
    StockInfo("MARUTI.NS", "Maruti Suzuki", "Auto", "Large"),
    StockInfo("TATAMOTORS.NS", "Tata Motors", "Auto", "Large"),
    StockInfo("M&M.NS", "Mahindra & Mahindra", "Auto", "Large"),
    StockInfo("BAJAJ-AUTO.NS", "Bajaj Auto", "Auto", "Large"),
    StockInfo("EICHERMOT.NS", "Eicher Motors", "Auto", "Large"),
    StockInfo("HEROMOTOCO.NS", "Hero MotoCorp", "Auto", "Mid"),
    StockInfo("TVSMOTOR.NS", "TVS Motor Company", "Auto", "Mid"),
    StockInfo("BOSCHLTD.NS", "Bosch Limited", "Auto", "Mid"),
    StockInfo("MOTHERSON.NS", "Samvardhana Motherson", "Auto", "Mid"),
    StockInfo("BALKRISIND.NS", "Balkrishna Industries", "Auto", "Mid"),

    # ── Energy / Oil & Gas ─────────────────────────────────────────
    StockInfo("RELIANCE.NS", "Reliance Industries", "Energy", "Large"),
    StockInfo("ONGC.NS", "Oil & Natural Gas Corp", "Energy", "Large"),
    StockInfo("BPCL.NS", "Bharat Petroleum", "Energy", "Mid"),
    StockInfo("IOC.NS", "Indian Oil Corp", "Energy", "Mid"),
    StockInfo("GAIL.NS", "GAIL India", "Energy", "Mid"),
    StockInfo("PETRONET.NS", "Petronet LNG", "Energy", "Mid"),
    StockInfo("ATGL.NS", "Adani Total Gas", "Energy", "Mid"),

    # ── Metals & Mining ────────────────────────────────────────────
    StockInfo("TATASTEEL.NS", "Tata Steel", "Metals", "Large"),
    StockInfo("JSWSTEEL.NS", "JSW Steel", "Metals", "Large"),
    StockInfo("HINDALCO.NS", "Hindalco Industries", "Metals", "Large"),
    StockInfo("VEDL.NS", "Vedanta", "Metals", "Mid"),
    StockInfo("COALINDIA.NS", "Coal India", "Metals", "Large"),
    StockInfo("JINDALSTEL.NS", "Jindal Steel & Power", "Metals", "Mid"),
    StockInfo("NMDC.NS", "NMDC Limited", "Metals", "Mid"),
    StockInfo("SAIL.NS", "Steel Authority of India", "Metals", "Mid"),

    # ── Infra / Cement / Construction ─────────────────────────────
    StockInfo("LT.NS", "Larsen & Toubro", "Infra", "Large"),
    StockInfo("ULTRACEMCO.NS", "UltraTech Cement", "Infra", "Large"),
    StockInfo("GRASIM.NS", "Grasim Industries", "Infra", "Large"),
    StockInfo("SHREECEM.NS", "Shree Cement", "Infra", "Mid"),
    StockInfo("AMBUJACEM.NS", "Ambuja Cements", "Infra", "Mid"),
    StockInfo("ACC.NS", "ACC Limited", "Infra", "Mid"),
    StockInfo("ADANIPORTS.NS", "Adani Ports & SEZ", "Infra", "Large"),
    StockInfo("ADANIENT.NS", "Adani Enterprises", "Infra", "Large"),
    StockInfo("DLF.NS", "DLF Limited", "Infra", "Mid"),
    StockInfo("GMRINFRA.NS", "GMR Airports Infra", "Infra", "Mid"),

    # ── Telecom ────────────────────────────────────────────────────
    StockInfo("BHARTIARTL.NS", "Bharti Airtel", "Telecom", "Large"),
    StockInfo("IDEA.NS", "Vodafone Idea", "Telecom", "Mid"),
    StockInfo("INDUSTOWER.NS", "Indus Towers", "Telecom", "Mid"),

    # ── Consumer Durables / Retail ─────────────────────────────────
    StockInfo("TITAN.NS", "Titan Company", "Consumer Durables", "Large"),
    StockInfo("DMART.NS", "Avenue Supermarts", "Consumer Durables", "Large"),
    StockInfo("TRENT.NS", "Trent Limited", "Consumer Durables", "Mid"),
    StockInfo("HAVELLS.NS", "Havells India", "Consumer Durables", "Mid"),
    StockInfo("VOLTAS.NS", "Voltas", "Consumer Durables", "Mid"),
    StockInfo("DIXON.NS", "Dixon Technologies", "Consumer Durables", "Mid"),
    StockInfo("CROMPTON.NS", "Crompton Greaves Consumer", "Consumer Durables", "Mid"),

    # ── Chemicals ──────────────────────────────────────────────────
    StockInfo("PIDILITIND.NS", "Pidilite Industries", "Chemicals", "Large"),
    StockInfo("SRF.NS", "SRF Limited", "Chemicals", "Mid"),
    StockInfo("UPL.NS", "UPL Limited", "Chemicals", "Mid"),
    StockInfo("AARTIIND.NS", "Aarti Industries", "Chemicals", "Mid"),
    StockInfo("DEEPAKNTR.NS", "Deepak Nitrite", "Chemicals", "Mid"),
    StockInfo("PIIND.NS", "PI Industries", "Chemicals", "Mid"),

    # ── Power / Utilities ──────────────────────────────────────────
    StockInfo("NTPC.NS", "NTPC Limited", "Power", "Large"),
    StockInfo("POWERGRID.NS", "Power Grid Corp", "Power", "Large"),
    StockInfo("TATAPOWER.NS", "Tata Power", "Power", "Mid"),
    StockInfo("ADANIPOWER.NS", "Adani Power", "Power", "Mid"),
    StockInfo("NHPC.NS", "NHPC Limited", "Power", "Mid"),
    StockInfo("TORNTPOWER.NS", "Torrent Power", "Power", "Mid"),

    # ── Other (cement/paints/diversified) ──────────────────────────
    StockInfo("ASIANPAINT.NS", "Asian Paints", "Consumer Durables", "Large"),
    StockInfo("BERGEPAINT.NS", "Berger Paints", "Consumer Durables", "Mid"),
    StockInfo("SIEMENS.NS", "Siemens India", "Infra", "Large"),
    StockInfo("ABB.NS", "ABB India", "Infra", "Mid"),
    StockInfo("CUMMINSIND.NS", "Cummins India", "Infra", "Mid"),
    StockInfo("HAL.NS", "Hindustan Aeronautics", "Infra", "Large"),
    StockInfo("BEL.NS", "Bharat Electronics", "Infra", "Large"),
]


# 10-stock subset for fast iteration / smoke tests
MINI_UNIVERSE: list[StockInfo] = [
    StockInfo("RELIANCE.NS", "Reliance Industries", "Energy", "Large"),
    StockInfo("TCS.NS", "Tata Consultancy Services", "IT", "Large"),
    StockInfo("HDFCBANK.NS", "HDFC Bank", "Banking", "Large"),
    StockInfo("TATAMOTORS.NS", "Tata Motors", "Auto", "Large"),
    StockInfo("SUNPHARMA.NS", "Sun Pharmaceutical", "Pharma", "Large"),
    StockInfo("HINDUNILVR.NS", "Hindustan Unilever", "FMCG", "Large"),
    StockInfo("TATASTEEL.NS", "Tata Steel", "Metals", "Large"),
    StockInfo("BHARTIARTL.NS", "Bharti Airtel", "Telecom", "Large"),
    StockInfo("LT.NS", "Larsen & Toubro", "Infra", "Large"),
    StockInfo("TITAN.NS", "Titan Company", "Consumer Durables", "Large"),
]


def tickers(universe: list[StockInfo] = None) -> list[str]:
    return [s.ticker for s in (universe or UNIVERSE)]


def sector_map(universe: list[StockInfo] = None) -> dict[str, str]:
    return {s.ticker: s.sector for s in (universe or UNIVERSE)}


def universe_df(universe: list[StockInfo] = None) -> pd.DataFrame:
    u = universe or UNIVERSE
    return pd.DataFrame([{
        "ticker": s.ticker, "name": s.name,
        "sector": s.sector, "market_cap_tier": s.market_cap_tier,
    } for s in u])
