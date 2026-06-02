# Getting Real Data for Backtesting

## Option A — yfinance (easiest, automatic)
```bash
pip install -r requirements.txt
python main.py          # auto-downloads Nifty + India VIX from Yahoo Finance
```
Works on your local machine. The cloud sandbox blocks outbound connections.

---

## Option B — NSE bhavcopy (real options prices)
Downloads actual daily options chain from NSE archives. Used for `--real-options` mode.
```bash
python main.py --real-options   # downloads + uses actual NSE options prices
```
NSE archives go back to ~2008. Weekly options data available from 2019.

---

## Option C — Your own CSV files
Export Nifty historical data from NSE / TradingView / any broker:

**Nifty CSV format** (required columns):
```
Date,Open,High,Low,Close,Volume
2019-01-01,10910,10987,10800,10910,123456
...
```

**VIX CSV format** (optional, recommended):
```
Date,VIX
2019-01-01,15.3
...
```

**Where to download:**
- Nifty 50: https://www.nseindia.com/report-detail/eq_security → search NIFTY 50
- India VIX: https://www.nseindia.com/market-data/india-vix → Download historical data

**Run with CSV:**
```bash
python main.py --csv nifty_data.csv vix_data.csv
```
If no VIX CSV is provided, HV×1.2 is used as a rough IV proxy.

---

## Running All Modes
```bash
python main.py                          # real data (yfinance)
python main.py --sample                 # synthetic data (offline test)
python main.py --csv nifty.csv vix.csv  # your own CSV
python main.py --real-options           # real NSE options prices
python main.py --sweep                  # find best SD/wing params
python main.py signal                   # this week's trade signal
```
