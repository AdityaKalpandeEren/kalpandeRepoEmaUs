import os
from dotenv import load_dotenv

load_dotenv()

# --- Telegram ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# No exchange credentials needed - yfinance (Yahoo Finance) is free and
# keyless. That's the main practical difference from the NSE bot: no
# API key, no daily token refresh, nothing to rotate.

# --- Strategy parameters (same logic as BTC/NSE bots) ---
EMA_PERIOD = 20
VOLUME_AVG_PERIOD = 20
VOLUME_MULTIPLIER = 2.0          # signal candle volume must be > 2x the avg
CANDLE_INTERVAL_MINUTES = 5
RISK_REWARD_RATIO = 1.5
MAX_RISK_PCT = 0.015             # skip signal if stop-loss implies >1.5% risk

# --- VWAP retest-for-long parameters ---
RETEST_TREND_LOOKBACK = 5        # how many prior candles to check for an established above-VWAP trend
RETEST_MIN_CANDLES_ABOVE = 3     # at least this many of those prior candles must close above VWAP
RETEST_TOUCH_BUFFER_PCT = 0.001  # 0.1% buffer - counts as "touching" VWAP even if it doesn't hit exactly

# --- Runtime ---
POLL_SECONDS = 60                # how often the loop checks for new candles
SKIP_FIRST_MINUTES = 15          # ignore signals in first 15 min after market open

# --- Session hours ---
# Regular NYSE/NASDAQ trading hours. Index symbols (^GSPC, ^IXIC, ^DJI)
# follow the same session. Futures (ES=F, NQ=F, CL=F, GC=F, ...) trade
# nearly 24/5 on CME Globex, but this bot only scans them during the
# core session below for simplicity and to keep the session-VWAP model
# (which resets each session) meaningful - see README for how to widen
# this if you want overnight futures coverage.
MARKET_OPEN_HOUR = 4
MARKET_OPEN_MINUTE = 0
MARKET_CLOSE_HOUR = 20
MARKET_CLOSE_MINUTE = 0
MARKET_TIMEZONE = "America/New_York"    # zoneinfo handles EST/EDT (DST) automatically

# --- VWAP timeline ---
# Controls WHEN "the day" starts for VWAP purposes ONLY - candle
# fetching/scanning still starts at MARKET_OPEN_HOUR above regardless.
#   false (default) - VWAP accumulates from the first candle of the day,
#     which is 4:00 AM ET pre-market (see strategy/indicators.py
#     add_vwap docstring for exactly why) - pre-market volume/price is
#     baked into VWAP.
#   true - VWAP instead resets at the 9:30 AM regular-session open, the
#     "textbook" definition most traders assume VWAP means.
VWAP_REGULAR_SESSION_ONLY = os.getenv("VWAP_REGULAR_SESSION_ONLY", "false").lower() == "true"
REGULAR_SESSION_OPEN_HOUR = 9
REGULAR_SESSION_OPEN_MINUTE = 30

# --- Paper trading (main.py only - see paper_trading/tracker.py) ---
# When on, every signal main.py generates also opens a virtual position
# that gets tracked forward (filled on the next candle's open, closed on
# target/stop/EOD-square-off) into paper_trades.db, so you can run
# `python -m paper_trading.generate_report` to see real accuracy without
# risking money. Needs main.py running continuously - see that module's
# docstring for why.
PAPER_TRADING_ENABLED = os.getenv("PAPER_TRADING_ENABLED", "true").lower() == "true"
