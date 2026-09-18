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


# ═══════════════════════════════════════════════════════════════════
# RESEARCH ENGINE (strategy/strategies.py, strategy/regime.py)
# Used ONLY by the multi-strategy backtest research engine. The live
# alert path (main.py / scan_once.py) does not read anything below
# this line, so tuning these cannot change your live alerts.
# ═══════════════════════════════════════════════════════════════════

# --- EMA structure ---
EMA_FAST = 9
EMA_SLOW = 21

# --- Volatility ---
ATR_PERIOD = 14
ATR_STOP_MULT = 1.5              # stop distance in ATRs from entry
STRUCT_STOP_BUFFER_ATR = 0.25    # extra room beyond a structural level
ATR_PERCENTILE_LOOKBACK = 50

# --- Structure / retest ---
STRUCT_LOOKBACK = 10             # candles defining swing high/low
RETEST_WINDOW = 5                # how recently a breakout must have happened
RETEST_TOUCH_ATR = 0.20          # how close counts as "touching" a level
OPENING_RANGE_MINUTES = 30

# --- VWAP reclaim model ---
VWAP_RECLAIM_LOOKBACK = 6
VWAP_RECLAIM_MIN_BARS = 4        # bars on the wrong side before a reclaim counts
CONFLUENCE_MAX_ATR = 0.5         # EMA and VWAP within this = one zone

# --- Regime classifier ---
REGIME_FILTER_ENABLED = os.getenv("REGIME_FILTER_ENABLED", "true").lower() == "true"
REGIME_BLOCK_HIGH_VOL = True
REGIME_ATR_HIGH_PCT = 0.85       # ATR percentile rank above this = "high" vol
REGIME_ATR_LOW_PCT = 0.20
REGIME_SLOPE_MIN = 0.0002        # EMA slope magnitude to count as a directional vote
REGIME_EMA_SEPARATION_MIN = 0.001  # |fast-slow|/price for a "strong" trend

# --- Short asymmetry (see strategies.py docstring) ---
SHORT_STOP_ATR_MULT_EXTRA = 0.25   # downside moves are faster: wider stops
SHORT_RVOL_EXTRA = 0.5             # demand more volume confirmation on shorts
SHORT_MIN_SESSION_CANDLES = 12     # no shorts in the opening candles

# --- Scoring engine ---
SCORE_WEIGHTS = {
    "trend": 0.25,
    "momentum": 0.15,
    "volume": 0.15,
    "vwap": 0.20,
    "volatility": 0.10,
    "price_action": 0.15,
}
SCORE_MIN_TOTAL = 55.0           # combined 0-100 score needed to fire
SCORE_MIN_COMPONENT = 0.10       # floor, applied ONLY to SCORE_FLOOR_COMPONENTS
# Which components the floor applies to. Applying it to all six turns the
# weighted score back into AND-logic, where one weak component vetoes the
# setup - which defeats the point of scoring. Only the two that define
# whether the trade is directionally valid at all are floored; volume,
# volatility and price-action are allowed to be weak if the rest is strong.
SCORE_FLOOR_COMPONENTS = ("trend", "vwap")
SCORE_SEPARATION_FULL = 0.004    # |fast-slow|/price scoring 1.0 on trend
SCORE_SLOPE_FULL = 0.0008        # EMA slope scoring 1.0 on momentum
# VWAP distance is scored as a Z-SCORE, not in ATRs - see the long
# comment in strategies.py::_component_scores. Session VWAP is anchored
# at the open, so raw distance grows mechanically through a trending
# session and is not a usable overextension measure.
SCORE_VWAP_IDEAL_Z = 1.0         # z-score of VWAP distance scoring 1.0
SCORE_VWAP_FADE_Z = 2.5          # score fades to 0 this far beyond ideal        # further than ideal, score fades over this span
SCORE_ATR_IDEAL = 0.5            # ideal ATR percentile rank

# --- Research backtest mechanics ---
MIN_WARMUP_CANDLES = 25          # candles before any model may fire
MAX_TRADES_PER_SYMBOL_DAY = 3    # per model, per direction
SIGNAL_COOLDOWN_CANDLES = 6      # gap between same-model signals

# --- Costs (applied to every simulated research trade) ---
# Zero-cost backtests are the single most common way an intraday
# strategy looks profitable and then isn't. These are deliberately ON.
SLIPPAGE_BPS = 2.0               # basis points per side
COMMISSION_BPS = 1.0             # basis points per side

# --- Position sizing (reporting only; R-multiples are size-agnostic) ---
ACCOUNT_EQUITY = 100000.0
RISK_PER_TRADE_PCT = 0.005       # 0.5%; sweep with --risk-sweep
