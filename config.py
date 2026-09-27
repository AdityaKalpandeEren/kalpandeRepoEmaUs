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

# --- EMA stack breakout model (I) - 10/20/50, long only ---
# Separate from EMA_FAST/EMA_SLOW above (9/21, used by the regime
# classifier and the scoring engine) - this is the specific 10/20/50
# ribbon the user already reads by eye on their charts.
EMA_STACK_FAST = 10
EMA_STACK_MID = 20
EMA_STACK_SLOW = 50
EMA_STACK_MIN_RVOL = 1.2         # a breakout wants real volume behind it -
                                  # unlike a VWAP touch/test, this is the one
                                  # place high volume is being read as
                                  # confirmation, not climax exhaustion
EMA_STACK_MAX_EXTENSION_ATR = 3.0  # skip breakouts already this many ATRs
                                    # above the 50 EMA - chasing an extended
                                    # move rather than catching a fresh one

# --- J: VWAP standard-deviation band reversion (long only) ---
# From a cited QuantConnect study (100 liquid NASDAQ names): buying at
# the lower 2-SD VWAP band showed ~61% win rate at ~1.4:1 R:R, and the
# rarer 3-SD touch ~71% - the interesting claim is a high win rate
# WITHOUT collapsing R:R, which is exactly what our own RR-sweep this
# session could never get past 0.5R at a similar win rate. Worth
# testing on our own data rather than trusting the citation.
# Uses vwap_z (already computed by add_vwap_features) directly.
VWAP_BAND_Z_THRESHOLD = 2.0       # SD below VWAP required to arm the model
VWAP_BAND_MIN_RVOL = 1.0

# --- K: Larry Connors RSI(2) mean reversion (long only) ---
# Widely cited 75-79% win rate - but that literature is on DAILY bars
# with multi-day holds and a 200-day-MA trend filter; this is an
# intraday adaptation (5-min RSI(2), EMA_STACK_SLOW as the trend
# filter in place of the 200-day MA), so the win rate that literature
# reports does NOT directly transfer - it's a reason to test the idea,
# not a result to expect.
RSI2_PERIOD = 2
RSI2_OVERSOLD = 10.0
RSI2_TREND_FILTER_EMA = EMA_STACK_SLOW   # only buy dips above this EMA

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

# --- Aggressor filter (D/E/G) ---
# "Aggressor share" is the same geometric buy/sell split the live
# check_vwap_retest already computes (strategy/screener.py::_candle_aggressor),
# reframed as "how much of the signal candle's own range closed in the
# trade's favor" - 1.0 means it closed at the exact high (long) or exact
# low (short) with zero give-back.
#
# Data-driven, not assumed: a feature/outcome study over D_VWAP_RECLAIM,
# E_VWAP_REJECTION and G_CONFLUENCE (DRAM + IREN, 42 days each, ~390
# decided trades) found the naive read backwards - candles at >=0.95
# aggressor share (near-zero give-back) were the WORST-performing bucket
# in all three models individually, not the best:
#   D: below 0.95 win 38.2%/+ -0.03R  vs  >=0.95 win 29.2%/-0.39R
#   E: below 0.95 win 42.4%/+0.07R    vs  >=0.95 win 26.8%/-0.40R
#   G: below 0.95 win 34.9%/-0.13R    vs  >=0.95 win 25.0%/-0.38R
# A close pinned to the exact extreme with nothing pushing back reads as
# a climax/exhaustion print, not stronger conviction - closer to what
# "buying/selling the top tick" means for the other side. Re-validate
# on a broader symbol set before trusting this beyond the discovery set.
VWAP_MAX_AGGRESSOR = 0.95

# --- Candle-quality stack (E_VWAP_REJECTION) ---
# Same discovery run (DRAM + IREN) found that stacking three checks on
# the signal candle - some volume behind it, AND a moderate body (not
# a full-range no-wick print, which the aggressor filter above already
# discourages but this narrows further), AND the aggressor filter - is
# much stronger than aggressor alone:
#   aggressor<0.95 alone:                    n=125  win 42.4%  avgR +0.072
#   + rvol>1.0 + body 0.3-0.7:               n=26   win 61.5%  avgR +0.570
# Held up split by symbol (DRAM 64%, IREN 58%) and by direction (long
# 53%, short 73%) rather than being carried by one slice - but n=26 is
# still small. MUST be re-validated on a broader symbol set before
# this number is trusted; see broad-set results before/after in the
# commit/PR notes.
VWAP_REJECTION_MIN_RVOL = 1.0
VWAP_REJECTION_BODY_MIN = 0.3
VWAP_REJECTION_BODY_MAX = 0.7

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
# Override with e.g. ACCOUNT_EQUITY=10000 on the command line to size
# the P&L/drawdown columns to your actual budget - win rate and R
# multiples themselves don't change with this, only the currency math.
ACCOUNT_EQUITY = float(os.getenv("ACCOUNT_EQUITY", "100000"))
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "0.005"))  # 0.5%; sweep with --risk-sweep


# ═══════════════════════════════════════════════════════════════════
# SWING/POSITIONAL RESEARCH (strategy/swing_strategy.py,
# backtest/run_swing_research.py) - "SWING_DAYS_STR"
#
# A DIFFERENT engine from everything above: daily bars, multi-day
# holds (days, not minutes), no session VWAP, no EOD square-off. Kept
# fully separate from the intraday research engine rather than forced
# into evaluate_all() - that loop's regime classifier and stop
# convention are built around session VWAP and ATR-scaled intraday
# risk, neither of which means the same thing on a daily bar. Nothing
# here is read by the intraday engine or by live/main.py.
#
# The entry logic is grounded in Mark Minervini's published "Trend
# Template" (Trade Like a Stock Market Wizard) - moving-average stack
# alignment (fast > mid > slow, all trending up), price near its
# highs rather than its lows, and a volume-expansion breakout day -
# the closest thing to a "world class", publicly documented breakout
# system for swing/position equity trading. Minervini's own template
# uses the 50/150/200-day MAs; this uses the periods requested here
# (30/50/60) instead, so treat it as the same STRUCTURE, not a
# reproduction of his published results.
# ═══════════════════════════════════════════════════════════════════
SWING_EMA_FAST = 30
SWING_EMA_MID = 50
SWING_EMA_SLOW = 60
SWING_STRUCT_LOOKBACK = 50       # trading days defining the breakout level (~10 weeks)
SWING_VOLUME_AVG_PERIOD = 50     # "expected" volume baseline - ~10 trading weeks
SWING_MIN_VOLUME_RATIO = 1.5     # breakout day volume must be >= this x the baseline
SWING_MIN_ABOVE_52W_LOW_PCT = 0.25   # Minervini filter: price >= 25% above its 52-week low
SWING_MAX_BELOW_52W_HIGH_PCT = 0.25  # Minervini filter: price within 25% of its 52-week high
SWING_TARGET_PCT = 0.08          # +8% profit target
SWING_STOP_PCT = 0.02            # -2% stop-loss (4:1 reward:risk by construction)
SWING_MAX_HOLD_DAYS = 120        # safety cap so a backtest position can't hold forever
SWING_MIN_WARMUP_DAYS = SWING_EMA_SLOW + 5


# ═══════════════════════════════════════════════════════════════════
# ML META-LABEL FILTER (strategy/ml_features.py, model L_ML_META in
# strategy/strategies.py, backtest/ml/)
#
# NOT a strategy that invents its own entries. It's a classifier
# trained on this engine's own historical trade outcomes that decides
# which of the 12 existing models' candidate trades are worth taking
# (Lopez de Prado's "meta-labeling"). See backtest/ml/train_meta_model.py
# for the honest, walk-forward out-of-sample report - do not assume this
# helps just because it exists; the trainer's own printout says whether
# it beat "take every candidate" out of sample.
# ═══════════════════════════════════════════════════════════════════
ML_META_MODEL_PATH = "backtest/ml/model/meta_model.joblib"
ML_META_MIN_PROB = float(os.getenv("ML_META_MIN_PROB", "0.55"))   # predict_proba floor to fire
ML_META_TEST_FRAC = 0.3          # last N% of rows BY TIME held out, never trained on


# ═══════════════════════════════════════════════════════════════════
# ML META-LABEL FILTER V2 (model L_ML_META_V2 in strategy/strategies.py,
# strategy/market_context.py, strategy/news_catalyst.py,
# strategy/ml_features_v2.py, backtest/ml/*_v2.py)
#
# Same meta-labeling idea as L_ML_META above (the 12 base models propose,
# a classifier decides), with four changes - each one there because the
# V1 data or V1 design showed a concrete gap, not added for its own sake:
#
#  1. MARKET CONTEXT features (VIX/VXN level vs their 50-day SMA, VIX
#     term structure, % of a broad universe above its 50-day SMA, QQQ/SPY
#     intraday trend) - V1 only ever saw the symbol's own candles, so it
#     could not tell a calm tape from a risk-off one.
#  2. CATALYST features that are point-in-time backtestable: earnings
#     proximity + last EPS surprise, today's gap, time-of-day relative
#     volume ("is this stock in play today"), relative strength vs QQQ.
#  3. SESSION GATE - V1's own dataset: pre-market (-0.21R), after-hours
#     (-0.11R, mostly squared off at 8pm) and 15:00-16:00 (-0.28R) were
#     the worst buckets. V2 only trades the regular session window below
#     and flattens at the regular close instead of 8pm after-hours.
#  4. DYNAMIC EXITS + NEWS - an open V2 trade is closed early if the
#     market turns against it (QQQ drop / VIX spike since entry). Live,
#     headline news is read for the symbol AND the broad market: a bad
#     catalyst vetoes a long (a good one vetoes a short), an aligned one
#     slightly lowers the probability bar.
#
# NEWS IS LIVE-ONLY. Yahoo serves only the latest ~10 headlines per
# ticker - there is no historical archive, so headline sentiment cannot
# be backtested or trained on. It is therefore NOT an ML feature (that
# would be train/serve skew: trained always-neutral, served non-neutral),
# only a rule-based overlay applied at live inference time. Earnings
# dates/surprises and gap/volume ARE historical, which is why those are
# the catalyst inputs the model actually learns from.
# ═══════════════════════════════════════════════════════════════════
ML_V2_MODEL_PATH = "backtest/ml/model/meta_model_v2.joblib"
ML_V2_DATASET_PATH = "backtest/ml/data/dataset_v2.csv"
ML_V2_CACHE_DIR = "backtest/ml/cache"
# Probability floor. "auto" = use the threshold the trainer picked on its
# VALIDATION block (stored in the model bundle) - never one tuned on test.
ML_V2_MIN_PROB = os.getenv("ML_V2_MIN_PROB", "auto")

# --- Session gate (minutes after the 9:30 ET regular open) ---
ML_V2_SESSION_START_MIN = 15      # 9:45 - skip the opening auction chaos
ML_V2_SESSION_END_MIN = 330       # 15:00 - last-hour entries were the worst bucket
ML_V2_FLAT_MIN = 385              # 15:55 - flatten at the regular close, not 8pm

# --- Hard market-risk vetoes (rule-based, applied before the model) ---
# Standard risk-off markers. The ML only saw ~60 days of one market
# regime and cannot learn a crash it never saw; these rules cover that.
ML_V2_VETO_ENABLED = os.getenv("ML_V2_VETO_ENABLED", "true").lower() == "true"
ML_V2_VIX_MAX_LONG = 30.0         # no new longs with VIX at/above this
ML_V2_VIX_TERM_MAX_LONG = 1.05    # VIX/VIX3M above this = backwardation (panic)
ML_V2_VIX_DAY_SPIKE_MAX_LONG = 0.12   # VIX up >12% on the day = risk-off tape

# --- Dynamic "bad catalyst during the trade" exit ---
# Backtestable proxy for a market-wide negative catalyst hitting an open
# trade: the index moving hard against the position, or volatility
# spiking, after entry. Exits at that bar's close.
ML_V2_SHOCK_EXIT_ENABLED = os.getenv("ML_V2_SHOCK_EXIT_ENABLED", "true").lower() == "true"
ML_V2_SHOCK_QQQ_PCT = 0.006       # QQQ moves 0.6% against the trade since entry
ML_V2_SHOCK_VIX_PCT = 0.08        # VIX up 8% since entry (longs) / down 8% (shorts)

# --- Breadth universe (% above 50-day SMA) ---
# A fixed, sector-diverse large-cap list rather than the watchlist, so the
# breadth reading does not change meaning when the watchlist changes.
ML_V2_BREADTH_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "ORCL", "CRM",
    "ADBE", "AMD", "INTC", "CSCO", "QCOM", "TXN", "IBM", "NOW", "INTU", "AMAT",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "V", "MA",
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT", "AMGN",
    "XOM", "CVX", "COP", "SLB", "EOG",
    "WMT", "COST", "HD", "LOW", "MCD", "NKE", "SBUX", "TGT", "PG", "KO", "PEP",
    "CAT", "DE", "HON", "GE", "BA", "UPS", "RTX", "LMT", "UNP",
    "NEE", "DUK", "SO", "AMT", "PLD", "LIN", "DIS", "NFLX", "T", "VZ", "CMCSA",
]

# --- Live news catalyst overlay (see strategy/news_catalyst.py) ---
ML_V2_NEWS_ENABLED = os.getenv("ML_V2_NEWS_ENABLED", "true").lower() == "true"
ML_V2_NEWS_SYMBOL_LOOKBACK_HOURS = 24
ML_V2_NEWS_MARKET_LOOKBACK_HOURS = 8
ML_V2_NEWS_MARKET_TICKERS = ["SPY", "QQQ"]
ML_V2_NEWS_VETO = 0.35            # |score| at/above this against the trade = no trade
ML_V2_NEWS_MARKET_VETO = 0.45     # market-wide news needs to be clearer to veto
ML_V2_NEWS_BOOST = 0.35           # aligned score at/above this ...
ML_V2_NEWS_BOOST_PROB = 0.03      # ... lowers the probability bar by this much
ML_V2_NEWS_EXIT = 0.45            # open trade + fresh opposing news this strong = exit alert
ML_V2_NEWS_CACHE_SECONDS = 600
# Optional LLM scoring of headlines (Claude). Off by default: it costs
# money per call and the keyword scorer works without any API key. Needs
# `pip install anthropic` and ANTHROPIC_API_KEY (or `ant auth login`).
ML_V2_NEWS_LLM_ENABLED = os.getenv("ML_V2_NEWS_LLM_ENABLED", "false").lower() == "true"
ML_V2_NEWS_LLM_MODEL = os.getenv("ML_V2_NEWS_LLM_MODEL", "claude-opus-5")

# --- Training ---
ML_V2_VAL_FRAC = 0.2              # middle block (by DATE) - model/threshold selection
ML_V2_TEST_FRAC = 0.2             # last block (by DATE) - untouched, reported once
# A threshold must keep at least this many validation trades, on enough
# distinct days. The first long-only run chose a config off a 51-trade
# slice that didn't hold up - with 8 configs x 21 thresholds tried, a
# small or single-day slice looking good is mostly luck.
ML_V2_MIN_TRADES_FOR_THRESHOLD = 30
ML_V2_MIN_DAYS_FOR_THRESHOLD = 4      # ... spread over at least this many trading days

# Portfolio-level cap: at most this many NEW V2 trades per day across ALL
# symbols, first-come by time (what a single trader can actually take).
# Without it, one strong up-day produced 394 of 453 test trades and the
# whole "edge" was that one day - a per-trade t-stat of 3.3 that was 0.26
# once trades were grouped by day. Applied in the trainer's evaluation
# and in live_ml_v2.py.
ML_V2_MAX_TRADES_PER_DAY = int(os.getenv("ML_V2_MAX_TRADES_PER_DAY", "10"))

# Computed (and used by the hard vetoes / reason strings) but NOT fed to
# the classifier. Each is ~constant within a day and drifts slowly across
# days, so in a ~40-trading-day dataset it works as a date label: the
# model gets ~26 real observations of it and memorises which WEEK was
# good (the first run ranked raw VIX level #1 and its probabilities
# shifted wholesale on the test days). Revisit once the context cache has
# accumulated many months of history.
ML_V2_EXCLUDE_FEATURES = [
    "vix", "vix_vs_sma50", "vxn_vs_sma50",
    "breadth_pct50", "breadth_chg5",
    "qqq_vs_sma50_d", "spy_vs_sma50_d", "spy_vs_sma200_d",
]


# ═══════════════════════════════════════════════════════════════════
# LIVE RESEARCH ALERTS + PAPER TRADING (paper_trading/research_live.py,
# hooked into scan_once.py after the production alerts)
#
# Runs selected research strategies live on every scan: Telegram alert
# per new signal, a paper trade filled/exited exactly like the backtest,
# and an end-of-day report (win rate, expectancy, risk sweep, all-time
# totals) after config.MARKET_CLOSE_HOUR. The production EMA-cross /
# VWAP alerts in scan_once.py are unaffected - set
# LIVE_RESEARCH_ENABLED=false to switch this whole block off.
# ═══════════════════════════════════════════════════════════════════
LIVE_RESEARCH_ENABLED = os.getenv("LIVE_RESEARCH_ENABLED", "true").lower() == "true"
LIVE_RESEARCH_STRATEGIES = [s.strip() for s in os.getenv(
    "LIVE_RESEARCH_STRATEGIES",
    "K_RSI2_REVERSION,SCORE_ENGINE,L_ML_META,L_ML_META_V2,J_VWAP_BAND_REVERSION",
).split(",") if s.strip()]
LIVE_RESEARCH_DIRECTIONS = [d.strip() for d in os.getenv(
    "LIVE_RESEARCH_DIRECTIONS", "long").split(",") if d.strip()]
LIVE_RESEARCH_TELEGRAM = os.getenv("LIVE_RESEARCH_TELEGRAM", "true").lower() == "true"
# Optional separate Telegram chat for research/paper alerts, so they don't
# mix with the production alerts. Empty = same chat as production.
LIVE_RESEARCH_CHAT_ID = os.getenv("LIVE_RESEARCH_CHAT_ID", "")
# Where state lives between cron runs (the workflow caches this folder).
LIVE_STATE_DIR = os.getenv("LIVE_STATE_DIR", "live_state")
# A signal found late (e.g. the cron skipped a few runs) is still
# paper-traded for backtest parity, but only alerted if its candle closed
# within this many minutes - a 40-minute-old entry alert isn't actionable.
LIVE_ALERT_MAX_AGE_MIN = 15
# L_ML_META_V2 with its live context (fresh tape + news overlay). False =
# backtest-identical V2 (no news), for strict parity.
LIVE_RESEARCH_V2_LIVE_CONTEXT = os.getenv("LIVE_RESEARCH_V2_LIVE_CONTEXT", "true").lower() == "true"
