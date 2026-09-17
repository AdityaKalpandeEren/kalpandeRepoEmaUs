# US Market Trade Alert Bot (Telegram)

Same two-signal logic as the BTC and NSE bots — **EMA-cross** and
**VWAP-retest** — applied to US stocks, indices, and futures, using
[Yahoo Finance](https://finance.yahoo.com) via the free `yfinance`
library as the data source. **It never places trades — you execute
manually.**

This is the simplest of the three to run: **no API key, no account
signup, no daily token refresh.** Yahoo Finance's intraday endpoint is
free and keyless. The only credential you need at all is your Telegram
bot token.

1. **EMA-cross** — EMA(20) crossover + price above session VWAP +
   volume > 2x average.
2. **VWAP retest** — price has been trading above session VWAP
   (established uptrend), the candle dips to touch VWAP as support and
   closes back above it on a bullish candle, plus a buy/sell volume
   split ("aggressor") using the same geometric close-position-in-range
   method as the companion Volume Footprint TradingView indicator.

Each setup fires **at most once per symbol per day**, independently.

## What tickers work

Yahoo Finance uses one ticker format for all three asset types:

| Type | Format | Examples |
|---|---|---|
| Stocks | plain symbol | `AAPL`, `MSFT`, `NVDA`, `TSLA` |
| Indices | caret prefix | `^GSPC` (S&P 500), `^IXIC` (Nasdaq Composite), `^DJI` (Dow), `^VIX` |
| Futures | `=F` suffix | `ES=F` (S&P 500), `NQ=F` (Nasdaq 100), `CL=F` (crude oil), `GC=F` (gold) |

Edit `watchlist.txt` — one ticker per line.

## 1. Create your Telegram bot

1. Message **@BotFather** on Telegram → `/newbot` → follow the prompts
   → copy the bot token it gives you.
2. Message your new bot anything, then visit
   `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in a browser
   to find your numeric `chat.id`.

## 2. Install and configure

```bash
cd us_alert_bot
pip install -r requirements.txt

cp .env.example .env
# edit .env: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
```

## 3. Test Telegram wiring (before worrying about market data)

```bash
python -c "from alerts.telegram_bot import send_alert; send_alert('✅ Test message from us_alert_bot')"
```

Confirm it lands in your Telegram chat before moving on.

## 4. Run a scan

```bash
python scan_once.py
```

During market hours (9:30 AM–4:00 PM ET, weekdays — the script checks
this itself using US Eastern time, DST-aware) this prints a status line
per symbol and sends a Telegram alert for any real signal. Outside
market hours it prints one line and exits immediately.

For continuous local running instead of one-shot:
```bash
python main.py
```

## 5. Run it for free on a schedule (GitHub Actions)

Push this project to its own GitHub repo, same pattern as your other
two bots:

```bash
git init
git add .
git commit -m "Initial US alert bot: EMA-cross + VWAP-retest signals"
git branch -M main
git remote add origin https://github.com/<you>/<repo-name>.git
git push -u origin main
```

Then repo → **Settings → Secrets and variables → Actions** → add:
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

That's it — only two secrets, since there's no exchange token to
manage. `.github/workflows/us-alert-scan.yml` runs on a schedule
during US market hours automatically from here on. Test it once
manually first: repo → **Actions** tab → **US Alert Scan** →
**Run workflow**, during market hours, and check the run's logs.

## VWAP logic — exactly which timeline it uses

This is worth stating precisely, because "VWAP" means different things
on different platforms and the difference changes which alerts fire.

**The formula** (`strategy/indicators.py` → `add_vwap`), identical in
the US, NSE and BTC bots:

```
typical_price = (high + low + close) / 3
VWAP = cumsum(typical_price × volume) / cumsum(volume)
```

**The timeline it accumulates over** — this is the part that differs
between bots:

| Bot | VWAP resets | Why |
|---|---|---|
| **US (this bot), default** | 4:00 AM ET, first pre-market candle | `MARKET_OPEN_HOUR=4` + `prepost=True`, so the first candle of "today" is pre-market |
| **US, `VWAP_REGULAR_SESSION_ONLY=true`** | 9:30 AM ET regular open | The "textbook" VWAP most traders mean |
| **NSE** | 9:15 AM IST session open | NSE has no pre-market candle feed here |
| **BTC** | never (rolling 24h window) | Crypto has no session boundary to reset on |

So by default **this bot's VWAP includes pre-market price and volume.**
Pre-market is thin, so a few hundred shares at an odd price can drag
VWAP noticeably away from where a 9:30-anchored VWAP would sit — which
means a "price reclaimed VWAP" alert at 10:00 AM may not match what
you see on a broker chart that anchors at 9:30.

To switch to the 9:30 anchor, set in `.env`:
```
VWAP_REGULAR_SESSION_ONLY=true
```

With that on, pre-market candles get `vwap = NaN`, and every
`check_*()` function already treats a missing VWAP as "no signal". So
the bot still *scans* from 4:00 AM (EMA-cross alerts can still fire
pre-market, since those don't need VWAP), it just won't fire a
VWAP-based signal until 9:30 when a real VWAP value exists. Nothing
else changes — `MARKET_OPEN_HOUR` still controls scanning, this flag
only controls VWAP's anchor.

Both modes are verified by the backtester, so the honest way to pick
between them is to run the same date range twice, once with each
setting, and compare the reports.

## Testing on historical data

If backtesting isn't working, **run the diagnostic first**:

```bash
python -m backtest.check_data_access
```

It walks the stack one layer at a time (dependencies → can we reach
Yahoo → today's intraday bars → a historical range → actual lookback
limits per interval) and tells you which layer broke, rather than
handing you a raw traceback.

**Why live scanning can work while backtesting fails.** These are
genuinely different requests. `yfinance` 1.x depends on `curl_cffi`,
a compiled libcurl wrapper Yahoo now effectively requires — it blocks
plain `requests` sessions as bot traffic. A single "give me today"
call is light and usually slips through. A historical range request
is much heavier and is exactly what Yahoo rate-limits and bot-blocks
first. So "works live, can't backtest" is usually **throttling, not a
bug** — which is why `get_historical_candles()` now splits every range
into small chunks (6 days for 1m bars, 15 for 5/15m, 45 for 1h),
retries with backoff, and pauses between chunks.

**Yahoo's real lookback limits** — asking for more silently returns
fewer days, it doesn't error:

| Interval | History available |
|---|---|
| 1m | ~30 days |
| 2m / 5m / 15m / 30m / 90m | ~60 days |
| 60m / 1h | ~730 days |

So a 6-month backtest on 5-minute bars is **not possible from Yahoo at
all** — not a bug you can fix. Use 1h bars for long ranges, or accept
a ~60-day window on 5m bars. (This is the one real advantage the NSE
bot has: Upstox serves far deeper intraday history.)

### Running a backtest

```bash
# start small to confirm it works
python -m backtest.run_backtest --symbols AAPL --from 2025-09-01 --to 2025-09-12

# then widen
python -m backtest.run_backtest --from 2025-08-15 --to 2025-09-14

# one strategy only
python -m backtest.run_backtest --symbols AAPL,MSFT \
    --from 2025-08-15 --to 2025-09-14 --strategies VWAP_RETEST
```

It prints a trade-by-trade log, then a summary (win rate, avg R,
expectancy — overall, per strategy, per symbol), and writes
`backtest/results/backtest_trades_<timestamp>.csv` and
`backtest/results/backtest_report_<timestamp>.md`.

Same accounting as paper trading below: fills at the *next* candle's
open (never the signal candle's own close, since you couldn't have
acted before it closed), and if one candle touches both stop and
target, the **stop** is counted first — so the win rate is a
conservative lower bound.

If a run comes back with far fewer trading days than you asked for,
that's the lookback limit above, not a failure.

## A note on futures and "session" VWAP

This bot's VWAP is a *session* VWAP — cumulative from the start of
today's fetched candles, same model the NSE bot uses. That's a clean,
well-defined signal for stocks and indices, which have one clear daily
session. Futures technically trade nearly 24/5 on CME Globex, so a
"session" is a looser concept for them; this bot only scans futures
during the 9:30 AM–4:00 PM ET window (the same hours as stocks) so the
VWAP stays meaningful, rather than trying to model the full overnight
session. If you want genuine round-the-clock futures coverage, that's
a bigger change (continuous VWAP that doesn't reset, similar to how
the BTC bot's rolling-window VWAP works) — ask if you want that built.

## Testing accuracy: paper trading

While `main.py` runs continuously, it now also opens a *virtual*
position for every signal, tracks it forward candle by candle (fills
on the next candle's open — not the signal candle's own close, since
in real life you'd only see the Telegram alert after that candle
closed), and closes it on target, stop-loss, or the 8pm ET session
close (squared off at last price) — no real orders, no capital at
risk. Results are logged to `paper_trades.db`.

It's on by default. To turn it off, set in `.env`:
```
PAPER_TRADING_ENABLED=false
```

**This needs `main.py` running continuously** (your own machine, a
free-tier VM, etc.) — not the GitHub Actions `scan_once.py` path —
because a pending paper trade needs to still be there on the *next*
poll to get filled and tracked; an ephemeral CI runner won't persist
that state between runs.

Generate an accuracy report anytime:
```bash
python -m paper_trading.generate_report
# or a specific window
python -m paper_trading.generate_report --from 2025-09-01 --to 2025-09-10
```

This prints a win-rate / avg-R breakdown (overall, per-strategy,
per-symbol) and writes `paper_trading/results/paper_trading_trades_<timestamp>.csv`
and `..._report_<timestamp>.md`. Same accounting rules as the NSE
bot's version:
- **Win rate (target vs stop only)** — of trades that actually
  resolved, what % hit target.
- **Win rate (incl. EOD as loss)** — same, but counts trades still
  open at the session close (squared off, not stopped/targeted) as
  losses — stricter and more conservative.
- **Avg R per trade (expectancy)** — the number that matters for
  whether this makes money over time: average result per trade, in
  multiples of what you risked. Positive = profitable on average even
  below a 50% win rate, as long as winners (capped at
  `config.RISK_REWARD_RATIO`, 1.5R) outweigh losers (-1R) often enough.

If you also want a *historical* backtest here (run the strategies
against real past Yahoo Finance candles, like the NSE bot's
`backtest/run_backtest.py` does against Upstox history), that's a
straightforward addition on top of this — Yahoo's intraday history is
more limited than Upstox's (roughly the last 7 days for 1-minute bars,
60 days for 5/15-minute bars), but the same walk-forward simulator
would work unchanged. Ask if you want that built too.

## Tuning the strategy

Same knobs as the NSE bot, in `config.py`:
- `VOLUME_MULTIPLIER`, `RISK_REWARD_RATIO`, `MAX_RISK_PCT`
- `RETEST_TREND_LOOKBACK`, `RETEST_MIN_CANDLES_ABOVE`, `RETEST_TOUCH_BUFFER_PCT`

## Project structure

```
us_alert_bot/
├── config.py
├── data/yfinance_client.py    # free, keyless Yahoo Finance intraday candles
├── strategy/
│   ├── indicators.py           # EMA, session VWAP (see "VWAP logic" above), avg volume
│   ├── screener.py             # EMA-cross + VWAP-retest + VWAP-broad-test trigger logic
│   └── trade_engine.py         # shared trade-outcome simulation (backtest AND paper trading)
├── backtest/
│   ├── check_data_access.py    # RUN THIS FIRST if historical data isn't working
│   ├── data_loader.py          # fetches + chunks historical candles, splits into per-day sessions
│   ├── simulator.py            # walks each day candle-by-candle, converts signals to trades
│   └── run_backtest.py         # CLI: python -m backtest.run_backtest --from ... --to ...
├── paper_trading/
│   ├── tracker.py              # live virtual-position tracking, driven by main.py's poll loop
│   ├── report.py                # win rate / avg-R / expectancy report
│   └── generate_report.py      # CLI: python -m paper_trading.generate_report
├── alerts/
│   ├── telegram_bot.py         # sends both alert message types ($, ET)
│   └── logger.py               # SQLite alert history
├── watchlist.txt
├── main.py                     # always-on loop (also drives paper trading)
├── scan_once.py                # single-pass scan for GitHub Actions
├── .github/workflows/us-alert-scan.yml
└── .env / requirements.txt
```

## Honest caveat about yfinance

`yfinance` is an unofficial wrapper around Yahoo Finance's public
endpoints, not a documented, contractually-supported API — Yahoo can
change or rate-limit it without notice, and it has broken before and
been fixed by the library's maintainers. For a free, keyless option
across stocks/indices/futures, it's the most practical choice, but if
you ever want a fully-documented paid alternative (Polygon.io, Alpaca,
Databento), the `data/` layer is the only thing you'd need to swap out
— `strategy/` and `alerts/` don't care where the candles came from.
