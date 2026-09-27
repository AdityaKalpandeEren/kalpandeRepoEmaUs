# Handoff: ML V2 filter + live alerts & paper trading for research strategies

Source repo: `us_alert_bot` (branch `v9_4_code_changes`, all work below uncommitted as of 2026-09-27).
Purpose: let a new Claude Code session in ANOTHER repo (e.g. the NSE bot) reuse this logic. Give it this file first:
"Read HANDOFF_ML_V2_AND_LIVE_PAPER.md and port <part> into this repo."

Contents: 1 files · 2 live alerts + paper trading · 3 V2 design · 4 ML lessons · 5 open items ·
**6 full specs of K_RSI2_REVERSION, SCORE_ENGINE, L_ML_META, L_ML_META_V2, J_VWAP_BAND_REVERSION** ·
**7 porting to NSE (what must change)**

---

## 1. What exists (files in us_alert_bot)

| File | Role |
|---|---|
| `strategy/strategies.py` | 12 base research models (A–K, `SCORE_ENGINE`), `L_ML_META` (V1), `L_ML_META_V2`, `evaluate_all()` |
| `strategy/market_context.py` | Point-in-time market + catalyst features (VIX/VXN/VIX3M, QQQ/SPY, breadth, earnings, gap, RVOL); disk cache; live mode |
| `strategy/news_catalyst.py` | Live-only headline scoring (lexicon default, optional Claude), entry veto/boost, exit check |
| `strategy/ml_features_v2.py` | V2 feature list + extraction (shared by training and inference) |
| `strategy/trade_engine.py` | Fill/exit simulation: `simulate_forward_directional`, `simulate_forward_v2`, `build_research_trade` (costs) |
| `backtest/research_simulator.py`, `backtest/run_research.py`, `backtest/research_report.py` | Backtest engine + report (ranking by expectancy, risk sweep) |
| `backtest/ml/build_dataset_v2.py`, `backtest/ml/train_meta_model_v2.py` | V2 dataset builder (parallel) and trainer (train/val/test by date) |
| `backtest/ml/model/meta_model*.joblib` | Trained V1 / V2 models (scikit-learn **1.7.2** — pin it) |
| `paper_trading/research_live.py` | Live Telegram alerts + paper trading + EOD report for chosen research strategies |
| `scan_once.py` | Production one-pass scanner; research pass hooked in AFTER production alerts |
| `live_ml_v2.py` | Standalone long-running V2 runner with open-trade monitoring (optional) |
| `.github/workflows/us-alert-scan.yml` | Actions job: state cache restore/save, concurrency, 12-min timeout, EOD artifact |
| `config.py` | All settings: `ML_V2_*`, `LIVE_RESEARCH_*` blocks at the end |

---

## 2. Live alerts + paper trading (the part to port first)

**Execution model:** a short process (`python scan_once.py`) started by cron at ANY minute. State lives in
`LIVE_STATE_DIR` (`live_state/`), cached between GitHub runs; on a normal machine it just stays on disk.

**Per run:**
1. Production alerts run first, untouched; research code is wrapped in try/except so it can never block them.
2. `ResearchPass.begin()` — file lock (no overlapping runs), load state, day rollover (sends a late EOD report if the previous day's never went out).
3. For each symbol, `process_symbol(symbol, df)` with the SAME candles production fetched:
   - drop the still-forming bar (`timestamp + 5min <= now`)
   - update open/pending trades by **replaying the backtest's own exit functions** on the closed bars
   - evaluate **every candle closed since the last run** (`last_ts` per symbol) with `evaluate_all(...)` → irregular cron can't skip/duplicate
   - per-symbol limits exactly as backtest: cooldown `SIGNAL_COOLDOWN_CANDLES`, max `MAX_TRADES_PER_SYMBOL_DAY` per (strategy, direction)
   - alert only signals whose candle is ≤ `LIVE_ALERT_MAX_AGE_MIN` old; older backlog signals are paper-traded silently
4. `end()` — one batched Telegram message for entries, one for exits (plain text, no Markdown: names contain `_`), save state.
5. After `MARKET_CLOSE_HOUR` (20:00 ET): `finish_day()` — final scan of late candles (no alerts), final exits with `session_over=True`, square off, EOD report once.

**Parity rules (why results equal the backtest):**
- fill = NEXT candle's open; skip if it opens through the stop (status `SKIPPED_GAP`)
- simulator returning `EOD_SQUAREOFF` mid-day means "ran out of candles" → still OPEN; only a real exit after the session
- standard models square off on the day's last candle (after-hours, 19:55 ET); V2 at 15:55 ET
- trade record via `build_research_trade` (same slippage/commission)
- **Verified:** replaying 7 past days with jittered/skipped cron → 142/142 trades identical to `simulate_day_research`

**EOD report:** today's summary per strategy (win %, expectancy R, total R, PF, max DD), risk sweep at
0.25/0.5/0.75/1% of `ACCOUNT_EQUITY`, all-time paper block (`paper_trades_all.csv`, idempotent by trade id),
plus the backtest-format `.md` attached. On demand: `python -m paper_trading.research_live --report [--send]`.

**Config:** `LIVE_RESEARCH_ENABLED`, `LIVE_RESEARCH_STRATEGIES` (default
`K_RSI2_REVERSION,SCORE_ENGINE,L_ML_META,L_ML_META_V2,J_VWAP_BAND_REVERSION`), `LIVE_RESEARCH_DIRECTIONS=long`,
`LIVE_RESEARCH_TELEGRAM`, `LIVE_RESEARCH_CHAT_ID` (separate chat), `LIVE_STATE_DIR`, `LIVE_ALERT_MAX_AGE_MIN=15`,
`LIVE_RESEARCH_V2_LIVE_CONTEXT` (V2 news on/off).

**Workflow requirements:** `concurrency: {group: us-alert-scan, cancel-in-progress: false}`; restore cache with
`restore-keys: live-state-`, save with a per-run key `live-state-${run_id}-${run_attempt}`; paths `live_state` and
`backtest/ml/cache`; cron must fire at least once after 20:00 ET for the EOD report.

---

## 3. L_ML_META_V2 (FROZEN in source repo — do not change there unless the user says so)

Meta-labeling: the 12 base models propose candidates; a classifier picks the best and requires p ≥ threshold.
- **Gates:** regular session entries 9:45–15:00 ET; long vetoes VIX ≥ 30, VIX/VIX3M ≥ 1.05, VIX +12% on day.
- **Features:** V1 candle features + intraday market (SPY/QQQ day & 30-min returns, QQQ vs VWAP, VIX/VXN changes,
  VIX term) + catalyst footprint (rel. strength vs QQQ, gap, time-of-day RVOL, earnings days since/to next, EPS surprise,
  distance from highs) + stop geometry. Excluded from the model (date-label effect on ~40 days):
  `vix, vix_vs_sma50, vxn_vs_sma50, breadth_pct50, breadth_chg5, qqq_vs_sma50_d, spy_vs_sma50_d, spy_vs_sma200_d`.
- **Exits:** stop / target (1.5R) / SHOCK_EXIT (QQQ −0.6% or VIX +8% since entry) / SESSION_CLOSE 15:55 ET.
- **News (live only, not backtestable, not an ML feature):** symbol veto ≤ −0.35, market veto ≤ −0.45,
  aligned boost ≥ +0.35 lowers bar by 0.03, exit alert ≤ −0.45. Headlines must name the company in the title;
  opinion/listicle titles score 0; market feed must be macro-related.
- **Honest status:** out-of-sample test (8 days, 10 trades/day cap) ≈ −0.02R, 4/7 days positive → **no proven edge**.
  Backtests over Jul 29–Sep 25 2026 are IN-SAMPLE for the saved model. Needs forward paper results.

---

## 4. Lessons that apply to any ML trading work

1. **No lookahead:** intraday context = last bar ≤ signal time; daily context = strictly prior sessions; earnings count from the reaction session.
2. **Day-constant regime features act as date labels** on short datasets — exclude until you have many months.
3. **Split by DATE into train / validation / test**; choose model + threshold on validation only; report test once.
4. **Judge by day, not by trade:** one strong day produced a per-trade t = 3.3 that was 0.26 day-clustered. Report days-positive, day-level t, result without the best day, and use a portfolio daily cap.
5. Evaluate the way the model trades (best candidate per bar, cooldown, caps), not per raw candidate row.
6. Pin scikit-learn to the training version when shipping `.joblib` models.

---

## 5. Open items in the source repo

- **Revoke the Telegram bot token** committed in `.env.example` (on `main` too); use secrets only.
- Commit + PR `v9_4_code_changes` → `main` (not done; needs user approval).
- Production `scan_once.py` has no alert de-duplication (322 repeat alerts for 4 symbols in a replayed day) — pre-existing, untouched.
- Offered, not built: `--train-until` for clean out-of-sample backtests; `--research-only` flag for local runs; catalyst watchlist (news-triggered focus list).
- Pre/post-market scanning (4:00–20:00 ET, `prepost=True`) and VWAP from 4:00 AM are unchanged vs production (verified: VWAP diff 0.0).

---

## 6. Model specs: K_RSI2_REVERSION, SCORE_ENGINE, L_ML_META, L_ML_META_V2, J_VWAP_BAND_REVERSION

Everything below is exactly what the code in `strategy/strategies.py` + `strategy/indicators.py` +
`strategy/regime.py` does, with the US defaults from `config.py`. Port these pieces **together** — every model
depends on the shared indicators, regime gate, stop/target convention and engine rules in 6.1.

### 6.1 Shared foundation (all five models)

**Candles:** 5-min, one session per DataFrame (indicators reset every day). US uses pre+post market (4:00–20:00 ET).

**Indicators (`enrich()`, computed once per day, all causal):**
| Column | Definition |
|---|---|
| `ema_9`, `ema_21` | EMA of close (`EMA_FAST=9`, `EMA_SLOW=21`); `ema_10/20/50` also computed |
| `ema_9_slope`, `ema_21_slope` | `(ema - ema.shift(3)) / (ema.shift(3) * 3)` — fractional change per candle |
| `vwap` | cumulative Σ(typical price × vol) / Σ vol from the first candle of the day; typical = (H+L+C)/3 |
| `vwap_dist_pct` | `(close − vwap) / vwap` |
| `vwap_z` | `(close − vwap) / rolling_std(close − vwap, 20, min 5)` |
| `vwap_slope` | `(vwap − vwap.shift(3)) / (vwap.shift(3) * 3)` |
| `rsi_2` | Wilder RSI, period 2 (`ewm(alpha=1/2)` of gains/losses); 100 when avg loss = 0 |
| `atr_14` | Wilder ATR of true range (`ewm(alpha=1/14)`) |
| `atr_pct_rank` | rolling 50-candle percentile rank of `atr_14` (min 5) |
| `rvol` | volume / mean of the PREVIOUS 20 candles' volume (shifted) |
| `struct_high/low` | rolling 10-candle max high / min low, **shifted 1** |
| `hh, hl, lh, ll` | `struct_high > struct_high.shift(10)` etc. |
| `or_high/or_low` | first 30 min high/low, hidden until the range completes |

**Regime (`classify(row)`):**
- volatility: `atr_pct_rank ≥ 0.85` high, `≤ 0.20` low, else normal
- direction votes (−3..+3): ema_9 vs ema_21 (±1); ema_9_slope beyond ±0.0002 (±1); close vs vwap (±1)
- `wide` = |ema_9 − ema_21| / close ≥ 0.001
- votes ≥3 & wide → strong_bull; ≥2 → weak_bull; ≤−3 & wide → strong_bear; ≤−2 → weak_bear; else neutral
- `trend_strength = min(1, |votes|/3 × (1.5 if wide else 0.75))`

**No-trade gate (`is_tradeable`)**, applied before any model: block `neutral`; block volatility `high`; no longs in
`strong_bear`; no shorts in `strong_bull`. Shorts also need ≥ 12 candles into the session.

**Stop / target (`_stop_and_target`) — one convention for every model:**
- long stop = close − 1.5 × ATR; if a structural level is given, widen to `level − 0.25 × ATR` when that is LOWER (never tighten)
- short: ATR multiple 1.75 (1.5 + 0.25), widen to `level + 0.25 × ATR` when higher
- target = entry ± 1.5 × risk (`RISK_REWARD_RATIO`)
- skip the signal if risk / entry > 1.5% (`MAX_RISK_PCT`)

**Engine rules (backtest + live identical):** start after 25 candles; per symbol/day max 3 trades per (model,
direction); 6-candle cooldown; fill at next candle open (skip if through the stop); stop checked before target when a
candle touches both; square off at the day's last candle; costs 2 bps slippage + 1 bps commission per side;
R-multiple measured against the ORIGINAL stop distance.

### 6.2 K_RSI2_REVERSION — Connors RSI(2) dip-buy (LONG only)
Fires when ALL hold on the signal candle:
- `rsi_2 ≤ 10` (`RSI2_OVERSOLD`)
- `close > ema_50` (`RSI2_TREND_FILTER_EMA` = 50) — dip in an uptrend, not a falling knife
- `close > open` — the candle already turning up

Structural level for the stop = candle low. Score 70. (Connors' 75–79% win rate is for DAILY bars multi-day;
this is an intraday adaptation — test it, don't expect it.)

### 6.3 J_VWAP_BAND_REVERSION — buy the lower VWAP band (LONG only)
Fires when ALL hold:
- `vwap_z ≤ −2.0` (`VWAP_BAND_Z_THRESHOLD`) — stretched 2+ SD below VWAP (z ≠ 0 and vwap > 0)
- `close > open` — reverting
- `rvol ≥ 1.0` (`VWAP_BAND_MIN_RVOL`)

Structural level = candle low. Score 70. (Source claim: QuantConnect study, ~61% win at ~1.4:1 on 100 NASDAQ names.)

### 6.4 SCORE_ENGINE — weighted six-component score (long + short)
Six continuous 0..1 components (`_component_scores`), `sign` = +1 long / −1 short:
| Component | Formula | Weight |
|---|---|---|
| trend | ordered (ema_9 > ema_21 for long) ? min(1, (|ema_9−ema_21|/close) / 0.004) : 0 | 0.25 |
| momentum | clip(sign × ema_9_slope / 0.0008, 0, 1) | 0.15 |
| volume | clip((rvol − 1) / (thr − 1), 0, 1), thr = 2.0 long / 2.5 short | 0.15 |
| vwap | z = sign × vwap_z; z ≤ 0 → 0; z ≤ 1 → z; else max(0, 1 − (z − 1)/2.5) | 0.20 |
| volatility | max(0, 1 − |atr_pct_rank − 0.5| / 0.5) | 0.10 |
| price_action | 0.6 × (1 if hh else 0.5 if hl else 0) [ll/lh for short] + 0.4 × (close > open for long) | 0.15 |

Fires when `100 × Σ weight × component ≥ 55` AND trend ≥ 0.10 AND vwap ≥ 0.10 (floor only on those two).
Structural level = candle low (long) / high (short). Score = the total.

### 6.5 L_ML_META — V1 meta-label filter
- Runs all 12 base models (A–K + SCORE_ENGINE) on the candle; for each candidate builds a 33-feature row:
  `rsi_2, atr_pct_rank, rvol, vwap_dist_pct, vwap_z, vwap_slope, ema_fast_slope, ema_slow_slope`, the six component
  scores, `trend_strength, session_position (candle count), direction_long, regime_vol_high/low,
  regime_trend_bull/bear` + one-hot of the 12 source models.
- `HistGradientBoostingClassifier(max_depth=4, max_iter=200, learning_rate=0.05, l2=1.0)`, label = R > 0.
- Takes the highest-probability candidate if p ≥ 0.55 (`ML_META_MIN_PROB`); stop/target stay the base model's.
- Training: `backtest/ml/build_dataset.py` → `train_meta_model.py` (last 30% by time held out).
- Status: never beat baseline out of sample.

### 6.6 L_ML_META_V2
See section 3 for the design; code in `strategies.py` (`ml_v2_candidates`, `model_l_ml_meta_v2`),
`market_context.py`, `ml_features_v2.py`, `news_catalyst.py`, trainer in section 4's lessons.
Pipeline: `build_dataset_v2.py` (parallel, labels with V2 exits) → `train_meta_model_v2.py` (8 configs × 21 thresholds
on validation, min 30 trades / 4 days, 10 trades/day cap, day-level verdict) → `meta_model_v2.joblib`
(chosen: `hgb_shallow`: depth 3, lr 0.03, 250 iters, min_samples_leaf 150, l2 5, max_features 0.5; threshold 0.55).

---

## 7. Porting to the NSE bot — what must change

**Do NOT copy the trained `.joblib` models.** They learned US stocks, US hours and US context. Port the code,
rebuild the datasets on NSE data, retrain, and re-validate. Everything rule-based (K, J, SCORE_ENGINE) ports directly.

| Area | US version | NSE version |
|---|---|---|
| Timezone | `America/New_York` | `Asia/Kolkata` |
| Session | 4:00–20:00 ET incl. pre/post | 9:15–15:30 IST, no extended hours (pre-open 9:00–9:08 is an auction, not candles) |
| VWAP start | first candle (4:00 ET) | 9:15 IST candle |
| Square-off | day's last candle (19:55 ET); V2 15:55 | intraday MIS auto square-off ~15:20 by brokers — square off at **15:15–15:20**, not 15:30 |
| V2 entry window | 9:45–15:00 ET | e.g. 9:30–14:45 IST (re-derive from NSE data by time bucket, as done for US) |
| Warmup | 25 candles | 25 × 5 min = 11:20 IST if counted from 9:15 — probably too late; consider 12–15 candles and re-test |
| Shorts | long-only live (cash equity) | intraday shorting allowed (MIS) — test both directions, but US shorts were strongly negative |
| Costs | 2 + 1 bps per side | much higher: brokerage + STT (0.025% sell side intraday) + exchange + GST + stamp ≈ 3–6 bps per side — model it properly or results will be too optimistic |
| Data | yfinance 5-min (`.NS` suffix works, 60-day cap) | yfinance `RELIANCE.NS` or broker API (Kite/Upstox) — broker data is more reliable and longer |
| Volatility index | ^VIX, ^VXN, ^VIX3M | **^INDIAVIX** only — no term-structure equivalent; drop `vix_term`, VIX3M veto, VXN features |
| Market tape | QQQ / SPY | NIFTY 50 (`^NSEI`), BANKNIFTY (`^NSEBANK`) — rs_vs_index, index vs VWAP, index returns |
| Breadth universe | 76 US large caps | NIFTY 50 constituents |
| Earnings | `yfinance.get_earnings_dates` | often missing for `.NS` — use NSE corporate results calendar or drop earnings features |
| News | Yahoo headlines, macro regex (Fed, CPI, tariffs) | Yahoo `.NS` feeds are thin — consider Moneycontrol/ET RSS or broker news; macro terms: RBI, repo rate, inflation/CPI India, FII/DII, budget, SEBI |
| V2 vetoes | VIX ≥ 30, VIX/VIX3M ≥ 1.05, VIX +12% day | India VIX thresholds need re-deriving (typical range differs; e.g. start with India VIX ≥ 25 and +15% day, then validate) |
| Holidays | NYSE calendar | NSE holiday list (and special Muhurat session) |

**Suggested order for the NSE session:**
1. Port shared foundation (6.1) + K, J, SCORE_ENGINE; backtest on NSE data with NSE costs.
2. Port live alerts + paper trading (section 2) and verify parity by replaying past days (the 142/142 test).
3. Only then port V1/V2 ML: rebuild datasets on NSE data, retrain, apply the lessons in section 4 — expect to need
   more history than 60 days before any ML result means anything.
