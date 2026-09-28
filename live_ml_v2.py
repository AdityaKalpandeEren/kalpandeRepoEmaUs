"""
Live runner for model L_ML_META_V2 - alerts on new V2 signals AND keeps
watching every trade it alerted on until it is closed.

    python live_ml_v2.py                         # watchlist.txt, long only
    python live_ml_v2.py --symbols META,NVDA,AAPL
    python live_ml_v2.py --no-telegram           # console only

Separate from main.py on purpose: main.py's live alerts (EMA cross /
VWAP retest) are untouched by anything here.

Every poll, for each symbol:
  1. Uses CLOSED 5-min candles only (Yahoo's last row is the bar still
     forming - acting on it would be acting on a price that can still
     change before the bar closes).
  2. Runs L_ML_META_V2 exactly as the backtest does (same enrichment,
     same regime gate, same model function) with the live context:
     fresh VIX/VXN/QQQ tape, and the news-catalyst overlay - a strong
     opposing headline for the stock or the market vetoes the entry.
  3. For every open V2 trade, checks in this order and sends an EXIT
     alert on the first that fires:
        STOP / TARGET        - the candle's range touched the level
        SESSION_CLOSE        - regular-session flatten time reached
        SHOCK_EXIT           - QQQ moved hard against the trade, or VIX
                               spiked, since entry (same rule as backtest)
        NEWS_EXIT            - fresh bad catalyst for the stock or the
                               market (live-only, see news_catalyst.py)

Trades are tracked in paper_trading/ml_v2_positions.json (survives a
restart) and every close is appended to paper_trading/ml_v2_trades.csv,
so live results can be compared against the backtest over time - which
is the only way the news overlay's value will ever be measured.
"""
import argparse
import csv
import json
import os
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

import config
from data.yfinance_client import get_intraday_candles
from strategy.indicators import enrich
from strategy.market_context import (get_context, set_live_mode, reset_context,
                                     minutes_since_open)
from strategy.strategies import evaluate_all, model_l_ml_meta_v2, _load_ml_v2
from strategy import news_catalyst

MARKET_TZ = ZoneInfo(config.MARKET_TIMEZONE)
POSITIONS_PATH = "paper_trading/ml_v2_positions.json"
TRADES_PATH = "paper_trading/ml_v2_trades.csv"
TRADE_FIELDS = ["symbol", "direction", "signal_time", "entry_time", "entry", "stop_loss", "target",
                "exit_time", "exit_price", "outcome", "r_multiple", "reason", "exit_reason"]


def load_watchlist() -> list:
    with open("watchlist.txt") as f:
        syms = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return list(dict.fromkeys(syms))


def _md(text: str) -> str:
    """Escape Telegram-Markdown specials (model names are full of '_')."""
    for ch in ("_", "*", "`", "["):
        text = text.replace(ch, "\\" + ch)
    return text


def notify(msg: str, telegram: bool):
    print(msg)
    if telegram:
        try:
            from alerts.telegram_bot import send_alert
            send_alert(_md(msg))
        except Exception as e:
            print(f"  (telegram failed: {e!r})")


def closed_candles(df: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Drop the still-forming bar."""
    if df.empty:
        return df
    bar = timedelta(minutes=config.CANDLE_INTERVAL_MINUTES)
    return df[df["timestamp"] + bar <= now].reset_index(drop=True)


def enrich_live(df: pd.DataFrame) -> pd.DataFrame:
    return enrich(
        df.copy(),
        ema_fast=config.EMA_FAST, ema_slow=config.EMA_SLOW,
        atr_period=config.ATR_PERIOD, vol_period=config.VOLUME_AVG_PERIOD,
        struct_lookback=config.STRUCT_LOOKBACK, or_minutes=config.OPENING_RANGE_MINUTES,
        candle_minutes=config.CANDLE_INTERVAL_MINUTES,
    )


# ═══════════════════════════════════════════════════════════════════
# Position book
# ═══════════════════════════════════════════════════════════════════

def load_positions() -> dict:
    try:
        with open(POSITIONS_PATH) as f:
            return json.load(f)
    except Exception:
        return {"open": [], "day_counts": {}, "last_signal": {}}


def save_positions(book: dict):
    os.makedirs(os.path.dirname(POSITIONS_PATH), exist_ok=True)
    tmp = POSITIONS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(book, f, indent=2, default=str)
    os.replace(tmp, POSITIONS_PATH)


def log_close(pos: dict):
    new_file = not os.path.exists(TRADES_PATH)
    os.makedirs(os.path.dirname(TRADES_PATH), exist_ok=True)
    with open(TRADES_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRADE_FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(pos)


def _r_multiple(pos: dict, exit_price: float) -> float:
    risk = abs(pos["entry"] - pos["stop_loss"])
    if risk <= 0:
        return 0.0
    move = exit_price - pos["entry"] if pos["direction"] == "long" else pos["entry"] - exit_price
    return round(move / risk, 3)


def close_position(book, pos, exit_time, exit_price, outcome, why, telegram):
    pos.update(exit_time=str(exit_time), exit_price=round(float(exit_price), 4),
               outcome=outcome, exit_reason=why, r_multiple=_r_multiple(pos, float(exit_price)))
    book["open"].remove(pos)
    log_close(pos)
    notify(f"EXIT {pos['symbol']} {pos['direction'].upper()} [L_ML_META_V2] {outcome} "
           f"@ {pos['exit_price']} ({pos['r_multiple']:+.2f}R) - {why}", telegram)


# ═══════════════════════════════════════════════════════════════════
# Monitoring open trades
# ═══════════════════════════════════════════════════════════════════

def monitor(book, symbol, df, ctx, telegram):
    for pos in [p for p in book["open"] if p["symbol"] == symbol]:
        after = df[df["timestamp"] > pd.Timestamp(pos["signal_time"])]
        if after.empty:
            continue
        if pos.get("entry_time") is None:
            # Filled on the open of the first candle after the signal,
            # exactly like the backtest; stop kept, target unchanged.
            first = after.iloc[0]
            pos["entry_time"] = str(first["timestamp"])
            pos["entry"] = round(float(first["open"]), 4)
            if (pos["direction"] == "long" and pos["entry"] <= pos["stop_loss"]) or \
               (pos["direction"] == "short" and pos["entry"] >= pos["stop_loss"]):
                close_position(book, pos, first["timestamp"], pos["entry"], "NO_FILL",
                               "opened through the stop", telegram)
                continue
        long_ = pos["direction"] == "long"
        sign = 1.0 if long_ else -1.0
        q0, v0 = _num(pos.get("qqq_ref")), _num(pos.get("vix_ref"))
        checked = pos.get("checked_through")
        bars = after if not checked else after[after["timestamp"] > pd.Timestamp(checked)]
        closed = False
        for _, bar in bars.iterrows():
            ts = bar["timestamp"]
            exit_ = None
            if (long_ and bar["low"] <= pos["stop_loss"]) or (not long_ and bar["high"] >= pos["stop_loss"]):
                exit_ = (pos["stop_loss"], "STOP", "stop hit")
            elif (long_ and bar["high"] >= pos["target"]) or (not long_ and bar["low"] <= pos["target"]):
                exit_ = (pos["target"], "TARGET", "target hit")
            elif minutes_since_open(ts) >= config.ML_V2_FLAT_MIN:
                exit_ = (bar["close"], "SESSION_CLOSE", "regular-session close")
            elif config.ML_V2_SHOCK_EXIT_ENABLED:
                q, v = _num(ctx.asof("QQQ", ts)), _num(ctx.asof("VIX", ts))
                if q0 and q and sign * (q / q0 - 1) <= -config.ML_V2_SHOCK_QQQ_PCT:
                    exit_ = (bar["close"], "SHOCK_EXIT", f"QQQ {100*(q/q0-1):+.2f}% since entry")
                elif v0 and v and sign * (v / v0 - 1) >= config.ML_V2_SHOCK_VIX_PCT:
                    exit_ = (bar["close"], "SHOCK_EXIT", f"VIX {100*(v/v0-1):+.1f}% since entry")
            if exit_:
                close_position(book, pos, ts, exit_[0], exit_[1], exit_[2], telegram)
                closed = True
                break
            pos["checked_through"] = str(ts)
        if not closed and config.ML_V2_NEWS_ENABLED:
            why = news_catalyst.exit_check(symbol, pos["direction"])
            if why:
                last = df.iloc[-1]
                close_position(book, pos, last["timestamp"], last["close"], "NEWS_EXIT", why, telegram)


def _num(v):
    """float or None (NaN/None/garbage -> None), for JSON-stored refs."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if v == v and v > 0 else None


# ═══════════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════════

def in_trading_window(now: datetime) -> bool:
    if now.weekday() >= 5:
        return False
    mso = minutes_since_open(pd.Timestamp(now))
    return config.ML_V2_SESSION_START_MIN <= mso <= config.ML_V2_FLAT_MIN + config.CANDLE_INTERVAL_MINUTES


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", help="Comma-separated tickers (default: watchlist.txt)")
    p.add_argument("--directions", default="long",
                   help="long (default - cash equity) or long,short")
    p.add_argument("--no-telegram", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    telegram = not args.no_telegram
    directions = tuple(d.strip() for d in args.directions.split(",") if d.strip())
    symbols = ([s.strip().upper() for s in args.symbols.split(",")] if args.symbols
               else load_watchlist())

    set_live_mode(True)
    bundle = _load_ml_v2()
    if bundle is None:
        print(f"No V2 model at {config.ML_V2_MODEL_PATH}. Build + train it first:\n"
              f"  python -m backtest.ml.build_dataset_v2\n"
              f"  python -m backtest.ml.train_meta_model_v2")
        return
    print(f"L_ML_META_V2 live: {len(symbols)} symbols | directions {list(directions)} | "
          f"model {bundle.get('config')} trained through {bundle.get('trained_through')} | "
          f"news overlay {'ON' if config.ML_V2_NEWS_ENABLED else 'OFF'}"
          f"{' (LLM)' if config.ML_V2_NEWS_LLM_ENABLED else ''}")

    book = load_positions()
    today = None
    models = {"L_ML_META_V2": model_l_ml_meta_v2}

    while True:
        now = datetime.now(MARKET_TZ)
        if now.date() != today:
            today = now.date()
            reset_context()
            book["day_counts"], book["last_signal"], book["day_total"] = {}, {}, 0
            # A position still open from an earlier day means the runner
            # wasn't running at that day's close - its real exit is
            # unknown, so log it as such rather than guess an R-multiple
            # from today's (overnight-gapped) prices.
            for pos in [p for p in book["open"]
                        if pd.Timestamp(p["signal_time"]).tz_convert(MARKET_TZ).date() < today]:
                pos.update(outcome="STALE_UNKNOWN", r_multiple="",
                           exit_reason="runner was not running at that session's close")
                book["open"].remove(pos)
                log_close(pos)
                print(f"Stale open position from an earlier day logged as unknown: {pos['symbol']}")
            save_positions(book)
        if not in_trading_window(now):
            print(f"[{now:%H:%M:%S} ET] outside the V2 window "
                  f"({len(book['open'])} open) - sleeping")
            time.sleep(60)
            continue

        ctx = get_context()
        ctx.refresh_live()
        for symbol in symbols:
            try:
                raw = get_intraday_candles(symbol, config.CANDLE_INTERVAL_MINUTES)
                df = closed_candles(raw, now)
                if len(df) < config.MIN_WARMUP_CANDLES:
                    continue
                monitor(book, symbol, df, ctx, telegram)

                enriched = enrich_live(df)
                for sig in evaluate_all(symbol, enriched, models, directions):
                    key = f"{symbol}|{sig.direction}"
                    if book["day_counts"].get(key, 0) >= config.MAX_TRADES_PER_SYMBOL_DAY:
                        continue
                    last = book["last_signal"].get(key)
                    cooldown = timedelta(minutes=config.SIGNAL_COOLDOWN_CANDLES
                                         * config.CANDLE_INTERVAL_MINUTES)
                    sig_ts = pd.Timestamp(sig.candle_time)
                    if last and sig_ts - pd.Timestamp(last) < cooldown:
                        continue
                    if any(p["symbol"] == symbol and p["direction"] == sig.direction
                           for p in book["open"]):
                        continue
                    if book.get("day_total", 0) >= config.ML_V2_MAX_TRADES_PER_DAY:
                        continue
                    book["open"].append({
                        "symbol": symbol, "direction": sig.direction,
                        "signal_time": str(sig_ts), "entry_time": None, "entry": sig.entry,
                        "stop_loss": sig.stop_loss, "target": sig.target, "reason": sig.reason,
                        "qqq_ref": ctx.asof("QQQ", sig_ts), "vix_ref": ctx.asof("VIX", sig_ts),
                    })
                    book["day_counts"][key] = book["day_counts"].get(key, 0) + 1
                    book["day_total"] = book.get("day_total", 0) + 1
                    book["last_signal"][key] = str(sig_ts)
                    notify(f"ENTRY {symbol} {sig.direction.upper()} [L_ML_META_V2] "
                           f"~{sig.entry} stop {sig.stop_loss} target {sig.target} | "
                           f"{sig.reason}", telegram)
            except Exception as e:
                print(f"[{now:%H:%M:%S} ET] {symbol}: error {e!r}")
        save_positions(book)
        time.sleep(config.POLL_SECONDS)


if __name__ == "__main__":
    main()
