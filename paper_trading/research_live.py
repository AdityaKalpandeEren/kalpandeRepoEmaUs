"""
Live Telegram alerts + paper trading for selected RESEARCH strategies
(config.LIVE_RESEARCH_STRATEGIES - by default K_RSI2_REVERSION,
SCORE_ENGINE, L_ML_META, L_ML_META_V2, L_ML_META_V2_2, L_ML_META_V1_2,
J_VWAP_BAND_REVERSION),
long only, with an end-of-day report in the same format as
backtest/run_research.py.

Designed for scan_once.py's execution model: a short process started by
a cron trigger at ANY minute (not aligned to candle boundaries), that
must remember what it already did. Everything it needs to remember lives
in config.LIVE_STATE_DIR (the GitHub workflow caches that folder between
runs; on a normal machine it simply stays on disk).

BACKTEST PARITY is the design rule - the paper results are only worth
comparing to backtests if "a trade" means the same thing in both:
  - signals: strategy.strategies.evaluate_all on the same enriched
    candles (same regime gate, same models, same config)
  - only CLOSED candles are evaluated; every candle closed since the last
    run is processed in order, so an irregular cron can't skip or repeat
    one
  - same per-symbol limits as backtest/research_simulator.py: cooldown
    config.SIGNAL_COOLDOWN_CANDLES, max config.MAX_TRADES_PER_SYMBOL_DAY
    per (strategy, direction)
  - fill at the NEXT candle's open; skipped if that open is already
    through the stop (the backtest skips those too)
  - exits replayed with the backtest's own functions
    (simulate_forward_directional / simulate_forward_v2 for V2), and the
    trade record built by build_research_trade with the same costs
  - standard models square off on the day's last candle (after-hours,
    like the backtest); L_ML_META_V2 at 15:55 ET per its own rules

One deliberate live-only difference: L_ML_META_V2 runs with its live
context, which includes the news overlay (config.ML_V2_NEWS_ENABLED) -
the backtest can't have that. Set ML_V2_NEWS_ENABLED=false for strict
parity.

    python -m paper_trading.research_live --report        # print today's report
    python -m paper_trading.research_live --report --send # ...and send it
"""
import argparse
import csv
import json
import math
import os
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import requests

import config
from strategy.indicators import enrich
from strategy.trade_engine import (simulate_forward_directional, simulate_forward_v2,
                                   build_research_trade, ResearchTrade)

MARKET_TZ = ZoneInfo(config.MARKET_TIMEZONE)
STATE_FILE = "state.json"
ALL_TRADES_CSV = "paper_trades_all.csv"


# ═══════════════════════════════════════════════════════════════════
# Telegram (plain text - strategy names are full of '_', which breaks
# the Markdown parse mode alerts/telegram_bot.py uses)
# ═══════════════════════════════════════════════════════════════════

def _tg(method: str, **kwargs):
    if not config.LIVE_RESEARCH_TELEGRAM or not config.TELEGRAM_BOT_TOKEN:
        return
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/{method}"
    chat_id = config.LIVE_RESEARCH_CHAT_ID or config.TELEGRAM_CHAT_ID
    try:
        resp = requests.post(url, data={"chat_id": chat_id, **kwargs.pop("data", {})},
                             timeout=20, **kwargs)
        if resp.status_code != 200:
            print(f"[research-live] Telegram {method} failed: {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        print(f"[research-live] Telegram {method} error: {e!r}")


def send_text(text: str):
    # Telegram caps a message at 4096 chars - split on line boundaries.
    chunk = ""
    for line in text.splitlines(keepends=True):
        if len(chunk) + len(line) > 3900:
            _tg("sendMessage", data={"text": chunk})
            chunk = ""
        chunk += line
    if chunk.strip():
        _tg("sendMessage", data={"text": chunk})


def send_document(path: str, caption: str = ""):
    with open(path, "rb") as f:
        _tg("sendDocument", data={"caption": caption[:1000]}, files={"document": f})


# ═══════════════════════════════════════════════════════════════════
# State
# ═══════════════════════════════════════════════════════════════════

def _path(name: str) -> str:
    os.makedirs(config.LIVE_STATE_DIR, exist_ok=True)
    return os.path.join(config.LIVE_STATE_DIR, name)


def _new_state(day: str) -> dict:
    return {"date": day, "last_ts": {}, "counts": {}, "last_fired": {},
            "trades": [], "report_sent": False}


def load_state() -> dict:
    try:
        with open(_path(STATE_FILE)) as f:
            return json.load(f)
    except Exception:
        return _new_state("")


def save_state(state: dict):
    tmp = _path(STATE_FILE + f".tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1, default=str)
    os.replace(tmp, _path(STATE_FILE))


@contextmanager
def state_lock():
    """Stops two overlapping runs (a slow run + the next cron tick) from
    both editing the state. The loser skips its research pass - the
    next run catches up on every candle it missed anyway."""
    import fcntl
    fh = open(_path("lock"), "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        yield False
        return
    try:
        yield True
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


# ═══════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════
_models = None


def _selected_models() -> dict:
    global _models
    if _models is None:
        from strategy.strategies import ENTRY_MODELS
        wanted = config.LIVE_RESEARCH_STRATEGIES
        unknown = [w for w in wanted if w not in ENTRY_MODELS]
        if unknown:
            print(f"[research-live] unknown strategies ignored: {unknown}")
        _models = {k: ENTRY_MODELS[k] for k in wanted if k in ENTRY_MODELS}
        if any(k in _models for k in ("L_ML_META_V2", "L_ML_META_V2_2", "L_ML_META_V1_2")):
            # V2's live mode: fresh VIX/QQQ tape + news overlay (V1_2: tape only).
            from strategy.market_context import set_live_mode
            set_live_mode(config.LIVE_RESEARCH_V2_LIVE_CONTEXT)
    return _models


def _enrich(day_df: pd.DataFrame) -> pd.DataFrame:
    # Identical to backtest/research_simulator.py::_enrich_day.
    return enrich(
        day_df.copy(),
        ema_fast=config.EMA_FAST, ema_slow=config.EMA_SLOW,
        atr_period=config.ATR_PERIOD, vol_period=config.VOLUME_AVG_PERIOD,
        struct_lookback=config.STRUCT_LOOKBACK, or_minutes=config.OPENING_RANGE_MINUTES,
        candle_minutes=config.CANDLE_INTERVAL_MINUTES,
    )


def closed_candles(df: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Drop the bar that is still forming (Yahoo returns it as the last row)."""
    if df is None or df.empty:
        return df
    bar = timedelta(minutes=config.CANDLE_INTERVAL_MINUTES)
    return df[df["timestamp"] + bar <= now].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# Per-pass lifecycle
# ═══════════════════════════════════════════════════════════════════

class ResearchPass:
    """One scan_once.py run: begin() -> process_symbol() per symbol -> end()."""

    def __init__(self, now: datetime):
        self.now = now
        self.day = now.date().isoformat()
        self.state = None
        self.new_alerts = []
        self.closed_alerts = []
        self._lock_cm = None

    def begin(self) -> bool:
        self._lock_cm = state_lock()
        if not self._lock_cm.__enter__():
            print("[research-live] another run holds the state lock - skipping this pass")
            self._lock_cm = None
            return False
        self.state = load_state()
        if self.state.get("date") != self.day:
            self._rollover()
        _selected_models()
        return True

    def _rollover(self):
        """New trading day: finish the previous one if its EOD report never
        went out (e.g. no cron run after the close), then reset."""
        prev = self.state
        if prev.get("date") and prev.get("trades") and not prev.get("report_sent"):
            for t in prev["trades"]:
                if t["status"] == "OPEN":
                    _close_at_last_seen(t)
                elif t["status"] == "PENDING":
                    t["status"] = "NEVER_FILLED"
            send_eod_report(prev, late=True)
        self.state = _new_state(self.day)
        save_state(self.state)

    # ---------- per symbol ----------

    def process_symbol(self, symbol: str, raw_df: pd.DataFrame, allow_new: bool = True):
        """raw_df = today's candles as scan_once already fetched them."""
        if self.state is None:
            return
        df = closed_candles(raw_df, self.now)
        if df is None or df.empty:
            return
        self._update_trades(symbol, df)
        if allow_new:
            self._scan_new_candles(symbol, df)

    def _scan_new_candles(self, symbol: str, df: pd.DataFrame):
        from strategy.strategies import evaluate_all
        models = _selected_models()
        if not models or len(df) < config.MIN_WARMUP_CANDLES + 1:
            return
        last_ts = self.state["last_ts"].get(symbol)
        last_ts = pd.Timestamp(last_ts) if last_ts else None
        todo = [i for i in range(config.MIN_WARMUP_CANDLES, len(df))
                if last_ts is None or df["timestamp"].iat[i] > last_ts]
        if not todo:
            return
        enriched = _enrich(df)
        directions = tuple(config.LIVE_RESEARCH_DIRECTIONS)
        for i in todo:
            sub = enriched.iloc[: i + 1]
            try:
                signals = evaluate_all(symbol, sub, models, directions)
            except Exception as e:
                print(f"[research-live] {symbol} evaluate error: {e!r}")
                signals = []
            for sig in signals:
                key = f"{symbol}|{sig.strategy}|{sig.direction}"
                if self.state["counts"].get(key, 0) >= config.MAX_TRADES_PER_SYMBOL_DAY:
                    continue
                if i - self.state["last_fired"].get(key, -10**9) < config.SIGNAL_COOLDOWN_CANDLES:
                    continue
                self.state["counts"][key] = self.state["counts"].get(key, 0) + 1
                self.state["last_fired"][key] = i
                trade = {
                    "id": f"{key}|{sig.candle_time}",
                    "symbol": symbol, "strategy": sig.strategy, "direction": sig.direction,
                    "regime": sig.regime, "trend_strength": sig.trend_strength,
                    "score": sig.score, "reason": sig.reason,
                    "exit_mode": getattr(sig, "exit_mode", "standard"),
                    "signal_time": str(sig.candle_time), "signal_price": sig.entry,
                    "stop_loss": sig.stop_loss, "target": sig.target,
                    "status": "PENDING",
                }
                self.state["trades"].append(trade)
                age_min = (self.now - pd.Timestamp(sig.candle_time).to_pydatetime()).total_seconds() / 60
                if age_min <= config.LIVE_ALERT_MAX_AGE_MIN + config.CANDLE_INTERVAL_MINUTES:
                    self.new_alerts.append(trade)
                else:
                    print(f"[research-live] {symbol} {sig.strategy}: backlog signal from "
                          f"{sig.candle_time} paper-traded but not alerted (stale)")
        self.state["last_ts"][symbol] = str(df["timestamp"].iat[todo[-1]])

    def _update_trades(self, symbol: str, df: pd.DataFrame, session_over: bool = False):
        ts_index = {str(t): k for k, t in enumerate(df["timestamp"])}
        for t in self.state["trades"]:
            if t["symbol"] != symbol or t["status"] not in ("PENDING", "OPEN"):
                continue
            sig_idx = ts_index.get(t["signal_time"])
            if sig_idx is None:
                continue
            fill_idx = sig_idx + 1
            if fill_idx >= len(df):
                continue   # next candle hasn't closed yet
            entry = float(df["open"].iat[fill_idx])
            if t["status"] == "PENDING":
                if (t["direction"] == "long" and entry <= t["stop_loss"]) or \
                   (t["direction"] == "short" and entry >= t["stop_loss"]):
                    t["status"] = "SKIPPED_GAP"   # the backtest never books these
                    continue
                t.update(status="OPEN", entry=entry, entry_time=str(df["timestamp"].iat[fill_idx]))

            if t["exit_mode"] == "v2":
                from strategy.market_context import get_context
                exit_time, exit_price, outcome, exit_idx = simulate_forward_v2(
                    df, fill_idx, t["direction"], entry, t["stop_loss"], t["target"], get_context())
            else:
                exit_time, exit_price, outcome, exit_idx = simulate_forward_directional(
                    df, fill_idx, t["direction"], entry, t["stop_loss"], t["target"])
            t["last_price"] = float(df["close"].iat[-1])
            t["last_time"] = str(df["timestamp"].iat[-1])
            # EOD_SQUAREOFF from the simulator only means "ran out of candles".
            # It's a real exit only once the session is over.
            if outcome == "EOD_SQUAREOFF" and not session_over:
                continue
            _finalize(t, exit_time, exit_price, outcome, exit_idx - fill_idx)
            self.closed_alerts.append(t)

    # ---------- end of pass ----------

    def end(self):
        if self.state is None:
            return
        try:
            entries, exits = _alertable(self.new_alerts), _alertable(self.closed_alerts)
            if entries:
                send_text(_format_entries(entries))
            if exits:
                send_text(_format_exits(exits))
            save_state(self.state)
        finally:
            if self._lock_cm is not None:
                self._lock_cm.__exit__(None, None, None)
                self._lock_cm = None

    def finish_day(self, fetch):
        """After the session close: final exit check on every open trade
        with the day's last candles, square off what's left, send the EOD
        report once. `fetch(symbol)` returns today's candles."""
        if self.state is None or self.state.get("report_sent"):
            return
        # Every symbol scanned today gets one last pass: candles that closed
        # after the final in-session run can still hold a signal that fills
        # on the day's last candle (the backtest books those too). Paper-
        # traded for parity, but not alerted - it's after the close.
        symbols = set(self.state["last_ts"]) | {t["symbol"] for t in self.state["trades"]
                                                 if t["status"] in ("PENDING", "OPEN")}
        for symbol in sorted(symbols):
            try:
                df = closed_candles(fetch(symbol), self.now)
                if df is None or df.empty:
                    continue
                n_alerts = len(self.new_alerts)
                self._scan_new_candles(symbol, df)
                del self.new_alerts[n_alerts:]
                self._update_trades(symbol, df, session_over=True)
            except Exception as e:
                print(f"[research-live] EOD fetch failed for {symbol}: {e!r}")
        for t in self.state["trades"]:
            if t["status"] == "OPEN":
                _close_at_last_seen(t)
            elif t["status"] == "PENDING":
                t["status"] = "NEVER_FILLED"
        exits = _alertable(self.closed_alerts)
        if exits:
            send_text(_format_exits(exits))
        self.closed_alerts = []
        send_eod_report(self.state)
        save_state(self.state)


def _alertable(trades: list) -> list:
    """Trades whose strategy isn't muted (config.LIVE_RESEARCH_SILENT_STRATEGIES).
    Muted strategies are still paper-traded and in the EOD report."""
    silent = set(config.LIVE_RESEARCH_SILENT_STRATEGIES)
    return [t for t in trades if t["strategy"] not in silent]


def _finalize(t: dict, exit_time, exit_price, outcome, candles_held):
    sig = SimpleNamespace(symbol=t["symbol"], strategy=t["strategy"], direction=t["direction"],
                          stop_loss=t["stop_loss"], target=t["target"], regime=t["regime"],
                          trend_strength=t["trend_strength"], score=t["score"])
    rt = build_research_trade(signal=sig, entry_time=pd.Timestamp(t["entry_time"]), entry=t["entry"],
                              exit_time=pd.Timestamp(str(exit_time)), exit_price=float(exit_price),
                              outcome=outcome, candles_held=int(candles_held), apply_costs=True)
    t.update(status="CLOSED", outcome=outcome, exit_time=str(exit_time),
             exit_price=round(float(exit_price), 4), r_multiple=rt.r_multiple,
             pnl_pct=rt.pnl_pct, pnl_currency=rt.pnl_currency, candles_held=int(candles_held),
             research_trade=asdict(rt))


def _close_at_last_seen(t: dict):
    price = t.get("last_price", t.get("entry"))
    when = t.get("last_time", t.get("entry_time"))
    _finalize(t, when, price, "EOD_SQUAREOFF", 0)


# ═══════════════════════════════════════════════════════════════════
# Messages
# ═══════════════════════════════════════════════════════════════════

def _fmt_time(ts: str) -> str:
    try:
        return pd.Timestamp(ts).tz_convert(MARKET_TZ).strftime("%H:%M")
    except Exception:
        return str(ts)[11:16]


def _format_entries(trades: list) -> str:
    lines = [f"📈 PAPER SIGNALS ({len(trades)}) - research strategies, next-candle fill"]
    for t in trades:
        risk = t["signal_price"] - t["stop_loss"] if t["direction"] == "long" else t["stop_loss"] - t["signal_price"]
        risk_pct = risk / t["signal_price"] * 100 if t["signal_price"] else 0
        lines.append(
            f"\n{t['symbol']} {t['direction'].upper()} | {t['strategy']} | candle {_fmt_time(t['signal_time'])} ET\n"
            f"  ~{t['signal_price']}  stop {t['stop_loss']} ({risk_pct:.2f}%)  target {t['target']}\n"
            f"  {t['regime']} | {t['reason'][:160]}")
    return "\n".join(lines)


def _format_exits(trades: list) -> str:
    lines = [f"🔔 PAPER EXITS ({len(trades)})"]
    for t in trades:
        mark = "✅" if t["r_multiple"] > 0 else "❌"
        lines.append(f"{mark} {t['symbol']} {t['strategy']} {t['outcome']} "
                     f"{t['entry']} -> {t['exit_price']}  {t['r_multiple']:+.2f}R "
                     f"({_fmt_time(t['entry_time'])}-{_fmt_time(t['exit_time'])})")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# EOD report (same summary/markdown code as backtest/run_research.py)
# ═══════════════════════════════════════════════════════════════════

def _research_trades(trades: list) -> list:
    out = []
    for t in trades:
        if t.get("status") == "CLOSED" and t.get("research_trade"):
            d = dict(t["research_trade"])
            out.append(ResearchTrade(**d))
    return out


def _closed_rows(state: dict) -> list:
    return [dict(t["research_trade"], date=state["date"], trade_id=t["id"], reason=t.get("reason", ""))
            for t in state["trades"] if t.get("status") == "CLOSED" and t.get("research_trade")]


def _append_all_time(state: dict):
    """Idempotent by trade id - a day can never be counted twice, even if
    its report is regenerated or re-sent."""
    path = _path(ALL_TRADES_CSV)
    done = set()
    if os.path.exists(path):
        done = set(pd.read_csv(path, usecols=["trade_id"])["trade_id"].astype(str))
    rows = [r for r in _closed_rows(state) if r["trade_id"] not in done]
    if not rows:
        return
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


def _load_all_time(exclude_ids=()) -> list:
    path = _path(ALL_TRADES_CSV)
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    df = df[~df["trade_id"].astype(str).isin(set(exclude_ids))]
    fields = ResearchTrade.__dataclass_fields__
    return [ResearchTrade(**{k: r[k] for k in fields}) for r in df.to_dict("records")]


def _risk_sweep(total_r: float, dd_r: float) -> str:
    lines = [f"Risk sweep (equity {config.ACCOUNT_EQUITY:,.0f}):",
             f"{'RISK':>6}{'P&L':>12}{'MAX DD':>12}{'DD%':>7}"]
    for risk in (0.0025, 0.005, 0.0075, 0.01):
        budget = config.ACCOUNT_EQUITY * risk
        lines.append(f"{risk*100:>5.2f}%{total_r*budget:>12,.0f}{dd_r*budget:>12,.0f}"
                     f"{abs(dd_r*budget)/config.ACCOUNT_EQUITY*100:>6.1f}%")
    return "\n".join(lines)


def _summary_text(title: str, summary: dict) -> str:
    from backtest.research_report import _rank
    if summary.get("total_trades", 0) == 0:
        return f"{title}\nNo closed trades."
    o = summary["overall"]
    lines = [title,
             f"Trades {o['trades']} | Win {o['win_rate_pct']}% (W{o['wins']}/L{o['losses']}/EOD{o['scratches']})",
             f"Expectancy {o['expectancy_r']}R | Total {o['total_r']}R | PF {o['profit_factor']} | "
             f"MaxDD {o['max_drawdown_r']}R",
             "",
             f"{'STRATEGY':<22}{'N':>4}{'WIN%':>7}{'EXP_R':>8}{'TOT_R':>8}"]
    for r in _rank(summary):
        lines.append(f"{r['strategy']:<22}{r['trades']:>4}{r['win_rate_pct']:>7}"
                     f"{r['expectancy_r']:>8}{r['total_r']:>8}")
    lines += ["", _risk_sweep(o["total_r"], o["max_drawdown_r"])]
    return "\n".join(lines)


def build_eod_report(state: dict, include_all_time: bool = True):
    """Returns (text, markdown_path)."""
    from backtest.research_report import write_research_report, summarize
    today = _research_trades(state["trades"])
    counts = {}
    for t in state["trades"]:
        counts[t["status"]] = counts.get(t["status"], 0) + 1
    meta = {"from": state["date"], "to": state["date"], "interval": config.CANDLE_INTERVAL_MINUTES,
            "symbols": len({t['symbol'] for t in state['trades']}), "days": 1,
            "costs": f"{config.SLIPPAGE_BPS}+{config.COMMISSION_BPS}bps/side",
            "regime_filter": "ON" if config.REGIME_FILTER_ENABLED else "OFF"}
    out_dir = _path("reports")
    res = write_research_report(today, out_dir, meta)
    text = _summary_text(f"📊 PAPER TRADING EOD REPORT - {state['date']} "
                         f"({', '.join(config.LIVE_RESEARCH_DIRECTIONS)} only)", res["summary"])
    skipped = {k: v for k, v in counts.items() if k != "CLOSED"}
    if skipped:
        text += "\n\nNot traded: " + ", ".join(f"{k} {v}" for k, v in skipped.items())
    if include_all_time:
        prior = _load_all_time(exclude_ids=[t["id"] for t in state["trades"]])
        all_trades = prior + today
        if prior:
            days = len({str(t.entry_time)[:10] for t in all_trades})
            text += "\n\n" + _summary_text(f"📚 ALL PAPER TRADING TO DATE ({days} days)",
                                            summarize(all_trades))
    text += ("\n\nSame fills, exits and costs as the backtest. Small daily samples are noise - "
             "judge strategies on the all-time block.")
    return text, res["report"]


def send_eod_report(state: dict, late: bool = False):
    text, md_path = build_eod_report(state)
    if late:
        text = "(sent late - no run after the close that day)\n" + text
    print(text)
    send_text(text)
    try:
        send_document(md_path, caption=f"Full report {state['date']}")
    except Exception as e:
        print(f"[research-live] report upload failed: {e!r}")
    _append_all_time(state)
    state["report_sent"] = True


def session_over(now: datetime) -> bool:
    close_t = now.replace(hour=config.MARKET_CLOSE_HOUR, minute=config.MARKET_CLOSE_MINUTE,
                          second=0, microsecond=0)
    return now >= close_t


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report", action="store_true", help="Print today's report so far")
    p.add_argument("--send", action="store_true", help="With --report: also send it to Telegram")
    args = p.parse_args()
    state = load_state()
    if args.report:
        if not state.get("date"):
            print("No paper-trading state yet.")
            return
        text, md = build_eod_report(state)
        print(text)
        print(f"\nMarkdown report: {md}")
        if args.send:
            send_text(text)
            send_document(md, caption=f"Report {state['date']} (on demand)")


if __name__ == "__main__":
    main()
