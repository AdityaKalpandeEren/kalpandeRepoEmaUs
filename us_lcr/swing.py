"""
US LCR SWING - a second paper book on the SAME LCR entries, carried overnight.

Research (us_lcr/research_swing.py, 343 trades, Aug-Oct 2026): with the
intraday trailing stop, holding overnight loses; with a fixed 8% stop,
selling at the close of the 3rd session after the entry day was the best
max hold (+0.52%/trade, both halves positive: +0.45% / +0.62%; 4-5 days turn
negative). Weak evidence (42 sessions, day-t ~1.0, 16 variants) - UNPROVEN.

Rules
  entry   copies every intraday LCR fill (same price, same time)
  size    fixed $10,000 per trade from its own $100,000 book; a signal is
          skipped when 10 positions are already open
  stop    8% under the entry; checked on every 5-min bar (a bar opening below
          it fills at that open)
  exit    otherwise at the 15:55 ET bar close of the 3rd session after the
          entry day
State: live_state/us_lcr/swing/ (state.json + trades.csv, LCR's columns), so the
existing `lcr` job cache and the app pick it up with no workflow change.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import time as dtime

import pandas as pd

from us_lcr import strategy as S
from us_scr import data as SD

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.environ.get("US_LCR_SWING_STATE_DIR", os.path.join(ROOT, "live_state", "us_lcr", "swing"))
EQUITY, NOTIONAL, MAX_OPEN = 100_000.0, 10_000.0, 10
STOP_PCT, HOLD_SESSIONS = 0.08, 3
EXIT_BAR = dtime(15, 55)
HEADER = "🏛️🌙 LCR SWING (max 3 days, 8% stop)"
NOTE = "Paper only - UNPROVEN"


def _p(name):
    os.makedirs(STATE_DIR, exist_ok=True)
    return os.path.join(STATE_DIR, name)


def load_state():
    try:
        return json.load(open(_p("state.json")))
    except Exception:
        return {"positions": [], "copied": [], "cash": EQUITY}


def save_state(st):
    tmp = _p("state.json.tmp")
    json.dump(st, open(tmp, "w"), indent=1, default=str)
    os.replace(tmp, _p("state.json"))


def open_from_intraday(sw, intraday_positions, today, notify):
    """Copy every new intraday LCR fill into the swing book."""
    for p in intraday_positions:
        if p.get("status") not in ("OPEN", "CLOSED") or not p.get("entry"):
            continue
        key = f"{p['symbol']}|{p['signal_ts']}"
        if key in sw["copied"]:
            continue
        sw["copied"].append(key)
        sw["taken"] = sw.get("taken", 0) + 1 if sw.get("date") == today else 1
        sw["date"] = today
        held = [q for q in sw["positions"] if q["status"] == "OPEN"]
        if len(held) >= MAX_OPEN or sw["cash"] < NOTIONAL:
            notify(f"{HEADER}\n⏭️ skipped {p['symbol']}: book full ({len(held)}/{MAX_OPEN} open) | {NOTE}")
            continue
        entry = float(p["entry"])
        shares = int(NOTIONAL // entry)
        if shares <= 0:
            continue
        stop = round(entry * (1 - STOP_PCT), 4)
        sw["cash"] -= shares * entry
        sw["positions"].append({"symbol": p["symbol"], "status": "OPEN", "signal_ts": p["signal_ts"],
                                "entry_ts": p["entry_ts"], "entry": entry, "stop_initial": stop, "shares": shares,
                                "entry_date": today, "bucket": p.get("bucket"), "pct": p.get("pct"),
                                "rvol": p.get("rvol"), "perfect": p.get("perfect"), "checked_ts": p["signal_ts"]})
        notify(f"{HEADER}\n🌙 BUY {p['symbol']} {shares} sh @ ${entry:.2f} (same fill as intraday LCR)\n"
               f"🛑 stop ${stop:.2f} (-8%) | exit at the 15:55 ET close of the 3rd session after today | {NOTE}")


def check(p, x):
    """Walk 5-min bars after the last checked one. Returns (exit_price, exit_ts, outcome) or None."""
    x = x[(x.index > pd.Timestamp(p["checked_ts"])) & (x.index.time >= dtime(9, 30)) & (x.index.time <= EXIT_BAR)]
    entry_day = pd.Timestamp(p["entry_date"]).date()
    entry_ts = pd.Timestamp(p["entry_ts"])
    for ts, b in x.iterrows():
        if ts > entry_ts and float(b["open"]) <= p["stop_initial"]:     # opened below the stop (e.g. overnight gap)
            return float(b["open"]), ts, "STOP_GAP"
        if float(b["low"]) <= p["stop_initial"]:
            return p["stop_initial"], ts, "STOP"
        sessions = len({d for d in x.index[x.index <= ts].date if d > entry_day} | set(p.get("sessions", [])))
        if sessions >= HOLD_SESSIONS and ts.time() >= EXIT_BAR:
            return float(b["close"]), ts, f"DAY{HOLD_SESSIONS}_CLOSE"
    if len(x):
        p["checked_ts"] = str(x.index[-1])
        p["sessions"] = sorted({str(d) for d in x.index.date if d > entry_day} | set(p.get("sessions", [])))
    return None


def manage(sw, now, notify):
    held = [p for p in sw["positions"] if p["status"] == "OPEN"]
    if not held:
        return
    bars = {}
    for s, x in SD._yf(sorted({p["symbol"] for p in held}), period="5d", interval="5m", prepost=False).items():
        x.index = pd.DatetimeIndex(x.index).tz_convert(SD.ET)
        bars[s] = x[x.index < pd.Timestamp(now).floor("5min")]
    for p in held:
        x = bars.get(p["symbol"])
        if x is None or x.empty:
            continue
        hit = check(p, x)
        if hit is None:
            continue
        px, ts, why = hit
        exit_px = px * (1 - S.SLIP)
        pnl = p["shares"] * (exit_px - p["entry"])
        risk = p["entry"] - p["stop_initial"]
        p.update(status="CLOSED", exit=round(exit_px, 4), exit_ts=str(ts), outcome=why,
                 R=round((exit_px - p["entry"]) / risk, 2), ret_pct=round((exit_px / p["entry"] - 1) * 100, 2),
                 pnl=round(pnl, 2))
        sw["cash"] += p["shares"] * exit_px
        f = _p("trades.csv")
        new = not os.path.exists(f)
        with open(f, "a", newline="") as fh:
            row = {"date": p["entry_date"], **{k: p.get(k) for k in ("symbol", "bucket", "signal_ts", "entry_ts", "entry",
                                                                       "stop_initial", "exit_ts", "exit", "outcome", "R",
                                                                       "ret_pct", "pnl", "pct", "rvol", "shares", "perfect")}}
            wr = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                wr.writeheader()
            wr.writerow(row)
        notify(f"{HEADER}\n{'✅' if pnl > 0 else '❌'} EXIT {p['symbol']} ${p['entry']:.2f} -> ${exit_px:.2f} "
               f"({p['ret_pct']:+.2f}%) {why} | paper P&L ${pnl:+,.0f} | {NOTE}")
    sw["positions"] = [p for p in sw["positions"] if p["status"] == "OPEN"]
    sw["copied"] = sw["copied"][-500:]


def day_report(sw, today, notify):
    if sw.get("reported") == today:
        return
    sw["reported"] = today
    lines = [f"🏛️🌙📊 LCR SWING - PAPER {today}"]
    open_ = [p for p in sw["positions"] if p["status"] == "OPEN"]
    lines.append(f"Open {len(open_)}/{MAX_OPEN}: " + (", ".join(f"{p['symbol']} (since {p['entry_date']})" for p in open_) or "none"))
    f = _p("trades.csv")
    if os.path.exists(f):
        t = pd.read_csv(f)
        lines.append(f"ALL-TIME closed {len(t)}: ${t['pnl'].sum():+,.0f} | avg {t['ret_pct'].mean():+.2f}%/trade | "
                     f"win {(t['pnl'] > 0).mean() * 100:.0f}%")
    lines.append(f"Cash ${sw['cash']:,.0f} of ${EQUITY:,.0f} | {NOTE}")
    notify("\n".join(lines))


def step(intraday_st, now, telegram):
    """Called from us_lcr.live.run once per run."""
    from us_lcr.live import notify as _n
    note = lambda t: _n(t, telegram)
    sw = load_state()
    today = str(now.date())
    open_from_intraday(sw, intraday_st.get("positions", []), today, note)
    manage(sw, now, note)
    if now.hour * 60 + now.minute >= 16 * 60:
        day_report(sw, today, note)
    save_state(sw)
