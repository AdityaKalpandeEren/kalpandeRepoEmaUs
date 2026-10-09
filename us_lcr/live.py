"""
US LCR (Large-Cap Runners) - mid / large / mega caps with unusual volume,
live paper trading, one step per workflow run (09:30-16:10 ET).

    python -m us_lcr.live                 # what the US Alert Scan `lcr` job runs
    python -m us_lcr.live --no-telegram

Each run:
  1. DYNAMIC WATCHLIST: free Yahoo screener for listed stocks with market
     cap >= $2B, up >= 2% today on volume, + open positions; tagged MID
     ($2-10B) / LARGE ($10-200B) / MEGA (>= $200B).
  2. 🏛️📡 VOLUME WATCH: >= 3x 10-day volume while up < 2% (watch only).
  3. 5-min bars for the list; every NEW setup on a closed bar
     (us_lcr/strategy.py, time-adjusted volume shocker + daily EMA filter)
     -> paper BUY at the next bar's open; exits re-simulated each run with
     the SAME function as research; day report after 16:00 ET.
Paper only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import pandas as pd

from us_lcr import strategy as S
from us_scr import data as SD

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.environ.get("US_LCR_STATE_DIR", os.path.join(ROOT, "live_state", "us_lcr"))
V_MAX = int(os.environ.get("LCR_MAX_TRADES_PER_DAY", "13"))   # user, 2026-10-09 (was 10)
EQUITY, RISK_PER_TRADE = 100_000.0, 0.005
MAX_WATCH = 80
VOL_WATCH_X = 3.0
HEADER = "🏛️🏛️ LARGE-CAP RUNNER (US LCR) 🏛️🏛️"
NOTE = "Paper only."


def _p(name):
    os.makedirs(STATE_DIR, exist_ok=True)
    return os.path.join(STATE_DIR, name)


def load_state():
    try:
        return json.load(open(_p("state.json")))
    except Exception:
        return {}


def save_state(st):
    tmp = _p("state.json.tmp")
    json.dump(st, open(tmp, "w"), indent=1, default=str)
    os.replace(tmp, _p("state.json"))


def notify(text, telegram):
    print(text, flush=True)
    if telegram:
        from paper_trading.research_live import send_text
        send_text(text)


def bucket(mcap) -> str:
    if not mcap or mcap != mcap:
        return "?"
    return "MEGA" if mcap >= 200e9 else ("LARGE" if mcap >= 10e9 else "MID")


def _money(x):
    return f"${x / 1e9:,.1f}B" if x and x >= 1e9 else "?"


def screen(min_change=2.0, min_volume=500_000, size=250) -> pd.DataFrame:
    import yfinance as yf
    from yfinance import EquityQuery as EQ
    q = EQ("and", [EQ("eq", ["region", "us"]), EQ("gte", ["intradaymarketcap", 2_000_000_000]),
                   EQ("gt", ["percentchange", min_change]), EQ("gt", ["dayvolume", min_volume])])
    try:
        r = yf.screen(q, sortField="percentchange", sortAsc=False, size=size)
    except Exception as e:
        print(f"[LCR] screener failed: {e!r}")
        return pd.DataFrame()
    rows = []
    for x in r.get("quotes", []):
        if x.get("exchange") not in SD.LISTED or x.get("quoteType") != "EQUITY":
            continue
        avg = x.get("averageDailyVolume10Day") or 0
        rows.append({"symbol": x["symbol"], "pct": x.get("regularMarketChangePercent"), "price": x.get("regularMarketPrice"),
                     "volume": x.get("regularMarketVolume"), "vol_x": (x.get("regularMarketVolume") or 0) / avg if avg else np.nan,
                     "mcap": x.get("marketCap")})
    return pd.DataFrame(rows)


def volume_watch(st, telegram):
    import yfinance as yf
    from yfinance import EquityQuery as EQ
    try:
        q = EQ("and", [EQ("eq", ["region", "us"]), EQ("gte", ["intradaymarketcap", 2_000_000_000]),
                       EQ("gt", ["dayvolume", 1_000_000])])
        r = yf.screen(q, sortField="dayvolume", sortAsc=False, size=250)
    except Exception as e:
        print(f"[LCR] volume watch failed: {e!r}")
        return
    sent = set(st.setdefault("vol_watch_sent", []))
    lines = []
    for x in r.get("quotes", []):
        if x.get("exchange") not in SD.LISTED or x.get("quoteType") != "EQUITY" or not x.get("marketCap"):
            continue
        avg, vol, pct = x.get("averageDailyVolume10Day") or 0, x.get("regularMarketVolume") or 0, x.get("regularMarketChangePercent") or 0
        if avg and vol / avg >= VOL_WATCH_X and -2 <= pct < S.WATCH_PCT * 100 and x["symbol"] not in sent:
            sent.add(x["symbol"])
            lines.append(f"📡 {x['symbol']} [{bucket(x['marketCap'])}] {pct:+.1f}% @ ${x.get('regularMarketPrice', 0):.2f} | "
                         f"volume {vol / avg:.1f}x 10-day avg | mcap {_money(x['marketCap'])}")
    st["vol_watch_sent"] = sorted(sent)
    if lines:
        notify("🏛️📡 LCR VOLUME WATCH - large caps with unusual volume, not moved yet (watch only)\n" + "\n".join(lines[:10]), telegram)


def daily_ctx(syms, st) -> dict:
    cache = st.setdefault("daily", {})
    need = [s for s in syms if s not in cache]
    if need:
        d = SD.daily_bars(need, period="1y")
        today = pd.Timestamp(st["date"])
        for s, g in (d.groupby("symbol") if not d.empty else []):
            g = g[g["date"] < today].sort_values("date")
            if len(g) < 30:
                cache[s] = None
                continue
            cache[s] = {"prev_close": float(g["close"].iloc[-1]), "avg_vol20": float(g["volume"].tail(20).mean()),
                        "emas": S.daily_emas(g["close"])}
    return cache


def run(telegram: bool) -> bool:
    now = SD.now_et()
    if now.weekday() >= 5:
        return False
    mins = now.hour * 60 + now.minute
    st = load_state()
    today = str(now.date())
    if st.get("date") != today:
        st = {"date": today, "positions": [], "last_ts": {}, "taken": 0}
    if mins < 9 * 60 + 30 or st.get("reported"):
        return False
    w = screen()
    if mins >= 9 * 60 + 35 and mins < 15 * 60 + 30 and not st.get(f"vw_{now.hour}"):
        volume_watch(st, telegram)
        st[f"vw_{now.hour}"] = True
    held = [p["symbol"] for p in st["positions"] if p["status"] in ("PENDING", "OPEN")]
    syms = list(dict.fromkeys(list(w.get("symbol", [])) + held))[:MAX_WATCH]
    ctx = daily_ctx(syms, st)
    meta = w.set_index("symbol") if not w.empty else pd.DataFrame()
    bars = {}
    for i in range(0, len(syms), 40):
        for s, x in SD._yf(syms[i:i + 40], period="1d", interval="5m", prepost=True).items():
            x.index = pd.DatetimeIndex(x.index).tz_convert(SD.ET)
            x = x[x.index < pd.Timestamp(now).floor("5min")]
            if len(x):
                bars[s] = x
    for s in syms:
        x, c = bars.get(s), ctx.get(s)
        if x is None or not c:
            continue
        f = S.bar_features(x, c["prev_close"], c["avg_vol20"])
        last = st["last_ts"].get(s)
        new = [ts for ts in S.setups(x, f, c["emas"]) if last is None or str(ts) > last]
        st["last_ts"][s] = str(x.index[-1])
        for ts in new:
            if st["taken"] >= V_MAX or ts < x.index[-1] - pd.Timedelta(minutes=10):
                continue
            if any(p["symbol"] == s and p["status"] in ("PENDING", "OPEN") for p in st["positions"]):
                continue
            if sum(p["symbol"] == s for p in st["positions"]) >= S.MAX_PER_SYMBOL_DAY:
                continue
            r, close = f.loc[ts], float(x.loc[ts, "close"])
            stop = S.initial_stop(x, x.index.get_loc(ts), close)
            mc = meta.loc[s, "mcap"] if s in meta.index else None
            st["taken"] += 1
            st["positions"].append({"symbol": s, "status": "PENDING", "signal_ts": str(ts), "bucket": bucket(mc),
                                    "pct": round(float(r["pct"]) * 100, 2), "rvol": round(float(r["rvol"]), 1)})
            notify(f"{HEADER}\n🚀 BUY {s} [{bucket(mc)} {_money(mc)}] ~${close:.2f} (next 5-min bar open)\n"
                   f"⚡ {float(r['pct']) * 100:+.1f}% today | volume {float(r['rvol']):.1f}x normal for this time of day\n"
                   f"📈 daily EMAs: {S.ema_tags(close, c['emas'])}\n"
                   f"🛑 stop ~${stop:.2f} ({(stop / close - 1) * 100:+.1f}%) | trail after +1R | max {S.MAX_HOLD_MIN} min, flat 15:55 ET\n"
                   f"trade {st['taken']}/{V_MAX} today | {NOTE}", telegram)
    manage(st, bars, telegram)
    if mins >= 16 * 60 and not st.get("reported"):
        day_report(st, telegram)
        st["reported"] = True
    save_state(st)
    return True


def manage(st, bars, telegram):
    for p in st["positions"]:
        if p["status"] not in ("PENDING", "OPEN"):
            continue
        x = bars.get(p["symbol"])
        sig = pd.Timestamp(p["signal_ts"])
        if x is None or sig not in x.index:
            continue
        tr = S.simulate(p["symbol"], x, sig, final=False)
        if tr is None:
            continue
        if p["status"] == "PENDING":
            stop0 = S.initial_stop(x, x.index.get_loc(sig), tr.entry)
            p.update(status="OPEN", entry=round(tr.entry, 4), entry_ts=str(tr.entry_ts), stop_initial=round(stop0, 4),
                     shares=int(EQUITY * RISK_PER_TRADE / max(tr.entry - stop0, 0.01)))
        if tr.outcome == "OPEN":
            p["stop_now"] = round(tr.stop0, 4)
            continue
        pnl = p["shares"] * (tr.exit - tr.entry)
        p.update(status="CLOSED", exit=round(tr.exit, 4), exit_ts=str(tr.exit_ts), outcome=tr.outcome,
                 R=round(tr.R, 2), ret_pct=round(tr.ret_pct, 2), pnl=round(pnl, 2))
        f = _p("trades.csv")
        new = not os.path.exists(f)
        with open(f, "a", newline="") as fh:
            row = {"date": st["date"], **{k: p.get(k) for k in ("symbol", "bucket", "signal_ts", "entry_ts", "entry",
                                                                  "stop_initial", "exit_ts", "exit", "outcome", "R",
                                                                  "ret_pct", "pnl", "pct", "rvol", "shares")}}
            wr = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                wr.writeheader()
            wr.writerow(row)
        mins = int((tr.exit_ts - tr.entry_ts).total_seconds() // 60)
        notify(f"🏛️ LCR EXIT {'✅' if tr.R > 0 else '❌'} {p['symbol']} ${tr.entry:.2f} -> ${tr.exit:.2f} "
               f"({tr.ret_pct:+.2f}%, {tr.R:+.2f}R) {tr.outcome} after {mins} min | paper P&L ${pnl:+,.0f}", telegram)


def day_report(st, telegram):
    ps = [p for p in st["positions"] if p["status"] == "CLOSED"]
    lines = [f"🏛️📊 US LCR LARGE-CAP RUNNERS - PAPER RESULT {st['date']}"]
    if not ps:
        lines.append("No large-cap runner trade today.")
    for p in ps:
        lines.append(f"{'✅' if p['R'] > 0 else '❌'} {p['symbol']} [{p['bucket']}]: {p['entry']:.2f} -> {p['exit']:.2f} "
                     f"({p['ret_pct']:+.2f}%, {p['R']:+.2f}R, {p['outcome']}) ${p['pnl']:+,.0f}")
    if ps:
        lines.append(f"Day: {sum(p['R'] for p in ps):+.2f}R | ${sum(p['pnl'] for p in ps):+,.0f} ({RISK_PER_TRADE * 100:.1f}% risk/trade)")
    f = _p("trades.csv")
    if os.path.exists(f):
        t = pd.read_csv(f)
        lines.append(f"ALL-TIME ({t['date'].nunique()} days, {len(t)} trades): {t['R'].sum():+.2f}R | "
                     f"win {(t['R'] > 0).mean() * 100:.0f}% | ${t['pnl'].sum():+,.0f}")
    notify("\n".join(lines), telegram)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-telegram", action="store_true")
    a = ap.parse_args()
    changed = False
    try:
        changed = run(not a.no_telegram)
    finally:
        out = os.environ.get("GITHUB_OUTPUT")
        if out:
            with open(out, "a") as fh:
                fh.write(f"changed={'true' if changed else 'false'}\n")


if __name__ == "__main__":
    main()
