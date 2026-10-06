"""
US SCR (Small-Cap Runners) - live paper trading, one step per workflow run.

    python -m us_scr.live                 # what the US Alert Scan `scr` job runs
    python -m us_scr.live --no-telegram   # console only

Each run (every ~2 min, 07:00-16:10 ET):
  1. DYNAMIC WATCHLIST: Yahoo screener for listed small / micro caps
     (< $2B, >= $1, no OTC) up >= 10% today on volume, + yesterday's movers
     (pre-market carry-over), + open positions. Max 80 names.
  2. VOLUME WATCH: small caps trading >= 5x their 10-day average volume while
     still up < 10% - "suspicious volume before the move" - one alert per
     name per day (watch only, no trade).
  3. 5-minute bars (pre-market included) for the list; every NEW setup on a
     closed bar (us_scr/strategy.py: new high of day on >= 2x volume above
     VWAP, 09:35-15:30 ET) -> paper BUY at the next bar's open, fixed 8%
     stop, trail under bar lows after +1R. At most V_MAX trades a day, one
     position per stock. The ML score is shown and logged, NOT used as a
     filter (it did not beat the plain rule out of sample, 2026-10-06).
  4. Exits are re-simulated each run with the SAME function as research
     (strategy.simulate, live mode) -> Telegram exit alert; day report after
     16:00 ET.
Alerts use their own 🔥 format so they never look like the other strategies.
Paper only - nothing is sent to a broker.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import timedelta

import numpy as np
import pandas as pd

from us_scr import data as D
from us_scr import strategy as S

STATE_DIR = os.environ.get("US_SCR_STATE_DIR", os.path.join(D.ROOT, "live_state", "us_scr"))
MODEL = os.path.join(D.ROOT, "us_scr", "model", "scr_model.joblib")
V_MAX = int(os.environ.get("SCR_MAX_TRADES_PER_DAY", "23"))   # user, 2026-10-06 (was 10; full-day window)
MAX_WATCH = 80
VOL_WATCH_RVOL = 5.0
EQUITY = 100_000.0
RISK_PER_TRADE = 0.005            # 0.5% of paper equity at risk per trade (tight stops -> small size)
START_MIN, END_MIN = 7 * 60, 16 * 60 + 10
HEADER = "🔥🔥 SMALL-CAP RUNNER (US SCR) 🔥🔥"
RISK_LINE = "⚠️ HIGH RISK micro/small cap - no circuit limits, can move ±50% in minutes. Paper only."


def _p(name):
    os.makedirs(STATE_DIR, exist_ok=True)
    return os.path.join(STATE_DIR, name)


def load_state() -> dict:
    try:
        return json.load(open(_p("state.json")))
    except Exception:
        return {}


def save_state(st: dict) -> None:
    tmp = _p("state.json.tmp")
    json.dump(st, open(tmp, "w"), indent=1, default=str)
    os.replace(tmp, _p("state.json"))


def notify(text: str, telegram: bool) -> None:
    print(text, flush=True)
    if telegram:
        from paper_trading.research_live import send_text
        send_text(text)


def _log_trade(row: dict) -> None:
    f = _p("trades.csv")
    new = not os.path.exists(f)
    with open(f, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def _fmt_money(x) -> str:
    return f"${x / 1e6:,.0f}M" if x and x >= 1e6 else (f"${x:,.0f}" if x else "?")


# ─── watchlist ───────────────────────────────────────────────────────────

def build_watch(st: dict) -> pd.DataFrame:
    movers = D.screen_live(min_change=S.WATCH_PCT * 100, min_volume=200_000)
    carry = [s for s in st.get("carry", []) if s not in set(movers.get("symbol", []))]
    held = [p["symbol"] for p in st.get("positions", []) if p["status"] in ("PENDING", "OPEN")]
    names = list(dict.fromkeys(list(movers.get("symbol", [])) + held + carry))[:MAX_WATCH]
    meta = movers.set_index("symbol") if not movers.empty else pd.DataFrame()
    return pd.DataFrame({"symbol": names}).join(meta, on="symbol") if names else pd.DataFrame(columns=["symbol"])


def volume_watch(st: dict, telegram: bool) -> None:
    """Big volume, little move yet (watch only)."""
    try:
        import yfinance as yf
        from yfinance import EquityQuery as EQ
        q = EQ("and", [EQ("eq", ["region", "us"]), EQ("lt", ["intradaymarketcap", D.MAX_MCAP]),
                       EQ("gt", ["dayvolume", 1_000_000]), EQ("gt", ["intradayprice", D.MIN_PRICE])])
        r = yf.screen(q, sortField="dayvolume", sortAsc=False, size=250)
    except Exception as e:
        print(f"[SCR] volume watch screener failed: {e!r}")
        return
    sent = set(st.setdefault("vol_watch_sent", []))
    lines = []
    for x in r.get("quotes", []):
        if x.get("exchange") not in D.LISTED or x.get("quoteType") != "EQUITY" or not x.get("marketCap"):
            continue                                               # no OTC, funds or names without a market cap
        avg = x.get("averageDailyVolume10Day") or 0
        vol, pct = x.get("regularMarketVolume") or 0, x.get("regularMarketChangePercent") or 0
        if avg and vol / avg >= VOL_WATCH_RVOL and -3 <= pct < S.WATCH_PCT * 100 and x["symbol"] not in sent:
            sent.add(x["symbol"])
            lines.append(f"📡 {x['symbol']} {pct:+.1f}% @ ${x.get('regularMarketPrice', 0):.2f} | volume {vol / avg:.1f}x 10-day avg "
                         f"| mcap {_fmt_money(x.get('marketCap'))}")
    st["vol_watch_sent"] = sorted(sent)
    if lines:
        notify("📡📡 SCR VOLUME WATCH - unusual volume, not moved yet (watch only, no trade)\n" + "\n".join(lines[:10]), telegram)


# ─── one run ─────────────────────────────────────────────────────────────

def run(telegram: bool) -> bool:
    import joblib
    now = D.now_et()
    if now.weekday() >= 5:
        print("Weekend - nothing to do.")
        return False
    mins = now.hour * 60 + now.minute
    st = load_state()
    today = str(now.date())
    if st.get("date") != today:
        st = {"date": today, "carry": st.get("tomorrow_carry", []), "positions": [], "last_ts": {}, "taken": 0}
    if mins < START_MIN or st.get("reported"):
        return False
    bundle = joblib.load(MODEL) if os.path.exists(MODEL) else None
    watch = build_watch(st)
    if mins >= 9 * 60 + 35 and mins < 15 * 60 + 30 and not st.get("vol_watch_done_" + str(now.hour)):
        volume_watch(st, telegram)                                  # at most once per hour
        st["vol_watch_done_" + str(now.hour)] = True
    syms = list(watch["symbol"]) if not watch.empty else []
    if not syms:
        save_state(st)
        return True
    bars = {}
    for i in range(0, len(syms), 40):
        for s, x in D._yf(syms[i:i + 40], period="1d", interval="5m", prepost=True).items():
            x.index = pd.DatetimeIndex(x.index).tz_convert(D.ET)
            x = x[x.index < pd.Timestamp(now).floor("5min")]         # closed bars only
            if len(x):
                bars[s] = x
                D.store(s, x)
    daily = _daily_ctx(syms, st)
    changed = False
    for sym in syms:
        x = bars.get(sym)
        c = daily.get(sym)
        if x is None or c is None or not c.get("prev_close"):
            continue
        f = S.bar_features(x, c["prev_close"], c.get("avg_vol20") or np.nan)
        last = st["last_ts"].get(sym)
        new = [ts for ts in S.setups(x, f) if last is None or str(ts) > last]
        st["last_ts"][sym] = str(x.index[-1])
        for ts in new:
            if st["taken"] >= V_MAX:
                break
            if any(p["symbol"] == sym and p["status"] in ("PENDING", "OPEN") for p in st["positions"]):
                continue
            if sum(p["symbol"] == sym for p in st["positions"]) >= S.MAX_PER_SYMBOL_DAY:
                continue
            if ts < x.index[-1] - pd.Timedelta(minutes=10):
                continue                                           # stale backlog setup - don't trade it late
            # ML score is shown and logged but NOT used as a filter: it did not
            # beat the plain rule out of sample (2026-10-06, 40 sessions of data).
            score = _score(bundle, f.loc[ts], c, x.loc[ts, "close"], ts)
            st["taken"] += 1
            changed = True
            row, meta = f.loc[ts], (watch.set_index("symbol").loc[sym] if sym in set(watch["symbol"]) else {})
            stop_hint = float(x.loc[ts, "close"]) * (1 - S.STOP_PCT)
            p = {"symbol": sym, "status": "PENDING", "signal_ts": str(ts), "score": round(float(score), 3),
                 "pct": round(float(row["pct"]) * 100, 1), "rvol": round(float(row["rvol"]), 1) if row["rvol"] == row["rvol"] else None,
                 "mcap": meta.get("mcap") if hasattr(meta, "get") else None}
            st["positions"].append(p)
            notify(f"{HEADER}\n🚀 BUY {sym} ~${float(x.loc[ts, 'close']):.2f} (next 5-min bar open)\n"
                   f"⚡ {p['pct']:+.1f}% today | volume {p['rvol']}x avg | new high of day on {float(row['vol_ratio']):.1f}x bar volume | "
                   f"mcap {_fmt_money(p['mcap'])}\n"
                   f"🛑 stop ~${stop_hint:.2f} ({(stop_hint / float(x.loc[ts, 'close']) - 1) * 100:+.1f}%) | trail under bar lows after +1R | "
                   f"max {S.MAX_HOLD_MIN} min, flat 15:55 ET\n🧠 ML score {score:+.2f} (experimental, not a filter) | trade {st['taken']}/{V_MAX} today\n"
                   f"🧪 UNPROVEN strategy - backtest ~0% before costs; forward paper test\n{RISK_LINE}",
                   telegram)
    changed |= manage(st, bars, telegram)
    if mins >= 16 * 60 and not st.get("reported"):
        day_report(st, telegram)
        movers = list(watch["symbol"]) if not watch.empty else []
        st["tomorrow_carry"] = movers[:40]
        st["reported"] = True
        changed = True
    save_state(st)
    return True


def _daily_ctx(syms, st) -> dict:
    """prev_close / avg_vol20 / prev_ret / ret_5d / max_ret_20d per symbol, fetched once a day."""
    cache = st.setdefault("daily", {})
    need = [s for s in syms if s not in cache]
    if need:
        d = D.daily_bars(need, period="3mo")
        mcap = D.universe().set_index("symbol")["mcap"].to_dict()
        if not d.empty:
            today = pd.Timestamp(st["date"])
            for s, g in d.groupby("symbol"):
                g = g[g["date"] < today].sort_values("date")
                if len(g) < 7:
                    continue
                cl, vol = g["close"].values, g["volume"].values
                rets = cl[1:] / cl[:-1] - 1
                cache[s] = {"prev_close": float(cl[-1]), "avg_vol20": float(np.mean(vol[-20:])),
                            "prev_ret": float(cl[-1] / cl[-2] - 1), "ret_5d": float(cl[-1] / cl[-6] - 1),
                            "max_ret_20d": float(np.max(rets[-20:])), "mcap": mcap.get(s)}
    return cache


def _score(bundle, r, c, close, ts) -> float:
    if bundle is None:
        return 0.0
    import math
    feats = {"pct": r["pct"], "log_rvol": math.log1p(max(r["rvol"], 0)) if r["rvol"] == r["rvol"] else np.nan,
             "log_dollar": math.log10(max(r["cum_dollar"], 1)), "vol_ratio": r["vol_ratio"], "dist_vwap": r["dist_vwap"],
             "minute": r["minute"], "n_hod": r["n_hod"], "bar_range": r["bar_range"], "close_loc": r["close_loc"],
             "ret_3": r["ret_3"], "reg_open_gap": r["reg_open_gap"], "stop_pct": r["stop_pct"],
             "prev_ret": c.get("prev_ret"), "ret_5d": c.get("ret_5d"), "max_ret_20d": c.get("max_ret_20d"),
             "log_mcap": math.log10(c["mcap"]) if c.get("mcap") else np.nan,
             "log_price": math.log10(max(float(close), 0.01)), "pre_session": int(ts.time() < S.dtime(9, 30))}
    X = pd.DataFrame([feats])[bundle["features"]].astype(float)
    return float(bundle["model"].predict(X)[0])


def manage(st, bars, telegram) -> bool:
    """Fill pending entries, re-simulate open positions with the research exit rules."""
    changed = False
    for p in st["positions"]:
        if p["status"] not in ("PENDING", "OPEN"):
            continue
        x = bars.get(p["symbol"])
        if x is None:
            continue
        sig = pd.Timestamp(p["signal_ts"])
        if sig not in x.index:
            continue
        tr = S.simulate(p["symbol"], x, sig, final=False)
        if tr is None:
            continue                                               # next bar not closed yet
        if p["status"] == "PENDING":
            stop0 = tr.entry * (1 - S.STOP_PCT)                          # same rule as simulate()
            p.update(status="OPEN", entry=round(tr.entry, 4), entry_ts=str(tr.entry_ts), stop_initial=round(stop0, 4))
            p["shares"] = int(EQUITY * RISK_PER_TRADE / max(tr.entry - p["stop_initial"], 0.01))
            changed = True
        if tr.outcome == "OPEN":
            p["stop_now"] = round(tr.stop0, 4)
            continue
        p.update(status="CLOSED", exit=round(tr.exit, 4), exit_ts=str(tr.exit_ts), outcome=tr.outcome,
                 R=round(tr.R, 2), ret_pct=round(tr.ret_pct, 2), pnl=round(p.get("shares", 0) * (tr.exit - tr.entry), 2))
        _log_trade({"date": st["date"], **{k: p.get(k) for k in ("symbol", "signal_ts", "entry_ts", "entry", "stop_initial",
                                                                  "exit_ts", "exit", "outcome", "R", "ret_pct", "pnl", "score",
                                                                  "pct", "rvol", "shares")}})
        icon = "✅" if tr.R > 0 else "❌"
        mins = int((tr.exit_ts - tr.entry_ts).total_seconds() // 60)
        notify(f"🔥 SCR EXIT {icon} {p['symbol']} ${tr.entry:.2f} -> ${tr.exit:.2f} ({tr.ret_pct:+.1f}%, {tr.R:+.2f}R) "
               f"{tr.outcome} after {mins} min | paper P&L ${p['pnl']:+,.0f}", telegram)
        changed = True
    return changed


def day_report(st, telegram) -> None:
    ps = [p for p in st["positions"] if p["status"] == "CLOSED"]
    lines = [f"🔥📊 US SCR SMALL-CAP RUNNERS - PAPER RESULT {st['date']}"]
    if not ps:
        lines.append("No runner trade today (no setup scored above the ML bar).")
    for p in ps:
        lines.append(f"{'✅' if p['R'] > 0 else '❌'} {p['symbol']}: {p['entry']:.2f} -> {p['exit']:.2f} ({p['ret_pct']:+.1f}%, "
                     f"{p['R']:+.2f}R, {p['outcome']}) ${p['pnl']:+,.0f}")
    if ps:
        lines.append(f"Day: {sum(p['R'] for p in ps):+.2f}R | ${sum(p['pnl'] for p in ps):+,.0f} on ${EQUITY:,.0f} paper "
                     f"({RISK_PER_TRADE * 100:.1f}% risk/trade)")
    f = _p("trades.csv")
    if os.path.exists(f):
        t = pd.read_csv(f)
        lines.append(f"ALL-TIME ({t['date'].nunique()} days, {len(t)} trades): {t['R'].sum():+.2f}R | win {(t['R'] > 0).mean() * 100:.0f}% | "
                     f"${t['pnl'].sum():+,.0f}")
    lines.append(RISK_LINE)
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
