#!/usr/bin/env python3
"""
App data export (schema 1) - one compact JSON with every paper-trading
system of this repo, read by the "Nayan Kalpande Appreciation Society"
Android app (code name AlgoPulse).

The separate `app-data.yml` workflow runs this twice per run, after each
production run. It only READS production's state caches and run history: it
never saves a cache, never trades, never alerts, so it cannot affect the bots.

  plan   - which state caches changed since the last publish (only those are
           restored - NSE V5's cache alone is ~180 MB), plus a small health
           snapshot of the production workflow's recent runs (GitHub API, GH_TOKEN)
  build  - read the restored state files, normalise every system to one
           schema, carry over the systems whose cache did not change, write
           dashboard.json.gz (published as an asset of the `dashboard-data`
           release: public URL, no token needed in the app)

    python3 app_data/export.py plan  --market US --prev prev.json.gz
    python3 app_data/export.py build --market US --root . --prev prev.json.gz --out dashboard.json.gz

The master copy lives in the app repo (pipeline/export.py); the US and NSE bot
repos carry identical copies in app_data/ - edit the master, copy to both.
Standard library only: runs on the runner's system python, no pip install.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

SCHEMA = 1
RESEARCH_TRADE_SESSIONS = 10      # research strategies trade hundreds a day: keep each trade for the last N sessions
                                  # (daily totals always cover the full history)

# Source groups: cache-key prefix -> what it holds. The restore steps in the
# workflow use exactly these prefixes (and the jobs' own `path:` lists).
MARKETS = {
    "US": {
        "ccy": "USD", "tz": "America/New_York", "open": "09:30", "close": "16:00", "bench": "SPY",
        "workflow": "us-alert-scan.yml",     # the production workflow whose runs the health section reports
        "groups": {"research": "live-state-", "us_v3": "us-v3-", "us_v5": "us-v5-",
                   "us_scr": "us-scr-", "us_lcr": "us-lcr-"},
    },
    "NSE": {
        "ccy": "INR", "tz": "Asia/Kolkata", "open": "09:15", "close": "15:30", "bench": "NIFTY 50",
        "workflow": "nse-alert-scan.yml",
        "groups": {"research": "nse-live-state-", "nse_v33": "nse-v33-", "nse_v5": "nse-v5-",
                   "nse_scr": "nse-scr-", "nse_lcr": "nse-lcr-"},
    },
}

RESEARCH_NAMES = {   # code name -> (display name, group, one-line description)
    "SCORE_ENGINE": ("Score Engine", "Research", "Multi-factor score on 5-min candles, long only, research paper"),
    "J_VWAP_BAND_REVERSION": ("VWAP Band Reversion", "Research", "Mean reversion from the lower VWAP band"),
    "K_RSI2_REVERSION": ("RSI-2 Reversion", "Research", "Short-term RSI(2) oversold bounce"),
    "L_ML_META": ("ML Meta V1", "ML filters", "ML meta-filter on top of the rule setups (V1)"),
    "L_ML_META_V1_2": ("ML Meta V1.2", "ML filters", "V1 meta-filter, no-news variant"),
    "L_ML_META_V2": ("ML Meta V2", "ML filters", "V2 meta-filter with live news overlay"),
    "L_ML_META_V2_2": ("ML Meta V2.2", "ML filters", "V2 meta-filter, keyword-news variant"),
    "L_ML_META_V3": ("ML Meta V3", "ML filters", "V3 meta-filter (trained threshold)"),
}

# US V3 sizes 1R as DAILY_RISK / V3_MAX of a $100k paper account. V3_MAX was 12
# on 2026-10-01/02 (5% / 12 = 0.42%) and 5 from 2026-10-05 (PR #11: 1%/trade).
US_V3_EQUITY = 100_000.0
US_V3_RISK = [("2026-10-05", 0.01), ("0000-00-00", 0.05 / 12)]


# ─── small helpers ────────────────────────────────────────────────────────

def fnum(x, nd=None):
    """float or None; tolerant of '', 'nan', None."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return round(v, nd) if nd is not None else v


def ts(x):
    """'2026-10-09 09:35:00-04:00' -> '2026-10-09T09:35:00-04:00' (None-safe)."""
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in ("nan", "none", "nat"):
        return None
    return s.replace(" ", "T", 1) if len(s) > 10 and s[10] == " " else s


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def mtime_iso(*paths):
    ms = [os.path.getmtime(p) for p in paths if p and os.path.exists(p)]
    return datetime.fromtimestamp(max(ms), timezone.utc).isoformat(timespec="seconds") if ms else None


def now_utc():
    return datetime.now(timezone.utc).replace(microsecond=0)


def truthy(x):
    return str(x).strip().lower() in ("true", "1", "yes")


def clean(d):
    """Drop None values (smaller JSON; the app treats missing as unknown)."""
    return {k: v for k, v in d.items() if v is not None and v != "" and v != []}


# ─── aggregation ──────────────────────────────────────────────────────────

def days_from_trades(trades):
    """Daily totals per trade date: the app rolls these up to week / month /
    year / all-time without needing every trade."""
    acc = defaultdict(lambda: {"n": 0, "w": 0, "l": 0, "pnl": 0.0, "gw": 0.0, "gl": 0.0, "r": 0.0, "rw": 0.0,
                               "rl": 0.0, "r2": 0.0, "rn": 0, "best": None, "worst": None, "fees": 0.0})
    for t in trades:
        a = acc[t["d"]]
        p = t.get("pnl") or 0.0
        a["n"] += 1
        a["pnl"] += p
        if p > 0:
            a["w"] += 1
            a["gw"] += p
        elif p < 0:
            a["l"] += 1
            a["gl"] += p
        r = t.get("r")
        if r is not None:
            a["rn"] += 1
            a["r"] += r
            a["r2"] += r * r
            if r > 0:
                a["rw"] += r
            elif r < 0:
                a["rl"] += r
        a["best"] = p if a["best"] is None else max(a["best"], p)
        a["worst"] = p if a["worst"] is None else min(a["worst"], p)
        a["fees"] += t.get("fees") or 0.0
    out = []
    for d in sorted(acc):
        a = acc[d]
        out.append(clean({"d": d, "n": a["n"], "w": a["w"], "l": a["l"], "pnl": round(a["pnl"], 2),
                          "gw": round(a["gw"], 2), "gl": round(a["gl"], 2),
                          "r": round(a["r"], 3) if a["rn"] else None, "rw": round(a["rw"], 3) if a["rn"] else None,
                          "rl": round(a["rl"], 3) if a["rn"] else None, "r2": round(a["r2"], 3) if a["rn"] else None,
                          "rn": a["rn"] or None, "best": round(a["best"], 2), "worst": round(a["worst"], 2),
                          "fees": round(a["fees"], 2) or None}))
    return out


def system(sid, name, code, group, style, desc, trades=None, opens=None, status=None, risk_unit=None,
           keep_sessions=None, src=None, **extra):
    trades = sorted(trades or [], key=lambda t: (t["d"], t.get("xt") or t.get("et") or ""))
    days = days_from_trades(trades)
    shown, trades_from = trades, None
    if keep_sessions and days and len(days) > keep_sessions:
        trades_from = days[-keep_sessions]["d"]
        shown = [t for t in trades if t["d"] >= trades_from]
    s = {"id": sid, "name": name, "code": code, "group": group, "style": style, "kind": "trades", "desc": desc,
         "src": src, "since": days[0]["d"] if days else None, "risk_unit": risk_unit, "status": status or {},
         "days": days, "trades": [clean(t) for t in shown], "trades_from": trades_from,
         "open": [clean(o) for o in (opens or [])]}
    s.update(extra)
    return clean(s) | {"days": days, "trades": s["trades"], "open": s["open"]}


def r_of(side, entry, exit_, stop):
    """Gross R from prices (for systems that log money but not R)."""
    if None in (entry, exit_, stop) or entry == stop:
        return None
    risk = abs(entry - stop)
    move = (exit_ - entry) if side == "L" else (entry - exit_)
    return round(move / risk, 3)


# ─── research strategies (scan job: paper_trading/research_live.py) ───────

def build_research(root, mk):
    ls = os.path.join(root, "live_state")
    csv_path, st_path = os.path.join(ls, "paper_trades_all.csv"), os.path.join(ls, "state.json")
    rows = read_csv(csv_path)
    st = read_json(st_path, {}) or {}
    have = {r.get("trade_id") for r in rows}
    for t in st.get("trades", []):          # today's closed trades are appended to the CSV only at the EOD report
        if t.get("status") == "CLOSED" and t.get("research_trade") and t.get("id") not in have:
            rows.append({**t["research_trade"], "date": st.get("date"), "trade_id": t["id"], "reason": t.get("reason", "")})
    ratios = sorted(abs(fnum(r.get("pnl_currency"), 6) / fnum(r.get("r_multiple"), 6)) for r in rows
                    if fnum(r.get("r_multiple")) and abs(fnum(r.get("r_multiple"))) > 0.2 and fnum(r.get("pnl_currency")))
    risk_unit = round(ratios[len(ratios) // 2]) if ratios else 500.0      # 0.5% of the 100k paper account
    by = defaultdict(list)
    for r in rows:
        if not r.get("date") or not r.get("strategy"):
            continue
        by[r["strategy"]].append({
            "d": r["date"], "sym": r.get("symbol"),
            "side": "S" if str(r.get("direction", "")).lower() == "short" else "L",
            "et": ts(r.get("entry_time")), "xt": ts(r.get("exit_time")), "ep": fnum(r.get("entry"), 4),
            "xp": fnum(r.get("exit_price"), 4), "sl": fnum(r.get("stop_loss"), 4), "tp": fnum(r.get("target"), 4),
            "q": fnum(r.get("position_size"), 2), "out": r.get("outcome"), "r": fnum(r.get("r_multiple"), 3),
            "pnl": fnum(r.get("pnl_currency"), 2), "ret": fnum(r.get("pnl_pct"), 3),
            "hold": int(float(r["candles_held"])) * 5 if fnum(r.get("candles_held")) is not None else None,
            "note": (r.get("reason") or "")[:120] or None})
    opens = defaultdict(list)
    counts = defaultdict(lambda: defaultdict(int))
    for t in st.get("trades", []):
        counts[t.get("strategy")][t.get("status")] += 1
        if t.get("status") != "OPEN":
            continue
        side = "S" if str(t.get("direction", "")).lower() == "short" else "L"
        ep, last, sl = fnum(t.get("entry")), fnum(t.get("last_price")), fnum(t.get("stop_loss"))
        r_now = r_of(side, ep, last, sl)
        opens[t["strategy"]].append({"sym": t.get("symbol"), "side": side, "et": ts(t.get("entry_time")),
                                     "ep": fnum(ep, 4), "sl": fnum(sl, 4), "tp": fnum(t.get("target"), 4),
                                     "last": fnum(last, 4), "r": r_now,
                                     "upnl": round(r_now * risk_unit, 2) if r_now is not None else None})
    updated = mtime_iso(csv_path, st_path)
    out = []
    for code in sorted(set(by) | set(opens), key=lambda c: (RESEARCH_NAMES.get(c, ("", "Research"))[1] != "ML filters", c)):
        name, group, desc = RESEARCH_NAMES.get(code, (code.replace("_", " ").title(), "Research", "Research strategy"))
        c = counts.get(code, {})
        today = f"today {st.get('date')}: {c.get('CLOSED', 0)} closed, {c.get('OPEN', 0)} open" if st.get("date") else None
        out.append(system(f"research_{code.lower()}", name, code, group, "intraday", desc, trades=by.get(code),
                          opens=opens.get(code), risk_unit=risk_unit, keep_sessions=RESEARCH_TRADE_SESSIONS,
                          status=clean({"level": "ok", "text": today, "updated": updated, "asof": st.get("date")})))
    return out


# ─── US models ────────────────────────────────────────────────────────────

def _us_v3_risk(day):
    for start, risk in US_V3_RISK:
        if day >= start:
            return US_V3_EQUITY * risk
    return US_V3_EQUITY * US_V3_RISK[-1][1]


def build_us_v3(root, mk):
    d = os.path.join(root, "live_state", "us_v3")
    st = read_json(os.path.join(d, "state.json"), {}) or {}
    trades = []
    for r in read_csv(os.path.join(d, "trades.csv")):
        R = fnum(r.get("R"), 3)
        side = "S" if str(r.get("side")).strip() in ("-1", "-1.0") else "L"
        trades.append({"d": r["date"], "sym": r["symbol"],
                       "side": side, "et": ts(r.get("entry_ts")), "xt": ts(r.get("exit_ts")),
                       "ep": fnum(r.get("entry"), 4), "xp": fnum(r.get("exit"), 4), "sl": fnum(r.get("stop"), 4),
                       "out": r.get("outcome"), "r": R,
                       "pnl": round(R * _us_v3_risk(r["date"]), 2) if R is not None else None,
                       "ret": fnum(r.get("ret_pct"), 3),
                       "tags": [f"ML score {float(r['pred']):.2f}" if fnum(r.get("pred")) is not None else None,
                                f"RVOL {float(r['rvol']):.1f}x" if fnum(r.get("rvol")) is not None else None]})
    opens = [{"sym": p["symbol"], "side": "S" if p.get("side") == -1 else "L", "et": ts(p.get("entry_ts")),
              "ep": fnum(p.get("entry"), 4), "sl": fnum(p.get("stop"), 4)}
             for p in st.get("positions", []) if p.get("status") == "OPEN" and st.get("date")]
    cap = st.get("cap")
    text = f"{st.get('date')}: {len(st.get('taken', []))} taken, limit {cap}/day" if st.get("date") else None
    level = "warn" if "THROTTLED" in str(st.get("cap_reason", "")) else "ok"
    for t in trades:
        t["tags"] = [x for x in t["tags"] if x]
    return [system("us_v3", "V3 Intraday ORB", "us_v3", "Models", "intraday",
                   "Opening-range breakout on stocks in play, ML-ranked (LightGBM), long + short, 1% risk/trade",
                   trades=trades, opens=opens, risk_unit=_us_v3_risk("9999-12-31"),
                   status=clean({"level": level, "text": text, "note": st.get("cap_reason"), "asof": st.get("date"),
                                 "updated": mtime_iso(os.path.join(d, "state.json"), os.path.join(d, "trades.csv"))}))]


def _runners(sid, name, code, desc, d, qty_key, tag_fn, opens_fn, style="intraday", side_key=None):
    st = read_json(os.path.join(d, "state.json"), {}) or {}
    trades = []
    for r in read_csv(os.path.join(d, "trades.csv")):
        side = "S" if str(r.get(side_key or "", "")).upper() == "SHORT" else "L"
        trades.append({"d": r["date"], "sym": r["symbol"],
                       "side": side, "et": ts(r.get("entry_ts")), "xt": ts(r.get("exit_ts")),
                       "ep": fnum(r.get("entry"), 4), "xp": fnum(r.get("exit"), 4),
                       "sl": fnum(r.get("stop_initial"), 4), "q": fnum(r.get(qty_key), 0), "out": r.get("outcome"),
                       "r": fnum(r.get("R"), 3), "pnl": fnum(r.get("pnl"), 2), "ret": fnum(r.get("ret_pct"), 3),
                       "tags": [x for x in tag_fn(r) if x]})
    taken = st.get("taken")
    text = f"{st.get('date')}: {taken} taken today" if st.get("date") else None
    return system(sid, name, code, "Models", style, desc, trades=trades, opens=opens_fn(st),
                  status=clean({"level": "ok", "text": text, "asof": st.get("date"),
                                "updated": mtime_iso(os.path.join(d, "state.json"), os.path.join(d, "trades.csv"))}))


def _open_runner(st, qty_key="shares"):
    return [{"sym": p["symbol"], "side": "S" if str(p.get("side", "")).upper() == "SHORT" else "L",
             "et": ts(p.get("entry_ts")), "ep": fnum(p.get("entry"), 4), "sl": fnum(p.get("stop_initial"), 4),
             "stop_now": fnum(p.get("stop_now"), 4), "q": fnum(p.get(qty_key), 0)}
            for p in st.get("positions", []) if p.get("status") == "OPEN"]


def build_us_scr(root, mk):
    def tags(r):
        news = fnum(r.get("news_score"))
        return [f"{r.get('pillars')}/5 pillars" if r.get("pillars") else None,
                f"+{float(r['pct']):.0f}% day" if fnum(r.get("pct")) is not None else None,
                f"RVOL {float(r['rvol']):.0f}x" if fnum(r.get("rvol")) is not None else None,
                f"news {news:.2f} ({r.get('news_method')})" if news else None]
    return [_runners("us_scr", "SCR Small-Cap Runners", "us_scr",
                     "Small/micro-cap runners (< $2B) with unusual volume, A+ pullback setups only",
                     os.path.join(root, "live_state", "us_scr"), "shares", tags, _open_runner)]


def build_us_lcr(root, mk):
    def tags(r):
        return ["⭐ PERFECT" if truthy(r.get("perfect")) else None, r.get("bucket"),
                f"+{float(r['pct']):.1f}% day" if fnum(r.get("pct")) is not None else None,
                f"RVOL {float(r['rvol']):.1f}x" if fnum(r.get("rvol")) is not None else None]
    return [_runners("us_lcr", "LCR Large-Cap Runners", "us_lcr",
                     "Mid/large/mega caps (>= $2B) with time-adjusted volume shocks, pullback above the daily EMAs",
                     os.path.join(root, "live_state", "us_lcr"), "shares", tags, _open_runner)]


# ─── swing portfolios (V5, both markets) ──────────────────────────────────

def _portfolio(sid, name, code, desc, d, bench_key, bench_name):
    st = read_json(os.path.join(d, "state.json"), {}) or {}
    fills = [{"d": r["date"], "sym": r["symbol"], "act": r.get("side"), "q": fnum(r.get("qty"), 0),
              "px": fnum(r.get("price"), 4), "cf": fnum(r.get("cash_flow"), 2), "pnl": fnum(r.get("realised_pnl"), 2)}
             for r in read_csv(os.path.join(d, "trades.csv"))]
    start_eq = fnum(st.get("start_equity")) or 0.0
    daily = st.get("daily", [])
    fills_by_day = defaultdict(list)
    for f in fills:
        fills_by_day[f["d"]].append(f)
    days, prev = [], start_eq
    for row in daily:
        eq = fnum(row.get("equity"))
        if eq is None:
            continue
        fl = fills_by_day.get(row["date"], [])
        sells = [f["pnl"] for f in fl if f.get("act") == "SELL" and f.get("pnl")]
        pnl = round(eq - prev, 2)
        days.append(clean({"d": row["date"], "n": len(fl), "w": sum(p > 0 for p in sells),
                           "l": sum(p < 0 for p in sells), "pnl": pnl,
                           "gw": round(max(pnl, 0.0), 2), "gl": round(min(pnl, 0.0), 2),
                           "best": round(max(sells), 2) if sells else None,
                           "worst": round(min(sells), 2) if sells else None}) | {"n": len(fl), "pnl": pnl})
        prev = eq
    eq_now = fnum(daily[-1].get("equity")) if daily else start_eq
    holdings = []
    for sym, p in (st.get("positions") or {}).items():
        q, cost, last = fnum(p.get("qty")), fnum(p.get("cost")), fnum(p.get("last_px"))
        val = q * last if q is not None and last is not None else None
        holdings.append(clean({"sym": sym, "q": q, "cost": fnum(cost, 2), "last": fnum(last, 4),
                               "val": round(val, 2) if val is not None else None,
                               "upnl": round(val - cost, 2) if val is not None and cost is not None else None,
                               "w": round(val / eq_now * 100, 2) if val is not None and eq_now else None,
                               "since": p.get("since"), "sector": p.get("sector")}))
    holdings.sort(key=lambda h: -(h.get("val") or 0))
    nxt = None
    if st.get("sessions_since_rebalance") is not None:
        nxt = f"rebalanced {st.get('last_rebalance')}, {st['sessions_since_rebalance']} session(s) ago"
    s = system(sid, name, code, "Models", "swing", desc,
               status=clean({"level": "warn" if st.get("rebalance_pending") else "ok", "text": nxt,
                             "asof": st.get("last_day"),
                             "updated": mtime_iso(os.path.join(d, "state.json"))}))
    s.update({"kind": "portfolio", "days": days, "since": daily[0]["date"] if daily else st.get("start"),
              "fills": fills, "holdings": holdings, "bench": bench_name,
              "equity": [clean({"d": r["date"], "eq": fnum(r.get("equity"), 2), "bm": fnum(r.get(bench_key), 4)})
                         for r in daily],
              "start_equity": start_eq, "bench_start": fnum(st.get(f"{bench_key}_start"), 4),
              "cash": fnum(st.get("cash"), 2)})
    return s


def build_us_v5(root, mk):
    return [_portfolio("us_v5", "V5 Swing Portfolio", "us_v5",
                       "Top-20 S&P 500 names by the V5 factor model, rebalanced every 5 sessions, $100k paper",
                       os.path.join(root, "live_state", "us_v5"), "spy", "SPY")]


def build_nse_v5(root, mk):
    return [_portfolio("nse_v5", "V5 Swing Portfolio", "nse_v5",
                       "Top-20 NSE names by the V5 model, rebalanced every 5 sessions, Rs 10 lakh paper",
                       os.path.join(root, "live_state", "v5"), "nifty", "NIFTY 50")]


# ─── NSE models ───────────────────────────────────────────────────────────

def build_nse_v33(root, mk):
    d = os.path.join(root, "live_state", "v33")
    st = read_json(os.path.join(d, "state.json"), {}) or {}
    trades = []
    for r in read_csv(os.path.join(d, "trades.csv")):
        ep, xp, sl = fnum(r.get("entry")), fnum(r.get("exit")), fnum(r.get("stop"))
        trades.append({"d": r["date"], "sym": r["symbol"], "side": "L",
                       "et": ts(r.get("entry_time")), "xt": ts(r.get("exit_time")), "ep": fnum(ep, 2),
                       "xp": fnum(xp, 2), "sl": fnum(sl, 2), "q": fnum(r.get("qty"), 0), "out": r.get("outcome"),
                       "r": r_of("L", ep, xp, sl), "pnl": fnum(r.get("net"), 2), "gross": fnum(r.get("gross"), 2),
                       "fees": fnum(r.get("charges"), 2),
                       "ret": round(fnum(r.get("net_pct")) * 100, 3) if fnum(r.get("net_pct")) is not None else None,
                       "tags": [x for x in (f"rank {r.get('rank')}" if r.get("rank") else None,
                                            f"score {float(r['score']):.4f}" if fnum(r.get("score")) is not None else None,
                                            "late fill" if truthy(r.get("late_fill")) else None) if x]})
    opens = [{"sym": p["symbol"], "side": "L", "et": ts(p.get("entry_time")), "ep": fnum(p.get("entry"), 2),
              "sl": fnum(p.get("stop"), 2), "q": fnum(p.get("qty"), 0)}
             for p in st.get("picks", []) if p.get("status") == "OPEN"]
    text = (f"{st.get('date')}: {len(st.get('picks', []))} picks, model trained through {st.get('model_trained_through')}"
            if st.get("date") else None)
    return [system("nse_v33", "V3.3 Intraday Picks", "v33", "Models", "intraday",
                   "Top-5 morning picks from the V3.3 ranking model, square-off 15:10, exact Zerodha-style charges",
                   trades=trades, opens=opens,
                   status=clean({"level": "ok", "text": text, "asof": st.get("date"),
                                 "updated": mtime_iso(os.path.join(d, "state.json"), os.path.join(d, "trades.csv"))}))]


def build_nse_scr(root, mk):
    def tags(r):
        return [r.get("cap"), f"+{float(r['pct']):.1f}% day" if fnum(r.get("pct")) is not None else None,
                f"RVOL {float(r['rvol']):.1f}x" if fnum(r.get("rvol")) is not None else None]
    return [_runners("nse_scr", "SCR Small-Cap Runners", "nse_scr",
                     "NSE small caps with volume spikes: pullback longs + fade-the-spike shorts, intraday",
                     os.path.join(root, "live_state", "nse_scr"), "qty", tags,
                     lambda st: _open_runner(st, "qty"), side_key="side")]


def build_nse_lcr(root, mk):
    d = os.path.join(root, "live_state", "nse_lcr")
    st = read_json(os.path.join(d, "state.json"), {}) or {}
    trades = []
    for r in read_csv(os.path.join(d, "trades.csv")):
        ep, xp, sl = fnum(r.get("entry_raw")), fnum(r.get("exit")), fnum(r.get("stop"))
        kind = (r.get("kind") or "").upper()
        trades.append({"d": r.get("exit_date"),
                       "sym": r["symbol"], "side": "L", "et": ts(r.get("entry_ts")), "xt": ts(r.get("exit_ts")),
                       "ep": fnum(ep, 2), "xp": fnum(xp, 2), "sl": fnum(sl, 2), "q": fnum(r.get("qty"), 0),
                       "out": r.get("outcome"), "r": r_of("L", ep, xp, sl), "pnl": fnum(r.get("pnl"), 2),
                       "ret": fnum(r.get("net_pct"), 3),
                       "tags": [x for x in ("⭐ PERFECT" if truthy(r.get("perfect")) else None,
                                            "SWING" if kind == "SWING" else "INTRADAY", r.get("bucket"),
                                            f"{r.get('sessions')} session(s)" if kind == "SWING" else None) if x]})
    opens = [{"sym": p["symbol"], "side": "L", "et": ts(p.get("entry_ts")), "ep": fnum(p.get("entry_raw"), 2),
              "sl": fnum(p.get("stop"), 2), "q": fnum(p.get("qty"), 0), "note": "intraday"}
             for p in st.get("positions", []) if p.get("status") == "OPEN"]
    for p in st.get("swing", []):
        q, ep, last = fnum(p.get("qty")), fnum(p.get("entry_raw")), fnum(p.get("last_px"))
        opens.append({"sym": p["symbol"], "side": "L", "et": ts(p.get("entry_ts")), "ep": fnum(ep, 2),
                      "sl": fnum(p.get("stop"), 2), "q": q, "last": fnum(last, 2),
                      "upnl": round(q * (last - ep), 2) if None not in (q, ep, last) else None,
                      "note": f"swing, {p.get('sessions', 0)} session(s)"})
    text = f"{st.get('date')}: {st.get('taken', 0)} taken, {len(st.get('swing', []))} swing open" if st.get("date") else None
    return [system("nse_lcr", "LCR Large-Cap Runners", "nse_lcr", "Models", "hybrid",
                   "NIFTY 250 + large caps with volume shocks above the 10/20-day EMAs; intraday, carried up to 5 sessions",
                   trades=trades, opens=opens,
                   status=clean({"level": "ok", "text": text, "asof": st.get("date"),
                                 "updated": mtime_iso(os.path.join(d, "state.json"), os.path.join(d, "trades.csv"))}))]


BUILDERS = {"research": build_research, "us_v3": build_us_v3, "us_v5": build_us_v5, "us_scr": build_us_scr,
            "us_lcr": build_us_lcr, "nse_v33": build_nse_v33, "nse_v5": build_nse_v5, "nse_scr": build_nse_scr,
            "nse_lcr": build_nse_lcr}

# a group counts as restored this run when its state is present in the workspace
PRESENT = {"research": ("live_state/paper_trades_all.csv", "live_state/state.json"),
           "us_v3": ("live_state/us_v3",), "us_v5": ("live_state/us_v5",), "us_scr": ("live_state/us_scr",),
           "us_lcr": ("live_state/us_lcr",), "nse_v33": ("live_state/v33",), "nse_v5": ("live_state/v5",),
           "nse_scr": ("live_state/nse_scr",), "nse_lcr": ("live_state/nse_lcr",)}

ORDER = ["us_v3", "us_v5", "us_scr", "us_lcr", "nse_v33", "nse_v5", "nse_scr", "nse_lcr", "research"]


# ─── health (workflow runs) ───────────────────────────────────────────────

def gh_api(path):
    """GET a GitHub API path with the gh CLI (runner) or plain HTTPS + GH_TOKEN."""
    try:
        out = subprocess.run(["gh", "api", path], capture_output=True, text=True, timeout=60)
        return json.loads(out.stdout) if out.returncode == 0 else None
    except FileNotFoundError:
        import urllib.request
        headers = {"Accept": "application/vnd.github+json"}
        if os.environ.get("GH_TOKEN"):                   # an empty "Bearer " is rejected even for public data
            headers["Authorization"] = f"Bearer {os.environ['GH_TOKEN']}"
        req = urllib.request.Request(f"https://api.github.com/{path}", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception:
            return None
    except Exception:
        return None


def health_snapshot(repo, workflow):
    """The production workflow's runs of the last 3 hours + the job results of its newest finished run."""
    since = (now_utc() - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    data = gh_api(f"repos/{repo}/actions/workflows/{workflow}/runs?per_page=100&created=%3E%3D{since}")
    if not data:
        return None
    runs = list(data.get("workflow_runs", []))
    runs.sort(key=lambda r: r["created_at"])
    counts = defaultdict(int)
    for r in runs:
        counts[r.get("conclusion") or r.get("status") or "unknown"] += 1
    times = [datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")) for r in runs]
    gaps = [(b - a).total_seconds() / 60 for a, b in zip(times, times[1:])]
    done = [r for r in runs if r.get("status") == "completed"]
    jobs = []
    if done:
        jd = gh_api(f"repos/{repo}/actions/runs/{done[-1]['id']}/jobs?per_page=50") or {}
        jobs = [clean({"name": j.get("name"), "result": j.get("conclusion") or j.get("status"),
                       "started": j.get("started_at"), "done": j.get("completed_at")}) for j in jd.get("jobs", [])]
    fails = [clean({"id": r["id"], "at": r["created_at"], "result": r.get("conclusion"), "url": r.get("html_url")})
             for r in runs if r.get("conclusion") in ("failure", "timed_out", "startup_failure")][-5:]
    return clean({"window_h": 3, "runs": dict(counts), "total": len(runs),
                  "last_run_at": runs[-1]["created_at"] if runs else None,
                  "max_gap_min": round(max(gaps), 1) if gaps else None,
                  "median_gap_min": round(sorted(gaps)[len(gaps) // 2], 1) if gaps else None,
                  "last_jobs": jobs, "last_jobs_run": done[-1]["id"] if done else None, "failures": fails})


# ─── plan / build ─────────────────────────────────────────────────────────

def load_prev(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with gzip.open(path, "rt") as f:
            prev = json.load(f)
        return prev if prev.get("schema") == SCHEMA else None
    except Exception:
        return None


def newest_cache_keys(repo, groups):
    """Newest cache key per group prefix (one API call; caches sorted newest first)."""
    data = gh_api(f"repos/{repo}/actions/caches?per_page=100&sort=created_at&direction=desc") or {}
    newest = {}
    for c in data.get("actions_caches", []):
        key = c.get("key", "")
        for g, prefix in groups.items():
            rest = key[len(prefix):]
            if key.startswith(prefix) and re.fullmatch(r"\d+-\d+", rest) and g not in newest:
                newest[g] = key            # '<prefix><run_id>-<attempt>' only (skips e.g. us-v5-done-*)
    return newest


def cmd_plan(a):
    mk = MARKETS[a.market]
    prev = load_prev(a.prev) or {}
    have = prev.get("sources", {})
    newest = newest_cache_keys(a.repo, mk["groups"])
    lines = []
    for g in mk["groups"]:
        key = newest.get(g, "")
        restore = key if key and (key != have.get(g) or a.force or g not in {s.get("src") for s in prev.get("systems", [])}) else ""
        lines.append(f"{g}={restore}")
        print(f"[plan] {g:9s} newest={key or '-':40s} published={have.get(g, '-'):40s} restore={'yes' if restore else 'no'}",
              file=sys.stderr)
    lines.append(f"sources={json.dumps(newest, separators=(',', ':'))}")
    print("\n".join(lines))
    if a.health_out:
        snap = health_snapshot(a.repo, mk["workflow"])
        if snap:
            with open(a.health_out, "w") as f:
                json.dump(snap, f)


def cmd_build(a):
    mk = MARKETS[a.market]
    prev = load_prev(a.prev) or {}
    prev_by_src = defaultdict(list)
    for s in prev.get("systems", []):
        prev_by_src[s.get("src")].append(s)
    sources = dict(prev.get("sources", {}))
    newest = json.loads(a.sources) if a.sources else {}
    systems, notes = [], []
    if not newest:
        notes.append("could not list the state caches this run - figures may be from an earlier run")
    for g in sorted(mk["groups"], key=ORDER.index):
        present = any(os.path.exists(os.path.join(a.root, p)) for p in PRESENT[g])
        if present:
            try:
                built = BUILDERS[g](a.root, mk)
                for s in built:
                    s["src"] = g
                systems += built
                if newest.get(g):
                    sources[g] = newest[g]
                continue
            except Exception as e:                       # never lose the dashboard over one system
                notes.append(f"{g}: export failed ({type(e).__name__}: {e}) - showing the last good data")
                print(f"[build] {g} FAILED: {e!r}", file=sys.stderr)
        systems += prev_by_src.get(g, [])
    health = {}
    if a.health and os.path.exists(a.health):
        health = read_json(a.health, {}) or {}
    elif prev.get("health"):
        health = prev["health"]
    run_id = os.environ.get("GITHUB_RUN_ID")
    out = {"schema": SCHEMA, "market": a.market, "ccy": mk["ccy"], "tz": mk["tz"],
           "session": {"open": mk["open"], "close": mk["close"]}, "bench": mk["bench"],
           "generated_at": now_utc().isoformat().replace("+00:00", "Z"),
           "repo": a.repo, "run_id": int(run_id) if run_id and run_id.isdigit() else None,
           "health": health, "notes": notes, "sources": sources, "systems": systems}
    raw = json.dumps(out, separators=(",", ":"), ensure_ascii=False).encode()
    with gzip.open(a.out, "wb", compresslevel=9) as f:
        f.write(raw)
    n_tr = sum(len(s.get("trades", [])) for s in systems)
    print(f"[build] {a.market}: {len(systems)} systems, {n_tr} trades shown, json {len(raw) / 1e3:.0f} KB, "
          f"gz {os.path.getsize(a.out) / 1e3:.0f} KB -> {a.out}", file=sys.stderr)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "build"):
        q = sub.add_parser(name)
        q.add_argument("--market", required=True, choices=sorted(MARKETS))
        q.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
        q.add_argument("--prev", help="previously published dashboard.json.gz (carry-over + change detection)")
    pl = sub.choices["plan"]
    pl.add_argument("--force", action="store_true", help="restore every group")
    pl.add_argument("--health-out", help="write the workflow-runs health snapshot here")
    b = sub.choices["build"]
    b.add_argument("--root", default=".")
    b.add_argument("--out", default="dashboard.json.gz")
    b.add_argument("--sources", help="JSON {group: cache key} from the plan step")
    b.add_argument("--health", help="health snapshot written by the plan step")
    a = p.parse_args(argv)
    (cmd_plan if a.cmd == "plan" else cmd_build)(a)


if __name__ == "__main__":
    main()
