"""
US V5.0 live paper trading (swing ranking model) - one step per trading day.

    python -m us_v5.live                  # what the US Alert Scan `v5` job runs
    python -m us_v5.live --no-telegram    # console only
    python -m us_v5.live --report         # print the portfolio state

Each trading day, on the first run after 09:35 ET (cron-job.org trigger):
  1. refresh Yahoo daily bars up to YESTERDAY's close (S&P 500 + context)
  2. rebuild V5 features and score the top-100 universe with the committed
     model (us_v5/model/us_v5_model.joblib), smoothed exactly as in research
  3. every 5th session (and on the first day): rebalance to the top-20
     target at TODAY'S OPEN (Yahoo daily bar open) + slippage + SEC/FINRA
     fees; holdings stay while ranked within the top 50 (no-trade band)
  4. Telegram: rebalance orders (BUY / SELL / HOLD) or a daily portfolio
     update, all-time P&L vs SPY since the start

State: live_state/us_v5/state.json + trades.csv. Idempotent: a second run on
the same day does nothing. Paper only - no orders are sent anywhere.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.environ.get("US_V5_STATE_DIR", os.path.join(ROOT, "live_state", "us_v5"))
MODEL = os.path.join(ROOT, "us_v5", "model", "us_v5_model.joblib")
WINDOW_DAYS = 1100                 # calendar days of history (> 252 sessions + 200-day warm-ups)
START_HHMM = 9 * 60 + 35           # today's official open is known a few minutes after 09:30
REWEIGHT_TOL = 0.25


def _path(name: str) -> str:
    os.makedirs(STATE_DIR, exist_ok=True)
    return os.path.join(STATE_DIR, name)


def load_state() -> dict:
    try:
        with open(_path("state.json")) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(st: dict) -> None:
    tmp = _path("state.json.tmp")
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1, default=str)
    os.replace(tmp, _path("state.json"))


def notify(text: str, telegram: bool, table: list | None = None) -> None:
    """`table` lines go in a monospace <pre> block so Telegram keeps columns
    aligned; falls back to plain text if HTML is rejected."""
    plain = text + ("\n" + "\n".join(table) if table else "")
    print(plain, flush=True)
    if not telegram:
        return
    from paper_trading import research_live as rl
    if table:
        msg = html.escape(text) + "\n<pre>" + html.escape("\n".join(table)) + "</pre>"
        if len(msg) < 3900:
            import config
            import requests
            if config.LIVE_RESEARCH_TELEGRAM and config.TELEGRAM_BOT_TOKEN:
                try:
                    r = requests.post(f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage",
                                      data={"chat_id": config.LIVE_RESEARCH_CHAT_ID or config.TELEGRAM_CHAT_ID,
                                            "text": msg, "parse_mode": "HTML"}, timeout=20)
                    if r.status_code == 200:
                        return
                    print(f"[US V5] Telegram HTML failed: {r.status_code} {r.text[:150]}")
                except Exception as e:
                    print(f"[US V5] Telegram error: {e!r}")
    rl.send_text(plain)


def _holding_rows(st: dict, prices: dict, equity: float) -> list:
    rows = []
    for sym, p in st["positions"].items():
        px = float(prices.get(sym, p["last_px"]))
        val = p["qty"] * px
        rows.append((sym, p["qty"], (val / p["cost"] - 1) * 100, val - p["cost"], val / equity * 100))
    return sorted(rows, key=lambda r: -r[2])


def holdings_table(st: dict, prices: dict, equity: float) -> list:
    """Every holding, best first - 36 chars wide (fits a phone)."""
    rows = _holding_rows(st, prices, equity)
    if not rows:
        return []
    out = [f"{'Stock':<7} {'Qty':>5} {'P&L%':>6} {'P&L $':>8} {'Wt%':>5}", "-" * 36]
    out += [f"{s[:7]:<7} {q:>5} {pct:>+6.1f} {pnl:>+8,.0f} {wt:>5.1f}" for s, q, pct, pnl, wt in rows]
    cost = sum(p["cost"] for p in st["positions"].values())
    pnl = sum(r[3] for r in rows)
    out += ["-" * 36,
            f"{'Total':<7} {len(rows):>5} {pnl / cost * 100:>+6.1f} {pnl:>+8,.0f} {sum(r[4] for r in rows):>5.0f}",
            f"Up {sum(r[3] > 0 for r in rows)} | Down {sum(r[3] < 0 for r in rows)} | Cash ${st['cash']:,.0f}"]
    return out


def _set_output(changed: bool) -> None:
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"changed={'true' if changed else 'false'}\n")


# ─── data ────────────────────────────────────────────────────────────────

def todays_opens(symbols: list[str], today: date) -> dict:
    """Today's official open per symbol from Yahoo's daily bar (None if the
    bar for today is missing - holiday, halt or Yahoo lag)."""
    import yfinance as yf
    out = {s: None for s in symbols}
    if not symbols:
        return out
    for attempt in range(3):
        try:
            d = yf.download(sorted(symbols), start=str(today), end=str(today + timedelta(days=1)), interval="1d",
                            auto_adjust=False, progress=False, group_by="ticker", threads=True)
            break
        except Exception as e:
            print(f"[US V5] open-price download failed ({e!r}), retry {attempt + 1}", flush=True)
            time.sleep(5)
    else:
        return out
    if d is None or d.empty:
        return out
    for s in symbols:
        try:
            x = d[s] if s in d.columns.get_level_values(0) else None
            if x is None:
                continue
            x = x[pd.DatetimeIndex(x.index).tz_localize(None).normalize() == pd.Timestamp(today)]
            o = float(x["Open"].iloc[0]) if len(x) else float("nan")
            if o == o and o > 0:
                out[s] = o
        except Exception:
            continue
    return out


def score_previous_close(bundle: dict, today: date):
    """Smoothed V5 scores at the latest session before today + its info."""
    from us_v5 import data as data_mod
    from us_v5 import features as feat
    from us_v5.core import config as cfgmod
    from us_v5.scoring import score_live
    cfg = cfgmod.load({"data.start": str(today - timedelta(days=WINDOW_DAYS)),
                       "data.end": str(today - timedelta(days=1))})
    for k in ("universe", "label", "features"):
        cfg[k] = bundle["config"][k]
    data = data_mod.load_data(cfg, refresh_days=10)
    df = feat.build(data, cfg, with_targets=False)
    days = sorted(df["date"].unique())[-40:]
    recent = df[df["date"].isin(days)]
    s = score_live(bundle, recent)
    last = s["date"].max()
    scores = s[s["date"] == last].set_index("entity")["score"]
    info = recent[recent["date"] == last].set_index("entity")
    day = data["panel"][data["panel"]["date"] == last]
    closes = day.set_index("symbol")["close"]
    spy = data["context"]["SPY"].dropna()
    return pd.Timestamp(last), scores, info, closes, spy, data["panel"]


def equity_at(st: dict, prices: dict) -> float:
    return st["cash"] + sum(p["qty"] * prices.get(sym, p["last_px"]) for sym, p in st["positions"].items())


# ─── one trading day ─────────────────────────────────────────────────────

def run_day(telegram: bool) -> bool:
    """Returns True if state changed."""
    import joblib
    from us_v5.core import costs as cost_mod
    from us_v5.core import sizing

    now = datetime.now(ET)
    today = now.date()
    if now.weekday() >= 5:
        print("Weekend - nothing to do.")
        return False
    st = load_state()
    if st.get("last_day") == str(today):
        print(f"[US V5] already done for {today}.")
        return False
    if now.hour * 60 + now.minute < START_HHMM:
        print(f"{now:%H:%M} ET - waiting for 09:35 ET (needs today's opening prices).")
        return False
    # Holiday: no SPY bar for today -> not a session, nothing counted or traded.
    if todays_opens(["SPY"], today)["SPY"] is None:
        if st.get("closed_day") != str(today):
            st = st or {}
            st.update(closed_day=str(today), last_day=str(today))
            save_state(st)
            notify(f"US V5.0 SWING PAPER {today}: no SPY bar today (market holiday?) - no session counted, "
                   f"no trades.", telegram)
            return True
        return False

    bundle = joblib.load(MODEL)
    cfg = bundle["config"]
    t0 = time.time()
    last, scores, info, closes, spy, panel = score_previous_close(bundle, today)
    print(f"[US V5] scored {len(scores)} stocks from the {last.date()} close in {time.time() - t0:.0f} s", flush=True)

    if not st or "positions" not in st:
        st = {"start": str(today), "cash": float(cfg["portfolio"]["capital"]), "positions": {},
              "sessions_since_rebalance": 10 ** 6, "equity_history": [], "spy_start": None, "start_equity": None}
    prices_prev = {sym: float(closes.get(sym, p["last_px"])) for sym, p in st["positions"].items()}
    for sym, p in st["positions"].items():
        p["last_px"] = prices_prev[sym]
    eq_prev = equity_at(st, prices_prev)
    spy_last = float(spy.loc[:last].iloc[-1])
    st["spy_start"] = st["spy_start"] or spy_last
    st["start_equity"] = st["start_equity"] or eq_prev
    st["sessions_since_rebalance"] = st.get("sessions_since_rebalance", 0) + 1
    never_invested = not st["positions"] and not os.path.exists(_path("trades.csv"))
    rebalance = (st["sessions_since_rebalance"] >= cfg["portfolio"]["rebalance_days"]
                 or st.get("rebalance_pending", False) or never_invested)

    lines = []
    if rebalance:
        c = cfg["costs"]
        held = {p["entity"] for p in st["positions"].values()}
        names = sizing.select(scores, held, cfg["portfolio"]["top_n"], cfg["portfolio"]["hold_buffer"])
        ex = sizing.Exposure(cfg)
        ex.history = [float(x) for x in st["equity_history"]][-ex.window:]
        w = sizing.weights(names, info["vol_63"], info["industry"], cfg) * ex.scale(eq_prev)
        st["equity_history"] = ex.history
        target = {info.loc[e, "symbol"]: float(wt) for e, wt in w.items()}
        opens = todays_opens(sorted(set(target) | set(st["positions"])), today)
        eq_open = st["cash"] + sum(p["qty"] * (opens.get(sym) or p["last_px"]) for sym, p in st["positions"].items())
        buys, sells, holds = [], [], []
        for sym in list(st["positions"]):                       # sells / reductions first
            p, px = st["positions"][sym], opens.get(sym)
            tgt = target.get(sym, 0.0) * eq_open
            cur = p["qty"] * (px or p["last_px"])
            if tgt > 0 and (cur - tgt) / max(tgt, 1) <= REWEIGHT_TOL:
                holds.append(sym)
                continue
            if px is None:
                lines.append(f"⚠️ {sym}: no opening price - sell postponed")
                continue
            qty = p["qty"] if tgt <= 0 else int((cur - tgt) // px)
            if qty <= 0:
                holds.append(sym)
                continue
            fill = cost_mod.fill_price(px, "sell", c)
            gross = qty * fill
            proceeds = gross - cost_mod.order_charges(gross, "sell", c)
            basis = p["cost"] * qty / p["qty"]
            st["cash"] += proceeds
            p["qty"] -= qty
            p["cost"] -= basis
            pnl = proceeds - basis
            _log_trade(today, "SELL", sym, qty, fill, proceeds, pnl)
            sells.append(f"{sym} {qty} @ {fill:.2f} ({pnl:+,.0f}, {pnl / basis * 100:+.1f}%)")
            if p["qty"] <= 0:
                del st["positions"][sym]
            else:
                holds.append(sym)
        for sym, wt in sorted(target.items(), key=lambda kv: -kv[1]):   # buys / increases
            px = opens.get(sym)
            cur = st["positions"].get(sym, {}).get("qty", 0) * (px or 0)
            tgt = wt * eq_open
            if px is None or tgt <= 0 or (cur > 0 and (tgt - cur) / tgt < REWEIGHT_TOL):
                if px is None and sym not in st["positions"]:
                    lines.append(f"⚠️ {sym}: no opening price - buy skipped")
                continue
            fill = cost_mod.fill_price(px, "buy", c)
            qty = int((tgt - cur) // fill)
            if qty <= 0:
                continue
            gross = qty * fill
            fee = cost_mod.order_charges(gross, "buy", c)
            if gross + fee > st["cash"]:
                qty = int((st["cash"] * 0.999) // fill)
                if qty <= 0:
                    continue
                gross, fee = qty * fill, cost_mod.order_charges(qty * fill, "buy", c)
            st["cash"] -= gross + fee
            ent = next(e for e in w.index if info.loc[e, "symbol"] == sym)
            p = st["positions"].setdefault(sym, {"entity": ent, "qty": 0, "cost": 0.0, "since": str(today),
                                                 "last_px": px, "sector": str(info.loc[ent, "industry"])})
            p["qty"] += qty
            p["cost"] += gross + fee
            p["last_px"] = px
            _log_trade(today, "BUY", sym, qty, fill, -(gross + fee), 0.0)
            buys.append(f"{sym} {qty} @ {fill:.2f} ({wt * 100:.1f}%)")
        st["sessions_since_rebalance"] = 0
        st["last_rebalance"] = str(today)
        st["rebalance_pending"] = bool(target) and not buys and not st["positions"]
        lines = [f"📈 US V5.0 SWING PAPER - REBALANCE {today} (signal: {last.date()} close; fills at today's "
                 f"open + {c['slippage_bps']} bps slippage + SEC/FINRA fees)"] \
            + ([f"BUY ({len(buys)}): " + "; ".join(buys)] if buys else []) \
            + ([f"SELL ({len(sells)}): " + "; ".join(sells)] if sells else []) \
            + ([f"HOLD ({len(holds)}): " + ", ".join(sorted(holds))] if holds else []) + lines
        prices_now = {sym: (opens.get(sym) or p["last_px"]) for sym, p in st["positions"].items()}
        eq_now = equity_at(st, prices_now)
    else:
        eq_now, prices_now = eq_prev, prices_prev
        lines = [f"📊 US V5.0 SWING PAPER {today} (marked at the {last.date()} close) - next rebalance in "
                 f"{cfg['portfolio']['rebalance_days'] - st['sessions_since_rebalance']} session(s)"]
    tot = eq_now / st["start_equity"] - 1
    sp = spy_last / st["spy_start"] - 1
    invested = eq_now - st["cash"]
    lines.append(f"Portfolio ${eq_now:,.0f} ({len(st['positions'])} stocks, {invested / eq_now * 100:.0f}% invested) | "
                 f"since {st['start']}: {tot * 100:+.2f}% vs SPY {sp * 100:+.2f}% | paper only")
    st["last_day"] = str(today)
    daily = st.setdefault("daily", [])
    if daily and daily[-1].get("date") == str(today):
        daily.pop()
    daily.append({"date": str(today), "equity": round(eq_now, 2), "spy": spy_last})
    save_state(st)
    notify("\n".join(lines), telegram, holdings_table(st, prices_now, eq_now))
    return True


def _log_trade(day, side, sym, qty, px, cash_flow, pnl):
    f = _path("trades.csv")
    new = not os.path.exists(f)
    with open(f, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["date", "side", "symbol", "qty", "price", "cash_flow", "realised_pnl"])
        w.writerow([day, side, sym, qty, round(px, 2), round(cash_flow, 2), round(pnl, 2)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-telegram", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        print(json.dumps(load_state(), indent=1, default=str)[:4000])
        return
    changed = False
    try:
        changed = run_day(not a.no_telegram)
    finally:
        _set_output(changed)


if __name__ == "__main__":
    main()
