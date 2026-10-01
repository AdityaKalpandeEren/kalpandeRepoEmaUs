"""
US V3 live paper trading + Telegram alerts (Stocks-in-Play ORB + ML ranker).

    python -m us_v3.live                 # one step - what the `v3` workflow job runs every trigger
    python -m us_v3.live --no-telegram
    python -m us_v3.live --retrain       # force a model retrain now

Each trigger (cron-job.org, every 2 min) does whatever is due, idempotently:
  09:00-09:34 ET  daily prep (once): refresh daily bars + earnings dates;
                  retrain the model if it is older than 7 days (weekly
                  learning on the growing 5-min store), recalibrate the
                  entry threshold from walk-forward predictions
  09:35-10:30 ET  refresh today's 5-min bars; find NEW opening-range
                  breakouts (first 5-min bar green -> long above its high,
                  red -> short below its low); score each with the model;
                  paper-enter if score >= threshold, at most V3_MAX (12)
                  per day at DAILY_RISK/V3_MAX (~0.42%) risk each, one entry per symbol per day, at the LIVE price
                  (stop = 10% of daily ATR from that price)
  until 15:55 ET  stop checks on every closed 5-min bar since entry
  15:55-16:10 ET  close everything at the 15:55 bar close; Telegram day
                  report + all-time stats (once)
Guards: AUTO-THROTTLE - if the last 10 live trading days are net negative,
only 5 trades/day are allowed until that rolling sum turns positive (Telegram
says so). DRIFT ALERT - if the last 30 closed trades fall well short of the
R the model predicted for them (t < -2), the day report warns that the market
no longer matches what the model learned.
Paper only - nothing is sent to a broker. Shorts are paper-only too
(V3_DIRECTIONS=long for long-only).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime, time as dtime

import numpy as np
import pandas as pd

from us_v3 import data as D
from us_v3 import research as R

ET = D.ET
STATE_DIR = os.environ.get("US_V3_STATE_DIR", os.path.join("live_state", "us_v3"))
MODEL_PATH = os.path.join(STATE_DIR, "model.joblib")
V3_MAX = int(os.environ.get("V3_MAX_TRADES_PER_DAY", "12"))
# Total risk per day stays ~5% of equity however many trades are allowed:
# 12 trades/day -> ~0.42% risk each (research: 20/day at 1% each had a -56% drawdown).
DAILY_RISK = float(os.environ.get("V3_DAILY_RISK", "0.05"))
RISK_PER_TRADE = DAILY_RISK / V3_MAX
V3_DIRECTIONS = {d.strip() for d in os.environ.get("V3_DIRECTIONS", "long,short").split(",") if d.strip()}
SETUP = "ORB_ATR"            # the configuration selected in research (ML top-5, 10% ATR stop)
RETRAIN_DAYS = 7


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


def symbols() -> list[str]:
    from backtest.run_backtest import load_symbol_list
    return list(dict.fromkeys(D.tradable(load_symbol_list("watchlist.txt"))))


# ─── model (weekly learning) ─────────────────────────────────────────────

def train(syms: list[str]) -> dict:
    import joblib
    import lightgbm as lgb
    df = R.build(syms)                                   # every complete day in the growing store
    df["pred"] = R.walk_forward(df)                      # out-of-sample scores -> threshold calibration
    oos = df[(df["setup"] == SETUP) & df["pred"].notna()]
    # threshold = typical V3_MAX-th best out-of-sample score of a day (so ~V3_MAX entries/day)
    kth = oos.groupby("date")["pred"].apply(lambda s: s.nlargest(V3_MAX).min() if len(s) >= V3_MAX else s.min())
    thr = float(max(np.nanmedian(kth), 0.0))
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, random_state=42, verbose=-1)
    m.fit(df[R.FEATURES], df["R"].clip(-3, 5))
    top = oos.sort_values(["date", "pred"], ascending=[True, False]).groupby("date").head(V3_MAX)
    top = top[top["pred"] > 0]
    info = {"trained_at": datetime.now(ET).isoformat(timespec="minutes"), "days": int(df["date"].nunique()),
            "rows": int(len(df)), "threshold": thr, "oos_top_avgR": round(float(top["R"].mean()), 3) if len(top) else None,
            "oos_days": int(oos["date"].nunique()), "max_per_day": V3_MAX}
    joblib.dump({"model": m, "features": R.FEATURES, "info": info}, MODEL_PATH)
    return info


def load_model():
    import joblib
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


# ─── trading steps ───────────────────────────────────────────────────────

def _log(row: dict) -> None:
    f = _p("trades.csv")
    new = not os.path.exists(f)
    with open(f, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


MAX_CORR = float(os.environ.get("V3_MAX_CORR", "0.7"))

# Auto-throttle: if the last THROTTLE_LOOKBACK traded days are net negative,
# allow only THROTTLE_MAX trades/day until the rolling sum turns positive.
THROTTLE_LOOKBACK = 10
THROTTLE_MIN_DAYS = 5
THROTTLE_MAX = 5
# Drift alert: live results vs what the model expected for the same trades.
DRIFT_LOOKBACK = 30          # most recent closed trades
DRIFT_MIN_TRADES = 15
DRIFT_T = -2.0               # t-stat of (realised R - predicted R) below this -> alert


def live_trades() -> pd.DataFrame:
    f = _p("trades.csv")
    return pd.read_csv(f) if os.path.exists(f) else pd.DataFrame()


def daily_cap() -> tuple[int, str]:
    """Today's max trades and a one-line reason."""
    t = live_trades()
    if t.empty:
        return V3_MAX, "normal (no live history yet)"
    days = t.groupby("date")["R"].sum().sort_index().tail(THROTTLE_LOOKBACK)
    if len(days) < THROTTLE_MIN_DAYS:
        return V3_MAX, f"normal ({len(days)} live days, throttle needs {THROTTLE_MIN_DAYS})"
    tot = float(days.sum())
    if tot < 0:
        return min(THROTTLE_MAX, V3_MAX), (f"THROTTLED to {min(THROTTLE_MAX, V3_MAX)}/day: last {len(days)} live days "
                                           f"{tot:+.1f}R (resumes {V3_MAX}/day when that turns positive)")
    return V3_MAX, f"normal: last {len(days)} live days {tot:+.1f}R"


def drift_check() -> str | None:
    """Alert text if recent live trades fall well short of the model's own
    expectation (predicted R) - a sign the market no longer matches what the
    model learned. None if fine or too few trades."""
    t = live_trades()
    if t.empty or "pred" not in t:
        return None
    t = t.dropna(subset=["R", "pred"]).tail(DRIFT_LOOKBACK)
    if len(t) < DRIFT_MIN_TRADES:
        return None
    gap = t["R"] - t["pred"]
    sd = gap.std(ddof=1)
    tstat = gap.mean() / (sd / np.sqrt(len(gap))) if sd > 0 else 0.0
    if tstat < DRIFT_T:
        return (f"⚠️ US V3 DRIFT: last {len(t)} trades averaged {t['R'].mean():+.2f}R vs {t['pred'].mean():+.2f}R "
                f"expected by the model (t {tstat:.1f}). The market may have changed since training - results are "
                f"below what the model learned. Next retrain uses the new data; consider pausing if it persists.")
    return None


def _price_now(sym: str, now):
    """Latest traded price at `now` (the forming bar's last close), never later."""
    x = D.load(sym)
    x = x[x.index <= now] if len(x) else x
    return (float(x["close"].iloc[-1]), x.index[-1]) if len(x) else (None, None)


def _daily_returns(sym: str, now, n: int = 20) -> pd.Series:
    d = D.load(sym, "1d")
    if d.empty:
        return pd.Series(dtype=float)
    return d[d.index < pd.Timestamp(now.date())]["close"].pct_change().tail(n)


def _too_correlated(sym: str, held: list[str], now) -> str | None:
    """Return the held symbol `sym` moves with (20-day daily-return corr > MAX_CORR):
    four crypto-miner shorts at once are one bet, not four (V2/J clustering lesson)."""
    a = _daily_returns(sym, now)
    for h in held:
        b = _daily_returns(h, now)
        j = pd.concat([a, b], axis=1).dropna()
        if len(j) >= 10 and j.iloc[:, 0].corr(j.iloc[:, 1]) > MAX_CORR:
            return h
    return None


def scan_breakouts(st, now, syms, bundle, telegram):
    ctx = R.context(now=now)
    today = pd.Timestamp(now.date())
    new_lines = []
    taken = st.setdefault("taken", [])
    seen = st.setdefault("seen", [])
    for sym in syms:
        if sym in seen:
            continue
        t = R.symbol_days(sym, ctx, now=now, last_days=20)
        if t.empty:
            continue
        t = t[(t["date"] == today) & (t["setup"] == SETUP)]
        if t.empty:
            continue
        seen.append(sym)                                  # one decision per symbol per day
        row = t.iloc[0].copy()
        row["is_late"], row["stop_range"] = 0, 0
        side = int(row["side"])
        if (side > 0 and "long" not in V3_DIRECTIONS) or (side < 0 and "short" not in V3_DIRECTIONS):
            continue
        X = pd.DataFrame([row[bundle["features"]].astype(float)])
        pred = float(bundle["model"].predict(X)[0])
        thr = bundle["info"]["threshold"]
        if pred < thr or len(taken) >= st.get("cap", V3_MAX):
            continue
        twin = _too_correlated(sym, taken, now)
        if twin:
            print(f"[US V3] skip {sym}: moves with {twin} (corr > {MAX_CORR})", flush=True)
            continue
        px, pts = _price_now(sym, now)
        if px is None:
            continue
        stop = px - side * 0.10 * float(row["atr_d"])
        pos = {"symbol": sym, "side": side, "entry": px, "entry_ts": str(pts), "stop": round(stop, 4),
               "level": float(row["entry"]), "pred": round(pred, 3), "rvol": round(float(row["rvol_open"]), 2),
               "gap": round(float(row["gap"]), 4), "earnings": bool(row["earnings_overnight"]), "status": "OPEN"}
        taken.append(sym)
        st.setdefault("positions", []).append(pos)
        word = "BUY" if side > 0 else "SHORT"
        new_lines.append(f"{'🚀' if side > 0 else '🔻'} {word} {sym} @ {px:.2f}  stop {stop:.2f}  "
                         f"(breakout {row['entry']:.2f}, RVOL {row['rvol_open']:.1f}x, gap {row['gap'] * 100:+.1f}%"
                         f"{', EARNINGS' if row['earnings_overnight'] else ''}, score {pred:+.2f})")
    if new_lines:
        notify(f"🧠 US V3 PAPER (Stocks-in-Play ORB) {today.date()} - {len(taken)}/{st.get('cap', V3_MAX)} today\n"
               + "\n".join(new_lines) + "\nExit: stop or 15:55 ET. Paper only.", telegram)
        return True
    return False


def check_exits(st, now, telegram, force_close=False) -> bool:
    lines = []
    for p in st.get("positions", []):
        if p["status"] != "OPEN":
            continue
        x = R._closed(D.load(p["symbol"]), now)
        after = x[x.index > pd.Timestamp(p["entry_ts"])]
        after = after[after.index.time < dtime(16, 0)]
        side, stop = p["side"], p["stop"]
        exit_px = None
        for ts, b in after.iterrows():
            if (side > 0 and b.low <= stop) or (side < 0 and b.high >= stop):
                exit_px = min(stop, b.open) if side > 0 else max(stop, b.open)
                p.update(status="CLOSED", exit=exit_px, exit_ts=str(ts), outcome="STOP")
                break
            if ts.time() >= dtime(15, 55):
                exit_px = float(b.close)
                p.update(status="CLOSED", exit=exit_px, exit_ts=str(ts), outcome="EOD")
                break
        if exit_px is None and force_close and len(after):
            exit_px = float(after["close"].iloc[-1])
            p.update(status="CLOSED", exit=exit_px, exit_ts=str(after.index[-1]), outcome="EOD")
        if p["status"] == "CLOSED":
            risk = abs(p["entry"] - stop)
            p["R"] = round((side * (p["exit"] - p["entry"]) - R.SLIP * (p["entry"] + p["exit"])) / risk, 3)
            p["ret_pct"] = round(100 * (side * (p["exit"] / p["entry"] - 1) - 2 * R.SLIP), 3)
            _log({"date": now.date(), **{k: p[k] for k in ("symbol", "side", "entry", "exit", "stop", "outcome", "R",
                                                           "ret_pct", "pred", "rvol", "entry_ts", "exit_ts")}})
            if p["outcome"] == "STOP":
                lines.append(f"🛑 {p['symbol']} stopped @ {p['exit']:.2f} ({p['R']:+.2f}R)")
    if lines:
        notify("US V3 PAPER\n" + "\n".join(lines), telegram)
    return bool(lines)


def day_report(st, now, telegram):
    ps = st.get("positions", [])
    lines = [f"📊 US V3 PAPER RESULT {now.date()} (Stocks-in-Play ORB + ML)"]
    if not ps:
        lines.append("No trades today (no breakout scored above the threshold).")
    for p in ps:
        mark = "✅" if p.get("R", 0) > 0 else "❌"
        lines.append(f"{mark} {'LONG' if p['side'] > 0 else 'SHORT'} {p['symbol']}: {p['entry']:.2f} -> {p.get('exit', float('nan')):.2f} "
                     f"({p.get('outcome', '?')}) {p.get('R', 0):+.2f}R, {p.get('ret_pct', 0):+.2f}%")
    if ps:
        day_r = sum(p.get('R', 0) for p in ps)
        lines.append(f"Day: {day_r:+.2f}R over {len(ps)} trades ({RISK_PER_TRADE * 100:.2f}% risk/trade -> "
                     f"{day_r * RISK_PER_TRADE * 100:+.2f}% of equity)")
    f = _p("trades.csv")
    if os.path.exists(f):
        t = pd.read_csv(f)
        dd = t.groupby("date")["R"].sum()
        lines.append(f"ALL-TIME ({len(dd)} days, {len(t)} trades): {t['R'].sum():+.2f}R | win {100 * (t['R'] > 0).mean():.0f}% | "
                     f"avg {t['R'].mean():+.3f}R | positive days {(dd > 0).sum()}/{len(dd)}  (need ~40+ days before judging)")
    m = load_model()
    if m:
        lines.append(f"Model: trained {m['info']['trained_at']} on {m['info']['days']} days; threshold {m['info']['threshold']:+.2f}")
    lines.append(f"Daily limit today: {st.get('cap', V3_MAX)} ({st.get('cap_reason', 'normal')})")
    nxt, why = daily_cap()
    if nxt != st.get("cap", V3_MAX):
        lines.append(("🐢 Tomorrow: " if nxt < V3_MAX else "✅ Tomorrow: back to ") + f"{nxt}/day - {why}")
    drift = drift_check()
    if drift:
        lines.append(drift)
    notify("\n".join(lines), telegram)


def run(telegram: bool, force_retrain: bool = False) -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5 and not force_retrain:
        print("Weekend - nothing to do.")
        return False
    st = load_state()
    today = str(now.date())
    if st.get("date") != today:
        st = {"date": today}
    mins = now.hour * 60 + now.minute
    syms = symbols()
    changed = False

    if not st.get("prepped") and (mins >= 9 * 60 or force_retrain):
        t0 = time.time()
        D.refresh_intraday(syms + D.CONTEXT, period="60d" if not os.path.exists(D._path("SPY")) else "5d")
        D.refresh_daily(syms + D.CONTEXT)
        D.refresh_earnings(syms)
        m = load_model()
        stale = (m is None or (datetime.now(ET) - datetime.fromisoformat(m["info"]["trained_at"])).days >= RETRAIN_DAYS
                 or m["info"].get("max_per_day") != V3_MAX)          # threshold depends on the daily limit
        if stale or force_retrain:
            info = train(syms)
            print(f"[US V3] retrained: {info}", flush=True)
            notify(f"🧠 US V3 model retrained ({info['days']} days of data, {info['rows']} candidates); entry threshold "
                   f"{info['threshold']:+.2f}; out-of-sample top picks avg {info['oos_top_avgR']}R", telegram)
        st["prepped"] = True
        changed = True
        print(f"[US V3] prep done in {time.time() - t0:.0f} s", flush=True)
        if force_retrain:
            save_state(st)
            return True

    if mins < 9 * 60 + 35 or st.get("reported"):
        save_state(st) if changed else None
        return changed

    D.refresh_intraday(syms + D.CONTEXT, period="1d")
    bundle = load_model()
    if bundle is not None and bundle["info"].get("max_per_day") != V3_MAX:
        info = train(syms)                                 # daily limit changed -> recalibrate now, not next week
        notify(f"🧠 US V3 model recalibrated for up to {V3_MAX} trades/day; threshold {info['threshold']:+.2f}", telegram)
        bundle, changed = load_model(), True
    if bundle is None:
        print("[US V3] no model yet")
        return changed
    if "cap" not in st:
        st["cap"], why = daily_cap()
        st["cap_reason"] = why
        changed = True
        if st["cap"] < V3_MAX:
            notify(f"🐢 US V3 auto-throttle: {why}", telegram)
    if mins <= 10 * 60 + 30:
        changed |= scan_breakouts(st, now, syms, bundle, telegram)
    changed |= check_exits(st, now, telegram)
    if mins >= 16 * 60:                                   # the 15:55 bar has closed
        check_exits(st, now, telegram, force_close=True)
        day_report(st, now, telegram)
        st["reported"] = True
        changed = True
    if changed:
        save_state(st)
    return changed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-telegram", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    a = ap.parse_args()
    changed = False
    try:
        changed = run(not a.no_telegram, a.retrain)
    finally:
        out = os.environ.get("GITHUB_OUTPUT")
        if out:
            with open(out, "a") as f:
                f.write(f"changed={'true' if changed else 'false'}\n")


if __name__ == "__main__":
    main()
