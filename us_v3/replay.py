"""
US V3 replay: re-run past live days through the research engine and compare
with what the live bot actually did - is a bad day the MODEL or the EXECUTION?

    python -m us_v3.replay --model <model.joblib> --trades <trades.csv>
    python -m us_v3.replay --model ... --trades ... --days 2026-10-01,2026-10-02

Use the model the live bot used that day (the `us-v3-*` artifact) and its
trades.csv. Refresh the store first (us_v3.data.refresh_intraday) so it has
the days being replayed.

For each day, three views of the same ORB_ATR breakouts:
  RESEARCH  what the backtest books: entry AT the breakout level on the
            breakout bar, stop 10% ATR from that level, entry-bar stop only
            on a close through it (research._sim). Picks = the top-cap
            breakouts by score >= threshold (the model's own ranking).
  LIVE-SIM  live's rules on the same bars: candidates in the order live sees
            them (breakout bar closes, then watchlist order), correlation
            filter, daily cap; entry at the OPEN of the bar after the
            breakout bar (live's first chance), stop 10% ATR from that
            price, stop checks from the bar after entry (live.check_exits).
  LIVE      trades.csv as booked.
RESEARCH vs LIVE-SIM = cost of executing one bar late at the live price.
LIVE-SIM vs LIVE     = timing / price noise of the 2-minute trigger.
Same score in all three; the `pred` live logged is checked against the replay
score (a mismatch means live saw different data / features).
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, time as dtime

import joblib
import numpy as np
import pandas as pd

from us_v3 import data as D
from us_v3 import live as L
from us_v3 import research as R

OUT_DIR = os.path.join("backtest", "results", "us_v3")


def candidates(day: pd.Timestamp, syms: list[str], ctx: dict, bundle: dict) -> pd.DataFrame:
    """Every ORB_ATR breakout on `day` (full-day bars), scored like live."""
    rows = []
    for k, sym in enumerate(syms):
        t = R.symbol_days(sym, ctx, last_days=25)
        if t.empty:
            continue
        t = t[(t["date"] == day) & (t["setup"] == L.SETUP)]
        if t.empty:
            continue
        row = t.iloc[0].copy()
        row["is_late"], row["stop_range"] = 0, 0           # exactly as live.scan_breakouts
        row["pred"] = float(bundle["model"].predict(pd.DataFrame([row[bundle["features"]].astype(float)]))[0])
        row["wl_order"] = k
        rows.append(row)
    return pd.DataFrame(rows)


def _rth(sym: str, day: pd.Timestamp) -> pd.DataFrame:
    x = D.load(sym)
    x = x[x.index.date == day.date()]
    return x[(x.index.time >= R.OPEN) & (x.index.time < R.CLOSE)]


def live_sim(c: pd.Series, day: pd.Timestamp) -> dict | None:
    """Live's execution on one breakout: enter at the next bar's open, stop
    10% ATR from there, stops checked from the bar after entry."""
    rth = _rth(c["symbol"], day)
    i = int(c["mins"]) // 5 + 1                          # bar after the breakout bar
    if i >= len(rth):
        return None
    side, a = int(c["side"]), float(c["atr_d"])
    entry = float(rth["open"].iloc[i])
    stop = entry - side * 0.10 * a
    px, xts, outcome = None, None, None
    for ts, b in rth.iloc[i + 1:].iterrows():
        if (side > 0 and b.low <= stop) or (side < 0 and b.high >= stop):
            px, xts, outcome = (min(stop, b.open) if side > 0 else max(stop, b.open)), ts, "STOP"
            break
        if ts.time() >= dtime(15, 55):
            px, xts, outcome = float(b.close), ts, "EOD"
            break
    if px is None:
        px, xts, outcome = float(rth["close"].iloc[-1]), rth.index[-1], "EOD"
    risk = abs(entry - stop)
    r = (side * (px - entry) - R.SLIP * (entry + px)) / risk
    return {"sim_entry_ts": str(rth.index[i]), "sim_entry": entry, "sim_stop": stop, "sim_exit": px,
            "sim_exit_ts": str(xts), "sim_outcome": outcome, "sim_R": r}


def replay_day(day: pd.Timestamp, syms: list[str], ctx: dict, bundle: dict, cap: int,
               live: pd.DataFrame) -> dict:
    thr = bundle["info"]["threshold"]
    c = candidates(day, syms, ctx, bundle)
    if c.empty:
        return {"day": day, "cands": c}
    passed = c[c["pred"] >= thr]

    # RESEARCH: the model's top `cap` by score
    research = passed.sort_values("pred", ascending=False).head(cap)

    # LIVE-SIM: live's order (breakout time, then watchlist), corr filter, cap
    at = pd.Timestamp.combine(day.date(), dtime(10, 30)).tz_localize(D.ET)
    taken, sims = [], []
    for _, r in passed.sort_values(["mins", "wl_order"]).iterrows():
        if len(taken) >= cap:
            break
        if L._too_correlated(r["symbol"], taken, at):
            continue
        s = live_sim(r, day)
        if s is None:
            continue
        taken.append(r["symbol"])
        sims.append({**r.to_dict(), **s})
    sim = pd.DataFrame(sims)

    # LIVE trades vs the replay of the same symbols
    lv = live[live["date"] == str(day.date())].copy()
    if len(lv):
        m = c.set_index("symbol")
        lv["replay_pred"] = lv["symbol"].map(m["pred"])
        lv["research_entry"] = lv["symbol"].map(m["entry"])
        lv["research_R"] = lv["symbol"].map(m["R"])
        lv["research_outcome"] = lv["symbol"].map(m["outcome"])
        lv["breakout_bar"] = lv["symbol"].map(m["mins"]).map(
            lambda v: f"{9 + (30 + int(v)) // 60}:{(30 + int(v)) % 60:02d}" if v == v else "")
        risk = (lv["symbol"].map(m["entry"]) - lv["symbol"].map(m["stop"])).abs()
        # live entry vs the breakout level, in research R (+ = worse price than research)
        lv["entry_gap_R"] = lv["side"] * (lv["entry"] - lv["research_entry"]) / risk
    return {"day": day, "cands": c, "passed": passed, "research": research, "sim": sim, "live": lv, "thr": thr}


def _fmt_day(res: dict) -> str:
    d, out = res["day"].date(), []
    c = res["cands"]
    if c.empty:
        return f"\n=== {d}: no ORB breakouts in the store (refresh it?)"
    out.append(f"\n=== {d}: {len(c)} ORB breakouts, {len(res['passed'])} scored >= threshold {res['thr']:+.3f}")
    r, s, lv = res["research"], res["sim"], res["live"]
    out.append(f"RESEARCH  {len(r):>2} trades  sum {r['R'].sum():+6.2f}R  avg {r['R'].mean() if len(r) else 0:+.2f}R  "
               f"win {(r['R'] > 0).mean() * 100 if len(r) else 0:.0f}%  stops {(r['outcome'] == 'STOP').sum()}")
    out.append(f"LIVE-SIM  {len(s):>2} trades  sum {s['sim_R'].sum() if len(s) else 0:+6.2f}R  "
               f"avg {s['sim_R'].mean() if len(s) else 0:+.2f}R  win {(s['sim_R'] > 0).mean() * 100 if len(s) else 0:.0f}%  "
               f"stops {(s['sim_outcome'] == 'STOP').sum() if len(s) else 0}")
    out.append(f"LIVE      {len(lv):>2} trades  sum {lv['R'].sum() if len(lv) else 0:+6.2f}R  "
               f"avg {lv['R'].mean() if len(lv) else 0:+.2f}R  win {(lv['R'] > 0).mean() * 100 if len(lv) else 0:.0f}%")
    if len(r):
        out.append("\n  research picks (top by score):")
        for _, t in r.iterrows():
            out.append(f"   {t['symbol']:<6} {'L' if t['side'] > 0 else 'S'} pred {t['pred']:+.2f}  rvol {t['rvol_open']:.2f}  "
                       f"entry {t['entry']:.2f} @ +{int(t['mins'])}m  -> {t['outcome']:<4} {t['R']:+.2f}R")
    if len(s):
        out.append("\n  live-sim picks (live's order + next-bar entry):")
        for _, t in s.iterrows():
            out.append(f"   {t['symbol']:<6} {'L' if t['side'] > 0 else 'S'} pred {t['pred']:+.2f}  research {t['R']:+.2f}R  "
                       f"| next-bar entry {t['sim_entry']:.2f} -> {t['sim_outcome']:<4} {t['sim_R']:+.2f}R")
    if len(lv):
        out.append("\n  live trades vs replay of the same symbol:")
        for _, t in lv.iterrows():
            out.append(f"   {t['symbol']:<6} {'L' if t['side'] > 0 else 'S'} pred live {t['pred']:+.2f} / replay {t['replay_pred']:+.2f}  "
                       f"breakout {t['breakout_bar']}  level {t['research_entry']:.2f} live entry {t['entry']:.2f} "
                       f"(entry {t['entry_gap_R']:+.2f}R vs level, + = worse)  research {t['research_outcome']} {t['research_R']:+.2f}R | live {t['R']:+.2f}R")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=L.MODEL_PATH, help="model.joblib the live bot used")
    ap.add_argument("--trades", default=os.path.join(L.STATE_DIR, "trades.csv"), help="live trades.csv")
    ap.add_argument("--days", help="comma-separated YYYY-MM-DD (default: every day in trades.csv)")
    ap.add_argument("--cap", type=int, default=L.V3_MAX)
    a = ap.parse_args()

    bundle = joblib.load(a.model)
    live = pd.read_csv(a.trades) if os.path.exists(a.trades) else pd.DataFrame(columns=["date"])
    days = a.days.split(",") if a.days else sorted(live["date"].astype(str).unique())
    syms = L.symbols()
    ctx = R.context()
    print(f"model trained {bundle['info']['trained_at']} on {bundle['info']['days']} days, "
          f"threshold {bundle['info']['threshold']:+.3f}, cap {a.cap}, {len(syms)} symbols")

    results = [replay_day(pd.Timestamp(d), syms, ctx, bundle, a.cap, live) for d in days]
    for res in results:
        print(_fmt_day(res))

    tot = {k: sum(float(res[k][col].sum()) for res in results if k in res and len(res[k]))
           for k, col in (("research", "R"), ("sim", "sim_R"), ("live", "R"))}
    print(f"\nTOTAL  research {tot['research']:+.2f}R | live-sim {tot['sim']:+.2f}R | live {tot['live']:+.2f}R")

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for k in ("research", "sim", "live"):
        parts = [res[k].assign(view=k) for res in results if k in res and len(res[k])]
        if parts:
            pd.concat(parts).to_csv(os.path.join(OUT_DIR, f"replay_{k}_{stamp}.csv"), index=False)
    print(f"CSVs: {OUT_DIR}/replay_*_{stamp}.csv")


if __name__ == "__main__":
    main()
