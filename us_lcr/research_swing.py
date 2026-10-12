"""
US LCR intraday + swing research: how many days should a trade be carried?

    python -m us_lcr.research_swing

Same entries as live LCR (pullback, price above the 10- and 20-day EMAs,
structural stop capped at 3%, next-bar open + 5 bps). Exits compared on the
SAME trades (only trades with 5 later sessions of data):
  intraday (live)  today's live exit: breakeven after +1R, trail under bar
                   lows, max 180 min, flat 15:55
  hold N (N=0..5)  keep the original stop; checked on every 5-min bar of the
                   entry day and every later daily bar (a gap below it exits at
                   the open); otherwise sell at the close of the N-th session
                   after the entry day (N=0 = entry-day close)
Net of 5 bps slippage per side. Judged on both halves of the trade dates.
"""
from __future__ import annotations

import logging
import math
import os

import numpy as np
import pandas as pd

from us_lcr import research as R
from us_lcr import strategy as S

log = logging.getLogger("us_lcr")
MAX_DAYS = 5


def fresh_daily(symbols: list[str]) -> pd.DataFrame:
    from us_scr.data import daily_bars
    d = daily_bars(sorted(symbols), period="6mo")
    d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None).dt.normalize()
    return d.sort_values(["symbol", "date"])


def trades(days):
    S.SETUP_MODE, S.EMA_MODE, S.STOP_PCT = "pullback", "min", 0.03
    out = []
    for d in days:
        x = d["x"]
        f = S.bar_features(x, d["prev_close"], d["avg_vol20"])
        busy, n = None, 0
        for ts in S.setups(x, f, d["emas"], "pullback", "min"):
            if n >= S.MAX_PER_SYMBOL_DAY or (busy is not None and ts < busy):
                continue
            tr = S.simulate(d["symbol"], x, ts)
            if tr is None:
                continue
            n, busy = n + 1, tr.exit_ts
            i = x.index.get_loc(ts)
            stop = S.initial_stop(x, i, tr.entry)
            # entry day with the ORIGINAL stop, held to the close
            px, why = None, None
            for j in range(i + 1, len(x)):
                b = x.iloc[j]
                if j > i + 1 and float(b["open"]) <= stop:
                    px, why = float(b["open"]), "STOP_GAP"
                    break
                if float(b["low"]) <= stop:
                    px, why = stop, "STOP"
                    break
            day_close = float(x[x.index.time <= S.FLAT].iloc[-1]["close"])
            out.append({"symbol": d["symbol"], "date": d["date"], "entry": tr.entry, "stop": stop,
                        "intraday_ret": tr.ret_pct, "d0_px": px, "d0_why": why, "d0_close": day_close})
    return pd.DataFrame(out)


def hold_returns(t: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    by = {s: g.set_index("date") for s, g in daily.groupby("symbol")}
    rows = []
    for r in t.itertuples():
        g = by.get(r.symbol)
        if g is None:
            continue
        nxt = g[g.index > r.date].head(MAX_DAYS)
        if len(nxt) < MAX_DAYS:
            continue                                  # not enough later sessions yet
        rec = {"symbol": r.symbol, "date": r.date, "intraday": r.intraday_ret}
        exit_px = None if pd.isna(r.d0_px) else float(r.d0_px)   # None = still open after the entry day
        for n in range(0, MAX_DAYS + 1):
            if exit_px is None and n > 0:
                b = nxt.iloc[n - 1]
                if b["open"] <= r.stop:
                    exit_px = float(b["open"])
                elif b["low"] <= r.stop:
                    exit_px = r.stop
            px = exit_px if exit_px is not None else (r.d0_close if n == 0 else float(nxt.iloc[n - 1]["close"]))
            rec[f"hold{n}"] = (px * (1 - S.SLIP) / r.entry - 1) * 100
        rec["risk_pct"] = (1 - r.stop / r.entry) * 100
        rows.append(rec)
    return pd.DataFrame(rows)


def stats(h: pd.DataFrame, col: str) -> dict:
    day = h.groupby("date")[col].mean()
    return {"avg%": round(h[col].mean(), 3), "win%": round((h[col] > 0).mean() * 100, 1),
            "avgR": round((h[col] / h["risk_pct"]).mean(), 3),
            "day_t": round(day.mean() / day.std() * math.sqrt(len(day)), 2) if len(day) > 2 and day.std() > 0 else np.nan}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    days = R.load_days()
    t = trades(days)
    log.info("live-rule LCR trades: %d on %d days", len(t), t["date"].nunique())
    daily = fresh_daily(t["symbol"].unique().tolist())
    h = hold_returns(t, daily)
    dates = sorted(h["date"].unique())
    half = dates[len(dates) // 2]
    log.info("trades with %d later sessions: %d (%s .. %s), halves split %s", MAX_DAYS, len(h),
             pd.Timestamp(dates[0]).date(), pd.Timestamp(dates[-1]).date(), pd.Timestamp(half).date())
    rows = []
    for col in ["intraday"] + [f"hold{n}" for n in range(MAX_DAYS + 1)]:
        rows.append({"exit": col, **{f"all_{k}": v for k, v in stats(h, col).items()},
                     **{f"A_{k}": v for k, v in stats(h[h["date"] < half], col).items()},
                     **{f"B_{k}": v for k, v in stats(h[h["date"] >= half], col).items()}})
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).to_string(index=False))
    h.to_parquet(os.path.join(R.CACHE, "swing_hold_returns.parquet"), index=False)


if __name__ == "__main__":
    main()
