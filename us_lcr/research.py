"""
US LCR research: every setup on every mid / large / mega-cap volume-shocker
day (daily: volume >= 2x the 20-day average and intraday high >= +2%,
last ~58 days, Yahoo 5-min bars), live rules, 5 bps slippage per side.

    python -m us_lcr.research

No lookahead: the live trigger needs >= 2x the average DAILY volume by the
signal bar and >= +2%, so it can only fire on days that end that way. The
daily EMAs use closes up to the PREVIOUS session. Variants (setup x EMA
filter x stop cap) are judged on the first half of the days and checked on
the second half.
"""
from __future__ import annotations

import itertools
import logging
import math
import os

import numpy as np
import pandas as pd

from us_lcr import strategy as S

log = logging.getLogger("us_lcr")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "us_lcr", "cache")


def load_days():
    daily = pd.read_parquet(os.path.join(CACHE, "daily.parquet")).sort_values(["symbol", "date"])
    ev = pd.read_parquet(os.path.join(CACHE, "events.parquet"))
    uni = pd.read_parquet(os.path.join(CACHE, "universe.parquet")).set_index("symbol")
    out = []
    for sym, g in ev.groupby("symbol"):
        f = os.path.join(CACHE, "bars", f"{sym}.parquet")
        if not os.path.exists(f):
            continue
        bars = pd.read_parquet(f)
        dsym = daily[daily.symbol == sym].set_index("date")
        for day in g.date:
            hist = dsym[dsym.index < day]
            if len(hist) < 30:
                continue
            x = bars[bars.index.date == day.date()]
            if len(x) < 40:
                continue
            out.append({"symbol": sym, "date": day, "x": x, "prev_close": float(hist.close.iloc[-1]),
                        "avg_vol20": float(hist.volume.tail(20).mean()), "emas": S.daily_emas(hist.close),
                        "bucket": uni.loc[sym, "bucket"] if sym in uni.index else "?"})
    return out


def run(days, mode, ema_mode, stop_pct):
    S.STOP_PCT = stop_pct
    rows = []
    for d in days:
        x = d["x"]
        f = S.bar_features(x, d["prev_close"], d["avg_vol20"])
        busy, n = None, 0
        for ts in S.setups(x, f, d["emas"], mode, ema_mode):
            if n >= S.MAX_PER_SYMBOL_DAY or (busy is not None and ts < busy):
                continue
            tr = S.simulate(d["symbol"], x, ts, mode=mode)
            if tr is None:
                continue
            n += 1
            busy = tr.exit_ts
            rows.append({"symbol": d["symbol"], "date": d["date"], "bucket": d["bucket"], "entry_ts": tr.entry_ts,
                         "R": tr.R, "ret_pct": tr.ret_pct, "outcome": tr.outcome, "hour": tr.entry_ts.hour,
                         "pct": float(f.loc[ts, "pct"]), "rvol": float(f.loc[ts, "rvol"])})
    return pd.DataFrame(rows)


def stats(t):
    if t.empty:
        return {"n": 0}
    day = t.groupby("date")["ret_pct"].mean()
    return {"n": len(t), "win%": round((t.R > 0).mean() * 100, 1), "avg_R": round(t.R.mean(), 3),
            "net%": round(t.ret_pct.mean(), 3),
            "day_t": round(day.mean() / day.std() * math.sqrt(len(day)), 2) if len(day) > 2 and day.std() > 0 else np.nan}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    days = load_days()
    alld = sorted({d["date"] for d in days})
    half = alld[len(alld) // 2]
    log.info("shocker stock-days: %d (%s .. %s), split %s", len(days), alld[0].date(), alld[-1].date(), half.date())
    rows = []
    for mode, em, sp in itertools.product(("pullback", "hod"), ("none", "above", "stacked"), (0.02, 0.03)):
        t = run(days, mode, em, sp)
        a, b = t[t.date < half], t[t.date >= half]
        rows.append({"setup": mode, "ema": em, "stop": sp, **{f"A_{k}": v for k, v in stats(a).items()},
                     **{f"B_{k}": v for k, v in stats(b).items()}})
        t.to_parquet(os.path.join(CACHE, f"trades_{mode}_{em}_{int(sp * 100)}.parquet"), index=False)
    r = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    log.info("RESULTS (net of 5 bps slippage per side):\n%s", r.to_string(index=False))
    r.to_csv(os.path.join(CACHE, "grid.csv"), index=False)


if __name__ == "__main__":
    main()
