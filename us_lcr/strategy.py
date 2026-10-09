"""
US LCR (Large-Cap Runners) - mid / large / mega caps (market cap >= $2B)
with unusual volume. Same mechanics as US SCR (us_scr/strategy.py), tuned
for liquid large caps, plus a DAILY EMA trend filter (10 / 30 / 40 / 60 /
180-day EMAs of the close, as of the PREVIOUS session - known before the open).

Everything on bar t uses data up to the CLOSE of bar t only.

WATCH  TIME-ADJUSTED volume (by bar t) >= RVOL_MIN x what a normal day has
       traded by that time (a volume shocker; see VOL_CURVE), up >= WATCH_PCT vs the previous close,
       dollar volume today >= MIN_DOLLAR_VOL, above VWAP.
TREND  EMA_MODE: "none" | "above10" (price above the 10-day EMA only) |
       "above" (price above all five daily EMAs) |
       "stacked" (EMA10 > EMA30 > EMA40 > EMA60 > EMA180 and price above EMA10).
SETUP  SETUP_MODE: "pullback" (recent high of day, pullback on lighter volume,
       reclaim of the previous bar's high) | "hod" (new high of day on
       >= VOL_SURGE x bar volume).
RESEARCH (2026-10-08, us_lcr/research.py; 42 sessions, 10,829 candidate days,
       12 variants, halves A/B): pullback continuation is positive in BOTH
       halves for every EMA setting; the EMA filter lifts it (none +0.15% /
       +0.17% per trade -> above all 5 EMAs +0.26% / +0.29%, win 49% / 55%,
       ~290 trades); HOD breakouts lose. Live = pullback + above + 3% stop.
       Not statistically proven (t 0.3-1.6) - forward paper test.
ENTRY  next 5-min bar's open + SLIP; stop structural, capped at STOP_PCT;
       breakeven after +1R then trail under bar lows; max MAX_HOLD_MIN;
       flat 15:55 ET.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time as dtime

import numpy as np
import pandas as pd

RVOL_MIN = 2.0
WATCH_PCT = 0.02
MIN_DOLLAR_VOL = 20_000_000
VOL_SURGE = 2.0
STOP_PCT = 0.03
SLIP = 0.0005                    # 5 bps per side - liquid large caps
MAX_HOLD_MIN = 180
PB_LOOKBACK, PB_MIN_PCT = 6, 0.01
REG_START, LAST_ENTRY, FLAT = dtime(9, 35), dtime(15, 30), dtime(15, 55)
MAX_PER_SYMBOL_DAY = 2
EMA_SPANS = (10, 30, 40, 60, 180)
SETUP_MODE = "pullback"
EMA_MODE = "above"          # price above all five daily EMAs (chosen 2026-10-08, see below)


@dataclass
class Trade:
    symbol: str
    day: object
    signal_ts: pd.Timestamp
    entry_ts: pd.Timestamp
    entry: float
    stop0: float
    exit_ts: pd.Timestamp | None = None
    exit: float | None = None
    outcome: str = "OPEN"
    R: float = 0.0
    ret_pct: float = 0.0


def daily_emas(closes: pd.Series) -> dict:
    """EMAs of daily closes (pass closes up to the PREVIOUS session only)."""
    return {f"ema{n}": float(closes.ewm(span=n, adjust=False).mean().iloc[-1]) for n in EMA_SPANS}


def ema_ok(price: float, emas: dict, mode: str | None = None) -> bool:
    mode = mode or EMA_MODE
    if mode == "none" or not emas:
        return True
    vals = [emas[f"ema{n}"] for n in EMA_SPANS]
    if mode == "above":
        return all(price > v for v in vals)
    if mode == "above10":                          # only the short-term trend: price above the 10-day EMA
        return price > emas["ema10"]
    if mode == "stacked":
        return all(a > b for a, b in zip(vals, vals[1:])) and price > vals[0]
    raise ValueError(mode)


def ema_tags(price: float, emas: dict) -> str:
    """'✅10 ✅30 ✅40 ❌60 ✅180' - price above each daily EMA (for alerts)."""
    return " ".join(f"{'✅' if price > emas[f'ema{n}'] else '❌'}{n}d" for n in EMA_SPANS)


# Typical share of a US stock's daily volume traded by each time (cumulative,
# regular session; heavier at the open and the close). Pre-market bars count
# toward the first point.
VOL_CURVE = [(9 * 60 + 30, 0.02), (9 * 60 + 45, 0.09), (10 * 60, 0.15), (10 * 60 + 30, 0.25), (11 * 60, 0.33),
             (12 * 60, 0.45), (13 * 60, 0.55), (14 * 60, 0.65), (15 * 60, 0.78), (15 * 60 + 30, 0.87), (16 * 60, 1.0)]


def expected_share(ts) -> float:
    m = ts.hour * 60 + ts.minute + 5                                  # volume known at the bar's close
    xs, ys = zip(*VOL_CURVE)
    return float(np.interp(m, xs, ys))


def bar_features(x: pd.DataFrame, prev_close: float, avg_vol20: float) -> pd.DataFrame:
    f = pd.DataFrame(index=x.index)
    c, h, lo, v = x["close"], x["high"], x["low"], x["volume"].fillna(0)
    f["pct"] = c / prev_close - 1
    f["cum_vol"] = v.cumsum()
    tp = (h + lo + c) / 3
    f["cum_dollar"] = (v * tp).cumsum()
    # TIME-ADJUSTED relative volume: volume so far vs the volume a normal day
    # has traded by this time (U-shaped US intraday profile, VOL_CURVE).
    exp_share = pd.Series([expected_share(ts) for ts in x.index], index=x.index)
    f["rvol"] = f["cum_vol"] / (avg_vol20 * exp_share) if avg_vol20 and avg_vol20 > 0 else np.nan
    f["rvol_day"] = f["cum_vol"] / avg_vol20 if avg_vol20 and avg_vol20 > 0 else np.nan
    f["hod_prior"] = h.cummax().shift()
    f["new_hod"] = c > f["hod_prior"]
    f["vol_ratio"] = v / v.where(v > 0).shift().rolling(12, min_periods=3).median()
    f["vwap"] = (tp * v).cumsum() / f["cum_vol"].replace(0, np.nan)
    f["dist_vwap"] = c / f["vwap"] - 1
    return f


def pullback_low(h, lo, v, c, i):
    j0 = max(0, i - PB_LOOKBACK)
    if i - j0 < 2:
        return None
    k = j0 + int(np.argmax(h[j0:i]))
    hod = h[:i].max()
    if h[k] < hod * 0.999 or k >= i - 1:
        return None
    pl = lo[k + 1:i].min()
    if pl <= hod * (1 - PB_MIN_PCT) and v[k + 1:i].mean() < v[k] and c[i] > h[i - 1] and c[i] >= hod * 0.985:
        return float(pl)
    return None


def setups(x: pd.DataFrame, f: pd.DataFrame, emas: dict | None = None, mode: str | None = None,
           ema_mode: str | None = None) -> pd.Index:
    mode = mode or SETUP_MODE
    t = x.index.time
    watch = ((t >= REG_START) & (t <= LAST_ENTRY) & (f["rvol"] >= RVOL_MIN) & (f["pct"] >= WATCH_PCT)
             & (f["cum_dollar"] >= MIN_DOLLAR_VOL) & (f["dist_vwap"] > 0)).fillna(False).values
    c, h, lo, v = (x[k].to_numpy(float) for k in ("close", "high", "low", "volume"))
    out = []
    for i in range(2, len(x)):
        if not watch[i] or not ema_ok(c[i], emas or {}, ema_mode):
            continue
        if mode == "hod":
            if bool(f["new_hod"].iloc[i]) and (f["vol_ratio"].iloc[i] or 0) >= VOL_SURGE:
                out.append(i)
        elif pullback_low(h, lo, v, c, i) is not None:
            out.append(i)
    return x.index[out]


def initial_stop(x: pd.DataFrame, i: int, entry: float, mode: str | None = None) -> float:
    mode = mode or SETUP_MODE
    c, h, lo, v = (x[k].to_numpy(float) for k in ("close", "high", "low", "volume"))
    s = pullback_low(h, lo, v, c, i) if mode == "pullback" else float(lo[i])
    s = entry * (1 - STOP_PCT) if s is None else s
    return max(min(s, entry * 0.998), entry * (1 - STOP_PCT))


def simulate(sym, x: pd.DataFrame, sig_ts, final: bool = True, mode: str | None = None) -> Trade | None:
    i = x.index.get_loc(sig_ts)
    if i + 1 >= len(x):
        return None
    eb, ets = x.iloc[i + 1], x.index[i + 1]
    if ets.time() >= FLAT:
        return None
    entry = float(eb["open"]) * (1 + SLIP)
    stop = initial_stop(x, i, entry, mode)
    risk = entry - stop
    tr = Trade(sym, ets.date(), sig_ts, ets, entry, stop)
    best, deadline = entry, ets + pd.Timedelta(minutes=MAX_HOLD_MIN)
    for j in range(i + 1, len(x)):
        ts, b = x.index[j], x.iloc[j]
        if j > i + 1 and float(b["open"]) <= stop:
            px, out = float(b["open"]), "STOP_GAP"
        elif float(b["low"]) <= stop:
            px, out = stop, ("STOP" if stop < entry else "TRAIL")
        elif ts >= deadline or ts.time() >= FLAT:
            px, out = float(b["close"]), ("EOD" if ts.time() >= FLAT else "TIME")
        else:
            best = max(best, float(b["close"]))
            if best - entry >= risk:
                stop = max(stop, entry, float(b["low"]))
            continue
        e = px * (1 - SLIP)
        tr.exit_ts, tr.exit, tr.outcome = ts, e, out
        tr.R, tr.ret_pct = (e - entry) / risk, (e / entry - 1) * 100
        return tr
    if not final:
        tr.stop0 = stop
        tr.R = (float(x.iloc[-1]["close"]) - entry) / risk
        return tr
    e = float(x.iloc[-1]["close"]) * (1 - SLIP)
    tr.exit_ts, tr.exit, tr.outcome = x.index[-1], e, "EOD"
    tr.R, tr.ret_pct = (e - entry) / risk, (e / entry - 1) * 100
    return tr
