"""
US SCR (Small-Cap Runners) - the rules, shared by research and the live job.

Everything on bar t uses data up to the CLOSE of bar t only.

WATCH  (the dynamic list): today's change vs the previous close >= WATCH_PCT,
       dollar volume traded today >= MIN_DOLLAR_VOL, price >= MIN_PRICE.
SESSION regular hours only (09:35-15:30 entries): Yahoo reports no pre-market
       volume, so a pre-market run can't be volume-confirmed; pre-market PRICE
       moves still count (today's % change, the opening gap).
SETUP  pullback continuation ("first pullback" of the small-cap momentum
       playbook): the stock made its high of day within the last PB_LOOKBACK
       bars, then pulled back >= PB_MIN_PCT on LIGHTER volume than the
       high-of-day bar, and bar t now closes above the previous bar's high
       (the reclaim), above VWAP and within 3% of the high of day.
       (Chosen 2026-10-06 over the plain high-of-day breakout: it lost the
       least in BOTH halves of a 48-variant test - about -0.43% vs -1.1% /
       -0.3% per trade - but no small-cap setup was positive in both halves.)
ENTRY  the NEXT bar's open + slippage (an alert needs time to reach a human).
STOP   structural: below the pullback low, never more than STOP_PCT (8%) below
       the entry (and at least 0.5% below it). A bar that OPENS below the stop (gap / trading
       halt) exits at that open - small caps can skip straight through a stop.
WINDOW entries 09:35-15:30 ET (whole regular session - some runners only start in
       the afternoon; user, 2026-10-06), at most 15 a day (live cap).
EXIT   after +1R the stop moves to breakeven, then trails under each closed
       bar's low (ride the run, leave when it stalls); out after MAX_HOLD_MIN,
       and flat by 15:55 ET. Slippage again on the exit.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time as dtime

import numpy as np
import pandas as pd

WATCH_PCT = 0.10
MIN_DOLLAR_VOL = 1_000_000
MIN_PRICE = 1.0
VOL_SURGE = 2.0
STOP_PCT = 0.08            # fixed stop below the entry (backtest: bar-low / 5% stops got shaken out by 5-min noise)
MAX_STOP_PCT = STOP_PCT
MAX_HOLD_MIN = 180
PILLAR_RVOL, PILLAR_SHARES, PILLAR_PRICE, PILLAR_FLOAT = 5.0, 1_000_000, (1.0, 20.0), 20_000_000
PB_LOOKBACK = 6            # bars back to look for the high of day
PB_MIN_PCT = 0.02          # pullback depth from the high of day
SLIP_REGULAR = 0.0025
SLIP_PRE = 0.005
PRE_START, REG_START, LAST_ENTRY, FLAT = dtime(7, 0), dtime(9, 35), dtime(15, 30), dtime(15, 55)   # entries: whole regular session
MAX_PER_SYMBOL_DAY = 2


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
    session: str = "regular"


def slip(ts) -> float:
    return SLIP_PRE if ts.time() < dtime(9, 30) else SLIP_REGULAR


def bar_features(x: pd.DataFrame, prev_close: float, avg_vol20: float) -> pd.DataFrame:
    """Per-bar causal features for one stock-day (x: 5-min bars 04:00-20:00 ET)."""
    f = pd.DataFrame(index=x.index)
    c, h, lo, v = x["close"], x["high"], x["low"], x["volume"].fillna(0)
    f["pct"] = c / prev_close - 1
    f["cum_vol"] = v.cumsum()
    f["cum_dollar"] = (v * (h + lo + c) / 3).cumsum()
    f["rvol"] = f["cum_vol"] / avg_vol20 if avg_vol20 and avg_vol20 > 0 else np.nan
    f["hod_prior"] = h.cummax().shift()
    f["new_hod"] = c > f["hod_prior"]
    # Yahoo reports 0 volume on pre/post-market bars - ignore them in the volume baseline
    med12 = v.where(v > 0).shift().rolling(12, min_periods=3).median()
    f["vol_ratio"] = v / med12.replace(0, np.nan)
    tp = (h + lo + c) / 3
    f["vwap"] = (tp * v).cumsum() / f["cum_vol"].replace(0, np.nan)
    f["dist_vwap"] = c / f["vwap"] - 1
    f["n_hod"] = f["new_hod"].astype(int).cumsum()
    f["bar_range"] = (h - lo) / c
    f["close_loc"] = (c - lo) / (h - lo).replace(0, np.nan)
    f["ret_3"] = c / c.shift(3) - 1
    f["minute"] = x.index.hour * 60 + x.index.minute
    reg = x.index.time >= dtime(9, 30)
    f["reg_open_gap"] = np.nan
    if reg.any():
        f.loc[reg, "reg_open_gap"] = x.loc[reg, "open"].iloc[0] / prev_close - 1
    f["stop_pct"] = 1 - lo / c
    return f


def setups(x: pd.DataFrame, f: pd.DataFrame, sessions=("regular",)) -> pd.Index:
    """Bars where the rule fires (signal at the bar close)."""
    t = x.index.time
    in_pre = (t >= PRE_START) & (t < dtime(9, 25))
    in_reg = (t >= REG_START) & (t <= LAST_ENTRY)
    ok_time = (in_pre & ("pre" in sessions)) | (in_reg & ("regular" in sessions))
    watch = (ok_time & (f["pct"] >= WATCH_PCT) & (f["cum_dollar"] >= MIN_DOLLAR_VOL)
             & (x["close"] >= MIN_PRICE) & (f["dist_vwap"] > 0)).fillna(False).values
    c, h, lo, v = (x[k].to_numpy(float) for k in ("close", "high", "low", "volume"))
    keep = []
    for i in range(2, len(x)):
        if watch[i] and pullback_low(h, lo, v, c, i) is not None:
            keep.append(i)
    return x.index[keep]


def pullback_low(h, lo, v, c, i):
    """Pullback low if bar i is a pullback-continuation reclaim (else None)."""
    j0 = max(0, i - PB_LOOKBACK)
    if i - j0 < 2:
        return None
    k = j0 + int(np.argmax(h[j0:i]))
    hod = h[:i].max()
    if h[k] < hod * 0.999 or k >= i - 1:            # the high of day must be recent, with >= 1 pullback bar after it
        return None
    pl = lo[k + 1:i].min()
    lighter = v[k + 1:i].mean() < v[k]
    if pl <= hod * (1 - PB_MIN_PCT) and lighter and c[i] > h[i - 1] and c[i] >= hod * 0.97:
        return float(pl)
    return None


def five_pillars(rvol: float, cum_vol: float, price: float, pct: float, mcap: float | None) -> list:
    """The small-cap momentum "5 pillars" that can be measured (the news
    pillar is read separately, live only): [(name, met), ...].
    Float has no free history -> shares outstanding (mcap / price) is the
    proxy (float <= shares outstanding). Backtest 2026-10-06 (pullback
    setup, 912 trades): 5/5 = +1.57%/trade (both halves positive: +0.49% /
    +2.62%) vs about -0.5% for every other group - but driven by a few big
    winners (best +136%, median -6.9%, day-level t 0.38)."""
    shares = (mcap or 0) / price if price else 0
    return [("RVOL≥5x", bool(rvol == rvol and rvol >= PILLAR_RVOL)), ("vol≥1M sh", bool(cum_vol >= PILLAR_SHARES)),
            ("$1-20", bool(PILLAR_PRICE[0] <= price <= PILLAR_PRICE[1])), ("up≥10%", bool(pct >= WATCH_PCT)),
            ("float≤20M*", bool(0 < shares <= PILLAR_FLOAT))]


def initial_stop(x: pd.DataFrame, i: int, entry: float) -> float:
    """Structural stop for the signal at bar i: the pullback low, capped at STOP_PCT."""
    c, h, lo, v = (x[k].to_numpy(float) for k in ("close", "high", "low", "volume"))
    pl = pullback_low(h, lo, v, c, i)
    pl = entry * (1 - STOP_PCT) if pl is None else pl
    return max(min(pl, entry * 0.995), entry * (1 - STOP_PCT))


def simulate(sym, x: pd.DataFrame, sig_ts: pd.Timestamp, final: bool = True) -> Trade | None:
    """Paper trade for a signal at the close of bar sig_ts (same day only).
    final=False (live): if the bars so far reach no exit, return the trade
    with outcome OPEN and its current (possibly trailed) stop in stop0."""
    i = x.index.get_loc(sig_ts)
    if i + 1 >= len(x):
        return None
    eb = x.iloc[i + 1]
    ets = x.index[i + 1]
    if ets.time() >= FLAT:
        return None
    entry = float(eb["open"]) * (1 + slip(ets))
    stop = initial_stop(x, i, entry)
    risk = entry - stop
    tr = Trade(sym, ets.date(), sig_ts, ets, entry, stop, session="pre" if ets.time() < dtime(9, 30) else "regular")
    best = entry
    deadline = ets + pd.Timedelta(minutes=MAX_HOLD_MIN)
    for j in range(i + 1, len(x)):
        ts, b = x.index[j], x.iloc[j]
        if j > i + 1 and float(b["open"]) <= stop:              # gap / halt through the stop
            px, out = float(b["open"]), "STOP_GAP"
        elif float(b["low"]) <= stop:
            px, out = stop, ("STOP" if stop < entry else "TRAIL")
        elif ts >= deadline or ts.time() >= FLAT:
            px, out = float(b["close"]), ("EOD" if ts.time() >= FLAT else "TIME")
        else:
            best = max(best, float(b["close"]))
            if best - entry >= risk:                             # +1R reached: protect, then trail bar lows
                stop = max(stop, entry, float(b["low"]))
            continue
        exitp = px * (1 - slip(ts))
        tr.exit_ts, tr.exit, tr.outcome = ts, exitp, out
        tr.R = (exitp - entry) / risk
        tr.ret_pct = (exitp / entry - 1) * 100
        return tr
    if not final:                                                # live: still open after the last closed bar
        tr.stop0 = stop
        tr.R = (float(x.iloc[-1]["close"]) - entry) / risk
        return tr
    b = x.iloc[-1]
    exitp = float(b["close"]) * (1 - SLIP_REGULAR)
    tr.exit_ts, tr.exit, tr.outcome = x.index[-1], exitp, "EOD"
    tr.R, tr.ret_pct = (exitp - entry) / risk, (exitp / entry - 1) * 100
    return tr
