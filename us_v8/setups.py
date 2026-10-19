"""
V8 "leaders" swing research - indicators and entry signals (market-agnostic).

Setups copied from the traders who published them, fixed BEFORE any test:

  BO   Qullamaggie breakout inside Minervini's trend template
       (qullamaggie.com "My 3 timeless setups"; Minervini "Trade Like a Stock
       Market Wizard"): a stock in a stage-2 uptrend (close > SMA50 > SMA150 >
       SMA200, SMA200 rising for a month, >= 30% above its 52-week low, within
       25% of its 52-week high) that is a relative-strength leader (IBD-style
       weighted 3/6/9/12-month return in the top 15% of the liquid market),
       surfing its rising 10/20-day averages in a TIGHT 10-day range
       (high-low <= 2.5 x ADR) and closing within one ADR of that range's
       high. Next session: buy-stop at the 10-day high; skip if it opens more
       than one ADR above (too extended); stop = entry - min(ADR, 8%).
  EP   Qullamaggie episodic pivot: gap up >= gap_min (10% US / 8% NSE) on a
       volume shock (traded value >= 3x its 50-day average) that holds the gap
       (close above open). Buy at that close; stop = the gap day's low
       (at most 12% away).
  ATH  Wilcox & Crittenden (2005) / Zarattini, Pagani & Wilcox (2025) trend
       following: close at an all-time high (history in the data, >= 1 year),
       buy next open, 10 x ATR(42) trailing stop from the highest close,
       exit at the next open after a close below it.

Inputs are wide DataFrames (index = trading dates, columns = entities):
ADJUSTED open/high/low/close (any scale), traded value in money, raw close
(for the price floor) and an optional universe mask. Every value on row t
uses data up to and including the close of t only.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


# Cash / bond ETFs are not trends: money-market ETFs make a "new high" almost every day
# (seen live: LIQUID1 bought on day one). Gold / silver / equity ETFs stay in.
CASH_BOND_ETF = re.compile(r"LIQ|^CASHIETF$|^E?BBETF|GILT(?!S$)|GSEC|SDL")


def cash_like(symbols) -> list[bool]:
    return [bool(CASH_BOND_ETF.search(str(s))) for s in symbols]


def _sma(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=n).mean()


def indicators(o, h, l, c, val, raw_c, p: dict, universe: pd.DataFrame | None = None) -> dict:
    ind = {}
    hist = c.notna().cumsum()                                  # sessions of history so far
    med20 = val.rolling(20, min_periods=15).median()
    liquid = (med20 >= p["min_value"]) & (raw_c >= p["min_price"]) & c.notna()
    if universe is not None:
        liquid &= universe.reindex(index=c.index, columns=c.columns).fillna(False).astype(bool)
    ind["liquid"] = liquid
    ind["hist"] = hist

    for n in (10, 20, 50, 150, 200):
        ind[f"sma{n}"] = _sma(c, n)
    ind["adr"] = (h / l).rolling(14, min_periods=10).mean() - 1.0          # Qullamaggie's ADR, as a fraction
    prev_c = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - prev_c).abs(), (l - prev_c).abs()))
    ind["atr42"] = tr.rolling(42, min_periods=30).mean()

    # IBD-style relative strength (40% last quarter + 20% each earlier quarter), ranked across liquid stocks
    roc = {k: c / c.shift(k) - 1 for k in (63, 126, 189, 252)}
    rs_raw = 0.4 * roc[63] + 0.2 * roc[126] + 0.2 * roc[189] + 0.2 * roc[252]
    ind["rs"] = rs_raw.where(liquid & (hist >= 253)).rank(axis=1, pct=True)

    # --- BO: trend template + tight base near its high ---
    sma50, sma150, sma200 = ind["sma50"], ind["sma150"], ind["sma200"]
    template = ((c > sma50) & (sma50 > sma150) & (sma150 > sma200) & (sma200 > sma200.shift(21))
                & (c >= 1.30 * l.rolling(252, min_periods=200).min())
                & (c >= 0.75 * h.rolling(252, min_periods=200).max()))
    pivot = h.rolling(10, min_periods=10).max()
    tight = (pivot - l.rolling(10, min_periods=10).min()) / c <= p["bo_tight_adr"] * ind["adr"]
    near = c >= pivot * (1 - ind["adr"])
    surf = (ind["sma10"] > ind["sma20"]) & (c > ind["sma20"])
    ind["bo"] = liquid & template & (ind["rs"] >= p["bo_rs_min"]) & tight & near & surf
    ind["pivot"] = pivot

    # --- EP: gap + volume shock that holds into the close (known at the close of the gap day) ---
    gap = o / prev_c - 1
    vratio = val / val.shift(1).rolling(50, min_periods=30).mean()
    liquid_prev = ((med20.shift(1) >= p["min_value"]) & (raw_c.shift(1) >= p["min_price"]))
    if universe is not None:
        liquid_prev &= universe.reindex(index=c.index, columns=c.columns).fillna(False).astype(bool)
    ep = liquid_prev & (gap >= p["ep_gap"]) & (vratio >= p["ep_vol"]) & (c > o) & (hist >= 60)
    ind["ep"] = ep
    ind["ep_fresh"] = ep & ((prev_c / c.shift(127) - 1) <= p["ep_fresh_max"])   # "best if it hasn't rallied in 3-6 months"
    ind["vratio"] = vratio

    # --- ATH: close >= highest close of all prior history (>= 1 year of it), or of the
    #     last `ath_lookback` sessions (what a live job with ~2 years of data can see) ---
    n = p.get("ath_lookback")
    prior_max = c.shift(1).cummax() if not n else c.shift(1).rolling(n, min_periods=n).max()
    moves = ind["atr42"] / c >= p.get("min_atr_pct", 0.0025)        # not cash-like (money-market ETFs make new highs daily)
    ind["ath"] = liquid & moves & (hist >= max(253, (n or 0) + 1)) & (c >= prior_max)
    ind["ath_504"] = liquid & moves & (hist >= 505) & (c >= c.shift(1).rolling(504, min_periods=504).max())
    return ind


def regime(index_close: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Qullamaggie's market filter: index 10-day average above its 20-day average."""
    s = index_close.reindex(dates).ffill()
    return (s.rolling(10).mean() > s.rolling(20).mean()).fillna(False)
