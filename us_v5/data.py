"""
US V5 data: S&P 500 membership, Yahoo daily bars, and the point-in-time
large-cap panel (the US counterpart of NSE swing/data/panel.py).

MEMBERSHIP. Today's S&P 500 list from Wikipedia (symbol, GICS sector,
"Date added"); a committed copy lives in us_v5/seed/sp500.csv so research
and the live job never depend on Wikipedia being reachable.

PRICES. Yahoo daily bars with auto_adjust=False: Open/High/Low/Close are
split-adjusted, `Adj Close` is split + dividend adjusted. Adjusted O/H/L use
the same per-day factor (Adj Close / Close), so returns are TOTAL returns
(dividends reinvested) - and the SPY benchmark is built the same way.
Dollar volume = Close * Volume (both split-adjusted, so splits don't move it).

UNIVERSE (point in time). On the first trading day of each month, using data
up to the previous session only: stocks already in the S&P 500 by then
(Date added <= that day), with >= min_history sessions and close >=
min_price, ranked by median dollar volume over value_window sessions; the
top `size` are members for that month.

KNOWN BIAS: Yahoo has no history for delisted tickers, so companies that left
the index (bankruptcies, takeovers) are missing. Results are optimistic by
that amount; research compares against an equal-weight portfolio of the
same universe, which carries the same bias, to isolate ranking skill.
"""
from __future__ import annotations

import io
import logging
import os
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED_SP500 = os.path.join(ROOT, "us_v5", "seed", "sp500.csv")
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
BARS_FILE = "daily_bars.parquet"
BATCH = 80


# ─── membership ──────────────────────────────────────────────────────────

def fetch_sp500() -> pd.DataFrame:
    """Current S&P 500 from Wikipedia: symbol (Yahoo form), name, sector,
    sub_industry, date_added."""
    import requests
    r = requests.get(WIKI_URL, headers={"User-Agent": "Mozilla/5.0 (us_v5 research)"}, timeout=30)
    r.raise_for_status()
    t = pd.read_html(io.StringIO(r.text))[0]
    out = pd.DataFrame({
        "symbol": t["Symbol"].str.replace(".", "-", regex=False).str.strip(),
        "name": t["Security"], "sector": t["GICS Sector"], "sub_industry": t["GICS Sub-Industry"],
        "date_added": pd.to_datetime(t["Date added"], errors="coerce"),
    })
    if len(out) < 450:
        raise ValueError(f"S&P 500 table looks wrong ({len(out)} rows)")
    return out


def load_sp500(refresh: bool = False) -> pd.DataFrame:
    """Committed seed, optionally refreshed from Wikipedia (seed kept on failure)."""
    if refresh or not os.path.exists(SEED_SP500):
        try:
            df = fetch_sp500()
            os.makedirs(os.path.dirname(SEED_SP500), exist_ok=True)
            df.to_csv(SEED_SP500, index=False)
        except Exception as e:
            if not os.path.exists(SEED_SP500):
                raise
            log.warning("S&P 500 refresh failed (%s) - using the committed seed", e)
    df = pd.read_csv(SEED_SP500, parse_dates=["date_added"])
    return df


# ─── Yahoo daily bars ────────────────────────────────────────────────────

def _download(symbols: list[str], start: date, end: date) -> pd.DataFrame:
    """Long frame: date, symbol, open, high, low, close, adj_close, volume."""
    import yfinance as yf
    parts = []
    for i in range(0, len(symbols), BATCH):
        chunk = symbols[i:i + BATCH]
        for attempt in range(3):
            try:
                d = yf.download(chunk, start=str(start), end=str(end + timedelta(days=1)), auto_adjust=False,
                                progress=False, group_by="ticker", threads=True)
                break
            except Exception as e:                       # network hiccup -> retry
                log.warning("yahoo batch %d failed (%s), retry %d", i, e, attempt + 1)
                time.sleep(3 * (attempt + 1))
        else:
            continue
        if d is None or d.empty:
            continue
        for s in chunk:
            if s not in d.columns.get_level_values(0):
                continue
            x = d[s].dropna(subset=["Close"])
            if x.empty:
                continue
            parts.append(pd.DataFrame({
                "date": pd.DatetimeIndex(x.index).tz_localize(None).normalize(), "symbol": s,
                "open": x["Open"].values, "high": x["High"].values, "low": x["Low"].values,
                "close": x["Close"].values, "adj_close": x["Adj Close"].values, "volume": x["Volume"].values,
            }))
        log.info("yahoo: %d/%d symbols", min(i + BATCH, len(symbols)), len(symbols))
    if not parts:
        return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"])
    return pd.concat(parts, ignore_index=True)


def load_bars(cache: str, symbols: list[str], start: date, end: date, refresh_days: int = 0) -> pd.DataFrame:
    """Cached daily bars. A full download the first time; afterwards only the
    last `refresh_days` calendar days (Yahoo can revise the latest bars) and
    any symbols not cached yet. Dividends after the cached window change the
    whole Adj Close history, so a refresh re-downloads a symbol in full when
    its newest overlapping Adj Close no longer matches."""
    f = os.path.join(cache, BARS_FILE)
    old = pd.read_parquet(f) if os.path.exists(f) else pd.DataFrame()
    have = set(old["symbol"].unique()) if not old.empty else set()
    new_syms = [s for s in symbols if s not in have]
    parts = [old] if not old.empty else []
    if new_syms:
        log.info("yahoo: full history for %d symbols", len(new_syms))
        parts.append(_download(new_syms, start, end))
    if refresh_days and have:
        # from the cache's last bar (a gap of weeks must not leave a hole), at least refresh_days back
        upd_start = min(end - timedelta(days=refresh_days), old["date"].max().date() - timedelta(days=3))
        upd = _download(sorted(have & set(symbols)), upd_start, end)
        if not upd.empty:
            # dividend / split since the cache was built -> history rescaled -> refetch fully
            chk = old.merge(upd, on=["date", "symbol"], suffixes=("", "_n"))
            chk = chk[chk["date"] == chk.groupby("symbol")["date"].transform("min")]
            moved = chk.loc[(chk["adj_close"] / chk["adj_close_n"] - 1).abs() > 1e-4, "symbol"].unique().tolist()
            if moved:
                log.info("yahoo: %d symbols re-adjusted (dividend/split) - full refetch", len(moved))
                parts = [p[~p["symbol"].isin(moved)] for p in parts]
                parts.append(_download(moved, start, end))
                upd = upd[~upd["symbol"].isin(moved)]
            parts.append(upd)
    if not parts:
        return pd.DataFrame()
    bars = pd.concat(parts, ignore_index=True).drop_duplicates(["date", "symbol"], keep="last")
    bars = bars.sort_values(["symbol", "date"]).reset_index(drop=True)
    for c in ("open", "high", "low", "close", "adj_close", "volume"):
        bars[c] = bars[c].astype("float64")
    if new_syms or refresh_days:
        os.makedirs(cache, exist_ok=True)
        bars.to_parquet(f, index=False)
    return bars[(bars["date"] >= pd.Timestamp(start)) & (bars["date"] <= pd.Timestamp(end))].reset_index(drop=True)


# ─── panel ───────────────────────────────────────────────────────────────

def assemble(bars: pd.DataFrame, members: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """bars: long daily bars. members: symbol, date_added. Returns the adjusted
    panel with entity, ret_cc, adj O/H/L/C, value, n_days, in_universe."""
    eq = bars.dropna(subset=["close", "adj_close"]).copy()
    eq = eq[(eq["close"] > 0) & (eq["adj_close"] > 0)]
    eq = eq.sort_values(["symbol", "date"]).reset_index(drop=True)
    eq["entity"] = eq["symbol"]
    g = eq.groupby("entity", sort=False)
    eq["prevclose"] = g["close"].shift()
    eq["ret_cc"] = g["adj_close"].pct_change()
    first = eq["entity"] != eq["entity"].shift()
    eq.loc[first, "ret_cc"] = 0.0
    bad = ~np.isfinite(eq["ret_cc"]) | (eq["ret_cc"].abs() > 0.6)       # bad prints -> neutral day
    eq.loc[bad, "ret_cc"] = 0.0
    k = eq["adj_close"] / eq["close"]
    for c in ("open", "high", "low"):
        eq[f"adj_{c}"] = eq[c] * k
    eq["prevclose_adj"] = g["adj_close"].shift()
    eq["value"] = eq["close"] * eq["volume"]
    eq["trades"] = np.nan
    eq["neutral_day"] = bad
    eq["n_days"] = g.cumcount() + 1
    added = members.set_index("symbol")["date_added"]
    eq["date_added"] = eq["symbol"].map(added)
    return add_universe(eq, cfg)


def add_universe(eq: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    u = cfg["universe"]
    eq = eq.sort_values(["entity", "date"])
    eq["_med_val"] = eq.groupby("entity")["value"].transform(
        lambda s: s.rolling(u["value_window"], min_periods=u["value_window"] // 2).median())
    days = pd.DatetimeIndex(sorted(eq["date"].unique()))
    month_first = days[~days.to_period("M").duplicated()]
    members = []
    for d in month_first:
        i = days.get_loc(d)
        if i == 0:
            continue
        prev = days[i - 1]                                   # data up to the prior session only
        snap = eq[eq["date"] == prev]
        snap = snap[(snap["n_days"] >= u["min_history"]) & (snap["close"] >= u["min_price"])
                    & (snap["date_added"].isna() | (snap["date_added"] <= d))]
        top = snap.nlargest(u["size"], "_med_val")["entity"]
        members.append(pd.DataFrame({"month": d.to_period("M"), "entity": top.values}))
    mem = pd.concat(members, ignore_index=True)
    mem["in_universe"] = True
    eq["month"] = eq["date"].dt.to_period("M")
    eq = eq.merge(mem, on=["month", "entity"], how="left")
    eq["in_universe"] = eq["in_universe"].astype("boolean").fillna(False).astype(bool)
    return eq.drop(columns=["month", "_med_val"]).sort_values(["entity", "date"]).reset_index(drop=True)


def load_data(cfg: dict, refresh_days: int = 0, refresh_members: bool = False) -> dict:
    """Everything research / live need: panel, sector map, context closes."""
    from us_v5.core import config as cfgmod
    cache = cfgmod.path(cfg, "cache")
    start = pd.Timestamp(cfg["data"]["start"]).date()
    end = pd.Timestamp(cfg["data"]["end"]).date() if cfg["data"]["end"] else date.today() - timedelta(days=1)
    members = load_sp500(refresh_members)
    ctx_syms = list(cfg["data"]["context"])
    bars = load_bars(cache, sorted(set(members["symbol"]) | set(ctx_syms)), start, end, refresh_days)
    ctx = bars[bars["symbol"].isin(ctx_syms)]
    closes = ctx.pivot(index="date", columns="symbol", values="adj_close").sort_index()
    stocks = bars[bars["symbol"].isin(set(members["symbol"]))]
    panel = assemble(stocks, members, cfg)
    sector = members.set_index("symbol")["sector"]
    return {"panel": panel, "industry": sector, "context": closes, "members": members,
            "start": start, "end": end}
