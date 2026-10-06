"""
US SCR (Small-Cap Runners) data - all free, no API keys.

  universe()      every LISTED US small / micro cap (Nasdaq's public screener
                  file: market cap < $2B, price >= $1, Nasdaq/NYSE/NYSE
                  American - no OTC / pink sheets), cached daily
  screen_live()   today's movers right now: Yahoo screener query for small /
                  micro caps up >= X% on volume (the automated version of an
                  app's "top % gainers" list), plus unusual-volume names
  daily_bars()    daily OHLCV (prev close, 20-day average volume, prior runs)
  intraday()      5-minute bars incl. pre/post market (Yahoo serves ~60 days)

The 5-minute bars of every stock-day the model looks at are kept in a
growing store (us_scr/cache/bars/<SYMBOL>.parquet) so later retrains can
use more than Yahoo's 60-day window.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "us_scr", "cache")
BARS = os.path.join(CACHE, "bars")
ET = "America/New_York"
LISTED = {"NMS", "NGM", "NCM", "NYQ", "ASE", "BTS", "PCX"}       # Yahoo exchange codes (no PNK / OTC)
MAX_MCAP = 2_000_000_000
MIN_PRICE = 1.0
UA = {"User-Agent": "Mozilla/5.0 (us_scr research)", "Accept": "application/json"}


def _p(*parts):
    p = os.path.join(CACHE, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


# ─── universe ────────────────────────────────────────────────────────────

def universe(refresh: bool = False, max_age_h: float = 20) -> pd.DataFrame:
    """symbol, name, mcap, price, sector, industry, country - listed small/micro caps."""
    f = _p("universe.parquet")
    if os.path.exists(f) and not refresh and (time.time() - os.path.getmtime(f)) < max_age_h * 3600:
        return pd.read_parquet(f)
    import requests
    try:
        r = requests.get("https://api.nasdaq.com/api/screener/stocks?tableonly=true&download=true", headers=UA, timeout=40)
        r.raise_for_status()
        d = pd.DataFrame(r.json()["data"]["rows"])
    except Exception as e:
        if os.path.exists(f):
            log.warning("Nasdaq list unavailable (%s) - using cached universe", e)
            return pd.read_parquet(f)
        raise
    d["mcap"] = pd.to_numeric(d["marketCap"], errors="coerce")
    d["price"] = pd.to_numeric(d["lastsale"].str.replace("$", "", regex=False), errors="coerce")
    d["symbol"] = d["symbol"].str.strip().str.replace("/", "-", regex=False).str.replace("^", "-P", regex=False)
    d = d[(d["mcap"] > 0) & (d["mcap"] < MAX_MCAP) & (d["price"] >= MIN_PRICE)]
    d = d[~d["symbol"].str.contains(r"[\^\.]|-P", regex=True)]               # no preferreds / share classes with dots
    d = d[~d["name"].str.contains(r"\b(?:Warrants?|Rights?|Units?|Preferred|Notes?|Debentures?)\b", case=False, regex=True)]
    out = d[["symbol", "name", "mcap", "price", "sector", "industry", "country"]].reset_index(drop=True)
    out.to_parquet(f, index=False)
    return out


def screen_live(min_change: float = 10.0, min_volume: int = 300_000, size: int = 250) -> pd.DataFrame:
    """Small / micro caps moving now (Yahoo screener). Columns: symbol, pct,
    price, volume, avg_vol, rvol, mcap, exchange, prev_close, day_high."""
    import yfinance as yf
    from yfinance import EquityQuery as EQ
    q = EQ("and", [EQ("eq", ["region", "us"]), EQ("lt", ["intradaymarketcap", MAX_MCAP]),
                   EQ("gt", ["percentchange", min_change]), EQ("gt", ["dayvolume", min_volume]),
                   EQ("gt", ["intradayprice", MIN_PRICE])])
    for attempt in range(3):
        try:
            r = yf.screen(q, sortField="percentchange", sortAsc=False, size=size)
            break
        except Exception as e:
            log.warning("screener failed (%s), retry %d", e, attempt + 1)
            time.sleep(3)
    else:
        return pd.DataFrame()
    rows = []
    for x in r.get("quotes", []):
        if x.get("exchange") not in LISTED or x.get("quoteType") != "EQUITY":
            continue
        avg = x.get("averageDailyVolume10Day") or x.get("averageDailyVolume3Month") or 0
        rows.append({"symbol": x["symbol"], "pct": x.get("regularMarketChangePercent"), "price": x.get("regularMarketPrice"),
                     "volume": x.get("regularMarketVolume"), "avg_vol": avg,
                     "rvol": (x.get("regularMarketVolume") or 0) / avg if avg else np.nan,
                     "mcap": x.get("marketCap"), "exchange": x.get("exchange"),
                     "prev_close": x.get("regularMarketPreviousClose"), "day_high": x.get("regularMarketDayHigh"),
                     "name": x.get("shortName") or x.get("longName")})
    return pd.DataFrame(rows)


# ─── bars ────────────────────────────────────────────────────────────────

def _yf(symbols, **kw) -> dict:
    """{symbol: frame} from one batched yfinance download (retries)."""
    import yfinance as yf
    for attempt in range(3):
        try:
            df = yf.download(symbols, group_by="ticker", progress=False, auto_adjust=False, threads=True, **kw)
            break
        except Exception as e:
            log.warning("yahoo download failed (%s), retry %d", e, attempt + 1)
            time.sleep(5 * (attempt + 1))
    else:
        return {}
    out = {}
    if df is None or df.empty:
        return out
    for s in symbols:
        try:
            x = df[s] if isinstance(df.columns, pd.MultiIndex) else df
        except KeyError:
            continue
        x = x.dropna(subset=["Close"])
        if not x.empty:
            out[s] = x.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
    return out


def daily_bars(symbols: list[str], period: str = "6mo", batch: int = 100) -> pd.DataFrame:
    """Long frame date, symbol, open, high, low, close, volume (split-adjusted)."""
    parts = []
    for i in range(0, len(symbols), batch):
        for s, x in _yf(symbols[i:i + batch], period=period, interval="1d").items():
            x = x.copy()
            x.index = pd.DatetimeIndex(x.index).tz_localize(None).normalize()
            parts.append(x.assign(symbol=s).rename_axis("date").reset_index())
        log.info("daily: %d/%d", min(i + batch, len(symbols)), len(symbols))
        time.sleep(0.5)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def intraday(symbols: list[str], period: str = "60d", batch: int = 40, keep_days: dict | None = None) -> dict:
    """5-min bars (pre/post) per symbol, ET index. keep_days={sym: set(dates)}
    keeps only those days (saves memory). Also merged into the growing store."""
    out = {}
    for i in range(0, len(symbols), batch):
        for s, x in _yf(symbols[i:i + batch], period=period, interval="5m", prepost=True).items():
            x.index = pd.DatetimeIndex(x.index).tz_convert(ET)
            x.index.name = "ts"
            if keep_days is not None:
                x = x[pd.Index(x.index.date).isin(keep_days.get(s, set()))]
            if x.empty:
                continue
            store(s, x)
            out[s] = x
        log.info("5m: %d/%d", min(i + batch, len(symbols)), len(symbols))
        time.sleep(1)
    return out


def store(sym: str, x: pd.DataFrame) -> None:
    f = _p("bars", f"{sym}.parquet")
    if os.path.exists(f):
        old = pd.read_parquet(f)
        x = pd.concat([old, x])
        x = x[~x.index.duplicated(keep="last")].sort_index()
    x.to_parquet(f)


def load_store(sym: str) -> pd.DataFrame:
    f = os.path.join(BARS, f"{sym}.parquet")
    return pd.read_parquet(f) if os.path.exists(f) else pd.DataFrame()


def now_et() -> datetime:
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo(ET))
