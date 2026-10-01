"""
US V3 data: 5-minute candles incl. pre-market (4:00-9:30 ET) and after-hours
(16:00-20:00 ET) for the watchlist + market / sector ETFs, daily bars for ATR,
and earnings dates.

GROWING STORE. Yahoo serves only ~60 days of 5-minute history. Every refresh
merges the new download into backtest/ml/cache/v3us/<SYM>.parquet and keeps
everything already stored, so the dataset (and the model retrained on it)
keeps growing past 60 days for as long as the bot runs.
"""
from __future__ import annotations

import logging
import os
import time

import pandas as pd

log = logging.getLogger(__name__)
STORE = os.path.join("backtest", "ml", "cache", "v3us")
ET = "America/New_York"

# market + sector context (ETFs trade like stocks; never traded by V3 itself)
CONTEXT = ["SPY", "QQQ", "IWM", "SMH", "XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XBI", "ARKK", "^VIX"]


def tradable(symbols: list[str]) -> list[str]:
    """Drop indices / futures / FX (^GSPC, NG=F, ...) - they can't be bought."""
    return [s for s in symbols if not s.startswith("^") and "=" not in s and s not in CONTEXT]


def _path(sym: str, kind: str = "5m") -> str:
    os.makedirs(STORE, exist_ok=True)
    return os.path.join(STORE, f"{sym.replace('^', '_')}_{kind}.parquet")


def refresh_intraday(symbols: list[str], batch: int = 40, period: str = "60d") -> dict:
    """Download 5-minute bars (pre/post included) and merge into the store."""
    import yfinance as yf
    stats = {"symbols": 0, "new_rows": 0}
    for i in range(0, len(symbols), batch):
        chunk = symbols[i:i + batch]
        for attempt in range(3):
            try:
                df = yf.download(chunk, period=period, interval="5m", prepost=True, group_by="ticker",
                                 progress=False, auto_adjust=False, threads=True)
                break
            except Exception as e:                       # Yahoo throttling
                log.warning("download failed (%s), retrying: %r", chunk[0], e)
                time.sleep(5 * (attempt + 1))
        else:
            continue
        for s in chunk:
            try:
                x = df[s] if isinstance(df.columns, pd.MultiIndex) else df
            except KeyError:
                continue
            x = x.dropna(subset=["Close"])
            if x.empty:
                continue
            x = x.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
            x.index = x.index.tz_convert(ET)
            x.index.name = "ts"
            f = _path(s)
            if os.path.exists(f):
                old = pd.read_parquet(f)
                n0 = len(old)
                x = pd.concat([old, x])
                x = x[~x.index.duplicated(keep="last")].sort_index()
                stats["new_rows"] += len(x) - n0
            else:
                stats["new_rows"] += len(x)
            x.to_parquet(f)
            stats["symbols"] += 1
        time.sleep(1)
    return stats


def refresh_daily(symbols: list[str], period: str = "2y") -> None:
    import yfinance as yf
    for i in range(0, len(symbols), 60):
        chunk = symbols[i:i + 60]
        df = yf.download(chunk, period=period, interval="1d", group_by="ticker", progress=False,
                         auto_adjust=True, threads=True)
        for s in chunk:
            try:
                x = df[s] if isinstance(df.columns, pd.MultiIndex) else df
            except KeyError:
                continue
            x = x.dropna(subset=["Close"]).rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
            if len(x):
                x.index = pd.to_datetime(x.index).tz_localize(None)
                x.index.name = "date"
                x.to_parquet(_path(s, "1d"))


def refresh_earnings(symbols: list[str]) -> None:
    """Earnings timestamps (Yahoo), cached per symbol; refreshed if older than 3 days."""
    import yfinance as yf
    for s in symbols:
        f = _path(s, "earn")
        if os.path.exists(f) and time.time() - os.path.getmtime(f) < 3 * 86400:
            continue
        try:
            e = yf.Ticker(s).get_earnings_dates(limit=16)
            ts = pd.DatetimeIndex(e.index) if e is not None and len(e) else pd.DatetimeIndex([])
        except Exception:
            ts = pd.DatetimeIndex([])
        pd.DataFrame({"ts": ts.tz_convert(ET) if ts.tz is not None else ts}).to_parquet(f)


def load(sym: str, kind: str = "5m") -> pd.DataFrame:
    f = _path(sym, kind)
    return pd.read_parquet(f) if os.path.exists(f) else pd.DataFrame()
