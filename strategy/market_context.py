"""
Point-in-time market + catalyst context for model L_ML_META_V2.

The base models (A-K, SCORE_ENGINE) only ever see ONE symbol's candles
for ONE session. This module supplies what they can't see - the state of
the broad market and whether the stock has a catalyst - as numbers the
V2 classifier can learn from:

  MARKET   VIX / VXN level vs their own 50-day SMA and their move today,
           VIX/VIX3M term structure (>1 = backwardation = panic), % of a
           fixed large-cap universe above its 50-day SMA (breadth),
           QQQ/SPY intraday trend and daily trend.
  CATALYST earnings proximity and last EPS surprise, today's gap vs the
           prior close, time-of-day relative volume ("is it in play"),
           relative strength vs QQQ, distance from recent highs.

NO LOOKAHEAD - every lookup is "as of" the signal candle:
  - intraday series (VIX, QQQ, ...): the last 5-min bar whose timestamp
    is <= the signal candle's timestamp. Both bars close at the same
    moment, which is when the signal is evaluated, so this is concurrent
    information, not future information.
  - daily series: only days strictly BEFORE the signal's date (today's
    daily bar isn't final until the close).
  - earnings: a report only counts once its reaction session has begun
    (pre-market report -> that day; after-close report -> next day).
The same code path serves backtests, dataset building and live
inference, so train-time and serve-time features can't drift apart.

Data is cached on disk under config.ML_V2_CACHE_DIR and MERGED with each
new fetch, so the intraday context history grows past Yahoo's ~60-day
5-min retention the longer you keep using it.
"""
import os
import pickle
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config

MARKET_TZ = ZoneInfo(config.MARKET_TIMEZONE)
NAN = float("nan")

# Intraday context series and the Yahoo ticker behind each.
INTRADAY_SERIES = {"VIX": "^VIX", "VXN": "^VXN", "VIX3M": "^VIX3M", "QQQ": "QQQ", "SPY": "SPY"}
DAILY_SERIES = {"VIX": "^VIX", "VXN": "^VXN", "QQQ": "QQQ", "SPY": "SPY"}

_INTRADAY_TTL = 60 * 60          # backtest: refetch market intraday at most hourly
_INTRADAY_TTL_LIVE = 4 * 60      # live: keep the tape fresh
_DAILY_TTL = 12 * 60 * 60
_EARNINGS_TTL = 3 * 24 * 60 * 60
_DAILY_LOOKBACK_DAYS = 420       # enough for a 200-day SMA plus slack


def _nan_safe(v) -> float:
    try:
        if v is None or pd.isna(v):
            return NAN
        return float(v)
    except (TypeError, ValueError):
        return NAN


def _ratio(a, b) -> float:
    """a / b - 1, NaN when either side is unusable."""
    a, b = _nan_safe(a), _nan_safe(b)
    if np.isnan(a) or np.isnan(b) or b == 0:
        return NAN
    return a / b - 1.0


def _regular_open_ts(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.normalize() + pd.Timedelta(hours=config.REGULAR_SESSION_OPEN_HOUR,
                                         minutes=config.REGULAR_SESSION_OPEN_MINUTE)


def minutes_since_open(ts) -> float:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize(MARKET_TZ)
    ts = ts.tz_convert(MARKET_TZ)
    return (ts - _regular_open_ts(ts)).total_seconds() / 60.0


# ═══════════════════════════════════════════════════════════════════
# Disk cache
# ═══════════════════════════════════════════════════════════════════

def _cache_path(name: str) -> str:
    os.makedirs(config.ML_V2_CACHE_DIR, exist_ok=True)
    safe = name.replace("^", "IDX_").replace("/", "_").replace("=", "_")
    return os.path.join(config.ML_V2_CACHE_DIR, f"{safe}.pkl")


def _cache_load(name: str):
    try:
        with open(_cache_path(name), "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _cache_save(name: str, payload: dict):
    # Written to a temp file then renamed, so a crash mid-write can never
    # leave a truncated pickle that poisons every later run.
    path = _cache_path(name)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    os.replace(tmp, path)


def _fresh_daily(cached) -> bool:
    """A daily-bar cache is only reusable on the SAME market date it was
    fetched: one fetched mid-session yesterday holds a partial bar that
    would otherwise be read today as yesterday's final close."""
    if cached is None:
        return False
    fetched = cached.get("fetched", 0)
    if time.time() - fetched >= _DAILY_TTL:
        return False
    return datetime.fromtimestamp(fetched, MARKET_TZ).date() == datetime.now(MARKET_TZ).date()


def _merge_frames(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if old is None or old.empty:
        return new
    if new is None or new.empty:
        return old
    out = pd.concat([old, new])
    # keep="last": a bar fetched again later (e.g. a partial live bar that
    # has since completed) replaces the earlier copy.
    return out.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# As-of lookup helpers
# ═══════════════════════════════════════════════════════════════════

class _AsOfSeries:
    """Sorted timestamp -> row lookup: 'last row at or before t'."""

    def __init__(self, df: pd.DataFrame):
        df = df.sort_values("timestamp").reset_index(drop=True)
        self.df = df
        self.ts = df["timestamp"].values.astype("datetime64[ns]").astype(np.int64) if len(df) else np.array([], dtype=np.int64)

    def idx_at(self, t: pd.Timestamp, max_age_minutes: float = 30.0) -> int:
        if not len(self.ts):
            return -1
        key = pd.Timestamp(t).tz_convert("UTC").value
        i = int(np.searchsorted(self.ts, key, side="right")) - 1
        if i < 0:
            return -1
        if (key - self.ts[i]) / 6e10 > max_age_minutes:
            return -1   # stale - e.g. an index that doesn't print pre-market
        return i

    def value(self, t, col: str = "close", max_age_minutes: float = 30.0) -> float:
        i = self.idx_at(t, max_age_minutes)
        return NAN if i < 0 else _nan_safe(self.df[col].iat[i])


class _DailySeries:
    """Daily bars with 'as of the prior session' lookups."""

    def __init__(self, df: pd.DataFrame):
        df = df.copy()
        df["date"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(MARKET_TZ).dt.date
        df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        c = df["close"]
        df["sma50"] = c.rolling(50, min_periods=40).mean()
        df["sma200"] = c.rolling(200, min_periods=150).mean()
        df["high20"] = df["high"].rolling(20, min_periods=10).max()
        df["vol20"] = df["volume"].rolling(20, min_periods=10).mean()
        prev_c = c.shift(1)
        tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_c).abs(),
                        (df["low"] - prev_c).abs()], axis=1).max(axis=1)
        df["atr14_pct"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=10).mean() / c
        df["ret5"] = c / c.shift(5) - 1
        df["ret20"] = c / c.shift(20) - 1
        self.df = df
        self.dates = np.array(df["date"].values, dtype="datetime64[D]")

    def prior_idx(self, d) -> int:
        """Index of the last session strictly before date d."""
        if not len(self.dates):
            return -1
        return int(np.searchsorted(self.dates, np.datetime64(d, "D"), side="left")) - 1

    def prior(self, d, col: str) -> float:
        i = self.prior_idx(d)
        return NAN if i < 0 else _nan_safe(self.df[col].iat[i])


# ═══════════════════════════════════════════════════════════════════
# Fetchers (all Yahoo, all keyless)
# ═══════════════════════════════════════════════════════════════════

def _fetch_intraday(ticker: str, days: int) -> pd.DataFrame:
    from data.yfinance_client import get_historical_candles
    today = datetime.now(MARKET_TZ)
    start = (today - timedelta(days=days)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")
    return get_historical_candles(ticker, config.CANDLE_INTERVAL_MINUTES, start, end, pause_seconds=0.5)


def _fetch_daily(ticker: str) -> pd.DataFrame:
    from data.yfinance_client import get_daily_candles
    today = datetime.now(MARKET_TZ)
    start = (today - timedelta(days=_DAILY_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    return get_daily_candles(ticker, start, today.strftime("%Y-%m-%d"))


def _fetch_breadth_closes(tickers) -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(list(tickers), period="2y", interval="1d", auto_adjust=False,
                      progress=False, group_by="ticker", threads=True)
    closes = {}
    for t in tickers:
        try:
            s = raw[t]["Close"].dropna()
            if len(s) > 60:
                closes[t] = s
        except Exception:
            continue
    df = pd.DataFrame(closes)
    df.index = pd.to_datetime(df.index).date
    return df


def _fetch_earnings(symbol: str) -> pd.DataFrame:
    """Earnings timestamps + EPS surprise. Empty for ETFs/indices."""
    import yfinance as yf
    try:
        e = yf.Ticker(symbol).get_earnings_dates(limit=24)
    except Exception:
        return pd.DataFrame(columns=["ts", "surprise"])
    if e is None or e.empty:
        return pd.DataFrame(columns=["ts", "surprise"])
    out = pd.DataFrame({
        "ts": pd.to_datetime(e.index),
        "surprise": pd.to_numeric(e.get("Surprise(%)"), errors="coerce").values,
    })
    if out["ts"].dt.tz is None:
        out["ts"] = out["ts"].dt.tz_localize(MARKET_TZ)
    else:
        out["ts"] = out["ts"].dt.tz_convert(MARKET_TZ)
    return out.sort_values("ts").reset_index(drop=True)


def _next_weekday(d):
    d = d + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _reaction_date(ts: pd.Timestamp):
    """First session that trades on this report: same day if it came out
    before the regular open, otherwise the next weekday."""
    if ts.hour * 60 + ts.minute < config.REGULAR_SESSION_OPEN_HOUR * 60 + config.REGULAR_SESSION_OPEN_MINUTE:
        return ts.date()
    return _next_weekday(ts.date())


# ═══════════════════════════════════════════════════════════════════
# The context object
# ═══════════════════════════════════════════════════════════════════

class MarketContext:
    """Lazily loads (and caches) everything, then answers feature lookups.

    live=False (backtest / dataset build): market intraday is loaded once
    per run over Yahoo's full 5-min window. live=True: the intraday tape
    is refreshed every few minutes via refresh_live().
    """

    def __init__(self, live: bool = False, verbose: bool = False):
        self.live = live
        self.verbose = verbose
        self._intraday = {}       # name -> _AsOfSeries
        self._daily = {}          # name -> _DailySeries (market + symbols)
        self._earnings = {}       # symbol -> DataFrame(ts, surprise, reaction)
        self._breadth = None      # pd.Series date -> pct above sma50
        self._market_loaded = False
        self._last_live_refresh = 0.0

    def _log(self, msg):
        if self.verbose:
            print(msg)

    # ---------- loading ----------

    def _load_intraday(self, name: str, ticker: str, ttl: float):
        cached = _cache_load(f"intraday_{ticker}")
        fresh = cached is not None and (time.time() - cached.get("fetched", 0)) < ttl
        df = cached["df"] if cached else None
        if not fresh:
            try:
                days = 5 if (self.live and df is not None and not df.empty) else 59
                new = _fetch_intraday(ticker, days)
                df = _merge_frames(df, new)
                _cache_save(f"intraday_{ticker}", {"fetched": time.time(), "df": df})
            except Exception as e:
                self._log(f"[context] intraday {ticker} fetch failed: {e}")
        if df is None or df.empty:
            self._intraday[name] = _AsOfSeries(pd.DataFrame(columns=["timestamp", "close"]))
            return
        df = df.copy()
        # Session VWAP of the regular session (used for QQQ/SPY "above VWAP").
        if name in ("QQQ", "SPY"):
            ts = df["timestamp"]
            reg = ts.dt.hour * 60 + ts.dt.minute >= (config.REGULAR_SESSION_OPEN_HOUR * 60
                                                      + config.REGULAR_SESSION_OPEN_MINUTE)
            day = ts.dt.date
            tpv = ((df["high"] + df["low"] + df["close"]) / 3 * df["volume"]).where(reg, 0.0)
            vol = df["volume"].where(reg, 0.0)
            cum_tpv = tpv.groupby(day).cumsum()
            cum_vol = vol.groupby(day).cumsum().replace(0, np.nan)
            df["vwap"] = (cum_tpv / cum_vol).where(reg)
        self._intraday[name] = _AsOfSeries(df)

    def _load_daily(self, name: str, ticker: str) -> _DailySeries:
        cached = _cache_load(f"daily_{ticker}")
        if _fresh_daily(cached):
            df = cached["df"]
        else:
            try:
                df = _fetch_daily(ticker)
                _cache_save(f"daily_{ticker}", {"fetched": time.time(), "df": df})
            except Exception as e:
                self._log(f"[context] daily {ticker} fetch failed: {e}")
                df = cached["df"] if cached else pd.DataFrame(
                    columns=["timestamp", "open", "high", "low", "close", "volume"])
        ser = _DailySeries(df) if not df.empty else None
        self._daily[name] = ser
        return ser

    def _load_breadth(self):
        cached = _cache_load("breadth_closes")
        if _fresh_daily(cached):
            closes = cached["df"]
        else:
            try:
                closes = _fetch_breadth_closes(config.ML_V2_BREADTH_UNIVERSE)
                _cache_save("breadth_closes", {"fetched": time.time(), "df": closes})
            except Exception as e:
                self._log(f"[context] breadth fetch failed: {e}")
                closes = cached["df"] if cached else pd.DataFrame()
        if closes is None or closes.empty:
            self._breadth = pd.Series(dtype=float)
            return
        sma = closes.rolling(50, min_periods=40).mean()
        above = (closes > sma).where(sma.notna())
        pct = above.mean(axis=1, skipna=True)
        pct.index = pd.to_datetime(pd.Index(pct.index)).date
        self._breadth = pct.sort_index()

    def ensure_market(self):
        if self._market_loaded:
            return
        ttl = _INTRADAY_TTL_LIVE if self.live else _INTRADAY_TTL
        for name, ticker in INTRADAY_SERIES.items():
            self._load_intraday(name, ticker, ttl)
        for name, ticker in DAILY_SERIES.items():
            self._load_daily(f"MKT_{name}", ticker)
        self._load_breadth()
        self._market_loaded = True
        self._last_live_refresh = time.time()

    def refresh_live(self):
        """Live loop hook: re-pull the intraday tape if it's gone stale."""
        if not self._market_loaded:
            self.ensure_market()
            return
        if time.time() - self._last_live_refresh < _INTRADAY_TTL_LIVE:
            return
        for name, ticker in INTRADAY_SERIES.items():
            self._load_intraday(name, ticker, _INTRADAY_TTL_LIVE)
        self._last_live_refresh = time.time()

    def ensure_symbol(self, symbol: str):
        if symbol in self._daily and symbol in self._earnings:
            return
        if symbol not in self._daily:
            self._load_daily(symbol, symbol)
        if symbol not in self._earnings:
            cached = _cache_load(f"earnings_{symbol}")
            if cached is not None and (time.time() - cached.get("fetched", 0)) < _EARNINGS_TTL:
                e = cached["df"]
            else:
                e = _fetch_earnings(symbol)
                try:
                    _cache_save(f"earnings_{symbol}", {"fetched": time.time(), "df": e})
                except Exception:
                    pass
            if not e.empty:
                e = e.copy()
                e["reaction"] = e["ts"].map(_reaction_date)
            self._earnings[symbol] = e

    # ---------- lookups ----------

    def asof(self, name: str, t, col: str = "close") -> float:
        """Intraday context value (e.g. asof('QQQ', ts)) at time t."""
        self.ensure_market()
        ser = self._intraday.get(name)
        return NAN if ser is None else ser.value(t, col)

    def _earnings_features(self, symbol: str, d) -> dict:
        e = self._earnings.get(symbol)
        out = {"earn_days_since": 120.0, "earn_days_to_next": 120.0,
               "earn_surprise": NAN, "earn_reaction_today": 0.0}
        if e is None or e.empty:
            return out
        past = e[e["reaction"] <= d]
        if not past.empty:
            last = past.iloc[-1]
            out["earn_days_since"] = float(min(120, (d - last["reaction"]).days))
            out["earn_surprise"] = float(np.clip(_nan_safe(last["surprise"]), -100, 100)) \
                if not np.isnan(_nan_safe(last["surprise"])) else NAN
            out["earn_reaction_today"] = 1.0 if last["reaction"] == d else 0.0
        future = e[e["reaction"] > d]
        if not future.empty:
            out["earn_days_to_next"] = float(min(120, (future.iloc[0]["reaction"] - d).days))
        return out

    def features(self, symbol: str, df: pd.DataFrame, direction: str) -> dict:
        """Every V2 context/catalyst feature for the signal candle df.iloc[-1].

        `df` is today's session slice up to and including the signal
        candle (the same slice the base models saw).
        """
        self.ensure_market()
        self.ensure_symbol(symbol)
        row = df.iloc[-1]
        ts = pd.Timestamp(row["timestamp"])
        d = ts.tz_convert(MARKET_TZ).date()
        close = _nan_safe(row["close"])
        sign = 1.0 if direction == "long" else -1.0
        mso = minutes_since_open(ts)
        t30 = ts - pd.Timedelta(minutes=30)

        f = {"minutes_since_open": mso}

        # --- volatility complex ---
        vix = self.asof("VIX", ts)
        vxn = self.asof("VXN", ts)
        vix3m = self.asof("VIX3M", ts)
        dv, dn = self._daily.get("MKT_VIX"), self._daily.get("MKT_VXN")
        f["vix"] = vix
        f["vix_vs_sma50"] = _ratio(vix, dv.prior(d, "sma50")) if dv else NAN
        f["vix_chg_day"] = _ratio(vix, dv.prior(d, "close")) if dv else NAN
        f["vix_chg_30m"] = _ratio(vix, self.asof("VIX", t30))
        f["vix_term"] = (vix / vix3m) if (vix3m and not np.isnan(vix3m) and not np.isnan(vix)) else NAN
        f["vxn_vs_sma50"] = _ratio(vxn, dn.prior(d, "sma50")) if dn else NAN
        f["vxn_chg_day"] = _ratio(vxn, dn.prior(d, "close")) if dn else NAN

        # --- index tape ---
        qqq = self.asof("QQQ", ts)
        spy = self.asof("SPY", ts)
        dq, ds = self._daily.get("MKT_QQQ"), self._daily.get("MKT_SPY")
        f["qqq_ret_day"] = _ratio(qqq, dq.prior(d, "close")) if dq else NAN
        f["qqq_ret_30m"] = _ratio(qqq, self.asof("QQQ", t30))
        f["qqq_vs_vwap"] = _ratio(qqq, self.asof("QQQ", ts, "vwap"))
        f["spy_ret_day"] = _ratio(spy, ds.prior(d, "close")) if ds else NAN
        f["qqq_vs_sma50_d"] = _ratio(dq.prior(d, "close"), dq.prior(d, "sma50")) if dq else NAN
        f["spy_vs_sma50_d"] = _ratio(ds.prior(d, "close"), ds.prior(d, "sma50")) if ds else NAN
        f["spy_vs_sma200_d"] = _ratio(ds.prior(d, "close"), ds.prior(d, "sma200")) if ds else NAN

        # --- breadth: % of universe above its 50-day SMA, prior close ---
        b = self._breadth
        if b is not None and len(b):
            prior = b[b.index < d]
            f["breadth_pct50"] = _nan_safe(prior.iloc[-1]) if len(prior) else NAN
            f["breadth_chg5"] = (_nan_safe(prior.iloc[-1]) - _nan_safe(prior.iloc[-6])) if len(prior) >= 6 else NAN
        else:
            f["breadth_pct50"] = f["breadth_chg5"] = NAN

        # --- symbol daily structure (prior sessions only) ---
        sd = self._daily.get(symbol)
        prev_close = sd.prior(d, "close") if sd else NAN
        f["sym_ret5_d"] = sd.prior(d, "ret5") if sd else NAN
        f["sym_ret20_d"] = sd.prior(d, "ret20") if sd else NAN
        f["sym_atr_pct_d"] = sd.prior(d, "atr14_pct") if sd else NAN
        f["sym_vs_sma50_d"] = _ratio(prev_close, sd.prior(d, "sma50")) if sd else NAN
        f["sym_dist_high20"] = _ratio(close, sd.prior(d, "high20")) if sd else NAN
        f["sym_dist_prev_high"] = _ratio(close, sd.prior(d, "high")) if sd else NAN

        # --- today's catalyst footprint: gap, move, volume, range ---
        reg_mask = df["timestamp"] >= _regular_open_ts(ts)
        reg = df[reg_mask]
        reg_open = _nan_safe(reg["open"].iat[0]) if len(reg) else close
        f["gap_pct"] = _ratio(reg_open, prev_close)
        f["sym_ret_day"] = _ratio(close, prev_close)
        f["sym_ret_since_open"] = _ratio(close, reg_open)
        f["rs_vs_qqq_day"] = f["sym_ret_day"] - f["qqq_ret_day"] \
            if not (np.isnan(f["sym_ret_day"]) or np.isnan(f["qqq_ret_day"])) else NAN
        past30 = df[df["timestamp"] <= t30]
        sym30 = _ratio(close, past30["close"].iat[-1]) if len(past30) else NAN
        f["rs_vs_qqq_30m"] = sym30 - f["qqq_ret_30m"] \
            if not (np.isnan(sym30) or np.isnan(f["qqq_ret_30m"])) else NAN
        vol20 = sd.prior(d, "vol20") if sd else NAN
        if len(reg) and vol20 and not np.isnan(vol20) and vol20 > 0:
            elapsed = min(1.0, max(mso + config.CANDLE_INTERVAL_MINUTES, 5) / 390.0)
            f["tod_rvol"] = float(reg["volume"].sum()) / (vol20 * elapsed)
        else:
            f["tod_rvol"] = NAN
        if len(reg):
            hi, lo = float(reg["high"].max()), float(reg["low"].min())
            f["day_range_pos"] = (close - lo) / (hi - lo) if hi > lo else 0.5
        else:
            f["day_range_pos"] = NAN

        f.update(self._earnings_features(symbol, d))

        # --- direction-aligned versions (positive = tailwind for THIS trade) ---
        def _al(v):
            return sign * v if not np.isnan(v) else NAN
        f["align_qqq_day"] = _al(f["qqq_ret_day"])
        f["align_qqq_30m"] = _al(f["qqq_ret_30m"])
        f["align_rs_day"] = _al(f["rs_vs_qqq_day"])
        f["align_gap"] = _al(f["gap_pct"])
        f["align_vix_chg_day"] = _al(-f["vix_chg_day"]) if not np.isnan(f["vix_chg_day"]) else NAN
        f["align_earn_surprise"] = _al(f["earn_surprise"]) if not np.isnan(f["earn_surprise"]) else NAN
        return f

    # ---------- rule-based risk vetoes ----------

    def risk_veto(self, f: dict, direction: str):
        """Hard market-risk vetoes (config.ML_V2_*). Returns a reason
        string if the trade must be skipped, else None."""
        if not config.ML_V2_VETO_ENABLED or direction != "long":
            return None
        vix, term, spike = f.get("vix", NAN), f.get("vix_term", NAN), f.get("vix_chg_day", NAN)
        if not np.isnan(vix) and vix >= config.ML_V2_VIX_MAX_LONG:
            return f"VIX {vix:.1f} >= {config.ML_V2_VIX_MAX_LONG}"
        if not np.isnan(term) and term >= config.ML_V2_VIX_TERM_MAX_LONG:
            return f"VIX/VIX3M {term:.2f} backwardation"
        if not np.isnan(spike) and spike >= config.ML_V2_VIX_DAY_SPIKE_MAX_LONG:
            return f"VIX +{spike*100:.0f}% today"
        return None


_shared = {}
_mode = {"live": False}


def set_live_mode(flag: bool):
    """Called once by a live runner so model calls get the live context
    (fresh tape + news overlay) instead of the backtest one."""
    _mode["live"] = bool(flag)


def is_live_mode() -> bool:
    return _mode["live"]


def reset_context(live: bool = None):
    """Drop the in-memory context (a live runner calls this at the start
    of each new trading day so prior-close lookups pick up yesterday's
    final bars)."""
    if live is None:
        live = _mode["live"]
    _shared.pop("live" if live else "backtest", None)


def get_context(live: bool = None) -> MarketContext:
    """Process-wide shared context (one per mode), so every model call
    in a run reuses the same loaded data."""
    if live is None:
        live = _mode["live"]
    key = "live" if live else "backtest"
    if key not in _shared:
        _shared[key] = MarketContext(live=live)
    return _shared[key]
