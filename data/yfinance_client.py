import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

import config

MARKET_TZ = ZoneInfo(config.MARKET_TIMEZONE)


def get_intraday_candles(symbol: str, interval_minutes: int) -> pd.DataFrame:
    """Fetch today's intraday candles for one symbol via Yahoo Finance.

    Works for US stocks (AAPL, MSFT), indices (^GSPC, ^IXIC, ^DJI, ^VIX),
    and futures continuous contracts (ES=F, NQ=F, YM=F, CL=F, GC=F, SI=F).
    No API key required. Yahoo's intraday history is limited to the last
    60 days for 5m/15m bars (1m bars only cover the last 7 days) -
    period="5d" is fetched for enough candles to warm up EMA/avg-volume
    early in the session, but the result is filtered down to just
    today (pre-market + regular + after-hours, since prepost=True) -
    VWAP and avg-volume need to reset every session, not accumulate
    across the whole 5-day lookback window.
    """
    interval = f"{interval_minutes}m"
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="5d", interval=interval, auto_adjust=False, prepost=True)

    if df is None or df.empty:
        raise ValueError(
            f"No data returned for '{symbol}'. Check the ticker: NASDAQ/NYSE "
            f"stocks as-is (AAPL, MSFT, NVDA), indices with a caret prefix "
            f"(^GSPC, ^IXIC, ^DJI), futures continuous contracts with '=F' "
            f"(ES=F, NQ=F, CL=F, GC=F). Also possible: outside market hours "
            f"Yahoo sometimes returns nothing at all rather than yesterday's data."
        )

    df = df.reset_index()
    time_col = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(columns={
        time_col: "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(MARKET_TZ)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(MARKET_TZ)

    today = datetime.now(MARKET_TZ).date()
    df = df[df["timestamp"].dt.date == today].reset_index(drop=True)

    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


# Yahoo's real intraday lookback limits (approx, as of current yfinance):
#   1m               -> last 30 days of history total
#   2m/5m/15m/30m/90m -> last 60 days of history total
#   60m/1h           -> last 730 days of history total
# We request in smaller chunks than even those limits, below, because a
# single request spanning weeks/months is far more likely to get
# rate-limited or bot-blocked by Yahoo than several smaller ones - this
# is exactly where the "dependency" problems people hit with historical
# yfinance data usually surface (see get_historical_candles' docstring).
_CHUNK_DAYS = {1: 6, 2: 15, 5: 15, 15: 15, 30: 15, 60: 45, 90: 15}


def _chunk_ranges(start: datetime, end: datetime, chunk_days: int):
    cur_start = start
    while cur_start < end:
        cur_end = min(end, cur_start + timedelta(days=chunk_days))
        yield cur_start, cur_end
        cur_start = cur_end


def get_historical_candles(symbol: str, interval_minutes: int, start_date: str, end_date: str,
                            max_retries: int = 3, pause_seconds: float = 1.5) -> pd.DataFrame:
    """
    Fetch MULTI-DAY historical candles for backtesting - unlike
    get_intraday_candles above, which only ever returns TODAY's candles.

    This is where a yfinance dependency/network problem will actually
    surface, if you have one: `pip install yfinance` also pulls in
    curl_cffi (a compiled libcurl wrapper Yahoo now requires to avoid
    its own bot-detection blocking plain `requests` sessions). A single
    "give me today" call rarely trips that layer; a historical range
    request - many more backend calls, sometimes fetched in bulk - is
    exactly what does. If a call below fails repeatedly and the error
    message mentions curl_cffi, an SSL/TLS error, "requires curl_cffi
    session", or a version conflict, that's a dependency/environment
    problem, not a bug in this bot - see README "Testing on historical
    data" for the fix checklist. A plain "no data" with no exception
    just means Yahoo has nothing for that symbol/date/interval combo
    (e.g. asking for 1-minute bars from 6 months ago - Yahoo simply
    doesn't keep that).
    """
    chunk_days = _CHUNK_DAYS.get(interval_minutes, 15)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)  # make end_date inclusive

    interval = f"{interval_minutes}m"
    ticker = yf.Ticker(symbol)
    frames = []

    for chunk_start, chunk_end in _chunk_ranges(start_dt, end_dt, chunk_days):
        chunk_df = None
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                chunk_df = ticker.history(
                    start=chunk_start.strftime("%Y-%m-%d"),
                    end=chunk_end.strftime("%Y-%m-%d"),
                    interval=interval,
                    auto_adjust=False,
                    prepost=True,
                )
                last_error = None
                break
            except Exception as e:
                last_error = e
                time.sleep(pause_seconds * attempt)  # backoff: 1.5s, 3s, 4.5s...

        if last_error is not None:
            raise RuntimeError(
                f"Failed to fetch {symbol} {chunk_start.date()}..{chunk_end.date()} after "
                f"{max_retries} attempts. Underlying error: {last_error!r}. "
                f"If that mentions curl_cffi / SSL / TLS / 'requires curl_cffi session', this "
                f"is a yfinance dependency problem - see README 'Testing on historical data' "
                f"for the fix. If it's a plain timeout, you may just be rate-limited - wait a "
                f"few minutes and try again, or fetch a smaller date range."
            ) from last_error

        if chunk_df is not None and not chunk_df.empty:
            frames.append(chunk_df)
        time.sleep(pause_seconds)  # be polite between chunks even on success

    if not frames:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    combined = pd.concat(frames)
    combined = combined.reset_index()
    time_col = "Datetime" if "Datetime" in combined.columns else "Date"
    combined = combined.rename(columns={
        time_col: "timestamp", "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })

    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    if combined["timestamp"].dt.tz is None:
        combined["timestamp"] = combined["timestamp"].dt.tz_localize(MARKET_TZ)
    else:
        combined["timestamp"] = combined["timestamp"].dt.tz_convert(MARKET_TZ)

    combined = combined.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        combined[col] = pd.to_numeric(combined[col])
    return combined[["timestamp", "open", "high", "low", "close", "volume"]]
