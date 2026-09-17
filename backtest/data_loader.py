"""
Fetches historical intraday candles across a date range and splits them
back into per-trading-day sessions, since the bot's indicators (VWAP,
EMA, avg-volume) all reset every session - exactly like the live bot
only ever seeing "today's" candles.
"""
import pandas as pd

from data.yfinance_client import get_historical_candles


def load_symbol_history(symbol: str, interval_minutes: int, from_date: str, to_date: str) -> pd.DataFrame:
    """
    Returns one combined DataFrame (columns: timestamp, open, high, low,
    close, volume) spanning every trading day in [from_date, to_date].
    Use group_by_day() to split it before running strategy checks.

    Yahoo's own intraday retention limits how far back you can go (see
    data/yfinance_client.py's _CHUNK_DAYS comment): roughly the last 30
    days for 1-minute bars, 60 days for 5/15/30-minute bars. Asking for
    more than that just returns fewer days than requested, not an error.
    """
    return get_historical_candles(symbol, interval_minutes, from_date, to_date)


def group_by_day(df: pd.DataFrame) -> dict:
    """{date: day_df} - each day_df is one trading session's candles,
    in the same shape the live bot's per-symbol df is."""
    if df.empty:
        return {}
    df = df.copy()
    df["_date"] = df["timestamp"].dt.date
    return {d: g.drop(columns="_date").reset_index(drop=True) for d, g in df.groupby("_date")}
