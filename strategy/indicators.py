import pandas as pd

import config


def add_ema(df: pd.DataFrame, period: int, column: str = "close") -> pd.DataFrame:
    df[f"ema_{period}"] = df[column].ewm(span=period, adjust=False).mean()
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Session VWAP - cumulative from the start of the (intraday-only) dataframe.

    Timeline this resets on: by default (config.VWAP_REGULAR_SESSION_ONLY =
    false), it accumulates from the very FIRST candle yfinance_client.py
    handed us for "today" - which, because MARKET_OPEN_HOUR=4 and
    prepost=True, is 4:00 AM ET pre-market, not the 9:30 AM regular
    open. So by default, pre-market price/volume IS baked into VWAP.

    Set config.VWAP_REGULAR_SESSION_ONLY=true to instead reset VWAP at
    the 9:30 AM regular-session open (the more common "textbook"
    definition most traders mean by "VWAP") - pre-market/after-hours
    candles get vwap=NaN, which check_signal/check_vwap_retest/
    check_vwap_broad_TEST already treat as "no signal", so this simply
    means no EMA/VWAP alerts fire before 9:30 even though the bot keeps
    scanning (and paper trading keeps tracking) from 4:00 AM onward.
    """
    df = df.copy()
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = typical_price * df["volume"]

    if config.VWAP_REGULAR_SESSION_ONLY and not df.empty:
        session_date = df["timestamp"].iloc[0].date()
        tz = df["timestamp"].iloc[0].tzinfo
        session_start = pd.Timestamp(
            year=session_date.year, month=session_date.month, day=session_date.day,
            hour=config.REGULAR_SESSION_OPEN_HOUR, minute=config.REGULAR_SESSION_OPEN_MINUTE,
            tz=tz,
        )
        in_regular_session = df["timestamp"] >= session_start
        tp_vol = tp_vol.where(in_regular_session, 0.0)
        vol_for_vwap = df["volume"].where(in_regular_session, 0.0)
    else:
        in_regular_session = None
        vol_for_vwap = df["volume"]

    cum_tp_vol = tp_vol.cumsum()
    cum_vol = vol_for_vwap.cumsum().replace(0, pd.NA)
    df["vwap"] = cum_tp_vol / cum_vol
    if in_regular_session is not None:
        df.loc[~in_regular_session, "vwap"] = pd.NA
    return df


def add_avg_volume(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Average of the PRECEDING `period` candles (shifted so it excludes the current candle)."""
    df[f"avg_vol_{period}"] = df["volume"].rolling(window=period, min_periods=1).mean().shift(1)
    return df
