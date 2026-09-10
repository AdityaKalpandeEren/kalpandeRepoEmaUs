import pandas as pd


def add_ema(df: pd.DataFrame, period: int, column: str = "close") -> pd.DataFrame:
    df[f"ema_{period}"] = df[column].ewm(span=period, adjust=False).mean()
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Session VWAP - cumulative from the start of the (intraday-only) dataframe."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (typical_price * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum().replace(0, pd.NA)
    df["vwap"] = cum_tp_vol / cum_vol
    return df


def add_avg_volume(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Average of the PRECEDING `period` candles (shifted so it excludes the current candle)."""
    df[f"avg_vol_{period}"] = df["volume"].rolling(window=period, min_periods=1).mean().shift(1)
    return df
