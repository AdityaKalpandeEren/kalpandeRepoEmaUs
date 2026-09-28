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


# ═══════════════════════════════════════════════════════════════════
# Research-layer indicators. Added for the multi-strategy research
# engine (strategy/strategies.py); nothing above this line changed,
# and none of the live signal functions call anything below.
# ═══════════════════════════════════════════════════════════════════

def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """True-range ATR (Wilder). Used for volatility-scaled stops and
    for the regime classifier's volatility axis."""
    df = df.copy()
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["tr"] = tr
    df[f"atr_{period}"] = tr.ewm(alpha=1 / period, adjust=False, min_periods=1).mean()
    return df


def add_atr_percentile(df: pd.DataFrame, period: int = 14, lookback: int = 50) -> pd.DataFrame:
    """Where the current ATR sits within its own recent distribution,
    0..1. Absolute ATR isn't comparable across symbols or price levels;
    its percentile is, which is what makes a single volatility
    threshold meaningful across a whole watchlist."""
    df = df.copy()
    col = f"atr_{period}"
    if col not in df.columns:
        df = add_atr(df, period)
    df["atr_pct_rank"] = df[col].rolling(lookback, min_periods=5).rank(pct=True)
    return df


def add_ema_slope(df: pd.DataFrame, period: int, lookback: int = 3) -> pd.DataFrame:
    """EMA slope normalised by price, so it's comparable across
    symbols. Expressed as fractional change per candle."""
    df = df.copy()
    col = f"ema_{period}"
    if col not in df.columns:
        df = add_ema(df, period)
    df[f"ema_{period}_slope"] = (df[col] - df[col].shift(lookback)) / (df[col].shift(lookback) * lookback)
    return df


def add_relative_volume(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Current candle volume vs the average of the PRECEDING `period`
    candles. Shifted, so the current candle never inflates its own
    baseline."""
    df = df.copy()
    baseline = df["volume"].rolling(period, min_periods=1).mean().shift(1)
    df["rvol"] = df["volume"] / baseline.replace(0, pd.NA)
    return df


def add_rsi(df: pd.DataFrame, period: int = 2, column: str = "close") -> pd.DataFrame:
    """Wilder's RSI. Added specifically for Larry Connors' RSI(2)
    mean-reversion model (see strategies.py::model_k_rsi2_reversion) -
    the classic short-period RSI, not the standard 14-period trend
    oscillator."""
    df = df.copy()
    delta = df[column].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
    df.loc[avg_loss == 0, f"rsi_{period}"] = 100.0
    return df


def add_vwap_features(df: pd.DataFrame, dev_lookback: int = 20) -> pd.DataFrame:
    """VWAP distance in standard deviations (a z-score, so it's
    comparable across symbols) plus VWAP slope. `add_vwap` must have
    run first."""
    df = df.copy()
    if "vwap" not in df.columns:
        df = add_vwap(df)
    vwap = pd.to_numeric(df["vwap"], errors="coerce")
    diff = df["close"] - vwap
    rolling_sd = diff.rolling(dev_lookback, min_periods=5).std()
    df["vwap_dist_pct"] = diff / vwap
    df["vwap_z"] = diff / rolling_sd.replace(0, pd.NA)
    df["vwap_slope"] = (vwap - vwap.shift(3)) / (vwap.shift(3) * 3)
    return df


def add_structure(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """Rolling structural highs/lows plus higher-high / higher-low
    flags. The rolling extremes are SHIFTED by one candle so the
    current candle's own high can't be the level it is being tested
    against - that would be circular and would silently leak the
    future into a breakout test."""
    df = df.copy()
    df["struct_high"] = df["high"].rolling(lookback, min_periods=2).max().shift(1)
    df["struct_low"] = df["low"].rolling(lookback, min_periods=2).min().shift(1)
    df["hh"] = df["struct_high"] > df["struct_high"].shift(lookback)
    df["hl"] = df["struct_low"] > df["struct_low"].shift(lookback)
    df["lh"] = df["struct_high"] < df["struct_high"].shift(lookback)
    df["ll"] = df["struct_low"] < df["struct_low"].shift(lookback)
    return df


def add_opening_range(df: pd.DataFrame, minutes: int, candle_minutes: int) -> pd.DataFrame:
    """Opening-range high/low from the first `minutes` of the session's
    candles, broadcast to every row of the day. Used by the ORB model."""
    df = df.copy()
    n = max(1, minutes // candle_minutes)
    if len(df) < n:
        df["or_high"] = pd.NA
        df["or_low"] = pd.NA
        return df
    df["or_high"] = df["high"].iloc[:n].max()
    df["or_low"] = df["low"].iloc[:n].min()
    # The opening range isn't known until it completes; before that it
    # must not be visible, or the ORB model would be trading on data
    # from its own future.
    df.loc[df.index[:n], "or_high"] = pd.NA
    df.loc[df.index[:n], "or_low"] = pd.NA
    return df


def enrich(df: pd.DataFrame, ema_fast: int, ema_slow: int, atr_period: int,
           vol_period: int, struct_lookback: int, or_minutes: int,
           candle_minutes: int) -> pd.DataFrame:
    """One pass that attaches every research feature. Called once per
    candle-slice by the strategy engine so each model doesn't recompute
    the same columns."""
    df = add_ema(df, ema_fast)
    df = add_ema(df, ema_slow)
    df = add_ema_slope(df, ema_fast)
    df = add_ema_slope(df, ema_slow)
    df = add_ema(df, config.EMA_STACK_FAST)
    df = add_ema(df, config.EMA_STACK_MID)
    df = add_ema(df, config.EMA_STACK_SLOW)
    df = add_vwap(df)
    df = add_vwap_features(df)
    df = add_rsi(df, config.RSI2_PERIOD)
    df = add_atr(df, atr_period)
    df = add_atr_percentile(df, atr_period)
    df = add_relative_volume(df, vol_period)
    df = add_structure(df, struct_lookback)
    df = add_opening_range(df, or_minutes, candle_minutes)
    return df
