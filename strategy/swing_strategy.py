"""
SWING_DAYS_STR - swing/positional breakout strategy.

Separate engine from strategy/strategies.py (the intraday, session-VWAP
research models). This one runs on DAILY candles and holds a position
for days, not minutes - see config.py's "SWING/POSITIONAL RESEARCH"
block for why it isn't folded into the intraday evaluate_all() loop.

Entry (long only, matching the rest of this version - cash equity, no
shorting): the 30/50/60-day EMAs are stacked bullish (fast > mid > slow,
an established multi-week uptrend, not a fast blip), price breaks out
above its own recent (SWING_STRUCT_LOOKBACK-day) high, that breakout
day's volume is well above its own trailing average (real demand behind
the move, not a quiet drift through the level), and two filters lifted
from Minervini's published Trend Template: price is meaningfully above
its 52-week low (not a dead-cat bounce) and within reach of its 52-week
high (participating in the stock's own best form, not lagging it).

Exit: fixed +8% target / -2% stop from the fill price (config
SWING_TARGET_PCT / SWING_STOP_PCT) - a 4:1 reward:risk by construction,
not the ATR-scaled convention the intraday models share, because this
was asked for as a specific percentage-based swing exit.
"""
from dataclasses import dataclass
from typing import Optional

import pandas as pd

import config
from strategy.indicators import add_ema, add_avg_volume, add_structure


@dataclass
class SwingSignal:
    symbol: str
    entry: float
    stop_loss: float
    target: float
    reason: str
    signal_date: object


def _f(v, default=0.0) -> float:
    try:
        if v is None or pd.isna(v):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def enrich_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Attach every feature SWING_DAYS_STR needs to a daily-bar
    dataframe (columns: timestamp, open, high, low, close, volume)."""
    df = add_ema(df, config.SWING_EMA_FAST)
    df = add_ema(df, config.SWING_EMA_MID)
    df = add_ema(df, config.SWING_EMA_SLOW)
    df = add_avg_volume(df, config.SWING_VOLUME_AVG_PERIOD)
    df = add_structure(df, config.SWING_STRUCT_LOOKBACK)
    df = df.copy()
    lookback_252 = min(252, len(df))
    df["low_52w"] = df["low"].rolling(lookback_252, min_periods=20).min().shift(1)
    df["high_52w"] = df["high"].rolling(lookback_252, min_periods=20).max().shift(1)
    return df


def check_swing_breakout(symbol: str, df: pd.DataFrame) -> Optional[SwingSignal]:
    """Evaluate the LAST row of an already-enriched daily dataframe
    (see enrich_daily) for a SWING_DAYS_STR entry. `df` must only
    contain candles up to and including the signal day - no lookahead."""
    if len(df) < config.SWING_MIN_WARMUP_DAYS:
        return None
    row = df.iloc[-1]

    e_fast = _f(row.get(f"ema_{config.SWING_EMA_FAST}"))
    e_mid = _f(row.get(f"ema_{config.SWING_EMA_MID}"))
    e_slow = _f(row.get(f"ema_{config.SWING_EMA_SLOW}"))
    close = _f(row.get("close"))
    struct_hi = _f(row.get("struct_high"))
    avg_vol_col = f"avg_vol_{config.SWING_VOLUME_AVG_PERIOD}"
    avg_vol = _f(row.get(avg_vol_col))
    volume = _f(row.get("volume"))
    low_52w = _f(row.get("low_52w"))
    high_52w = _f(row.get("high_52w"))

    if e_fast <= 0 or e_mid <= 0 or e_slow <= 0 or struct_hi <= 0 or avg_vol <= 0 or low_52w <= 0 or high_52w <= 0:
        return None

    stacked = e_fast > e_mid > e_slow
    broke_out = close > struct_hi
    vol_ratio = volume / avg_vol
    volume_confirmed = vol_ratio >= config.SWING_MIN_VOLUME_RATIO
    above_52w_low = close >= low_52w * (1 + config.SWING_MIN_ABOVE_52W_LOW_PCT)
    near_52w_high = close >= high_52w * (1 - config.SWING_MAX_BELOW_52W_HIGH_PCT)

    if not (stacked and broke_out and volume_confirmed and above_52w_low and near_52w_high):
        return None

    entry = close
    stop_loss = entry * (1 - config.SWING_STOP_PCT)
    target = entry * (1 + config.SWING_TARGET_PCT)
    reason = (
        f"EMA{config.SWING_EMA_FAST}/{config.SWING_EMA_MID}/{config.SWING_EMA_SLOW} stacked, "
        f"broke {struct_hi:.2f} on {vol_ratio:.1f}x volume, "
        f"{(close/low_52w-1)*100:.0f}% above 52w low, {(1-close/high_52w)*100:.0f}% below 52w high"
    )
    return SwingSignal(
        symbol=symbol,
        entry=round(entry, 4),
        stop_loss=round(stop_loss, 4),
        target=round(target, 4),
        reason=reason,
        signal_date=row.get("timestamp"),
    )
