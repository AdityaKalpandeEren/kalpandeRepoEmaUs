from dataclasses import dataclass
from typing import Optional
import pandas as pd

import config
from strategy.indicators import add_ema, add_vwap, add_avg_volume


@dataclass
class Signal:
    symbol: str
    entry: float
    stop_loss: float
    target: float
    reason: str
    candle_time: pd.Timestamp


def check_signal(symbol: str, df: pd.DataFrame) -> Optional[Signal]:
    if len(df) < config.EMA_PERIOD + 2:
        return None  # not enough candles yet to trust the EMA/avg-volume

    df = add_ema(df, config.EMA_PERIOD)
    df = add_vwap(df)
    df = add_avg_volume(df, config.VOLUME_AVG_PERIOD)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    ema_col = f"ema_{config.EMA_PERIOD}"
    avg_vol_col = f"avg_vol_{config.VOLUME_AVG_PERIOD}"

    if pd.isna(last[avg_vol_col]) or last[avg_vol_col] == 0:
        return None

    crossed_above_ema = prev["close"] <= prev[ema_col] and last["close"] > last[ema_col]
    above_vwap = last["close"] > last["vwap"]
    high_volume = last["volume"] > config.VOLUME_MULTIPLIER * last[avg_vol_col]

    if not (crossed_above_ema and above_vwap and high_volume):
        return None

    entry = last["close"]
    stop_loss = df["low"].iloc[-3:].min()  # recent swing low as structural stop
    risk = entry - stop_loss
    if risk <= 0:
        return None

    risk_pct = risk / entry
    if risk_pct > config.MAX_RISK_PCT:
        return None  # stop is too wide for this setup - skip rather than force it

    target = entry + risk * config.RISK_REWARD_RATIO

    reason = (
        f"Crossed above EMA{config.EMA_PERIOD}, above VWAP, "
        f"volume {last['volume']:.0f} vs avg {last[avg_vol_col]:.0f} "
        f"({last['volume'] / last[avg_vol_col]:.1f}x)"
    )

    return Signal(
        symbol=symbol,
        entry=round(entry, 2),
        stop_loss=round(stop_loss, 2),
        target=round(target, 2),
        reason=reason,
        candle_time=last["timestamp"],
    )


def evaluate(symbol: str, df: pd.DataFrame) -> dict:
    """Like check_signal, but always returns the current indicator readout
    (even when nothing triggers) so you can see the bot is actually working
    and how close/far the current candle is from a signal."""
    if len(df) < config.EMA_PERIOD + 2:
        return {"symbol": symbol, "status": f"warming up ({len(df)}/{config.EMA_PERIOD + 2} candles)"}

    df = add_ema(df, config.EMA_PERIOD)
    df = add_vwap(df)
    df = add_avg_volume(df, config.VOLUME_AVG_PERIOD)

    last = df.iloc[-1]
    prev = df.iloc[-2]
    ema_col = f"ema_{config.EMA_PERIOD}"
    avg_vol_col = f"avg_vol_{config.VOLUME_AVG_PERIOD}"

    if pd.isna(last[avg_vol_col]) or last[avg_vol_col] == 0:
        return {"symbol": symbol, "status": "no avg-volume data yet"}

    crossed_above_ema = prev["close"] <= prev[ema_col] and last["close"] > last[ema_col]
    above_vwap = last["close"] > last["vwap"]
    vol_ratio = last["volume"] / last[avg_vol_col]

    return {
        "symbol": symbol,
        "status": "OK",
        "price": round(last["close"], 2),
        "ema": round(last[ema_col], 2),
        "vwap": round(last["vwap"], 2),
        "ema_cross": crossed_above_ema,
        "above_vwap": above_vwap,
        "vol_ratio": round(vol_ratio, 2),
        "vol_needed": config.VOLUME_MULTIPLIER,
    }


@dataclass
class RetestSignal:
    symbol: str
    entry: float
    stop_loss: float
    target: float
    vwap: float
    buy_volume: float
    sell_volume: float
    buy_share_pct: float
    aggressor: str
    candle_time: pd.Timestamp


def _candle_aggressor(row) -> tuple:
    """Splits one candle's volume into buy/sell using the geometric method:
    where close sits inside the bar's high-low range. This is the same
    fallback formula the 'Geometric' engine in the Volume Footprint
    indicator uses (buyShare = (close - low) / (high - low)) - plain
    OHLCV arithmetic, not the footprint/intrabar data that needs a
    premium TradingView plan."""
    bar_range = row["high"] - row["low"]
    buy_share = (row["close"] - row["low"]) / bar_range if bar_range > 0 else 0.5
    buy_volume = row["volume"] * buy_share
    sell_volume = row["volume"] * (1 - buy_share)
    aggressor = "BUY" if buy_share > 0.55 else "SELL" if buy_share < 0.45 else "NEUTRAL"
    return buy_volume, sell_volume, buy_share, aggressor


def check_vwap_retest(symbol: str, df: pd.DataFrame) -> Optional[RetestSignal]:
    """VWAP-retest-for-long: price has been trading above VWAP (established
    uptrend context for the session), the current candle dips down to
    touch/test VWAP as support, and closes back above it on a bullish
    candle. Uses the US session VWAP (cumulative from market open), unlike
    the BTC bot's rolling 24h VWAP - crypto has no session reset, NSE does."""
    if len(df) < config.RETEST_TREND_LOOKBACK + 2:
        return None

    df = add_vwap(df)

    last = df.iloc[-1]
    trend_window = df.iloc[-(config.RETEST_TREND_LOOKBACK + 1):-1]

    above_count = (trend_window["close"] > trend_window["vwap"]).sum()
    established_uptrend = above_count >= config.RETEST_MIN_CANDLES_ABOVE
    if not established_uptrend:
        return None

    if pd.isna(last["vwap"]):
        return None

    touch_buffer = last["vwap"] * config.RETEST_TOUCH_BUFFER_PCT
    touched_vwap = last["low"] <= last["vwap"] + touch_buffer
    held_above = last["close"] > last["vwap"]
    bullish_candle = last["close"] > last["open"]

    if not (touched_vwap and held_above and bullish_candle):
        return None

    entry = last["close"]
    stop_loss = last["low"]
    risk = entry - stop_loss
    if risk <= 0:
        return None
    target = entry + risk * config.RISK_REWARD_RATIO

    buy_volume, sell_volume, buy_share, aggressor = _candle_aggressor(last)

    return RetestSignal(
        symbol=symbol,
        entry=round(entry, 2),
        stop_loss=round(stop_loss, 2),
        target=round(target, 2),
        vwap=round(last["vwap"], 2),
        buy_volume=round(buy_volume, 2),
        sell_volume=round(sell_volume, 2),
        buy_share_pct=round(buy_share * 100, 1),
        aggressor=aggressor,
        candle_time=last["timestamp"],
    )
