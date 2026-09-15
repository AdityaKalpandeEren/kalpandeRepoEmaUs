"""
Shared trade-outcome simulation, used by BOTH the historical backtester
(backtest/) and the live paper trader (paper_trading/) so "accuracy"
means the exact same thing in both places: same entry/stop/target math,
same rule for deciding whether a trade's target or its stop was hit
first, same definition of a win/loss/scratch.

Fill-price rule (applies in both backtest and live paper trading):
a signal is generated from a CLOSED candle, so the earliest you could
realistically act on a Telegram alert is the OPEN of the candle right
after it - not that candle's own close. Both this module's callers use
that next-candle-open as the simulated entry price, and re-derive the
target from that real fill price (the stop-loss from the signal itself
is kept, since it's a structural level, not a price you "chase").
"""
from dataclasses import dataclass
import pandas as pd


@dataclass
class TradeResult:
    symbol: str
    strategy: str            # "EMA_CROSS" | "VWAP_RETEST" | "VWAP_BROAD_TEST"
    entry_time: object
    exit_time: object
    entry: float
    stop_loss: float
    target: float
    exit_price: float
    outcome: str              # "TARGET" | "STOP" | "EOD_SQUAREOFF" | "NO_FILL"
    r_multiple: float
    pnl_pct: float
    candles_held: int


def simulate_forward(day_df: pd.DataFrame, entry_idx: int, entry: float,
                      stop_loss: float, target: float):
    """
    Walk forward from candle `entry_idx` (inclusive - this is the fill
    candle itself, so its own high/low can already trigger an exit) to
    the end of `day_df` (one trading day's candles) to see which level
    is hit first.

    Conservative same-candle rule: if a single candle's range touches
    BOTH the stop and the target, the STOP is assumed to have been hit
    first (worst case) - since plain OHLC data doesn't tell us the real
    intrabar sequence. This makes the reported win rate a lower bound,
    not an optimistic one.

    Never hit either level by end of the session -> square off at the
    day's last close (matches how a manual trader following these
    alerts would actually behave: no overnight NSE positions).

    Returns (exit_time, exit_price, outcome, exit_idx).
    """
    for i in range(entry_idx, len(day_df)):
        row = day_df.iloc[i]
        hit_stop = row["low"] <= stop_loss
        hit_target = row["high"] >= target
        if hit_stop:
            return row["timestamp"], stop_loss, "STOP", i
        if hit_target:
            return row["timestamp"], target, "TARGET", i

    last = day_df.iloc[-1]
    return last["timestamp"], last["close"], "EOD_SQUAREOFF", len(day_df) - 1


def build_trade_result(symbol, strategy, entry_time, entry, stop_loss, target,
                        exit_time, exit_price, outcome, candles_held) -> TradeResult:
    risk = entry - stop_loss
    pnl = exit_price - entry
    r_multiple = pnl / risk if risk > 0 else 0.0
    pnl_pct = pnl / entry * 100 if entry else 0.0
    return TradeResult(
        symbol=symbol,
        strategy=strategy,
        entry_time=entry_time,
        exit_time=exit_time,
        entry=round(entry, 2),
        stop_loss=round(stop_loss, 2),
        target=round(target, 2),
        exit_price=round(exit_price, 2),
        outcome=outcome,
        r_multiple=round(r_multiple, 2),
        pnl_pct=round(pnl_pct, 2),
        candles_held=candles_held,
    )
