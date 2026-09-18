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


# ═══════════════════════════════════════════════════════════════════
# DIRECTIONAL SIMULATION (research engine)
# The functions above are long-only: simulate_forward assumes the
# target sits above entry and the stop below it. Shorts invert that,
# so they need their own path rather than a flag bolted onto the old
# one. Nothing above this line changed - the live paper trader and the
# original backtest still call the long-only functions unchanged.
# ═══════════════════════════════════════════════════════════════════
import config as _config


@dataclass
class ResearchTrade:
    symbol: str
    strategy: str
    direction: str
    regime: str
    trend_strength: float
    score: float
    entry_time: object
    exit_time: object
    entry: float
    stop_loss: float
    target: float
    exit_price: float
    outcome: str
    r_multiple: float
    pnl_pct: float
    candles_held: int
    risk_pct: float          # stop distance as a fraction of entry
    position_size: float     # shares implied by config risk sizing
    pnl_currency: float


def simulate_forward_directional(day_df, entry_idx: int, direction: str,
                                  entry: float, stop_loss: float, target: float):
    """Directional version of simulate_forward.

    Same conservative rule as the long-only version: if one candle's
    range touches both the stop and the target, the STOP is taken
    first, because plain OHLC cannot tell us the intrabar order. For a
    short that means the high (stop) is checked before the low
    (target).

    Returns (exit_time, exit_price, outcome, exit_idx).
    """
    for i in range(entry_idx, len(day_df)):
        row = day_df.iloc[i]
        if direction == "long":
            if row["low"] <= stop_loss:
                return row["timestamp"], stop_loss, "STOP", i
            if row["high"] >= target:
                return row["timestamp"], target, "TARGET", i
        else:
            if row["high"] >= stop_loss:
                return row["timestamp"], stop_loss, "STOP", i
            if row["low"] <= target:
                return row["timestamp"], target, "TARGET", i

    last = day_df.iloc[-1]
    return last["timestamp"], last["close"], "EOD_SQUAREOFF", len(day_df) - 1


def _apply_costs(entry: float, exit_price: float, direction: str):
    """Slippage + commission, charged on BOTH sides, against the trade.
    Entry fills worse than shown and exits fill worse than shown - the
    realistic direction, not the flattering one."""
    cost_frac = (_config.SLIPPAGE_BPS + _config.COMMISSION_BPS) / 10000.0
    if direction == "long":
        return entry * (1 + cost_frac), exit_price * (1 - cost_frac)
    return entry * (1 - cost_frac), exit_price * (1 + cost_frac)


def build_research_trade(signal, entry_time, entry, exit_time, exit_price,
                          outcome, candles_held, apply_costs: bool = True) -> ResearchTrade:
    """Builds the trade record, charging costs and sizing the position
    from config.RISK_PER_TRADE_PCT.

    The R-multiple is computed against the ORIGINAL risk (the stop
    distance the strategy actually chose), not the post-cost entry, so
    "1R" keeps meaning "the risk you signed up for". Costs therefore
    show up as a drag on R rather than by quietly redefining R - which
    is what makes a cost-on vs cost-off comparison readable.
    """
    direction = signal.direction
    stop_loss, target = signal.stop_loss, signal.target
    risk = abs(entry - stop_loss)

    fill_entry, fill_exit = (_apply_costs(entry, exit_price, direction)
                             if apply_costs else (entry, exit_price))

    gross = (fill_exit - fill_entry) if direction == "long" else (fill_entry - fill_exit)
    r_multiple = gross / risk if risk > 0 else 0.0
    pnl_pct = gross / entry * 100 if entry else 0.0
    risk_pct = risk / entry if entry else 0.0

    risk_budget = _config.ACCOUNT_EQUITY * _config.RISK_PER_TRADE_PCT
    position_size = risk_budget / risk if risk > 0 else 0.0

    return ResearchTrade(
        symbol=signal.symbol,
        strategy=signal.strategy,
        direction=direction,
        regime=signal.regime,
        trend_strength=signal.trend_strength,
        score=signal.score,
        entry_time=entry_time,
        exit_time=exit_time,
        entry=round(entry, 4),
        stop_loss=round(stop_loss, 4),
        target=round(target, 4),
        exit_price=round(exit_price, 4),
        outcome=outcome,
        r_multiple=round(r_multiple, 3),
        pnl_pct=round(pnl_pct, 3),
        candles_held=candles_held,
        risk_pct=round(risk_pct, 5),
        position_size=round(position_size, 2),
        pnl_currency=round(gross * position_size, 2),
    )
