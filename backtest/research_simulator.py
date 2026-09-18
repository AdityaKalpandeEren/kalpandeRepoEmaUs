"""
Research simulator: runs every entry model, in both directions, over
one trading day's candles, walking forward causally exactly the way the
live bot sees them.

Differences from backtest/simulator.py (which is kept unchanged so the
original three live signals stay comparable to their own history):
  - runs the 9 research models instead of the 3 live signals
  - supports SHORT as well as LONG
  - allows several trades per model per day, with a cooldown, because
    comparing 9 models on 1 trade/day/symbol gives a sample too small
    to distinguish any of them from noise
  - charges slippage and commission
  - tags every trade with the regime it was opened in

No lookahead: each model only ever sees df.iloc[:i+1], and the fill is
always the OPEN of candle i+1.
"""
import pandas as pd

import config
from strategy.indicators import enrich
from strategy.strategies import evaluate_all, ENTRY_MODELS
from strategy.trade_engine import simulate_forward_directional, build_research_trade


def _enrich_day(day_df: pd.DataFrame, candle_minutes: int) -> pd.DataFrame:
    return enrich(
        day_df.copy(),
        ema_fast=config.EMA_FAST,
        ema_slow=config.EMA_SLOW,
        atr_period=config.ATR_PERIOD,
        vol_period=config.VOLUME_AVG_PERIOD,
        struct_lookback=config.STRUCT_LOOKBACK,
        or_minutes=config.OPENING_RANGE_MINUTES,
        candle_minutes=candle_minutes,
    )


def simulate_day_research(symbol: str, day_df: pd.DataFrame, candle_minutes: int,
                           models: dict = None, directions=("long", "short"),
                           apply_costs: bool = True) -> list:
    """Returns a list of ResearchTrade for one symbol, one trading day."""
    models = models or ENTRY_MODELS
    day_df = day_df.reset_index(drop=True)
    if len(day_df) < config.MIN_WARMUP_CANDLES + 2:
        return []

    # Enrich ONCE for the whole day, then slice. Every feature is
    # causal (rolling/shifted), so row i of the enriched frame contains
    # only information available at candle i - slicing it is equivalent
    # to recomputing on the slice, but far faster.
    enriched = _enrich_day(day_df, candle_minutes)

    trades = []
    last_fired = {}   # (model, direction) -> candle index
    counts = {}       # (model, direction) -> trades taken today

    for i in range(config.MIN_WARMUP_CANDLES, len(enriched) - 1):
        sub = enriched.iloc[: i + 1]
        signals = evaluate_all(symbol, sub, models, directions)

        for sig in signals:
            key = (sig.strategy, sig.direction)
            if counts.get(key, 0) >= config.MAX_TRADES_PER_SYMBOL_DAY:
                continue
            if i - last_fired.get(key, -10**9) < config.SIGNAL_COOLDOWN_CANDLES:
                continue

            fill_idx = i + 1
            entry_price = float(day_df.iloc[fill_idx]["open"])

            # The stop was derived from the signal candle's close. If
            # the next open has already gapped through it, the trade
            # was never takeable - skip rather than book a fictional
            # instant loss.
            if sig.direction == "long" and entry_price <= sig.stop_loss:
                continue
            if sig.direction == "short" and entry_price >= sig.stop_loss:
                continue

            exit_time, exit_price, outcome, exit_idx = simulate_forward_directional(
                day_df, fill_idx, sig.direction, entry_price, sig.stop_loss, sig.target
            )

            trades.append(build_research_trade(
                signal=sig,
                entry_time=day_df.iloc[fill_idx]["timestamp"],
                entry=entry_price,
                exit_time=exit_time,
                exit_price=exit_price,
                outcome=outcome,
                candles_held=exit_idx - fill_idx,
                apply_costs=apply_costs,
            ))
            last_fired[key] = i
            counts[key] = counts.get(key, 0) + 1

    return trades
