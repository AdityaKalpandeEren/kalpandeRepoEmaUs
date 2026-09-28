"""
Walks one trading day's candles causally - exactly the way the live
bot sees them, one candle at a time, never looking ahead - and fires
the same strategy functions main.py/scan_once.py call. Each strategy
fires at most once per symbol per day here (mirroring main.py's
*_alerted_today dedup), so the trade count you get matches what you'd
actually act on.

No lookahead bias: check_signal/check_vwap_retest/check_vwap_broad_TEST
are only ever given candles up to and including the signal candle. The
simulated entry fill happens on the NEXT candle's open (see
strategy/trade_engine.py's docstring for why), never the signal
candle's own close.
"""
import pandas as pd

import config
from strategy.screener import check_signal, check_vwap_retest, check_vwap_broad_TEST
from strategy.trade_engine import simulate_forward, build_trade_result

STRATEGIES = {
    "EMA_CROSS": check_signal,
    "VWAP_RETEST": check_vwap_retest,
    "VWAP_BROAD_TEST": check_vwap_broad_TEST,
}


def _min_candles_needed() -> int:
    return max(config.EMA_PERIOD + 2, config.RETEST_TREND_LOOKBACK + 2, 2)


def simulate_day(symbol: str, day_df: pd.DataFrame, strategies: dict = None) -> list:
    """Returns a list of strategy.trade_engine.TradeResult for one symbol,
    one trading day."""
    strategies = strategies or STRATEGIES
    day_df = day_df.reset_index(drop=True)
    results = []
    already_fired = set()
    min_needed = _min_candles_needed()

    if len(day_df) < min_needed + 1:  # +1: need a following candle to fill on
        return results

    for i in range(min_needed - 1, len(day_df) - 1):
        sub_df = day_df.iloc[: i + 1].copy()

        for strat_name, check_fn in strategies.items():
            if strat_name in already_fired:
                continue
            try:
                signal = check_fn(symbol, sub_df.copy())
            except Exception:
                continue
            if not signal:
                continue

            already_fired.add(strat_name)

            fill_idx = i + 1
            entry_price = day_df.iloc[fill_idx]["open"]
            stop_loss = signal.stop_loss
            risk = entry_price - stop_loss
            if risk <= 0:
                continue  # gapped through the stop at the open - not a tradeable fill
            target = entry_price + risk * config.RISK_REWARD_RATIO

            exit_time, exit_price, outcome, exit_idx = simulate_forward(
                day_df, fill_idx, entry_price, stop_loss, target
            )

            trade = build_trade_result(
                symbol=symbol,
                strategy=strat_name,
                entry_time=day_df.iloc[fill_idx]["timestamp"],
                entry=entry_price,
                stop_loss=stop_loss,
                target=target,
                exit_time=exit_time,
                exit_price=exit_price,
                outcome=outcome,
                candles_held=exit_idx - fill_idx,
            )
            results.append(trade)

        if len(already_fired) == len(strategies):
            break  # every requested strategy already fired once today for this symbol

    return results
