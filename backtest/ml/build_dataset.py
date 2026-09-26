"""
Builds the labeled training set for the ML meta-label filter (model
L_ML_META in strategy/strategies.py).

For every candle, every one of the 12 base models (A-K + SCORE_ENGINE)
proposes zero or more candidate trades, long and/or short. Each
candidate that fires becomes ONE training row: its features
(strategy/ml_features.py) plus whether it actually won, using the
EXACT SAME forward-simulation, fill rule and cost model as
backtest/run_research.py (strategy.trade_engine.simulate_forward_directional
+ build_research_trade) - so "label=1" means the same thing here as
"WIN" does in the research report.

    python -m backtest.ml.build_dataset --days 60 --max-symbols 80
    python -m backtest.ml.build_dataset --days 60 --symbols AAPL,MSFT,NVDA,AMZN,META,TSLA,GOOGL

Output: backtest/ml/data/dataset.csv - one row per candidate trade, in
chronological order (NOT shuffled - the trainer needs the time order
intact to do a walk-forward split instead of a leaking k-fold. See
train_meta_model.py for why that matters).

This reuses the exact per-day simulation loop research_simulator.py
uses (same warmup, cooldown, per-symbol/day trade cap, no-fill-on-gap
rule) so the population of candidates trained on matches the
population run_research.py would report on - the only difference is
this script also keeps the FEATURES at signal time, which the report
path throws away.
"""
import argparse
import csv
import os
import sys
from datetime import datetime, timedelta

import config
from backtest.data_loader import load_symbol_history, group_by_day
from backtest.run_backtest import _preflight_dates, load_symbol_list
from strategy.indicators import enrich
from strategy.strategies import BASE_MODELS, _component_scores
from strategy.regime import classify, is_tradeable
from strategy.trade_engine import simulate_forward_directional, build_research_trade
from strategy.ml_features import extract_features, FEATURE_COLUMNS

BOOKKEEPING_COLUMNS = ["symbol", "entry_time", "source_strategy", "direction",
                        "regime", "outcome", "r_multiple", "label"]
CSV_COLUMNS = BOOKKEEPING_COLUMNS + FEATURE_COLUMNS


def _enrich_day(day_df, candle_minutes):
    return enrich(
        day_df.copy(),
        ema_fast=config.EMA_FAST, ema_slow=config.EMA_SLOW,
        atr_period=config.ATR_PERIOD, vol_period=config.VOLUME_AVG_PERIOD,
        struct_lookback=config.STRUCT_LOOKBACK, or_minutes=config.OPENING_RANGE_MINUTES,
        candle_minutes=candle_minutes,
    )


def build_rows_for_day(symbol, day_df, candle_minutes, directions, apply_costs=True):
    rows = []
    day_df = day_df.reset_index(drop=True)
    if len(day_df) < config.MIN_WARMUP_CANDLES + 2:
        return rows
    enriched = _enrich_day(day_df, candle_minutes)

    last_fired = {}
    counts = {}
    for i in range(config.MIN_WARMUP_CANDLES, len(enriched) - 1):
        sub = enriched.iloc[: i + 1]
        row = sub.iloc[-1]
        regime = classify(row)

        for direction in directions:
            if not is_tradeable(regime, direction):
                continue
            if direction == "short" and len(sub) < config.SHORT_MIN_SESSION_CANDLES:
                continue

            for name, fn in BASE_MODELS.items():
                key = (name, direction)
                if counts.get(key, 0) >= config.MAX_TRADES_PER_SYMBOL_DAY:
                    continue
                if i - last_fired.get(key, -10**9) < config.SIGNAL_COOLDOWN_CANDLES:
                    continue
                try:
                    sig = fn(symbol, sub, direction, regime)
                except Exception:
                    continue
                if sig is None:
                    continue

                fill_idx = i + 1
                entry_price = float(day_df.iloc[fill_idx]["open"])
                if direction == "long" and entry_price <= sig.stop_loss:
                    continue
                if direction == "short" and entry_price >= sig.stop_loss:
                    continue

                exit_time, exit_price, outcome, exit_idx = simulate_forward_directional(
                    day_df, fill_idx, direction, entry_price, sig.stop_loss, sig.target)
                trade = build_research_trade(
                    signal=sig,
                    entry_time=day_df.iloc[fill_idx]["timestamp"],
                    entry=entry_price,
                    exit_time=exit_time,
                    exit_price=exit_price,
                    outcome=outcome,
                    candles_held=exit_idx - fill_idx,
                    apply_costs=apply_costs,
                )

                comps = _component_scores(row, sub, direction)
                feats = extract_features(row, sub, direction, regime, comps, name)

                record = {
                    "symbol": symbol,
                    "entry_time": str(trade.entry_time),
                    "source_strategy": name,
                    "direction": direction,
                    "regime": regime.label,
                    "outcome": outcome,
                    "r_multiple": trade.r_multiple,
                    "label": 1 if trade.r_multiple > 0 else 0,
                }
                record.update(feats)
                rows.append(record)

                last_fired[key] = i
                counts[key] = counts.get(key, 0) + 1

    return rows


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", help="Comma-separated tickers. Defaults to watchlist.txt.")
    p.add_argument("--from", dest="from_date", help="YYYY-MM-DD (or use --days)")
    p.add_argument("--to", dest="to_date", help="YYYY-MM-DD (or use --days)")
    p.add_argument("--days", type=int, default=60, help="Last N days ending today (default 60, "
                    "Yahoo's max retention for 5-min bars)")
    p.add_argument("--interval", type=int, default=config.CANDLE_INTERVAL_MINUTES)
    p.add_argument("--directions", default="long,short",
                    help="long | short | long,short (default: both - more training data; "
                         "the live bot's own default stays long-only regardless)")
    p.add_argument("--no-costs", action="store_true")
    p.add_argument("--max-symbols", type=int, default=80)
    p.add_argument("--out", default="backtest/ml/data/dataset.csv")
    return p.parse_args()


def main():
    args = parse_args()
    if args.days:
        today = datetime.now()
        args.to_date = today.strftime("%Y-%m-%d")
        args.from_date = (today - timedelta(days=args.days)).strftime("%Y-%m-%d")
    if not _preflight_dates(args.from_date, args.to_date, args.interval):
        sys.exit(1)

    symbols = ([s.strip() for s in args.symbols.split(",")] if args.symbols
               else load_symbol_list("watchlist.txt"))
    if len(symbols) > args.max_symbols:
        print(f"Capping {len(symbols)} symbols to --max-symbols {args.max_symbols}.")
        symbols = symbols[: args.max_symbols]

    directions = tuple(d.strip() for d in args.directions.split(",") if d.strip())
    apply_costs = not args.no_costs

    print(f"Building ML dataset: {len(symbols)} symbols | {args.from_date}..{args.to_date} "
          f"| {args.interval}-min | directions={list(directions)}\n")

    # Written to a temp file and moved into place with os.replace() only on
    # success - NOT opened in-place at args.out. Two runs sharing the same
    # default --out (e.g. this script launched again while an earlier run
    # is still going) used to silently truncate each other's file mid-write
    # and interleave rows, corrupting the dataset without any error. Now
    # the worst case is "last one to finish wins cleanly", and a run that
    # crashes never touches the last good file at all.
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp_out = f"{args.out}.tmp.{os.getpid()}"
    total_rows = 0
    with open(tmp_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for n, symbol in enumerate(symbols, 1):
            print(f"[{n}/{len(symbols)}] {symbol} ...", end=" ", flush=True)
            try:
                hist = load_symbol_history(symbol, args.interval, args.from_date, args.to_date)
            except Exception as e:
                print(f"FAILED: {e}")
                continue

            days = group_by_day(hist)
            if not days:
                print("no data")
                continue

            sym_rows = []
            for _, day_df in days.items():
                sym_rows.extend(build_rows_for_day(symbol, day_df, args.interval,
                                                     directions, apply_costs))
            for r in sym_rows:
                writer.writerow(r)
            total_rows += len(sym_rows)
            print(f"{len(days)} days, {len(sym_rows)} candidate trades")

    os.replace(tmp_out, args.out)
    print(f"\nTotal candidate trades written: {total_rows}")
    print(f"Saved: {args.out}")
    if total_rows < 500:
        print("\nWarning: under 500 rows is thin for training a classifier with ~30 "
              "features - widen --days/--max-symbols/--directions if Yahoo allows, or "
              "expect the trainer to refuse / warn about an unreliable split.")


if __name__ == "__main__":
    main()
