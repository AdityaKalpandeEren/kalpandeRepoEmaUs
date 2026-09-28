"""
Builds the labeled training set for model L_ML_META_V2.

Same idea as build_dataset.py (V1) - every base-model candidate becomes
one row of features + did-it-win - with the V2 differences that matter:

  - candidates come from strategies.ml_v2_candidates(), the SAME
    function the live model calls: session gate, hard risk vetoes,
    context/catalyst features and all. The training population is
    exactly the population V2 is later asked to judge.
  - labels come from simulate_forward_v2 (flat at the regular close,
    SHOCK_EXIT on a market move against the trade) - so label=1 means
    "won under the exits V2 will actually use".
  - symbols are processed in parallel worker processes; the shared
    market context is fetched once up front and served to workers from
    the disk cache.

    python -m backtest.ml.build_dataset_v2 --days 59 --max-symbols 150
    python -m backtest.ml.build_dataset_v2 --days 59 --symbols AAPL,MSFT,NVDA --workers 2

Output: config.ML_V2_DATASET_PATH, chronological, one row per candidate.
"""
import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

import config
from backtest.data_loader import load_symbol_history, group_by_day
from backtest.run_backtest import _preflight_dates, load_symbol_list
from strategy.indicators import enrich
from strategy.ml_features_v2 import FEATURE_COLUMNS_V2
from strategy.regime import classify, is_tradeable
from strategy.trade_engine import simulate_forward_v2, build_research_trade

BOOKKEEPING_COLUMNS = ["symbol", "date", "entry_time", "source_strategy", "direction",
                       "regime", "outcome", "r_multiple", "label"]
CSV_COLUMNS = BOOKKEEPING_COLUMNS + FEATURE_COLUMNS_V2


def _enrich_day(day_df, candle_minutes):
    return enrich(
        day_df.copy(),
        ema_fast=config.EMA_FAST, ema_slow=config.EMA_SLOW,
        atr_period=config.ATR_PERIOD, vol_period=config.VOLUME_AVG_PERIOD,
        struct_lookback=config.STRUCT_LOOKBACK, or_minutes=config.OPENING_RANGE_MINUTES,
        candle_minutes=candle_minutes,
    )


def build_rows_for_day(symbol, day_df, candle_minutes, directions, ctx, apply_costs=True):
    from strategy.strategies import ml_v2_candidates, ml_v2_in_session

    rows = []
    day_df = day_df.reset_index(drop=True)
    if len(day_df) < config.MIN_WARMUP_CANDLES + 2:
        return rows
    enriched = _enrich_day(day_df, candle_minutes)

    last_fired, counts = {}, {}
    for i in range(config.MIN_WARMUP_CANDLES, len(enriched) - 1):
        sub = enriched.iloc[: i + 1]
        if not ml_v2_in_session(sub):
            continue
        row = sub.iloc[-1]
        regime = classify(row)

        for direction in directions:
            # Same gates evaluate_all() applies before any model runs.
            if not is_tradeable(regime, direction):
                continue
            if direction == "short" and len(sub) < config.SHORT_MIN_SESSION_CANDLES:
                continue

            try:
                cands = ml_v2_candidates(symbol, sub, direction, regime, ctx)
            except Exception as e:
                print(f"  [{symbol}] candidate error at {row['timestamp']}: {e!r}")
                continue

            for name, sig, feats in cands:
                key = (name, direction)
                if counts.get(key, 0) >= config.MAX_TRADES_PER_SYMBOL_DAY:
                    continue
                if i - last_fired.get(key, -10**9) < config.SIGNAL_COOLDOWN_CANDLES:
                    continue

                fill_idx = i + 1
                entry_price = float(day_df.iloc[fill_idx]["open"])
                if direction == "long" and entry_price <= sig.stop_loss:
                    continue
                if direction == "short" and entry_price >= sig.stop_loss:
                    continue

                exit_time, exit_price, outcome, exit_idx = simulate_forward_v2(
                    day_df, fill_idx, direction, entry_price, sig.stop_loss, sig.target, ctx)
                trade = build_research_trade(
                    signal=sig, entry_time=day_df.iloc[fill_idx]["timestamp"], entry=entry_price,
                    exit_time=exit_time, exit_price=exit_price, outcome=outcome,
                    candles_held=exit_idx - fill_idx, apply_costs=apply_costs)

                record = {
                    "symbol": symbol,
                    "date": str(trade.entry_time.date()),
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


def process_symbol(symbol, interval, from_date, to_date, directions, apply_costs):
    """Worker entry point: all rows for one symbol, or an error string."""
    from strategy.market_context import get_context
    ctx = get_context(live=False)
    try:
        hist = load_symbol_history(symbol, interval, from_date, to_date)
    except Exception as e:
        return symbol, [], f"FAILED: {e}"
    days = group_by_day(hist)
    if not days:
        return symbol, [], "no data"
    try:
        ctx.ensure_symbol(symbol)
    except Exception as e:
        return symbol, [], f"context FAILED: {e}"
    rows = []
    for _, day_df in days.items():
        rows.extend(build_rows_for_day(symbol, day_df, interval, directions, ctx, apply_costs))
    return symbol, rows, f"{len(days)} days, {len(rows)} candidates"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", help="Comma-separated tickers. Defaults to watchlist.txt.")
    p.add_argument("--from", dest="from_date")
    p.add_argument("--to", dest="to_date")
    p.add_argument("--days", type=int, default=59,
                   help="Last N days ending today (default 59 - Yahoo's 5-min retention)")
    p.add_argument("--interval", type=int, default=config.CANDLE_INTERVAL_MINUTES)
    p.add_argument("--directions", default="long,short",
                   help="Default both: more training rows; direction is itself a feature")
    p.add_argument("--no-costs", action="store_true")
    p.add_argument("--max-symbols", type=int, default=150)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default=config.ML_V2_DATASET_PATH)
    return p.parse_args()


def main():
    args = parse_args()
    if args.days and not (args.from_date and args.to_date):
        today = datetime.now()
        args.to_date = today.strftime("%Y-%m-%d")
        args.from_date = (today - timedelta(days=args.days)).strftime("%Y-%m-%d")
    if not _preflight_dates(args.from_date, args.to_date, args.interval):
        sys.exit(1)

    symbols = ([s.strip().upper() for s in args.symbols.split(",")] if args.symbols
               else load_symbol_list("watchlist.txt"))
    symbols = list(dict.fromkeys(s for s in symbols if s))   # de-dupe, keep order
    if len(symbols) > args.max_symbols:
        print(f"Capping {len(symbols)} symbols to --max-symbols {args.max_symbols}.")
        symbols = symbols[: args.max_symbols]
    directions = tuple(d.strip() for d in args.directions.split(",") if d.strip())
    apply_costs = not args.no_costs

    print(f"Building V2 dataset: {len(symbols)} symbols | {args.from_date}..{args.to_date} "
          f"| {args.interval}-min | directions={list(directions)} | workers={args.workers}")

    # Warm the shared market context ONCE here so workers read it from the
    # disk cache instead of all hitting Yahoo for the same VIX/QQQ data.
    from strategy.market_context import get_context
    print("Loading market context (VIX, VXN, VIX3M, QQQ, SPY, breadth)...", flush=True)
    get_context(live=False).ensure_market()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp_out = f"{args.out}.tmp.{os.getpid()}"
    total = 0
    all_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process_symbol, s, args.interval, args.from_date, args.to_date,
                            directions, apply_costs): s for s in symbols}
        for n, fut in enumerate(as_completed(futs), 1):
            sym = futs[fut]
            try:
                _, rows, status = fut.result()
            except Exception as e:
                rows, status = [], f"CRASHED: {e!r}"
            all_rows.extend(rows)
            total += len(rows)
            print(f"[{n}/{len(symbols)}] {sym}: {status}", flush=True)

    # Chronological order - the trainer splits by date and must never see
    # the future in its training block.
    all_rows.sort(key=lambda r: (r["entry_time"], r["symbol"], r["source_strategy"], r["direction"]))
    with open(tmp_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)
    os.replace(tmp_out, args.out)

    print(f"\nTotal candidate trades written: {total}")
    print(f"Saved: {args.out}")
    if total < 1000:
        print("\nWarning: under ~1000 rows is thin for ~75 features. Widen --max-symbols.")


if __name__ == "__main__":
    main()
