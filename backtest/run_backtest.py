"""
Historical backtest for the US bot's EMA-cross / VWAP-retest /
VWAP-broad-test signals - answers "how often would these alerts
actually have won?" using real Yahoo Finance historical candles,
walked forward exactly the way the live bot sees them (see
backtest/simulator.py and strategy/trade_engine.py docstrings for the
no-lookahead / fill rules).

Usage examples (run from the us_alert_bot/ directory):

    # EASIEST - relative to today, so it's always inside Yahoo's window
    python -m backtest.run_backtest --symbols AAPL --days 30

    python -m backtest.run_backtest --days 45

    # explicit dates work too (must be recent - see the note below)
    python -m backtest.run_backtest --symbols AAPL,MSFT,NVDA \\
        --from 2026-08-20 --to 2026-09-16

    python -m backtest.run_backtest --days 30 \\
        --strategies EMA_CROSS,VWAP_RETEST

Notes:
- No API key needed - yfinance is free and keyless.
- DATES MUST BE RECENT. Yahoo only retains ~7 days of 1-minute bars,
  ~60 days of 5/15/30-minute bars, and ~730 days of 1-hour bars.
  Asking for anything older returns NOTHING ("must be within the last
  60 days") - it is not rate-limiting and retrying will not help.
  --days N avoids the whole problem; otherwise this script checks your
  dates up-front and tells you the valid window before fetching.
- If a fetch fails outright (not just "no data", an actual exception)
  and the error mentions curl_cffi, SSL/TLS, or "requires curl_cffi
  session" - that's a yfinance dependency problem in your environment,
  not a bug here. See README "Testing on historical data" for the fix.
- Results are written to backtest/results/ as a CSV (every simulated
  trade) and a Markdown summary report.
"""
import argparse
import sys
from datetime import datetime, timedelta

import config
from backtest.data_loader import load_symbol_history, group_by_day
from backtest.simulator import simulate_day, STRATEGIES
from paper_trading.report import write_report, print_summary

# Yahoo's hard intraday retention limits. Asking for anything older than
# this returns NOTHING (with a "must be within the last N days" notice
# per request) - it is not a rate-limit and not something retries fix.
# Checked up-front by _preflight_dates() so you find out before firing
# off hundreds of doomed requests.
_MAX_LOOKBACK_DAYS = {1: 7, 2: 60, 5: 60, 15: 60, 30: 60, 60: 730, 90: 60}


def _preflight_dates(from_date: str, to_date: str, interval: int) -> bool:
    """Returns True if the range is usable. Prints a concrete fix if not."""
    try:
        from_dt = datetime.strptime(from_date, "%Y-%m-%d")
        to_dt = datetime.strptime(to_date, "%Y-%m-%d")
    except ValueError:
        print("Dates must be YYYY-MM-DD.")
        return False

    if from_dt > to_dt:
        print(f"--from ({from_date}) is after --to ({to_date}).")
        return False

    today = datetime.now()
    max_days = _MAX_LOOKBACK_DAYS.get(interval, 60)
    earliest = today - timedelta(days=max_days)

    if to_dt > today:
        print(f"--to ({to_date}) is in the future. Today is {today.strftime('%Y-%m-%d')}.")
        return False

    if from_dt < earliest:
        print("=" * 64)
        print(f"DATE RANGE TOO OLD for {interval}-minute bars.")
        print("=" * 64)
        print(f"  You asked for : {from_date} .. {to_date}")
        print(f"  Today is      : {today.strftime('%Y-%m-%d')}")
        print(f"  Yahoo keeps   : {max_days} days of {interval}m bars "
              f"(back to {earliest.strftime('%Y-%m-%d')})")
        print()
        print("  This is a hard Yahoo limit - not rate-limiting, and retrying won't help.")
        print()
        print("  Fix - either use a recent range:")
        suggested_from = max(earliest + timedelta(days=1), from_dt)
        print(f"    --from {suggested_from.strftime('%Y-%m-%d')} --to {today.strftime('%Y-%m-%d')}")
        print("  ...or keep your dates and use a coarser interval with deeper history:")
        for iv in (60,):
            iv_earliest = today - timedelta(days=_MAX_LOOKBACK_DAYS[iv])
            if from_dt >= iv_earliest:
                print(f"    --interval {iv}   (has {_MAX_LOOKBACK_DAYS[iv]} days, "
                      f"back to {iv_earliest.strftime('%Y-%m-%d')})")
        print("=" * 64)
        return False

    return True


def load_symbol_list(filename: str) -> list:
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", help="Comma-separated tickers. Defaults to watchlist.txt.")
    p.add_argument("--from", dest="from_date", help="YYYY-MM-DD (or use --days)")
    p.add_argument("--to", dest="to_date", help="YYYY-MM-DD (or use --days)")
    p.add_argument("--days", type=int,
                    help="Shortcut for the last N days ending today - avoids picking "
                         "dates outside Yahoo's retention window. E.g. --days 30")
    p.add_argument("--interval", type=int, default=config.CANDLE_INTERVAL_MINUTES,
                    help=f"Candle size in minutes (default: {config.CANDLE_INTERVAL_MINUTES}, matches config.py)")
    p.add_argument("--strategies", default="EMA_CROSS,VWAP_RETEST,VWAP_BROAD_TEST",
                    help="Comma-separated subset of EMA_CROSS,VWAP_RETEST,VWAP_BROAD_TEST")
    p.add_argument("--out", default="backtest/results", help="Output directory for the CSV + report")
    return p.parse_args()


def main():
    args = parse_args()

    # --days N is a shortcut for "the last N days ending today", which
    # sidesteps the single most common mistake: picking dates that are
    # outside Yahoo's retention window without realising it.
    if args.days:
        today = datetime.now()
        args.to_date = today.strftime("%Y-%m-%d")
        args.from_date = (today - timedelta(days=args.days)).strftime("%Y-%m-%d")
        print(f"--days {args.days} -> {args.from_date} .. {args.to_date}\n")

    if not args.from_date or not args.to_date:
        print("Provide either --from and --to, or --days N.")
        sys.exit(1)

    if not _preflight_dates(args.from_date, args.to_date, args.interval):
        sys.exit(1)

    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else load_symbol_list("watchlist.txt")
    strategy_names = [s.strip() for s in args.strategies.split(",")]
    strategies = {k: v for k, v in STRATEGIES.items() if k in strategy_names}
    if not strategies:
        print(f"No valid strategies in '{args.strategies}'. Choose from: {list(STRATEGIES)}")
        sys.exit(1)

    print(f"Backtesting {len(symbols)} symbol(s) from {args.from_date} to {args.to_date}, "
          f"{args.interval}-min candles, strategies: {list(strategies)}\n")

    all_trades = []
    for symbol in symbols:
        print(f"Fetching {symbol} ...")
        try:
            hist = load_symbol_history(symbol, args.interval, args.from_date, args.to_date)
        except Exception as e:
            print(f"  FAILED for {symbol}: {e}")
            continue

        days = group_by_day(hist)
        if not days:
            print("  no historical candles returned for this range")
            continue
        print(f"  {len(days)} trading day(s) of data")

        for day, day_df in days.items():
            trades = simulate_day(symbol, day_df, strategies)
            all_trades.extend(trades)
            if trades:
                detail = ", ".join(f"{t.strategy}:{t.outcome}({t.r_multiple}R)" for t in trades)
                print(f"    {day}: {detail}")

    print(f"\nTotal trades simulated: {len(all_trades)}")
    if not all_trades:
        print("Nothing to report - try a wider date range, more symbols, or looser strategies.")
        return

    result = write_report(all_trades, args.out, label="backtest")
    print_summary(result["summary"])
    print(f"\nSaved trade log:   {result['csv']}")
    print(f"Saved summary:     {result['report']}")


if __name__ == "__main__":
    main()
