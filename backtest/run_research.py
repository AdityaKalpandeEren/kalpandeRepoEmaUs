"""
Multi-strategy research backtest: runs 9 entry models (8 classic entry
types + a weighted scoring engine), long AND short, over the same
historical candles, and ranks them by expectancy.

    # compare everything over the last 45 days
    python -m backtest.run_research --days 45

    # a focused, liquid subset (recommended first run)
    python -m backtest.run_research --symbols AAPL,MSFT,NVDA,AMZN,META,TSLA --days 45

    # one model only
    python -m backtest.run_research --days 45 --strategies G_CONFLUENCE

    # longs only
    python -m backtest.run_research --days 45 --directions long

    # measure what the regime filter is actually worth
    python -m backtest.run_research --days 45 --no-regime-filter

    # measure what costs are actually worth
    python -m backtest.run_research --days 45 --no-costs

    # position-sizing sweep (0.25 / 0.5 / 0.75 / 1.0% risk per trade)
    python -m backtest.run_research --days 45 --risk-sweep

Models:
  A_BREAKOUT          immediate structural breakout
  B_BREAKOUT_CLOSE    breakout confirmed by candle close
  C_BREAKOUT_RETEST   breakout, retest, continuation
  D_VWAP_RECLAIM      cross back through VWAP
  E_VWAP_REJECTION    rejection off VWAP in trend direction
  F_EMA_PULLBACK      pullback to fast EMA in an EMA trend
  G_CONFLUENCE        EMA and VWAP agreeing as one zone
  H_ORB_VWAP          opening-range breakout + VWAP confirmation
  SCORE_ENGINE        weighted 6-component score

Read the caveats at the end of the printed summary before acting on
any ranking. Nine models on one dataset means the winner is partly
luck; the report says so per strategy.
"""
import argparse
import sys
from datetime import datetime, timedelta

import config
from backtest.data_loader import load_symbol_history, group_by_day
from backtest.research_simulator import simulate_day_research
from backtest.research_report import write_research_report, print_research_summary
from backtest.run_backtest import _preflight_dates, load_symbol_list
from strategy.strategies import ENTRY_MODELS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", help="Comma-separated tickers. Defaults to watchlist.txt.")
    p.add_argument("--from", dest="from_date", help="YYYY-MM-DD (or use --days)")
    p.add_argument("--to", dest="to_date", help="YYYY-MM-DD (or use --days)")
    p.add_argument("--days", type=int, help="Last N days ending today (recommended)")
    p.add_argument("--interval", type=int, default=config.CANDLE_INTERVAL_MINUTES)
    p.add_argument("--strategies", default=",".join(ENTRY_MODELS),
                    help="Comma-separated subset of the model names above")
    p.add_argument("--directions", default="long,short", help="long | short | long,short")
    p.add_argument("--no-regime-filter", action="store_true",
                    help="Disable the NO-TRADE regime gate, to measure what it's worth")
    p.add_argument("--no-costs", action="store_true",
                    help="Disable slippage/commission, to measure what they cost you")
    p.add_argument("--risk-sweep", action="store_true",
                    help="Report currency P&L at 0.25/0.5/0.75/1.0%% risk per trade")
    p.add_argument("--max-symbols", type=int, default=40,
                    help="Cap symbols fetched (default 40). Yahoo throttles large runs.")
    p.add_argument("--out", default="backtest/results")
    return p.parse_args()


def main():
    args = parse_args()

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

    if args.no_regime_filter:
        config.REGIME_FILTER_ENABLED = False
    apply_costs = not args.no_costs

    symbols = ([s.strip() for s in args.symbols.split(",")] if args.symbols
               else load_symbol_list("watchlist.txt"))
    if len(symbols) > args.max_symbols:
        print(f"Watchlist has {len(symbols)} symbols; capping at {args.max_symbols} "
              f"(raise with --max-symbols, but Yahoo will throttle).\n")
        symbols = symbols[: args.max_symbols]

    wanted = [s.strip() for s in args.strategies.split(",")]
    models = {k: v for k, v in ENTRY_MODELS.items() if k in wanted}
    if not models:
        print(f"No valid strategies in '{args.strategies}'. Choose from: {list(ENTRY_MODELS)}")
        sys.exit(1)

    directions = tuple(d.strip() for d in args.directions.split(",") if d.strip())

    print(f"Research backtest: {len(symbols)} symbols | {args.interval}-min candles")
    print(f"Models    : {list(models)}")
    print(f"Directions: {list(directions)}")
    print(f"Costs     : {'ON' if apply_costs else 'OFF'} "
          f"({config.SLIPPAGE_BPS}bps slip + {config.COMMISSION_BPS}bps commission per side)")
    print(f"Regime gate: {'ON' if config.REGIME_FILTER_ENABLED else 'OFF'}\n")

    all_trades = []
    total_days = 0
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

        sym_trades = []
        for _, day_df in days.items():
            sym_trades.extend(simulate_day_research(
                symbol, day_df, args.interval, models, directions, apply_costs))
        total_days = max(total_days, len(days))
        all_trades.extend(sym_trades)
        print(f"{len(days)} days, {len(sym_trades)} trades")

    print(f"\nTotal trades simulated: {len(all_trades)}")
    if not all_trades:
        print("Nothing to report. Try more symbols, a longer window, or --no-regime-filter "
              "(the regime gate may be blocking everything).")
        return

    meta = {
        "from": args.from_date, "to": args.to_date, "interval": args.interval,
        "symbols": len(symbols), "days": total_days,
        "costs": f"{config.SLIPPAGE_BPS}+{config.COMMISSION_BPS}bps/side" if apply_costs else "NONE",
        "regime_filter": "ON" if config.REGIME_FILTER_ENABLED else "OFF",
    }
    result = write_research_report(all_trades, args.out, meta)
    print_research_summary(result["summary"])

    if args.risk_sweep:
        print("\n" + "=" * 78)
        print("POSITION-SIZING SWEEP")
        print("=" * 78)
        total_r = result["summary"]["overall"]["total_r"]
        dd_r = result["summary"]["overall"]["max_drawdown_r"]
        print(f"Account equity: {config.ACCOUNT_EQUITY:,.0f}")
        print(f"{'RISK/TRADE':>12}{'TOTAL P&L':>16}{'MAX DD':>16}{'DD %':>10}")
        for risk in (0.0025, 0.005, 0.0075, 0.01):
            budget = config.ACCOUNT_EQUITY * risk
            print(f"{risk*100:>11.2f}%{total_r*budget:>16,.0f}{dd_r*budget:>16,.0f}"
                  f"{abs(dd_r*budget)/config.ACCOUNT_EQUITY*100:>9.1f}%")
        print("\nR-multiples are size-agnostic, so this is a linear scaling of the same")
        print("result - it tells you the drawdown you would have had to sit through, which")
        print("is usually what decides the size you can actually trade, not the P&L.")

    print(f"\nSaved trade log: {result['csv']}")
    print(f"Saved report   : {result['report']}")

    print("\n" + "=" * 78)
    print("BEFORE YOU ACT ON THIS")
    print("=" * 78)
    print("1. Yahoo caps 5-min history at ~60 days, so this is ONE market period.")
    print("   A model that wins here has not been validated across regimes.")
    print("2. Nine models on one dataset: the winner is partly selection luck.")
    print("   Re-run the top 2-3 on a DIFFERENT date range before believing them.")
    print("3. Check the t-stat. Below 2.0 the result is inside the noise band,")
    print("   no matter how good the win rate looks.")
    print("4. Read 'Strategy x regime' in the report - a model that is negative")
    print("   overall but strong in one regime needs a filter, not deletion.")


if __name__ == "__main__":
    main()
