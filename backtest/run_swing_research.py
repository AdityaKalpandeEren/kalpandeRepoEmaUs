"""
SWING_DAYS_STR backtest: the swing/positional breakout model
(strategy/swing_strategy.py) on DAILY bars, holding a position for
days rather than minutes, exiting on a fixed +8%/-2% target/stop.

This is a separate engine from backtest/run_research.py (intraday,
5-min, session-VWAP) - see config.py's "SWING/POSITIONAL RESEARCH"
block for why. Daily bars aren't capped at ~60 days like intraday
data, so this can pull years of history for a much larger, more
trustworthy sample than the intraday models get.

    # 2 years of daily bars, one ticker
    python -m backtest.run_swing_research --symbols DRAM --days 730

    # the whole watchlist (capped), $10,000 account, 1% risk/trade
    ACCOUNT_EQUITY=10000 RISK_PER_TRADE_PCT=0.01 python -m backtest.run_swing_research \
      --days 730 --max-symbols 25 --risk-sweep
"""
import argparse
import sys
from datetime import datetime, timedelta

import pandas as pd

import config
from data.yfinance_client import get_daily_candles
from strategy.swing_strategy import enrich_daily, check_swing_breakout
from strategy.trade_engine import ResearchTrade, _apply_costs
from backtest.research_report import trades_to_dataframe, summarize
from backtest.run_backtest import load_symbol_list


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", help="Comma-separated tickers. Defaults to watchlist.txt.")
    p.add_argument("--from", dest="from_date", help="YYYY-MM-DD (or use --days)")
    p.add_argument("--to", dest="to_date", help="YYYY-MM-DD (or use --days)")
    p.add_argument("--days", type=int, default=730, help="Last N calendar days ending today (default 730 = ~2y)")
    p.add_argument("--no-costs", action="store_true", help="Disable slippage/commission")
    p.add_argument("--risk-sweep", action="store_true", help="Report currency P&L at 0.25/0.5/0.75/1.0%% risk per trade")
    p.add_argument("--max-symbols", type=int, default=40, help="Cap symbols fetched (default 40)")
    p.add_argument("--out", default="backtest/results")
    return p.parse_args()


def simulate_symbol(symbol: str, daily_df: pd.DataFrame, apply_costs: bool) -> list:
    """Walk forward one symbol's enriched daily bars, one position at a
    time, entering at the next day's OPEN after a signal (never the
    signal day's own close - that's not a realistic fill) and exiting
    on the first day the stop or target is touched (stop wins a
    same-day tie, the same conservative rule the intraday engine
    uses), or after SWING_MAX_HOLD_DAYS with no resolution."""
    df = enrich_daily(daily_df)
    trades = []
    i = config.SWING_MIN_WARMUP_DAYS
    n = len(df)

    while i < n - 1:
        sub = df.iloc[: i + 1]
        sig = check_swing_breakout(symbol, sub)
        if sig is None:
            i += 1
            continue

        fill_idx = i + 1
        entry_time = df.iloc[fill_idx]["timestamp"]
        entry = float(df.iloc[fill_idx]["open"])
        # Re-derive both target and stop from the REAL fill price, not
        # the signal candle's close - both are pure percentages here
        # (not a structural level worth keeping fixed), so there's no
        # reason to chase the stale signal-day numbers.
        target = entry * (1 + config.SWING_TARGET_PCT)
        stop = entry * (1 - config.SWING_STOP_PCT)
        if entry <= stop:
            i = fill_idx + 1
            continue

        outcome, exit_idx = None, None
        max_idx = min(n - 1, fill_idx + config.SWING_MAX_HOLD_DAYS)
        for j in range(fill_idx, max_idx + 1):
            row = df.iloc[j]
            hit_stop = row["low"] <= stop
            hit_target = row["high"] >= target
            if hit_stop:
                outcome, exit_idx = "STOP", j
                exit_price = stop
                break
            if hit_target:
                outcome, exit_idx = "TARGET", j
                exit_price = target
                break
        if outcome is None:
            exit_idx = max_idx
            outcome = "EOD_SQUAREOFF"   # timeout, not an intraday EOD - reusing the vocabulary the report already knows
            exit_price = float(df.iloc[exit_idx]["close"])

        exit_time = df.iloc[exit_idx]["timestamp"]
        risk = entry - stop
        fill_entry, fill_exit = _apply_costs(entry, exit_price, "long") if apply_costs else (entry, exit_price)
        gross = fill_exit - fill_entry
        r_multiple = gross / risk if risk > 0 else 0.0
        pnl_pct = gross / entry * 100 if entry else 0.0
        risk_pct = risk / entry if entry else 0.0
        risk_budget = config.ACCOUNT_EQUITY * config.RISK_PER_TRADE_PCT
        position_size = risk_budget / risk if risk > 0 else 0.0
        pnl_currency = position_size * gross

        trades.append(ResearchTrade(
            symbol=symbol, strategy="SWING_DAYS_STR", direction="long",
            regime="n/a", trend_strength=0.0, score=100.0,
            entry_time=entry_time, exit_time=exit_time,
            entry=round(entry, 4), stop_loss=round(stop, 4), target=round(target, 4),
            exit_price=round(exit_price, 4), outcome=outcome,
            r_multiple=round(r_multiple, 3), pnl_pct=round(pnl_pct, 3),
            candles_held=exit_idx - fill_idx,
            risk_pct=round(risk_pct, 5), position_size=round(position_size, 2),
            pnl_currency=round(pnl_currency, 2),
        ))
        i = exit_idx + 1   # one position at a time - resume scanning after this trade closes

    return trades


def _print_summary(summary: dict, meta: dict):
    if summary.get("total_trades", 0) == 0:
        print("No trades generated.")
        return
    o = summary["overall"]
    print(f"\nPeriod: {meta['from']} to {meta['to']}  |  daily bars  |  {meta['symbols']} symbols")
    print(f"Costs: {meta['costs']}  |  Exit: +{config.SWING_TARGET_PCT*100:.0f}% target / "
          f"-{config.SWING_STOP_PCT*100:.0f}% stop  |  Max hold: {config.SWING_MAX_HOLD_DAYS} trading days\n")
    print("=" * 78)
    print("SWING_DAYS_STR RESULTS")
    print("=" * 78)
    print(f"Trades: {o['trades']}  (W {o['wins']} / L {o['losses']} / Timeout {o['scratches']})")
    print(f"Win rate: {o['win_rate_pct']}% "
          f"{'±' + str(o['win_rate_se_pct']) + '%' if o['win_rate_se_pct'] is not None else ''}")
    print(f"Expectancy: {o['expectancy_r']} R/trade   |   Total: {o['total_r']} R")
    pf = o['profit_factor'] if o['profit_factor'] is not None else 'inf'
    print(f"Profit factor: {pf}   |   Max drawdown: {o['max_drawdown_r']} R")
    print(f"t-stat: {o['t_stat']}  ({'below 2.0 = not distinguishable from noise' if o['t_stat'] < 2.0 else 'above 2.0'})")
    print(f"Avg days held: {o['avg_candles_held']}")
    if not o["sample_adequate"]:
        print("[SAMPLE TOO SMALL - under 30 decided trades, do not trust this number]")

    print("\nBy symbol:")
    for sym, g in sorted(summary.get("by_symbol", {}).items(), key=lambda kv: kv[1]["expectancy_r"], reverse=True):
        tag = "" if g["sample_adequate"] else "  [small sample]"
        print(f"  {sym:8s} n={g['trades']:<4d} win={g['win_rate_pct']:>5.1f}%  "
              f"exp={g['expectancy_r']:>7.3f} R  totR={g['total_r']:>7.2f}{tag}")


def main():
    args = parse_args()
    if args.days and not (args.from_date or args.to_date):
        to_dt = datetime.today()
        from_dt = to_dt - timedelta(days=args.days)
        from_date, to_date = from_dt.strftime("%Y-%m-%d"), to_dt.strftime("%Y-%m-%d")
    else:
        from_date = args.from_date
        to_date = args.to_date or datetime.today().strftime("%Y-%m-%d")

    symbols = ([s.strip().upper() for s in args.symbols.split(",") if s.strip()]
               if args.symbols else load_symbol_list("watchlist.txt")[:args.max_symbols])
    apply_costs = not args.no_costs

    all_trades = []
    for idx, sym in enumerate(symbols, 1):
        print(f"[{idx}/{len(symbols)}] {sym} ...", end=" ", flush=True)
        try:
            daily_df = get_daily_candles(sym, from_date, to_date)
        except Exception as e:
            print(f"skip ({e})")
            continue
        if daily_df is None or daily_df.empty or len(daily_df) < config.SWING_MIN_WARMUP_DAYS + 2:
            print(f"not enough data ({0 if daily_df is None else len(daily_df)} bars)")
            continue
        trades = simulate_symbol(sym, daily_df, apply_costs)
        print(f"{len(daily_df)} daily bars, {len(trades)} trades")
        all_trades.extend(trades)

    meta = {
        "from": from_date, "to": to_date, "symbols": len(symbols),
        "costs": f"{config.SLIPPAGE_BPS}+{config.COMMISSION_BPS}bps/side" if apply_costs else "NONE",
    }
    summary = summarize(all_trades)
    _print_summary(summary, meta)

    if args.risk_sweep and summary.get("total_trades", 0):
        print("\n" + "=" * 78)
        print("POSITION-SIZING SWEEP")
        print("=" * 78)
        total_r = summary["overall"]["total_r"]
        dd_r = summary["overall"]["max_drawdown_r"]
        print(f"Account equity: {config.ACCOUNT_EQUITY:,.0f}")
        print(f"{'RISK/TRADE':>12}{'TOTAL P&L':>16}{'MAX DD':>16}{'DD %':>10}")
        for risk in (0.0025, 0.005, 0.0075, 0.01):
            budget = config.ACCOUNT_EQUITY * risk
            print(f"{risk*100:>11.2f}%{total_r*budget:>16,.0f}{dd_r*budget:>16,.0f}"
                  f"{abs(dd_r*budget)/config.ACCOUNT_EQUITY*100:>9.1f}%")

    if all_trades:
        import os
        os.makedirs(args.out, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(args.out, f"swing_trades_{stamp}.csv")
        trades_to_dataframe(all_trades).to_csv(csv_path, index=False)
        print(f"\nSaved trade log: {csv_path}")

    print("\n" + "=" * 78)
    print("BEFORE YOU ACT ON THIS")
    print("=" * 78)
    print("1. Fixed +8%/-2% exits are NOT ATR-scaled like the intraday models -")
    print("   they don't adapt to how volatile a given stock actually is.")
    print("2. Entry/exit filters (EMA stack, 52-week position, volume ratio) are a")
    print("   reasonable published structure (Minervini's Trend Template), not a")
    print("   proven edge on THIS data until the numbers above say so.")
    print("3. t-stat below 2.0 means the result is inside the noise band.")


if __name__ == "__main__":
    sys.exit(main())
