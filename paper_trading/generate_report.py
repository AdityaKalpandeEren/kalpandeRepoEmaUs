"""
Generates an accuracy report from the live paper-trading log
(paper_trades.db) - run this anytime (daily, weekly, whenever) to see
how the alerts have actually performed since main.py started tracking
them.

Usage:
    python -m paper_trading.generate_report
    python -m paper_trading.generate_report --from 2025-09-01 --to 2025-09-10
"""
import argparse
import sqlite3

import pandas as pd

from paper_trading.tracker import DB_FILE
from paper_trading.report import write_report, print_summary
from strategy.trade_engine import TradeResult

CLOSED_OUTCOMES = ("TARGET", "STOP", "EOD_SQUAREOFF")


def load_closed_trades(trade_date_from: str = None, trade_date_to: str = None) -> list:
    conn = sqlite3.connect(DB_FILE)
    query = "SELECT * FROM paper_trades WHERE status='CLOSED' AND outcome IN (?,?,?)"
    params = list(CLOSED_OUTCOMES)
    if trade_date_from:
        query += " AND trade_date >= ?"
        params.append(trade_date_from)
    if trade_date_to:
        query += " AND trade_date <= ?"
        params.append(trade_date_to)

    try:
        df = pd.read_sql_query(query, conn, params=params)
    except pd.errors.DatabaseError:
        df = pd.DataFrame()
    finally:
        conn.close()

    trades = []
    for _, row in df.iterrows():
        trades.append(TradeResult(
            symbol=row["symbol"],
            strategy=row["strategy"],
            entry_time=row["entry_time"],
            exit_time=row["exit_time"],
            entry=row["entry"],
            stop_loss=row["stop_loss"],
            target=row["target"],
            exit_price=row["exit_price"],
            outcome=row["outcome"],
            r_multiple=row["r_multiple"],
            pnl_pct=row["pnl_pct"],
            candles_held=0,
        ))
    return trades


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="from_date", help="YYYY-MM-DD, inclusive")
    p.add_argument("--to", dest="to_date", help="YYYY-MM-DD, inclusive")
    p.add_argument("--out", default="paper_trading/results")
    args = p.parse_args()

    trades = load_closed_trades(args.from_date, args.to_date)
    print(f"Closed paper trades found: {len(trades)}")
    if not trades:
        print("Nothing to report yet - let main.py run through at least one full "
              "market session with PAPER_TRADING_ENABLED=true.")
        return

    result = write_report(trades, args.out, label="paper_trading")
    print_summary(result["summary"])
    print(f"\nSaved trade log:   {result['csv']}")
    print(f"Saved summary:     {result['report']}")


if __name__ == "__main__":
    main()
