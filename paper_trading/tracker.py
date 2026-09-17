"""
Live paper-trading tracker. Uses the SAME entry/exit rules as the
historical backtester (strategy/trade_engine.py) so the two are
directly comparable, but driven candle-by-candle by main.py's normal
polling loop instead of a pre-downloaded history.

Lifecycle of one paper trade, mirrors backtest/simulator.py's fill rule:

  PENDING_FILL  -- created the moment a signal fires, on the signal
                   candle's own close. No price is locked in yet.
  OPEN          -- filled at the OPEN of the next candle that appears
                   after the signal (never the signal candle's own
                   close - see trade_engine.py docstring for why).
  CLOSED        -- exited on TARGET, STOP, EOD_SQUAREOFF (still open
                   at session close - squared off at last price), or
                   NO_FILL (signal fired but price gapped through the
                   stop before a fill was possible, or the session
                   ended before the next candle arrived).

Requires main.py to keep running continuously (a persistent host, e.g.
your own machine or a free-tier VM) - a PENDING_FILL trade needs to
still be here on the NEXT poll to get filled and tracked. This will
NOT work correctly if you only run scan_once.py on an ephemeral
GitHub Actions runner, since the DB (and the in-progress trade) won't
persist between runs unless you persist paper_trades.db yourself.
"""
import os
import sqlite3

import pandas as pd

import config

DB_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper_trades.db")


def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            strategy TEXT,
            status TEXT,              -- PENDING_FILL | OPEN | CLOSED
            trade_date TEXT,          -- YYYY-MM-DD, for daily/report filtering
            signal_time TEXT,
            entry_time TEXT,
            entry REAL,
            stop_loss REAL,
            target REAL,
            exit_time TEXT,
            exit_price REAL,
            outcome TEXT,             -- TARGET | STOP | EOD_SQUAREOFF | NO_FILL
            r_multiple REAL,
            pnl_pct REAL
        )
        """
    )
    conn.commit()
    conn.close()


def has_open_or_pending(symbol: str, strategy: str, trade_date: str) -> bool:
    conn = sqlite3.connect(DB_FILE)
    row = conn.execute(
        """SELECT COUNT(*) FROM paper_trades
           WHERE symbol=? AND strategy=? AND trade_date=? AND status IN ('PENDING_FILL','OPEN')""",
        (symbol, strategy, trade_date),
    ).fetchone()
    conn.close()
    return row[0] > 0


def open_pending(symbol: str, strategy: str, signal, signal_time, trade_date: str):
    """Registers a trade the moment a signal fires. No fill yet."""
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        """INSERT INTO paper_trades (symbol, strategy, status, trade_date, signal_time, stop_loss)
           VALUES (?, ?, 'PENDING_FILL', ?, ?, ?)""",
        (symbol, strategy, trade_date, str(signal_time), signal.stop_loss),
    )
    conn.commit()
    conn.close()


def fill_pending(symbol: str, latest_row):
    """Fills every PENDING_FILL trade for this symbol whose signal candle
    is strictly before `latest_row` - i.e. only once a genuinely NEW
    candle has appeared since the signal fired, at that new candle's
    open. Recomputes target off the real fill price, same as the
    backtester. Safe to call every poll (idempotent once filled)."""
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute(
        "SELECT id, stop_loss, signal_time FROM paper_trades WHERE symbol=? AND status='PENDING_FILL'",
        (symbol,),
    ).fetchall()

    latest_ts = pd.Timestamp(latest_row["timestamp"])
    for trade_id, stop_loss, signal_time in rows:
        if latest_ts <= pd.Timestamp(signal_time):
            continue  # still the same candle the signal fired on

        entry = float(latest_row["open"])
        risk = entry - stop_loss
        if risk <= 0:
            conn.execute(
                """UPDATE paper_trades SET status='CLOSED', outcome='NO_FILL',
                   exit_time=?, exit_price=? WHERE id=?""",
                (str(latest_row["timestamp"]), entry, trade_id),
            )
            continue

        target = entry + risk * config.RISK_REWARD_RATIO
        conn.execute(
            """UPDATE paper_trades SET status='OPEN', entry_time=?, entry=?, target=? WHERE id=?""",
            (str(latest_row["timestamp"]), entry, target, trade_id),
        )
    conn.commit()
    conn.close()


def check_open_trades(symbol: str, latest_row):
    """Checks every OPEN trade for this symbol against the latest
    candle's high/low. Conservative same-candle rule: if both stop and
    target are touched in one candle, STOP wins (see trade_engine.py).
    Safe to call every poll."""
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute(
        "SELECT id, entry, stop_loss, target FROM paper_trades WHERE symbol=? AND status='OPEN'",
        (symbol,),
    ).fetchall()

    for trade_id, entry, stop_loss, target in rows:
        hit_stop = latest_row["low"] <= stop_loss
        hit_target = latest_row["high"] >= target
        if hit_stop:
            _close(conn, trade_id, entry, stop_loss, stop_loss, "STOP", latest_row["timestamp"])
        elif hit_target:
            _close(conn, trade_id, entry, stop_loss, target, "TARGET", latest_row["timestamp"])

    conn.commit()
    conn.close()


def square_off_eod(symbol: str, latest_row):
    """Call once at/after session close: closes any still-OPEN trade at
    the last available price, and discards any PENDING_FILL trade that
    never got a chance to fill today."""
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute(
        "SELECT id, entry, stop_loss FROM paper_trades WHERE symbol=? AND status='OPEN'",
        (symbol,),
    ).fetchall()
    for trade_id, entry, stop_loss in rows:
        _close(conn, trade_id, entry, stop_loss, latest_row["close"], "EOD_SQUAREOFF", latest_row["timestamp"])

    conn.execute(
        "UPDATE paper_trades SET status='CLOSED', outcome='NO_FILL' WHERE symbol=? AND status='PENDING_FILL'",
        (symbol,),
    )
    conn.commit()
    conn.close()


def _close(conn, trade_id, entry, stop_loss, exit_price, outcome, exit_time):
    risk = entry - stop_loss
    r_multiple = (exit_price - entry) / risk if risk > 0 else 0.0
    pnl_pct = (exit_price - entry) / entry * 100 if entry else 0.0
    conn.execute(
        """UPDATE paper_trades SET status='CLOSED', exit_time=?, exit_price=?, outcome=?,
           r_multiple=?, pnl_pct=? WHERE id=?""",
        (str(exit_time), exit_price, outcome, round(r_multiple, 2), round(pnl_pct, 2), trade_id),
    )
