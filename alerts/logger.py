import sqlite3
import os
from datetime import datetime

DB_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "alerts_log.db")


def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            entry REAL,
            stop_loss REAL,
            target REAL,
            reason TEXT,
            candle_time TEXT,
            sent_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def log_alert(signal):
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        """INSERT INTO alerts (symbol, entry, stop_loss, target, reason, candle_time, sent_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            signal.symbol,
            signal.entry,
            signal.stop_loss,
            signal.target,
            signal.reason,
            str(signal.candle_time),
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()
