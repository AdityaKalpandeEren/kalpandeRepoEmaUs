"""LCR SWING exit rules on synthetic 5-min bars (offline)."""
import pandas as pd

from us_lcr import swing as W


def _bars(days_prices):
    """days_prices: list of (date, open, low, close) per day -> 5-min bars 09:30..15:55."""
    rows = []
    for d, o, lo, c in days_prices:
        idx = pd.date_range(f"{d} 09:30", f"{d} 15:55", freq="5min", tz="America/New_York")
        for k, ts in enumerate(idx):
            rows.append({"ts": ts, "open": o if k == 0 else c, "high": max(o, c) * 1.001, "low": lo if k == 1 else min(o, c) * 0.999, "close": c})
    return pd.DataFrame(rows).set_index("ts")


def _pos():
    return {"symbol": "X", "status": "OPEN", "signal_ts": "2026-10-05 10:00:00-04:00", "entry_ts": "2026-10-05 10:05:00-04:00",
            "entry": 100.0, "stop_initial": 92.0, "shares": 100, "entry_date": "2026-10-05",
            "checked_ts": "2026-10-05 10:00:00-04:00"}


def test_exits_at_third_session_close():
    x = _bars([("2026-10-05", 100, 99, 101), ("2026-10-06", 101, 100, 102), ("2026-10-07", 102, 101, 103),
               ("2026-10-08", 103, 102, 104), ("2026-10-09", 104, 103, 105)])
    px, ts, why = W.check(_pos(), x)
    assert why == "DAY3_CLOSE" and str(ts.date()) == "2026-10-08" and px == 104


def test_stop_and_gap():
    x = _bars([("2026-10-05", 100, 99, 100), ("2026-10-06", 90, 89, 91)])     # gaps under the 92 stop
    px, ts, why = W.check(_pos(), x)
    assert why == "STOP_GAP" and px == 90
    x = _bars([("2026-10-05", 100, 99, 100), ("2026-10-06", 100, 91, 100)])   # next day's intraday low through the stop
    px, ts, why = W.check(_pos(), x)
    assert why == "STOP" and px == 92.0


def test_resumes_across_runs():
    p = _pos()
    x1 = _bars([("2026-10-05", 100, 99, 101), ("2026-10-06", 101, 100, 102)])
    assert W.check(p, x1) is None and p["sessions"] == ["2026-10-06"]
    x2 = _bars([("2026-10-07", 102, 101, 103), ("2026-10-08", 103, 102, 104)])
    px, ts, why = W.check(p, x2)
    assert why == "DAY3_CLOSE" and str(ts.date()) == "2026-10-08"
