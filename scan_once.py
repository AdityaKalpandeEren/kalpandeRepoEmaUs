"""
Runs ONE scan pass across the watchlist and exits - designed to be
triggered on a schedule (e.g. every 5 minutes during US market hours)
by GitHub Actions, mirroring the BTC/NSE bots' scan_once.py.

No auth needed at all here (Yahoo Finance via yfinance is keyless) -
this is the simplest of the three bots to run unattended: just the
Telegram secrets, nothing to refresh, ever.
"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import config
from data.yfinance_client import get_intraday_candles
from strategy.screener import check_signal, check_vwap_retest, check_vwap_broad_TEST, evaluate
from alerts.telegram_bot import send_alert, format_signal_message, format_retest_message

MARKET_TZ = ZoneInfo(config.MARKET_TIMEZONE)


def load_watchlist() -> list:
    with open("watchlist.txt") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def is_market_hours() -> bool:
    now = datetime.now(MARKET_TZ)
    if now.weekday() >= 5:
        return False
    open_t = now.replace(hour=config.MARKET_OPEN_HOUR, minute=config.MARKET_OPEN_MINUTE, second=0, microsecond=0)
    close_t = now.replace(hour=config.MARKET_CLOSE_HOUR, minute=config.MARKET_CLOSE_MINUTE, second=0, microsecond=0)
    active_from = open_t + timedelta(minutes=config.SKIP_FIRST_MINUTES)
    return active_from <= now <= close_t


def main():
    now = datetime.now(MARKET_TZ)
    if not is_market_hours():
        print(f"[{now.strftime('%H:%M:%S')} ET] Outside market hours - skipping this pass.")
        return

    watchlist = load_watchlist()

    for symbol in watchlist:
        try:
            df = get_intraday_candles(symbol, config.CANDLE_INTERVAL_MINUTES)

            status = evaluate(symbol, df)
            print(f"{symbol}: {status}")

            signal = check_signal(symbol, df)
            if signal:
                send_alert(format_signal_message(signal))
                print(f">>> EMA-CROSS ALERT SENT: {symbol}")

            # retest = check_vwap_retest(symbol, df)
            # if retest:
            #     send_alert(format_retest_message(retest))
            #     print(f">>> VWAP-RETEST ALERT SENT: {symbol} ({retest.aggressor})")
            
            retest = check_vwap_broad_TEST(symbol, df)
            if retest:
                send_alert(format_retest_message(retest))
                print(f">>> [TEST] BROAD VWAP ALERT SENT: {symbol} ({retest.aggressor})")
        except Exception as e:
            print(f"Error processing {symbol}: {e}")


if __name__ == "__main__":
    main()
