import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import config
from data.yfinance_client import get_intraday_candles
from strategy.screener import check_signal, check_vwap_retest, evaluate
from alerts.telegram_bot import send_alert, format_signal_message, format_retest_message
from alerts.logger import init_db, log_alert

MARKET_TZ = ZoneInfo(config.MARKET_TIMEZONE)


def load_watchlist() -> list:
    with open("watchlist.txt") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def is_market_hours() -> bool:
    # zoneinfo resolves America/New_York's UTC offset correctly for the
    # current date, so this line alone handles the EST/EDT (DST) switch -
    # no separate summer/winter logic needed, unlike a bot using a fixed
    # UTC offset would require.
    now = datetime.now(MARKET_TZ)
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    open_t = now.replace(hour=config.MARKET_OPEN_HOUR, minute=config.MARKET_OPEN_MINUTE, second=0, microsecond=0)
    close_t = now.replace(hour=config.MARKET_CLOSE_HOUR, minute=config.MARKET_CLOSE_MINUTE, second=0, microsecond=0)
    active_from = open_t + timedelta(minutes=config.SKIP_FIRST_MINUTES)
    return active_from <= now <= close_t


def main():
    init_db()
    watchlist = load_watchlist()
    ema_alerted_today = set()
    retest_alerted_today = set()
    today = datetime.now(MARKET_TZ).date()

    print(f"Watching {len(watchlist)} symbols: {watchlist}")

    while True:
        now = datetime.now(MARKET_TZ)

        if now.date() != today:
            ema_alerted_today.clear()
            retest_alerted_today.clear()
            today = now.date()

        if not is_market_hours():
            print(f"[{now.strftime('%H:%M:%S')} ET] Outside market hours, sleeping...")
            time.sleep(60)
            continue

        for symbol in watchlist:
            try:
                df = get_intraday_candles(symbol, config.CANDLE_INTERVAL_MINUTES)

                status = evaluate(symbol, df)
                if status["status"] == "OK":
                    print(
                        f"[{now.strftime('%H:%M:%S')} ET] {symbol}: price={status['price']} "
                        f"ema={status['ema']} vwap={status['vwap']} "
                        f"cross={status['ema_cross']} above_vwap={status['above_vwap']} "
                        f"vol_ratio={status['vol_ratio']}x (need >{status['vol_needed']}x)"
                    )
                else:
                    print(f"[{now.strftime('%H:%M:%S')} ET] {symbol}: {status['status']}")

                if symbol not in ema_alerted_today:
                    signal = check_signal(symbol, df)
                    if signal:
                        send_alert(format_signal_message(signal))
                        log_alert(signal)
                        ema_alerted_today.add(symbol)
                        print(f"[{now.strftime('%H:%M:%S')} ET] >>> EMA-CROSS ALERT SENT: {symbol}")

                if symbol not in retest_alerted_today:
                    retest = check_vwap_retest(symbol, df)
                    if retest:
                        send_alert(format_retest_message(retest))
                        retest_alerted_today.add(symbol)
                        print(f"[{now.strftime('%H:%M:%S')} ET] >>> VWAP-RETEST ALERT SENT: {symbol} ({retest.aggressor})")
            except Exception as e:
                print(f"[{now.strftime('%H:%M:%S')} ET] Error processing {symbol}: {e}")

        time.sleep(config.POLL_SECONDS)


if __name__ == "__main__":
    main()
