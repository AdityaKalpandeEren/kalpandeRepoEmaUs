import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import config
from data.yfinance_client import get_intraday_candles
from strategy.screener import check_signal, check_vwap_retest, evaluate
from alerts.telegram_bot import send_alert, format_signal_message, format_retest_message
from alerts.logger import init_db, log_alert
from paper_trading import tracker as paper_tracker

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


def is_at_or_past_close(now: datetime) -> bool:
    close_t = now.replace(hour=config.MARKET_CLOSE_HOUR, minute=config.MARKET_CLOSE_MINUTE, second=0, microsecond=0)
    return now >= close_t


def main():
    init_db()
    if config.PAPER_TRADING_ENABLED:
        paper_tracker.init_db()
        print("Paper trading: ON - every signal will also open/track a virtual trade "
              "in paper_trades.db (run `python -m paper_trading.generate_report` anytime for accuracy).")

    watchlist = load_watchlist()
    # Two independent "already alerted today" sets, so an EMA-cross alert
    # for a symbol doesn't block a later VWAP-retest alert for the same
    # symbol (and vice versa).
    ema_alerted_today = set()
    retest_alerted_today = set()
    today = datetime.now(MARKET_TZ).date()
    eod_squared_off_today = False

    print(f"Watching {len(watchlist)} symbols: {watchlist}")

    while True:
        now = datetime.now(MARKET_TZ)

        if now.date() != today:
            ema_alerted_today.clear()
            retest_alerted_today.clear()
            eod_squared_off_today = False
            today = now.date()

        if not is_market_hours():
            # Right after the 8pm ET session close, square off any paper
            # trades still OPEN from today before sleeping through the close.
            if config.PAPER_TRADING_ENABLED and not eod_squared_off_today and is_at_or_past_close(now):
                for symbol in watchlist:
                    try:
                        df = get_intraday_candles(symbol, config.CANDLE_INTERVAL_MINUTES)
                        if not df.empty:
                            paper_tracker.square_off_eod(symbol, df.iloc[-1])
                    except Exception as e:
                        print(f"[{now.strftime('%H:%M:%S')} ET] EOD square-off error for {symbol}: {e}")
                eod_squared_off_today = True
                print(f"[{now.strftime('%H:%M:%S')} ET] Paper trading: squared off any open positions for today.")

            print(f"[{now.strftime('%H:%M:%S')} ET] Outside market hours, sleeping...")
            time.sleep(60)
            continue

        trade_date = now.date().isoformat()

        for symbol in watchlist:
            try:
                df = get_intraday_candles(symbol, config.CANDLE_INTERVAL_MINUTES)
                if df.empty:
                    continue

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

                # Evaluated once per poll regardless of alert-dedup state, so
                # paper trading (its own independent dedup, per trade_date)
                # never misses a signal just because a Telegram alert for it
                # already went out earlier today.
                signal = check_signal(symbol, df)
                retest = check_vwap_retest(symbol, df)
                last_row = df.iloc[-1]

                if signal and symbol not in ema_alerted_today:
                    send_alert(format_signal_message(signal))
                    log_alert(signal)
                    ema_alerted_today.add(symbol)
                    print(f"[{now.strftime('%H:%M:%S')} ET] >>> EMA-CROSS ALERT SENT: {symbol}")

                if retest and symbol not in retest_alerted_today:
                    send_alert(format_retest_message(retest))
                    retest_alerted_today.add(symbol)
                    print(f"[{now.strftime('%H:%M:%S')} ET] >>> VWAP-RETEST ALERT SENT: {symbol} ({retest.aggressor})")

                if config.PAPER_TRADING_ENABLED:
                    # 1) fill anything still pending from an earlier candle
                    paper_tracker.fill_pending(symbol, last_row)
                    # 2) check anything currently open against this candle
                    paper_tracker.check_open_trades(symbol, last_row)
                    # 3) register any new signal as a pending virtual trade
                    if signal and not paper_tracker.has_open_or_pending(symbol, "EMA_CROSS", trade_date):
                        paper_tracker.open_pending(symbol, "EMA_CROSS", signal, last_row["timestamp"], trade_date)
                    if retest and not paper_tracker.has_open_or_pending(symbol, "VWAP_RETEST", trade_date):
                        paper_tracker.open_pending(symbol, "VWAP_RETEST", retest, last_row["timestamp"], trade_date)

            except Exception as e:
                print(f"[{now.strftime('%H:%M:%S')} ET] Error processing {symbol}: {e}")

        time.sleep(config.POLL_SECONDS)


if __name__ == "__main__":
    main()
