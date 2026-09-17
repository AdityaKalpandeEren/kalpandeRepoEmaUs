"""
Run this FIRST if historical backtesting isn't working.

    python -m backtest.check_data_access

The US bot's live scanning ("give me today's candles") and its
historical backtesting ("give me 30 days of candles") fail in
different ways, for different reasons - so "the bot works live but
backtest doesn't" is a very common and very confusing symptom. This
script walks the stack one layer at a time and tells you exactly which
layer broke, instead of leaving you with a raw traceback.

Layers tested, in order:
  1. Are yfinance + curl_cffi importable, and what versions?
  2. Can we reach Yahoo at all (simple daily bars)?
  3. Can we get TODAY's intraday bars (what the live bot needs)?
  4. Can we get a HISTORICAL intraday date range (what backtest needs)?
  5. How far back does each interval actually go for this symbol?

Layer 4 failing while layer 3 passes is the classic "works live,
can't backtest" case - almost always Yahoo rate-limiting/bot-blocking
the heavier range request, not a bug in this bot.
"""
import sys
from datetime import datetime, timedelta

TEST_SYMBOL = "AAPL"


def _ok(msg):
    print(f"  [PASS] {msg}")


def _fail(msg):
    print(f"  [FAIL] {msg}")


def step_1_imports():
    print("\n[1/5] Checking dependencies...")
    try:
        import yfinance as yf
        _ok(f"yfinance {yf.__version__}")
    except Exception as e:
        _fail(f"cannot import yfinance: {e!r}")
        print("      Fix: pip install -r requirements.txt")
        return False

    try:
        import curl_cffi
        _ok(f"curl_cffi {curl_cffi.__version__}")
    except Exception as e:
        _fail(f"cannot import curl_cffi: {e!r}")
        print("      yfinance 1.x needs curl_cffi - Yahoo blocks plain `requests` sessions.")
        print("      Fix: pip install 'curl_cffi>=0.7,<1.0'")
        print("      On old macOS/Linux with no prebuilt wheel this must COMPILE, which is")
        print("      where most install failures happen. If so, try a newer Python (3.11+).")
        return False

    try:
        import pandas as pd
        _ok(f"pandas {pd.__version__}")
    except Exception as e:
        _fail(f"cannot import pandas: {e!r}")
        return False
    return True


def step_2_connectivity():
    print("\n[2/5] Can we reach Yahoo at all (daily bars)...")
    import yfinance as yf
    try:
        df = yf.Ticker(TEST_SYMBOL).history(period="5d", interval="1d")
        if df is None or df.empty:
            _fail("reached Yahoo but got zero daily rows - likely rate-limited right now.")
            print("      Fix: wait 5-10 minutes and retry. Yahoo throttles by IP.")
            return False
        _ok(f"{len(df)} daily bars for {TEST_SYMBOL}")
        return True
    except Exception as e:
        _fail(f"{e!r}")
        _diagnose_exception(e)
        return False


def step_3_today_intraday():
    print("\n[3/5] Can we get TODAY's intraday bars (what the LIVE bot needs)...")
    import yfinance as yf
    try:
        df = yf.Ticker(TEST_SYMBOL).history(period="5d", interval="5m", prepost=True)
        if df is None or df.empty:
            _fail("zero intraday rows. If it's a weekend/holiday this can be normal.")
            return False
        _ok(f"{len(df)} 5-minute bars (last: {df.index[-1]})")
        return True
    except Exception as e:
        _fail(f"{e!r}")
        _diagnose_exception(e)
        return False


def step_4_historical_range():
    print("\n[4/5] Can we get a HISTORICAL range (what BACKTESTING needs)...")
    import yfinance as yf
    end = datetime.now()
    start = end - timedelta(days=10)
    try:
        df = yf.Ticker(TEST_SYMBOL).history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="5m",
            prepost=True,
        )
        if df is None or df.empty:
            _fail("zero rows for a 10-day 5m range.")
            print("      This is THE failure that blocks backtesting. Usually rate-limiting.")
            print("      Fix: wait a few minutes, then use a SHORTER range. The bot's own")
            print("      get_historical_candles() already chunks + retries to avoid this.")
            return False
        days = len(set(d.date() for d in df.index))
        _ok(f"{len(df)} bars across {days} trading days - historical backtesting will work")
        return True
    except Exception as e:
        _fail(f"{e!r}")
        _diagnose_exception(e)
        return False


def step_5_lookback_limits():
    print("\n[5/5] How far back does each interval actually go for this symbol...")
    import yfinance as yf
    for interval, expected in [("1m", "~30 days"), ("5m", "~60 days"),
                                ("15m", "~60 days"), ("1h", "~730 days")]:
        try:
            df = yf.Ticker(TEST_SYMBOL).history(period="max", interval=interval)
            if df is None or df.empty:
                print(f"  {interval:>4}: no data (expected {expected})")
                continue
            span = (df.index[-1] - df.index[0]).days
            print(f"  {interval:>4}: {span} days available (expected {expected})")
        except Exception as e:
            print(f"  {interval:>4}: error - {type(e).__name__}")


def _diagnose_exception(e):
    text = repr(e).lower()
    if "curl_cffi" in text or "requires curl_cffi session" in text:
        print("      -> curl_cffi problem. Reinstall: pip install --force-reinstall 'curl_cffi>=0.7,<1.0'")
    elif "ssl" in text or "tls" in text or "certificate" in text:
        print("      -> TLS/SSL problem, usually curl_cffi built against the wrong OpenSSL,")
        print("         or a corporate proxy/VPN intercepting HTTPS. Try off-VPN, or a newer Python.")
    elif "429" in text or "too many" in text or "rate" in text:
        print("      -> Rate-limited by Yahoo. Wait 5-10 minutes; use shorter ranges.")
    elif "404" in text:
        print("      -> Ticker not found. Check the symbol spelling.")
    elif "timeout" in text or "timed out" in text:
        print("      -> Network timeout. Check connectivity/proxy/firewall.")
    else:
        print("      -> Unrecognized error. The full repr above is the thing to search for.")


def main():
    print("=" * 62)
    print("US ALERT BOT - historical data access diagnostic")
    print("=" * 62)

    if not step_1_imports():
        print("\nStopped at dependencies - fix those first.")
        sys.exit(1)

    connectivity = step_2_connectivity()
    today_ok = step_3_today_intraday() if connectivity else False
    hist_ok = step_4_historical_range() if connectivity else False

    if hist_ok:
        step_5_lookback_limits()

    print("\n" + "=" * 62)
    print("SUMMARY")
    print("=" * 62)
    if hist_ok:
        print("Historical backtesting should work. Run:")
        print("  python -m backtest.run_backtest --symbols AAPL --from <YYYY-MM-DD> --to <YYYY-MM-DD>")
    elif today_ok:
        print("Live scanning works, but historical ranges don't - the classic")
        print("'works live, can't backtest' case. This is Yahoo throttling the heavier")
        print("range request, not a bug in the bot. Wait a few minutes and retry with a")
        print("short range (3-5 days) first; run_backtest.py chunks + retries automatically.")
    elif connectivity:
        print("Yahoo reachable for daily bars but not intraday. Intraday is more")
        print("aggressively rate-limited - wait and retry.")
    else:
        print("Can't reach Yahoo at all. Check network/proxy/VPN, then re-run this script.")


if __name__ == "__main__":
    main()
