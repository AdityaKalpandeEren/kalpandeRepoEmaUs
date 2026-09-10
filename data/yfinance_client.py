import pandas as pd
import yfinance as yf


def get_intraday_candles(symbol: str, interval_minutes: int) -> pd.DataFrame:
    """Fetch today's intraday candles for one symbol via Yahoo Finance.

    Works for US stocks (AAPL, MSFT), indices (^GSPC, ^IXIC, ^DJI, ^VIX),
    and futures continuous contracts (ES=F, NQ=F, YM=F, CL=F, GC=F, SI=F).
    No API key required. Yahoo's intraday history is limited to the last
    60 days for 5m/15m bars (1m bars only cover the last 7 days) - we
    only ever need today's session, so `period="1d"` is always enough.
    """
    interval = f"{interval_minutes}m"
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="1d", interval=interval, auto_adjust=False)

    if df is None or df.empty:
        raise ValueError(
            f"No data returned for '{symbol}'. Check the ticker: NASDAQ/NYSE "
            f"stocks as-is (AAPL, MSFT, NVDA), indices with a caret prefix "
            f"(^GSPC, ^IXIC, ^DJI), futures continuous contracts with '=F' "
            f"(ES=F, NQ=F, CL=F, GC=F). Also possible: outside market hours "
            f"Yahoo sometimes returns nothing at all rather than yesterday's data."
        )

    df = df.reset_index()
    # Intraday bars come back indexed as "Datetime"; a daily bar (if Yahoo
    # ever falls back to one) would be "Date" instead - handle both.
    time_col = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(columns={
        time_col: "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])
    return df[["timestamp", "open", "high", "low", "close", "volume"]]
