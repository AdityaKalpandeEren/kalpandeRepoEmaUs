"""US V8 broad-universe research data (survivorship-biased: current listings only).

    python -m us_v8.fetch_broad us_v8/cache/broad

Download daily bars 2013-01 .. today for current US listings >= $300M market cap
(Nasdaq public screener) into OUT/chunk_XX.parquet. Resumable."""
import json, os, sys, time, urllib.request
import pandas as pd, yfinance as yf
OUT = sys.argv[1]
lst = os.path.join(OUT, "universe.csv")
if not os.path.exists(lst):
    req = urllib.request.Request("https://api.nasdaq.com/api/screener/stocks?tableonly=true&download=true",
                                 headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    rows = json.load(urllib.request.urlopen(req, timeout=60))["data"]["rows"]
    u = pd.DataFrame(rows)
    u["mcap"] = pd.to_numeric(u["marketCap"], errors="coerce")
    u["px"] = pd.to_numeric(u["lastsale"].str.replace("$", "", regex=False), errors="coerce")
    u = u[(u.mcap >= 3e8) & (u.px >= 5) & ~u.symbol.str.contains(r"[\^/ ]", regex=True)]
    u["symbol"] = u.symbol.str.strip().str.replace(".", "-", regex=False)
    u[["symbol", "name", "mcap", "sector", "industry", "ipoyear"]].to_csv(lst, index=False)
u = pd.read_csv(lst)
syms = sorted(u.symbol.unique())
print(len(syms), "symbols", flush=True)
for k in range(0, len(syms), 150):
    f = os.path.join(OUT, f"chunk_{k // 150:03d}.parquet")
    if os.path.exists(f):
        continue
    batch = syms[k:k + 150]
    for attempt in range(3):
        try:
            d = yf.download(batch, start="2013-01-01", interval="1d", auto_adjust=False, actions=False,
                            group_by="ticker", threads=True, progress=False)
            break
        except Exception as e:
            print("retry", k, e, flush=True); time.sleep(30)
    rows = []
    for s in batch:
        if s not in d.columns.get_level_values(0):
            continue
        x = d[s].dropna(how="all")
        if x.empty:
            continue
        x = x.rename(columns=str.lower).reset_index().rename(columns={"Date": "date", "adj close": "adj_close"})
        x["symbol"] = s
        rows.append(x[["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"]])
    pd.concat(rows).to_parquet(f, index=False) if rows else None
    print(f"chunk {k // 150}: {len(rows)}/{len(batch)}", flush=True)
    time.sleep(3)
print("done", flush=True)
