"""
US V8 "leaders" research: the published setups of the most successful momentum
traders (us_v8/setups.py - same code as the NSE repo's v8/) on US large caps.

Data: Yahoo daily bars 2013-01 .. 2026-10 (us_v5/cache/daily_bars.parquet) of
the CURRENT S&P 500, each stock tradable only from its S&P "date added"
(us_v5/seed/sp500.csv). Yahoo has no delisted tickers, so stocks that left the
index (and every bankruptcy) are missing: SURVIVORSHIP BIAS - judge against
the equal-weight average of the same stocks, not only SPY.
Prices: OHLC scaled by adj_close / close (splits + dividends). Liquid = 20-day
median dollar volume >= $20M and price >= $10.
Costs: slippage 5 bps at the open/close, 15 bps for stop orders (buy-stop
entries, stop-loss exits); SEC fee on sells; no commission. $100k book.

Variants (fixed before the first run, all reported) - same as the NSE run:
  BO_10, BO_20, BO_10_nf, EP_10, EP_20, EP_fresh_10, BOEP_10, ATH_10ATR,
  ATH504_10ATR (2-year high instead of all-time high - what a live job can see)
  (EP gap >= 10% here; NSE used 8% because of circuit bands.)
Market filter: SPY 10-day average above its 20-day average.
Windows: one continuous run 2014-01 .. 2026-10:
  WF 2017-01 .. 2025-06 and its halves A / B, PV 2025-07 .. 2026-10.
Keep rule (fixed in advance): beat SPY AND the equal-weight universe in both WF halves.

    python -m us_v8.research
"""
from __future__ import annotations

import argparse
import logging
import os
import time

import numpy as np
import pandas as pd

from us_v8 import engine, setups

log = logging.getLogger("us_v8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "us_v8", "results")
BARS = os.path.join(ROOT, "us_v5", "cache", "daily_bars.parquet")
SEED = os.path.join(ROOT, "us_v5", "seed", "sp500.csv")
START, END = "2014-01-01", "2026-10-02"
WINDOWS = {"EARLY": ("2014-01-01", "2016-12-31"), "WF": ("2017-01-01", "2025-06-30"),
           "WF_A": ("2017-01-01", "2021-06-30"), "WF_B": ("2021-07-01", "2025-06-30"),
           "PV": ("2025-07-01", "2026-10-02")}
RF = 0.0                                                   # Sharpe on raw returns (US cash yield varied 0-5%)

PARAMS = {"min_value": 2e7, "min_price": 10.0, "ep_gap": 0.10, "ep_vol": 3.0, "ep_fresh_max": 0.30,
          "bo_rs_min": 0.85, "bo_tight_adr": 2.5}
BOOK = {"start_equity": 100_000, "risk_pct": 0.01, "max_pos": 10, "max_weight": 0.20, "partial_day": 3,
        "partial_frac": 1 / 3, "bo_stop_cap": 0.08, "ep_stop_cap": 0.12, "ep_skip_close_at_high": False,
        "fill_mode": "pessimistic", "regime": True, "trail_ma": 10}
VARIANTS = {
    "BO_10": {"setups": ["BO"]},
    "BO_20": {"setups": ["BO"], "trail_ma": 20},
    "BO_10_nf": {"setups": ["BO"], "regime": False},
    "EP_10": {"setups": ["EP"]},
    "EP_20": {"setups": ["EP"], "trail_ma": 20},
    "EP_fresh_10": {"setups": ["EP"], "ep_signal": "ep_fresh"},
    "BOEP_10": {"setups": ["BO", "EP"]},
    "ATH_10ATR": {"setups": ["ATH"], "regime": False, "max_pos": 30, "max_weight": 0.10},
    "ATH504_10ATR": {"setups": ["ATH"], "regime": False, "max_pos": 30, "max_weight": 0.10, "ath_signal": "ath_504"},
}


class UsCosts:
    buy_rate = 0.0

    def __init__(self, slip_open=0.0005, slip_stop=0.0015, sec_fee=0.0000278):
        self.s = {"open": slip_open, "close": slip_open, "stop": slip_stop}
        self.sec = sec_fee

    def slip(self, kind):
        return self.s[kind]

    def buy(self, money):
        return 0.0

    def sell(self, money):
        return self.sec * money


BROAD = os.path.join(ROOT, "us_v8", "cache", "broad")


def load(universe: str = "sp500"):
    """sp500: current S&P 500, each from its date added. broad: every current US listing
    >= $300M market cap (us_v8/fetch_broad.py) - even more survivorship-biased."""
    if universe == "broad":
        import glob
        b = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(BROAD, "chunk_*.parquet")))])
        b = b.dropna(subset=["close", "adj_close"]).drop_duplicates(["date", "symbol"])
        b = b[(b["close"] > 0) & (b["volume"] >= 0)]
    else:
        seed = pd.read_csv(SEED)
        b = pd.read_parquet(BARS)
        b = b[b["symbol"].isin(seed["symbol"])].copy()
    b["date"] = pd.to_datetime(b["date"]).dt.tz_localize(None)
    k = b["adj_close"] / b["close"]
    for col in ("open", "high", "low"):
        b[f"adj_{col}"] = b[col] * k
    b["value"] = b["close"] * b["volume"]
    b = b.sort_values(["symbol", "date"])
    med = b.groupby("symbol")["value"].transform(lambda s: s.rolling(20, min_periods=15).median())
    keep = b.loc[(med >= PARAMS["min_value"]) & (b["close"] >= PARAMS["min_price"]), "symbol"].unique()
    b = b[b["symbol"].isin(keep)]
    wide = {key: b.pivot(index="date", columns="symbol", values=v).astype("float32" if universe == "broad" else "float64")
            for key, v in (("o", "adj_open"), ("h", "adj_high"), ("l", "adj_low"), ("c", "adj_close"),
                           ("val", "value"), ("raw", "close"))}
    dates = wide["c"].index
    if universe == "broad":
        uni = None
    else:
        added = pd.to_datetime(seed.set_index("symbol")["date_added"])
        uni = pd.DataFrame({s: dates >= added.get(s, pd.Timestamp("2100-01-01")) for s in wide["c"].columns}, index=dates)
    spy = pd.read_parquet(BARS, filters=[("symbol", "==", "SPY")]).set_index("date")["adj_close"].reindex(dates)
    return wide, uni, spy


def arrays(wide, ind):
    c = wide["c"]
    px = {"dates": c.index, "entities": list(c.columns), "O": wide["o"].to_numpy(float), "H": wide["h"].to_numpy(float),
          "L": wide["l"].to_numpy(float), "C": c.to_numpy(float), "CF": c.ffill().to_numpy(float)}
    lv = c.notna().to_numpy()
    px["last_valid"] = np.where(lv.any(0), len(c) - 1 - np.argmax(lv[::-1], axis=0), -1)
    na = {k: ind[k].fillna(False).to_numpy(bool) for k in ("bo", "ep", "ep_fresh", "ath", "ath_504")}
    for k in ("pivot", "adr", "rs", "vratio", "atr42", "sma10", "sma20"):
        na[k] = ind[k].to_numpy(float)
    return px, na


def window_stats(ret, w, bench=None):
    a, b = WINDOWS[w]
    return engine.perf(ret[(ret.index >= a) & (ret.index <= b)], RF, bench)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--universe", default="sp500", choices=("sp500", "broad"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    global RES
    RES = os.path.join(ROOT, "us_v8", "results", a.universe)
    os.makedirs(RES, exist_ok=True)
    t = time.time()
    wide, uni, spy = load(a.universe)
    ind = setups.indicators(wide["o"], wide["h"], wide["l"], wide["c"], wide["val"], wide["raw"], PARAMS, uni)
    px, na = arrays(wide, ind)
    reg = setups.regime(spy, px["dates"])
    spy_r = spy.pct_change(fill_method=None)
    # equal-weight benchmark of the same stocks: every stock liquid at the previous close
    ew = wide["c"].pct_change(fill_method=None).where(ind["liquid"].shift(1, fill_value=False)).mean(axis=1)
    yrs = len(px["dates"]) / 252
    log.info("%d stocks, %d sessions (%.0fs) | signals/yr BO %.0f EP %.0f EP_fresh %.0f ATH %.0f | filter on %.0f%%",
             len(px["entities"]), len(px["dates"]), time.time() - t, na["bo"].sum() / yrs, na["ep"].sum() / yrs,
             na["ep_fresh"].sum() / yrs, na["ath"].sum() / yrs, 100 * reg.mean())
    costs = UsCosts()
    rows, rets = [], {}
    todo = {k: v for k, v in VARIANTS.items() if not a.only or k in a.only.split(",")}
    for name, over in todo.items():
        res = engine.run(px, na, {**BOOK, **over}, costs, START, END, reg)
        r = res.daily["ret"]
        rets[name] = r
        res.trades.to_csv(os.path.join(RES, f"trades_{name}.csv"), index=False)
        res.daily.to_csv(os.path.join(RES, f"daily_{name}.csv"))
        row = {"variant": name}
        for w in WINDOWS:
            s = window_stats(r, w)
            row.update({f"{w}_CAGR": s.get("CAGR"), f"{w}_Sharpe": s.get("Sharpe"), f"{w}_MaxDD": s.get("MaxDD")})
        tr = res.trades
        wf_tr = tr[(tr["entry_date"] >= WINDOWS["WF"][0]) & (tr["entry_date"] <= WINDOWS["WF"][1])] if len(tr) else tr
        row.update({f"WF_{k}": v for k, v in engine.trade_stats(wf_tr).items()})
        row["avg_exposure"] = (res.daily["invested"] / res.daily["equity"]).mean()
        row["cost_drag"] = res.daily["costs"].iloc[-1] / res.daily["equity"].mean() / (len(r) / 252)
        rows.append(row)
        log.info("%-12s WF CAGR %+.1f%% Sharpe %.2f MaxDD %.1f%% | PV %+.1f%% | trades %d", name, 100 * row["WF_CAGR"],
                 row["WF_Sharpe"], 100 * row["WF_MaxDD"], 100 * (row["PV_CAGR"] or 0), len(tr))
    t = pd.DataFrame(rows).set_index("variant")
    ref = {"SPY": spy_r.fillna(0), "EW universe": ew.fillna(0)}
    for nm, r in ref.items():
        row = {}
        for w in WINDOWS:
            s = window_stats(r, w)
            row.update({f"{w}_CAGR": s.get("CAGR"), f"{w}_Sharpe": s.get("Sharpe"), f"{w}_MaxDD": s.get("MaxDD")})
        t.loc[nm, list(row)] = list(row.values())
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    cols = [f"{w}_{m}" for w in WINDOWS for m in ("CAGR", "Sharpe", "MaxDD")]
    print("\nPERFORMANCE after costs (CAGR / Sharpe rf=0 / MaxDD):")
    print(t[cols].astype(float).round(3).to_string())
    tcols = [c for c in t.columns if c.startswith("WF_") and c.split("_", 1)[1] in
             ("Trades", "Win%", "AvgWin%", "AvgLoss%", "AvgR", "PF", "Hold", "Top7%share")] + ["avg_exposure", "cost_drag"]
    print("\nTRADES inside WF:")
    print(t.loc[list(todo), tcols].astype(float).round(3).to_string())
    print("\nKEEP RULE (beat SPY and EW universe CAGR in WF_A and WF_B):")
    for k in todo:
        ok = all(t.loc[k, f"{h}_CAGR"] > max(t.loc["SPY", f"{h}_CAGR"], t.loc["EW universe", f"{h}_CAGR"])
                 for h in ("WF_A", "WF_B"))
        print(f"  {k:12s} {'KEEP' if ok else 'no'}")
    t.to_csv(os.path.join(RES, "summary.csv"))
    by_year = pd.DataFrame({k: v.groupby(v.index.year).apply(lambda x: (1 + x).prod() - 1) for k, v in {**rets, **ref}.items()})
    print("\nCALENDAR-YEAR RETURNS:")
    print((100 * by_year).round(1).to_string())
    by_year.to_csv(os.path.join(RES, "by_year.csv"))


if __name__ == "__main__":
    main()
