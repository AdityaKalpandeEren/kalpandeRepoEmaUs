"""
US V3 research: "Stocks in Play" intraday engine + ML ranker, walk-forward.

    python -m us_v3.research            # uses the growing store (us_v3.data)

Pipeline per (symbol, day):
  1. Session facts known at 9:35 ET: opening-range (first 5-min bar) OHLCV,
     opening relative volume vs its own 14-day average (RVOL - the "stocks
     in play" filter of Zarattini-Barbon-Aziz), gap vs prior close,
     pre-market return / volume / range, earnings-overnight flag, daily
     ATR(14), prior-day return, distance from the 20-day high.
  2. Setups (one entry per symbol per day - V2 re-bought TENB 3x):
       ORB   first 5-min bar green -> buy when price trades above its high
             (red -> short below its low), first break before 10:30;
             stop 10% of daily ATR (paper's rule), exit 15:55 close.
       LATE  15:30 entry in the direction of the market's first-half-hour
             move (Gao-Han-Li-Zhou intraday momentum) for stocks moving the
             same way, exit 15:55.
  3. Features at the entry bar: relative strength vs SPY/QQQ/SMH since the
     open, VWAP distance, entry-bar volume, minutes since open, VIX change...
  4. ML (LightGBM) predicts the trade's net R; walk-forward by DAY
     (expanding window, retrain every 5 days, no same-day leakage).
Costs: 3 bps/side slippage (commission-free broker); every R is net.
Portfolio: each day take the top K candidates (rule: by RVOL; ML: by
predicted R > 0); 1% equity risk per trade; daily P&L = mean of R x K%.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import time as dtime

import lightgbm as lgb
import numpy as np
import pandas as pd

from us_v3 import data as D

log = logging.getLogger("us_v3")
SLIP = 0.0003            # per side, regular hours
OPEN, CLOSE = dtime(9, 30), dtime(16, 0)


# ─── per symbol-day facts ────────────────────────────────────────────────

def _sessions(x: pd.DataFrame):
    t = x.index.time
    pre = x[(t >= dtime(4, 0)) & (t < OPEN)]
    rth = x[(t >= OPEN) & (t < CLOSE)]
    post = x[(t >= CLOSE)]
    return pre, rth, post


def _closed(x: pd.DataFrame, now) -> pd.DataFrame:
    """Live mode: keep only bars that have finished (ts + 5 min <= now)."""
    if now is None or x.empty:
        return x
    return x[x.index + pd.Timedelta(minutes=5) <= now]


def _before(s: pd.Series, day: pd.Timestamp) -> float:
    """Value on the last bar strictly before `day` (NaN if none)."""
    s = s[s.index < day]
    return float(s.iloc[-1]) if len(s) else np.nan


def symbol_days(sym: str, ctx_ret: dict, now=None, last_days: int | None = None) -> pd.DataFrame:
    """One row per trading day: opening facts + candidate trades.
    now (live): use only closed bars and accept today's partial session."""
    x = _closed(D.load(sym), now)
    dly = D.load(sym, "1d")
    if x.empty or dly.empty:
        return pd.DataFrame()
    if last_days:
        keep = sorted(set(x.index.date))[-last_days:]
        x = x[pd.Index(x.index.date).isin(keep)]
    if now is not None:
        dly = dly[dly.index < pd.Timestamp(now.date())]          # today's daily bar is unfinished
    e = D.load(sym, "earn")
    earn_ts = pd.DatetimeIndex(e["ts"]) if len(e) else pd.DatetimeIndex([])
    tr = pd.concat([dly["high"] - dly["low"], (dly["high"] - dly["close"].shift()).abs(),
                    (dly["low"] - dly["close"].shift()).abs()], axis=1).max(axis=1)
    # Daily facts known before the session = the value on the last daily bar
    # BEFORE the day (_before). Not shift(1)+asof: live drops today's
    # unfinished bar, and the shift then lagged live one extra day (stale
    # ATR / 20-day high / prior return vs training - fixed 2026-10-05).
    atr = tr.rolling(14).mean()
    hi20 = dly["high"].rolling(20).max()
    dret = dly["close"].pct_change()
    rows = []
    prev_close, or_vols, pm_vols = None, [], []
    for day, g in x.groupby(x.index.date):
        pre, rth, post = _sessions(g)
        live_today = now is not None and day == now.date()
        if (len(rth) < 70 and not live_today) or not len(rth) or rth.index[0].time() != OPEN:
            if len(rth) and not live_today:
                prev_close = float(rth["close"].iloc[-1])
            continue
        orb = rth.iloc[0]
        dkey = pd.Timestamp(day)
        r = {"symbol": sym, "date": dkey, "or_open": orb.open, "or_high": orb.high, "or_low": orb.low,
             "or_close": orb.close, "or_vol": orb.volume, "pm_vol": float(pre["volume"].sum())}
        r["rvol_open"] = orb.volume / np.mean(or_vols[-14:]) if len(or_vols) >= 10 and np.mean(or_vols[-14:]) > 0 else np.nan
        r["pm_rvol"] = r["pm_vol"] / np.mean(pm_vols[-14:]) if len(pm_vols) >= 10 and np.mean(pm_vols[-14:]) > 0 else np.nan
        r["gap"] = orb.open / prev_close - 1 if prev_close else np.nan
        r["pm_ret"] = float(pre["close"].iloc[-1]) / prev_close - 1 if prev_close and len(pre) else np.nan
        r["pm_range"] = (pre["high"].max() / pre["low"].min() - 1) if len(pre) else np.nan
        a = _before(atr, dkey)
        r["atr_d"] = a
        r["atr_pct"] = a / orb.open if a == a else np.nan
        r["or_ret"] = orb.close / orb.open - 1
        r["or_range_atr"] = (orb.high - orb.low) / a if a and a == a else np.nan
        r["prev_ret"] = _before(dret, dkey)
        r["dist_hi20"] = orb.open / _before(hi20, dkey) - 1
        prev_close_ts = pd.Timestamp.combine(day, CLOSE).tz_localize(D.ET) - pd.Timedelta(days=1)
        today_open_ts = pd.Timestamp.combine(day, OPEN).tz_localize(D.ET)
        r["earnings_overnight"] = float(((earn_ts > prev_close_ts - pd.Timedelta(days=3)) & (earn_ts <= today_open_ts)).any()) if len(earn_ts) else 0.0
        r["post_ret_prev"] = np.nan
        rows.append((r, rth))
        or_vols.append(orb.volume)
        pm_vols.append(r["pm_vol"])
        prev_close = float(rth["close"].iloc[-1])
    out = []
    for r, rth in rows:
        out += [t for t in trades_for_day(r, rth, ctx_ret) if t]
    return pd.DataFrame(out)


def _sim(rth: pd.DataFrame, i_entry: int, side: int, entry: float, stop: float, t_exit: dtime):
    """Walk bars from i_entry. On the ENTRY bar the intrabar path is unknown
    (its low may print before the breakout), so the stop counts there only if
    the bar CLOSES through it; from the next bar on, any touch stops out."""
    for j in range(i_entry, len(rth)):
        b = rth.iloc[j]
        if j == i_entry:
            if (side > 0 and b.close <= stop) or (side < 0 and b.close >= stop):
                return stop, rth.index[j], "STOP"
            if rth.index[j].time() >= t_exit:
                return float(b.close), rth.index[j], "EOD"
            continue
        if side > 0 and b.low <= stop:
            px = min(stop, b.open) if j > i_entry else stop
            return px, rth.index[j], "STOP"
        if side < 0 and b.high >= stop:
            px = max(stop, b.open) if j > i_entry else stop
            return px, rth.index[j], "STOP"
        if rth.index[j].time() >= t_exit:
            return float(b.close), rth.index[j], "EOD"
    b = rth.iloc[-1]
    return float(b.close), rth.index[-1], "EOD"


def trades_for_day(r: dict, rth: pd.DataFrame, ctx: dict) -> list:
    out = []
    a = r["atr_d"]
    if not (a == a) or a <= 0:
        return out
    day = r["date"]
    c = ctx.get(day)
    if c is None:
        return out
    cum_vol = rth["volume"].cumsum()
    vwap = (rth["close"] * rth["volume"]).cumsum() / cum_vol.replace(0, np.nan)
    day_open = rth["open"].iloc[0]

    def feats(i: int, entry: float) -> dict:
        ts = rth.index[i]
        prior = rth.iloc[:i]
        f = dict(r)
        f.update({"mins": (ts.hour * 60 + ts.minute) - 570,
                  "ret_since_open": entry / day_open - 1,
                  "rs_spy": (entry / day_open - 1) - c["spy"].get(ts, np.nan),
                  "rs_qqq": (entry / day_open - 1) - c["qqq"].get(ts, np.nan),
                  "rs_smh": (entry / day_open - 1) - c["smh"].get(ts, np.nan),
                  "spy_since_open": c["spy"].get(ts, np.nan), "vix_chg": c["vix"].get(ts, np.nan),
                  "spy_gap": c["spy_gap"], "vwap_dist": entry / vwap.iloc[i - 1] - 1 if i > 0 else 0.0,
                  "bar_vol_ratio": rth["volume"].iloc[i - 1] / max(prior["volume"].mean(), 1) if i > 0 else 1.0,
                  "day_range_atr": (prior["high"].max() - prior["low"].min()) / a if i > 0 else 0.0})
        return f

    # ORB (Zarattini et al.): direction from the first 5-min bar
    side = 1 if r["or_close"] > r["or_open"] else (-1 if r["or_close"] < r["or_open"] else 0)
    if side:
        lvl = r["or_high"] if side > 0 else r["or_low"]
        for i in range(1, len(rth)):
            ts = rth.index[i]
            if ts.time() > dtime(10, 30):
                break
            b = rth.iloc[i]
            hit = b.high >= lvl if side > 0 else b.low <= lvl
            if hit:
                entry = max(lvl, b.open) if side > 0 else min(lvl, b.open)
                f0 = feats(i, entry)
                # ORB_ATR: the paper's stop (10% of daily ATR); ORB_RANGE: stop at the
                # other side of the opening range (wider, common practitioner variant)
                for name, stop in (("ORB_ATR", entry - side * 0.10 * a),
                                   ("ORB_RANGE", r["or_low"] if side > 0 else r["or_high"])):
                    if side * (entry - stop) <= 0:
                        continue
                    px, xts, outcome = _sim(rth, i, side, entry, stop, dtime(15, 55))
                    out.append(_trade(name, side, i, entry, stop, px, xts, outcome, dict(f0)))
                break
    # LATE: market intraday momentum, 15:30 -> 15:55
    li = np.searchsorted(rth.index.time, dtime(15, 30))
    if li < len(rth) and c["spy_first30"] == c["spy_first30"] and c["spy_first30"] != 0:
        ms = 1 if c["spy_first30"] > 0 else -1
        entry = float(rth["open"].iloc[li])
        stock_dir = np.sign(entry / day_open - 1)
        if stock_dir == ms:
            stop = entry - ms * 0.10 * a
            px, xts, outcome = _sim(rth, li, ms, entry, stop, dtime(15, 55))
            out.append(_trade("LATE", ms, li, entry, stop, px, xts, outcome, feats(li, entry)))
    return out


def _trade(setup, side, i, entry, stop, px, xts, outcome, f):
    risk = abs(entry - stop)
    gross = side * (px - entry)
    cost = SLIP * (entry + px)
    f.update({"setup": setup, "side": side, "entry": entry, "stop": stop, "exit": px, "exit_ts": str(xts),
              "outcome": outcome, "R": (gross - cost) / risk, "R_gross": gross / risk, "ret": side * (px / entry - 1) - 2 * SLIP})
    return f


def context(now=None) -> dict:
    """Per day: SPY/QQQ/SMH return since the open and VIX change at each bar."""
    out = {}
    series = {k: _closed(D.load(s), now) for k, s in (("spy", "SPY"), ("qqq", "QQQ"), ("smh", "SMH"), ("vix", "^VIX"))}
    spy = series["spy"]
    prev = None
    for day, g in spy.groupby(spy.index.date):
        _, rth, _ = _sessions(g)
        if len(rth) < 70 and not (now is not None and day == now.date() and len(rth)):
            continue
        d = {}
        for k, s in series.items():
            if s.empty:
                d[k] = {}
                continue
            gg = s[s.index.date == day]
            _, rr, _ = _sessions(gg)
            if not len(rr):
                d[k] = {}
                continue
            o = rr["open"].iloc[0]
            d[k] = (rr["open"] / o - 1).to_dict()            # info at the START of each bar
        o = rth["open"].iloc[0]
        b30 = rth[rth.index.time < dtime(10, 0)]
        d["spy_first30"] = float(b30["close"].iloc[-1] / prev - 1) if prev and len(b30) else np.nan
        d["spy_gap"] = float(o / prev - 1) if prev else np.nan
        out[pd.Timestamp(day)] = d
        prev = float(rth["close"].iloc[-1])
    return out


# ─── dataset, walk-forward, portfolio ────────────────────────────────────

FEATURES = ["rvol_open", "pm_rvol", "gap", "pm_ret", "pm_range", "atr_pct", "or_ret", "or_range_atr", "prev_ret",
            "dist_hi20", "earnings_overnight", "mins", "ret_since_open", "rs_spy", "rs_qqq", "rs_smh",
            "vwap_dist", "bar_vol_ratio", "day_range_atr", "side", "is_late", "stop_range"]
# deliberately excluded: day-constant market features (spy_gap, spy_since_open, vix_chg) - on a few
# dozen days they act as date labels (lesson from V2); market info enters only via relative strength.


def build(symbols: list[str]) -> pd.DataFrame:
    ctx = context()
    parts = [symbol_days(s, ctx) for s in symbols]
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    df["is_late"] = (df["setup"] == "LATE").astype(int)
    df["stop_range"] = (df["setup"] == "ORB_RANGE").astype(int)
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


def walk_forward(df: pd.DataFrame, min_train_days: int = 20, step: int = 5, seed: int = 42) -> pd.Series:
    days = sorted(df["date"].unique())
    pred = pd.Series(np.nan, index=df.index)
    for k in range(min_train_days, len(days), step):
        train = df[df["date"] < days[k]]
        test = df[df["date"].isin(days[k:k + step])]
        y = train["R"].clip(-3, 5)
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                              random_state=seed, verbose=-1)
        m.fit(train[FEATURES], y)
        pred.loc[test.index] = m.predict(test[FEATURES])
    return pred


def portfolio(df: pd.DataFrame, pick: pd.Series, k: int, risk: float = 0.01) -> pd.Series:
    """Daily return when taking the chosen trades at `risk` of equity each
    (equal risk, at most k per day)."""
    sel = df[pick]
    return sel.groupby("date")["R"].apply(lambda r: r.head(k).sum() * risk)


def stats(daily: pd.Series, trades: pd.DataFrame) -> dict:
    daily = daily.reindex(sorted(daily.index)).fillna(0)
    t = daily.mean() / daily.std() * np.sqrt(len(daily)) if daily.std() > 0 else np.nan
    eq = (1 + daily).cumprod()
    return {"days": len(daily), "trades": len(trades), "win%": round(100 * (trades["R"] > 0).mean(), 1) if len(trades) else np.nan,
            "avgR": round(trades["R"].mean(), 3) if len(trades) else np.nan, "totR": round(trades["R"].sum(), 1),
            "day_ret%": round(100 * daily.mean(), 3), "pos_days": f"{(daily > 0).sum()}/{len(daily)}",
            "day_t": round(t, 2), "Sharpe_ann": round(daily.mean() / daily.std() * np.sqrt(252), 2) if daily.std() > 0 else np.nan,
            "total%": round(100 * (eq.iloc[-1] - 1), 1) if len(eq) else 0, "maxDD%": round(100 * (eq / eq.cummax() - 1).min(), 1) if len(eq) else 0}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="US V3 walk-forward research")
    ap.add_argument("--symbols", help="comma-separated tickers (default: watchlist.txt)")
    ap.add_argument("--days", type=int, help="use only the last N trading days in the store")
    ap.add_argument("--directions", default="long,short", help="long | short | long,short")
    ap.add_argument("--refresh", action="store_true", help="download the latest 5-min data first")
    ap.add_argument("--min-train-days", type=int, default=20)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    from backtest.run_backtest import load_symbol_list
    syms = [s.strip().upper() for s in a.symbols.split(",")] if a.symbols else load_symbol_list("watchlist.txt")
    syms = list(dict.fromkeys(D.tradable(syms)))
    if a.refresh:
        D.refresh_intraday(syms + D.CONTEXT)
        D.refresh_daily(syms + D.CONTEXT)
        D.refresh_earnings(syms)
    df = build(syms)
    if a.days:
        keep = sorted(df["date"].unique())[-a.days:]
        df = df[df["date"].isin(keep)]
    dirs = {"long": 1, "short": -1}
    df = df[df["side"].isin([dirs[d.strip()] for d in a.directions.split(",") if d.strip() in dirs])]
    if df["date"].nunique() <= a.min_train_days:
        raise SystemExit(f"need more than {a.min_train_days} days (have {df['date'].nunique()}): raise --days or lower --min-train-days")
    out_dir = os.path.join("backtest", "results", "us_v3")
    os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(os.path.join(out_dir, "candidates.parquet"))
    log.info("candidates: %d trades, %d days, %d symbols", len(df), df["date"].nunique(), df["symbol"].nunique())
    df["pred"] = walk_forward(df, min_train_days=a.min_train_days)
    test = df[df["pred"].notna()].copy()
    tdays = sorted(test["date"].unique())
    log.info("walk-forward test: %d days (%s .. %s)", len(tdays), tdays[0].date(), tdays[-1].date())

    rows = []

    def add(name, sub, order_col, k, ascending=False):
        s_ = sub.sort_values(["date", order_col], ascending=[True, ascending])
        top = s_.groupby("date").head(k)
        rows.append({"strategy": name, **stats(portfolio(top, pd.Series(True, index=top.index), k), top)})
        return top

    for setup in ("ORB_ATR", "ORB_RANGE"):
        o = test[test["setup"] == setup]
        add(f"{setup} random 20/day (no filter), L+S", o.sample(frac=1, random_state=1), "date", 20)
        add(f"{setup} top-20 RVOL (Zarattini), L+S", o[o["rvol_open"] >= 1], "rvol_open", 20)
        add(f"{setup} top-20 RVOL, LONG only", o[(o["rvol_open"] >= 1) & (o["side"] > 0)], "rvol_open", 20)
        for k in (5, 10):
            add(f"V3 ML {setup} top-{k} (pred>0), L+S", o[o["pred"] > 0], "pred", k)
            add(f"V3 ML {setup} top-{k} (pred>0), LONG only", o[(o["pred"] > 0) & (o["side"] > 0)], "pred", k)
    late = test[test["setup"] == "LATE"]
    add("LATE momentum top-20 RVOL", late, "rvol_open", 20)
    add("V3 ML LATE top-10 (pred>0)", late[late["pred"] > 0], "pred", 10)
    add("V3 ML any setup top-10 (pred>0), L+S", test[test["pred"] > 0].drop_duplicates(["date", "symbol"]), "pred", 10)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    test.to_parquet(os.path.join(out_dir, "test_trades.parquet"))
    pd.set_option("display.width", 250)
    print(res.to_string(index=False))
    for setup, g0 in test.groupby("setup"):
        ic = g0.groupby("date").apply(lambda g: g["pred"].corr(g["R"], method="spearman"), include_groups=False).dropna()
        print(f"ML rank-IC within {setup:9s}: mean {ic.mean():+.3f}, t {ic.mean() / ic.std() * np.sqrt(len(ic)):.2f}, "
              f"positive days {(ic > 0).mean():.0%}, gross avgR {g0['R_gross'].mean():+.3f}, net avgR {g0['R'].mean():+.3f}")
    json.dump({"test_days": len(tdays), "first": str(tdays[0].date()), "last": str(tdays[-1].date())},
              open(os.path.join(out_dir, "meta.json"), "w"))


if __name__ == "__main__":
    main()
