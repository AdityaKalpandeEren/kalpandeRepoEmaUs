"""
US SCR research: every rule setup on every small/micro-cap runner day in the
5-minute store, simulated with the live exit rules, then an ML ranker on top.

    python -m us_scr.research

Honesty:
  * candidate days = stock-days whose intraday high reached >= +10% vs the
    previous close; the rule itself needs >= +10% at the signal bar, so this
    selection only removes days that could never trigger (no lookahead)
  * ML scores are walk-forward BY DAY: the model for each day is trained only
    on earlier days (retrained every 5 days, >= 15 days of history)
  * the threshold is fixed on the first half of the out-of-sample days and
    checked on the second half; costs: 0.25% (0.5% pre-market) per side
  * universe = TODAY's listed small caps (stocks delisted in the last 60
    days are missing - a small survivorship bias)
"""
from __future__ import annotations

import logging
import math
import os

import numpy as np
import pandas as pd

from us_scr import data as D
from us_scr import strategy as S

log = logging.getLogger("us_scr")
OUT = os.path.join(D.ROOT, "us_scr", "results")
FEATURES = ["pct", "log_rvol", "log_dollar", "vol_ratio", "dist_vwap", "minute", "n_hod", "bar_range", "close_loc",
            "ret_3", "reg_open_gap", "stop_pct", "prev_ret", "ret_5d", "max_ret_20d", "log_mcap", "log_price",
            "pre_session"]


def daily_context(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.sort_values(["symbol", "date"]).copy()
    g = d.groupby("symbol")
    d["prev_close"] = g["close"].shift()
    d["avg_vol20"] = g["volume"].transform(lambda s: s.shift().rolling(20, min_periods=10).mean())
    d["prev_ret"] = d["prev_close"] / g["close"].shift(2) - 1
    d["ret_5d"] = d["prev_close"] / g["close"].shift(6) - 1
    d["max_ret_20d"] = g["close"].transform(lambda s: (s / s.shift()).shift().rolling(20, min_periods=5).max() - 1)
    d["hi_pct"] = d["high"] / d["prev_close"] - 1
    return d


def build(daily: pd.DataFrame, uni: pd.DataFrame) -> pd.DataFrame:
    ctx = daily_context(daily)
    ctx = ctx[ctx["hi_pct"] >= S.WATCH_PCT].set_index(["symbol", "date"])
    mcap = uni.set_index("symbol")["mcap"]
    rows = []
    syms = sorted({s for s, _ in ctx.index})
    for n, sym in enumerate(syms):
        bars = D.load_store(sym)
        if bars.empty:
            continue
        for day, x in bars.groupby(bars.index.date):
            key = (sym, pd.Timestamp(day))
            if key not in ctx.index:
                continue
            c = ctx.loc[key]
            if not (c["prev_close"] > 0):
                continue
            x = x.sort_index()
            f = S.bar_features(x, float(c["prev_close"]), float(c["avg_vol20"]))
            per_day = 0
            busy_until = None
            for ts in S.setups(x, f):
                if per_day >= S.MAX_PER_SYMBOL_DAY or (busy_until is not None and ts < busy_until):
                    continue                                   # one position per stock at a time
                tr = S.simulate(sym, x, ts)
                if tr is None:
                    continue
                per_day += 1
                busy_until = tr.exit_ts
                r = f.loc[ts]
                rows.append({"symbol": sym, "date": pd.Timestamp(day), "signal_ts": ts, "entry_ts": tr.entry_ts,
                             "exit_ts": tr.exit_ts, "entry": tr.entry, "stop0": tr.stop0, "exit": tr.exit,
                             "outcome": tr.outcome, "R": tr.R, "ret_pct": tr.ret_pct, "session": tr.session,
                             "pct": r["pct"], "log_rvol": math.log1p(max(r["rvol"], 0)) if r["rvol"] == r["rvol"] else np.nan,
                             "log_dollar": math.log10(max(r["cum_dollar"], 1)), "vol_ratio": r["vol_ratio"],
                             "dist_vwap": r["dist_vwap"], "minute": r["minute"], "n_hod": r["n_hod"],
                             "bar_range": r["bar_range"], "close_loc": r["close_loc"], "ret_3": r["ret_3"],
                             "reg_open_gap": r["reg_open_gap"], "stop_pct": 1 - tr.stop0 / tr.entry,
                             "prev_ret": c["prev_ret"], "ret_5d": c["ret_5d"], "max_ret_20d": c["max_ret_20d"],
                             "log_mcap": math.log10(mcap.get(sym, np.nan)) if mcap.get(sym, 0) > 0 else np.nan,
                             "log_price": math.log10(max(float(x.loc[ts, "close"]), 0.01)),
                             "pre_session": int(ts.time() < S.dtime(9, 30))})
        if n % 200 == 0:
            log.info("setups: %d/%d symbols, %d trades so far", n, len(syms), len(rows))
    return pd.DataFrame(rows).sort_values("entry_ts").reset_index(drop=True)


def walk_forward(t: pd.DataFrame, min_days: int = 15, every: int = 5) -> pd.Series:
    import lightgbm as lgb
    days = sorted(t["date"].unique())
    pred = pd.Series(np.nan, index=t.index)
    model = None
    for k, d in enumerate(days):
        if k < min_days:
            continue
        if model is None or (k - min_days) % every == 0:
            tr = t[t["date"] < d]
            y = tr["R"].clip(-1.5, 6)
            model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
                                      random_state=42).fit(tr[FEATURES], y)
        m = t["date"] == d
        pred[m] = model.predict(t.loc[m, FEATURES])
    return pred


def summary(x: pd.DataFrame, label: str) -> dict:
    if x.empty:
        return {"set": label, "trades": 0}
    day = x.groupby("date")["R"].sum()
    t = day.mean() / day.std() * math.sqrt(len(day)) if len(day) > 2 and day.std() > 0 else np.nan
    return {"set": label, "trades": len(x), "days": x["date"].nunique(), "win%": round((x["R"] > 0).mean() * 100, 1),
            "avg_R": round(x["R"].mean(), 3), "total_R": round(x["R"].sum(), 1), "med_ret%": round(x["ret_pct"].median(), 2),
            "avg_ret%": round(x["ret_pct"].mean(), 2), "day_t": round(t, 2), "pos_days%": round((day > 0).mean() * 100, 0),
            "worst_trade%": round(x["ret_pct"].min(), 1), "gap_stops": int((x["outcome"] == "STOP_GAP").sum())}


def daily_cap(x: pd.DataFrame, k: int, col: str = "pred") -> pd.DataFrame:
    """First-come trades of the day, at most k (as live: alerts in time order)."""
    return x.sort_values("entry_ts").groupby("date").head(k)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    os.makedirs(OUT, exist_ok=True)
    uni = pd.read_parquet(D._p("universe.parquet"))
    daily = pd.read_parquet(D._p("daily.parquet"))
    t = build(daily, uni)
    t.to_parquet(os.path.join(OUT, "trades_all.parquet"), index=False)
    log.info("trades: %d over %d days, %d symbols", len(t), t["date"].nunique(), t["symbol"].nunique())
    rows = [summary(t, "rule: all setups"), summary(t[t.session == "pre"], "rule: pre-market"),
            summary(t[t.session == "regular"], "rule: regular hours"),
            summary(daily_cap(t, 5), "rule: first 5/day")]
    t["pred"] = walk_forward(t)
    oos = t[t["pred"].notna()]
    days = sorted(oos["date"].unique())
    half = days[len(days) // 2]
    a, b = oos[oos["date"] < half], oos[oos["date"] >= half]
    best = None
    for q in (0.5, 0.7, 0.8, 0.9):
        thr = a["pred"].quantile(q)
        s = summary(daily_cap(a[a["pred"] >= thr], 5), f"ML q{q}")
        if s.get("trades", 0) >= 30 and (best is None or s["avg_R"] > best[1]["avg_R"]):
            best = (q, s, thr)
    q, _, thr = best
    rows += [summary(a, "OOS-A (tune half): rule, all"), summary(daily_cap(a[a["pred"] >= thr], 5), f"OOS-A: ML >= q{q}, max 5/day"),
             summary(b, "OOS-B (check half): rule, all"), summary(daily_cap(b, 5), "OOS-B: rule, first 5/day"),
             summary(daily_cap(b[b["pred"] >= thr], 5), f"OOS-B: ML >= q{q} (thr {thr:+.2f}), max 5/day")]
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    log.info("RESULTS (R = multiples of the stop distance; costs included):\n%s", res.to_string(index=False))
    log.info("OOS-A %s..%s | OOS-B %s..%s", a.date.min().date(), a.date.max().date(), b.date.min().date(), b.date.max().date())
    by_out = t.groupby("outcome")["R"].agg(["count", "mean"]).round(2)
    log.info("exits:\n%s", by_out.to_string())
    res.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    t.to_parquet(os.path.join(OUT, "trades_scored.parquet"), index=False)
    import lightgbm as lgb
    final = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=42).fit(t[FEATURES], t["R"].clip(-1.5, 6))
    import joblib
    os.makedirs(os.path.join(D.ROOT, "us_scr", "model"), exist_ok=True)
    joblib.dump({"model": final, "features": FEATURES, "threshold": float(thr), "q": q,
                 "trained_through": str(t["date"].max().date()), "days": int(t["date"].nunique()), "trades": len(t)},
                os.path.join(D.ROOT, "us_scr", "model", "scr_model.joblib"), compress=3)
    imp = pd.Series(final.feature_importances_, index=FEATURES).sort_values(ascending=False)
    log.info("feature importance:\n%s", imp.head(10).to_string())


if __name__ == "__main__":
    main()
