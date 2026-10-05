"""
US V3 improvement research: which EXECUTION and STOP make the ORB + ML ranker
hold up out of sample? Research only - live is untouched.

    python -m us_v3.improve              # build (cached) + walk-forward every variant
    python -m us_v3.improve --rebuild    # rebuild the base breakout table

Base: every ORB breakout (research.symbol_days, ORB_ATR rows = features +
breakout level/bar). Each variant re-simulates the SAME breakouts:
  entry  level  stop-entry order resting at the breakout level (fills there,
                as research assumes) - live can do this by scoring at the
                START of each bar (all features are known then)
         next   what live does today: market entry at the open of the bar
                after the breakout bar
  stop   atrXX  XX% of daily ATR(14) from the entry
         orr    the other side of the opening-range bar
Exit: stop or the 15:55 close (research._sim: on the entry bar only a close
through the stop counts). 3 bps/side slippage, every R net.

Model: research.walk_forward (LightGBM, expanding window, retrain every 5
days) trained on each variant's own R. Portfolio: each day the top K by
score with score > 0 (optional RVOL >= 1 / long-only filters), equal risk.

Honesty: the out-of-sample days are split in two. DEV = first half (pick a
variant there), HOLDOUT = second half (incl. the live days Oct 1-2) - only
a variant that also holds up on HOLDOUT is worth taking live.
"""
from __future__ import annotations

import argparse
import os
from datetime import time as dtime

import numpy as np
import pandas as pd

from us_v3 import data as D
from us_v3 import live as L
from us_v3 import research as R

CACHE = os.path.join("backtest", "ml", "cache", "v3us_improve_base.parquet")
OUT_DIR = os.path.join("backtest", "results", "us_v3")
STOPS = {"atr10": 0.10, "atr20": 0.20, "atr35": 0.35, "atr50": 0.50, "orr": None}
ENTRIES = ("level", "next")


def build_base(syms: list[str]) -> pd.DataFrame:
    ctx = R.context()
    parts = []
    for s in syms:
        t = R.symbol_days(s, ctx)
        if len(t):
            parts.append(t[t["setup"] == "ORB_ATR"])
    df = pd.concat(parts, ignore_index=True)
    df["is_late"], df["stop_range"] = 0, 0
    df["bar"] = (df["mins"] // 5).astype(int)            # breakout bar index in the RTH session
    df = df.rename(columns={"entry": "level"})
    keep = R.FEATURES + ["symbol", "date", "level", "bar", "atr_d", "or_high", "or_low"]
    return df[keep].sort_values(["date", "symbol"]).reset_index(drop=True)


def simulate(base: pd.DataFrame) -> pd.DataFrame:
    """R of every breakout under every (entry, stop) variant."""
    out = {f"{e}_{s}": np.full(len(base), np.nan) for e in ENTRIES for s in STOPS}
    for sym, g in base.groupby("symbol"):
        x = D.load(sym)
        days = {d: v for d, v in x.groupby(x.index.date)}
        for idx, r in g.iterrows():
            day = days.get(r["date"].date())
            if day is None:
                continue
            rth = day[(day.index.time >= R.OPEN) & (day.index.time < R.CLOSE)]
            side, a = int(r["side"]), float(r["atr_d"])
            for e in ENTRIES:
                i = int(r["bar"]) + (0 if e == "level" else 1)
                if i >= len(rth):
                    continue
                entry = float(r["level"]) if e == "level" else float(rth["open"].iloc[i])
                for s, k in STOPS.items():
                    stop = entry - side * k * a if k else (r["or_low"] if side > 0 else r["or_high"])
                    risk = side * (entry - stop)
                    if risk <= 0:
                        continue
                    px, _, _ = R._sim(rth, i, side, entry, stop, dtime(15, 55))
                    out[f"{e}_{s}"][base.index.get_loc(idx)] = (side * (px - entry) - R.SLIP * (entry + px)) / risk
    return pd.concat([base, pd.DataFrame(out, index=base.index)], axis=1)


def evaluate(df: pd.DataFrame, k: int = 5, risk: float = 0.01) -> pd.DataFrame:
    days = sorted(df["date"].unique())
    rows = []
    preds = {}
    for v in [f"{e}_{s}" for e in ENTRIES for s in STOPS]:
        d = df[df[v].notna()].copy()
        d["R"] = d[v]
        d["pred"] = R.walk_forward(d)
        preds[v] = d
    oos_days = [x for x in days if any(x in set(p.loc[p["pred"].notna(), "date"]) for p in preds.values())]
    half = len(oos_days) // 2
    blocks = {"DEV": oos_days[:half], "HOLDOUT": oos_days[half:], "ALL_OOS": oos_days}

    def book(name, d, pick, order):
        sel = d[pick].sort_values(["date", order], ascending=[True, False]).groupby("date").head(k)
        for b, bd in blocks.items():
            t = sel[sel["date"].isin(bd)]
            daily = (t.groupby("date")["R"].sum() * risk).reindex(bd).fillna(0)
            st = R.stats(daily, t)
            rows.append({"variant": name, "block": b, **st})

    for v, d in preds.items():
        o = d[d["pred"].notna()]
        book(f"{v} ML", o, o["pred"] > 0, "pred")
        book(f"{v} ML rvol>=1", o, (o["pred"] > 0) & (o["rvol_open"] >= 1), "pred")
        book(f"{v} ML long", o, (o["pred"] > 0) & (o["side"] > 0), "pred")
        book(f"{v} rule rvol top", o, o["rvol_open"] >= 1, "rvol_open")   # Zarattini, no model
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    if a.rebuild or not os.path.exists(CACHE):
        base = build_base(L.symbols())
        base.to_parquet(CACHE)
    base = pd.read_parquet(CACHE)
    print(f"{len(base)} ORB breakouts on {base['date'].nunique()} days")
    df = simulate(base)
    res = evaluate(df, k=a.k)
    os.makedirs(OUT_DIR, exist_ok=True)
    res.to_csv(os.path.join(OUT_DIR, f"improve_k{a.k}.csv"), index=False)
    piv = res.pivot_table(index="variant", columns="block", values=["avgR", "day_t", "totR", "maxDD%", "trades"])
    pd.set_option("display.width", 250, "display.max_rows", 200)
    cols = [(m, b) for m in ("trades", "avgR", "day_t", "maxDD%") for b in ("DEV", "HOLDOUT")]
    print(piv[cols].sort_values(("day_t", "DEV"), ascending=False).round(2).to_string())


if __name__ == "__main__":
    main()
