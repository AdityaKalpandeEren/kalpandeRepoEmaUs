"""
US V5.0 research run (port of NSE v5/run_v5.py).

    python -m us_v5.research                 # walk-forward 2017 -> mid-2025 + ablations
    python -m us_v5.research --holdout       # + the ONE-TIME clean holdout (Jul 2025 ->) + live model
    python -m us_v5.research --set v5.halflife=10

Pre-registered candidates (all EWMA-smoothed, all counted in the Deflated Sharpe):
  v5_ml      LightGBM (3 seeds) + XGBoost on the blended 5/10/20-day residual target
  v5_factor  52-week-high proximity + 12-1 momentum (no ML)
  v5_combo   50/50 rank blend of the two                  <- primary hypothesis
Selection rule: highest walk-forward net Sharpe. Baselines shown alongside:
SPY (total return) and an equal-weight portfolio of the same universe
(same survivorship bias as the model -> the fair test of ranking skill).

Honesty: the holdout (validation.holdout_start ->) has never been looked at
for the US; --holdout writes us_v5/results/holdout_used.json and refuses to
run twice without --force-holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import yaml

from us_v5 import data as data_mod
from us_v5 import features as feat
from us_v5.core import config as cfgmod
from us_v5.core import engine, gbm, html, metrics, walkforward
from us_v5.scoring import blend, fit_ml, predict_ml, raw_scores, smooth

log = logging.getLogger("us_v5")
CANDIDATES = ("v5_ml", "v5_factor", "v5_combo")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "us_v5", "model", "us_v5_model.joblib")


def _hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:10]


def wf_ml_scores(cfg, df, feats, out_dir) -> pd.DataFrame:
    f = os.path.join(out_dir, "wf_scores_v5_ml_raw.parquet")
    if os.path.exists(f):
        return pd.read_parquet(f)
    parts = []
    for fold in walkforward.folds(pd.DatetimeIndex(df["date"].unique()), cfg):
        train, test = walkforward.split(df, fold)
        walkforward.check_no_overlap(train, test)
        if test.empty:
            continue
        parts.append(predict_ml(fit_ml(train, feats, cfg), test, feats))
        log.info("v5_ml fold %s: train %d rows, test %d", fold.test_start.date(), len(train), len(test))
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(f, index=False)
    return out


# ─── backtest helpers ────────────────────────────────────────────────────

def backtest(cfg, df, prices, scores, start, end):
    info = df[["date", "entity", "vol_63", "industry"]]
    ctx = df.groupby("date")[["vix_pct_1y", "spy_dist_ma200"]].first()
    return engine.run(scores, prices, info, ctx, cfg, pd.Timestamp(start), pd.Timestamp(end))


def summarise(res, bench, cfg) -> dict:
    d = res.daily
    p = metrics.perf(d["ret"].iloc[1:], bench, cfg["report"]["risk_free"])
    years = len(d) / 252
    avg_eq = d["equity"].mean()
    p["Turnover_ann"] = d["turnover"].sum() / avg_eq / years / 2 if years > 0 else np.nan
    p["Cost_drag_ann"] = d["costs"].sum() / avg_eq / years if years > 0 else np.nan
    if not res.trades.empty:
        p["HitRate_trades"] = float((res.trades["ret"] > 0).mean())
        p["Trades"] = int(len(res.trades))
    return p


def ew_universe(panel: pd.DataFrame) -> pd.Series:
    """Equal-weight daily return of the universe members (gross, rebalanced daily)."""
    u = panel[panel["in_universe"]]
    return u.groupby("date")["ret_cc"].mean()


def ic_report(df: pd.DataFrame, s: pd.DataFrame) -> dict:
    m = df[["date", "entity", "fwd_5", "fwd_20", "target"]].merge(s, on=["date", "entity"])
    out = {}
    for col, step in (("fwd_5", 5), ("fwd_20", 20), ("target", 20)):
        ic = gbm.daily_ic(m, m["score"].values, target=col)
        no = ic.iloc[::step]
        out[f"IC_{col}"] = float(ic.mean())
        out[f"t_{col}"] = float(no.mean() / no.std() * np.sqrt(len(no))) if len(no) > 2 else np.nan
    return out


def record_trials(cfg: dict, rets: dict) -> list[float]:
    f = os.path.join(cfgmod.path(cfg, "results"), "trials.json")
    trials = json.load(open(f)) if os.path.exists(f) else {}
    h = _hash({k: v for k, v in cfg.items() if k != "paths"})
    for name, r in rets.items():
        trials[f"{h}:{name}"] = metrics.daily_sharpe(r)
    json.dump(trials, open(f, "w"), indent=1)
    return list(trials.values())


def bench_row(name: str, r: pd.Series, spy: pd.Series, cfg: dict) -> dict:
    p = metrics.perf(r, spy, cfg["report"]["risk_free"])
    return {"model": name, **p}


# ─── main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--end", default="2026-10-02")
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--force-holdout", action="store_true")
    a = ap.parse_args()
    overrides = {"data.end": a.end}
    for kv in a.set:
        k, v = kv.split("=", 1)
        overrides[k] = yaml.safe_load(v)
    cfg = cfgmod.load(overrides)
    cfgmod.setup_logging()
    np.random.seed(cfg["seed"])
    run = "run_" + _hash(cfg)
    out_dir = cfgmod.path(cfg, "results", run)
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=1, default=str)

    data = data_mod.load_data(cfg)
    dsf = os.path.join(cfgmod.path(cfg, "cache"), f"dataset_{_hash({k: cfg[k] for k in ('universe', 'label', 'features', 'data')})}.parquet")
    if os.path.exists(dsf):
        df = pd.read_parquet(dsf)
    else:
        df = feat.build(data, cfg)
        df.to_parquet(dsf, index=False)
    feats = feat.feature_columns(df)
    log.info("dataset %s, %d features, %s .. %s", df.shape, len(feats), df["date"].min().date(), df["date"].max().date())
    prices = engine.prepare_prices(data["panel"], set(df["entity"]))
    spy = data["context"]["SPY"].pct_change(fill_method=None)
    ew = ew_universe(data["panel"])
    hs, first = pd.Timestamp(cfg["validation"]["holdout_start"]), pd.Timestamp(cfg["validation"]["first_test"])
    end_wf = hs - pd.Timedelta(days=1)
    wf_df = df[df["date"] < hs]
    wf_test = wf_df[wf_df["date"] >= first]

    ml_raw = wf_ml_scores(cfg, wf_df, feats, out_dir)
    raw = raw_scores(ml_raw, wf_test.merge(ml_raw[["date", "entity"]], on=["date", "entity"]), cfg)
    raw["v5_ml"] = ml_raw
    scores = {k: smooth(v, cfg["v5"]["halflife"]) for k, v in raw.items()}

    rows, rets = [], {}
    for name, s in scores.items():
        res = backtest(cfg, df, prices, s, first, end_wf)
        rets[name] = res.daily["ret"].iloc[1:]
        rows.append({"model": name, **summarise(res, spy, cfg), **ic_report(wf_df, s)})
        res.daily.to_csv(os.path.join(out_dir, f"wf_daily_{name}.csv"))
        res.trades.to_csv(os.path.join(out_dir, f"wf_trades_{name}.csv"), index=False)
    table = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
    best = table.iloc[0]["model"]
    idx = rets[best].index
    benches = pd.DataFrame([bench_row("SPY (total return)", spy.reindex(idx).fillna(0), spy, cfg),
                            bench_row("EW universe (gross)", ew.reindex(idx).fillna(0), spy, cfg)])

    abl = []
    for label, over, hl in [("combo, no smoothing", {}, 0), ("combo, half-life 2", {}, 2),
                            ("combo, half-life 10", {}, 10), ("combo, top 10", {"portfolio.top_n": 10}, None),
                            ("combo, top 30", {"portfolio.top_n": 30}, None),
                            ("combo, no drawdown breaker", {"risk.dd_breaker": 9.9}, None),
                            ("combo, slippage x2", {"costs.slippage_bps": cfg["costs"]["slippage_bps"] * 2}, None)]:
        c2 = cfgmod.load({**overrides, **over})
        s2 = smooth(raw["v5_combo"], cfg["v5"]["halflife"] if hl is None else hl)
        r2 = backtest(c2, df, prices, s2, first, end_wf)
        rets[f"abl: {label}"] = r2.daily["ret"].iloc[1:]
        p2 = summarise(r2, spy, c2)
        abl.append({"variant": label, **{k: p2.get(k) for k in ("CAGR", "Sharpe", "MaxDD", "Turnover_ann", "Cost_drag_ann")}})
    abl = pd.DataFrame(abl)
    trial_sr = record_trials(cfg, rets)
    dsr = metrics.deflated_sharpe(rets[best], trial_sr)
    table.to_csv(os.path.join(out_dir, "walkforward_summary.csv"), index=False)
    abl.to_csv(os.path.join(out_dir, "ablations.csv"), index=False)
    cols = ["model", "CAGR", "Sharpe", "MaxDD", "Alpha_ann", "Beta", "Turnover_ann", "Cost_drag_ann",
            "IC_fwd_5", "t_fwd_5", "IC_fwd_20", "t_fwd_20"]
    log.info("walk-forward %s -> %s:\n%s", first.date(), end_wf.date(), table[cols].round(3).to_string(index=False))
    log.info("benchmarks:\n%s", benches[["model", "CAGR", "Sharpe", "MaxDD"]].round(3).to_string(index=False))
    log.info("ablations:\n%s", abl.round(3).to_string(index=False))
    log.info("selected %s; DSR %.3f over %d trials", best, dsr["DSR"], dsr["N_trials"])
    yearly = metrics.by_period(rets[best], spy, cfg["report"]["risk_free"])
    ew_y = ew.reindex(idx).fillna(0)
    yearly["EW_Return"] = [float((1 + ew_y[ew_y.index.year == int(y)]).prod() - 1) for y in yearly["period"]]
    log.info("per year (%s):\n%s", best, yearly[["period", "Return", "Bench_Return", "EW_Return", "Sharpe", "MaxDD"]].round(3).to_string(index=False))
    yearly.to_csv(os.path.join(out_dir, "per_year.csv"), index=False)
    benches.to_csv(os.path.join(out_dir, "benchmarks.csv"), index=False)

    ho = None
    if a.holdout:
        ho = run_holdout(cfg, df, feats, prices, spy, ew, best, out_dir, a.force_holdout)
    report(cfg, out_dir, table, benches, abl, dsr, rets, best, spy, ew, yearly, ho)


def run_holdout(cfg, df, feats, prices, spy, ew, best, out_dir, force):
    marker = os.path.join(cfgmod.path(cfg, "results"), "holdout_used.json")
    repeated = os.path.exists(marker)
    if repeated and not force:
        raise SystemExit(f"Holdout already used ({open(marker).read().strip()}); pass --force-holdout to accept that.")
    train, test = walkforward.holdout_split(df, cfg)
    walkforward.check_no_overlap(train, test)
    final = fit_ml(train, feats, cfg)
    raw = raw_scores(predict_ml(final, test, feats), test, cfg)
    hs = pd.Timestamp(cfg["validation"]["holdout_start"])
    rows, rets = [], {}
    for name, s in raw.items():
        r = backtest(cfg, df, prices, smooth(s, cfg["v5"]["halflife"]), hs, df["date"].max())
        rets[name] = r.daily["ret"].iloc[1:]
        rows.append({"model": name, **summarise(r, spy, cfg)})
        r.daily.to_csv(os.path.join(out_dir, f"holdout_daily_{name}.csv"))
    idx = rets[best].index
    rows.append(bench_row("SPY (total return)", spy.reindex(idx).fillna(0), spy, cfg))
    rows.append(bench_row("EW universe (gross)", ew.reindex(idx).fillna(0), spy, cfg))
    t = pd.DataFrame(rows)
    json.dump({"used_at": datetime.now().isoformat(timespec="seconds"), "selected": best}, open(marker, "w"))
    log.info("HOLDOUT %s -> %s%s:\n%s", hs.date(), df["date"].max().date(), " (REPEATED)" if repeated else "",
             t[["model", "CAGR", "Sharpe", "MaxDD", "Alpha_ann", "TotalReturn"]].round(3).to_string(index=False))
    t.to_csv(os.path.join(out_dir, "holdout_summary.csv"), index=False)

    # live model: refit on ALL labelled data
    all_train = df[df["target"].notna()]
    live = fit_ml(all_train, feats, cfg)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"models": live, "features": feats, "config": cfg, "selected": best,
                 "trained_through": str(all_train["label_end"].max().date())}, MODEL_PATH, compress=3)
    log.info("live model -> %s (%.1f MB)", MODEL_PATH, os.path.getsize(MODEL_PATH) / 1e6)
    try:
        import shap
        X = test[feats].sample(min(2000, len(test)), random_state=cfg["seed"])
        sv = shap.TreeExplainer(live[f"lgbm_{cfg['v5']['seeds'][0]}"]).shap_values(X)
        share = pd.Series(np.abs(sv).mean(axis=0), index=feats).sort_values(ascending=False)
        share = share / share.sum()
        share.to_csv(os.path.join(out_dir, "shap_importance.csv"))
        log.info("SHAP top 10:\n%s", share.head(10).round(3).to_string())
    except Exception as e:
        log.warning("SHAP skipped: %s", e)
        share = None
    return {"table": t, "rets": rets, "repeated": repeated, "shap": share}


def report(cfg, out_dir, table, benches, abl, dsr, rets, best, spy, ew, yearly, ho):
    rf = cfg["report"]["risk_free"]
    secs = []
    p = metrics.perf(rets[best], spy, rf)
    kpis = "".join(f"<div class='kpi'><span class='muted'>{k}</span><b>{html._fmt(k, p.get(k))}</b></div>"
                   for k in ("CAGR", "Sharpe", "MaxDD", "Calmar", "Bench_CAGR", "Alpha_ann"))
    secs.append(("", f"<div class='kpis'>{kpis}</div>"))
    secs.append(("Walk-forward (out of sample, after costs)", html.table(table.round(4)) + html.table(benches.round(4))))
    secs.append(("Ablations of v5_combo", html.table(abl.round(4))))
    secs.append(("Deflated Sharpe", html.table(pd.DataFrame([{"model": best, **dsr}]))))
    curves = {n: rets[n] for n in CANDIDATES}
    curves["SPY (total return)"] = spy.reindex(rets[best].index).fillna(0)
    curves["EW universe"] = ew.reindex(rets[best].index).fillna(0)
    secs.append(("Equity vs SPY", html.equity_chart(curves, "Walk-forward equity (log)") + html.drawdown_chart(curves)))
    secs.append(("Per year - " + best, html.table(yearly.round(4))))
    if ho:
        stamp = "<div class='warn'>REPEATED HOLDOUT</div>" if ho["repeated"] else "<p class='muted'>First and only look at the holdout.</p>"
        secs.append((f"Holdout {cfg['validation']['holdout_start']} ->", stamp + html.table(ho["table"].round(4))))
        if ho["shap"] is not None:
            secs.append(("SHAP (live model)", html.bar_chart(ho["shap"].head(25), "mean |SHAP| share")))
    secs.append(("Limitations", "<ul><li>Survivorship: Yahoo has no data for delisted tickers; the candidate pool is today's "
                 "S&P 500 (point-in-time by Date added). Compare against the EW universe row, which shares the bias.</li>"
                 "<li>Fills at the next open +/- slippage; no taxes; idle cash earns nothing.</li></ul>"))
    html.render(os.path.join(out_dir, "report.html"), "US V5.0 swing ranking - research report", secs,
                f"Generated {datetime.now():%Y-%m-%d %H:%M} · run {os.path.basename(out_dir)}")
    log.info("report: %s", os.path.join(out_dir, "report.html"))


if __name__ == "__main__":
    main()
