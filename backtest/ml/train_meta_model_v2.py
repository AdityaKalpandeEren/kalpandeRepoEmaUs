"""
Trains the classifier for model L_ML_META_V2 from config.ML_V2_DATASET_PATH
(built by backtest/ml/build_dataset_v2.py).

    python -m backtest.ml.train_meta_model_v2
    python -m backtest.ml.train_meta_model_v2 --directions long

THREE-WAY WALK-FORWARD SPLIT, BY TRADING DATE (not by row, not shuffled):

    |------------ TRAIN ------------|---- VAL ----|---- TEST ----|
                                     ^ choose model   ^ reported once,
                                       + threshold      never tuned on

  V1 picked nothing on held-out data - it reported a sweep over its own
  test block, which invites choosing the threshold that happened to look
  best there. V2 chooses the model configuration AND the probability
  threshold on the VALIDATION block only, then reports the TEST block
  once with those choices frozen. Splitting on dates (not rows) keeps
  one day's correlated trades from landing on both sides of a split.

EVALUATION MIMICS HOW THE MODEL TRADES. Several base models often fire
on the same bar; the live model scores them all and takes only the best
one, then observes the same per-symbol/day cooldown and trade cap as
every research model. Scoring every candidate row independently (as V1's
report did) would count one bar's opportunity up to 12 times. The
numbers here go through that same selection before any win rate is
computed.

ABLATION: the same model family is also trained on the V1-style
features alone (no market context / catalyst features). If the full
feature set doesn't beat that on validation, the extra features are not
helping and the report says so.

The deployed model is refit on ALL rows (train+val+test) with the frozen
configuration and threshold - the test report describes the procedure,
the final fit just gives it the most recent data.
"""
import argparse
import json
import math
import os

import numpy as np
import pandas as pd

import config
from strategy.ml_features_v2 import FEATURE_COLUMNS_V2, CONTEXT_COLUMNS, GEOMETRY_COLUMNS

# Model inputs: every V2 feature except config.ML_V2_EXCLUDE_FEATURES
# (day-level regime values that act as date labels on ~40 days of data).
MODEL_COLUMNS = [c for c in FEATURE_COLUMNS_V2 if c not in config.ML_V2_EXCLUDE_FEATURES]
BASE_ONLY_COLUMNS = [c for c in FEATURE_COLUMNS_V2
                     if c not in CONTEXT_COLUMNS and c not in GEOMETRY_COLUMNS]
THRESHOLDS = [round(x, 3) for x in np.arange(0.30, 0.801, 0.025)]


# ═══════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════

def _t_stat(r) -> float:
    r = np.asarray(r, dtype=float)
    n = len(r)
    if n <= 1:
        return 0.0
    sd = r.std(ddof=1)
    return 0.0 if sd == 0 or np.isnan(sd) else float(r.mean() / (sd / math.sqrt(n)))


def block(r) -> dict:
    r = np.asarray(r, dtype=float)
    if not len(r):
        return {"n": 0, "win_pct": 0.0, "exp_r": 0.0, "tot_r": 0.0, "t": 0.0, "pf": 0.0}
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    return {
        "n": int(len(r)),
        "win_pct": round(float((r > 0).mean() * 100), 1),
        "exp_r": round(float(r.mean()), 3),
        "tot_r": round(float(r.sum()), 1),
        "t": round(_t_stat(r), 2),
        "pf": round(float(gains / losses), 2) if losses > 0 else float("inf"),
    }


def day_block(sel: pd.DataFrame) -> dict:
    """Day-level view. Trades on the same day share one market, so they
    are NOT independent - the per-trade t-stat overstates confidence.
    Here each trading day is one observation (its mean R)."""
    if sel.empty:
        return {"days": 0, "days_pos": 0, "day_t": 0.0, "ex_best_day_exp_r": 0.0}
    per_day = sel.groupby("date")["r_multiple"]
    means, sums = per_day.mean(), per_day.sum()
    ex_best = sel[sel["date"] != sums.idxmax()]["r_multiple"]
    return {
        "days": int(len(means)),
        "days_pos": int((means > 0).sum()),
        "day_t": round(_t_stat(means.values), 2),
        "ex_best_day_exp_r": round(float(ex_best.mean()), 3) if len(ex_best) else 0.0,
    }


def fmt_days(name, d) -> str:
    return (f"  {name:<34}days+={d['days_pos']}/{d['days']}  day-t={d['day_t']}  "
            f"exp without best day={d['ex_best_day_exp_r']}R")


def fmt(name, b) -> str:
    return (f"  {name:<34}N={b['n']:<6}WIN%={b['win_pct']:<6}EXP_R={b['exp_r']:<8}"
            f"TOT_R={b['tot_r']:<9}PF={b['pf']:<6}t={b['t']}")


# ═══════════════════════════════════════════════════════════════════
# Trade selection - what the model would actually have taken
# ═══════════════════════════════════════════════════════════════════

def select_trades(df: pd.DataFrame, prob: np.ndarray, threshold: float) -> pd.DataFrame:
    """Apply the live selection rules to scored candidates:
    best candidate per (symbol, bar, direction) -> must clear threshold
    -> per (symbol, day, direction): cooldown + max trades/day
    -> per day across all symbols: config.ML_V2_MAX_TRADES_PER_DAY."""
    d = df.assign(prob=prob)
    d = d[d["prob"] >= threshold]
    if d.empty:
        return d
    d = d.sort_values("prob", ascending=False, kind="mergesort").drop_duplicates(
        ["symbol", "entry_time", "direction"], keep="first")
    d = d.sort_values("ts", kind="mergesort")
    cooldown = pd.Timedelta(minutes=config.SIGNAL_COOLDOWN_CANDLES * config.CANDLE_INTERVAL_MINUTES)
    keep = []
    last, count = {}, {}
    for idx, sym, day, direc, ts in zip(d.index, d["symbol"], d["date"], d["direction"], d["ts"]):
        key = (sym, day, direc)
        if count.get(key, 0) >= config.MAX_TRADES_PER_SYMBOL_DAY:
            continue
        if key in last and ts - last[key] < cooldown:
            continue
        keep.append(idx)
        last[key] = ts
        count[key] = count.get(key, 0) + 1
    d = d.loc[keep]
    # Portfolio cap: the first N trades of each day, in time order.
    return d.groupby("date", sort=False).head(config.ML_V2_MAX_TRADES_PER_DAY)


def baseline_trades(df: pd.DataFrame) -> pd.DataFrame:
    """'No model' reference under the SAME selection rules: every bar's
    candidate taken (first-listed model wins ties), same cooldown/cap."""
    return select_trades(df, np.ones(len(df)), 0.0)


# ═══════════════════════════════════════════════════════════════════
# Model family
# ═══════════════════════════════════════════════════════════════════

def model_configs():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    # Deliberately small and heavily regularised: ~60 days of data is one
    # market regime, and the enemy is memorising it. early_stopping is off
    # because its internal split is random, which would leak time.
    return {
        "hgb_shallow": lambda: HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.03, max_iter=250, min_samples_leaf=150,
            l2_regularization=5.0, max_features=0.5, early_stopping=False, random_state=42),
        "hgb_medium": lambda: HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.03, max_iter=300, min_samples_leaf=80,
            l2_regularization=3.0, max_features=0.5, early_stopping=False, random_state=42),
        "hgb_leafy": lambda: HistGradientBoostingClassifier(
            max_leaf_nodes=12, learning_rate=0.05, max_iter=200, min_samples_leaf=250,
            l2_regularization=10.0, max_features=0.4, early_stopping=False, random_state=42),
        "logreg": lambda: make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(C=0.05, max_iter=2000)),
    }


def best_threshold(df_val, prob_val, min_trades):
    """Threshold maximising the t-stat of selected trades' R on the
    validation block (rewards expectancy AND sample size, so a lucky
    handful of trades can't win), subject to min_trades."""
    best = None
    for th in THRESHOLDS:
        sel = select_trades(df_val, prob_val, th)
        if len(sel) < min_trades or sel["date"].nunique() < config.ML_V2_MIN_DAYS_FOR_THRESHOLD:
            continue
        b = block(sel["r_multiple"])
        if best is None or b["t"] > best[1]["t"]:
            best = (th, b)
    return best


def fit(cfg_factory, X, y):
    m = cfg_factory()
    m.fit(X, y)
    return m


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=config.ML_V2_DATASET_PATH)
    p.add_argument("--out", default=config.ML_V2_MODEL_PATH)
    p.add_argument("--directions", default="long,short",
                   help="Which directions to train/evaluate on. The live bot trades "
                        "long-only; a long-only model is 'long'.")
    p.add_argument("--val-frac", type=float, default=config.ML_V2_VAL_FRAC)
    p.add_argument("--test-frac", type=float, default=config.ML_V2_TEST_FRAC)
    p.add_argument("--min-trades", type=int, default=config.ML_V2_MIN_TRADES_FOR_THRESHOLD)
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.dataset):
        print(f"No dataset at {args.dataset}. Run backtest/ml/build_dataset_v2.py first.")
        return
    df = pd.read_csv(args.dataset)
    directions = [d.strip() for d in args.directions.split(",") if d.strip()]
    df = df[df["direction"].isin(directions)].copy()
    df["ts"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    missing = [c for c in FEATURE_COLUMNS_V2 if c not in df.columns]
    if missing:
        print(f"Dataset missing columns {missing[:5]}... - rebuild it with build_dataset_v2.py.")
        return
    if len(df) < 1000:
        print(f"Only {len(df)} rows - too thin for an honest 3-way split. Widen the dataset.")
        return

    dates = sorted(df["date"].unique())
    n_d = len(dates)
    n_test = max(1, int(round(n_d * args.test_frac)))
    n_val = max(1, int(round(n_d * args.val_frac)))
    train_dates = set(dates[: n_d - n_val - n_test])
    val_dates = set(dates[n_d - n_val - n_test: n_d - n_test])
    test_dates = set(dates[n_d - n_test:])
    tr = df[df["date"].isin(train_dates)]
    va = df[df["date"].isin(val_dates)]
    te = df[df["date"].isin(test_dates)]
    print(f"Directions: {directions} | rows {len(df)} | {n_d} trading days, "
          f"{df['symbol'].nunique()} symbols")
    print(f"  TRAIN {len(tr):>6} rows  {min(train_dates)} .. {max(train_dates)}  ({len(train_dates)} days)")
    print(f"  VAL   {len(va):>6} rows  {min(val_dates)} .. {max(val_dates)}  ({len(val_dates)} days)")
    print(f"  TEST  {len(te):>6} rows  {min(test_dates)} .. {max(test_dates)}  ({len(test_dates)} days)\n")

    # ---------------- model + feature-set selection on VAL ----------------
    configs = model_configs()
    results = []
    print("=" * 96)
    print("SELECTION ON VALIDATION BLOCK (train -> val)")
    print("=" * 96)
    print(fmt("baseline (take every bar, no model)", block(baseline_trades(va)["r_multiple"])))
    for fs_name, cols in (("full_v2", MODEL_COLUMNS), ("base_only", BASE_ONLY_COLUMNS)):
        for cfg_name, factory in configs.items():
            m = fit(factory, tr[cols], tr["label"])
            p_val = m.predict_proba(va[cols])[:, 1]
            bt = best_threshold(va, p_val, args.min_trades)
            if bt is None:
                print(f"  {fs_name}/{cfg_name:<12} no threshold keeps >= {args.min_trades} val trades")
                continue
            th, b = bt
            results.append((fs_name, cfg_name, cols, th, b))
            print(fmt(f"{fs_name}/{cfg_name} @p>={th}", b))

    if not results:
        print("\nNo configuration produced a usable threshold on validation. Nothing saved.")
        return
    fs_best, cfg_best, cols_best, th_best, b_best = max(results, key=lambda r: r[4]["t"])
    best_base_only = max((r for r in results if r[0] == "base_only"), key=lambda r: r[4]["t"], default=None)
    print(f"\nChosen on validation: {fs_best}/{cfg_best}, threshold p>={th_best} "
          f"(val t={b_best['t']}, exp {b_best['exp_r']}R over {b_best['n']} trades)")
    if best_base_only is not None and fs_best == "full_v2":
        print(f"Context/catalyst features vs base-only on validation: t {b_best['t']} vs "
              f"{best_base_only[4]['t']}, exp {b_best['exp_r']}R vs {best_base_only[4]['exp_r']}R")
    elif fs_best == "base_only":
        print("NOTE: the base-only feature set won on validation - the market-context and "
              "catalyst features did NOT add measurable value on this data.")

    # ---------------- untouched TEST ----------------
    factory = configs[cfg_best]
    trva = pd.concat([tr, va])
    m_trva = fit(factory, trva[cols_best], trva["label"])
    p_test = m_trva.predict_proba(te[cols_best])[:, 1]
    m_tr = fit(factory, tr[cols_best], tr["label"])
    p_test_tronly = m_tr.predict_proba(te[cols_best])[:, 1]

    base_te = baseline_trades(te)
    sel_te = select_trades(te, p_test, th_best)
    sel_te_tronly = select_trades(te, p_test_tronly, th_best)
    b_base, b_sel = block(base_te["r_multiple"]), block(sel_te["r_multiple"])

    print("\n" + "=" * 96)
    print("OUT-OF-SAMPLE TEST BLOCK (never used for any choice above)")
    print("=" * 96)
    print(fmt("baseline (every bar, no model)", b_base))
    print(fmt(f"V2 {cfg_best} p>={th_best} (fit train+val)", b_sel))
    print(fmt(f"V2 {cfg_best} p>={th_best} (fit train only)", block(sel_te_tronly["r_multiple"])))
    d_base, d_sel = day_block(base_te), day_block(sel_te)
    print(fmt_days("baseline by day", d_base))
    print(fmt_days("V2 by day", d_sel))
    for direc in directions:
        s = sel_te[sel_te["direction"] == direc]["r_multiple"]
        if len(s):
            print(fmt(f"   of which {direc}", block(s)))
    if len(sel_te):
        print("\n  Exit mix of selected test trades:",
              sel_te["outcome"].value_counts().to_dict())
        print("  Source models picked:", sel_te["source_strategy"].value_counts().head(6).to_dict())

    print()
    # Significance is judged at the DAY level (see day_block): a result
    # carried by one day, or positive on too few days, is not an edge.
    consistent = (d_sel["days"] >= 4 and d_sel["days_pos"] > d_sel["days"] / 2
                  and d_sel["ex_best_day_exp_r"] > 0)
    if b_sel["n"] < 20:
        verdict = (f"INCONCLUSIVE: only {b_sel['n']} test trades cleared the threshold - too few "
                   f"to judge. Don't trade it; gather more data.")
    elif b_sel["exp_r"] > 0 and d_sel["day_t"] >= 2.0 and consistent:
        verdict = (f"POSITIVE and consistent across days on the test block: {b_sel['win_pct']}% "
                   f"win, {b_sel['exp_r']}R/trade, {d_sel['days_pos']}/{d_sel['days']} days up, "
                   f"day-t={d_sel['day_t']}. Still ONE market period - paper-trade it before "
                   f"real money.")
    elif b_sel["exp_r"] > 0:
        verdict = (f"POSITIVE but NOT proven: {b_sel['win_pct']}% win, {b_sel['exp_r']}R/trade, "
                   f"but {d_sel['days_pos']}/{d_sel['days']} days up, day-t={d_sel['day_t']}, "
                   f"{d_sel['ex_best_day_exp_r']}R without its best day. Could be noise. "
                   f"Paper-trade only.")
    elif b_sel["exp_r"] > b_base["exp_r"]:
        verdict = (f"IMPROVES on baseline ({b_base['exp_r']}R -> {b_sel['exp_r']}R) but is still "
                   f"NEGATIVE expectancy. Better than taking every signal; not a money-maker. "
                   f"Do not trade it.")
    else:
        verdict = (f"FAILED: does not beat the baseline out of sample ({b_sel['exp_r']}R vs "
                   f"{b_base['exp_r']}R). Do not trade it.")
    print("VERDICT: " + verdict)

    print("\nTest-block threshold sweep (DESCRIPTIVE ONLY - the threshold above was fixed on "
          "validation; do not re-pick it from this table):")
    print(f"{'PROB>=':>8}{'N':>7}{'WIN%':>8}{'EXP_R':>9}{'TOT_R':>9}{'t':>7}")
    for th in THRESHOLDS[::2]:
        b = block(select_trades(te, p_test, th)["r_multiple"])
        if b["n"]:
            print(f"{th:>8.3f}{b['n']:>7}{b['win_pct']:>8}{b['exp_r']:>9}{b['tot_r']:>9}{b['t']:>7}")

    # ---------------- what the model leans on (validation, permutation) ----------------
    try:
        from sklearn.inspection import permutation_importance
        m_imp = fit(factory, tr[cols_best], tr["label"])
        imp = permutation_importance(m_imp, va[cols_best], va["label"], n_repeats=5,
                                     random_state=42, scoring="roc_auc")
        order = np.argsort(-imp.importances_mean)[:15]
        print("\nTop features by permutation importance on validation (ROC-AUC drop):")
        for i in order:
            print(f"  {cols_best[i]:<24}{imp.importances_mean[i]:+.4f}")
    except Exception as e:
        print(f"(permutation importance skipped: {e!r})")

    # ---------------- final fit on everything, save ----------------
    final = fit(factory, df[cols_best], df["label"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    import joblib
    bundle = {
        "model": final,
        "feature_columns": list(cols_best),
        "threshold": th_best,
        "config": cfg_best,
        "feature_set": fs_best,
        "directions": directions,
        "trained_through": str(max(dates)),
    }
    joblib.dump(bundle, args.out)

    meta = {
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "dataset": args.dataset, "rows": len(df), "symbols": int(df["symbol"].nunique()),
        "directions": directions,
        "split_days": {"train": len(train_dates), "val": len(val_dates), "test": len(test_dates)},
        "test_range": [min(test_dates), max(test_dates)],
        "chosen": {"feature_set": fs_best, "config": cfg_best, "threshold": th_best},
        "val_selected": b_best, "test_baseline": b_base, "test_selected": b_sel,
        "test_baseline_days": d_base, "test_selected_days": d_sel,
        "max_trades_per_day": config.ML_V2_MAX_TRADES_PER_DAY,
        "verdict": verdict,
    }
    meta_path = os.path.splitext(args.out)[0] + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\nSaved model : {args.out}  (final fit on all {len(df)} rows, threshold {th_best})")
    print(f"Saved report: {meta_path}")


if __name__ == "__main__":
    main()
