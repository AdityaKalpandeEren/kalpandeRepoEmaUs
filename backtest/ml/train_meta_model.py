"""
Trains the meta-label classifier for model L_ML_META from
backtest/ml/data/dataset.csv (built by backtest/ml/build_dataset.py).

    python -m backtest.ml.train_meta_model
    python -m backtest.ml.train_meta_model --min-prob 0.6

VALIDATION IS WALK-FORWARD, NOT K-FOLD: rows are sorted by entry_time
and the last config.ML_META_TEST_FRAC of them (by TIME, not at random)
are held out as one untouched test block. A shuffled k-fold would let
the model train on trades that happened AFTER some of the trades in
its own "test" fold - leakage that would make this report lie to you
in exactly the way that made the 70-80% win-rate ask unsafe in the
first place. This script refuses to shuffle for that reason.

What gets printed is the entire point of the exercise: the OUT-OF-
SAMPLE win rate / expectancy of "take every candidate the base models
propose" vs "take only candidates the model rates above --min-prob".
If the filtered numbers are not better than the baseline on the held-
out block, the script says so plainly - a model that doesn't beat the
baseline out of sample is still saved (so you can inspect/backtest it),
but the printed verdict will NOT claim success.
"""
import argparse
import json
import math
import os

import pandas as pd

import config
from strategy.ml_features import FEATURE_COLUMNS


def _t_stat(r: pd.Series) -> float:
    n = len(r)
    if n <= 1:
        return 0.0
    std = r.std()
    if std == 0 or pd.isna(std):
        return 0.0
    return float(r.mean() / (std / math.sqrt(n)))


def _report_block(label: str, r: pd.Series) -> dict:
    n = len(r)
    if n == 0:
        return {"n": 0, "win_pct": 0.0, "exp_r": 0.0, "tot_r": 0.0, "t_stat": 0.0}
    win_pct = float((r > 0).mean() * 100)
    return {
        "n": n,
        "win_pct": round(win_pct, 1),
        "exp_r": round(float(r.mean()), 3),
        "tot_r": round(float(r.sum()), 2),
        "t_stat": round(_t_stat(r), 2),
    }


def _print_block(name: str, b: dict):
    print(f"  {name:<28}N={b['n']:<6} WIN%={b['win_pct']:<6} "
          f"EXP_R={b['exp_r']:<7} TOT_R={b['tot_r']:<8} t={b['t_stat']}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="backtest/ml/data/dataset.csv")
    p.add_argument("--out", default=config.ML_META_MODEL_PATH)
    p.add_argument("--test-frac", type=float, default=config.ML_META_TEST_FRAC)
    p.add_argument("--min-prob", type=float, default=config.ML_META_MIN_PROB,
                    help="Threshold used ONLY for the printed report; the saved model "
                         "always outputs a probability, config.ML_META_MIN_PROB decides "
                         "live/backtest firing at inference time.")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.dataset):
        print(f"No dataset at {args.dataset}. Run backtest/ml/build_dataset.py first.")
        return

    df = pd.read_csv(args.dataset)
    if len(df) < 200:
        print(f"Only {len(df)} rows - too thin to train/validate honestly. "
              f"Widen backtest/ml/build_dataset.py's --days/--max-symbols/--directions.")
        return

    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["entry_time"]).sort_values("entry_time").reset_index(drop=True)

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        print(f"Dataset missing feature columns {missing} - was it built with an older "
              f"strategy/ml_features.py? Rebuild it.")
        return

    split_idx = int(len(df) * (1 - args.test_frac))
    split_idx = max(split_idx, 1)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    print(f"Rows: {len(df)} total -> train {len(train_df)} "
          f"({train_df['entry_time'].min()} .. {train_df['entry_time'].max()}), "
          f"test {len(test_df)} ({test_df['entry_time'].min() if len(test_df) else '-'} "
          f".. {test_df['entry_time'].max() if len(test_df) else '-'})\n")

    if train_df["label"].nunique() < 2:
        print("Training block has only one class (all wins or all losses) - can't train. "
              "Widen the dataset.")
        return

    from sklearn.ensemble import HistGradientBoostingClassifier

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["label"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["label"]

    model = HistGradientBoostingClassifier(
        max_depth=4,
        max_iter=200,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(X_train, y_train)

    test_df = test_df.copy()
    test_df["prob"] = model.predict_proba(X_test)[:, 1]

    baseline = _report_block("baseline", test_df["r_multiple"])
    filtered_mask = test_df["prob"] >= args.min_prob
    filtered = _report_block("filtered", test_df.loc[filtered_mask, "r_multiple"])

    print("=" * 78)
    print(f"OUT-OF-SAMPLE REPORT (held-out block, never seen during training)")
    print("=" * 78)
    _print_block("ALL base-model candidates", baseline)
    _print_block(f"ML-filtered (p>={args.min_prob})", filtered)

    print()
    if filtered["n"] < 20:
        print(f"VERDICT: only {filtered['n']} test-set trades cleared p>={args.min_prob} - "
              f"too few to trust this threshold. Lower --min-prob, widen the dataset, or "
              f"treat this run as inconclusive.")
    elif filtered["exp_r"] > baseline["exp_r"] and filtered["win_pct"] > baseline["win_pct"]:
        print(f"VERDICT: filter improved BOTH win rate ({baseline['win_pct']}% -> "
              f"{filtered['win_pct']}%) and expectancy ({baseline['exp_r']}R -> "
              f"{filtered['exp_r']}R) out of sample. Still only {filtered['n']} test "
              f"trades - re-validate on a later window before trusting it further, and "
              f"note the t-stat above before treating either number as decided.")
    elif filtered["exp_r"] > baseline["exp_r"]:
        print(f"VERDICT: filter improved expectancy ({baseline['exp_r']}R -> "
              f"{filtered['exp_r']}R) but not win rate. That's a legitimate and common "
              f"outcome - expectancy is what pays, not win rate - but if your goal is "
              f"specifically win rate, this threshold isn't achieving it out of sample.")
    else:
        print(f"VERDICT: the filter did NOT beat the baseline out of sample "
              f"(EXP_R {filtered['exp_r']}R vs baseline {baseline['exp_r']}R). Do not "
              f"trust this model's win-rate lift; the in-sample fit does not generalize "
              f"at this threshold/dataset size. Try --min-prob sweep below, more data, "
              f"or accept the base models' own numbers.")

    print("\nThreshold sweep on the SAME held-out block (for picking config.ML_META_MIN_PROB):")
    print(f"{'PROB>=':>8}{'N':>8}{'WIN%':>8}{'EXP_R':>9}{'TOT_R':>9}{'t':>7}")
    for thresh in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
        sub = test_df.loc[test_df["prob"] >= thresh, "r_multiple"]
        b = _report_block("", sub)
        print(f"{thresh:>8.2f}{b['n']:>8}{b['win_pct']:>8}{b['exp_r']:>9}{b['tot_r']:>9}{b['t_stat']:>7}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import joblib
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS}, args.out)

    meta = {
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "dataset": args.dataset,
        "rows_total": len(df), "rows_train": len(train_df), "rows_test": len(test_df),
        "train_range": [str(train_df["entry_time"].min()), str(train_df["entry_time"].max())],
        "test_range": [str(test_df["entry_time"].min()) if len(test_df) else None,
                        str(test_df["entry_time"].max()) if len(test_df) else None],
        "baseline_oos": baseline, "filtered_oos_at_min_prob": filtered,
        "min_prob_used_for_report": args.min_prob,
    }
    meta_path = os.path.splitext(args.out)[0] + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved model : {args.out}")
    print(f"Saved report: {meta_path}")
    print("\nNext: python -m backtest.run_research --strategies L_ML_META --days 45 "
          "--directions long,short   (this is a FRESH backtest run, still on the same ~60-day "
          "Yahoo window as training - it is not independent confirmation, just a consistency "
          "check against the report above using run_research's own metrics.)")


if __name__ == "__main__":
    main()
