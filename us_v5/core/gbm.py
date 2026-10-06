"""
Gradient-boosting rankers and Optuna tuning.

  lgbm_rank   LightGBM LambdaRank, one query group per date, relevance 0..4
  lgbm_reg    LightGBM regression on the per-day percentile rank of the
              forward return (a ranked target: robust to outliers)
  xgb_reg     XGBoost regression on the same ranked target

Tuning: Optuna maximises the mean daily Spearman rank-IC on an inner
validation block carved from the END of the fold's training dates (with a
purge gap), so hyper-parameters never see the fold's test period.
"""
from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

log = logging.getLogger(__name__)

GBM_NAMES = ("lgbm_rank", "lgbm_reg", "xgb_reg")


def daily_ic(df: pd.DataFrame, score: pd.Series | np.ndarray, target: str = "fwd_ret") -> pd.Series:
    """Spearman rank IC per date."""
    d = pd.DataFrame({"date": df["date"].values, "s": np.asarray(score), "y": df[target].values}).dropna()
    return d.groupby("date").apply(lambda g: g["s"].corr(g["y"], method="spearman")
                                   if len(g) > 5 else np.nan, include_groups=False).dropna()


def _groups(df: pd.DataFrame) -> np.ndarray:
    return df.groupby("date", sort=False).size().values


def _suggest(trial, name: str) -> dict:
    if name.startswith("lgbm"):
        return {
            "num_leaves": trial.suggest_int("num_leaves", 7, 63, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 1000, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.9),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
        }
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 10, 500, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.9),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
    }


def _make(name: str, params: dict, seed: int):
    common = dict(n_estimators=2000, random_state=seed, n_jobs=6)
    if name == "lgbm_rank":
        return lgb.LGBMRanker(objective="lambdarank", lambdarank_truncation_level=20, subsample_freq=1,
                              verbose=-1, **common, **params)
    if name == "lgbm_reg":
        return lgb.LGBMRegressor(objective="regression", subsample_freq=1, verbose=-1, **common, **params)
    if name == "xgb_reg":
        return xgb.XGBRegressor(tree_method="hist", objective="reg:squarederror", **common, **params)
    raise ValueError(name)


def _target(name: str, df: pd.DataFrame) -> np.ndarray:
    return df["relevance"].astype(int).values if name == "lgbm_rank" else df["fwd_rank"].values


def fit(name: str, params: dict, train: pd.DataFrame, val: pd.DataFrame | None, features: list[str],
        seed: int, early_stopping: int = 50):
    """Fit one model; early-stops on `val` if given. Returns (model, best_iteration)."""
    m = _make(name, params, seed)
    Xtr, ytr = train[features], _target(name, train)
    if name == "lgbm_rank":
        kw = {"group": _groups(train)}
        if val is not None:
            kw.update(eval_set=[(val[features], _target(name, val))], eval_group=[_groups(val)],
                      eval_at=[10], callbacks=[lgb.early_stopping(early_stopping, verbose=False)])
        m.fit(Xtr, ytr, **kw)
        return m, m.best_iteration_ or m.n_estimators
    if name == "lgbm_reg":
        kw = {}
        if val is not None:
            kw.update(eval_set=[(val[features], _target(name, val))],
                      callbacks=[lgb.early_stopping(early_stopping, verbose=False)])
        m.fit(Xtr, ytr, **kw)
        return m, m.best_iteration_ or m.n_estimators
    if val is not None:
        m.set_params(early_stopping_rounds=early_stopping)
        m.fit(Xtr, ytr, eval_set=[(val[features], _target(name, val))], verbose=False)
        return m, (m.best_iteration or 0) + 1
    m.fit(Xtr, ytr, verbose=False)
    return m, m.n_estimators


def predict(model, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict(df[features])


def tune_and_fit(name: str, train: pd.DataFrame, features: list[str], cfg: dict) -> tuple[object, dict]:
    """Optuna inside `train` only, then refit on all of `train` with the best
    params and the tuned number of trees. Returns (model, info)."""
    import optuna                                   # research-only dependency (lazy)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    seed = cfg["seed"]
    v = cfg["validation"]
    dates = np.sort(train["date"].unique())
    cut = dates[int(len(dates) * (1 - v["inner_val_frac"]))]
    gap = cfg["label"]["horizon"] + 1 + v["embargo_days"]
    inner_tr = train[train["label_end"] < cut]
    cut_i = np.searchsorted(dates, cut)
    inner_va = train[train["date"] >= dates[min(cut_i + gap, len(dates) - 1)]]

    def objective(trial):
        p = _suggest(trial, name)
        m, _ = fit(name, p, inner_tr, inner_va, features, seed, cfg["models"]["early_stopping_rounds"])
        return float(daily_ic(inner_va, predict(m, inner_va, features)).mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=cfg["models"]["optuna_trials"],
                   timeout=cfg["models"]["optuna_timeout_s"])
    best = study.best_params
    _, n_trees = fit(name, best, inner_tr, inner_va, features, seed, cfg["models"]["early_stopping_rounds"])
    final = _make(name, {**best}, seed)
    final.set_params(n_estimators=max(int(n_trees * 1.1), 20))
    Xtr, ytr = train[features], _target(name, train)
    if name == "lgbm_rank":
        final.fit(Xtr, ytr, group=_groups(train))
    elif name == "xgb_reg":
        final.fit(Xtr, ytr, verbose=False)
    else:
        final.fit(Xtr, ytr)
    return final, {"params": best, "inner_ic": study.best_value, "n_trees": n_trees,
                   "trials": len(study.trials)}
