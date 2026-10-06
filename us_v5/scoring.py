"""
US V5 scoring shared by research and the live job: per-day rank, EWMA
smoothing, ML ensemble prediction, factor blend. Imports only numpy /
pandas / the model libraries (no plotting, no scipy).
"""
from __future__ import annotations

import pandas as pd

from us_v5 import features as feat
from us_v5.core import gbm


def to_rank(s: pd.DataFrame) -> pd.DataFrame:
    s = s.copy()
    s["score"] = s.groupby("date")["score"].rank(pct=True)
    return s


def smooth(s: pd.DataFrame, halflife: float) -> pd.DataFrame:
    """Per-stock EWMA of the per-day score rank (causal). halflife <= 0 = none."""
    s = to_rank(s).sort_values(["entity", "date"])
    if halflife and halflife > 0:
        s["score"] = s.groupby("entity")["score"].transform(lambda x: x.ewm(halflife=halflife).mean())
    return s.sort_values(["date", "entity"]).reset_index(drop=True)


def blend(a: pd.DataFrame, b: pd.DataFrame, wa: float) -> pd.DataFrame:
    m = to_rank(a).merge(to_rank(b), on=["date", "entity"], suffixes=("_a", "_b"))
    m["score"] = wa * m["score_a"] + (1 - wa) * m["score_b"]
    return m[["date", "entity", "score"]]


def fit_ml(train: pd.DataFrame, feats: list[str], cfg: dict) -> dict:
    v5 = cfg["v5"]
    models = {}
    m, info = gbm.tune_and_fit("lgbm_reg", train, feats, cfg)
    models[f"lgbm_{v5['seeds'][0]}"] = m
    for sd in v5["seeds"][1:]:
        m2 = gbm._make("lgbm_reg", info["params"], sd)
        m2.set_params(n_estimators=max(int(info["n_trees"] * 1.1), 20))
        m2.fit(train[feats], train["fwd_rank"].values)
        models[f"lgbm_{sd}"] = m2
    mx, _ = gbm.tune_and_fit("xgb_reg", train, feats, cfg)
    models["xgb"] = mx
    return models


def predict_ml(models: dict, df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    parts = {}
    for name, m in models.items():
        p = pd.DataFrame({"date": df["date"].values, "entity": df["entity"].values, "score": m.predict(df[feats])})
        parts[name] = to_rank(p).set_index(["date", "entity"])["score"]
    lg = pd.concat([v for k, v in parts.items() if k.startswith("lgbm")], axis=1).mean(axis=1)
    return pd.concat([lg, parts["xgb"]], axis=1).mean(axis=1).rename("score").reset_index()


def raw_scores(ml: pd.DataFrame, df: pd.DataFrame, cfg: dict) -> dict:
    fac = pd.DataFrame({"date": df["date"].values, "entity": df["entity"].values,
                        "score": feat.factor_score(df).values})
    return {"v5_ml": ml, "v5_factor": fac, "v5_combo": blend(ml, fac, cfg["v5"]["combo_weight_ml"])}


def score_live(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Smoothed score of the selected candidate for every row of df (live)."""
    cfg = bundle["config"]
    ml = predict_ml(bundle["models"], df, bundle["features"])
    return smooth(raw_scores(ml, df, cfg)[bundle["selected"]], cfg["v5"]["halflife"])
