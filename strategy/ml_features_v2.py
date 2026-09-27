"""
Feature extraction for model L_ML_META_V2 - the V1 candle features
(strategy/ml_features.py) plus market context and catalyst features
(strategy/market_context.py).

Imported by BOTH backtest/ml/build_dataset_v2.py and
strategy/strategies.py::model_l_ml_meta_v2, for the same train/serve
parity reason V1 documents.

Differences from V1's list:
  - `session_position` (candle count since 4:00 AM, so it silently
    meant something different pre-market vs regular hours) is replaced
    by `minutes_since_open`, measured from the 9:30 regular open.
  - Missing values stay NaN instead of being filled with 0 - the
    gradient-boosted trees handle NaN natively, and a 0 for "VIX
    unknown" would be read as "VIX is zero".
"""
import math

import numpy as np

import config
from strategy.ml_features import FEATURE_COLUMNS as V1_COLUMNS, extract_features

CONTEXT_COLUMNS = [
    "minutes_since_open",
    # volatility complex
    "vix", "vix_vs_sma50", "vix_chg_day", "vix_chg_30m", "vix_term",
    "vxn_vs_sma50", "vxn_chg_day",
    # index tape + breadth
    "qqq_ret_day", "qqq_ret_30m", "qqq_vs_vwap", "spy_ret_day",
    "qqq_vs_sma50_d", "spy_vs_sma50_d", "spy_vs_sma200_d",
    "breadth_pct50", "breadth_chg5",
    # symbol daily structure
    "sym_ret5_d", "sym_ret20_d", "sym_atr_pct_d", "sym_vs_sma50_d",
    "sym_dist_high20", "sym_dist_prev_high",
    # today's catalyst footprint
    "gap_pct", "sym_ret_day", "sym_ret_since_open", "rs_vs_qqq_day", "rs_vs_qqq_30m",
    "tod_rvol", "day_range_pos",
    # earnings catalyst
    "earn_days_since", "earn_days_to_next", "earn_surprise", "earn_reaction_today",
    # direction-aligned (positive = tailwind for this trade)
    "align_qqq_day", "align_qqq_30m", "align_rs_day", "align_gap",
    "align_vix_chg_day", "align_earn_surprise",
]

# Trade-geometry features: how wide the base model's stop is. A 1.5R
# target is a very different ask on a 0.2% stop than on a 1.4% one.
GEOMETRY_COLUMNS = ["risk_pct", "risk_atr"]

FEATURE_COLUMNS_V2 = ([c for c in V1_COLUMNS if c != "session_position"]
                      + CONTEXT_COLUMNS + GEOMETRY_COLUMNS)


def _num(v) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return math.nan
    return v if np.isfinite(v) else math.nan


def extract_features_v2(row, df, direction, regime, comps, source_strategy,
                        context_feats: dict, signal) -> dict:
    """One V2 feature row. `context_feats` is MarketContext.features()
    for this symbol/candle/direction - computed once by the caller and
    shared across every base-model candidate on the same candle."""
    feats = extract_features(row, df, direction, regime, comps, source_strategy)
    feats.pop("session_position", None)
    feats.update(context_feats)
    entry = _num(signal.entry)
    risk = abs(entry - _num(signal.stop_loss))
    atr = _num(row.get(f"atr_{config.ATR_PERIOD}"))
    feats["risk_pct"] = risk / entry if entry else math.nan
    feats["risk_atr"] = risk / atr if atr and not math.isnan(atr) else math.nan
    return {c: _num(feats.get(c)) for c in FEATURE_COLUMNS_V2}
