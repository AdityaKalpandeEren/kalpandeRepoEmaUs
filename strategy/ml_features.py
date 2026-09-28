"""
Shared feature extraction for the ML meta-label filter (model
L_ML_META in strategy/strategies.py).

Meta-labeling (Lopez de Prado): instead of asking ML to invent trade
entries from scratch on ~a few thousand rows - which is how you get a
model that memorizes noise - the existing rule-based models (A-K,
SCORE_ENGINE) keep proposing WHERE to trade, and a classifier is
trained on THIS engine's own historical trade outcomes to predict
whether each specific proposed trade wins. That's a much easier,
better-posed problem than "predict price", and it's the legitimate way
to raise win rate: by learning which setups the existing rules already
propose are worth taking, not by inventing a black box.

This module is imported by BOTH the dataset builder
(backtest/ml/build_dataset.py) and the live filter
(strategy/strategies.py::model_l_ml_meta), so the features seen at
train time and at inference time are computed by the exact same code.
That parity is the whole game - any drift between the two silently
invalidates the model (train/serve skew), and would not show up as an
error, only as quietly wrong live probabilities.
"""
import pandas as pd

import config

# The 12 base models this filter chooses among. A static list (not
# imported from strategy.strategies.ENTRY_MODELS) to avoid a circular
# import - strategies.py imports THIS module. Keep in sync with the
# keys of BASE_MODELS in strategy/strategies.py.
STRATEGY_NAMES = [
    "A_BREAKOUT", "B_BREAKOUT_CLOSE", "C_BREAKOUT_RETEST", "D_VWAP_RECLAIM",
    "E_VWAP_REJECTION", "F_EMA_PULLBACK", "G_CONFLUENCE", "H_ORB_VWAP",
    "I_EMA_STACK_BREAKOUT", "J_VWAP_BAND_REVERSION", "K_RSI2_REVERSION",
    "SCORE_ENGINE",
]

_NUMERIC_COLUMNS = [
    "rsi_2", "atr_pct_rank", "rvol",
    "vwap_dist_pct", "vwap_z", "vwap_slope",
    "ema_fast_slope", "ema_slow_slope",
    "trend_score", "momentum_score", "volume_score",
    "vwap_score", "volatility_score", "price_action_score",
    "trend_strength", "session_position",
    "direction_long", "regime_vol_high", "regime_vol_low",
    "regime_trend_bull", "regime_trend_bear",
]

# Full ordered feature list a trained model expects. build_dataset.py
# writes exactly these columns (plus bookkeeping columns like label/
# r_multiple/symbol that are NOT features); train_meta_model.py trains
# on exactly these; model_l_ml_meta feeds exactly these at inference.
FEATURE_COLUMNS = _NUMERIC_COLUMNS + [f"src_{name}" for name in STRATEGY_NAMES]


def _f(v, default=0.0) -> float:
    try:
        if v is None or pd.isna(v):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def extract_features(row, df, direction: str, regime, comps: dict, source_strategy: str) -> dict:
    """One feature row for one candidate trade.

    `comps` is the 6-component 0..1 score dict from
    strategy.strategies._component_scores - computed by the caller (not
    here) so this module has no import-time dependency on strategies.py.
    `source_strategy` is which base model proposed this candidate; it's
    one-hot encoded so the classifier can learn "trust G_CONFLUENCE more
    than A_BREAKOUT" if the data says so.
    """
    feats = {
        "rsi_2": _f(row.get(f"rsi_{config.RSI2_PERIOD}"), 50.0),
        "atr_pct_rank": _f(row.get("atr_pct_rank"), 0.5),
        "rvol": _f(row.get("rvol"), 1.0),
        "vwap_dist_pct": _f(row.get("vwap_dist_pct")),
        "vwap_z": _f(row.get("vwap_z")),
        "vwap_slope": _f(row.get("vwap_slope")),
        "ema_fast_slope": _f(row.get(f"ema_{config.EMA_FAST}_slope")),
        "ema_slow_slope": _f(row.get(f"ema_{config.EMA_SLOW}_slope")),
        "trend_score": _f(comps.get("trend")),
        "momentum_score": _f(comps.get("momentum")),
        "volume_score": _f(comps.get("volume")),
        "vwap_score": _f(comps.get("vwap")),
        "volatility_score": _f(comps.get("volatility")),
        "price_action_score": _f(comps.get("price_action")),
        "trend_strength": _f(getattr(regime, "trend_strength", 0.0)),
        "session_position": float(len(df)),
        "direction_long": 1.0 if direction == "long" else 0.0,
        "regime_vol_high": 1.0 if getattr(regime, "volatility", "") == "high" else 0.0,
        "regime_vol_low": 1.0 if getattr(regime, "volatility", "") == "low" else 0.0,
        "regime_trend_bull": 1.0 if getattr(regime, "is_bullish", False) else 0.0,
        "regime_trend_bear": 1.0 if getattr(regime, "is_bearish", False) else 0.0,
    }
    for name in STRATEGY_NAMES:
        feats[f"src_{name}"] = 1.0 if source_strategy == name else 0.0
    return feats


def feature_vector(feats: dict, columns=None) -> list:
    """Order a feature dict into the list a sklearn model expects."""
    columns = columns or FEATURE_COLUMNS
    return [_f(feats.get(c)) for c in columns]
