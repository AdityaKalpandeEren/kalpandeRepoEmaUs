"""
Multi-strategy research engine.

Eight entry models plus a weighted scoring engine, each returning the
same StrategySignal object so the backtester can run them all over the
same candles and compare them on equal terms. Every model supports both
LONG and SHORT, and the two are NOT mirror images - see the note on
short asymmetry below.

Every model here shares one stop-loss convention: an ATR-scaled stop,
widened to respect structure where structure is nearby. Using one stop
rule across all models is deliberate. If each model had its own stop
logic, a win-rate difference between models would be partly a stop
difference, and you could not attribute the result to the entry itself.

WHAT THIS ENGINE DOES NOT DO, and why you should care:
  - It does not prove any of these models work. It measures them on a
    sample that, given Yahoo's 60-day intraday retention, is one market
    period. A model that wins over 40 trading days of one regime has
    not been validated; it has been described.
  - Testing 9 models on one dataset means the best-looking result is
    partly selection luck. Treat the ranking as a hypothesis to test on
    fresh data, never as a conclusion. See README.

SHORT ASYMMETRY (why shorts are not mirrored longs):
  - Downside moves are faster and more volatile than upside moves of
    the same size, so short stops are widened by
    config.SHORT_STOP_ATR_MULT_EXTRA.
  - Volume expansion means something different: selling climaxes
    frequently mark exhaustion, not continuation, so the short models
    require a HIGHER relative-volume bar than the longs
    (SHORT_RVOL_EXTRA) before treating volume as confirmation.
  - Shorts are not permitted in the first bars of the session, where
    upward drift and opening auctions dominate.
Whether those asymmetries actually help is an empirical question the
backtest can answer - they are config values, not hard-coded truths.
"""
from dataclasses import dataclass
from typing import Optional

import pandas as pd

import config
from strategy.regime import classify, is_tradeable, Regime


@dataclass
class StrategySignal:
    symbol: str
    strategy: str
    direction: str            # "long" | "short"
    entry: float              # reference price (the signal candle's close)
    stop_loss: float
    target: float
    regime: str
    trend_strength: float
    score: float              # 0..100; hard-rule models report their confirmation count scaled
    reason: str
    candle_time: object


def _f(v, default=0.0) -> float:
    try:
        if v is None or pd.isna(v):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _atr(row) -> float:
    return _f(row.get(f"atr_{config.ATR_PERIOD}"))


def _stop_and_target(row, direction: str, structural_level: float = None):
    """One stop convention for every model (see module docstring).

    Stop = ATR * multiplier from entry, extended to sit just beyond a
    structural level when that level is nearby. Target = stop distance *
    reward:risk. Returns (stop, target, risk) or None when the ATR is
    unusable or the implied risk breaches config.MAX_RISK_PCT.
    """
    entry = _f(row.get("close"))
    atr = _atr(row)
    if entry <= 0 or atr <= 0:
        return None

    mult = config.ATR_STOP_MULT
    if direction == "short":
        mult += config.SHORT_STOP_ATR_MULT_EXTRA

    if direction == "long":
        stop = entry - atr * mult
        if structural_level is not None and not pd.isna(structural_level):
            buffered = structural_level - atr * config.STRUCT_STOP_BUFFER_ATR
            # Only ever widen to structure, never tighten to it: a
            # structural level INSIDE the ATR stop would make the stop
            # artificially tight and flatter the win rate.
            if buffered < stop:
                stop = buffered
        risk = entry - stop
        target = entry + risk * config.RISK_REWARD_RATIO
    else:
        stop = entry + atr * mult
        if structural_level is not None and not pd.isna(structural_level):
            buffered = structural_level + atr * config.STRUCT_STOP_BUFFER_ATR
            if buffered > stop:
                stop = buffered
        risk = stop - entry
        target = entry - risk * config.RISK_REWARD_RATIO

    if risk <= 0:
        return None
    if risk / entry > config.MAX_RISK_PCT:
        return None   # stop too wide for this setup - skip rather than force it
    return stop, target, risk


def _rvol_threshold(direction: str) -> float:
    base = config.VOLUME_MULTIPLIER
    return base + (config.SHORT_RVOL_EXTRA if direction == "short" else 0.0)


def _build(symbol, name, direction, row, regime: Regime, reason, score, structural_level=None):
    st = _stop_and_target(row, direction, structural_level)
    if st is None:
        return None
    stop, target, _ = st
    return StrategySignal(
        symbol=symbol,
        strategy=name,
        direction=direction,
        entry=round(_f(row.get("close")), 4),
        stop_loss=round(stop, 4),
        target=round(target, 4),
        regime=regime.label,
        trend_strength=regime.trend_strength,
        score=round(score, 1),
        reason=reason,
        candle_time=row.get("timestamp"),
    )


def _session_position(df) -> int:
    """How many candles into the session we are. Used to block shorts
    in the opening candles and to skip the warm-up entirely."""
    return len(df)


# ═══════════════════════════════════════════════════════════════════
# ENTRY MODELS
# Each takes the enriched slice up to and including the signal candle
# and returns a StrategySignal or None. None of them may look at
# df.iloc[-1] beyond its own close - no future data.
# ═══════════════════════════════════════════════════════════════════

def model_a_breakout(symbol, df, direction, regime) -> Optional[StrategySignal]:
    """A. Immediate breakout: price trades through recent structure.
    Fires on the break itself, no close confirmation. Expected to be
    the noisiest of the breakout family - included precisely so the
    others have a baseline to beat."""
    row = df.iloc[-1]
    hi, lo = _f(row.get("struct_high")), _f(row.get("struct_low"))
    if direction == "long":
        if hi <= 0 or _f(row.get("high")) <= hi:
            return None
        return _build(symbol, "A_BREAKOUT", direction, row, regime,
                      f"high broke structure {hi:.2f}", 50, lo)
    if lo <= 0 or _f(row.get("low")) >= lo:
        return None
    return _build(symbol, "A_BREAKOUT", direction, row, regime,
                  f"low broke structure {lo:.2f}", 50, hi)


def model_b_breakout_close(symbol, df, direction, regime) -> Optional[StrategySignal]:
    """B. Breakout + candle close beyond the level. The standard
    remedy for A's false breaks: costs you part of the move in exchange
    for filtering wicks."""
    row = df.iloc[-1]
    hi, lo = _f(row.get("struct_high")), _f(row.get("struct_low"))
    close = _f(row.get("close"))
    if direction == "long":
        if hi <= 0 or close <= hi:
            return None
        return _build(symbol, "B_BREAKOUT_CLOSE", direction, row, regime,
                      f"closed above structure {hi:.2f}", 60, lo)
    if lo <= 0 or close >= lo:
        return None
    return _build(symbol, "B_BREAKOUT_CLOSE", direction, row, regime,
                  f"closed below structure {lo:.2f}", 60, hi)


def model_c_breakout_retest(symbol, df, direction, regime) -> Optional[StrategySignal]:
    """C. Breakout, then retest of the broken level, then continuation.
    Requires a break within the recent lookback, a pullback to the
    level, and a candle closing back in the breakout direction."""
    if len(df) < config.STRUCT_LOOKBACK + 3:
        return None
    row = df.iloc[-1]
    recent = df.iloc[-(config.RETEST_WINDOW + 1):-1]
    atr = _atr(row)
    if atr <= 0 or recent.empty:
        return None
    close = _f(row.get("close"))
    tol = atr * config.RETEST_TOUCH_ATR

    if direction == "long":
        level = _f(recent["struct_high"].min())
        if level <= 0:
            return None
        broke = (recent["close"] > recent["struct_high"]).any()
        retested = _f(row.get("low")) <= level + tol
        held = close > level and close > _f(row.get("open"))
        if not (broke and retested and held):
            return None
        return _build(symbol, "C_BREAKOUT_RETEST", direction, row, regime,
                      f"retested broken level {level:.2f} and held", 75, _f(row.get("low")))

    level = _f(recent["struct_low"].max())
    if level <= 0:
        return None
    broke = (recent["close"] < recent["struct_low"]).any()
    retested = _f(row.get("high")) >= level - tol
    held = close < level and close < _f(row.get("open"))
    if not (broke and retested and held):
        return None
    return _build(symbol, "C_BREAKOUT_RETEST", direction, row, regime,
                  f"retested broken level {level:.2f} and rejected", 75, _f(row.get("high")))


def model_d_vwap_reclaim(symbol, df, direction, regime) -> Optional[StrategySignal]:
    """D. VWAP reclaim: price was on the wrong side of VWAP, then
    crosses and closes through it. A regime-change model rather than a
    continuation model."""
    if len(df) < config.VWAP_RECLAIM_LOOKBACK + 2:
        return None
    row, prev = df.iloc[-1], df.iloc[-2]
    vwap, close = _f(row.get("vwap")), _f(row.get("close"))
    if vwap <= 0:
        return None
    window = df.iloc[-(config.VWAP_RECLAIM_LOOKBACK + 1):-1]

    if direction == "long":
        was_below = (window["close"] < window["vwap"]).sum() >= config.VWAP_RECLAIM_MIN_BARS
        crossed = _f(prev.get("close")) <= _f(prev.get("vwap")) and close > vwap
        if not (was_below and crossed):
            return None
        return _build(symbol, "D_VWAP_RECLAIM", direction, row, regime,
                      f"reclaimed VWAP {vwap:.2f} from below", 65, _f(row.get("low")))

    was_above = (window["close"] > window["vwap"]).sum() >= config.VWAP_RECLAIM_MIN_BARS
    crossed = _f(prev.get("close")) >= _f(prev.get("vwap")) and close < vwap
    if not (was_above and crossed):
        return None
    return _build(symbol, "D_VWAP_RECLAIM", direction, row, regime,
                  f"lost VWAP {vwap:.2f} from above", 65, _f(row.get("high")))


def model_e_vwap_rejection(symbol, df, direction, regime) -> Optional[StrategySignal]:
    """E. VWAP rejection: price tests VWAP from the trending side and
    is rejected, continuing away from it. The mirror of D - included so
    the two can be compared directly, since they cannot both be right
    about what VWAP means in the same regime."""
    row = df.iloc[-1]
    vwap, close = _f(row.get("vwap")), _f(row.get("close"))
    atr = _atr(row)
    if vwap <= 0 or atr <= 0:
        return None
    tol = atr * config.RETEST_TOUCH_ATR

    if direction == "long":
        touched = _f(row.get("low")) <= vwap + tol
        rejected = close > vwap and close > _f(row.get("open"))
        if not (touched and rejected):
            return None
        return _build(symbol, "E_VWAP_REJECTION", direction, row, regime,
                      f"rejected off VWAP {vwap:.2f} as support", 70, _f(row.get("low")))

    touched = _f(row.get("high")) >= vwap - tol
    rejected = close < vwap and close < _f(row.get("open"))
    if not (touched and rejected):
        return None
    return _build(symbol, "E_VWAP_REJECTION", direction, row, regime,
                  f"rejected off VWAP {vwap:.2f} as resistance", 70, _f(row.get("high")))


def model_f_ema_pullback(symbol, df, direction, regime) -> Optional[StrategySignal]:
    """F. EMA pullback: in an established EMA trend, price pulls back
    to the fast EMA and resumes. Pure trend continuation - should be
    strongly regime-dependent, which the per-regime report will show."""
    row = df.iloc[-1]
    fast = _f(row.get(f"ema_{config.EMA_FAST}"))
    slow = _f(row.get(f"ema_{config.EMA_SLOW}"))
    close, atr = _f(row.get("close")), _atr(row)
    if fast <= 0 or slow <= 0 or atr <= 0:
        return None
    tol = atr * config.RETEST_TOUCH_ATR

    if direction == "long":
        if not (fast > slow):
            return None
        pulled = _f(row.get("low")) <= fast + tol
        resumed = close > fast and close > _f(row.get("open"))
        if not (pulled and resumed):
            return None
        return _build(symbol, "F_EMA_PULLBACK", direction, row, regime,
                      f"pullback to EMA{config.EMA_FAST} held", 70, _f(row.get("low")))

    if not (fast < slow):
        return None
    pulled = _f(row.get("high")) >= fast - tol
    resumed = close < fast and close < _f(row.get("open"))
    if not (pulled and resumed):
        return None
    return _build(symbol, "F_EMA_PULLBACK", direction, row, regime,
                  f"pullback to EMA{config.EMA_FAST} rejected", 70, _f(row.get("high")))


def model_g_confluence(symbol, df, direction, regime) -> Optional[StrategySignal]:
    """G. EMA + VWAP confluence: the fast EMA and VWAP sit within a
    fraction of an ATR of each other and price reacts to the combined
    zone. The thesis is that two independent reference levels agreeing
    makes a stronger level - this model exists to test that claim, not
    to assume it."""
    row = df.iloc[-1]
    fast = _f(row.get(f"ema_{config.EMA_FAST}"))
    vwap = _f(row.get("vwap"))
    close, atr = _f(row.get("close")), _atr(row)
    if fast <= 0 or vwap <= 0 or atr <= 0:
        return None
    if abs(fast - vwap) > atr * config.CONFLUENCE_MAX_ATR:
        return None   # levels too far apart to be one zone

    zone_hi, zone_lo = max(fast, vwap), min(fast, vwap)
    tol = atr * config.RETEST_TOUCH_ATR

    if direction == "long":
        touched = _f(row.get("low")) <= zone_hi + tol
        held = close > zone_hi and close > _f(row.get("open"))
        if not (touched and held):
            return None
        return _build(symbol, "G_CONFLUENCE", direction, row, regime,
                      f"EMA/VWAP confluence {zone_lo:.2f}-{zone_hi:.2f} held", 80, _f(row.get("low")))

    touched = _f(row.get("high")) >= zone_lo - tol
    held = close < zone_lo and close < _f(row.get("open"))
    if not (touched and held):
        return None
    return _build(symbol, "G_CONFLUENCE", direction, row, regime,
                  f"EMA/VWAP confluence {zone_lo:.2f}-{zone_hi:.2f} rejected", 80, _f(row.get("high")))


def model_h_orb(symbol, df, direction, regime) -> Optional[StrategySignal]:
    """H. Opening-range breakout + VWAP confirmation. The opening range
    is only visible after it completes (see add_opening_range), so this
    cannot fire inside its own range."""
    row = df.iloc[-1]
    or_hi, or_lo = _f(row.get("or_high")), _f(row.get("or_low"))
    close, vwap = _f(row.get("close")), _f(row.get("vwap"))
    if or_hi <= 0 or or_lo <= 0 or vwap <= 0:
        return None

    if direction == "long":
        if not (close > or_hi and close > vwap):
            return None
        return _build(symbol, "H_ORB_VWAP", direction, row, regime,
                      f"broke opening range {or_hi:.2f} above VWAP", 70, or_lo)
    if not (close < or_lo and close < vwap):
        return None
    return _build(symbol, "H_ORB_VWAP", direction, row, regime,
                  f"broke opening range {or_lo:.2f} below VWAP", 70, or_hi)


# ═══════════════════════════════════════════════════════════════════
# SCORING ENGINE
# ═══════════════════════════════════════════════════════════════════

def _component_scores(row, df, direction: str) -> dict:
    """Six 0..1 component scores. Each is continuous rather than
    boolean, which is the entire point of the exercise: a hard rule
    throws away the difference between 'barely passed' and 'passed
    overwhelmingly', and that difference may carry most of the signal."""
    close = _f(row.get("close"))
    fast = _f(row.get(f"ema_{config.EMA_FAST}"))
    slow = _f(row.get(f"ema_{config.EMA_SLOW}"))
    slope = _f(row.get(f"ema_{config.EMA_FAST}_slope"))
    vwap = _f(row.get("vwap"))
    atr = _atr(row)
    rvol = _f(row.get("rvol"))
    atr_rank = _f(row.get("atr_pct_rank"), 0.5)
    sign = 1.0 if direction == "long" else -1.0

    # TREND: EMA ordering plus normalised separation.
    sep = abs(fast - slow) / close if close else 0.0
    ordered = (fast > slow) if direction == "long" else (fast < slow)
    trend = min(1.0, sep / config.SCORE_SEPARATION_FULL) if ordered else 0.0

    # MOMENTUM: EMA slope in the trade's direction.
    momentum = min(1.0, max(0.0, sign * slope / config.SCORE_SLOPE_FULL))

    # VOLUME: relative volume above 1.0, saturating at the threshold.
    thresh = _rvol_threshold(direction)
    volume = min(1.0, max(0.0, (rvol - 1.0) / max(thresh - 1.0, 1e-9))) if rvol else 0.0

    # VWAP: on the right side, scaled by how stretched the move is, with
    # a penalty for genuine overextension.
    #
    # This uses the Z-SCORE of the VWAP distance, not the raw ATR
    # distance, and the reason matters: session VWAP is anchored at the
    # open, so in a trending session price moves monotonically away from
    # it - 4+ ATRs above VWAP by mid-afternoon is ordinary, not extreme.
    # Scoring raw ATR distance therefore punished exactly the trending
    # bars a trend model wants. The z-score normalises by the distance's
    # own recent dispersion, so "stretched" means stretched relative to
    # how this symbol has been behaving today, which is the thing an
    # overextension penalty is actually trying to capture.
    z = sign * _f(row.get("vwap_z"))
    if not vwap or z <= 0:
        vwap_score = 0.0
    elif z <= config.SCORE_VWAP_IDEAL_Z:
        vwap_score = z / config.SCORE_VWAP_IDEAL_Z
    else:
        over = z - config.SCORE_VWAP_IDEAL_Z
        vwap_score = max(0.0, 1.0 - over / config.SCORE_VWAP_FADE_Z)

    # VOLATILITY: mid-range ATR percentile scores highest. Too low and
    # breakouts fail; too high and stops are unreliable.
    volatility = max(0.0, 1.0 - abs(atr_rank - config.SCORE_ATR_IDEAL) / config.SCORE_ATR_IDEAL)

    # PRICE ACTION: structure in the trade's direction plus candle close.
    if direction == "long":
        struct = 1.0 if bool(row.get("hh")) else (0.5 if bool(row.get("hl")) else 0.0)
        candle = 1.0 if close > _f(row.get("open")) else 0.0
    else:
        struct = 1.0 if bool(row.get("ll")) else (0.5 if bool(row.get("lh")) else 0.0)
        candle = 1.0 if close < _f(row.get("open")) else 0.0
    price_action = 0.6 * struct + 0.4 * candle

    return {
        "trend": trend, "momentum": momentum, "volume": volume,
        "vwap": vwap_score, "volatility": volatility, "price_action": price_action,
    }


def model_score_engine(symbol, df, direction, regime) -> Optional[StrategySignal]:
    """Weighted scoring engine: fires when the combined 0..100 score
    clears config.SCORE_MIN_TOTAL.

    This is the direct test of "does a weighted score beat hard boolean
    rules?" - it uses the same inputs the boolean models use, so any
    difference in the report is attributable to the combination method,
    not to different information.
    """
    row = df.iloc[-1]
    comps = _component_scores(row, df, direction)
    total = 100.0 * sum(comps[k] * w for k, w in config.SCORE_WEIGHTS.items())
    if total < config.SCORE_MIN_TOTAL:
        return None

    # Floor check, applied only to the components that decide whether the
    # trade is directionally valid at all (config.SCORE_FLOOR_COMPONENTS).
    # Flooring ALL six would make this AND-logic in disguise and defeat the
    # purpose of comparing a score against the boolean models.
    floored = getattr(config, "SCORE_FLOOR_COMPONENTS", tuple(comps))
    if any(comps[k] < config.SCORE_MIN_COMPONENT for k in floored if k in comps):
        return None

    struct_level = _f(row.get("low")) if direction == "long" else _f(row.get("high"))
    reason = " ".join(f"{k}={v:.2f}" for k, v in comps.items())
    return _build(symbol, "SCORE_ENGINE", direction, row, regime, reason, total, struct_level)


# ═══════════════════════════════════════════════════════════════════

ENTRY_MODELS = {
    "A_BREAKOUT": model_a_breakout,
    "B_BREAKOUT_CLOSE": model_b_breakout_close,
    "C_BREAKOUT_RETEST": model_c_breakout_retest,
    "D_VWAP_RECLAIM": model_d_vwap_reclaim,
    "E_VWAP_REJECTION": model_e_vwap_rejection,
    "F_EMA_PULLBACK": model_f_ema_pullback,
    "G_CONFLUENCE": model_g_confluence,
    "H_ORB_VWAP": model_h_orb,
    "SCORE_ENGINE": model_score_engine,
}


def evaluate_all(symbol: str, df: pd.DataFrame, models: dict = None,
                 directions=("long", "short")) -> list:
    """Run every requested model in every requested direction on the
    enriched slice `df`, applying the regime gate. Returns the signals
    that fired."""
    models = models or ENTRY_MODELS
    if len(df) < config.MIN_WARMUP_CANDLES:
        return []

    row = df.iloc[-1]
    regime = classify(row)
    out = []
    for direction in directions:
        if not is_tradeable(regime, direction):
            continue
        if direction == "short" and _session_position(df) < config.SHORT_MIN_SESSION_CANDLES:
            continue
        for name, fn in models.items():
            try:
                sig = fn(symbol, df, direction, regime)
            except Exception:
                continue
            if sig:
                out.append(sig)
    return out
