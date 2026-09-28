"""
Regime classifier.

The point of this module is NOT to improve any single entry signal. It
is to answer a different and more valuable question: "does this setup
only work in some conditions?" Every trade the research engine
simulates is tagged with the regime it was opened in, so the report can
slice win rate by regime. A model with a 45% overall win rate that is
62% in trending conditions and 28% in chop is not a bad model - it is a
good model with a missing filter, and you cannot see that without this
tag.

Two independent axes, deliberately kept separate rather than mashed
into one label:

  DIRECTION: strong_bull / weak_bull / neutral / weak_bear / strong_bear
     from EMA structure (fast vs slow), EMA slope, and where price sits
     relative to VWAP.

  VOLATILITY: low / normal / high
     from the ATR percentile rank - i.e. this symbol's current ATR
     against its own recent history, not an absolute number, so one
     threshold works across a whole watchlist.

A "regime" is the pair, e.g. strong_bull/normal.

Honest limitation: this classifier is computed from the SAME candles
the signal fires on, so it is a concurrent label, not a forecast. It
tells you the conditions a trade happened in. It does not predict the
conditions the trade will resolve in, and a regime can flip mid-trade.
"""
from dataclasses import dataclass

import pandas as pd

import config


@dataclass
class Regime:
    direction: str      # strong_bull | weak_bull | neutral | weak_bear | strong_bear
    volatility: str     # low | normal | high
    trend_strength: float   # 0..1, continuous version of `direction`

    @property
    def label(self) -> str:
        return f"{self.direction}/{self.volatility}"

    @property
    def is_trending(self) -> bool:
        return self.direction in ("strong_bull", "strong_bear")

    @property
    def is_bullish(self) -> bool:
        return self.direction in ("strong_bull", "weak_bull")

    @property
    def is_bearish(self) -> bool:
        return self.direction in ("strong_bear", "weak_bear")


def _safe(v, default=0.0) -> float:
    try:
        if v is None or pd.isna(v):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def classify(row) -> Regime:
    """Classify the regime at one (already enriched) candle."""
    fast_col = f"ema_{config.EMA_FAST}"
    slow_col = f"ema_{config.EMA_SLOW}"

    close = _safe(row.get("close"))
    ema_fast = _safe(row.get(fast_col))
    ema_slow = _safe(row.get(slow_col))
    slope = _safe(row.get(f"ema_{config.EMA_FAST}_slope"))
    vwap = _safe(row.get("vwap"))
    atr_rank = _safe(row.get("atr_pct_rank"), 0.5)

    # Volatility axis, from the ATR's own percentile rank.
    if atr_rank >= config.REGIME_ATR_HIGH_PCT:
        volatility = "high"
    elif atr_rank <= config.REGIME_ATR_LOW_PCT:
        volatility = "low"
    else:
        volatility = "normal"

    # Direction axis: three independent votes, so one noisy input can't
    # by itself declare a strong trend.
    votes = 0
    if ema_fast and ema_slow:
        votes += 1 if ema_fast > ema_slow else -1
    if slope:
        votes += 1 if slope > config.REGIME_SLOPE_MIN else (-1 if slope < -config.REGIME_SLOPE_MIN else 0)
    if vwap and close:
        votes += 1 if close > vwap else -1

    # Separation of the two EMAs, normalised by price: a trend with the
    # EMAs on top of each other is not a trend.
    separation = abs(ema_fast - ema_slow) / close if close else 0.0
    wide = separation >= config.REGIME_EMA_SEPARATION_MIN

    if votes >= 3 and wide:
        direction = "strong_bull"
    elif votes >= 2:
        direction = "weak_bull"
    elif votes <= -3 and wide:
        direction = "strong_bear"
    elif votes <= -2:
        direction = "weak_bear"
    else:
        direction = "neutral"

    trend_strength = min(1.0, abs(votes) / 3.0 * (1.5 if wide else 0.75))
    return Regime(direction=direction, volatility=volatility, trend_strength=round(trend_strength, 2))


def is_tradeable(regime: Regime, direction: str) -> bool:
    """Should a trade in `direction` ('long'/'short') be allowed in this
    regime at all? This is the 'NO TRADE' gate.

    Blocked cases and why:
      - neutral direction: chop. Both breakout and pullback models bleed
        here; this is the single most valuable filter in the system.
      - extreme volatility: stop distances become unreliable and gap
        risk dominates.
      - counter-trend: no longs in a strong_bear, no shorts in a
        strong_bull. Fading a strong trend intraday is a different
        strategy with a different risk profile.

    Turn the gate off entirely with config.REGIME_FILTER_ENABLED=false -
    which you should do at least once, to measure what the filter is
    actually worth rather than assuming it helps.
    """
    if not config.REGIME_FILTER_ENABLED:
        return True
    if regime.direction == "neutral":
        return False
    if regime.volatility == "high" and config.REGIME_BLOCK_HIGH_VOL:
        return False
    if direction == "long" and regime.direction == "strong_bear":
        return False
    if direction == "short" and regime.direction == "strong_bull":
        return False
    return True
