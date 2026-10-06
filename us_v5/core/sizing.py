"""
Portfolio construction and risk controls.

  select()        top-N by score, with a hold buffer: a current holding is
                  kept while its rank <= top_n * hold_buffer (cuts turnover)
  weights()       inverse-volatility (or equal) weights, capped per stock
                  and per sector (excess redistributed pro-rata)
  exposure()      drawdown circuit breaker + optional regime filter scale
                  the invested fraction; the rest stays in cash
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def select(scores: pd.Series, held: set[str], top_n: int, buffer: float) -> list[str]:
    """scores: entity -> score for one day (higher = better)."""
    ranked = scores.dropna().sort_values(ascending=False)
    rank = pd.Series(np.arange(1, len(ranked) + 1), index=ranked.index)
    keep = [e for e in held if e in rank.index and rank[e] <= top_n * buffer]
    keep = sorted(keep, key=lambda e: rank[e])[:top_n]
    for e in ranked.index:
        if len(keep) >= top_n:
            break
        if e not in keep:
            keep.append(e)
    return keep


def _cap(w: pd.Series, groups: pd.Series, name_cap: float, group_cap: float, iters: int = 100) -> pd.Series:
    """Enforce a per-name cap and a per-group cap together. Excess weight is
    redistributed pro-rata to names that still have room under BOTH caps;
    when nobody has room, it stays in cash (weights sum to < 1)."""
    w = w.astype(float).copy()
    for _ in range(iters):
        excess = 0.0
        over = w > name_cap
        excess += (w[over] - name_cap).sum()
        w[over] = name_cap
        gs = w.groupby(groups).sum()
        for g, tot in gs[gs > group_cap].items():
            m = groups == g
            excess += tot - group_cap
            w[m] *= group_cap / tot
        if excess <= 1e-12:
            break
        gs = w.groupby(groups).sum()
        room = (w < name_cap - 1e-12) & groups.map(gs < group_cap - 1e-12).astype(bool)
        if not room.any() or w[room].sum() <= 0:
            break
        w[room] += excess * w[room] / w[room].sum()
    return w


def weights(names: list[str], vol: pd.Series, sector: pd.Series, cfg: dict) -> pd.Series:
    r = cfg["risk"]
    if not names:
        return pd.Series(dtype=float)
    if r["sizing"] == "inverse_vol":
        v = vol.reindex(names).astype(float)
        v = v.fillna(v.median() if v.notna().any() else 0.02).clip(lower=0.005)
        w = 1 / v
    else:
        w = pd.Series(1.0, index=names)
    w = w / w.sum()
    return _cap(w, sector.reindex(names).fillna("UNKNOWN"), r["max_weight"], r["max_sector_weight"])


class Exposure:
    """Drawdown circuit breaker (+ optional regime filter).

    Drawdown is measured from the ROLLING 1-year peak of the equity seen at
    rebalances: an all-time peak can lock a half-invested portfolio out of
    ever recovering (it needs twice the move to climb back)."""

    def __init__(self, cfg: dict):
        self.r = cfg["risk"]
        n = max(int(252 / max(cfg["portfolio"]["rebalance_days"], 1)), 1)
        self.history: list[float] = []
        self.window = n
        self.tripped = False

    def scale(self, equity: float, ctx: dict | None = None) -> float:
        self.history = (self.history + [equity])[-self.window:]
        dd = equity / max(self.history) - 1
        if not self.tripped and dd <= -self.r["dd_breaker"]:
            self.tripped = True
        elif self.tripped and dd >= -self.r["dd_resume"]:
            self.tripped = False
        s = 0.5 if self.tripped else 1.0
        if self.r.get("regime_filter") and ctx:
            if (ctx.get("vix_pct_1y") or 0) > 0.8 and (ctx.get("spy_dist_ma200") or 0) < 0:
                s *= 0.5
        return s
