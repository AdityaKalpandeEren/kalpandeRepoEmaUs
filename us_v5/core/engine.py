"""
Daily portfolio simulation for V4 signals.

Timeline for a signal computed after the close of day t:
  t      close: rank stocks, choose target weights (data <= t only)
  t+1    OPEN : trade at open +/- slippage, pay US charges (SEC fee, FINRA TAF) + slippage
  t+1..  marked to market at each close (adjusted prices -> splits/bonuses
         don't create fake P&L)

Market-microstructure rules:
  * can't BUY a stock locked at its upper circuit on t+1 (open == high ==
    low > prev close); can't SELL one locked at lower circuit (the sell
    retries next session)
  * a position may not exceed risk.max_adv_frac of the stock's 20-day
    average traded value (as of t)
  * T+1: with t1_sale_proceeds_same_day=false, today's sale proceeds can
    only fund buys from the next session
  * a stock that stops trading for good (delisted / merged) is closed at
    its last traded close
  * small re-weights of existing holdings (< 25% of target) are skipped
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from us_v5.core import costs as cost_mod
from us_v5.core import sizing

log = logging.getLogger(__name__)
REWEIGHT_TOL = 0.25


@dataclass
class Result:
    daily: pd.DataFrame                       # date, equity, ret, cash, invested, turnover, costs, n_pos
    trades: pd.DataFrame                      # one row per closed position
    targets: pd.DataFrame = field(default_factory=pd.DataFrame)   # signal date -> chosen names


def prepare_prices(panel: pd.DataFrame, entities: set[str] | None = None) -> pd.DataFrame:
    """Price frame for the engine, with a causal 20-day average traded value."""
    p = panel if entities is None else panel[panel["entity"].isin(entities)]
    cols = ["date", "entity", "symbol", "open", "high", "low", "close", "prevclose", "adj_open", "adj_close", "value"]
    p = p[cols + (["prevclose_adj"] if "prevclose_adj" in p else [])].sort_values(["entity", "date"]).copy()
    pc = p["prevclose_adj"] if "prevclose_adj" in p else p["prevclose"]
    p["value20"] = p.groupby("entity")["value"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    p["locked_up"] = (p["open"] == p["high"]) & (p["high"] == p["low"]) & (p["open"] > pc)
    p["locked_down"] = (p["open"] == p["high"]) & (p["high"] == p["low"]) & (p["open"] < pc)
    return p


def run(scores: pd.DataFrame, prices: pd.DataFrame, info: pd.DataFrame, ctx: pd.DataFrame,
        cfg: dict, start: pd.Timestamp, end: pd.Timestamp) -> Result:
    """scores: date, entity, score (signal dates). info: date, entity, vol_63, industry.
    ctx: market features indexed by date. Trades in [start, end]."""
    c, pcfg = cfg["costs"], cfg["portfolio"]
    days = pd.DatetimeIndex(sorted(prices.loc[(prices["date"] >= start) & (prices["date"] <= end), "date"].unique()))
    by_day = {d: g.set_index("entity") for d, g in prices[prices["date"].isin(days)].groupby("date")}
    last_day = prices.groupby("entity")["date"].max()
    last_close = prices.sort_values("date").groupby("entity")["adj_close"].last()
    sc = {d: g.set_index("entity")["score"] for d, g in scores.groupby("date")}
    inf = {d: g.set_index("entity") for d, g in info[info["date"].isin(sc.keys())].groupby("date")}

    cash = float(pcfg["capital"])
    unsettled = 0.0
    units: dict[str, float] = {}
    mark: dict[str, float] = {}
    basis: dict[str, float] = {}            # rupees invested (incl. buy costs) in the open lot
    opened: dict[str, pd.Timestamp] = {}
    pending: pd.Series | None = None        # target weights to execute at next open
    exposure = sizing.Exposure(cfg)
    rows, trades, targets = [], [], []
    signal_days = [d for d in days if d in sc]
    rebalance_set = set(signal_days[::pcfg["rebalance_days"]])
    equity = cash

    def sell(e: str, px_adj: float, d: pd.Timestamp, frac: float = 1.0) -> tuple[float, float]:
        """Sell `frac` of the position at px_adj (adjusted price). Returns
        (value at the unslipped price, total cost incl. slippage)."""
        nonlocal cash, unsettled
        u = units[e] * frac
        mid = u * px_adj
        gross = u * cost_mod.fill_price(px_adj, "sell", c)
        proceeds = gross - cost_mod.order_charges(gross, "sell", c)
        if c["t1_sale_proceeds_same_day"]:
            cash += proceeds
        else:
            unsettled += proceeds
        b = basis[e] * frac
        units[e] -= u
        basis[e] -= b
        if frac >= 0.999:
            trades.append({"entity": e, "entry": opened[e], "exit": d, "invested": b, "proceeds": proceeds,
                           "ret": proceeds / b - 1 if b > 0 else np.nan})
            for dct in (units, basis, opened, mark):
                dct.pop(e, None)
        return mid, mid - proceeds

    for d in days:
        today = by_day.get(d)
        traded_value, paid = 0.0, 0.0
        if unsettled and not c["t1_sale_proceeds_same_day"]:
            cash += unsettled
            unsettled = 0.0

        # forced exits: stock no longer trades at all
        for e in [e for e in list(units) if last_day.get(e, d) < d]:
            v, fee = sell(e, last_close[e], d)
            traded_value += v
            paid += fee

        if pending is not None and today is not None:
            eq_open = cash + unsettled + sum(units[e] * (today["adj_open"].get(e, np.nan)
                                                         if e in today.index else mark.get(e, 0))
                                             for e in units)
            target_val = pending * eq_open
            # 1) sells / reductions
            for e in list(units):
                if e not in today.index:
                    continue
                row = today.loc[e]
                tgt = float(target_val.get(e, 0.0))
                cur = units[e] * row["adj_open"]
                if tgt <= 0 or (cur - tgt) / max(tgt, 1e-9) > REWEIGHT_TOL:
                    if row["locked_down"]:
                        continue
                    frac = 1.0 if tgt <= 0 else (cur - tgt) / cur
                    v, fee = sell(e, row["adj_open"], d, frac)
                    traded_value += v
                    paid += fee
            # 2) buys / increases
            for e, tgt in target_val.sort_values(ascending=False).items():
                if e not in today.index or tgt <= 0:
                    continue
                row = today.loc[e]
                if row["locked_up"]:
                    continue
                cur = units.get(e, 0.0) * row["adj_open"]
                if cur > 0 and (tgt - cur) / tgt < REWEIGHT_TOL:
                    continue
                want = tgt - cur
                fp = cost_mod.fill_price(row["adj_open"], "buy", c)
                fee = cost_mod.order_charges(want, "buy", c)
                spend = min(want + fee, cash)
                if spend <= 100:
                    continue
                gross = spend * want / (want + fee)
                fee = spend - gross
                units[e] = units.get(e, 0.0) + gross / fp
                basis[e] = basis.get(e, 0.0) + spend
                opened.setdefault(e, d)
                cash -= spend
                traded_value += gross
                paid += fee + gross * (fp / row["adj_open"] - 1)
            pending = None

        # mark to market at the close
        if today is not None:
            for e in units:
                if e in today.index:
                    mark[e] = today.at[e, "adj_close"]
        invested = sum(units[e] * mark.get(e, 0.0) for e in units)
        prev_equity = equity
        equity = cash + unsettled + invested
        rows.append({"date": d, "equity": equity, "ret": equity / prev_equity - 1 if rows else equity / pcfg["capital"] - 1,
                     "cash": cash + unsettled, "invested": invested, "turnover": traded_value,
                     "costs": paid, "n_pos": len(units)})

        # signal after today's close -> orders for tomorrow's open
        if d in rebalance_set and today is not None:
            s = sc[d]
            names = sizing.select(s, set(units), pcfg["top_n"], pcfg["hold_buffer"])
            dinfo = inf.get(d, pd.DataFrame())
            w = sizing.weights(names, dinfo.get("vol_63", pd.Series(dtype=float)),
                               dinfo.get("industry", pd.Series(dtype=object)), cfg)
            cx = ctx.loc[d].to_dict() if d in ctx.index else None
            w = w * exposure.scale(equity, cx)
            # ADV cap (as of the signal date)
            v20 = today["value20"].reindex(w.index)
            cap = (cfg["risk"]["max_adv_frac"] * v20 / max(equity, 1)).fillna(0)
            w = np.minimum(w, cap)
            pending = w
            targets.append({"date": d, "names": list(w.index), "weights": w.round(4).to_dict(),
                            "gross": float(w.sum())})

    daily = pd.DataFrame(rows).set_index("date")
    return Result(daily=daily, trades=pd.DataFrame(trades), targets=pd.DataFrame(targets))
