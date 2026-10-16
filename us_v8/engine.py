"""
V8 event-driven daily portfolio simulator for the setups in v8/setups.py
(market-agnostic: costs come in as an object, prices are adjusted and of any
scale - the book is kept in money, a position is money / entry price units).

One session, in order:
  1. open   - exits and 1/3 partial sales flagged at the last close fill at the open
  2. open   - entries from last close's signals, best score first, while slots,
              cash and the market filter allow:
                BO  buy-stop at the 10-day high (fill = max(open, level));
                    skipped if it opens > 1 ADR above the level or never trades it
                ATH buy at the open
  3. day    - stop orders: a gap below the stop fills at the open, else at the stop.
              Entry day of a BO buy-stop: the day's low vs the stop is ambiguous
              (was the low before or after the fill?) -> "pessimistic" = stopped,
              "optimistic" = only when it opened at/above the level.
  4. close  - EP entries at the close of the gap day (stop = gap day's low)
  5. close  - BO/EP: after `partial_day` sessions flag 1/3 for the next open and
              move the stop to breakeven; from then on the first close below the
              trailing average flags a full exit at the next open (Qullamaggie).
              ATH: stop = max(stop, highest close since entry - k x ATR42); a close
              below it flags an exit at the next open (Wilcox & Crittenden).
  6. mark to market at the close.

Position size: risk `risk_pct` of equity to the initial stop, capped at
`max_weight` of equity and by cash (no leverage).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class Result:
    daily: pd.DataFrame
    trades: pd.DataFrame
    cfg: dict = field(default_factory=dict)


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def run(px: dict, ind: dict, cfg: dict, cost, start, end, regime: pd.Series | None = None) -> Result:
    dates: pd.DatetimeIndex = px["dates"]
    ents = px["entities"]
    O, H, L, C, CF = px["O"], px["H"], px["L"], px["C"], px["CF"]
    last_valid = px["last_valid"]
    i0 = int(dates.searchsorted(pd.Timestamp(start)))
    i1 = int(dates.searchsorted(pd.Timestamp(end), side="right")) - 1
    i0 = max(i0, 1)

    setups = set(cfg["setups"])
    trail = ind[f"sma{cfg.get('trail_ma', 10)}"]
    ep_sig = ind[cfg.get("ep_signal", "ep")]
    reg = regime.reindex(dates).fillna(False).to_numpy() if (regime is not None and cfg.get("regime")) else None
    k_atr = cfg.get("ath_k", 10.0)
    pday, pfrac = cfg.get("partial_day", 3), cfg.get("partial_frac", 1 / 3)
    max_pos, max_w, risk_pct = cfg["max_pos"], cfg["max_weight"], cfg["risk_pct"]
    pessimistic = cfg.get("fill_mode", "pessimistic") == "pessimistic"

    cash = float(cfg["start_equity"])
    pos: dict[int, dict] = {}
    trades, rows = [], []
    eq_prev = cash
    costs_total = 0.0

    def sell(j, units, price_raw, kind_fill, i, reason, final):
        nonlocal cash, costs_total
        p = pos[j]
        price = price_raw * (1 - cost.slip(kind_fill))
        money = units * price
        ch = cost.sell(money)
        cash += money - ch
        costs_total += ch + units * price_raw * cost.slip(kind_fill)
        p["out"] += money - ch
        p["units"] -= units
        p["exit_px_sum"] += units * price_raw
        if final or p["units"] <= 1e-12:
            pnl = p["out"] - p["in"]
            trades.append({"kind": p["kind"], "entity": ents[j], "entry_date": dates[p["entry_i"]],
                           "exit_date": dates[i], "entry_px": p["entry_raw"],
                           "exit_px": p["exit_px_sum"] / p["units0"], "money_in": p["in"],
                           "pnl": pnl, "ret": pnl / p["in"], "R": pnl / p["risk_money"],
                           "sessions": p["days"], "reason": reason, "partial": p["partial"],
                           "init_stop_pct": p["stop_pct"]})
            del pos[j]

    def buy(kind, j, fill_raw, stop, fill_kind, i):
        nonlocal cash, costs_total
        if len(pos) >= max_pos or j in pos:
            return False
        fill = fill_raw * (1 + cost.slip(fill_kind))
        d = (fill - stop) / fill
        if not (d > 0) or _nan(fill):
            return False
        money = min(risk_pct * eq_prev / d, max_w * eq_prev, cash / (1 + cost.buy_rate))
        if money < cfg.get("min_ticket", 0.005) * eq_prev:
            return False
        ch = cost.buy(money)
        cash -= money + ch
        costs_total += ch + money * cost.slip(fill_kind) / (1 + cost.slip(fill_kind))
        pos[j] = {"kind": kind, "units": money / fill, "units0": money / fill, "entry_raw": fill_raw,
                  "entry_i": i, "stop": stop, "stop_pct": d, "days": 0, "partial": False, "flag": None,
                  "reason": None, "hi": fill_raw, "in": money + ch, "out": 0.0, "exit_px_sum": 0.0,
                  "risk_money": money * d, "fill_kind": fill_kind}
        return True

    for i in range(i0, i1 + 1):
        # 1. flagged exits / partials at the open
        for j in list(pos):
            p = pos[j]
            if i > last_valid[j]:                                  # delisted / no more data
                sell(j, p["units"], CF[i, j], "close", i, "DELISTED", True)
                continue
            if p["flag"] and not math.isnan(O[i, j]):
                if p["flag"] == "partial":
                    sell(j, p["units0"] * pfrac, O[i, j], "open", i, "PARTIAL", False)
                    if j in pos:
                        pos[j]["stop"] = max(pos[j]["stop"], pos[j]["entry_raw"])
                        pos[j]["partial"], pos[j]["flag"] = True, None
                else:
                    sell(j, p["units"], O[i, j], "open", i, p["reason"], True)

        # 2. entries at the open / buy-stops (signals from the last close)
        ok_prev = True if reg is None else bool(reg[i - 1])
        cands = []
        if ok_prev and "BO" in setups:
            for j in np.flatnonzero(ind["bo"][i - 1]):
                if j in pos or math.isnan(O[i, j]) or not H[i, j] > L[i, j]:
                    continue
                lvl, a = ind["pivot"][i - 1, j], ind["adr"][i - 1, j]
                if H[i, j] > lvl and O[i, j] <= lvl * (1 + a):
                    fr = max(O[i, j], lvl)
                    cands.append((ind["rs"][i - 1, j], "BO", j, fr, fr * (1 - min(a, cfg.get("bo_stop_cap", 0.08))),
                                  "open" if O[i, j] >= lvl else "stop"))
        if (True if (reg is None or not cfg.get("ath_regime", False)) else bool(reg[i - 1])) and "ATH" in setups:
            for j in np.flatnonzero(ind[cfg.get("ath_signal", "ath")][i - 1]):
                if j in pos or math.isnan(O[i, j]):
                    continue
                st = C[i - 1, j] - k_atr * ind["atr42"][i - 1, j]
                cands.append((ind["rs"][i - 1, j], "ATH", j, O[i, j], st, "open"))
        cands.sort(key=lambda x: -(x[0] if not math.isnan(x[0]) else -1))
        for sc, kind, j, fr, st, fk in cands:
            if len(pos) >= max_pos:
                break
            buy(kind, j, fr, st, fk, i)

        # 3. stop orders during the session (BO / EP)
        for j in list(pos):
            p = pos[j]
            if p["kind"] == "ATH" or math.isnan(O[i, j]) or p["flag"] == "exit":
                continue
            if p["entry_i"] == i:
                if L[i, j] <= p["stop"] and (pessimistic or p["fill_kind"] == "open"):
                    sell(j, p["units"], p["stop"], "stop", i, "STOP_DAY0", True)
                continue
            if O[i, j] <= p["stop"]:
                sell(j, p["units"], O[i, j], "open", i, "GAP_STOP" if not p["partial"] else "GAP_BE", True)
            elif L[i, j] <= p["stop"]:
                sell(j, p["units"], p["stop"], "stop", i, "STOP" if not p["partial"] else "BREAKEVEN", True)

        # 4. EP entries at the close of the gap day
        ok_now = True if reg is None else bool(reg[i])
        if ok_now and "EP" in setups:
            ec = []
            for j in np.flatnonzero(ep_sig[i]):
                if j in pos or math.isnan(C[i, j]):
                    continue
                if cfg.get("ep_skip_close_at_high") and C[i, j] >= H[i, j]:
                    continue                                   # NSE: closed at the high = maybe locked at the upper circuit
                st = max(L[i, j], C[i, j] * (1 - cfg.get("ep_stop_cap", 0.12)))
                ec.append((ind["vratio"][i, j], j, C[i, j], st))
            ec.sort(key=lambda x: -x[0])
            for sc, j, fr, st in ec:
                if len(pos) >= max_pos:
                    break
                buy("EP", j, fr, st, "close", i)

        # 5. close: partial / trailing flags
        for j in list(pos):
            p = pos[j]
            if math.isnan(C[i, j]):
                continue
            if p["entry_i"] < i:
                p["days"] += 1
            if p["kind"] == "ATH":
                p["hi"] = max(p["hi"], C[i, j])
                p["stop"] = max(p["stop"], p["hi"] - k_atr * ind["atr42"][i, j])
                if C[i, j] < p["stop"]:
                    p["flag"], p["reason"] = "exit", "ATR_TRAIL"
                continue
            if p["days"] >= pday and C[i, j] < trail[i, j]:
                p["flag"], p["reason"] = "exit", "TRAIL"
            elif not p["partial"] and p["days"] == pday and pfrac > 0:
                p["flag"] = "partial"

        # 6. mark to market
        inv = 0.0
        for j, p in pos.items():
            inv += p["units"] * CF[i, j]
        eq = cash + inv
        rows.append({"date": dates[i], "equity": eq, "cash": cash, "invested": inv, "n_pos": len(pos),
                     "costs": costs_total})
        eq_prev = eq

    daily = pd.DataFrame(rows).set_index("date")
    daily["ret"] = daily["equity"].pct_change().fillna(daily["equity"].iloc[0] / cfg["start_equity"] - 1)
    return Result(daily=daily, trades=pd.DataFrame(trades), cfg=cfg)


# ─── metrics ────────────────────────────────────────────────────────────

def perf(ret: pd.Series, rf: float = 0.0, bench: pd.Series | None = None) -> dict:
    r = ret.dropna()
    if len(r) < 20:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / TRADING_DAYS
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1.0
    sd = r.std()
    out = {"CAGR": cagr, "Vol": sd * math.sqrt(TRADING_DAYS),
           "Sharpe": (r.mean() - rf / TRADING_DAYS) / sd * math.sqrt(TRADING_DAYS) if sd > 0 else np.nan,
           "MaxDD": float((eq / eq.cummax() - 1).min()), "Total": eq.iloc[-1] - 1}
    out["Calmar"] = cagr / abs(out["MaxDD"]) if out["MaxDD"] < 0 else np.nan
    if bench is not None:
        b = bench.reindex(r.index).fillna(0)
        out["Bench_CAGR"] = (1 + b).prod() ** (1 / yrs) - 1
        if b.var() > 0:
            beta = np.cov(r, b)[0, 1] / b.var()
            out["Beta"] = beta
            out["Alpha"] = (r.mean() - beta * b.mean()) * TRADING_DAYS
    return out


def trade_stats(t: pd.DataFrame) -> dict:
    if t is None or t.empty:
        return {"Trades": 0}
    w, lo = t[t["pnl"] > 0], t[t["pnl"] <= 0]
    top = t.sort_values("pnl", ascending=False)
    n7 = max(1, int(round(0.07 * len(t))))
    return {"Trades": len(t), "Win%": 100 * len(w) / len(t),
            "AvgWin%": 100 * w["ret"].mean() if len(w) else 0.0, "AvgLoss%": 100 * lo["ret"].mean() if len(lo) else 0.0,
            "AvgR": t["R"].mean(), "PF": w["pnl"].sum() / -lo["pnl"].sum() if lo["pnl"].sum() < 0 else np.nan,
            "Hold": t["sessions"].median(), "Top7%share": top["pnl"].iloc[:n7].sum() / t["pnl"].sum() if t["pnl"].sum() > 0 else np.nan}
