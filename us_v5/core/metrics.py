"""Performance statistics, incl. the Deflated Sharpe Ratio."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


def perf(ret: pd.Series, bench: pd.Series | None = None, rf: float = 0.0) -> dict:
    """ret: daily net returns. rf: annual risk-free rate."""
    ret = ret.dropna()
    if len(ret) < 2:
        return {}
    eq = (1 + ret).cumprod()
    years = len(ret) / TRADING_DAYS
    cagr = eq.iloc[-1] ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1.0
    ex = ret - rf / TRADING_DAYS
    vol = ret.std() * math.sqrt(TRADING_DAYS)
    sharpe = ex.mean() / ret.std() * math.sqrt(TRADING_DAYS) if ret.std() > 0 else np.nan
    sharpe0 = ret.mean() / ret.std() * math.sqrt(TRADING_DAYS) if ret.std() > 0 else np.nan
    down = ex[ex < 0]
    sortino = ex.mean() / math.sqrt((down ** 2).mean()) * math.sqrt(TRADING_DAYS) if len(down) else np.nan
    dd = eq / eq.cummax() - 1
    mdd = dd.min()
    out = {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "Sharpe_rf0": sharpe0, "Sortino": sortino,
           "MaxDD": mdd, "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
           "HitRate_daily": (ret > 0).mean(), "Days": len(ret), "TotalReturn": eq.iloc[-1] - 1}
    if bench is not None:
        b = bench.reindex(ret.index).fillna(0)
        if b.var() > 0:
            beta = np.cov(ret, b)[0, 1] / b.var()
            alpha = (ret.mean() - beta * b.mean()) * TRADING_DAYS
            out.update({"Beta": beta, "Alpha_ann": alpha,
                        "Bench_CAGR": (1 + b).prod() ** (1 / years) - 1,
                        "Bench_MaxDD": ((1 + b).cumprod() / (1 + b).cumprod().cummax() - 1).min(),
                        "Bench_Sharpe": (b.mean() - rf / TRADING_DAYS) / b.std() * math.sqrt(TRADING_DAYS)})
    return out


def probabilistic_sharpe(ret: pd.Series, sr_benchmark: float = 0.0) -> float:
    """PSR: probability the true (per-period) Sharpe exceeds sr_benchmark."""
    r = ret.dropna()
    n = len(r)
    if n < 10 or r.std() == 0:
        return np.nan
    sr = r.mean() / r.std()
    g3, g4 = stats.skew(r), stats.kurtosis(r, fisher=False)
    denom = math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return float(stats.norm.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / denom))


def deflated_sharpe(ret: pd.Series, trial_sharpes: list[float]) -> dict:
    """Bailey & Lopez de Prado (2014). trial_sharpes: PER-PERIOD (daily) Sharpe
    of every configuration tried. Benchmark = expected max Sharpe of N
    unskilled trials; DSR = PSR against that benchmark."""
    n = max(len(trial_sharpes), 1)
    var = float(np.var(trial_sharpes, ddof=1)) if n > 1 else 0.0
    gamma = 0.5772156649
    if n > 1:
        sr0 = math.sqrt(var) * ((1 - gamma) * stats.norm.ppf(1 - 1 / n)
                                + gamma * stats.norm.ppf(1 - 1 / (n * math.e)))
    else:
        sr0 = 0.0
    return {"DSR": probabilistic_sharpe(ret, sr0), "SR0_daily": sr0, "SR0_annual": sr0 * math.sqrt(TRADING_DAYS),
            "N_trials": n, "PSR_vs_0": probabilistic_sharpe(ret, 0.0)}


def daily_sharpe(ret: pd.Series) -> float:
    r = ret.dropna()
    return float(r.mean() / r.std()) if r.std() > 0 else 0.0


REGIMES = [  # (name, start, end) - well-known US market phases
    ("2017 low-vol bull", "2017-01-01", "2018-01-26"),
    ("2018 Volmageddon + Q4 selloff", "2018-01-29", "2018-12-24"),
    ("2019 recovery", "2018-12-26", "2020-02-19"),
    ("2020 COVID crash", "2020-02-20", "2020-03-23"),
    ("2020-21 stimulus bull", "2020-03-24", "2021-12-31"),
    ("2022 rate-hike bear", "2022-01-03", "2022-10-12"),
    ("2022-23 recovery / AI rally", "2022-10-13", "2024-12-31"),
    ("2025 tariff shock", "2025-02-19", "2025-04-08"),
    ("2025-26", "2025-04-09", "2026-12-31"),
]


def by_period(ret: pd.Series, bench: pd.Series, rf: float) -> pd.DataFrame:
    rows = []
    for y, r in ret.groupby(ret.index.year):
        p = perf(r, bench, rf)
        rows.append({"period": str(y), **{k: p.get(k) for k in ("CAGR", "Sharpe", "MaxDD", "Bench_CAGR", "Alpha_ann")},
                     "Return": p.get("TotalReturn"),
                     "Bench_Return": float((1 + bench.reindex(r.index).fillna(0)).prod() - 1)})
    return pd.DataFrame(rows)


def by_regime(ret: pd.Series, bench: pd.Series, rf: float, ctx: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for name, a, b in REGIMES:
        r = ret[(ret.index >= a) & (ret.index <= b)]
        if len(r) < 5:
            continue
        bb = bench.reindex(r.index).fillna(0)
        rows.append({"regime": name, "days": len(r), "Return": float((1 + r).prod() - 1),
                     "Bench_Return": float((1 + bb).prod() - 1), "Sharpe": perf(r, bench, rf).get("Sharpe"),
                     "MaxDD": perf(r).get("MaxDD")})
    if ctx is not None:   # data-driven regimes from the market features
        c = ctx.reindex(ret.index)
        for label, mask in (("VIX top 20% (1y)", c["vix_pct_1y"] > 0.8),
                            ("VIX bottom 50% (1y)", c["vix_pct_1y"] < 0.5),
                            ("SPY above 200DMA", c["spy_dist_ma200"] > 0),
                            ("SPY below 200DMA", c["spy_dist_ma200"] <= 0)):
            r = ret[mask.fillna(False)]
            if len(r) >= 5:
                bb = bench.reindex(r.index).fillna(0)
                rows.append({"regime": label, "days": len(r), "Return": float((1 + r).prod() - 1),
                             "Bench_Return": float((1 + bb).prod() - 1),
                             "Sharpe": perf(r, bench, rf).get("Sharpe"), "MaxDD": np.nan})
    return pd.DataFrame(rows)


def monthly_table(ret: pd.Series) -> pd.DataFrame:
    m = (1 + ret).groupby([ret.index.year, ret.index.month]).prod() - 1
    t = m.unstack()
    t.index.name, t.columns.name = "year", "month"
    return t
