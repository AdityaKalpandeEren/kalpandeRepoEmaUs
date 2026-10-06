"""
US V5 features and targets (port of NSE swing/features/* + v5/features.py).

Every feature on row t uses data up to the close of t only (rolling / ewm /
shift(+k)); us_v5/tests checks that a feature row is identical whether or
not later data exists. Signals are computed after the close of t and traded
at the OPEN of t+1, so targets run from the t+1 open.

Differences from NSE V5:
  * no delivery % / F&O participant data in the US -> those features are
    dropped; volume shocks, up-volume share and 52-week-high timing stay
  * market = SPY (total return); context = QQQ/IWM/RSP vs SPY, VIX level,
    percentile and term structure (VIX/VIX3M), 10y yield, dollar, crude,
    credit (HYG vs IEF), universe breadth and dispersion
  * industry = GICS sector (11 groups; ~100 names leave too few peers per
    sub-industry)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

HORIZONS = (5, 10, 20)
META = ["date", "entity", "symbol", "industry", "label_end", "fwd_ret", "fwd_excess", "fwd_rank",
        "relevance", "in_universe", "target", "fwd_5", "fwd_10", "fwd_20"]
RANKED_PREFIXES = ("ret_", "mom_", "vol_", "downvol", "skew", "max_ret", "rsi", "macd", "atr", "bb_",
                   "dist_", "value_", "log_value", "beta", "resid_", "rel_", "idio", "sect_rel",
                   "up_value", "close_to", "days_since")


# ─── per-stock technicals (NSE swing/features/technical.py) ──────────────

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _days_since_max(c: pd.Series, w: int = 252, min_periods: int = 200) -> pd.Series:
    """Sessions since the rolling w-day max close (vectorised rolling argmax)."""
    a = c.to_numpy(dtype="float64")
    n = len(a)
    out = np.full(n, np.nan)
    if n == 0:
        return pd.Series(out, index=c.index)
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.concatenate([np.full(w - 1, -np.inf), a])
    win = sliding_window_view(pad, w)
    arg = np.argmax(np.where(np.isnan(win), -np.inf, win), axis=1)
    out = (w - 1 - arg).astype(float)
    valid = np.arange(n) + 1 >= min_periods
    out[~valid] = np.nan
    return pd.Series(out, index=c.index)


def per_entity(g: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    c, o, h, lo = g["adj_close"], g["adj_open"], g["adj_high"], g["adj_low"]
    r = g["ret_cc"]
    f = pd.DataFrame(index=g.index)
    for w in windows:
        f[f"ret_{w}"] = c / c.shift(w) - 1
    f["ret_1"] = r
    f["mom_12_1"] = c.shift(21) / c.shift(252) - 1
    f["mom_6_1"] = c.shift(21) / c.shift(126) - 1
    for w in (21, 63):
        f[f"vol_{w}"] = r.rolling(w, min_periods=w // 2).std()
    f["downvol_63"] = r.where(r < 0, 0).rolling(63, min_periods=30).std()
    f["vol_ratio_21_63"] = f["vol_21"] / f["vol_63"]
    f["skew_63"] = r.rolling(63, min_periods=40).skew()
    f["max_ret_21"] = r.rolling(21, min_periods=10).max()
    f["rsi_14"] = _rsi(c, 14)
    f["rsi_2"] = _rsi(c, 2)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = (ema12 - ema26) / c
    f["macd"] = macd
    f["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / c
    ma20, sd20 = c.rolling(20, min_periods=20).mean(), c.rolling(20, min_periods=20).std()
    f["bb_pos_20"] = (c - ma20) / (2 * sd20)
    for w in (50, 200):
        f[f"dist_ma{w}"] = c / c.rolling(w, min_periods=int(w * 0.8)).mean() - 1
    f["dist_52w_high"] = c / h.rolling(252, min_periods=200).max() - 1
    f["dist_52w_low"] = c / lo.rolling(252, min_periods=200).min() - 1
    f["gap_1"] = o / c.shift() - 1
    f["range_1"] = (h - lo) / c
    f["close_loc_1"] = (c - lo) / (h - lo).replace(0, np.nan)
    val = g["value"]
    lv = np.log1p(val)
    f["value_z_63"] = (lv - lv.rolling(63, min_periods=40).mean()) / lv.rolling(63, min_periods=40).std()
    f["value_ratio_5_63"] = val.rolling(5).mean() / val.rolling(63, min_periods=40).mean()
    f["log_value_63"] = lv.rolling(63, min_periods=40).mean()
    f["ret_ma_slope_20"] = ma20 / ma20.shift(5) - 1
    # V5 extras (no delivery data in the US)
    vz = (lv - lv.rolling(60, min_periods=40).mean()) / lv.rolling(60, min_periods=40).std()
    shock = (vz > 2).astype(float)
    f["vol_shock_up_20"] = (shock * (r > 0)).rolling(20, min_periods=10).sum()
    f["vol_shock_down_20"] = (shock * (r < 0)).rolling(20, min_periods=10).sum()
    f["up_value_share_20"] = val.where(r > 0, 0.0).rolling(20, min_periods=10).sum() / val.rolling(20, min_periods=10).sum()
    f["close_to_252_high"] = c / c.rolling(252, min_periods=200).max()
    f["days_since_252_high"] = _days_since_max(c)
    return f


def market_relative(panel: pd.DataFrame, mkt_ret: pd.Series) -> pd.DataFrame:
    """Beta to SPY (126d) and beta-adjusted residual returns."""
    out = pd.DataFrame(index=panel.index)
    m = panel["date"].map(mkt_ret)
    for _, idx in panel.groupby("entity", sort=False).groups.items():
        r = panel.loc[idx, "ret_cc"]
        mr = m.loc[idx]
        beta = r.rolling(126, min_periods=80).cov(mr) / mr.rolling(126, min_periods=80).var()
        out.loc[idx, "beta_126"] = beta
        lr, lmr = np.log1p(r), np.log1p(mr)
        for w in (5, 21, 63):
            sr, smr = np.expm1(lr.rolling(w).sum()), np.expm1(lmr.rolling(w).sum())
            out.loc[idx, f"resid_ret_{w}"] = sr - beta * smr
            out.loc[idx, f"rel_ret_{w}"] = sr - smr
        out.loc[idx, "idio_vol_63"] = (r - beta * mr).rolling(63, min_periods=40).std()
    return out


# ─── market / context (one row per date) ─────────────────────────────────

def market_features(dates: pd.DatetimeIndex, ctx: pd.DataFrame, u: pd.DataFrame) -> pd.DataFrame:
    """ctx: adjusted closes of context symbols (columns), indexed by date.
    u: universe rows (date, ret_cc, dist_ma50, dist_ma200, ret_21)."""
    f = pd.DataFrame(index=pd.DatetimeIndex(dates, name="date"))
    c = ctx.reindex(f.index).ffill()
    spy = c["SPY"]
    f["spy_ret_1"] = spy.pct_change(fill_method=None)
    for w in (5, 21, 63):
        f[f"spy_ret_{w}"] = spy / spy.shift(w) - 1
    f["spy_dist_ma200"] = spy / spy.rolling(200, min_periods=150).mean() - 1
    f["spy_vol_21"] = f["spy_ret_1"].rolling(21, min_periods=15).std()
    for s, col in (("QQQ", "qqq_vs_spy_21"), ("IWM", "iwm_vs_spy_21"), ("RSP", "rsp_vs_spy_21")):
        if s in c:
            f[col] = (c[s] / c[s].shift(21)) / (spy / spy.shift(21)) - 1
    if "^VIX" in c:
        vix = c["^VIX"]
        f["vix"] = vix
        f["vix_pct_1y"] = vix.rolling(252, min_periods=150).rank(pct=True)
        f["vix_chg_5"] = vix / vix.shift(5) - 1
        if "^VIX3M" in c:
            f["vix_term"] = vix / c["^VIX3M"]
    if "^TNX" in c:
        f["tnx_chg_5"] = c["^TNX"] - c["^TNX"].shift(5)
        f["tnx_chg_21"] = c["^TNX"] - c["^TNX"].shift(21)
    for s, name in (("DX-Y.NYB", "dxy"), ("CL=F", "oil")):
        if s in c:
            for w in (5, 21):
                f[f"{name}_ret_{w}"] = c[s] / c[s].shift(w) - 1
    if "HYG" in c and "IEF" in c:
        f["credit_21"] = (c["HYG"] / c["HYG"].shift(21)) / (c["IEF"] / c["IEF"].shift(21)) - 1
    f["breadth_ma50"] = (u["dist_ma50"] > 0).groupby(u["date"]).mean().reindex(f.index)
    f["breadth_ma200"] = (u["dist_ma200"] > 0).groupby(u["date"]).mean().reindex(f.index)
    adv = (u["ret_cc"] > 0).groupby(u["date"]).mean().reindex(f.index)
    f["adv_ratio_5"] = adv.rolling(5, min_periods=3).mean()
    f["xs_dispersion_21"] = u.groupby("date")["ret_21"].std().reindex(f.index)
    return f


# ─── targets (NSE swing/features/labels.py + v5 residual target) ─────────

def forward_returns(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """fwd = adj_open(t+1+H) / adj_open(t+1) - 1; label_end = date of t+1+H."""
    g = panel.groupby("entity", sort=False)
    o_in, o_out = g["adj_open"].shift(-1), g["adj_open"].shift(-(1 + horizon))
    return pd.DataFrame({"fwd_ret": o_out / o_in - 1, "label_end": g["date"].shift(-(1 + horizon))},
                        index=panel.index)


def residual_targets(df: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    """Blended residual-return rank target: for H in 5/10/20 remove the part
    explained by beta (cross-sectional OLS per day) and the sector mean, rank
    per day; target = mean of the three ranks; label_end = the 20-day end."""
    key = cand[["date", "entity"]]
    out = df[["date", "entity", "beta_126", "industry"]].copy()
    ranks = []
    for h in HORIZONS:
        fr = forward_returns(cand, h)
        m = pd.concat([key, fr], axis=1).rename(columns={"fwd_ret": f"fwd_{h}", "label_end": f"end_{h}"})
        out = out.merge(m, on=["date", "entity"], how="left")

        def _resid(g, h=h):
            gy, gb = g[f"fwd_{h}"], g["beta_126"].fillna(1.0)
            ok = gy.notna()
            res = pd.Series(np.nan, index=g.index)
            if ok.sum() < 10:
                return res
            b = np.polyfit(gb[ok], gy[ok], 1) if gb[ok].std() > 0 else (0.0, gy[ok].mean())
            r = gy - (b[0] * gb + b[1])
            return r - r.groupby(g["industry"]).transform("mean")
        out[f"resid_{h}"] = out.groupby("date", group_keys=False).apply(_resid, include_groups=False)
        ranks.append(out.groupby("date")[f"resid_{h}"].rank(pct=True))
    out["target"] = pd.concat(ranks, axis=1).mean(axis=1, skipna=False)
    out["label_end"] = out[f"end_{max(HORIZONS)}"]
    return out[["date", "entity", "target", "label_end"] + [f"fwd_{h}" for h in HORIZONS]]


def factor_score(df: pd.DataFrame) -> pd.Series:
    """52-week-high proximity (George-Hwang 2004, robust in the US) + 12-1
    momentum (Jegadeesh-Titman), rank-averaged per day."""
    r1 = df.groupby("date")["close_to_252_high"].rank(pct=True)
    r2 = df.groupby("date")["mom_12_1"].rank(pct=True)
    return (r1 + r2) / 2


# ─── dataset ─────────────────────────────────────────────────────────────

def build(data: dict, cfg: dict, with_targets: bool = True) -> pd.DataFrame:
    """One row per (universe member, date): features, ranks, context, targets."""
    panel, ctx = data["panel"], data["context"]
    cand = panel[panel["entity"].isin(panel.loc[panel["in_universe"], "entity"].unique())]
    cand = cand.sort_values(["entity", "date"]).reset_index(drop=True)
    log.info("features: %d candidate stocks, %d rows", cand["entity"].nunique(), len(cand))
    tech = pd.concat([per_entity(g, cfg["features"]["windows"]) for _, g in cand.groupby("entity", sort=False)])
    tech = tech.loc[cand.index]
    rel = market_relative(cand, ctx["SPY"].pct_change(fill_method=None))
    df = pd.concat([cand[["date", "entity", "symbol", "in_universe", "ret_cc"]], tech, rel], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    df["industry"] = df["entity"].map(data["industry"]).fillna("UNKNOWN")
    if with_targets:
        fr = forward_returns(cand, cfg["label"]["horizon"])
        df["fwd_ret_h"], df["label_end"] = fr["fwd_ret"], fr["label_end"]

    u = df[df["in_universe"]].copy()
    for w in (5, 21, 63):
        u[f"sect_rel_{w}"] = u[f"ret_{w}"] - u.groupby(["date", "industry"])[f"ret_{w}"].transform("median")
    u["industry_size"] = u.groupby(["date", "industry"])["entity"].transform("size")
    feat_cols = [c for c in u.columns if c not in META and c not in ("ret_cc", "fwd_ret_h")]
    if cfg["features"].get("cross_sectional", True):
        ranked = [c for c in feat_cols if c.startswith(RANKED_PREFIXES)]
        ranks = u.groupby("date")[ranked].rank(pct=True)
        ranks.columns = [f"xs_{c}" for c in ranked]
        u = pd.concat([u, ranks], axis=1)
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    u = u.merge(market_features(dates, ctx, u), left_on="date", right_index=True, how="left")

    if with_targets:
        tg = residual_targets(u, cand)
        u = u.drop(columns=["label_end"]).merge(tg, on=["date", "entity"], how="left")
        # the GBM / walk-forward code reads these names: learn the blended residual rank
        u["fwd_rank"] = u["target"]
        u["fwd_ret"] = u["target"]
        u = u.drop(columns=["fwd_ret_h"])
    drop = set(cfg["features"].get("drop") or [])
    u = u.drop(columns=[c for c in drop if c in u.columns])
    u = u.replace([np.inf, -np.inf], np.nan)
    num = [c for c in u.columns if c not in META and pd.api.types.is_float_dtype(u[c])]
    u[num] = u[num].astype("float32")                     # half the memory; trees don't care
    return u.sort_values(["date", "entity"]).reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    skip = set(META) | {"ret_cc"}
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
