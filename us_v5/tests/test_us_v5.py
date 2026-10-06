"""
US V5 checks (python -m pytest us_v5/tests -q). Needs the cached Yahoo bars
(us_v5/cache/daily_bars.parquet - built by `python -m us_v5.research`);
skipped when the cache is missing.

  * no lookahead: features for dates <= T are identical whether or not data
    after T exists (the live job scores the last day of a truncated panel)
  * walk-forward purge: no training label reaches into its test block
  * costs: US round trip ~= 2 x slippage + SEC fee
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from us_v5 import data as data_mod
from us_v5 import features as feat
from us_v5.core import config as cfgmod
from us_v5.core import costs, walkforward

CACHE = os.path.join(data_mod.ROOT, "us_v5", "cache", data_mod.BARS_FILE)
needs_cache = pytest.mark.skipif(not os.path.exists(CACHE), reason="no cached Yahoo bars")


def _data(cfg, end):
    bars = pd.read_parquet(CACHE)
    bars = bars[(bars["date"] >= pd.Timestamp(cfg["data"]["start"])) & (bars["date"] <= pd.Timestamp(end))]
    members = data_mod.load_sp500()
    ctx_syms = cfg["data"]["context"]
    ctx = bars[bars["symbol"].isin(ctx_syms)].pivot(index="date", columns="symbol", values="adj_close").sort_index()
    panel = data_mod.assemble(bars[bars["symbol"].isin(set(members["symbol"]))], members, cfg)
    return {"panel": panel, "industry": members.set_index("symbol")["sector"], "context": ctx}


@needs_cache
def test_no_lookahead():
    cfg = cfgmod.load({"data.start": "2016-01-01"})
    cut = "2019-06-28"
    short = feat.build(_data(cfg, cut), cfg, with_targets=False)
    full = feat.build(_data(cfg, "2020-06-30"), cfg, with_targets=False)
    cols = [c for c in feat.feature_columns(short) if c in full.columns]
    a = short[short["date"] <= cut].set_index(["date", "entity"])[cols].sort_index()
    b = full[full["date"] <= cut].set_index(["date", "entity"])[cols].sort_index()
    assert a.index.equals(b.index), "universe membership changed when later data was added"
    diff = (a - b).abs().max()
    bad = diff[diff > 1e-4]
    assert bad.empty, f"features changed with later data: {bad.to_dict()}"


@needs_cache
def test_target_starts_next_open():
    cfg = cfgmod.load({"data.start": "2018-01-01"})
    d = _data(cfg, "2019-12-31")
    p = d["panel"]
    g = p[p["entity"] == "AAPL"].reset_index(drop=True)
    fr = feat.forward_returns(g, 5)
    i = 100
    assert np.isclose(fr["fwd_ret"].iloc[i], g["adj_open"].iloc[i + 6] / g["adj_open"].iloc[i + 1] - 1)
    assert fr["label_end"].iloc[i] == g["date"].iloc[i + 6]


def test_purge():
    dates = pd.bdate_range("2015-01-01", "2020-12-31")
    df = pd.DataFrame({"date": np.repeat(dates, 3), "entity": np.tile(["A", "B", "C"], len(dates))})
    pos = {d: i for i, d in enumerate(dates)}
    df["label_end"] = [dates[min(pos[d] + 21, len(dates) - 1)] for d in df["date"]]
    df["fwd_ret"] = 0.0
    cfg = {"validation": {"first_test": "2017-01-01", "holdout_start": "2020-01-01", "refit_months": 12,
                          "embargo_days": 21}}
    for fold in walkforward.folds(pd.DatetimeIndex(dates), cfg):
        tr, te = walkforward.split(df, fold)
        walkforward.check_no_overlap(tr, te)
        assert tr["label_end"].max() < te["date"].min()


def test_costs():
    c = cfgmod.load()["costs"]
    rt = costs.round_trip_pct(10_000, c)
    assert 2 * c["slippage_bps"] / 1e4 < rt < 2 * c["slippage_bps"] / 1e4 + 0.0002
