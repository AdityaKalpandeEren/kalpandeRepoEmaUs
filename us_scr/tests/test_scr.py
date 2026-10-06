"""python -m pytest us_scr/tests -q  - synthetic bars, no network."""
import numpy as np
import pandas as pd

from us_scr import strategy as S


def _day(seed=0, spike_at=68):
    """5-min bars 04:00-15:55 ET: quiet, a volume surge + run-up from 09:40,
    a 3-bar pullback on lighter volume, a reclaim, then further waves."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-10-05 04:00", "2026-10-05 15:55", freq="5min", tz="America/New_York")
    n = len(idx)
    steps = rng.normal(0, 0.002, n)
    vol = rng.integers(2_000, 5_000, n).astype(float)
    k = spike_at
    while k + 9 < n:                                              # waves: 6 up bars, 3 pullback bars
        steps[k:k + 6] += 0.03
        steps[k + 6:k + 9] -= 0.012
        vol[k:k + 6] *= 40
        vol[k + 6:k + 9] *= 8
        k += 9
    px = 2.0 * np.cumprod(1 + steps)
    o = np.r_[px[0], px[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, px) * 1.002, "low": np.minimum(o, px) * 0.998,
                         "close": px, "volume": vol}, index=idx)


def test_setups_are_causal():
    x = _day()
    full = S.setups(x, S.bar_features(x, 1.8, 50_000))
    assert len(full) > 0
    for cut in (90, 110, 130):
        part = x.iloc[:cut]
        got = S.setups(part, S.bar_features(part, 1.8, 50_000))
        assert list(got) == [t for t in full if t <= part.index[-1]], "a setup changed when later bars were added"


def test_live_matches_research():
    x = _day(seed=3)
    sig = S.setups(x, S.bar_features(x, 1.8, 50_000))[0]
    done = S.simulate("T", x, sig, final=True)
    i = x.index.get_loc(sig)
    for cut in range(i + 2, len(x) + 1):
        tr = S.simulate("T", x.iloc[:cut], sig, final=False)
        if tr.outcome != "OPEN":                                    # the first run that sees the exit
            assert (tr.exit_ts, round(tr.exit, 6), tr.outcome) == (done.exit_ts, round(done.exit, 6), done.outcome)
            break
    else:
        assert done.outcome == "EOD"


def test_five_pillars():
    ok = S.five_pillars(rvol=8.0, cum_vol=3e6, price=4.0, pct=0.4, mcap=40e6)       # 10M shares out
    assert all(m for _, m in ok)
    bad = dict(S.five_pillars(rvol=float("nan"), cum_vol=5e5, price=25.0, pct=0.12, mcap=2e9))
    assert not bad["RVOL≥5x"] and not bad["vol≥1M sh"] and not bad["$1-20"] and bad["up≥10%"] and not bad["float≤20M*"]


def test_stop_is_structural_and_capped_and_gaps_fill_at_open():
    x = _day(seed=5)
    sig = S.setups(x, S.bar_features(x, 1.8, 50_000))[0]
    tr = S.simulate("T", x, sig)
    i = x.index.get_loc(sig)
    c, h, lo, v = (x[k].to_numpy(float) for k in ("close", "high", "low", "volume"))
    pl = S.pullback_low(h, lo, v, c, i)
    assert pl is not None
    assert abs(tr.stop0 - max(min(pl, tr.entry * 0.995), tr.entry * (1 - S.STOP_PCT))) < 1e-9
    assert 1 - tr.stop0 / tr.entry <= S.STOP_PCT + 1e-9
    y = x.copy()
    y.iloc[i + 2, y.columns.get_loc("open")] = tr.stop0 * 0.7       # halt / gap 30% below the stop
    y.iloc[i + 2, y.columns.get_loc("low")] = tr.stop0 * 0.65
    g = S.simulate("T", y, sig)
    assert g.outcome == "STOP_GAP" and g.exit < tr.stop0 * 0.71
