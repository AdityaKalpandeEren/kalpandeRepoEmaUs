"""python -m pytest us_lcr/tests -q - synthetic bars, no network."""
import numpy as np
import pandas as pd

from us_lcr import strategy as S

EMAS_UP = {"ema10": 98.0, "ema20": 96.5, "ema30": 95.0, "ema40": 94.0, "ema60": 92.0, "ema180": 85.0}


def _day(seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-10-07 09:30", "2026-10-07 15:55", freq="5min", tz="America/New_York")
    n = len(idx)
    steps = rng.normal(0, 0.0005, n)
    vol = rng.integers(40_000, 60_000, n).astype(float)
    k = 3
    while k + 9 < n:
        steps[k:k + 6] += 0.004
        steps[k + 6:k + 9] -= 0.003
        vol[k:k + 6] *= 6
        vol[k + 6:k + 9] *= 2
        k += 9
    px = 100 * np.cumprod(1 + steps)
    o = np.r_[px[0], px[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, px) * 1.0005, "low": np.minimum(o, px) * 0.9995,
                         "close": px, "volume": vol}, index=idx)


def _sig(x, emas=EMAS_UP, ema_mode=None):
    return S.setups(x, S.bar_features(x, 97.0, 1_500_000), emas, ema_mode=ema_mode)


def test_setups_are_causal():
    x = _day()
    full = _sig(x)
    assert len(full) > 0
    for cut in (20, 35, 50):
        part = x.iloc[:cut]
        assert list(_sig(part)) == [t for t in full if t <= part.index[-1]]


def test_live_matches_research():
    x = _day(seed=2)
    sig = _sig(x)[0]
    done = S.simulate("T", x, sig)
    i = x.index.get_loc(sig)
    for cut in range(i + 2, len(x) + 1):
        tr = S.simulate("T", x.iloc[:cut], sig, final=False)
        if tr.outcome != "OPEN":
            assert (tr.exit_ts, round(tr.exit, 6), tr.outcome) == (done.exit_ts, round(done.exit, 6), done.outcome)
            break


def test_ema_filter():
    assert S.ema_ok(100, EMAS_UP, "above") and S.ema_ok(100, EMAS_UP, "stacked")
    weak_long = {**EMAS_UP, "ema60": 103.0, "ema180": 110.0}       # above 10 & 20, below the long EMAs
    assert S.ema_ok(100, weak_long, "min") and not S.is_perfect(100, weak_long) and S.is_perfect(100, EMAS_UP)
    assert not S.ema_ok(100, {**EMAS_UP, "ema20": 101.0}, "min")
    below = {**EMAS_UP, "ema180": 101.0}
    assert not S.ema_ok(100, below, "above") and S.ema_ok(100, below, "none")
    x = _day()
    assert len(_sig(x, below, "above")) == 0 or all(x.loc[t, "close"] > 101 for t in _sig(x, below, "above"))


def test_time_adjusted_rvol():
    x = _day()
    f = S.bar_features(x, 97.0, 1_500_000)
    early = f.index[3]
    assert f.loc[early, "rvol"] > f.loc[early, "rvol_day"] * 3      # early volume is scaled up by the profile
