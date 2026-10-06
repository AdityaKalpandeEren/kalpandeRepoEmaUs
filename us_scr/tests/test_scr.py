"""python -m pytest us_scr/tests -q  - synthetic bars, no network."""
import numpy as np
import pandas as pd

from us_scr import strategy as S


def _day(seed=0, spike_at=68):
    """5-min bars 04:00-15:55 ET: quiet, then a volume surge + run-up."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-10-05 04:00", "2026-10-05 15:55", freq="5min", tz="America/New_York")
    n = len(idx)
    px = 2.0 * np.cumprod(1 + rng.normal(0, 0.002, n))
    px[spike_at:] *= np.linspace(1.0, 1.6, n - spike_at)          # the run
    vol = rng.integers(2_000, 5_000, n).astype(float)
    vol[spike_at:] *= 20
    o = np.r_[px[0], px[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, px) * 1.003, "low": np.minimum(o, px) * 0.997,
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


def test_stop_is_tight_and_gaps_fill_at_open():
    x = _day(seed=5)
    sig = S.setups(x, S.bar_features(x, 1.8, 50_000))[0]
    tr = S.simulate("T", x, sig)
    assert abs(1 - tr.stop0 / tr.entry - S.STOP_PCT) < 1e-9
    i = x.index.get_loc(sig)
    y = x.copy()
    y.iloc[i + 2, y.columns.get_loc("open")] = tr.stop0 * 0.7       # halt / gap 30% below the stop
    y.iloc[i + 2, y.columns.get_loc("low")] = tr.stop0 * 0.65
    g = S.simulate("T", y, sig)
    assert g.outcome == "STOP_GAP" and g.exit < tr.stop0 * 0.71
