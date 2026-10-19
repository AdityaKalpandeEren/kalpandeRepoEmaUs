# US V8 — top momentum traders' setups on US stocks (research only, nothing went live)

Started 2026-10-16 together with the NSE V8 study (NSE repo `v8/README.md`: same setup code, same engine). Setups,
fixed before the first run: **BO** Qullamaggie breakout inside Minervini's trend template, **EP** Qullamaggie episodic
pivot (gap ≥ 10% on ≥ 3× volume, holds into the close), **ATH** Wilcox & Crittenden all-time-high trend following with a
10 × ATR(42) trailing stop (and the 2-year-high version a live job could run). Rules: `us_v8/setups.py`; simulator:
`us_v8/engine.py`. Costs: 5 bps slippage at the open/close, 15 bps for stop orders, SEC fee. $100k book.
WF = 2017-01 → 2025-06 (halves A / B), PV = 2025-07 → 2026-10. Keep rule: beat SPY and the equal-weight average of the
same stocks in both WF halves.

    python -m us_v8.research                    # current S&P 500, each from its "date added"
    python -m us_v8.fetch_broad us_v8/cache/broad && python -m us_v8.research --universe broad

**Both universes are survivorship-biased** (Yahoo has no delisted tickers), so these results are optimistic.

## Results (after costs, Sharpe with rf = 0)

| Variant | S&P 500: WF CAGR / Sharpe | PV | Broad (~2,000 names ≥ $300M): WF CAGR / Sharpe | PV |
|---|---|---|---|---|
| BO, 10-day trail | −13.5% / −1.34 | −8.8% | −21.7% / −1.09 | −49.4% |
| BO, 20-day trail | −12.6% / −1.17 | −8.8% | −19.8% / −0.90 | −37.7% |
| BO, no market filter | −13.1% / −1.12 | −19.5% | −24.0% / −1.09 | −53.0% |
| EP, 10-day trail | −2.0% / −0.53 | +3.4% | +0.9% / 0.14 | −13.5% |
| EP, 20-day trail | −2.1% / −0.54 | +3.0% | +2.4% / 0.23 | +6.9% |
| EP, not rallied before | −2.3% / −0.66 | +1.1% | +0.4% / 0.09 | −8.1% |
| BO + EP | −14.7% / −1.40 | −6.2% | −13.5% / −0.55 | −36.2% |
| ATH, 10 × ATR | +6.3% / 0.46 | +11.6% | +10.3% / 0.56 | +9.6% |
| ATH 2-year high, 10 × ATR | +7.2% / 0.51 | +13.8% | +9.2% / 0.50 | +10.5% |
| **SPY** | **+14.6% / 0.82** | **+20.4%** | **+14.6% / 0.82** | **+20.4%** |
| Equal-weight same stocks | +14.5% / 0.80 | +14.6% | +12.1% / 0.64 | +18.2% |

**Nothing passed.** Breakouts on US large caps have no edge even before costs (S&P 500, optimistic fills, no costs:
−4% to −5% a year): average win +2.9% vs loss −2.1% — S&P names move ~2% a day, Qullamaggie trades 5%+ ADR names.
On the broad universe the breakouts lose more. ATH trend following, which works on NSE (+21%/yr), trails SPY in the US
2017-26 (mega-cap concentration; S&P names at all-time highs did worse than the index).

## What a real US test needs
Survivorship-free daily (and minute) history of all US stocks, including delisted ones — e.g. a free Alpaca account
(minute bars from 2016 via the free SIP feed > 15 min old) — to test the small/mid-cap setups these traders actually
trade and intraday ideas such as Zarattini, Aziz & Barbon's SPY intraday momentum (reported 19.6%/yr, Sharpe 1.33,
2007-2024; failed on NSE, see the NSE repo `v7/README.md` 3a).
