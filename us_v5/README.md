# US V5.0 — swing ranking model (live paper)

US port of the NSE V5.0 swing model (`nse_alert_bot` `v5/` + `swing/`).
A daily cross-sectional ranker for the 100 most liquid S&P 500 stocks. It runs live as the
`v5` job of **US Alert Scan**, on the same cron-job.org trigger.

- **When:** on the first run after 09:35 ET.
- **Signal:** it scores the universe from the previous close.
- **Rebalance:** every 5th session, it rebalances a **$100k** paper portfolio at today's open.
  - It buys the top 20, sized inverse to volatility, with caps of 15% per stock and 30% per sector.
  - A holding stays while it ranks in the top 50.
- **Costs:** 5 bps slippage per side, plus SEC/FINRA fees.
- **Telegram:** rebalance orders, or a daily portfolio update vs SPY.
- **Paper only.**

| | |
|---|---|
| Live runner | `python -m us_v5.live` (state: `live_state/us_v5/`) |
| Model | `us_v5/model/us_v5_model.joblib` (LightGBM ×3 seeds + XGBoost) |
| Research | `python -m us_v5.research` (walk-forward, ablations, Deflated Sharpe); `--holdout` runs the one-time holdout and refits the live model |
| Tests | `python -m pytest us_v5/tests -q` (no-lookahead, purge, costs) |

## What carries over from NSE V5
- **Target:** the average daily rank of residual returns over 5, 10 and 20 days, with beta and sector effects removed.
- **Scores:** smoothed with an EWMA (half-life 5 sessions).
- **Models:** an ML ensemble, a 52-week-high + 12-1 momentum factor, and a 50/50 combo of the two. The research run picks whichever has the best walk-forward Sharpe.
- **Validation:** purged walk-forward, retrained every 12 months.
- **Portfolio:** top 20 with a no-trade band, plus a drawdown breaker.

## What differs for the US
- **Data:** Yahoo daily bars, total-return adjusted (dividends included). The benchmark is SPY total return.
- **Universe:**
  - Each month it's the top 100 by median dollar volume among today's S&P 500 constituents.
  - A stock only counts once it was in the index, by its Wikipedia "Date added" (`us_v5/seed/sp500.csv`).
- **No delivery or F&O data.** Context features are used instead: QQQ/IWM/RSP vs SPY, VIX (level, percentile, VIX/VIX3M), the 10-year yield, the dollar, crude, credit (HYG vs IEF), and universe breadth and dispersion.
- **Costs:** about 0.10–0.11% round trip, against about 0.25% on NSE.

## Known bias
Yahoo has no history for delisted tickers (SIVB, FRC, ATVI…), so stocks that left the index
are missing. That makes the backtest look better than reality. The research report therefore also shows an **equal-weight
portfolio of the same universe**, which has the same bias. Only an edge over that portfolio reflects real ranking
skill. The forward paper record is the real test.

## Research result (2026-10-05, after costs)
| | Walk-forward 2017–2024 CAGR / Sharpe / MaxDD | Holdout Jul 2025–Oct 2026 return / Sharpe / MaxDD |
|---|---|---|
| **v5_factor (live)** | 13.8% / 0.58 / −28.5% | +32.2% / 1.02 / −12.8% |
| v5_ml | 11.7% / 0.52 / −30.1% | +7.8% / 0.23 / −9.9% |
| v5_combo | 11.8% / 0.51 / −30.1% | +11.3% / 0.45 / −11.2% |
| SPY (total return) | 14.4% / 0.60 / −33.7% | +27.1% / 1.30 / −8.9% |
| EW universe (gross) | 15.6% / 0.63 / −34.8% | +31.6% / 1.28 / −9.2% |

- **Walk-forward:** the selected `v5_factor` (52-week-high + 12-1 momentum) **lost to SPY**.
- **Ranking skill is weak:** IC is about 0.014, with t ≈ 1.4.
- **Holdout:** it beat SPY's return, but with a lower Sharpe and a deeper drawdown, and it was level with the equal-weight universe.
- **Live anyway, as a forward paper test:** at the user's request (2026-10-05). Treat it as unproven.
- **Model:** the live file still holds the ML ensemble, but `selected = v5_factor`, so live picks come from the factor score.
