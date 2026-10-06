# US SCR — Small-Cap Runners (live paper + alerts)

Automated version of a "today's top % gainers in small/micro caps" screen. It runs as the `scr` job of
**US Alert Scan**, on the same cron-job.org trigger, every ~2 minutes from 07:00 to 16:15 ET.

| Part | What it does |
|---|---|
| 🔥 Dynamic watchlist | Free Yahoo screener: **listed** US small/micro caps (< $2B, ≥ $1, no OTC) up ≥ 10% today on volume. Also yesterday's movers (pre-market carry-over) and open positions. Max 80 names, refreshed every run |
| 📡 Volume watch | Small caps trading ≥ 5x their 10-day average volume while still up < 10% ("suspicious volume before the move"). Watch-only alert, once per name per day, at most hourly |
| 🚀 Runner trades (paper) | A 5-min bar closes at a new high of day on ≥ 2x bar volume, above VWAP, 09:35–15:30 ET (whole regular session). Buy at the next bar's open, **fixed 8% stop**, breakeven after +1R, then trail under bar lows. Max 3 h hold, flat 15:55. Max 15 a day, one position per stock, 0.5% of $100k paper equity at risk per trade |
| 🧠 ML score | LightGBM on the setup's features, **shown and logged but not a filter**: it didn't beat the plain rule out of sample |

Alerts use their own 🔥 / 📡 format and a ⚠️ high-risk line, so they never look like the other strategies.

## Research (2026-10-06, `python -m us_scr.research`)
- **Data:** 2,927 listed small/micro caps (Nasdaq's public list). 5,662 runner stock-days (intraday high ≥ +10%) over the last ~40 sessions. That's Yahoo's 60-day limit for 5-min data. Costs: 0.25% slippage per side.
- **Breakout-chasing has about 0% edge before costs.** 24 rule variants were tested (entry, stop, setup, time window), choosing on the first 20 days and checking on the last 20, and none was positive in both halves.
- **The final rule** (8% stop, first 90 minutes): 373 trades, 34% win, **−0.06R / −0.5% per trade**. The original bar-low stop was −0.38R, stopped out 62% of the time.
- **No pre-market trading:** Yahoo reports **0 volume on pre/post-market bars**, so pre-market runs can't be volume-confirmed. Pre-market price moves still count (% change, opening gap).
- **Full-session entries** (09:35–15:30 ET, user request): 1,308 trades, 36% win, −0.09R / −0.69% per trade. By entry hour, 11:00–13:59 ET is the least bad (11:xx +0.05R) and 14:00–15:30 the worst (about −0.2R).
- **Status: UNPROVEN.** It's a forward paper test. Every watched bar is stored (`us_scr/cache/bars/`), so the model can be retrained on months of data instead of 40 days.

| | |
|---|---|
| Live | `python -m us_scr.live` (state `live_state/us_scr/`) |
| Research | `python -m us_scr.research` (needs `us_scr/cache`, built by the download in research) |
| Tests | `python -m pytest us_scr/tests -q` (causal setups, live = research exits, stop / gap fills) |
