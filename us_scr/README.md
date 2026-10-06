# US SCR — Small-Cap Runners (live paper + alerts)

Automated version of a "today's top % gainers in small/micro caps" screen. It runs as the `scr` job of
**US Alert Scan**, on the same cron-job.org trigger, every ~2 minutes from 07:00 to 16:15 ET.

| Part | What it does |
|---|---|
| 🔥 Dynamic watchlist | Free Yahoo screener: **listed** US small/micro caps (< $2B, ≥ $1, no OTC) up ≥ 10% today on volume. Also yesterday's movers (pre-market carry-over) and open positions. Max 80 names, refreshed every run |
| 📡 Volume watch | Small caps trading ≥ 5x their 10-day average volume **and ≥ 1M shares** while still up < 10% ("suspicious volume before the move"). Watch-only alert, once per name per day, at most hourly |
| 🚀 Runner trades (paper) | **Pullback continuation**: the high of day was made within the last 6 bars, the stock pulled back ≥ 2% on lighter volume, and a bar now closes above the previous bar's high, above VWAP and within 3% of the high (09:35–15:30 ET). Buy at the next bar's open; the **stop goes under the pullback low** (max 8%, median 3.8%); breakeven after +1R, then trail under bar lows. Max 3 h hold, flat 15:55. Max 15 a day, one position per stock, 0.5% of $100k paper equity at risk per trade |
| ⭐ 5-pillar tags | Each alert shows the small-cap momentum checklist: RVOL ≥ 5x, ≥ 1M shares, $1–20, up ≥ 10%, float ≤ 20M (shares outstanding as a proxy) → ⭐ A+ (5/5) / ✳️ strong (4/5) / ▫️ partial. Info only |
| 🧠 ML score | LightGBM on the setup's features, **shown and logged but not a filter**: it didn't beat the plain rule out of sample |

Alerts use their own 🔥 / 📡 format and a ⚠️ high-risk line, so they never look like the other strategies.

## Research (2026-10-06, `python -m us_scr.research`)
- **Data:** 2,927 listed small/micro caps (Nasdaq's public list). 5,662 runner stock-days (intraday high ≥ +10%) over the last ~40 sessions. That's Yahoo's 60-day limit for 5-min data. Costs: 0.25% slippage per side.
- **Breakout-chasing has about 0% edge before costs.** 24 rule variants were tested (entry, stop, setup, time window), choosing on the first 20 days and checking on the last 20, and none was positive in both halves.
- **The final rule** (8% stop, first 90 minutes): 373 trades, 34% win, **−0.06R / −0.5% per trade**. The original bar-low stop was −0.38R, stopped out 62% of the time.
- **No pre-market trading:** Yahoo reports **0 volume on pre/post-market bars**, so pre-market runs can't be volume-confirmed. Pre-market price moves still count (% change, opening gap).
- **Full-session entries** (09:35–15:30 ET, user request): 1,308 trades, 36% win, −0.09R / −0.69% per trade. By entry hour, 11:00–13:59 ET is the least bad (11:xx +0.05R) and 14:00–15:30 the worst (about −0.2R).
- **The research playbook was tested too** (2026-10-06; "5 pillars": RVOL ≥ 5x, millions of shares, $1–20, low float; setups: HOD breakout, pullback continuation, pre-market-high break, VWAP reclaim; structural vs fixed stops; trail vs 2R target). That's 48 variants, chosen on the first 20 sessions and checked on the last 20. **None was positive in both halves.** Before costs, everything is about 0%. Pullback continuation lost the least in both halves (about −0.43%), so it replaced the HOD breakout. Final rule: 912 trades, 38% win, −0.48% per trade.
- **Status: UNPROVEN.** It's a forward paper test. Every watched bar is stored (`us_scr/cache/bars/`), so the model can be retrained on months of data instead of 40 days.

| | |
|---|---|
| Live | `python -m us_scr.live` (state `live_state/us_scr/`) |
| Research | `python -m us_scr.research` (needs `us_scr/cache`, built by the download in research) |
| Tests | `python -m pytest us_scr/tests -q` (causal setups, live = research exits, stop / gap fills) |
