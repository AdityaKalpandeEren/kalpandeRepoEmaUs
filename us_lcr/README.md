# US LCR — Large-Cap Runners (alerts + paper)

The large-cap counterpart of US SCR: **mid ($2–10B), large ($10–200B) and mega (≥ $200B) caps with unusual volume**. It runs as the `lcr` job of **US Alert Scan**, every ~2 min from 09:30 to 16:15 ET.

| Part | What it does |
|---|---|
| 🏛️ Dynamic watchlist | Free Yahoo screener, rebuilt every run: listed stocks with market cap ≥ $2B, up ≥ 2% on volume (plus open positions). Up to 80 names, tagged MID / LARGE / MEGA |
| Volume shocker | **Time-adjusted** relative volume: volume so far vs what a normal day has traded by this time (U-shaped intraday profile) ≥ 2x, up ≥ 2%, ≥ $20M traded, above VWAP |
| 📈 EMA trend rule | **Minimum: price above the 10- and 20-day daily EMAs** (as of the previous close). Every alert shows ✅/❌ for all six (10/20/30/40/60/180). **Above all six = ⭐ PERFECT TRADE** |
| 🚀 Paper trades | **Pullback continuation**: recent high of day, ≥ 1% pullback on lighter volume, reclaim of the previous bar's high. Buy at the next bar's open (+5 bps), stop under the pullback low (max 3%), breakeven after +1R then trail, max 3 h, flat 15:55. Max 13 a day, 2 per stock, 0.5% risk per trade |
| 🏛️📡 Volume watch | ≥ 3x 10-day volume while up < 2% (watch only) |

## Research (2026-10-08, `python -m us_lcr.research`)
42 sessions, 10,829 candidate days (volume ≥ 1x, high ≥ +2%), 12 variants (pullback / HOD × EMA none / above / stacked × 2% / 3% stop), first half vs second half:

| Variant | First half / trade | Second half / trade |
|---|---|---|
| Pullback, no EMA filter, 3% | +0.15% | +0.17% |
| **Pullback, above all 5 EMAs, 3% (LIVE)** | **+0.26%** (t 1.55) | **+0.29%** |
| Pullback, stacked EMAs, 3% | +0.35% (t 2.20) | +0.16% (fewer trades) |
| HOD breakout (any EMA) | −0.17% to −0.04% | mixed |

- **Pullback continuation is positive in both halves for every EMA setting,** and the EMA filter lifts it.
- **HOD breakouts lose.**
- **10 & 20 EMA minimum rule** (2026-10-09, max 13 a day): +0.32% / +0.10% per trade, 7.9 a day. Its **⭐ PERFECT** part (above all six EMAs) made **+0.35% / +0.24%**; the not-perfect rest made +0.22% / **−0.29%**.
- **About 290 trades and t 0.3–1.6, so it's not statistically proven yet.** It's a forward paper test.
