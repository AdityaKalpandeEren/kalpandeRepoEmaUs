"""
US cash-equity charges, per order. All rates in config.costs.

  commission  per-share commission (0 at most US retail brokers)
  SEC fee     Section 31 fee on SELL proceeds only
  FINRA TAF   per share sold, capped per trade (approximated on value here:
              value / price is unknown in the engine, so TAF uses an average
              share price assumption - it is tiny either way)
  slippage    applied to the fill price: buy at open*(1+s), sell at open*(1-s)
No stamp duty / STT / DP charges in the US.
"""
from __future__ import annotations


def order_charges(value: float, side: str, c: dict) -> float:
    """Explicit charges ($) for one order of `value` $; excludes slippage."""
    if value <= 0:
        return 0.0
    fee = c.get("commission_per_order", 0.0)
    if side == "sell":
        fee += c["sec_fee"] * value
        shares = value / max(c.get("avg_share_price", 100.0), 1.0)
        fee += min(c["finra_taf_per_share"] * shares, c["finra_taf_cap"])
    return fee


def fill_price(price: float, side: str, c: dict) -> float:
    s = c["slippage_bps"] / 1e4
    return price * (1 + s) if side == "buy" else price * (1 - s)


def round_trip_pct(value: float, c: dict) -> float:
    """Total cost of buying and later selling `value` $, as a fraction (incl. slippage)."""
    s = c["slippage_bps"] / 1e4
    return (order_charges(value, "buy", c) + order_charges(value, "sell", c)) / value + 2 * s
