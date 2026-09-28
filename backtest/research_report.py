"""
Research reporting: ranks strategies and slices results by regime,
direction, and symbol.

Includes two things most retail backtest reports leave out, both of
which usually matter more than the headline win rate:

  1. EXPECTANCY IN R, not win rate, as the ranking key. A 70%-win-rate
     model that risks 1R to make 0.3R loses money. Win rate alone is
     not a performance measure and ranking by it produces bad decisions.

  2. A SAMPLE-SIZE WARNING per strategy. With ~40 trading days of data
     (Yahoo's intraday retention limit), most models will produce a few
     dozen trades. The standard error on a win rate from 30 trades is
     roughly +/-9 percentage points, which is wider than nearly every
     difference you will see between models. The report says so
     explicitly, per strategy, so a 3-point win-rate gap is not
     mistaken for an edge.
"""
import math
import os
from dataclasses import asdict
from datetime import datetime

import pandas as pd


def trades_to_dataframe(trades: list) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([asdict(t) for t in trades])


def _summarize(df: pd.DataFrame) -> dict:
    total = len(df)
    wins = int((df["outcome"] == "TARGET").sum())
    losses = int((df["outcome"] == "STOP").sum())
    scratches = int((df["outcome"] == "EOD_SQUAREOFF").sum())
    decided = wins + losses
    win_rate = wins / decided * 100 if decided else 0.0

    r = df["r_multiple"]
    expectancy = r.mean() if total else 0.0
    total_r = r.sum() if total else 0.0
    std_r = r.std() if total > 1 else 0.0

    gross_win = r[r > 0].sum()
    gross_loss = abs(r[r < 0].sum())
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0

    # Standard error on the win rate, the honest error bar on this
    # sample. Below ~30 decided trades the estimate is near-useless.
    if decided >= 2:
        p = wins / decided
        se = math.sqrt(p * (1 - p) / decided) * 100
    else:
        se = float("nan")

    # Expectancy t-stat: how many standard errors the mean R sits above
    # zero. Below ~2 the result is not distinguishable from noise.
    t_stat = (expectancy / (std_r / math.sqrt(total))) if total > 1 and std_r > 0 else 0.0

    equity = r.cumsum()
    drawdown = (equity - equity.cummax()).min() if total else 0.0

    return {
        "trades": total,
        "wins": wins,
        "losses": losses,
        "scratches": scratches,
        "win_rate_pct": round(win_rate, 1),
        "win_rate_se_pct": round(se, 1) if not math.isnan(se) else None,
        "expectancy_r": round(expectancy, 3),
        "total_r": round(total_r, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else None,
        "max_drawdown_r": round(drawdown, 2),
        "t_stat": round(t_stat, 2),
        "avg_candles_held": round(df["candles_held"].mean(), 1) if total else 0.0,
        "sample_adequate": decided >= 30,
    }


def summarize(trades: list) -> dict:
    df = trades_to_dataframe(trades)
    if df.empty:
        return {"total_trades": 0}

    out = {
        "total_trades": len(df),
        "overall": _summarize(df),
        "by_strategy": {k: _summarize(g) for k, g in df.groupby("strategy")},
        "by_direction": {k: _summarize(g) for k, g in df.groupby("direction")},
        "by_regime": {k: _summarize(g) for k, g in df.groupby("regime")},
        "by_symbol": {k: _summarize(g) for k, g in df.groupby("symbol")},
    }
    # Strategy x regime: the slice that reveals "good model, missing
    # filter". Kept to combinations with enough trades to mean anything.
    out["by_strategy_regime"] = {
        f"{s} @ {rg}": _summarize(g)
        for (s, rg), g in df.groupby(["strategy", "regime"]) if len(g) >= 10
    }
    return out


def _rank(summary: dict) -> list:
    """Strategies ranked by expectancy in R, not win rate."""
    rows = [{"strategy": k, **v} for k, v in summary.get("by_strategy", {}).items()]
    return sorted(rows, key=lambda r: r["expectancy_r"], reverse=True)


def _fmt_group(name: str, g: dict) -> str:
    se = f" ±{g['win_rate_se_pct']}" if g.get("win_rate_se_pct") is not None else ""
    warn = "" if g["sample_adequate"] else "   [SMALL SAMPLE - not conclusive]"
    pf = g["profit_factor"] if g["profit_factor"] is not None else "inf"
    return (
        f"### {name}{warn}\n"
        f"- Trades: {g['trades']}  (W {g['wins']} / L {g['losses']} / EOD {g['scratches']})\n"
        f"- Win rate: {g['win_rate_pct']}%{se}\n"
        f"- Expectancy: {g['expectancy_r']} R/trade   |   Total: {g['total_r']} R\n"
        f"- Profit factor: {pf}   |   Max drawdown: {g['max_drawdown_r']} R\n"
        f"- t-stat: {g['t_stat']}  (below 2.0 = not distinguishable from noise)\n"
        f"- Avg hold: {g['avg_candles_held']} candles\n"
    )


def _render_md(summary: dict, meta: dict) -> str:
    if summary.get("total_trades", 0) == 0:
        return "# Strategy research report\n\nNo trades generated.\n"

    lines = ["# Strategy research report", ""]
    lines.append(f"Period: {meta.get('from')} to {meta.get('to')}  |  "
                 f"{meta.get('interval')}-min candles  |  {meta.get('symbols')} symbols  |  "
                 f"{meta.get('days', '?')} trading days")
    lines.append(f"Costs applied: {meta.get('costs')}  |  Regime filter: {meta.get('regime_filter')}")
    lines.append("")
    lines.append("> Ranked by EXPECTANCY (R per trade), not win rate. A high win rate")
    lines.append("> with a poor payoff still loses money. Read the t-stat and the")
    lines.append("> sample-size warnings before believing any ranking below.")
    lines.append("")

    lines.append("## Strategy ranking")
    lines.append("")
    lines.append("| # | Strategy | Trades | Win% | Expectancy (R) | Total R | PF | t | Verdict |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(_rank(summary), 1):
        pf = r["profit_factor"] if r["profit_factor"] is not None else "inf"
        if not r["sample_adequate"]:
            verdict = "too few trades"
        elif r["expectancy_r"] > 0 and r["t_stat"] >= 2.0:
            verdict = "worth testing further"
        elif r["expectancy_r"] > 0:
            verdict = "positive but noisy"
        else:
            verdict = "negative expectancy"
        lines.append(f"| {i} | {r['strategy']} | {r['trades']} | {r['win_rate_pct']} | "
                     f"{r['expectancy_r']} | {r['total_r']} | {pf} | {r['t_stat']} | {verdict} |")
    lines.append("")

    lines.append(_fmt_group("Overall (all strategies pooled)", summary["overall"]))

    for section, title in [
        ("by_strategy", "By strategy"),
        ("by_direction", "Long vs short"),
        ("by_regime", "By market regime"),
        ("by_strategy_regime", "Strategy x regime (>=10 trades only)"),
    ]:
        lines.append(f"## {title}")
        for name, g in sorted(summary.get(section, {}).items(),
                               key=lambda kv: kv[1]["expectancy_r"], reverse=True):
            lines.append(_fmt_group(name, g))

    lines.append("## How to read this")
    lines.append("")
    lines.append("- **Expectancy** is the number that decides whether a model makes money.")
    lines.append("- **t-stat below 2.0** means the result is within noise, however good it looks.")
    lines.append("- **Strategy x regime** is the most actionable section: a model that is")
    lines.append("  negative overall but strongly positive in one regime is a model with a")
    lines.append("  missing filter, not a bad model.")
    lines.append("- Nine models were tested on one dataset, so the top result is partly")
    lines.append("  selection luck. Re-run the leaders on a different date range before")
    lines.append("  trusting the ranking.")
    return "\n".join(lines)


def write_research_report(trades: list, out_dir: str, meta: dict) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = trades_to_dataframe(trades)
    csv_path = os.path.join(out_dir, f"research_trades_{stamp}.csv")
    df.to_csv(csv_path, index=False)

    summary = summarize(trades)
    md_path = os.path.join(out_dir, f"research_report_{stamp}.md")
    with open(md_path, "w") as f:
        f.write(_render_md(summary, meta))

    return {"csv": csv_path, "report": md_path, "summary": summary}


def print_research_summary(summary: dict):
    if summary.get("total_trades", 0) == 0:
        print("No trades generated.")
        return

    print("\n" + "=" * 78)
    print("STRATEGY RANKING  (by expectancy in R, not win rate)")
    print("=" * 78)
    print(f"{'#':<3}{'STRATEGY':<20}{'N':>6}{'WIN%':>8}{'EXP_R':>9}{'TOT_R':>9}{'t':>7}  VERDICT")
    print("-" * 78)
    for i, r in enumerate(_rank(summary), 1):
        if not r["sample_adequate"]:
            verdict = "too few trades"
        elif r["expectancy_r"] > 0 and r["t_stat"] >= 2.0:
            verdict = "worth testing further"
        elif r["expectancy_r"] > 0:
            verdict = "positive but noisy"
        else:
            verdict = "negative"
        print(f"{i:<3}{r['strategy']:<20}{r['trades']:>6}{r['win_rate_pct']:>8}"
              f"{r['expectancy_r']:>9}{r['total_r']:>9}{r['t_stat']:>7}  {verdict}")

    o = summary["overall"]
    print("-" * 78)
    print(f"POOLED: {o['trades']} trades | win {o['win_rate_pct']}% | "
          f"expectancy {o['expectancy_r']}R | total {o['total_r']}R | maxDD {o['max_drawdown_r']}R")

    print("\n" + "=" * 78)
    print("BY REGIME  (the most actionable slice)")
    print("=" * 78)
    for name, g in sorted(summary.get("by_regime", {}).items(),
                           key=lambda kv: kv[1]["expectancy_r"], reverse=True):
        flag = "" if g["sample_adequate"] else "  [small sample]"
        print(f"  {name:<26} n={g['trades']:<5} win={g['win_rate_pct']:<6}% "
              f"exp={g['expectancy_r']:<7}R{flag}")

    print("\n" + "=" * 78)
    print("LONG vs SHORT")
    print("=" * 78)
    for name, g in summary.get("by_direction", {}).items():
        flag = "" if g["sample_adequate"] else "  [small sample]"
        print(f"  {name:<8} n={g['trades']:<5} win={g['win_rate_pct']:<6}% "
              f"exp={g['expectancy_r']:<7}R{flag}")
