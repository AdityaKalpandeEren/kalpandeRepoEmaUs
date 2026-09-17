"""
Turns a list of strategy.trade_engine.TradeResult into an accuracy
report - overall, and broken down by strategy and by symbol.
Self-contained here (rather than living in a shared backtest/ package
like the NSE bot has) since this round only adds paper trading to the
US bot - a historical backtester can be added the same way later if
you want one, following the NSE bot's backtest/ folder as the template.
"""
import os
from datetime import datetime
from dataclasses import asdict

import pandas as pd


def trades_to_dataframe(trades: list) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([asdict(t) if hasattr(t, "__dataclass_fields__") else dict(t) for t in trades])


def _summarize_group(df: pd.DataFrame) -> dict:
    total = len(df)
    wins = int((df["outcome"] == "TARGET").sum())
    losses = int((df["outcome"] == "STOP").sum())
    scratches = int((df["outcome"] == "EOD_SQUAREOFF").sum())
    decided = wins + losses  # win rate on trades that actually resolved, excl. EOD scratches
    win_rate = round(wins / decided * 100, 1) if decided else 0.0
    win_rate_incl_scratch = round(wins / total * 100, 1) if total else 0.0

    avg_r = round(df["r_multiple"].mean(), 2) if total else 0.0
    total_r = round(df["r_multiple"].sum(), 2) if total else 0.0
    avg_win_r = round(df.loc[df["outcome"] == "TARGET", "r_multiple"].mean(), 2) if wins else 0.0
    avg_loss_r = round(df.loc[df["outcome"] == "STOP", "r_multiple"].mean(), 2) if losses else 0.0

    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "scratches_eod": scratches,
        "win_rate_pct": win_rate,
        "win_rate_pct_incl_scratch": win_rate_incl_scratch,
        "avg_r_per_trade": avg_r,        # expectancy, in R (multiples of risk)
        "total_r": total_r,
        "avg_win_r": avg_win_r,
        "avg_loss_r": avg_loss_r,
    }


def summarize(trades: list) -> dict:
    df = trades_to_dataframe(trades)
    if df.empty:
        return {"total_trades": 0}

    overall = _summarize_group(df)
    return {
        "total_trades": overall["total_trades"],
        "overall": overall,
        "by_strategy": {name: _summarize_group(g) for name, g in df.groupby("strategy")},
        "by_symbol": {name: _summarize_group(g) for name, g in df.groupby("symbol")},
    }


def _render_group_md(name: str, g: dict) -> str:
    return (
        f"### {name}\n"
        f"- Trades: {g['total_trades']}\n"
        f"- Wins / Losses / EOD scratch: {g['wins']} / {g['losses']} / {g['scratches_eod']}\n"
        f"- Win rate (target vs stop only): {g['win_rate_pct']}%\n"
        f"- Win rate (EOD scratches counted as loss): {g['win_rate_pct_incl_scratch']}%\n"
        f"- Avg R per trade (expectancy): {g['avg_r_per_trade']}\n"
        f"- Avg winning trade: {g['avg_win_r']}R | Avg losing trade: {g['avg_loss_r']}R\n"
        f"- Total R across all trades: {g['total_r']}\n"
    )


def _render_markdown(summary: dict, label: str) -> str:
    if summary.get("total_trades", 0) == 0:
        return f"# {label} report\n\nNo trades were generated in this run.\n"

    lines = [f"# {label} report", "", _render_group_md("Overall", summary["overall"]), "## By strategy"]
    for name, g in summary["by_strategy"].items():
        lines.append(_render_group_md(name, g))
    lines.append("## By symbol")
    for name, g in summary["by_symbol"].items():
        lines.append(_render_group_md(name, g))
    return "\n".join(lines)


def write_report(trades: list, out_dir: str, label: str = "paper_trading") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = trades_to_dataframe(trades)
    csv_path = os.path.join(out_dir, f"{label}_trades_{stamp}.csv")
    df.to_csv(csv_path, index=False)

    summary = summarize(trades)
    md_path = os.path.join(out_dir, f"{label}_report_{stamp}.md")
    with open(md_path, "w") as f:
        f.write(_render_markdown(summary, label))

    return {"csv": csv_path, "report": md_path, "summary": summary}


def _print_group(g: dict):
    print(f"  Trades: {g['total_trades']}  (Win {g['wins']} / Loss {g['losses']} / EOD scratch {g['scratches_eod']})")
    print(f"  Win rate: {g['win_rate_pct']}% (target-vs-stop) | {g['win_rate_pct_incl_scratch']}% (incl. EOD as loss)")
    print(f"  Avg R/trade: {g['avg_r_per_trade']}  | Avg win: {g['avg_win_r']}R  | Avg loss: {g['avg_loss_r']}R")
    print(f"  Total R: {g['total_r']}")


def print_summary(summary: dict):
    if summary.get("total_trades", 0) == 0:
        print("No trades were generated in this run.")
        return
    print("\n=== OVERALL ===")
    _print_group(summary["overall"])
    print("\n=== BY STRATEGY ===")
    for name, g in summary["by_strategy"].items():
        print(f"\n-- {name} --")
        _print_group(g)
    print("\n=== BY SYMBOL ===")
    for name, g in summary["by_symbol"].items():
        print(f"\n-- {name} --")
        _print_group(g)
