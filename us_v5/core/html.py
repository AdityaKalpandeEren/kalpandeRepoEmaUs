"""Self-contained HTML research report (charts embedded as PNG)."""
from __future__ import annotations

import base64
import io
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from us_v5.core import metrics  # noqa: E402

CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--muted:#6e6e73;--line:#e5e5ea;--pos:#1a7f37;--neg:#c62828;--card:#f6f6f8}
@media (prefers-color-scheme: dark){:root{--bg:#141416;--fg:#f2f2f4;--muted:#a1a1a6;--line:#2c2c30;--pos:#4cc76a;--neg:#ff6b6b;--card:#1d1d20}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:24px 16px;max-width:1100px;margin:auto}
h1{font-size:26px;margin:0 0 4px} h2{font-size:19px;margin:32px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
.muted{color:var(--muted)} .warn{background:#fff4e5;color:#7a4b00;padding:10px 14px;border-radius:8px;margin:12px 0}
@media (prefers-color-scheme: dark){.warn{background:#3a2a10;color:#ffd08a}}
table{border-collapse:collapse;font-size:13px;margin:8px 0;display:block;overflow-x:auto}
th,td{padding:4px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
img{max-width:100%;height:auto;border-radius:8px;background:#fff}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.kpi{background:var(--card);border-radius:10px;padding:10px 12px}.kpi b{font-size:20px;display:block}
"""

PCT = {"CAGR", "Vol", "MaxDD", "HitRate_daily", "TotalReturn", "Alpha_ann", "Bench_CAGR", "Bench_MaxDD",
       "Return", "Bench_Return", "HitRate_trades", "Turnover_ann", "Cost_drag_ann", "Avg_gross"}


def _png(fig) -> str:
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f'<img alt="chart" src="data:image/png;base64,{base64.b64encode(b.getvalue()).decode()}">'


def _fmt(k: str, v) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        return f"{v * 100:.1f}%" if k in PCT else f"{v:.2f}"
    return str(v)


def table(df: pd.DataFrame) -> str:
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{_fmt(c, r[c])}</td>" for c in df.columns) + "</tr>"
                   for _, r in df.iterrows())
    return f"<table><tr>{head}</tr>{body}</table>"


def equity_chart(curves: dict[str, pd.Series], title: str) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, r in curves.items():
        ax.plot((1 + r.fillna(0)).cumprod(), label=name, lw=1.4 if "SPY" not in name else 1.0)
    ax.set_yscale("log"); ax.set_title(title); ax.legend(frameon=False); ax.grid(alpha=0.3)
    return _png(fig)


def drawdown_chart(curves: dict[str, pd.Series]) -> str:
    fig, ax = plt.subplots(figsize=(10, 2.8))
    for name, r in curves.items():
        eq = (1 + r.fillna(0)).cumprod()
        ax.plot(eq / eq.cummax() - 1, label=name, lw=1.0)
    ax.set_title("Drawdown"); ax.legend(frameon=False); ax.grid(alpha=0.3)
    return _png(fig)


def heatmap(ret: pd.Series, title: str) -> str:
    t = metrics.monthly_table(ret) * 100
    fig, ax = plt.subplots(figsize=(10, 0.35 * len(t) + 1.2))
    lim = np.nanmax(np.abs(t.values)) if t.size else 1
    im = ax.imshow(t.values, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(t.shape[1]), [str(m) for m in t.columns]); ax.set_yticks(range(len(t)), t.index)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7)
    ax.set_title(title); fig.colorbar(im, ax=ax, shrink=0.6)
    return _png(fig)


def bar_chart(s: pd.Series, title: str) -> str:
    s = s.sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.25 * len(s) + 1))
    ax.barh(s.index, s.values); ax.set_title(title); ax.grid(alpha=0.3, axis="x")
    return _png(fig)


def render(path: str, title: str, sections: list[tuple[str, str]], subtitle: str = "") -> None:
    body = "".join(f"<h2>{h}</h2>{html}" if h else html for h, html in sections)
    with open(path, "w") as f:
        f.write(f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' "
                f"content='width=device-width,initial-scale=1'><title>{title}</title><style>{CSS}</style>"
                f"</head><body><h1>{title}</h1><div class='muted'>{subtitle}</div>{body}</body></html>")
