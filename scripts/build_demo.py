"""Regenerate the GitHub Pages demo (docs/) from live data.

Run from the repo root:  python scripts/build_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tvchart import build_chart, enrich, load, save_html, trade_plan  # noqa: E402

DOCS = ROOT / "docs"
DEMO = DOCS / "demo"
DEMO.mkdir(parents=True, exist_ok=True)

JOBS = [
    ("aapl_1d", "AAPL", "1d", "1y", "dark", "long"),
    ("aapl_15m", "AAPL", "15m", "1mo", "light", None),
    ("nvda_1d", "NVDA", "1d", "1y", "dark", "long"),
    ("spy_1wk", "SPY", "1wk", "5y", "dark", None),
]

generated = []
for name, ticker, interval, period, theme, plan in JOBS:
    try:
        df, meta = load(ticker, interval=interval, period=period)
        df = enrich(df)
        plan_dict = trade_plan(df, direction=plan) if plan else None
        fig = build_chart(df, ticker, theme=theme, plan=plan_dict)
        out = DEMO / f"{name}.html"
        save_html(fig, out, include_plotlyjs=False, title=f"{ticker} {interval}")
        generated.append((name, ticker, interval, out.stat().st_size))
        print(f"  {name:<10} {ticker:<5} {interval:<4} {out.stat().st_size/1024:6.0f} KB")
    except Exception as exc:
        print(f"  {name}: FAILED {exc}", file=sys.stderr)

INDEX = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>tvchart — TradingView-style charts in Python</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #131722; color: #d1d4dc;
         font: 15px/1.6 Inter, -apple-system, "Segoe UI", sans-serif; }
  .wrap { max-width: 1080px; margin: 0 auto; padding: 56px 24px 80px; }
  h1 { font-size: 40px; line-height: 1.1; margin: 0 0 12px; letter-spacing: -0.02em; }
  h1 span { color: #26a69a; }
  p.lead { color: #9aa0ac; font-size: 18px; max-width: 720px; }
  code, pre { font-family: ui-monospace, "SF Mono", Menlo, monospace; }
  pre { background: #1b1f2b; border: 1px solid #2a2e39; border-radius: 10px;
        padding: 16px 18px; overflow-x: auto; font-size: 13.5px; }
  a { color: #26a69a; }
  .chart { border: 1px solid #2a2e39; border-radius: 12px; overflow: hidden; margin: 28px 0 8px; }
  iframe { width: 100%; height: 620px; border: 0; display: block; }
  .row { display: flex; gap: 12px; flex-wrap: wrap; margin: 18px 0 0; }
  .row a { border: 1px solid #2a2e39; border-radius: 8px; padding: 8px 14px;
           text-decoration: none; font-size: 13px; color: #d1d4dc; }
  .row a:hover { border-color: #26a69a; }
  .muted { color: #787b86; font-size: 13px; }
  h2 { margin-top: 44px; font-size: 22px; }
  table { border-collapse: collapse; width: 100%; font-size: 14px; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #2a2e39; }
  th { color: #787b86; font-weight: 500; }
</style>
</head>
<body>
<div class="wrap">
  <h1>tvchart <span>·</span> TradingView-style charts, in Python</h1>
  <p class="lead">A tested library and CLI that turns yfinance + pandas + Plotly into
  a three-panel technical chart, a JSON summary, and a multi-timeframe bias score.
  Indicators follow TradingView's own conventions.</p>

  <pre>pip install git+https://github.com/&lt;you&gt;/tvchart.git
tvchart AAPL -i 1d -p 1y --plan long --mtf -o aapl.html</pre>

  <div class="chart"><iframe src="demo/aapl_1d.html" title="AAPL chart"></iframe></div>
  <p class="muted">Live-generated AAPL daily chart. Scroll to zoom, drag to pan, use the
  toolbar for trendlines and rectangles.</p>

  <div class="row">
    <a href="demo/aapl_15m.html">AAPL 15m (light theme)</a>
    <a href="demo/nvda_1d.html">NVDA daily</a>
    <a href="demo/spy_1wk.html">SPY weekly, 5y</a>
  </div>

  <h2>What it replaces</h2>
  <table>
    <tr><th>Notebook approach</th><th>tvchart</th></tr>
    <tr><td>Simple-mean RSI</td><td>Wilder RMA, flat → 50</td></tr>
    <tr><td>Sample-stdev Bollinger</td><td>Population stdev, like TradingView</td></tr>
    <tr><td>Whole-history VWAP</td><td>Session VWAP intraday, anchored on daily</td></tr>
    <tr><td>Hard-coded stop/target</td><td>ATR + swing-structure plan</td></tr>
    <tr><td>Chart only</td><td>Chart + JSON + CSV + MTF bias</td></tr>
  </table>

  <h2>Limits</h2>
  <p class="muted">yfinance is an unofficial, delayed feed. No Pine Script, no saved
  drawings, no server-side alerts. Research tool, not an execution system.</p>
</div>
</body>
</html>
"""

(DOCS / "index.html").write_text(INDEX, encoding="utf-8")
print(f"  index.html written ({len(generated)} charts)")
