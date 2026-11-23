"""Command-line interface: ``tvchart AAPL -i 1d -p 6mo -o chart.html``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from . import chart as _chart
from . import data as _data
from . import indicators as _ind
from . import signals as _signals

MTF_PLAN = [("1wk", "2y", "1wk"), ("1d", "1y", "1d"), ("1h", "3mo", "1h"), ("15m", "1mo", "15m")]


def _print_summary(s: dict, file=None) -> None:
    p = lambda *a, **k: print(*a, **{**k, "file": file or sys.stdout})
    tr = s["trend"]
    p(f"\n{s['ticker']}  last {s['last_close']:,.2f}  ({s['change_pct']:+.2f}%)  [{s['last_bar']}]")
    p(f"  trend      : {tr['label']} ({tr['score']:+d}/4)")
    for r in tr["reasons"]:
        p(f"               - {r}")
    if s.get("rsi") is not None:
        p(f"  RSI        : {s['rsi']:.1f}")
    if s.get("adx") is not None:
        p(f"  ADX        : {s['adx']:.1f}")
    if s.get("atr_pct") is not None:
        p(f"  ATR        : {s['atr_pct']:.2f}% of price")
    lv = s["levels"]
    p(f"  support    : {lv.get('support')}")
    p(f"  resistance : {lv.get('resistance')}")
    sq = s.get("squeeze") or {}
    if sq:
        p(f"  BB width   : {sq['bandwidth']}% (pct {sq['percentile']} → {sq['state']})")
    ec = s["crosses"].get("ema_cross")
    if ec:
        p(f"  EMA cross  : {ec['direction']} {ec['bars_ago']} bars ago ({ec['date']})")
    mc = s["crosses"].get("macd_cross")
    if mc:
        p(f"  MACD cross : {mc['direction']} {mc['bars_ago']} bars ago ({mc['date']})")


def _mtf(ticker: str, theme: str) -> dict:
    frames = {}
    for label, period, interval in MTF_PLAN:
        try:
            df, _ = _data.load(ticker, interval=interval, period=period)
            frames[label] = _ind.enrich(df)
        except Exception as exc:  # a missing timeframe should not kill the report
            print(f"  ! {label} unavailable: {exc}", file=sys.stderr)
    if not frames:
        return {"bias": "no data", "score": None, "timeframes": []}
    return _signals.mtf_alignment(frames)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tvchart", description="TradingView-style technical charts for any ticker.")
    p.add_argument("ticker", help="Symbol, e.g. AAPL, NVDA, SPY")
    p.add_argument("-i", "--interval", default="1d", help="1m 5m 15m 30m 1h 1d 1wk 1mo (default 1d)")
    p.add_argument("-p", "--period", default=None, help="Lookback: 1mo 3mo 6mo 1y 2y 5y max")
    p.add_argument("--start", default=None, help="Start date YYYY-MM-DD (overrides --period)")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    p.add_argument("-o", "--out", default=None, help="HTML output path (default ./<TICKER>_<interval>.html)")
    p.add_argument(
        "--json",
        dest="json_out",
        default=None,
        metavar="PATH",
        help="Write the summary as JSON to PATH; use '-' for stdout",
    )
    p.add_argument("--csv", dest="csv_out", default=None, help="Write the enriched data as CSV")
    p.add_argument("--theme", default="dark", choices=sorted(_chart.THEMES))
    p.add_argument("--plan", default=None, choices=["long", "short"], help="Overlay an ATR-based trade plan")
    p.add_argument("--mtf", action="store_true", help="Also score weekly/daily/4h/15m alignment")
    p.add_argument("--no-events", action="store_true", help="Skip dividends/earnings markers")
    p.add_argument("--height", type=int, default=900)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--version", action="version", version=f"tvchart {__version__}")
    args = p.parse_args(argv)

    try:
        df, meta = _data.load(
            args.ticker, interval=args.interval, period=args.period,
            start=args.start, end=args.end, cache=not args.no_cache,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    df = _ind.enrich(df)
    df.attrs["interval"] = args.interval

    plan = _signals.trade_plan(df, direction=args.plan) if args.plan else None
    summary = _signals.summary(df, ticker=meta.ticker)
    summary["interval"] = args.interval
    summary["rows"] = meta.rows
    summary["range"] = [str(meta.start), str(meta.end)]
    if plan:
        summary["plan"] = plan
    if args.mtf:
        summary["mtf"] = _mtf(meta.ticker, args.theme)

    # Everything human-readable goes to stderr when the JSON is streamed to
    # stdout, so `tvchart ... --json - | jq` gets clean JSON.
    to_stdout = args.json_out != "-"
    human = sys.stdout if to_stdout else sys.stderr

    _print_summary(summary, file=human)
    if args.mtf and summary.get("mtf", {}).get("timeframes"):
        print("\n  multi-timeframe alignment:", file=human)
        for row in summary["mtf"]["timeframes"]:
            print(f"    {row['timeframe']:>4}  {row['label']:<18} score {row['score']:+d}  RSI {row['rsi']}", file=human)
        print(f"    bias: {summary['mtf']['bias']}  ({summary['mtf']['score']:+.2f})", file=human)

    fig = _chart.build_chart(
        df, meta.ticker, theme=args.theme, height=args.height, plan=plan,
        show_events=not args.no_events,
    )
    out = Path(args.out) if args.out else Path(f"{meta.ticker}_{args.interval}.html")
    _chart.save_html(fig, out, title=f"{meta.ticker} {meta.interval}")
    print(f"\n  chart  -> {out.resolve()}", file=human)

    if args.json_out:
        blob = json.dumps(summary, indent=2, default=str)
        if args.json_out == "-":
            sys.stdout.write(blob + "\n")
        else:
            Path(args.json_out).write_text(blob)
            print(f"  json   -> {Path(args.json_out).resolve()}", file=human)
    if args.csv_out:
        df.to_csv(args.csv_out)
        print(f"  csv    -> {Path(args.csv_out).resolve()}", file=human)

    return 0


def scan_main(argv: list[str] | None = None) -> int:
    """``tvchart-scan`` — batch a watchlist into an append-only history."""
    from . import scan as _scan

    p = argparse.ArgumentParser(
        prog="tvchart-scan",
        description="Scan a watchlist, append results to a JSONL history, print a ranked table.",
    )
    p.add_argument("watchlist", help="Path to a watchlist file, or a comma-separated ticker list")
    p.add_argument("-i", "--interval", default="1d", help="1m 5m 15m 30m 1h 1d 1wk 1mo (default 1d)")
    p.add_argument("-p", "--period", default=None, help="Lookback: 1mo 3mo 6mo 1y 2y 5y max")
    p.add_argument("--mtf", action="store_true", help="Also score weekly/daily/1h/15m alignment")
    p.add_argument("--plan", default=None, choices=["long", "short"], help="Attach an ATR trade plan")
    p.add_argument("--history", default="scan_history.jsonl", help="Append-only JSONL history path")
    p.add_argument("--csv", default=None, help="Write the current run as a CSV snapshot")
    p.add_argument("--delay", type=float, default=0.4, help="Seconds between tickers (default 0.4)")
    p.add_argument("--top", type=int, default=None, help="Only keep the N highest-scoring rows")
    p.add_argument("--json", dest="json_out", default=None, metavar="PATH",
                   help="Write this run's rows as JSON; '-' for stdout")
    p.add_argument("--quiet", action="store_true", help="Suppress the table")
    args = p.parse_args(argv)

    tickers = _scan.parse_watchlist(args.watchlist)
    if not tickers:
        print("error: empty watchlist", file=sys.stderr)
        return 2

    rows = _scan.scan_watchlist(
        tickers,
        interval=args.interval,
        period=args.period,
        mtf=args.mtf,
        plan=args.plan,
        history_path=args.history,
        csv_path=args.csv,
        delay=args.delay,
        top=args.top,
    )

    human = sys.stdout if args.json_out != "-" else sys.stderr
    if not args.quiet:
        ok = [r for r in rows if not r.get("error")]
        bad = [r for r in rows if r.get("error")]
        print(f"\n  {len(ok)} scanned  ·  {len(bad)} failed  ·  interval {args.interval}", file=human)
        print(
            f"  {'ticker':<7}{'score':>7}{'trend':>7}{'rsi':>6}{'adx':>6}{'atr%':>7}"
            f"{'bw%':>7}{'mtf':>7}  tags",
            file=human,
        )
        for r in ok:
            print(
                f"  {r['ticker']:<7}{r.get('setup_score', 0):>7.2f}"
                f"{r.get('trend_score') or 0:>7}{r.get('rsi') or 0:>6.1f}"
                f"{r.get('adx') or 0:>6.1f}{r.get('atr_pct') or 0:>7.2f}"
                f"{r.get('bb_bandwidth') or 0:>7.2f}{r.get('mtf_score') or 0:>7.2f}"
                f"  {r.get('tags', '')}",
                file=human,
            )
        for r in bad:
            print(f"  {r['ticker']:<7} FAILED  {r['error']}", file=human)
        print(f"\n  history -> {Path(args.history).resolve()}", file=human)
        if args.csv:
            print(f"  csv     -> {Path(args.csv).resolve()}", file=human)

    if args.json_out:
        blob = json.dumps(rows, indent=2, default=str)
        if args.json_out == "-":
            sys.stdout.write(blob + "\n")
        else:
            Path(args.json_out).write_text(blob)
            print(f"  json    -> {Path(args.json_out).resolve()}", file=human)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
