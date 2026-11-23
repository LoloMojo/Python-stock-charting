"""Batch scan: a watchlist in, an append-only history plus a ranked snapshot out.

The point is *change over time*.  Every run appends one row per ticker to a
JSONL file, so later runs can compute deltas against the previous observation of
the same ticker/timeframe.  A flat CSV snapshot of the current run is written
alongside for spreadsheets and quick eyeballing.

Design rules
------------
* Append-only history: never rewrite or delete past rows.  A bad run is visible
  in the data, not silently corrected.
* One row per (ticker, interval) per run, keyed by ``run_id``.
* Network failures are recorded as ``error`` rows and never abort the batch.
* ``setup_score`` is a transparent, documented composite — not a black box.
"""

from __future__ import annotations

import csv
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import data as _data
from . import indicators as _ind
from . import signals as _signals

# Weights for setup_score.  Kept module-level so they are auditable and tunable.
W_TREND = 2.0
W_MTF = 1.5
W_RSI = 1.0
W_ADX = 0.5
W_SQUEEZE = 0.5
W_EXTENSION = 0.5

FIELDS = [
    "run_id", "run_utc", "ticker", "interval", "last_bar", "last_close",
    "change_pct", "trend_score", "trend_label", "rsi", "adx", "atr_pct",
    "bb_bandwidth", "bb_percentile", "squeeze_state", "rel_volume",
    "support", "resistance", "dist_support_pct", "dist_resistance_pct",
    "ema_cross", "macd_cross", "mtf_bias", "mtf_score",
    "plan_direction", "plan_entry", "plan_stop", "plan_target", "plan_rr",
    "setup_score", "tags", "error",
    "prev_run_utc", "days_since_prev", "trend_delta", "rsi_delta",
    "close_delta_pct",
]


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def setup_score(row: dict) -> tuple[float, list[str]]:
    """Composite conviction score plus human-readable tags.

    Components (all normalised to roughly -1..1 before weighting):
      trend   : trend_state score / 4            weight 2.0
      mtf     : multi-timeframe alignment score   weight 1.5
      rsi     : (rsi - 50) / 50                   weight 1.0
      adx     : min(adx/40, 1) signed by trend    weight 0.5
      squeeze : (50 - bandwidth percentile)/50    weight 0.5
      ext     : penalty beyond 2 ATR from SMA20   weight 0.5

    A high score means "directional trend, timeframes agree, not stretched".
    It is a ranking device, not a probability.
    """
    tags: list[str] = []
    trend = row.get("trend_score") or 0
    score = W_TREND * _clip(trend / 4.0)

    mtf = row.get("mtf_score")
    if mtf is not None:
        score += W_MTF * _clip(float(mtf))
        if trend > 0 and mtf > 0.25:
            tags.append("mtf-aligned-up")
        elif trend < 0 and mtf < -0.25:
            tags.append("mtf-aligned-down")

    rsi = row.get("rsi")
    if rsi is not None:
        score += W_RSI * _clip((float(rsi) - 50.0) / 50.0)
        if rsi >= 70:
            tags.append("rsi-hot")
        elif rsi <= 30:
            tags.append("rsi-cold")

    adx = row.get("adx")
    if adx is not None:
        score += W_ADX * min(float(adx) / 40.0, 1.0) * (1 if trend >= 0 else -1)
        if adx >= 25:
            tags.append("trending")

    pct = row.get("bb_percentile")
    if pct is not None:
        score += W_SQUEEZE * _clip((50.0 - float(pct)) / 50.0)
        if pct <= 20:
            tags.append("squeeze")

    ext = row.get("extension_atr")
    if ext is not None:
        score -= W_EXTENSION * min(max(abs(float(ext)) - 2.0, 0.0), 2.0)
        if abs(float(ext)) > 2.0:
            tags.append("extended")

    rv = row.get("rel_volume")
    if rv is not None and rv >= 2.0:
        tags.append("vol-climax")

    if trend >= 3:
        tags.append("trend-up")
    elif trend <= -3:
        tags.append("trend-down")

    return round(score, 2), tags


def _row_from_frame(df, ticker: str, interval: str, mtf: dict | None, plan: str | None) -> dict:
    s = _signals.summary(df, ticker=ticker)
    last = df.iloc[-1]
    close = float(last["close"])
    levels = s.get("levels") or {}
    squeeze = s.get("squeeze") or {}
    crosses = s.get("crosses") or {}

    support = levels.get("support")
    resistance = levels.get("resistance")
    sma20 = last.get("sma_20")
    atr = last.get("atr")

    row = {
        "ticker": ticker,
        "interval": interval,
        "last_bar": s.get("last_bar"),
        "last_close": s.get("last_close"),
        "change_pct": s.get("change_pct"),
        "trend_score": (s.get("trend") or {}).get("score"),
        "trend_label": (s.get("trend") or {}).get("label"),
        "rsi": s.get("rsi"),
        "adx": s.get("adx"),
        "atr_pct": s.get("atr_pct"),
        "bb_bandwidth": squeeze.get("bandwidth"),
        "bb_percentile": squeeze.get("percentile"),
        "squeeze_state": squeeze.get("state"),
        "rel_volume": s.get("rel_volume"),
        "support": support,
        "resistance": resistance,
        "ema_cross": (crosses.get("ema_cross") or {}).get("direction"),
        "macd_cross": (crosses.get("macd_cross") or {}).get("direction"),
        "mtf_bias": (mtf or {}).get("bias"),
        "mtf_score": (mtf or {}).get("score"),
        "extension_atr": (
            round((close - float(sma20)) / float(atr), 2)
            if sma20 is not None and atr not in (None, 0) and not _isnan(atr)
            else None
        ),
    }

    if support and close:
        row["dist_support_pct"] = round((close / float(support) - 1) * 100, 2)
    if resistance and close:
        row["dist_resistance_pct"] = round((float(resistance) / close - 1) * 100, 2)

    if plan:
        p = _signals.trade_plan(df, direction=plan)
        row.update(
            plan_direction=p.get("direction"),
            plan_entry=p.get("entry"),
            plan_stop=p.get("stop"),
            plan_target=p.get("target"),
            plan_rr=p.get("risk_reward"),
        )

    score, tags = setup_score(row)
    row["setup_score"] = score
    row["tags"] = ",".join(tags)
    return row


def _isnan(x) -> bool:
    try:
        return x != x
    except Exception:
        return False


def load_history(path: Path) -> list[dict]:
    """Read the JSONL history, skipping malformed lines."""
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _previous(history: list[dict], ticker: str, interval: str) -> dict | None:
    for row in reversed(history):
        if row.get("ticker") == ticker and row.get("interval") == interval and not row.get("error"):
            return row
    return None


def _attach_deltas(row: dict, prev: dict | None) -> None:
    if not prev:
        return
    row["prev_run_utc"] = prev.get("run_utc")
    try:
        now = datetime.fromisoformat(row["run_utc"])
        then = datetime.fromisoformat(prev["run_utc"])
        row["days_since_prev"] = round((now - then).total_seconds() / 86400, 2)
    except Exception:
        row["days_since_prev"] = None
    for key in ("trend_score", "rsi", "last_close"):
        a, b = row.get(key), prev.get(key)
        if a is not None and b not in (None, 0):
            if key == "last_close":
                row["close_delta_pct"] = round((float(a) / float(b) - 1) * 100, 2)
            else:
                row[f"{key.split('_')[0]}_delta"] = round(float(a) - float(b), 2)


def scan_watchlist(
    tickers: list[str],
    *,
    interval: str = "1d",
    period: str | None = None,
    mtf: bool = False,
    plan: str | None = None,
    history_path: str | Path = "scan_history.jsonl",
    csv_path: str | Path | None = None,
    delay: float = 0.4,
    top: int | None = None,
) -> list[dict]:
    """Scan a watchlist and append the results to an append-only history.

    Returns the rows for this run, sorted by ``setup_score`` descending.
    """
    history_path = Path(history_path)
    history = load_history(history_path)
    run_id = uuid.uuid4().hex[:12]
    run_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows: list[dict] = []

    for i, ticker in enumerate(tickers):
        ticker = ticker.strip().upper()
        if not ticker:
            continue
        row: dict = {"run_id": run_id, "run_utc": run_utc, "ticker": ticker, "interval": interval}
        try:
            df, _meta = _data.load(ticker, interval=interval, period=period)
            df = _ind.enrich(df)
            mtf_block = None
            if mtf:
                mtf_block = _scan_mtf(ticker)
            row.update(_row_from_frame(df, ticker, interval, mtf_block, plan))
            _attach_deltas(row, _previous(history, ticker, interval))
        except Exception as exc:  # a single bad ticker must not kill the batch
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if delay and i < len(tickers) - 1:
            time.sleep(delay)

    rows.sort(key=lambda r: (r.get("error") is not None, -(r.get("setup_score") or -99)))
    if top:
        rows = rows[:top]

    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")

    if csv_path:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        cols = [c for c in FIELDS if c != "extension_atr"]
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in cols})

    return rows


def _scan_mtf(ticker: str) -> dict | None:
    """Score weekly/daily/4h/15m alignment, tolerating missing timeframes."""
    frames = {}
    for label, interval, period in (
        ("1wk", "1wk", "2y"), ("1d", "1d", "1y"), ("1h", "1h", "3mo"), ("15m", "15m", "1mo"),
    ):
        try:
            df, _ = _data.load(ticker, interval=interval, period=period)
            frames[label] = _ind.enrich(df)
        except Exception:
            continue
    if not frames:
        return None
    return _signals.mtf_alignment(frames)


def parse_watchlist(spec: str) -> list[str]:
    """Accept a file path or a comma/space separated list of tickers."""
    p = Path(spec)
    if p.exists() and p.is_file():
        text = p.read_text(encoding="utf-8")
        return [t.strip().upper() for t in text.replace(",", "\n").split() if t.strip()]
    return [t.strip().upper() for t in spec.replace(",", " ").split() if t.strip()]
