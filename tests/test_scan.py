"""Tests for the batch scan layer (offline: data loading is monkeypatched)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tvchart import scan as scan_mod


def _frame(n: int = 260, drift: float = 0.4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    close = 100 + np.cumsum(rng.normal(drift, 1.2, n))
    close = np.maximum(close, 5.0)
    high = close + rng.uniform(0.2, 1.5, n)
    low = close - rng.uniform(0.2, 1.5, n)
    open_ = close + rng.normal(0, 0.5, n)
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx
    )


@pytest.fixture()
def fake_load(monkeypatch):
    def _load(ticker, interval="1d", period=None, **kw):
        seed = sum(ord(c) for c in ticker) % 100
        drift = 0.6 if ticker.startswith("UP") else -0.6
        return _frame(drift=drift, seed=seed), type("Bar", (), {"ticker": ticker, "interval": interval,
                                                               "rows": 260, "start": "2024-01-01",
                                                               "end": "2024-12-31"})()

    monkeypatch.setattr(scan_mod._data, "load", _load)
    return _load


def test_setup_score_prefers_uptrend_over_downtrend():
    up = {"trend_score": 4, "mtf_score": 0.8, "rsi": 62, "adx": 30, "bb_percentile": 10, "extension_atr": 0.5}
    down = {"trend_score": -4, "mtf_score": -0.8, "rsi": 38, "adx": 30, "bb_percentile": 10, "extension_atr": 0.5}
    s_up, tags_up = scan_mod.setup_score(up)
    s_down, tags_down = scan_mod.setup_score(down)
    assert s_up > 0 > s_down
    assert "mtf-aligned-up" in tags_up
    assert "mtf-aligned-down" in tags_down
    assert "squeeze" in tags_up


def test_setup_score_penalises_extension():
    base = {"trend_score": 3, "mtf_score": 0.5, "rsi": 60, "adx": 28, "bb_percentile": 40, "extension_atr": 0.5}
    stretched = {**base, "extension_atr": 6.0}
    assert scan_mod.setup_score(base)[0] > scan_mod.setup_score(stretched)[0]
    assert "extended" in scan_mod.setup_score(stretched)[1]


def test_parse_watchlist_file_and_inline(tmp_path: Path):
    p = tmp_path / "w.txt"
    p.write_text("aapl\n, nvda\n\nspy\n")
    assert scan_mod.parse_watchlist(str(p)) == ["AAPL", "NVDA", "SPY"]
    assert scan_mod.parse_watchlist("aapl, msft") == ["AAPL", "MSFT"]


def test_scan_appends_history_and_computes_deltas(fake_load, tmp_path: Path):
    hist = tmp_path / "h.jsonl"
    csv_out = tmp_path / "run.csv"

    first = scan_mod.scan_watchlist(["UPX", "DOWNX"], interval="1d", history_path=hist,
                                    csv_path=csv_out, delay=0)
    assert len(first) == 2
    assert all("error" not in r for r in first)
    assert first[0]["ticker"] == "UPX"  # uptrend ranks first
    assert first[0]["setup_score"] > first[1]["setup_score"]
    assert hist.read_text().count("\n") == 2
    assert csv_out.exists()

    second = scan_mod.scan_watchlist(["UPX"], interval="1d", history_path=hist, delay=0)
    assert len(second) == 1
    row = second[0]
    assert row["prev_run_utc"] is not None
    assert row["days_since_prev"] is not None
    assert "trend_delta" in row and "rsi_delta" in row
    assert hist.read_text().count("\n") == 3


def test_scan_records_errors_without_aborting(fake_load, monkeypatch, tmp_path: Path):
    real = scan_mod._data.load

    def flaky(ticker, *a, **k):
        if ticker == "BAD":
            raise RuntimeError("no data")
        return real(ticker, *a, **k)

    monkeypatch.setattr(scan_mod._data, "load", flaky)
    rows = scan_mod.scan_watchlist(["UPX", "BAD"], history_path=tmp_path / "h.jsonl", delay=0)
    assert len(rows) == 2
    bad = [r for r in rows if r["ticker"] == "BAD"][0]
    assert "no data" in bad["error"]
    assert rows[-1]["ticker"] == "BAD"  # errors sort last


def test_history_skips_malformed_lines(tmp_path: Path):
    p = tmp_path / "h.jsonl"
    p.write_text('{"ticker": "AAPL", "run_utc": "2026-01-01T00:00:00+00:00"}\nnot json\n')
    rows = scan_mod.load_history(p)
    assert len(rows) == 1
    assert rows[0]["ticker"] == "AAPL"
