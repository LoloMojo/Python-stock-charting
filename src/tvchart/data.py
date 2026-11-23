"""Market data loading with caching and strict validation.

Wraps yfinance but normalises the output so the rest of the library never has
to deal with MultiIndex columns, missing sessions or empty frames.
"""

from __future__ import annotations

import hashlib
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import yfinance as yf

Interval = Literal["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"]

# yfinance limits: max lookback per interval when using `period`.
PERIOD_FOR_INTERVAL = {
    "1m": "7d",
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "1h": "2y",
    "1d": "max",
    "1wk": "max",
    "1mo": "max",
}

CACHE_DIR = Path.home() / ".cache" / "tvchart"
CACHE_TTL_SECONDS = 15 * 60


class DataError(RuntimeError):
    """Raised when no usable OHLCV data could be retrieved."""


@dataclass
class Bar:
    """Metadata about the loaded series."""

    ticker: str
    interval: str
    rows: int
    start: pd.Timestamp
    end: pd.Timestamp
    intraday: bool
    adjusted: bool


def _normalise(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise DataError(f"No data returned for {ticker!r} (delisted, wrong symbol or empty range).")

    if isinstance(df.columns, pd.MultiIndex):
        # yfinance returns (field, ticker) when a single ticker is requested.
        df = df.copy()
        df.columns = [str(c[0]) for c in df.columns]

    df = df.rename(columns=lambda c: str(c).strip().lower().replace(" ", "_"))
    rename = {
        "adj_close": "close",
        "adjclose": "close",
        "stock_splits": "splits",
        "capital_gains": "capital_gains",
    }
    df = df.rename(columns=rename)

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataError(f"{ticker}: missing columns {missing}; got {list(df.columns)}")

    df = df[required].copy()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0)

    # A bar where high < low is corrupt; drop it rather than silently poisoning
    # ATR / Bollinger calculations.
    bad = df["high"] < df["low"]
    if bad.any():
        df = df[~bad]

    if df.empty:
        raise DataError(f"{ticker}: every row was dropped as invalid.")

    return df


def _cache_path(ticker: str, interval: str, key: str) -> Path:
    digest = hashlib.sha1(f"{ticker}|{interval}|{key}".encode()).hexdigest()[:16]
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in ticker)
    return CACHE_DIR / f"{safe}_{interval}_{digest}.pkl"


def load(
    ticker: str,
    interval: str = "1d",
    period: str | None = None,
    start: str | None = None,
    end: str | None = None,
    *,
    auto_adjust: bool = True,
    prepost: bool = False,
    cache: bool = True,
    cache_ttl: int = CACHE_TTL_SECONDS,
) -> tuple[pd.DataFrame, Bar]:
    """Download OHLCV bars and return ``(dataframe, metadata)``.

    ``period`` wins over ``start``/``end`` when both are supplied, matching
    yfinance behaviour.  Intraday intervals get a sensible default period so a
    bare ``load("AAPL", "15m")`` does not silently return nothing.
    """
    ticker = ticker.strip().upper()
    interval = interval.lower()
    if interval not in PERIOD_FOR_INTERVAL:
        raise ValueError(f"Unsupported interval {interval!r}; use one of {sorted(PERIOD_FOR_INTERVAL)}")

    if period is None and start is None and end is None:
        period = PERIOD_FOR_INTERVAL[interval]
    if interval in ("1m", "5m", "15m", "30m") and period in (None, "max"):
        period = PERIOD_FOR_INTERVAL[interval]

    key = f"{period}|{start}|{end}|{auto_adjust}|{prepost}"
    path = _cache_path(ticker, interval, key)

    if cache and path.exists() and time.time() - path.stat().st_mtime < cache_ttl:
        try:
            with path.open("rb") as fh:
                df = pickle.load(fh)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return _finish(df, ticker, interval, auto_adjust)
        except Exception:
            pass  # corrupt cache entry -> fall through to a fresh download

    kwargs: dict = dict(
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        threads=False,
        prepost=prepost,
    )
    if period:
        kwargs["period"] = period
    else:
        kwargs["start"] = start
        if end:
            kwargs["end"] = end

    raw = yf.download(ticker, **kwargs)
    df = _normalise(raw, ticker)

    if cache:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as fh:
                pickle.dump(df, fh)
        except Exception:
            pass

    return _finish(df, ticker, interval, auto_adjust)


def _finish(df: pd.DataFrame, ticker: str, interval: str, adjusted: bool) -> tuple[pd.DataFrame, Bar]:
    intraday = interval.endswith("m") or interval.endswith("h")
    meta = Bar(
        ticker=ticker,
        interval=interval,
        rows=len(df),
        start=df.index[0],
        end=df.index[-1],
        intraday=intraday,
        adjusted=adjusted,
    )
    return df, meta


def dividends(ticker: str, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> pd.Series:
    """Dividend payments inside the window (empty series on failure)."""
    try:
        divs = yf.Ticker(ticker).dividends
        if divs is None or divs.empty:
            return pd.Series(dtype=float)
        idx = divs.index
        if getattr(idx, "tz", None) is not None:
            divs.index = idx.tz_localize(None)
        if start is not None:
            divs = divs[divs.index >= pd.Timestamp(start).tz_localize(None)]
        if end is not None:
            divs = divs[divs.index <= pd.Timestamp(end).tz_localize(None)]
        return divs
    except Exception:
        return pd.Series(dtype=float)


def earnings(ticker: str, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Earnings dates with EPS estimate/actual where available."""
    try:
        ed = yf.Ticker(ticker).get_earnings_dates(limit=24)
        if ed is None or ed.empty:
            return pd.DataFrame()
        idx = ed.index
        if getattr(idx, "tz", None) is not None:
            ed.index = idx.tz_localize(None)
        ed = ed.sort_index()
        if start is not None:
            ed = ed[ed.index >= pd.Timestamp(start).tz_localize(None)]
        if end is not None:
            ed = ed[ed.index <= pd.Timestamp(end).tz_localize(None)]
        keep = [c for c in ("EPS Estimate", "Reported EPS", "Surprise(%)") if c in ed.columns]
        return ed[keep]
    except Exception:
        return pd.DataFrame()
