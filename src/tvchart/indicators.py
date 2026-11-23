"""Technical indicators implemented with TradingView's exact conventions.

The notebook this project started from got several of these subtly wrong
(simple-average RSI, sample stdev Bollinger bands, whole-history VWAP).
Every function here documents the convention it follows so results line up
with what a TradingView chart shows.

All functions take a DataFrame with lowercase columns:
``open, high, low, close, volume`` and a DatetimeIndex.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "sma",
    "ema",
    "rma",
    "stdev",
    "rsi",
    "macd",
    "bollinger",
    "atr",
    "vwap",
    "stochastic",
    "adx",
    "donchian",
    "relative_volume",
    "enrich",
]


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential MA seeded with the first value (TradingView ``ta.ema``)."""
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's smoothing (TradingView ``ta.rma``), alpha = 1/length."""
    return series.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def stdev(series: pd.Series, length: int) -> pd.Series:
    """Population standard deviation — TradingView's ``ta.stdev`` default."""
    return series.rolling(length, min_periods=length).std(ddof=0)


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's RMA.

    Flat stretches (no gains and no losses) return 50 instead of NaN, which is
    what TradingView plots.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    out = out.where(avg_loss != 0, 100.0)
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return out


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD line, signal line and histogram.

    TradingView's histogram is ``macd - signal`` (not doubled, unlike some
    other platforms).
    """
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def bollinger(close: pd.Series, length: int = 20, mult: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands with population stdev (TradingView ``ta.bb``)."""
    mid = sma(close, length)
    sd = stdev(close, length)
    return pd.DataFrame(
        {"bb_mid": mid, "bb_upper": mid + mult * sd, "bb_lower": mid - mult * sd}
    )


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Average True Range (Wilder's RMA of true range)."""
    return rma(true_range(df), length)


def vwap(df: pd.DataFrame, anchor: str = "auto") -> pd.Series:
    """Volume-weighted average price.

    * ``session`` — resets each trading day. This is what TradingView's VWAP
      does, but on daily bars it degenerates to the typical price, so...
    * ``start`` — anchored from the first bar in the window (AVWAP).
    * ``auto`` — session for intraday, start for daily/weekly.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].replace(0, np.nan)

    intraday = isinstance(df.index, pd.DatetimeIndex) and (
        df.index.tz is not None or len(df.index) > 1 and df.index[0].hour != 0
    )
    if anchor == "auto":
        anchor = "session" if intraday else "start"

    if anchor == "session" and isinstance(df.index, pd.DatetimeIndex):
        session = df.index.tz_convert(None) if df.index.tz is not None else df.index
        session = session.normalize()
        cum_pv = (typical * vol).groupby(session).cumsum()
        cum_v = vol.groupby(session).cumsum()
        out = cum_pv / cum_v
        return out.fillna(typical)

    cum_pv = (typical * vol).cumsum()
    cum_v = vol.cumsum()
    return (cum_pv / cum_v).fillna(typical)


def stochastic(df: pd.DataFrame, k: int = 14, smooth_k: int = 3, d: int = 3) -> pd.DataFrame:
    """Stochastic oscillator — TradingView defaults 14/3/3."""
    lowest = df["low"].rolling(k, min_periods=k).min()
    highest = df["high"].rolling(k, min_periods=k).max()
    raw = 100 * (df["close"] - lowest) / (highest - lowest).replace(0, np.nan)
    k_line = sma(raw, smooth_k)
    d_line = sma(k_line, d)
    return pd.DataFrame({"stoch_k": k_line, "stoch_d": d_line})


def adx(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """ADX / +DI / -DI using Wilder's smoothing."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = rma(true_range(df), length)
    plus_di = 100 * rma(plus_dm, length) / tr.replace(0, np.nan)
    minus_di = 100 * rma(minus_dm, length) / tr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return pd.DataFrame({"adx": rma(dx, length), "plus_di": plus_di, "minus_di": minus_di})


def donchian(df: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    upper = df["high"].rolling(length, min_periods=length).max()
    lower = df["low"].rolling(length, min_periods=length).min()
    return pd.DataFrame({"dc_upper": upper, "dc_lower": lower, "dc_mid": (upper + lower) / 2})


def relative_volume(df: pd.DataFrame, length: int = 20) -> pd.Series:
    return df["volume"] / sma(df["volume"], length).replace(0, np.nan)


def enrich(
    df: pd.DataFrame,
    *,
    ema_fast: int = 50,
    ema_slow: int = 200,
    bb_length: int = 20,
    bb_mult: float = 2.0,
    rsi_length: int = 14,
    atr_length: int = 14,
    vwap_anchor: str = "auto",
) -> pd.DataFrame:
    """Return a copy of ``df`` with every indicator column appended."""
    out = df.copy()
    close = out["close"]

    out[f"ema_{ema_fast}"] = ema(close, ema_fast)
    out[f"ema_{ema_slow}"] = ema(close, ema_slow)
    out["sma_20"] = sma(close, 20)
    out = out.join(bollinger(close, bb_length, bb_mult))
    out["rsi"] = rsi(close, rsi_length)
    out = out.join(macd(close))
    out["atr"] = atr(out, atr_length)
    out["atr_pct"] = out["atr"] / close * 100
    out["vwap"] = vwap(out, vwap_anchor)
    out = out.join(stochastic(out))
    out = out.join(adx(out, 14))
    out = out.join(donchian(out, 20))
    out["rel_volume"] = relative_volume(out, 20)
    out["vol_sma_20"] = sma(out["volume"], 20)
    return out
