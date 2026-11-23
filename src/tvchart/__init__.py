"""tvchart — TradingView-style technical analysis in Python.

Quick start::

    from tvchart import load, enrich, build_chart, save_html

    df, meta = load("AAPL", interval="1d", period="6mo")
    df = enrich(df)
    save_html(build_chart(df, "AAPL"), "aapl.html")
"""

from .data import Bar, DataError, dividends, earnings, load
from .indicators import (
    adx,
    atr,
    bollinger,
    ema,
    enrich,
    macd,
    rsi,
    sma,
    stochastic,
    vwap,
)
from .signals import mtf_alignment, summary, trade_plan, trend_state
from .chart import build_chart, save_html

__version__ = "0.1.0"

__all__ = [
    "load", "Bar", "DataError", "dividends", "earnings",
    "enrich", "ema", "sma", "rsi", "macd", "bollinger", "atr", "vwap", "stochastic", "adx",
    "trend_state", "summary", "trade_plan", "mtf_alignment",
    "build_chart", "save_html",
]
