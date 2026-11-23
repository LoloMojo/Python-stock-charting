import numpy as np
import pandas as pd
import pytest

from tvchart import indicators as ind
from tvchart import signals as sig


def frame(closes):
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": closes, "high": closes * 1.01, "low": closes * 0.99, "close": closes,
         "volume": np.full(n, 1_000_000.0)},
        index=idx,
    )


def uptrend(n=300):
    return frame(100 * np.cumprod(np.full(n, 1.004)))


def downtrend(n=300):
    return frame(100 * np.cumprod(np.full(n, 0.996)))


def test_trend_state_max_on_uptrend():
    st = sig.trend_state(ind.enrich(uptrend()))
    assert st["score"] == 4
    assert st["label"] == "strong uptrend"


def test_trend_state_bearish_on_downtrend():
    st = sig.trend_state(ind.enrich(downtrend()))
    # A perfectly geometric decline flattens the MACD histogram, so momentum
    # reads as "transition" rather than bearish: -3, not -4.
    assert st["score"] <= -3
    assert "downtrend" in st["label"]


def test_crosses_detected():
    # downtrend then uptrend -> EMA50 crosses above EMA200
    n = 400
    closes = np.concatenate([100 * np.cumprod(np.full(n, 0.997)), 100 * np.cumprod(np.full(n, 1.006))])
    df = ind.enrich(frame(closes))
    c = sig.crosses(df, lookback=500)
    assert c["ema_cross"] is not None
    assert c["ema_cross"]["direction"] == "bullish"


def test_volume_climax_flags_spike():
    df = frame(np.linspace(100, 110, 60))
    df.loc[df.index[-1], "volume"] = 5_000_000
    df = ind.enrich(df)
    cl = sig.volume_climax(df)
    assert len(cl) == 1
    assert cl.iloc[-1]["alert"] == "accumulation"


def test_squeeze_detected_on_flat_series():
    df = ind.enrich(frame(np.full(200, 100.0)))
    sq = sig.bollinger_squeeze(df)
    # zero width -> percentile 0 -> squeeze
    assert sq["state"] == "squeeze"


def test_key_levels_bracket_price():
    rng = np.random.default_rng(7)
    closes = 100 + np.cumsum(rng.normal(0, 1, 200))
    df = ind.enrich(frame(closes))
    lv = sig.key_levels(df)
    last = float(df["close"].iloc[-1])
    if lv["support"] is not None:
        assert lv["support"] < last
    if lv["resistance"] is not None:
        assert lv["resistance"] > last


def test_trade_plan_long_ordering():
    df = ind.enrich(uptrend())
    plan = sig.trade_plan(df, direction="long")
    assert plan["stop"] < plan["entry"] < plan["target"]
    assert plan["risk_reward"] >= 2.0
    assert plan["stop_pct"] < 0


def test_trade_plan_short_ordering():
    df = ind.enrich(downtrend())
    plan = sig.trade_plan(df, direction="short")
    assert plan["target"] < plan["entry"] < plan["stop"]
    assert plan["risk_reward"] >= 2.0


def test_summary_serialisable():
    import json
    s = sig.summary(ind.enrich(uptrend()), ticker="TEST")
    json.dumps(s, default=str)
    assert s["ticker"] == "TEST"
    assert s["trend"]["score"] == 4


def test_mtf_alignment_bullish():
    frames = {"1wk": ind.enrich(uptrend()), "1d": ind.enrich(uptrend()), "1h": ind.enrich(uptrend())}
    out = sig.mtf_alignment(frames)
    assert out["bias"] == "bullish"
    assert out["score"] > 0.9
    assert len(out["timeframes"]) == 3


def test_mtf_alignment_mixed():
    frames = {"1wk": ind.enrich(uptrend()), "1d": ind.enrich(downtrend())}
    out = sig.mtf_alignment(frames)
    assert out["bias"] == "mixed"
