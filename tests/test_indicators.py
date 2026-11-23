import numpy as np
import pandas as pd
import pytest

from tvchart import indicators as ind


def frame(closes, highs=None, lows=None, volumes=None):
    n = len(closes)
    closes = np.asarray(closes, dtype=float)
    highs = np.asarray(highs if highs is not None else closes * 1.01, dtype=float)
    lows = np.asarray(lows if lows is not None else closes * 0.99, dtype=float)
    volumes = np.asarray(volumes if volumes is not None else np.full(n, 1000.0), dtype=float)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"open": closes, "high": highs, "low": lows, "close": closes, "volume": volumes}, index=idx)


def test_sma_matches_manual():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    out = ind.sma(s, 3)
    assert np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[4] == pytest.approx(4.0)


def test_ema_is_recursive_and_seeded():
    s = pd.Series([1, 2, 3, 4], dtype=float)
    out = ind.ema(s, 3)
    # alpha = 2/(3+1) = 0.5, seeded with the first value
    assert out.iloc[2] == pytest.approx(2.25)
    assert out.iloc[3] == pytest.approx(3.125)


def test_ema_of_constant_is_constant():
    s = pd.Series([7.0] * 10)
    assert ind.ema(s, 5).dropna().eq(7.0).all()


def test_rsi_monotonic_up_is_100():
    s = pd.Series(np.arange(1, 40, dtype=float))
    out = ind.rsi(s, 14).dropna()
    assert out.iloc[-1] == pytest.approx(100.0)


def test_rsi_monotonic_down_is_zero():
    s = pd.Series(np.arange(40, 1, -1, dtype=float))
    out = ind.rsi(s, 14).dropna()
    assert out.iloc[-1] == pytest.approx(0.0)


def test_rsi_flat_is_50():
    s = pd.Series([5.0] * 40)
    out = ind.rsi(s, 14).dropna()
    assert out.iloc[-1] == pytest.approx(50.0)


def test_rsi_bounded():
    rng = np.random.default_rng(0)
    s = pd.Series(100 + np.cumsum(rng.normal(0, 1, 300)))
    out = ind.rsi(s, 14).dropna()
    assert out.between(0, 100).all()


def test_bollinger_uses_population_stdev():
    rng = np.random.default_rng(1)
    s = pd.Series(100 + rng.normal(0, 2, 60))
    bb = ind.bollinger(s, 20, 2.0)
    window = s.iloc[-20:]
    expected_upper = window.mean() + 2 * window.std(ddof=0)
    assert bb["bb_upper"].iloc[-1] == pytest.approx(expected_upper, rel=1e-9)


def test_bollinger_constant_series_collapses():
    s = pd.Series([10.0] * 40)
    bb = ind.bollinger(s, 20, 2.0)
    assert bb["bb_upper"].iloc[-1] == pytest.approx(10.0)
    assert bb["bb_lower"].iloc[-1] == pytest.approx(10.0)


def test_macd_histogram_is_line_minus_signal():
    rng = np.random.default_rng(2)
    s = pd.Series(50 + np.cumsum(rng.normal(0, 1, 200)))
    m = ind.macd(s)
    diff = (m["macd"] - m["macd_signal"] - m["macd_hist"]).abs().max()
    assert diff < 1e-12


def test_atr_on_constant_range():
    closes = [10.0] * 40
    df = frame(closes, highs=[11.0] * 40, lows=[9.0] * 40)
    a = ind.atr(df, 14).dropna()
    assert a.iloc[-1] == pytest.approx(2.0)


def test_vwap_single_bar_is_typical_price():
    df = frame([10.0], highs=[12.0], lows=[9.0], volumes=[500])
    v = ind.vwap(df, anchor="start")
    assert v.iloc[0] == pytest.approx((12 + 9 + 10) / 3)


def test_vwap_session_resets_each_day():
    idx = pd.to_datetime(
        ["2024-01-02 09:30", "2024-01-02 10:30", "2024-01-03 09:30", "2024-01-03 10:30"]
    ).tz_localize("America/New_York")
    df = pd.DataFrame(
        {"open": [10, 20, 30, 40], "high": [10, 20, 30, 40], "low": [10, 20, 30, 40],
         "close": [10, 20, 30, 40], "volume": [1, 1, 1, 1]},
        index=idx,
    )
    v = ind.vwap(df, anchor="session")
    assert v.iloc[0] == pytest.approx(10)
    assert v.iloc[1] == pytest.approx(15)
    assert v.iloc[2] == pytest.approx(30)  # reset
    assert v.iloc[3] == pytest.approx(35)


def test_stochastic_bounded():
    rng = np.random.default_rng(3)
    closes = 100 + np.cumsum(rng.normal(0, 1, 120))
    df = frame(closes, highs=closes + 1, lows=closes - 1)
    st = ind.stochastic(df).dropna()
    assert st["stoch_k"].between(0, 100).all()
    assert st["stoch_d"].between(0, 100).all()


def test_adx_bounded():
    rng = np.random.default_rng(4)
    closes = 100 + np.cumsum(rng.normal(0, 1, 200))
    df = frame(closes, highs=closes + 1.5, lows=closes - 1.5)
    a = ind.adx(df, 14).dropna()
    assert a["adx"].between(0, 100).all()
    assert (a["plus_di"] >= 0).all()


def test_enrich_adds_expected_columns():
    rng = np.random.default_rng(5)
    closes = 100 + np.cumsum(rng.normal(0, 1, 260))
    df = frame(closes)
    out = ind.enrich(df)
    for col in ["ema_50", "ema_200", "bb_upper", "bb_lower", "rsi", "macd",
                "macd_signal", "macd_hist", "atr", "vwap", "stoch_k", "adx",
                "dc_upper", "rel_volume"]:
        assert col in out.columns, col
    assert len(out) == len(df)
