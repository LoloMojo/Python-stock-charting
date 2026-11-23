import numpy as np
import pandas as pd

from tvchart import indicators as ind
from tvchart import signals as sig
from tvchart.chart import build_chart, save_html


def frame(n=300):
    rng = np.random.default_rng(11)
    closes = 100 + np.cumsum(rng.normal(0.1, 1.2, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": closes + rng.normal(0, 0.4, n), "high": closes + 1.2, "low": closes - 1.2,
         "close": closes, "volume": rng.integers(500_000, 3_000_000, n).astype(float)},
        index=idx,
    )


def test_chart_builds_three_panels():
    df = ind.enrich(frame())
    fig = build_chart(df, "TEST", show_events=False)
    assert len(fig.data) >= 8
    names = [t.name for t in fig.data]
    assert "RSI" in names and "MACD" in names and "Volume" in names
    assert fig.layout.height == 900


def test_chart_with_plan_and_theme():
    df = ind.enrich(frame())
    plan = sig.trade_plan(df, "long")
    fig = build_chart(df, "TEST", plan=plan, theme="light", show_events=False)
    assert fig.layout.paper_bgcolor == "#ffffff"
    assert any(t.name == "Long target" for t in fig.data)


def test_save_html_is_self_contained(tmp_path):
    df = ind.enrich(frame())
    fig = build_chart(df, "TEST", show_events=False)
    path = save_html(fig, tmp_path / "chart.html")
    text = path.read_text()
    assert path.exists() and path.stat().st_size > 100_000
    assert "plotly" in text.lower()
    assert "TEST" in text
