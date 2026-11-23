"""Interactive Plotly chart builder with a TradingView-style layout.

Produces a three-panel chart (price + volume, RSI, MACD) with optional
corporate-event markers, a trade-plan overlay and a summary box.  The output
is a standalone HTML file that works offline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import data as _data
from . import signals as _signals

THEMES = {
    "dark": {
        "paper": "#131722",
        "plot": "#131722",
        "grid": "#1f2430",
        "text": "#d1d4dc",
        "muted": "#787b86",
        "up": "#26a69a",
        "down": "#ef5350",
        "ema_fast": "#f0b90b",
        "ema_slow": "#2962ff",
        "vwap": "#ff9800",
        "bb": "#5d606b",
        "rsi": "#b39ddb",
        "macd": "#42a5f5",
        "signal": "#ff9800",
        "box": "rgba(30,34,45,0.92)",
        "box_border": "#363a45",
    },
    "light": {
        "paper": "#ffffff",
        "plot": "#ffffff",
        "grid": "#e6e8ec",
        "text": "#131722",
        "muted": "#787b86",
        "up": "#089981",
        "down": "#f23645",
        "ema_fast": "#b8860b",
        "ema_slow": "#2962ff",
        "vwap": "#e65100",
        "bb": "#9aa0a6",
        "rsi": "#7e57c2",
        "macd": "#1976d2",
        "signal": "#ef6c00",
        "box": "rgba(255,255,255,0.95)",
        "box_border": "#d0d4dd",
    },
}


def _rangebreaks(df: pd.DataFrame, intraday: bool) -> list[dict]:
    breaks: list[dict] = [dict(bounds=["sat", "mon"])]
    if intraday:
        # Hide 16:00 -> 09:30 and anything outside a regular US session.
        breaks.append(dict(bounds=[16, 9.5], pattern="hour"))
    return breaks


def build_chart(
    df: pd.DataFrame,
    ticker: str,
    *,
    theme: str = "dark",
    height: int = 900,
    plan: dict | None = None,
    show_events: bool = True,
    rangeslider: bool = False,
    title: str | None = None,
) -> go.Figure:
    t = THEMES.get(theme, THEMES["dark"])
    intraday = isinstance(df.index, pd.DatetimeIndex) and (
        df.index.tz is not None or (len(df.index) > 1 and df.index[0].hour != 0)
    )

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.62, 0.19, 0.19],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]],
    )

    x = df.index

    # ---- panel 1: candles, moving averages, VWAP, bands -------------------
    fig.add_trace(
        go.Candlestick(
            x=x, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name=ticker,
            increasing=dict(line=dict(color=t["up"], width=1), fillcolor=t["up"]),
            decreasing=dict(line=dict(color=t["down"], width=1), fillcolor=t["down"]),
            hovertext=[
                f"O {o:,.2f}  H {h:,.2f}<br>L {l:,.2f}  C {c:,.2f}"
                for o, h, l, c in zip(df["open"], df["high"], df["low"], df["close"])
            ],
            hoverinfo="x+text",
        ),
        row=1, col=1,
    )

    ema_cols = [c for c in df.columns if c.startswith("ema_")]
    ema_cols.sort(key=lambda c: int(c.split("_")[1]))
    for i, col in enumerate(ema_cols):
        length = col.split("_")[1]
        color = t["ema_fast"] if i == 0 else t["ema_slow"]
        fig.add_trace(
            go.Scatter(
                x=x, y=df[col], name=f"EMA {length}", mode="lines",
                line=dict(color=color, width=1.4),
                hovertemplate=f"EMA {length}: %{{y:,.2f}}<extra></extra>",
            ),
            row=1, col=1,
        )

    if "bb_upper" in df:
        fig.add_trace(
            go.Scatter(x=x, y=df["bb_upper"], name="BB upper", mode="lines",
                       line=dict(color=t["bb"], width=1, dash="dot"),
                       hovertemplate="BB upper: %{y:,.2f}<extra></extra>"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=x, y=df["bb_lower"], name="BB lower", mode="lines",
                       line=dict(color=t["bb"], width=1, dash="dot"),
                       fill="tonexty", fillcolor="rgba(120,123,134,0.035)",
                       hovertemplate="BB lower: %{y:,.2f}<extra></extra>"),
            row=1, col=1,
        )

    if "vwap" in df:
        fig.add_trace(
            go.Scatter(x=x, y=df["vwap"], name="VWAP", mode="lines",
                       line=dict(color=t["vwap"], width=1.6),
                       hovertemplate="VWAP: %{y:,.2f}<extra></extra>"),
            row=1, col=1,
        )

    # volume on the secondary axis, coloured by candle direction
    vol_colors = [t["up"] if c >= o else t["down"] for o, c in zip(df["open"], df["close"])]
    fig.add_trace(
        go.Bar(x=x, y=df["volume"], name="Volume", marker_color=vol_colors,
               marker_line_width=0, opacity=0.35,
               hovertemplate="Vol: %{y:,.0f}<extra></extra>"),
        row=1, col=1, secondary_y=True,
    )
    # Pin the volume axis to 4x the max so bars sit in the bottom quarter of the
    # pane instead of covering the candles (TradingView's overlay behaviour).
    vol_max = float(df["volume"].max() or 0)
    if vol_max > 0:
        fig.update_yaxes(range=[0, vol_max * 4], secondary_y=True, row=1, col=1,
                         showticklabels=False, showgrid=False)

    # ---- volume climax markers -------------------------------------------
    climax = _signals.volume_climax(df).tail(12)
    if not climax.empty:
        fig.add_trace(
            go.Scatter(
                x=climax.index, y=climax["high"] * 1.012, mode="markers",
                name="Vol climax",
                marker=dict(
                    size=11, symbol="circle-open", line=dict(width=2.5),
                    color=[t["up"] if a == "accumulation" else t["down"] for a in climax["alert"]],
                ),
                customdata=climax[["alert", "rel_volume"]].to_numpy(),
                hovertemplate="<b>%{customdata[0]}</b><br>%{x|%Y-%m-%d %H:%M}<br>"
                              "Relative volume: %{customdata[1]:.1f}x<extra></extra>",
            ),
            row=1, col=1,
        )

    # ---- corporate events -------------------------------------------------
    if show_events:
        start = pd.Timestamp(df.index[0]).tz_localize(None) if getattr(df.index, "tz", None) is not None else df.index[0]
        end = pd.Timestamp(df.index[-1]).tz_localize(None) if getattr(df.index, "tz", None) is not None else df.index[-1]
        y_event = float(df["low"].min()) * 0.985

        divs = _data.dividends(ticker, start, end)
        if not divs.empty:
            fig.add_trace(
                go.Scatter(
                    x=divs.index, y=[y_event] * len(divs), mode="markers+text",
                    name="Dividends", text=["D"] * len(divs), textposition="middle center",
                    textfont=dict(color="#ffffff", size=9),
                    marker=dict(color="#29b6f6", size=13, symbol="circle"),
                    customdata=divs.to_numpy(),
                    hovertemplate="<b>Dividend</b><br>%{x|%Y-%m-%d}<br>$%{customdata:.2f}/share<extra></extra>",
                ),
                row=1, col=1,
            )

        earn = _data.earnings(ticker, start, end)
        if not earn.empty:
            est = earn.get("EPS Estimate", pd.Series(index=earn.index, dtype=float)).fillna(0).to_numpy()
            rep = earn.get("Reported EPS", pd.Series(index=earn.index, dtype=float)).fillna(0).to_numpy()
            fig.add_trace(
                go.Scatter(
                    x=earn.index, y=[y_event] * len(earn), mode="markers+text",
                    name="Earnings", text=["E"] * len(earn), textposition="middle center",
                    textfont=dict(color="#ffffff", size=9),
                    marker=dict(color="#ffa726", size=13, symbol="circle"),
                    customdata=list(zip(est, rep)),
                    hovertemplate="<b>Earnings</b><br>%{x|%Y-%m-%d}<br>"
                                  "Est. EPS: %{customdata[0]:.2f}<br>"
                                  "Reported: %{customdata[1]:.2f}<extra></extra>",
                ),
                row=1, col=1,
            )

    # ---- trade plan overlay ----------------------------------------------
    if plan:
        last_ts = x[-1]
        xs = [x[0], last_ts, last_ts, x[0]]
        fig.add_trace(
            go.Scatter(
                x=xs, y=[plan["target"]] * 2 + [plan["entry"]] * 2, fill="toself",
                fillcolor="rgba(38,166,154,0.10)", line=dict(color=t["up"], width=1, dash="dash"),
                name="Long target", legendgroup="plan", visible="legendonly",
                hovertemplate="Target: %{y:,.2f}<extra></extra>",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=xs, y=[plan["entry"]] * 2 + [plan["stop"]] * 2, fill="toself",
                fillcolor="rgba(239,83,80,0.10)", line=dict(color=t["down"], width=1, dash="dash"),
                name="Stop", legendgroup="plan", visible="legendonly",
                hovertemplate="Stop: %{y:,.2f}<extra></extra>",
            ),
            row=1, col=1,
        )

    # ---- panel 2: RSI -----------------------------------------------------
    if "rsi" in df:
        fig.add_trace(
            go.Scatter(x=x, y=df["rsi"], name="RSI", mode="lines",
                       line=dict(color=t["rsi"], width=1.4),
                       hovertemplate="RSI: %{y:.1f}<extra></extra>"),
            row=2, col=1,
        )
        for lvl in (70, 30):
            fig.add_hline(y=lvl, line_dash="dot", line_color=t["muted"], row=2, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor=t["down"], opacity=0.05, line_width=0, row=2, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor=t["up"], opacity=0.05, line_width=0, row=2, col=1)

    # ---- panel 3: MACD ----------------------------------------------------
    if "macd" in df:
        hist_colors = [t["up"] if v >= 0 else t["down"] for v in df["macd_hist"].fillna(0)]
        fig.add_trace(
            go.Bar(x=x, y=df["macd_hist"], name="Histogram", marker_color=hist_colors,
                   marker_line_width=0, opacity=0.55,
                   hovertemplate="Hist: %{y:.3f}<extra></extra>"),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(x=x, y=df["macd"], name="MACD", mode="lines",
                       line=dict(color=t["macd"], width=1.3),
                       hovertemplate="MACD: %{y:.3f}<extra></extra>"),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(x=x, y=df["macd_signal"], name="Signal", mode="lines",
                       line=dict(color=t["signal"], width=1.3),
                       hovertemplate="Signal: %{y:.3f}<extra></extra>"),
            row=3, col=1,
        )
        fig.add_hline(y=0, line_color=t["muted"], line_width=1, row=3, col=1)

    # ---- title / subtitle (kept out of the plotting area) -----------------
    st = _signals.trend_state(df)
    last = float(df["close"].iloc[-1])
    chg = (last / float(df["close"].iloc[-2]) - 1) * 100 if len(df) > 1 else 0.0
    head = f"<b>{ticker}</b>  {last:,.2f}  <span style='color:{t['up'] if chg >= 0 else t['down']}'>{chg:+.2f}%</span>"
    sub = [f"Trend: <b>{st['label']}</b> ({st['score']:+d}/4)"]
    if "rsi" in df and not pd.isna(df["rsi"].iloc[-1]):
        sub.append(f"RSI {df['rsi'].iloc[-1]:.1f}")
    if "adx" in df and not pd.isna(df["adx"].iloc[-1]):
        sub.append(f"ADX {df['adx'].iloc[-1]:.1f}")
    if "atr_pct" in df and not pd.isna(df["atr_pct"].iloc[-1]):
        sub.append(f"ATR {df['atr_pct'].iloc[-1]:.2f}%")
    if plan:
        sub.append(f"plan {plan['entry']:,.2f} → {plan['target']:,.2f} / stop {plan['stop']:,.2f} (R:R {plan['risk_reward']:.2f})")
    title_html = (
        f"{head}<br><span style='font-size:10px;color:{t['muted']}'>"
        + "  ·  ".join(sub)
        + "</span>"
    )

    # ---- layout -----------------------------------------------------------
    fig.update_layout(
        title=dict(
            text=title or title_html,
            font=dict(color=t["text"], size=15),
            x=0.008, xanchor="left", y=0.985, yanchor="top",
        ),
        template=None,
        paper_bgcolor=t["paper"],
        plot_bgcolor=t["plot"],
        font=dict(color=t["text"], size=11, family="Inter, -apple-system, Segoe UI, sans-serif"),
        height=height,
        margin=dict(l=8, r=8, t=64, b=56),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=t["box"], bordercolor=t["box_border"], font=dict(color=t["text"], size=11)),
        legend=dict(orientation="h", yanchor="top", y=-0.055, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=t["text"]),
                    itemwidth=30),
        xaxis_rangeslider_visible=rangeslider,
        bargap=0.25,
        dragmode="pan",
    )
    fig.update_xaxes(
        showgrid=True, gridcolor=t["grid"], zeroline=False,
        rangebreaks=_rangebreaks(df, intraday),
        showspikes=True, spikemode="across", spikesnap="cursor",
        spikecolor=t["muted"], spikethickness=1, spikedash="dot",
        tickfont=dict(color=t["muted"], size=10),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=t["grid"], zeroline=False,
        tickfont=dict(color=t["muted"], size=10),
        side="right",
    )
    fig.update_yaxes(title_text="", secondary_y=False, row=1, col=1)
    fig.update_yaxes(title_text="", secondary_y=True, row=1, col=1, showgrid=False,
                     tickformat="~s", tickfont=dict(color=t["muted"], size=9),
                     showline=False)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1, tickvals=[0, 30, 50, 70, 100])
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    fig.update_xaxes(showticklabels=True, row=3, col=1)

    return fig


def save_html(
    fig: go.Figure,
    path: str | Path,
    *,
    include_plotlyjs: bool = True,
    title: str | None = None,
) -> Path:
    """Write a self-contained HTML file (works offline, no CDN).

    The page body is repainted with the chart's own background colour so the
    chart does not float on a white page in dark mode.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(path),
        include_plotlyjs="inline" if include_plotlyjs else "cdn",
        full_html=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "modeBarButtonsToAdd": ["drawline", "drawrect", "eraseshape"],
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )

    bg = getattr(fig.layout, "paper_bgcolor", None) or "#131722"
    text = path.read_text(encoding="utf-8")
    style = (
        "<style>html,body{margin:0;padding:0;background:" + bg + ";}"
        ".plotly-graph-div{width:100%!important;}</style>"
    )
    if "</head>" in text:
        text = text.replace("</head>", style + "</head>", 1)
    if title and "<title>" in text:
        import re

        text = re.sub(r"<title>.*?</title>", f"<title>{title}</title>", text, count=1, flags=re.S)
    path.write_text(text, encoding="utf-8")
    return path
