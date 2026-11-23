"""Signal detection, key levels and multi-timeframe alignment.

Everything here works on an *enriched* DataFrame (see ``indicators.enrich``)
and returns plain Python objects so it can be serialised to JSON or rendered
as chart annotations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "crosses",
    "trend_state",
    "volume_climax",
    "bollinger_squeeze",
    "swing_levels",
    "key_levels",
    "trade_plan",
    "summary",
    "mtf_alignment",
]


def _last_cross(fast: pd.Series, slow: pd.Series, lookback: int = 60) -> dict | None:
    """Most recent fast/slow crossover inside the lookback window."""
    diff = (fast - slow).dropna()
    if len(diff) < 2:
        return None
    diff = diff.iloc[-lookback:]
    sign = np.sign(diff)
    flips = sign[sign != sign.shift(1)].dropna()
    if flips.empty:
        return None
    idx = flips.index[-1]
    return {
        "date": str(idx.date()) if hasattr(idx, "date") else str(idx),
        "direction": "bullish" if flips.iloc[-1] > 0 else "bearish",
        "bars_ago": int(len(diff) - diff.index.get_loc(idx)),
    }


def crosses(df: pd.DataFrame, fast_col: str = "ema_50", slow_col: str = "ema_200", lookback: int = 60) -> dict:
    ema_cross = _last_cross(df[fast_col], df[slow_col], lookback) if {fast_col, slow_col} <= set(df.columns) else None
    macd_cross = _last_cross(df["macd"], df["macd_signal"], lookback) if "macd" in df else None
    return {"ema_cross": ema_cross, "macd_cross": macd_cross}


def trend_state(df: pd.DataFrame) -> dict:
    """Stacked trend score in [-4, +4] with a plain-language label."""
    last = df.iloc[-1]
    score = 0
    reasons = []

    if "ema_50" in df and not pd.isna(last["ema_50"]):
        if last["close"] > last["ema_50"]:
            score += 1
            reasons.append("price above EMA50")
        else:
            score -= 1
            reasons.append("price below EMA50")
    if {"ema_50", "ema_200"} <= set(df.columns) and not pd.isna(last["ema_200"]):
        if last["ema_50"] > last["ema_200"]:
            score += 1
            reasons.append("EMA50 above EMA200 (golden structure)")
        else:
            score -= 1
            reasons.append("EMA50 below EMA200 (death structure)")
    if "macd_hist" in df and not pd.isna(last["macd_hist"]):
        # Momentum only counts when the histogram agrees with the MACD line's
        # side of zero; a shrinking negative histogram is a transition, not a
        # bullish signal.
        if last["macd"] > 0 and last["macd_hist"] > 0:
            score += 1
            reasons.append("MACD above zero with positive histogram")
        elif last["macd"] < 0 and last["macd_hist"] < 0:
            score -= 1
            reasons.append("MACD below zero with negative histogram")
        else:
            reasons.append("MACD in transition (line and histogram disagree)")
    if "rsi" in df and not pd.isna(last["rsi"]):
        if last["rsi"] > 50:
            score += 1
            reasons.append(f"RSI {last['rsi']:.1f} > 50")
        else:
            score -= 1
            reasons.append(f"RSI {last['rsi']:.1f} <= 50")

    labels = {-4: "strong downtrend", -3: "downtrend", -2: "weak downtrend", -1: "mildly bearish",
              0: "neutral", 1: "mildly bullish", 2: "weak uptrend", 3: "uptrend", 4: "strong uptrend"}
    return {"score": score, "label": labels[score], "reasons": reasons}


def volume_climax(df: pd.DataFrame, mult: float = 2.5, lookback: int = 20) -> pd.DataFrame:
    """Bars where volume exceeds ``mult`` x its 20-bar average."""
    if "vol_sma_20" not in df:
        return pd.DataFrame()
    mask = (df["volume"] > df["vol_sma_20"] * mult) & df["vol_sma_20"].notna()
    out = df[mask].copy()
    if out.empty:
        return out
    out["alert"] = np.where(out["close"] >= out["open"], "accumulation", "distribution")
    return out.tail(lookback)


def bollinger_squeeze(df: pd.DataFrame, lookback: int = 100) -> dict:
    """Bandwidth percentile: <20 = squeeze, >80 = expansion."""
    if "bb_upper" not in df:
        return {}
    width = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    width = width.dropna()
    if len(width) < 20:
        return {}
    window = width.iloc[-lookback:]
    pct = float((window < width.iloc[-1]).mean() * 100)
    return {
        "bandwidth": round(float(width.iloc[-1]) * 100, 2),
        "percentile": round(pct, 1),
        "state": "squeeze" if pct < 20 else "expansion" if pct > 80 else "normal",
    }


def swing_levels(df: pd.DataFrame, left: int = 3, right: int = 3, lookback: int = 120) -> dict:
    """Fractal pivot highs/lows — the horizontal levels a discretionary trader draws."""
    d = df.tail(lookback)
    highs, lows = [], []
    h, l = d["high"].to_numpy(), d["low"].to_numpy()
    idx = d.index
    for i in range(left, len(d) - right):
        if h[i] == max(h[i - left : i + right + 1]):
            highs.append((idx[i], float(h[i])))
        if l[i] == min(l[i - left : i + right + 1]):
            lows.append((idx[i], float(l[i])))
    return {"pivots_high": highs[-8:], "pivots_low": lows[-8:]}


def key_levels(df: pd.DataFrame, lookback: int = 120) -> dict:
    last = float(df["close"].iloc[-1])
    piv = swing_levels(df, lookback=lookback)
    highs = [p for _, p in piv["pivots_high"] if p > last]
    lows = [p for _, p in piv["pivots_low"] if p < last]
    out = {
        "last": round(last, 2),
        "resistance": round(min(highs), 2) if highs else None,
        "support": round(max(lows), 2) if lows else None,
    }
    if "dc_upper" in df:
        out["donchian_upper"] = _f(df["dc_upper"].iloc[-1])
        out["donchian_lower"] = _f(df["dc_lower"].iloc[-1])
    if len(df) >= 200:
        out["high_52w"] = _f(df["high"].tail(252).max())
        out["low_52w"] = _f(df["low"].tail(252).min())
    return out


def _f(x):
    try:
        return None if pd.isna(x) else round(float(x), 2)
    except Exception:
        return None


def trade_plan(df: pd.DataFrame, direction: str = "long", atr_mult: float = 1.5, rr: float = 2.0) -> dict:
    """ATR-based stop with a structure-aware target."""
    last = float(df["close"].iloc[-1])
    a = float(df["atr"].iloc[-1]) if "atr" in df and not pd.isna(df["atr"].iloc[-1]) else last * 0.02
    lv = key_levels(df)

    if direction == "long":
        stop = last - atr_mult * a
        if lv.get("support"):
            stop = max(stop, lv["support"] * 0.995)
        target = lv.get("resistance") or last + rr * (last - stop)
        target = max(target, last + rr * (last - stop))
    else:
        stop = last + atr_mult * a
        if lv.get("resistance"):
            stop = min(stop, lv["resistance"] * 1.005)
        target = lv.get("support") or last - rr * (stop - last)
        target = min(target, last - rr * (stop - last))

    risk = abs(last - stop)
    reward = abs(target - last)
    return {
        "direction": direction,
        "entry": round(last, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "risk_per_share": round(risk, 2),
        "reward_per_share": round(reward, 2),
        "risk_reward": round(reward / risk, 2) if risk else None,
        "stop_pct": round(-risk / last * 100, 2),
        "target_pct": round(reward / last * 100, 2),
    }


def summary(df: pd.DataFrame, ticker: str = "") -> dict:
    last = df.iloc[-1]
    out = {
        "ticker": ticker,
        "last_close": _f(last["close"]),
        "last_bar": str(df.index[-1]),
        "change_pct": _f((last["close"] / df["close"].iloc[-2] - 1) * 100) if len(df) > 1 else None,
        "trend": trend_state(df),
        "crosses": crosses(df),
        "levels": key_levels(df),
        "squeeze": bollinger_squeeze(df),
        "rsi": _f(last.get("rsi")),
        "atr_pct": _f(last.get("atr_pct")),
        "adx": _f(last.get("adx")),
        "rel_volume": _f(last.get("rel_volume")),
    }
    return out


def mtf_alignment(
    frames: dict[str, pd.DataFrame],
    weights: dict[str, float] | None = None,
) -> dict:
    """Score a stack of timeframes (e.g. 1wk/1d/1h/15m) into one bias.

    ``frames`` maps a label to an enriched DataFrame.  The score is the
    weighted mean of each timeframe's trend score, rescaled to [-1, 1].
    """
    weights = weights or {k: 1.0 for k in frames}
    rows, total_w, total = [], 0.0, 0.0
    for label, df in frames.items():
        if df is None or df.empty:
            continue
        st = trend_state(df)
        w = float(weights.get(label, 1.0))
        total += st["score"] * w
        total_w += w
        rows.append(
            {
                "timeframe": label,
                "score": st["score"],
                "label": st["label"],
                "close": _f(df["close"].iloc[-1]),
                "rsi": _f(df["rsi"].iloc[-1]) if "rsi" in df else None,
                "adx": _f(df["adx"].iloc[-1]) if "adx" in df else None,
            }
        )
    if not total_w:
        return {"bias": "no data", "score": None, "timeframes": []}
    score = total / total_w / 4.0
    bias = "bullish" if score > 0.35 else "bearish" if score < -0.35 else "mixed"
    return {"bias": bias, "score": round(score, 3), "timeframes": rows}
