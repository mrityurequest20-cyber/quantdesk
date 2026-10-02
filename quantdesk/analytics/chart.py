"""Chart reading, the way a discretionary trader would, but mechanised:
swing pivots, support/resistance zones, trend structure (HH/HL), floor pivots,
candlestick patterns, gaps, and 52-week context — summarised into a narrative."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators as ind


def swing_pivots(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    """Fractal highs/lows. A pivot at bar i is only *known* at bar i+right, so the
    `confirmed_at` column is what a live system may act on."""
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    rows = []
    for i in range(left, len(df) - right):
        win_h = h[i - left:i + right + 1]
        win_l = l[i - left:i + right + 1]
        if h[i] == win_h.max() and (win_h == h[i]).sum() == 1:
            rows.append((df.index[i], "high", h[i], df.index[i + right]))
        if l[i] == win_l.min() and (win_l == l[i]).sum() == 1:
            rows.append((df.index[i], "low", l[i], df.index[i + right]))
    return pd.DataFrame(rows, columns=["date", "type", "price", "confirmed_at"])


@dataclass
class Level:
    price: float
    kind: str              # support | resistance
    touches: int
    last_touch: pd.Timestamp
    strength: float


def support_resistance(df: pd.DataFrame, lookback: int = 250, max_levels: int = 4) -> list[Level]:
    """Cluster confirmed pivots within ~0.6 ATR into zones, score by touches and recency."""
    d = df.tail(lookback)
    piv = swing_pivots(d)
    if piv.empty:
        return []
    tol = float(ind.atr(d, 14).iloc[-1]) * 0.6
    last = float(d["close"].iloc[-1])
    prices = piv.sort_values("price").reset_index(drop=True)
    clusters: list[list[int]] = []
    for i, p in enumerate(prices["price"]):
        if clusters and p - prices.loc[clusters[-1][-1], "price"] <= tol:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    levels = []
    n = len(d)
    for c in clusters:
        sub = prices.loc[c]
        price = float(sub["price"].mean())
        last_touch = sub["date"].max()
        age = n - d.index.get_loc(last_touch)
        strength = len(c) * (1.0 + np.exp(-age / 60))
        levels.append(Level(price, "support" if price < last else "resistance", len(c), last_touch, strength))
    sup = sorted([l for l in levels if l.kind == "support"], key=lambda l: -l.price)[:max_levels]
    res = sorted([l for l in levels if l.kind == "resistance"], key=lambda l: l.price)[:max_levels]
    return sup + res


def floor_pivots(prev_high: float, prev_low: float, prev_close: float) -> dict[str, float]:
    p = (prev_high + prev_low + prev_close) / 3
    return {"P": p, "R1": 2 * p - prev_low, "S1": 2 * p - prev_high,
            "R2": p + (prev_high - prev_low), "S2": p - (prev_high - prev_low),
            "R3": prev_high + 2 * (p - prev_low), "S3": prev_low - 2 * (prev_high - p)}


def candle_patterns(df: pd.DataFrame) -> list[str]:
    """Patterns completed on the last bar."""
    if len(df) < 8:
        return []
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    body = abs(c[-1] - o[-1])
    rng = max(h[-1] - l[-1], 1e-9)
    upper = h[-1] - max(o[-1], c[-1])
    lower = min(o[-1], c[-1]) - l[-1]
    prev_body_top, prev_body_bot = max(o[-2], c[-2]), min(o[-2], c[-2])
    down_move = c[-1] < c[-6]
    up_move = c[-1] > c[-6]
    out = []
    if body / rng < 0.1:
        out.append("doji (indecision)")
    if lower > 2 * body and upper < body and c[-2] < c[-6]:
        out.append("hammer (bullish reversal after a decline)")
    if upper > 2 * body and lower < body and c[-2] > c[-6]:
        out.append("shooting star (bearish reversal after a rally)")
    if c[-1] > o[-1] and c[-2] < o[-2] and c[-1] >= prev_body_top and o[-1] <= prev_body_bot:
        out.append("bullish engulfing")
    if c[-1] < o[-1] and c[-2] > o[-2] and o[-1] >= prev_body_top and c[-1] <= prev_body_bot:
        out.append("bearish engulfing")
    if h[-1] < h[-2] and l[-1] > l[-2]:
        out.append("inside bar (volatility contraction)")
    if h[-1] > h[-2] and l[-1] < l[-2]:
        out.append("outside bar")
    ranges = (df["high"] - df["low"]).tail(7).to_numpy()
    if ranges[-1] == ranges.min():
        out.append("NR7 (narrowest range in 7 bars: expansion often follows)")
    gap = o[-1] / c[-2] - 1
    if abs(gap) > 0.01:
        out.append(f"{'gap up' if gap > 0 else 'gap down'} {gap:+.1%}")
    _ = down_move, up_move
    return out


def trend_structure(df: pd.DataFrame, lookback: int = 120) -> str:
    piv = swing_pivots(df.tail(lookback))
    highs = piv[piv["type"] == "high"]["price"].tail(3).to_numpy()
    lows = piv[piv["type"] == "low"]["price"].tail(3).to_numpy()
    if len(highs) < 2 or len(lows) < 2:
        return "undetermined"
    hh, hl = highs[-1] > highs[-2], lows[-1] > lows[-2]
    lh, ll = highs[-1] < highs[-2], lows[-1] < lows[-2]
    if hh and hl:
        return "uptrend (higher highs, higher lows)"
    if lh and ll:
        return "downtrend (lower highs, lower lows)"
    if lh and hl:
        return "contracting triangle (lower highs, higher lows)"
    if hh and ll:
        return "expanding / broadening"
    return "sideways"


@dataclass
class ChartReport:
    symbol: str
    date: pd.Timestamp
    close: float
    trend: str
    structure: str
    levels: list[Level]
    patterns: list[str]
    pivots: dict
    stats: dict = field(default_factory=dict)
    narrative: list[str] = field(default_factory=list)


def analyze_chart(symbol: str, df: pd.DataFrame) -> ChartReport:
    close = df["close"]
    last = float(close.iloc[-1])
    s20, s50, s200 = (float(ind.sma(close, n).iloc[-1]) for n in (20, 50, 200))
    a = ind.adx(df).iloc[-1]
    r = float(ind.rsi(close, 14).iloc[-1])
    m = ind.macd(close).iloc[-1]
    atr = float(ind.atr(df).iloc[-1])
    bb = ind.bollinger(close).iloc[-1]
    st = ind.supertrend(df).iloc[-1]
    hi52, lo52 = float(df["high"].tail(252).max()), float(df["low"].tail(252).min())
    prev = df.iloc[-2]
    piv = floor_pivots(prev["high"], prev["low"], prev["close"])
    levels = support_resistance(df)
    pats = candle_patterns(df)
    structure = trend_structure(df)

    above = sum(last > x for x in (s20, s50, s200) if np.isfinite(x))
    if above == 3 and a["adx"] > 20:
        trend = "strong uptrend"
    elif above == 0 and a["adx"] > 20:
        trend = "strong downtrend"
    elif last > s200:
        trend = "uptrend, consolidating" if a["adx"] < 20 else "uptrend"
    else:
        trend = "downtrend, basing" if a["adx"] < 20 else "downtrend"

    stats = {"sma20": s20, "sma50": s50, "sma200": s200, "adx": float(a["adx"]), "+di": float(a["plus_di"]),
             "-di": float(a["minus_di"]), "rsi14": r, "macd_hist": float(m["hist"]), "atr": atr,
             "atr_pct": atr / last, "bb_pct_b": float(bb["pct_b"]), "bb_width": float(bb["bandwidth"]),
             "supertrend": "long" if st["direction"] > 0 else "short", "high_52w": hi52, "low_52w": lo52,
             "from_52w_high": last / hi52 - 1, "from_52w_low": last / lo52 - 1}

    nar = [f"{symbol} closed at {last:,.2f}: {trend}; swing structure is {structure}."]
    nar.append(f"Price vs 20/50/200 SMA: {'above' if last > s20 else 'below'} / {'above' if last > s50 else 'below'} / "
               f"{'above' if last > s200 else 'below'}. ADX {a['adx']:.0f} ({'trending' if a['adx'] > 20 else 'no trend'}),"
               f" +DI {a['plus_di']:.0f} vs -DI {a['minus_di']:.0f}.")
    mom = "overbought" if r > 70 else "oversold" if r < 30 else "neutral"
    nar.append(f"Momentum: RSI14 {r:.0f} ({mom}), MACD histogram {m['hist']:+.2f}, Supertrend {stats['supertrend']}.")
    nar.append(f"Volatility: ATR {atr:,.2f} ({atr / last:.2%} of price); Bollinger %B {bb['pct_b']:.2f}, width {bb['bandwidth']:.2%}.")
    sup = [l for l in levels if l.kind == "support"]
    res = [l for l in levels if l.kind == "resistance"]
    if sup:
        nar.append("Support: " + ", ".join(f"{l.price:,.0f} (x{l.touches})" for l in sup[:3]))
    if res:
        nar.append("Resistance: " + ", ".join(f"{l.price:,.0f} (x{l.touches})" for l in res[:3]))
    nar.append(f"52-week range {lo52:,.0f} – {hi52:,.0f}; price is {stats['from_52w_high']:+.1%} from the high.")
    if pats:
        nar.append("Last bar: " + "; ".join(pats) + ".")
    return ChartReport(symbol, df.index[-1], last, trend, structure, levels, pats, piv, stats, nar)
