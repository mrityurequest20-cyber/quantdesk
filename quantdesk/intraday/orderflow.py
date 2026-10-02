"""Order-flow analytics: volume profile, footprint, delta / cumulative delta.

Two fidelity levels, same outputs:

* **Tick level** (exact-ish): trades carry price, size and — when the feed gives quotes —
  an aggressor side inferred with the quote rule (trade at/above ask = buyer-initiated,
  at/below bid = seller-initiated) and the tick rule as the fallback (Lee & Ready).
  Footprint bars hold buy/sell volume per price level, delta, and diagonal imbalances.
  This is where a tick-by-tick order-flow feed (Kite full-mode snapshots today, a
  GoCharting / vendor order-flow feed later) plugs in via `Trade`.
* **Bar level** (approximate): with only OHLCV candles, a bar's volume is spread over
  its range for the profile, and delta is estimated from where the bar closed within its
  range (close-location value). Everything computed this way is labelled approximate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Trade:
    ts: pd.Timestamp
    price: float
    size: float
    side: int = 0          # +1 buyer-initiated, -1 seller-initiated, 0 unknown
    bid: float | None = None
    ask: float | None = None


class TickClassifier:
    """Quote rule with tick-rule fallback. Keeps the last price and last non-zero tick."""

    def __init__(self):
        self.last_price: float | None = None
        self.last_dir = 0

    def classify(self, t: Trade) -> Trade:
        side = t.side
        if side == 0 and t.bid is not None and t.ask is not None and t.ask > t.bid:
            if t.price >= t.ask:
                side = 1
            elif t.price <= t.bid:
                side = -1
            else:  # inside the spread: nearer side wins
                mid = (t.bid + t.ask) / 2
                side = 1 if t.price > mid else -1 if t.price < mid else 0
        if side == 0 and self.last_price is not None:
            if t.price > self.last_price:
                side = 1
            elif t.price < self.last_price:
                side = -1
            else:
                side = self.last_dir                       # zero tick: inherit the last direction
        if t.price != self.last_price and self.last_price is not None:
            self.last_dir = 1 if t.price > self.last_price else -1
        self.last_price = t.price
        t.side = side
        return t


class KiteSnapshotAdapter:
    """Kite full-mode ticks are ~1/s snapshots with cumulative volume and top-of-book.
    Turn consecutive snapshots into synthetic trades: size = Δ cumulative volume at the
    snapshot's last price, classified against the snapshot's best bid/ask."""

    def __init__(self):
        self.last_volume: dict[int, float] = {}

    def to_trades(self, tick: dict) -> list[Trade]:
        tok = tick.get("instrument_token")
        vol = float(tick.get("volume_traded") or tick.get("volume") or 0)
        prev = self.last_volume.get(tok)
        self.last_volume[tok] = vol
        if prev is None or vol <= prev:
            return []
        depth = tick.get("depth") or {}
        bid = (depth.get("buy") or [{}])[0].get("price")
        ask = (depth.get("sell") or [{}])[0].get("price")
        ts = pd.Timestamp(tick.get("last_trade_time") or tick.get("exchange_timestamp") or pd.Timestamp.now())
        return [Trade(ts, float(tick["last_price"]), vol - prev, 0, bid or None, ask or None)]


# ---- volume profile ----------------------------------------------------------------------------
@dataclass
class Profile:
    prices: np.ndarray          # bin centres
    volume: np.ndarray
    poc: float
    vah: float
    val: float
    approximate: bool
    hvn: list[float] = field(default_factory=list)
    lvn: list[float] = field(default_factory=list)

    def position(self, price: float) -> str:
        if price > self.vah:
            return "above value"
        if price < self.val:
            return "below value"
        return "inside value"

    def to_dict(self) -> dict:
        return {"poc": self.poc, "vah": self.vah, "val": self.val, "approximate": self.approximate,
                "hvn": self.hvn[:3], "lvn": self.lvn[:3]}


def _value_area(prices: np.ndarray, vol: np.ndarray, pct: float = 0.70) -> tuple[float, float, float]:
    """CBOT value-area algorithm: start at the POC, add the larger adjacent pair until pct."""
    i_poc = int(np.argmax(vol))
    total = vol.sum()
    lo = hi = i_poc
    acc = vol[i_poc]
    while acc < pct * total and (lo > 0 or hi < len(vol) - 1):
        up = vol[hi + 1:hi + 3].sum() if hi < len(vol) - 1 else -1
        dn = vol[max(lo - 2, 0):lo].sum() if lo > 0 else -1
        if up >= dn:
            step = min(2, len(vol) - 1 - hi)
            acc += vol[hi + 1:hi + 1 + step].sum()
            hi += step
        else:
            step = min(2, lo)
            acc += vol[lo - step:lo].sum()
            lo -= step
    return float(prices[i_poc]), float(prices[hi]), float(prices[lo])


def _nodes(prices, vol) -> tuple[list[float], list[float]]:
    if len(vol) < 5:
        return [], []
    sm = np.convolve(vol, np.ones(3) / 3, mode="same")
    peaks = [i for i in range(1, len(sm) - 1) if sm[i] >= sm[i - 1] and sm[i] > sm[i + 1] and sm[i] > np.median(sm)]
    troughs = [i for i in range(1, len(sm) - 1) if sm[i] <= sm[i - 1] and sm[i] < sm[i + 1] and sm[i] < np.median(sm)]
    peaks.sort(key=lambda i: -sm[i])
    troughs.sort(key=lambda i: sm[i])
    return [float(prices[i]) for i in peaks], [float(prices[i]) for i in troughs]


def profile_from_bars(df: pd.DataFrame, tick: float | None = None, bins: int | None = None) -> Profile | None:
    """Spread each bar's volume uniformly across its high-low range (approximate profile)."""
    if df is None or df.empty or df["volume"].sum() <= 0:
        return None
    lo, hi = float(df["low"].min()), float(df["high"].max())
    if tick is None:
        tick = max((hi - lo) / (bins or 80), 1e-9)
    edges = np.arange(np.floor(lo / tick) * tick, np.ceil(hi / tick) * tick + tick, tick)
    if len(edges) < 2:
        edges = np.array([lo - tick, hi + tick])
    n = len(edges) - 1
    a = np.clip(np.searchsorted(edges, df["low"].to_numpy(), side="right") - 1, 0, n - 1)
    b = np.clip(np.searchsorted(edges, df["high"].to_numpy(), side="right") - 1, 0, n - 1)
    b = np.maximum(a, b)
    per = df["volume"].to_numpy(dtype=float) / (b - a + 1)
    diff = np.zeros(n + 1)                     # difference array: spread each bar's volume over its bins
    np.add.at(diff, a, per)
    np.add.at(diff, b + 1, -per)
    vol = np.cumsum(diff)[:n]
    centres = (edges[:-1] + edges[1:]) / 2
    poc, vah, val = _value_area(centres, vol)
    hvn, lvn = _nodes(centres, vol)
    return Profile(centres, vol, poc, vah, val, True, hvn, lvn)


def profile_from_trades(trades: list[Trade], tick: float) -> Profile | None:
    if not trades:
        return None
    px = np.array([round(t.price / tick) * tick for t in trades])
    sz = np.array([t.size for t in trades], dtype=float)
    levels = np.unique(px)
    vol = np.array([sz[px == p].sum() for p in levels])
    full = np.arange(levels.min(), levels.max() + tick / 2, tick)
    fv = np.zeros(len(full))
    fv[np.round((levels - full[0]) / tick).astype(int)] = vol
    poc, vah, val = _value_area(full, fv)
    hvn, lvn = _nodes(full, fv)
    return Profile(full, fv, poc, vah, val, False, hvn, lvn)


# ---- footprint -----------------------------------------------------------------------------------
@dataclass
class FootprintBar:
    start: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    levels: dict[float, list[float]]       # price -> [sell_volume, buy_volume]
    delta: float
    volume: float
    max_delta: float
    min_delta: float
    poc: float
    imbalances: list[tuple[float, str, float]]   # (price, 'buy'|'sell', ratio)

    @property
    def stacked(self) -> dict[str, int]:
        """Longest run of consecutive same-side imbalances (3+ is the classic signal)."""
        out = {"buy": 0, "sell": 0}
        for side in out:
            ps = sorted(p for p, s, _ in self.imbalances if s == side)
            run = best = 0
            for a, b in zip([None] + ps, ps):
                run = run + 1 if a is not None and abs(b - a) <= self._tick * 1.01 else 1
                best = max(best, run)
            out[side] = best
        return out

    _tick: float = 0.05


class FootprintBuilder:
    """Aggregate classified trades into fixed-interval footprint bars."""

    def __init__(self, tick: float, interval: str = "1min", imbalance_ratio: float = 3.0):
        self.tick = tick
        self.interval = pd.Timedelta(interval)
        self.ratio = imbalance_ratio
        self.classifier = TickClassifier()
        self.bars: list[FootprintBar] = []
        self._cur: dict | None = None

    def add(self, t: Trade) -> FootprintBar | None:
        t = self.classifier.classify(t)
        start = t.ts.floor(self.interval)
        done = None
        if self._cur is not None and start != self._cur["start"]:
            done = self._close()
        if self._cur is None:
            self._cur = {"start": start, "open": t.price, "high": t.price, "low": t.price, "close": t.price,
                         "levels": {}, "delta": 0.0, "vol": 0.0, "maxd": 0.0, "mind": 0.0}
        c = self._cur
        p = round(t.price / self.tick) * self.tick
        lv = c["levels"].setdefault(p, [0.0, 0.0])
        if t.side > 0:
            lv[1] += t.size
        elif t.side < 0:
            lv[0] += t.size
        c["delta"] += t.side * t.size
        c["vol"] += t.size
        c["maxd"], c["mind"] = max(c["maxd"], c["delta"]), min(c["mind"], c["delta"])
        c["high"], c["low"], c["close"] = max(c["high"], t.price), min(c["low"], t.price), t.price
        return done

    def _close(self) -> FootprintBar:
        c = self._cur
        prices = sorted(c["levels"])
        imb = []
        for i, p in enumerate(prices):
            # diagonal comparison: ask volume at p vs bid volume one tick lower, and vice versa
            buy = c["levels"][p][1]
            below = c["levels"].get(round((p - self.tick) / self.tick) * self.tick, [0, 0])[0]
            if buy > 0 and buy >= self.ratio * max(below, 1e-9) and buy > 0:
                imb.append((p, "buy", buy / max(below, 1.0)))
            sell = c["levels"][p][0]
            above = c["levels"].get(round((p + self.tick) / self.tick) * self.tick, [0, 0])[1]
            if sell > 0 and sell >= self.ratio * max(above, 1e-9):
                imb.append((p, "sell", sell / max(above, 1.0)))
        poc = max(prices, key=lambda p: sum(c["levels"][p])) if prices else c["close"]
        bar = FootprintBar(c["start"], c["open"], c["high"], c["low"], c["close"], c["levels"], c["delta"], c["vol"],
                           c["maxd"], c["mind"], poc, imb)
        bar._tick = self.tick
        self.bars.append(bar)
        self._cur = None
        return bar

    def flush(self) -> FootprintBar | None:
        return self._close() if self._cur else None

    def cvd(self) -> pd.Series:
        return pd.Series([b.delta for b in self.bars], index=[b.start for b in self.bars]).cumsum()


# ---- bar-level delta -----------------------------------------------------------------------------
def approx_delta(df: pd.DataFrame) -> pd.Series:
    """Close-location value × volume: +volume if the bar closed on its high, -volume on its low."""
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng
    return (clv.fillna(0) * df["volume"]).rename("approx_delta")


def delta_divergence(price: pd.Series, cvd: pd.Series, lookback: int = 20) -> int:
    """+1 bullish (price lower low, CVD higher low), -1 bearish (price higher high, CVD lower high), 0 none."""
    if len(price) < lookback * 2:
        return 0
    p1, p2 = price.iloc[-2 * lookback:-lookback], price.iloc[-lookback:]
    c1, c2 = cvd.iloc[-2 * lookback:-lookback], cvd.iloc[-lookback:]
    if p2.max() > p1.max() and c2.max() < c1.max():
        return -1
    if p2.min() < p1.min() and c2.min() > c1.min():
        return 1
    return 0
