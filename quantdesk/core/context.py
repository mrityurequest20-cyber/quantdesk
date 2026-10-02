"""What a strategy can see on bar t: fast point-in-time lookups, nothing after t."""
from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd


class FeatureTable:
    """A DataFrame frozen into numpy columns plus a timestamp→row map.
    Point lookups are ~20x faster than DataFrame.loc, which matters in a bar loop."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.index = df.index
        self._pos = {ts: i for i, ts in enumerate(df.index)}
        self._arr = {c: df[c].to_numpy() for c in df.columns}

    def pos(self, ts) -> int | None:
        return self._pos.get(ts)

    def at(self, ts, col: str, default: Any = np.nan):
        i = self._pos.get(ts)
        return default if i is None else self._arr[col][i]

    def row(self, ts) -> dict | None:
        i = self._pos.get(ts)
        if i is None:
            return None
        return {c: a[i] for c, a in self._arr.items()}

    def prev(self, ts, col: str, k: int = 1, default: Any = np.nan):
        i = self._pos.get(ts)
        return default if i is None or i - k < 0 else self._arr[col][i - k]

    def window(self, ts, col: str, n: int) -> np.ndarray:
        i = self._pos.get(ts)
        if i is None:
            return np.array([])
        return self._arr[col][max(0, i - n + 1): i + 1]

    def upto(self, ts) -> pd.DataFrame:
        i = self._pos.get(ts)
        return self.df.iloc[: (i + 1 if i is not None else 0)]


class MarketContext:
    def __init__(self, cfg, bars: dict[str, FeatureTable], regimes: dict[str, FeatureTable],
                 iv: dict[str, FeatureTable], pricer, calendar, costs):
        self.cfg = cfg
        self.bars = bars
        self.regimes = regimes
        self.iv = iv
        self.pricer = pricer
        self.calendar = calendar
        self.costs = costs
        self.benchmark = cfg.get("universe.benchmark", "NIFTY")
        self.events = cfg.events()
        self.ts: pd.Timestamp | None = None
        self.i = 0
        self.equity = 0.0
        self.open_trades: list = []
        self.closed_trades: list = []

    def at(self, ts: pd.Timestamp, i: int, equity: float, open_trades: list, closed_trades: list) -> "MarketContext":
        self.ts, self.i, self.equity = ts, i, equity
        self.open_trades, self.closed_trades = open_trades, closed_trades
        return self

    @property
    def date(self) -> dt.date:
        return self.ts.date()

    # ---- prices ------------------------------------------------------------------------
    def has(self, sym: str) -> bool:
        t = self.bars.get(sym)
        return t is not None and t.pos(self.ts) is not None

    def price(self, sym: str, field: str = "close") -> float:
        return float(self.bars[sym].at(self.ts, field))

    def bar(self, sym: str) -> dict | None:
        t = self.bars.get(sym)
        return t.row(self.ts) if t else None

    def history(self, sym: str, n: int | None = None) -> pd.DataFrame:
        df = self.bars[sym].upto(self.ts)
        return df if n is None else df.tail(n)

    # ---- regime / vol ------------------------------------------------------------------
    def regime(self, sym: str | None = None) -> str:
        t = self.regimes.get(sym or self.benchmark) or self.regimes.get(self.benchmark)
        return str(t.at(self.ts, "label", "range")) if t else "range"

    def regime_row(self, sym: str | None = None) -> dict:
        t = self.regimes.get(sym or self.benchmark) or self.regimes.get(self.benchmark)
        return (t.row(self.ts) if t else None) or {}

    def atm_iv(self, sym: str, field: str = "iv") -> float:
        """ATM implied vol (decimal) for an index, from India VIX scaled by `iv_beta`."""
        t = self.iv.get(sym)
        return float(t.at(self.ts, field)) if t else float("nan")

    def iv_row(self, sym: str) -> dict:
        t = self.iv.get(sym)
        return (t.row(self.ts) if t else None) or {}

    def event_within(self, days: int) -> str | None:
        d0 = self.date
        for d, name in self.events:
            if 0 <= (d - d0).days <= days:
                return f"{name} on {d}"
        return None

    def trades_for(self, strategy: str, symbol: str | None = None) -> list:
        return [t for t in self.open_trades if t.strategy == strategy and (symbol is None or t.symbol == symbol)]
