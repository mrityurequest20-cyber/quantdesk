"""Data provider interface. Every provider returns daily OHLCV frames in one shape:
DatetimeIndex (tz-naive, normalised to dates) and float columns open/high/low/close/volume."""
from __future__ import annotations

import abc
import logging

import pandas as pd

from .validation import clean_ohlcv

log = logging.getLogger(__name__)
COLUMNS = ["open", "high", "low", "close", "volume"]


class DataProvider(abc.ABC):
    name = "base"

    @abc.abstractmethod
    def history(self, symbol: str, start, end=None) -> pd.DataFrame:
        """Daily bars for `symbol` in [start, end]."""

    def universe(self, symbols: list[str], start, end=None) -> dict[str, pd.DataFrame]:
        out = {}
        for s in symbols:
            try:
                df = self.history(s, start, end)
            except Exception as exc:  # one bad symbol must not kill a scan
                log.warning("data: %s failed on %s: %s", self.name, s, exc)
                continue
            if df is not None and len(df):
                out[s] = df
        return out


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: str(c).lower().replace(" ", "_") for c in df.columns})
    if "adj_close" in df.columns and "close" not in df.columns:
        df = df.rename(columns={"adj_close": "close"})
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "date"
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df[COLUMNS].astype(float)
    return clean_ohlcv(df)


def align(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Restrict every frame to the union calendar, forward-filling only *gaps inside*
    a series (never before its first bar), so cross-sectional code can index by date."""
    if not frames:
        return frames
    idx = sorted(set().union(*[f.index for f in frames.values()]))
    idx = pd.DatetimeIndex(idx)
    out = {}
    for s, f in frames.items():
        g = f.reindex(idx)
        first = f.index[0]
        g.loc[g.index >= first, "close"] = g.loc[g.index >= first, "close"].ffill()
        for c in ("open", "high", "low"):
            g[c] = g[c].fillna(g["close"])
        g["volume"] = g["volume"].fillna(0.0)
        out[s] = g.loc[g.index >= first]
    return out
