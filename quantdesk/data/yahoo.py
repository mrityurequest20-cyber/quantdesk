"""Yahoo Finance provider (via yfinance) with an on-disk CSV cache.

NSE symbols map to `<SYMBOL>.NS`, indices to their caret tickers (^NSEI, ^NSEBANK, ^INDIAVIX)
via the `instruments` section of the config."""
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import pandas as pd

from .base import DataProvider, normalise

log = logging.getLogger(__name__)


class YahooProvider(DataProvider):
    name = "yahoo"

    def __init__(self, cfg, cache_dir: Path | None = None, max_cache_age_hours: float = 6.0):
        self.cfg = cfg
        self.cache_dir = cache_dir or (cfg.runtime_dir / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age = dt.timedelta(hours=max_cache_age_hours)

    def ticker(self, symbol: str) -> str:
        return self.cfg.instrument_spec(symbol).get("yahoo", f"{symbol}.NS")

    def history(self, symbol: str, start, end=None) -> pd.DataFrame:
        path = self.cache_dir / f"{symbol}.csv"
        if path.exists() and dt.datetime.now() - dt.datetime.fromtimestamp(path.stat().st_mtime) < self.max_age:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            df = self._download(symbol, start, end)
            df.to_csv(path)
        df = df.loc[pd.Timestamp(start):]
        if end is not None:
            df = df.loc[: pd.Timestamp(end)]
        return df

    def _download(self, symbol: str, start, end) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pip install yfinance to use the yahoo data source") from exc
        tkr = yf.Ticker(self.ticker(symbol))
        raw = tkr.history(start=pd.Timestamp(start).date().isoformat(),
                          end=(pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat() if end else None,
                          interval="1d", auto_adjust=True)
        if raw is None or raw.empty:
            raise RuntimeError(f"yahoo returned no rows for {symbol} ({self.ticker(symbol)})")
        return normalise(raw)
