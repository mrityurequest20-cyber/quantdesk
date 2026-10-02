"""CSV provider: one file per symbol, e.g. data/csv/NIFTY.csv with
date,open,high,low,close,volume. Handy for NSE bhavcopy exports or broker downloads."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import DataProvider, normalise


class CSVProvider(DataProvider):
    name = "csv"

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)

    def history(self, symbol: str, start, end=None) -> pd.DataFrame:
        path = self.dir / f"{symbol}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        date_col = next(c for c in df.columns if c.lower() in ("date", "datetime", "timestamp"))
        df = df.set_index(pd.to_datetime(df.pop(date_col)))
        df = normalise(df).loc[pd.Timestamp(start):]
        return df.loc[: pd.Timestamp(end)] if end is not None else df

    def save(self, symbol: str, df: pd.DataFrame) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{symbol}.csv"
        df.to_csv(path, index_label="date")
        return path
