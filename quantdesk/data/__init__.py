from __future__ import annotations

from .base import DataProvider, align
from .csv_store import CSVProvider
from .synthetic import SyntheticProvider
from .yahoo import YahooProvider


def make_provider(cfg, source: str | None = None, **kw) -> DataProvider:
    source = source or cfg.get("data.source", "yahoo")
    if source == "yahoo":
        return YahooProvider(cfg)
    if source == "csv":
        path = cfg.get("data.csv_dir", "data/csv")
        return CSVProvider(path if str(path).startswith("/") else cfg.root / path)
    if source == "synthetic":
        return SyntheticProvider(cfg, **kw)
    raise ValueError(f"unknown data source {source!r}")


__all__ = ["DataProvider", "CSVProvider", "SyntheticProvider", "YahooProvider", "make_provider", "align"]
