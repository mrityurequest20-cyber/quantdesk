"""Data quality: cleaning plus a report of anything suspicious (fed into the routine checks)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DataIssue:
    symbol: str
    severity: str        # WARN | FAIL
    message: str


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0].copy()
    for c in ("open", "high", "low"):
        df[c] = df[c].where(df[c] > 0, df["close"])
    # Enforce OHLC consistency: high is the max, low the min of the bar.
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    df["volume"] = df["volume"].fillna(0.0).clip(lower=0)
    return df


def audit(symbol: str, df: pd.DataFrame, max_abs_return: float = 0.20,
          as_of: pd.Timestamp | None = None, max_stale_days: int = 4) -> list[DataIssue]:
    issues: list[DataIssue] = []
    if df is None or df.empty:
        return [DataIssue(symbol, "FAIL", "no data")]
    rets = np.log(df["close"]).diff().dropna()
    big = rets[rets.abs() > np.log1p(max_abs_return)]
    for ts, r in big.items():
        issues.append(DataIssue(symbol, "WARN", f"{ts.date()} one-day move {np.expm1(r):+.1%} (bad tick or real gap?)"))
    flat = (df["high"] == df["low"]).tail(20).sum()
    if flat > 5 and symbol not in ("INDIAVIX",):
        issues.append(DataIssue(symbol, "WARN", f"{flat} of last 20 bars have high == low (illiquid or stale feed)"))
    if (df["volume"].tail(10) == 0).all() and df["volume"].sum() > 0:
        issues.append(DataIssue(symbol, "WARN", "zero volume for the last 10 bars"))
    gaps = df.index.to_series().diff().dt.days.dropna()
    long_gaps = gaps[gaps > 6]
    for ts, g in long_gaps.tail(3).items():
        issues.append(DataIssue(symbol, "WARN", f"{int(g)}-day gap ending {ts.date()}"))
    if as_of is not None:
        stale = (pd.Timestamp(as_of).normalize() - df.index[-1]).days
        if stale > max_stale_days:
            issues.append(DataIssue(symbol, "FAIL", f"last bar {df.index[-1].date()} is {stale} days old"))
    return issues
