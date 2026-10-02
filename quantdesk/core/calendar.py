"""NSE trading calendar and F&O expiry rules.

Expiry rule (2026): NIFTY weekly on Tuesday; monthly contracts on the last Tuesday of
the month. If the expiry day is a holiday, expiry moves to the previous trading day.
"""
from __future__ import annotations

import datetime as dt
from functools import lru_cache

import pandas as pd


class TradingCalendar:
    def __init__(self, holidays: set[dt.date] | None = None):
        self.holidays = frozenset(holidays or ())

    # ---- trading days ----------------------------------------------------------
    def is_trading_day(self, d: dt.date) -> bool:
        d = _as_date(d)
        return d.weekday() < 5 and d not in self.holidays

    def trading_days(self, start, end) -> pd.DatetimeIndex:
        days = pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end))
        return pd.DatetimeIndex([d for d in days if d.date() not in self.holidays])

    def next_trading_day(self, d) -> dt.date:
        d = _as_date(d) + dt.timedelta(days=1)
        while not self.is_trading_day(d):
            d += dt.timedelta(days=1)
        return d

    def prev_trading_day(self, d) -> dt.date:
        d = _as_date(d) - dt.timedelta(days=1)
        while not self.is_trading_day(d):
            d -= dt.timedelta(days=1)
        return d

    def trading_days_between(self, a, b) -> int:
        """Trading days in (a, b]."""
        a, b = _as_date(a), _as_date(b)
        if b <= a:
            return 0
        return len(self.trading_days(a + dt.timedelta(days=1), b))

    # ---- expiries ---------------------------------------------------------------
    def _adjust(self, d: dt.date) -> dt.date:
        return d if self.is_trading_day(d) else self.prev_trading_day(d)

    def monthly_expiry(self, year: int, month: int, weekday: int = 1) -> dt.date:
        return self._adjust(_last_weekday(year, month, weekday))

    def expiries(self, from_date, horizon_days: int = 120, weekday: int = 1, weekly: bool = True) -> list[dt.date]:
        """All expiries in [from_date, from_date + horizon] for an underlying."""
        start = _as_date(from_date)
        end = start + dt.timedelta(days=horizon_days)
        out: set[dt.date] = set()
        if weekly:
            d = start - dt.timedelta(days=7)
            d += dt.timedelta(days=(weekday - d.weekday()) % 7)
            while d <= end + dt.timedelta(days=7):
                out.add(self._adjust(d))
                d += dt.timedelta(days=7)
        y, m = start.year, start.month
        for _ in range(horizon_days // 28 + 3):
            out.add(self.monthly_expiry(y, m, weekday))
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        return sorted(e for e in out if start <= e <= end)

    def pick_expiry(self, from_date, target_dte: int, min_dte: int = 0, weekday: int = 1,
                    weekly: bool = True) -> dt.date:
        """Expiry closest to `target_dte` calendar days while leaving at least `min_dte`."""
        start = _as_date(from_date)
        cands = [e for e in self.expiries(start, max(target_dte * 3, 70), weekday, weekly)
                 if (e - start).days >= min_dte]
        if not cands:
            raise ValueError("no expiry found in horizon")
        return min(cands, key=lambda e: (abs((e - start).days - target_dte), e))


def _as_date(d) -> dt.date:
    if isinstance(d, pd.Timestamp):
        return d.date()
    if isinstance(d, dt.datetime):
        return d.date()
    return d


@lru_cache(maxsize=4096)
def _last_weekday(year: int, month: int, weekday: int) -> dt.date:
    nxt = dt.date(year + (month == 12), month % 12 + 1, 1)
    d = nxt - dt.timedelta(days=1)
    while d.weekday() != weekday:
        d -= dt.timedelta(days=1)
    return d


def year_fraction(from_date, expiry: dt.date) -> float:
    """ACT/365 time to a 15:30 expiry measured from a daily close (0 on expiry day)."""
    return max((_as_date(expiry) - _as_date(from_date)).days, 0) / 365.0
