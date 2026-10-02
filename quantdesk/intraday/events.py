"""Scheduled events, with times: what the desk stands aside for, and what it only needs to know.

An event that lands inside the session (RBI's policy at 10:00, the Union Budget at 11:00) blocks new entries for a
window around the announcement, not the whole day. One that lands after the close (FOMC at ~23:30 IST, US CPI at
18:00–19:00 IST, India CPI at 16:00) can't move today's tape: it is context today, and "since the last close"
context the next morning. An event without a time keeps the old rule: the whole session.

Heavyweight results come from NSE's event calendar (the warehouse's corp_events table): context only, untested."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

OPEN, CLOSE = dt.time(9, 15), dt.time(15, 30)


@dataclass
class Event:
    day: dt.date
    name: str
    time: dt.time | None = None          # IST; None = the whole session
    before_min: int = 15
    after_min: int = 45
    source: str = "calendar"

    @property
    def in_session(self) -> bool:
        return self.time is None or OPEN <= self.time < CLOSE

    def window(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        if self.time is None:
            return (pd.Timestamp(dt.datetime.combine(self.day, OPEN), tz="Asia/Kolkata"),
                    pd.Timestamp(dt.datetime.combine(self.day, CLOSE), tz="Asia/Kolkata"))
        t = pd.Timestamp(dt.datetime.combine(self.day, self.time), tz="Asia/Kolkata")
        return t - pd.Timedelta(minutes=self.before_min), t + pd.Timedelta(minutes=self.after_min)

    def label(self) -> str:
        return f"{self.name}{f' at {self.time:%H:%M}' if self.time else ''}"


def from_config(cfg) -> list[Event]:
    out = []
    for ev in cfg.get("calendar.events", []) or []:
        d = ev["date"] if isinstance(ev["date"], dt.date) else dt.date.fromisoformat(str(ev["date"]))
        t = ev.get("time")
        t = None if t in (None, "") else (t if isinstance(t, dt.time) else dt.time.fromisoformat(str(t)))
        out.append(Event(d, ev["name"], t, int(ev.get("before_min", 15)), int(ev.get("after_min", 45))))
    return sorted(out, key=lambda e: (e.day, e.time or dt.time(0)))


def heavyweight_results(corp: pd.DataFrame | None, heavy: list[str]) -> list[Event]:
    """Board meetings about financial results for the index heavyweights (time unknown: context only)."""
    if corp is None or corp.empty or not heavy:
        return []
    c = corp[corp["symbol"].isin(heavy) & corp["purpose"].astype(str).str.contains("result", case=False)]
    return [Event(pd.Timestamp(r.date).date(), f"{r.symbol} results (board meeting)", dt.time(23, 59), source="nse")
            for r in c.drop_duplicates(["date", "symbol"]).itertuples()]


def load_corp_events(folder: Path) -> pd.DataFrame | None:
    files = sorted(Path(folder).glob("corp_events_*.parquet"))
    if not files:
        return None
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


class EventBook:
    def __init__(self, events: list[Event]):
        self.events = sorted(events, key=lambda e: (e.day, e.time or dt.time(0)))

    def on(self, day: dt.date) -> list[Event]:
        return [e for e in self.events if e.day == day]

    def blocking(self, now) -> Event | None:
        """The in-session event whose window `now` is in (new entries wait), or None."""
        now = pd.Timestamp(now)
        for e in self.on(now.date()):
            if e.in_session:
                lo, hi = e.window()
                if lo <= now <= hi:
                    return e
        return None

    def since_last_close(self, day: dt.date, prev_session: dt.date | None) -> list[Event]:
        """Out-of-hours events between the previous session's close and this one's open."""
        def pre_open(e):
            return e.time is not None and e.time < OPEN
        def after_close(e):
            return e.time is not None and e.time >= CLOSE
        return [e for e in self.events if (prev_session is not None and e.day == prev_session and after_close(e))
                or (prev_session is not None and prev_session < e.day < day) or (e.day == day and pre_open(e))]

    def describe(self, day: dt.date, prev_session: dt.date | None = None) -> list[str]:
        """Labels for the session: what happened since the last close, then today's events in and after hours."""
        today = [f"{e.label()}{'' if e.in_session else ' (after the close)'}" for e in self.on(day)
                 if e.time is None or e.time >= OPEN]
        before = [f"since the last close: {e.label()}" for e in self.since_last_close(day, prev_session)]
        return before + today
