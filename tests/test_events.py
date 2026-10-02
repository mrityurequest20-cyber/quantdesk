"""Timed events: an announcement inside the session blocks entries around it, not all day; after-hours events are
context today and "since the last close" tomorrow; GIFT Nifty is read before the open and recorded."""
import datetime as dt

import pandas as pd

from quantdesk.intraday.events import Event, EventBook, from_config, heavyweight_results

IST = "Asia/Kolkata"


def ts(s):
    return pd.Timestamp(s, tz=IST)


def test_config_events_carry_times(cfg):
    ev = {(e.day, e.name): e for e in from_config(cfg)}
    rbi = ev[(dt.date(2026, 10, 7), "RBI MPC decision")]
    assert rbi.time == dt.time(10, 0) and rbi.in_session
    assert not ev[(dt.date(2026, 10, 28), "US FOMC decision")].in_session


def test_only_an_in_session_announcement_blocks_and_only_around_it(cfg):
    book = EventBook(from_config(cfg))
    assert book.blocking(ts("2026-10-07 09:40")) is None                        # before the window
    assert book.blocking(ts("2026-10-07 09:50")).name == "RBI MPC decision"   # 10:00 − 15 min
    assert book.blocking(ts("2026-10-07 10:40")).name == "RBI MPC decision"   # 10:00 + 45 min
    assert book.blocking(ts("2026-10-07 10:50")) is None                        # trading again
    assert book.blocking(ts("2026-10-28 11:00")) is None                        # FOMC lands at 23:30 IST
    assert book.blocking(ts("2026-10-14 14:00")) is None                        # US CPI at 18:00 IST
    whole = EventBook([Event(dt.date(2026, 10, 9), "unscheduled-time event")])
    assert whole.blocking(ts("2026-10-09 14:00")).name == "unscheduled-time event"   # no time: the whole session


def test_after_hours_events_are_context_the_next_morning(cfg):
    book = EventBook(from_config(cfg))
    assert book.describe(dt.date(2026, 10, 28), dt.date(2026, 10, 27)) == ["US FOMC decision at 23:30 (after the close)"]
    assert book.describe(dt.date(2026, 10, 29), dt.date(2026, 10, 28)) == ["since the last close: US FOMC decision at 23:30"]
    # 9 Dec FOMC lands at 00:30 IST on 10 Dec: before that session's open
    dec10 = book.describe(dt.date(2026, 12, 10), dt.date(2026, 12, 9))
    assert "since the last close: US FOMC decision (9 Dec, US time) at 00:30" in dec10
    assert "US CPI (Nov) at 19:00 (after the close)" in dec10


def test_heavyweight_results_are_context_only():
    corp = pd.DataFrame({"date": [dt.date(2026, 10, 17), dt.date(2026, 10, 17), dt.date(2026, 10, 18)],
                         "symbol": ["HDFCBANK", "TINYCO", "ICICIBANK"],
                         "purpose": ["Financial Results", "Financial Results", "Financial Results/Dividend"]})
    ev = heavyweight_results(corp, ["HDFCBANK", "ICICIBANK"])
    assert [e.name for e in ev] == ["HDFCBANK results (board meeting)", "ICICIBANK results (board meeting)"]
    book = EventBook(ev)
    assert book.blocking(ts("2026-10-17 13:00")) is None
    assert book.describe(dt.date(2026, 10, 19), dt.date(2026, 10, 16)) == [
        "since the last close: HDFCBANK results (board meeting) at 23:59",
        "since the last close: ICICIBANK results (board meeting) at 23:59"]


def test_gift_nifty_is_read_before_the_open_and_recorded(cfg, tmp_path):
    from quantdesk.intraday.engine import IntradayEngine
    from quantdesk.intraday.feeds import ReplayFeed
    from quantdesk.intraday.recorder import SessionRecorder
    from quantdesk.intraday.sim import IntradayBroker
    from quantdesk.intraday.synthetic import simulate_sessions
    from quantdesk.journal.journal import Journal
    day = dt.date(2026, 10, 5)
    bars, _ = simulate_sessions([dt.date(2026, 10, 1), day], seed=3)
    j = Journal()
    eng = IntradayEngine(cfg, ReplayFeed(bars, day), "model", j, IntradayBroker(cfg, starting_cash=20000),
                         SessionRecorder(tmp_path / "data"), say=None)
    calls = []

    def gift():
        calls.append(1)
        return pd.DataFrame([{"ts": ts("2026-10-05 08:50"), "last": 22624.5, "change": 133.5, "pct": 0.59,
                              "expiry": dt.date(2026, 10, 27), "contracts": 1.0, "nifty_close": 22421.95}])
    eng.gift_source = gift
    eng.preopen(ts("2026-10-05 08:50"))
    eng.preopen(ts("2026-10-05 08:52"))                                         # within 4 minutes: not asked again
    eng.preopen(ts("2026-10-05 08:55"))
    assert len(calls) == 2 and 0.004 < eng.gift["implied_gap"] < 0.006
    assert len(pd.read_csv(tmp_path / "data" / str(day) / "gift.csv")) == 2
    ev = j.events()
    assert ev["message"].str.contains("GIFT Nifty 22,624.5").sum() == 1      # said once
