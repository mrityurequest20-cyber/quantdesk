"""Running unattended: a session split across two runners (each capped at 6 hours) must carry on
exactly where the first left off, and a real option chain that can't be reached (NSE blocks many
cloud IPs) must degrade to the model chain instead of stopping the desk."""
import datetime as dt

import pandas as pd
import pytest

import quantdesk.intraday.engine as engine_mod
from quantdesk.core.calendar import TradingCalendar
from quantdesk.intraday.engine import IntradayEngine, close_out, run_live, run_replay
from quantdesk.intraday.feeds import ReplayFeed
from quantdesk.intraday.sim import IntradayBroker
from quantdesk.intraday.synthetic import simulate_sessions
from quantdesk.journal.journal import Journal


@pytest.fixture(scope="module")
def sessions():
    from quantdesk.config import DEFAULT_CONFIG, Config
    cal = TradingCalendar(Config.load(DEFAULT_CONFIG).holidays())
    days = [d.date() for d in cal.trading_days("2026-08-17", "2026-09-28")]
    bars, _ = simulate_sessions(days, seed=5)
    return bars, days


@pytest.fixture
def sim_clock(monkeypatch):
    """run_live sleeps on the wall clock; in tests each sleep advances the replay clock a minute."""
    feeds = []
    monkeypatch.setattr(engine_mod.time, "sleep", lambda s: [f.advance() for f in feeds])
    return feeds


def _find_handover(cfg, bars, days, tmp_path):
    """A day and a minute at which the desk holds an open position (so the hand-over matters)."""
    for d in days[-8:]:
        eng = IntradayEngine(cfg, ReplayFeed(bars, d), "model", Journal(), IntradayBroker(cfg, starting_cash=500000), say=None)
        eng.start_session(d)
        while eng.feed.advance():
            eng.step()
            now = eng.feed.now()
            if eng.open_trades and dt.time(10, 30) <= now.time() <= dt.time(14, 30):
                return d, now.time()
    pytest.skip("no open position mid-session in the sample")


def test_handover_resumes_the_session(cfg, sessions, tmp_path, sim_clock):
    bars, days = sessions
    day, t_hand = _find_handover(cfg, bars, days, tmp_path)
    j = Journal(tmp_path / "j.db")
    bstate = tmp_path / "broker.json"

    feed_a = ReplayFeed(bars, day)
    sim_clock.append(feed_a)
    a = IntradayEngine(cfg, feed_a, "model", j, IntradayBroker(cfg, starting_cash=500000, state_path=bstate), say=None)
    msg = run_live(a, t_hand, handover=True)
    assert msg.startswith("handed over") and a.open_trades                      # still holding at the hand-over
    assert not any(t.exit_reason == "square_off" for t in a.closed)             # nothing was squared off early
    held = {t.id for t in a.open_trades}
    done = {t.id for t in a.closed}

    # the second runner: new process, same journal and broker state, starts a couple of minutes later
    feed_b = ReplayFeed(bars, day)
    feed_b.clock = feed_a.clock + pd.Timedelta(minutes=2)
    sim_clock[:] = [feed_b]
    b = IntradayEngine(cfg, feed_b, "model", Journal(tmp_path / "j.db"),
                       IntradayBroker(cfg, starting_cash=500000, state_path=bstate), say=None, review_dir=tmp_path / "rev")
    review = run_live(b)
    assert {t.id for t in b.closed} >= held | done                              # both halves are in the day
    assert b.day_start_equity == pytest.approx(a.day_start_equity)
    assert not b.open_trades
    tr = Journal(tmp_path / "j.db").trades()
    day_tr = tr[tr["opened_at"].str.startswith(str(day))]
    assert (day_tr["status"] == "closed").all() and len(day_tr) == len(b.closed)
    assert len(day_tr) <= cfg.get("intraday.risk.max_trades_per_day")
    for tid in held | done:
        assert tid in review                                                    # the review covers the whole day
    assert (tmp_path / "rev" / f"{day}.md").exists()


def test_handover_nobody_picked_up_is_closed_after_the_bell(cfg, sessions, tmp_path, sim_clock):
    bars, days = sessions
    day, t_hand = _find_handover(cfg, bars, days, tmp_path)
    bstate = tmp_path / "broker.json"
    feed_a = ReplayFeed(bars, day)
    sim_clock.append(feed_a)
    a = IntradayEngine(cfg, feed_a, "model", Journal(tmp_path / "j.db"),
                       IntradayBroker(cfg, starting_cash=500000, state_path=bstate), say=None)
    run_live(a, t_hand, handover=True)
    assert a.open_trades
    late = ReplayFeed(bars, day)
    late.clock = late.close_ts + pd.Timedelta(minutes=20)
    b = IntradayEngine(cfg, late, "model", Journal(tmp_path / "j.db"),
                       IntradayBroker(cfg, starting_cash=500000, state_path=bstate), say=None)
    review = run_live(b)
    assert "session review" in review and not b.open_trades
    assert (Journal(tmp_path / "j.db").trades()["status"] == "closed").all()
    # and once it's closed, a later run has nothing to do
    assert "is over" in run_live(IntradayEngine(cfg, late, "model", Journal(tmp_path / "j.db"),
                                                IntradayBroker(cfg, starting_cash=500000, state_path=bstate), say=None))


def test_kill_switch_squares_off_at_current_prices(cfg, sessions, tmp_path, sim_clock):
    bars, days = sessions
    day, t_hand = _find_handover(cfg, bars, days, tmp_path)
    bstate = tmp_path / "broker.json"
    feed_a = ReplayFeed(bars, day)
    sim_clock.append(feed_a)
    a = IntradayEngine(cfg, feed_a, "model", Journal(tmp_path / "j.db"),
                       IntradayBroker(cfg, starting_cash=500000, state_path=bstate), say=None)
    run_live(a, t_hand, handover=True)                                          # cancelled mid-run
    held = {t.id for t in a.open_trades}
    stop = ReplayFeed(bars, day)
    stop.clock = feed_a.clock
    b = IntradayEngine(cfg, stop, "model", Journal(tmp_path / "j.db"),
                       IntradayBroker(cfg, starting_cash=500000, state_path=bstate), say=None)
    close_out(b)
    tr = Journal(tmp_path / "j.db").trades().set_index("id")
    assert (tr.loc[list(held), "exit_reason"] == "manual").all() and (tr["status"] == "closed").all()
    assert b.chain_df                                                           # exits used a calibrated chain
    assert close_out(b) == "nothing open to close"
    # the operator changes their mind and restarts: the day resumes with its trades counted
    again = ReplayFeed(bars, day)
    again.clock = feed_a.clock + pd.Timedelta(minutes=30)
    sim_clock[:] = [again]
    c = IntradayEngine(cfg, again, "model", Journal(tmp_path / "j.db"),
                       IntradayBroker(cfg, starting_cash=500000, state_path=bstate), say=None)
    run_live(c)
    assert c.day_start_equity == pytest.approx(a.day_start_equity) and held <= {t.id for t in c.closed}


def test_not_a_trading_day_and_late_handover(cfg, sessions, sim_clock):
    bars, days = sessions
    sunday = ReplayFeed(bars, days[-1])
    sunday.clock = pd.Timestamp("2026-09-27 10:00", tz="Asia/Kolkata")
    eng = IntradayEngine(cfg, sunday, "model", Journal(), IntradayBroker(cfg, starting_cash=500000), say=None)
    assert "not an NSE trading day" in run_live(eng)
    f = ReplayFeed(bars, days[-1])
    f.clock = f.open_ts + pd.Timedelta(hours=4)
    eng = IntradayEngine(cfg, f, "model", Journal(), IntradayBroker(cfg, starting_cash=500000), say=None)
    assert "nothing to do" in run_live(eng, dt.time(12, 0), handover=True)


class BrokenChain:
    """Stands in for NSE when it refuses the connection."""
    name = "nse"

    def __init__(self):
        self.calls = 0

    def expiries(self, u):
        raise ConnectionError("403 Forbidden")

    def chain(self, u, expiry, spot=None, ts=None):
        self.calls += 1
        raise ConnectionError("403 Forbidden")


def test_unreachable_chain_falls_back_to_the_model(cfg, sessions):
    bars, days = sessions
    broken = BrokenChain()
    j = Journal()
    eng = IntradayEngine(cfg, ReplayFeed(bars, days[-1]), broken, j, IntradayBroker(cfg, starting_cash=500000), say=None)
    run_replay(eng)
    th = j.thoughts(str(days[-1]))
    assert not th["action"].str.contains("no option chain").any()               # it kept thinking with prices
    assert eng.chain_name() == "model (no nse)"
    warn = j.events(level="WARN")
    msgs = warn[warn["message"].str.contains("pricing off the model chain")]
    assert 1 <= len(msgs) <= 8                                                  # said so, without spamming
    # backs off: ~2×125 refreshes in a session, but only 3 quick tries then one every 15 min per underlying
    assert broken.calls <= 2 * (3 + 26)
    assert all(t.meta.get("quote_source") == "model" for t in eng.closed)
