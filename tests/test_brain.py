"""The brain: global state without look-ahead, drivers and regime, research-weighted links that vote only
when validated (in their measured direction), the gap attributed to what happened overnight, news→drivers."""
import numpy as np
import pandas as pd
import pytest

from quantdesk.intraday.brain import Brain, GlobalFeed, news_drivers
from quantdesk.intraday.news import NewsItem

IST = "Asia/Kolkata"
NOW = pd.Timestamp("2026-09-29 11:00", tz=IST)


def _daily(dates, closes):
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c}, index=pd.DatetimeIndex(dates))


def _five(start_utc, n, path):
    idx = pd.date_range(start_utc, periods=n, freq="5min", tz="UTC")
    c = np.asarray(path, dtype=float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c}, index=idx)


def world(us_last=-0.02, vix_last=0.25, europe_last=0.012, asia_today=0.05, es_drift=-0.001, day="2026-09-29"):
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2026-05-01", pd.Timestamp(day) - pd.Timedelta(days=1))
    def series(last, vol):
        r = rng.normal(0, vol, len(days))
        r[-1] = last
        return _daily(days, 100 * np.exp(np.cumsum(r)))
    daily = {"SPX": series(us_last, 0.01), "USVIX": series(vix_last, 0.06), "STOXX": series(europe_last, 0.009),
             "BRENT": series(0.0, 0.02)}
    # Tokyo's session dated *today* runs during India's morning: it must not count as a prior session
    n225 = series(0.0, 0.01)
    n225.loc[pd.Timestamp(day)] = n225["close"].iloc[-1] * np.exp(asia_today)
    daily["N225"] = n225
    # S&P futures trading through India's morning; bars after NOW exist in the feed but must be ignored
    path = 5000 * np.exp(np.cumsum(np.r_[rng.normal(0, 4e-4, 40), rng.normal(es_drift, 5e-5, 40)]))
    intraday = {"ES": _five(f"{pd.Timestamp(day).date()} 00:00", 80, path)}
    return intraday, daily


def brain_with(links=None, edges=None, **kw):
    intraday, daily = world(**kw)
    gf = GlobalFeed(None, fetch=lambda now: (intraday, daily), keys=["ES", "SPX", "USVIX", "STOXX", "BRENT", "N225"])
    gf.refresh(NOW, force=True)
    return Brain(None, gf, links or [], edges or [])


def test_no_look_ahead_in_global_prices():
    b = brain_with()
    es = b.gfeed.market("ES", NOW)
    cutoff = NOW.tz_convert("UTC")
    assert pd.Timestamp(es["last_ts"]).tz_convert("UTC") <= cutoff
    assert b.gfeed.intraday["ES"].index.max() + pd.Timedelta(minutes=5) <= cutoff
    n225 = b.gfeed.market("N225", NOW)
    assert n225["prior_date"] == "2026-09-28" and abs(n225["prior_ret"]) < 0.04        # not today's +5% Tokyo session


def test_regime_stress_and_size():
    b = brain_with(us_last=-0.03, vix_last=0.30)
    st = b.think("NIFTY", NOW)
    assert st.regime == "risk-off" and st.regime_score < -0.25
    assert st.stress >= 2 and st.size_mult < 1
    calm = brain_with(us_last=0.001, vix_last=-0.01, europe_last=0.0, es_drift=0.0).think("NIFTY", NOW)
    assert calm.size_mult == 1.0


def test_only_validated_links_vote_in_their_measured_direction():
    edges = [{"id": "G-STOXX", "symbol": "BANKNIFTY", "verdict": "EDGE", "t": -2.55, "effect_bps": -5.4},
             {"id": "G-SPX", "symbol": "NIFTY", "verdict": "NO EDGE", "t": 2.56, "effect_bps": 4.5}]
    b = brain_with(edges=edges, europe_last=0.02)                                   # Europe rallied yesterday
    bnf = b.think("BANKNIFTY", NOW)
    ev = [e for e in bnf.evidence if e["factor"] == "global_europe"]
    assert ev and ev[0]["direction"] < 0 and "faded" in ev[0]["observation"]      # BANKNIFTY fades it → bearish
    nifty = b.think("NIFTY", NOW)
    assert not nifty.evidence and "don't vote" in nifty.narrative                 # the US lead died out of sample


def test_gap_attribution_and_news():
    links = [{"from": "SPX", "to": "NIFTY", "what": "opening gap", "beta": 0.2, "corr": 0.42, "t": 20, "n": 4000, "r2": 0.18}]
    b = brain_with(links=links, us_last=-0.02)
    news = [NewsItem(NOW - pd.Timedelta(minutes=20), "T", "Crude oil prices surge 3% on Middle East tensions", sentiment=-1.0),
            NewsItem(NOW - pd.Timedelta(minutes=10), "T", "Wall Street futures slide as Treasury yields jump", sentiment=-0.8)]
    st = b.think("NIFTY", NOW, news, gap=-0.006)
    assert st.gap["explained"] == pytest.approx(0.2 * b.gfeed.market("SPX", NOW)["prior_ret"])
    assert st.gap["parts"][0][0] == "US equities"
    crude = next(d for d in st.drivers if d["id"] == "crude")
    assert crude["news_n"] == 1 and crude["news_tone"] < 0
    assert set(news_drivers("Crude oil prices surge 3% on Middle East tensions")) >= {"crude", "fear"}
    assert "rates" in news_drivers("Fed signals higher for longer as Treasury yields jump")
    assert "Global:" in st.narrative and "gap" in st.narrative


def test_engine_thinks_globally(cfg, tmp_path):
    from quantdesk.core.calendar import TradingCalendar
    from quantdesk.intraday.engine import IntradayEngine, run_replay
    from quantdesk.intraday.feeds import ReplayFeed
    from quantdesk.intraday.sim import IntradayBroker
    from quantdesk.intraday.synthetic import simulate_sessions
    from quantdesk.journal.journal import Journal
    cal = TradingCalendar(cfg.holidays())
    days = [d.date() for d in cal.trading_days("2026-08-20", "2026-09-28")]
    bars, _ = simulate_sessions(days, seed=5)
    d = days[-1]
    intraday, daily = world(day=str(d), europe_last=0.02)
    edges = [{"id": "G-STOXX", "symbol": "BANKNIFTY", "verdict": "EDGE", "t": -2.55, "effect_bps": -5.4}]
    links = [{"from": "SPX", "to": "NIFTY", "what": "opening gap", "beta": 0.2, "corr": 0.42, "t": 20, "n": 4000, "r2": 0.18}]
    gf = GlobalFeed(cfg, fetch=lambda now: (intraday, daily), keys=["ES", "SPX", "USVIX", "STOXX", "BRENT", "N225"])
    j = Journal(tmp_path / "journal.db")
    eng = IntradayEngine(cfg, ReplayFeed(bars, d), "model", j, IntradayBroker(cfg, starting_cash=20000), say=None,
                         brain=Brain(cfg, gf, links, edges))
    run_replay(eng)
    bnf = j.thoughts(str(d), "BANKNIFTY")
    import json
    ev = bnf["evidence"].map(lambda e: {x["factor"]: x["direction"] for x in json.loads(e)})
    assert ev.map(lambda e: e.get("global_europe", 0) < 0).all()                      # Europe rallied → BANKNIFTY fade lean
    nifty = j.thoughts(str(d), "NIFTY")
    assert nifty["narrative"].str.contains("Global:").all()
    assert not nifty["evidence"].str.contains("global_").any()                       # nothing validated for NIFTY
    hb = j.get_state("intraday_live")
    assert hb["global"]["markets"]["SPX"]["prior_date"] < str(d)
    assert hb["views"]["NIFTY"]["brain"]["regime"] in ("risk-on", "risk-off", "mixed")
    assert hb["views"]["NIFTY"]["brain"]["gap"]["parts"][0][0] == "US equities"
