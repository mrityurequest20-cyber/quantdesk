"""Intraday desk: order flow, chains, features (no look-ahead), analyst, sizing, execution,
and a full synthetic session through the engine."""
import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from quantdesk.core.calendar import TradingCalendar
from quantdesk.intraday.chains import (COLUMNS, IntradayPricer, ModelOptionChain, NSEOptionChain, chain_analytics,
                                       fill_iv, time_to_expiry)
from quantdesk.intraday.engine import IntradayEngine, run_replay
from quantdesk.intraday.features import session_state
from quantdesk.intraday.feeds import IST, ReplayFeed
from quantdesk.intraday.orderflow import (FootprintBuilder, TickClassifier, Trade, approx_delta, profile_from_bars,
                                          profile_from_trades)
from quantdesk.intraday.recorder import SessionRecorder
from quantdesk.intraday.sim import IntradayBroker, QuoteMarker
from quantdesk.intraday.synthetic import simulate_sessions
from quantdesk.journal.journal import Journal


@pytest.fixture(scope="module")
def sessions():
    from quantdesk.config import DEFAULT_CONFIG, Config
    cal = TradingCalendar(Config.load(DEFAULT_CONFIG).holidays())
    days = [d.date() for d in cal.trading_days("2026-08-17", "2026-09-28")]
    bars, meta = simulate_sessions(days, seed=5)
    return bars, meta, days


def ts(s):
    return pd.Timestamp(s, tz=IST)


# ---- order flow -------------------------------------------------------------------------------------
def test_value_area_from_known_distribution():
    t = [Trade(ts("2026-09-28 10:00"), p, v) for p, v in [(100, 5), (101, 10), (102, 40), (103, 12), (104, 3), (105, 1)]]
    prof = profile_from_trades(t, tick=1.0)
    assert prof.poc == 102 and not prof.approximate
    covered = prof.volume[(prof.prices >= prof.val) & (prof.prices <= prof.vah)].sum()
    assert covered / prof.volume.sum() >= 0.70 and prof.val <= 102 <= prof.vah
    assert prof.position(106) == "above value" and prof.position(99) == "below value"


def test_profile_from_bars_conserves_volume():
    idx = pd.date_range(ts("2026-09-28 09:15"), periods=3, freq="min")
    df = pd.DataFrame({"open": [100, 101, 102], "high": [101, 103, 104], "low": [99, 100, 101], "close": [101, 102, 103],
                       "volume": [300.0, 500.0, 200.0]}, index=idx)
    prof = profile_from_bars(df, tick=0.5)
    assert prof.volume.sum() == pytest.approx(1000.0) and prof.approximate
    assert (approx_delta(df) > 0).all()                     # all bars closed near their highs


def test_tick_classifier_quote_then_tick_rule():
    c = TickClassifier()
    assert c.classify(Trade(ts("2026-09-28 10:00"), 100.05, 1, 0, 100.0, 100.05)).side == 1     # at the ask
    assert c.classify(Trade(ts("2026-09-28 10:00"), 100.00, 1, 0, 100.0, 100.05)).side == -1    # at the bid
    assert c.classify(Trade(ts("2026-09-28 10:00"), 100.10, 1)).side == 1                        # uptick
    assert c.classify(Trade(ts("2026-09-28 10:00"), 100.10, 1)).side == 1                        # zero tick inherits


def test_footprint_delta_and_imbalances():
    fb = FootprintBuilder(tick=0.05)
    t0 = ts("2026-09-28 10:00")
    for i in range(10):
        fb.add(Trade(t0 + pd.Timedelta(seconds=i), 100.05 + 0.05 * (i % 3), 10, 1))   # aggressive buying
    fb.add(Trade(t0 + pd.Timedelta(seconds=20), 100.00, 5, -1))
    bar = fb.add(Trade(t0 + pd.Timedelta(minutes=1), 100.1, 1, 1))                   # next minute closes the bar
    assert bar is not None and bar.delta == 95 and bar.volume == 105
    assert any(side == "buy" for _, side, _ in bar.imbalances)
    assert fb.cvd().iloc[-1] == 95


# ---- chains --------------------------------------------------------------------------------------------
NSE_V3 = {"records": {"timestamp": "28-Sep-2026 10:30:00", "underlyingValue": 25010.5, "data": [
    {"strikePrice": 24950, "expiryDates": "06-Oct-2026",
     "CE": {"lastPrice": 150.0, "buyPrice1": 149.5, "sellPrice1": 150.5, "impliedVolatility": 13.1, "openInterest": 1000,
            "changeinOpenInterest": 100, "totalTradedVolume": 5000, "underlyingValue": 25010.5},
     "PE": {"lastPrice": 90.0, "buyPrice1": 89.5, "sellPrice1": 90.5, "impliedVolatility": 13.9, "openInterest": 3000,
            "changeinOpenInterest": 900, "totalTradedVolume": 7000}},
    {"strikePrice": 25000, "expiryDates": "06-Oct-2026",
     "CE": {"lastPrice": 120.0, "buyPrice1": 119.0, "sellPrice1": 121.0, "impliedVolatility": 0, "openInterest": 5000,
            "changeinOpenInterest": 800, "totalTradedVolume": 9000},
     "PE": {"lastPrice": 110.0, "buyPrice1": 109.5, "sellPrice1": 110.5, "impliedVolatility": 13.5, "openInterest": 4000,
            "changeinOpenInterest": 400, "totalTradedVolume": 8000}},
    {"strikePrice": 25000, "expiryDates": "13-Oct-2026", "CE": {"lastPrice": 999.0}},
]}}


def test_nse_v3_parser_and_analytics():
    df = NSEOptionChain.parse(NSE_V3, "NIFTY", dt.date(2026, 10, 6))
    assert list(df.columns) == COLUMNS and list(df.index) == [24950.0, 25000.0]     # other expiry filtered out
    assert df.attrs["spot"] == 25010.5 and df.attrs["ts"] == ts("2026-09-28 10:30")
    assert np.isnan(df.at[25000.0, "ce_iv"])                                          # NSE "0" means no IV
    df = fill_iv(df, IntradayPricer())
    assert 5 < df.at[25000.0, "ce_iv"] < 40
    a = chain_analytics(df)
    assert a["atm_strike"] == 25000 and a["pcr_oi"] == pytest.approx(7000 / 6000)
    assert a["top_put_adds"][0] == 24950.0 and a["max_pain"] in (24950.0, 25000.0)


def test_time_to_expiry_is_minute_precise():
    e = dt.date(2026, 9, 29)
    assert time_to_expiry(ts("2026-09-29 15:30"), e) == 0
    assert time_to_expiry(ts("2026-09-29 09:30"), e) == pytest.approx(6 / 24 / 365)


def test_marker_uses_quote_iv_and_spread(cfg, sessions):
    bars, _, days = sessions
    cal = TradingCalendar(cfg.holidays())
    now = ts(f"{days[-1]} 11:00")
    mc = ModelOptionChain(cfg, cal, lambda u, t: (25000.0, 0.13))
    exp = mc.expiries("NIFTY", now)[1]
    ch = mc.chain("NIFTY", exp, ts=now)
    mk = QuoteMarker(IntradayPricer())
    mk.calibrate(ch, 65)
    from quantdesk.core.types import Instrument
    inst = Instrument.option("NIFTY", exp, 25000, "CE", 65)
    row = ch.loc[25000.0]
    assert mk.mid(inst, 25000.0, now) == pytest.approx((row.ce_bid + row.ce_ask) / 2, rel=0.02)
    b, a = mk.bid_ask(inst, 25000.0, now)
    assert a > b and mk.exit_price(inst, 65, 25000.0, now) == b and mk.exit_price(inst, -65, 25000.0, now) == a
    assert mk.mid(inst, 25100.0, now) > mk.mid(inst, 25000.0, now)                      # delta shows up


def test_prices_use_the_iv_the_quote_implies_not_the_printed_one(cfg, sessions):
    """29 Sep 2026: NSE printed 14.7% for a put whose ask implied 14.05%; valuing it at the printed IV made it
    'worth' ₹92 the moment it was bought for ₹85 (phantom EV at entry, phantom P&L at the mark)."""
    from quantdesk.core.types import Instrument
    from quantdesk.intraday.playbook import StrikePicker
    _, _, days = sessions
    cal = TradingCalendar(cfg.holidays())
    now = ts(f"{days[-1]} 11:00")
    mc = ModelOptionChain(cfg, cal, lambda u, t: (25000.0, 0.13))
    exp = mc.expiries("NIFTY", now)[1]
    ch = mc.chain("NIFTY", exp, ts=now)
    ch["ce_iv"] += 1.5                                          # the exchange's figure disagrees with its own quotes
    ch["pe_iv"] += 1.5
    mk = QuoteMarker(IntradayPricer())
    mk.calibrate(ch, 65)
    for K in (24800.0, 25000.0, 25200.0):
        row = ch.loc[K]
        inst = Instrument.option("NIFTY", exp, K, "PE", 65)
        assert mk.mid(inst, 25000.0, now) == pytest.approx((row.pe_bid + row.pe_ask) / 2, rel=0.003)
    rows = StrikePicker(IntradayPricer()).rows(ch, "PE", now).set_index("strike")
    pr, T = IntradayPricer(), time_to_expiry(now, exp)
    for K in (24800.0, 25000.0):
        assert pr.price(K, "PE", 25000.0, T, rows.at[K, "iv"] / 100) == pytest.approx(rows.at[K, "mid"], rel=0.003)


def test_a_stop_inside_one_minutes_noise_is_pushed_out(cfg, sessions):
    from quantdesk.intraday.analyst import MarketView
    from quantdesk.intraday.playbook import Playbook
    _, _, days = sessions
    cal = TradingCalendar(cfg.holidays())
    now = ts(f"{days[-1]} 11:00")
    mc = ModelOptionChain(cfg, cal, lambda u, t: (25000.0, 0.13))
    ch = mc.chain("NIFTY", mc.expiries("NIFTY", now)[0], ts=now)
    view = MarketView("NIFTY", now, 25000.0, "bearish", -0.8, 0.8, "trend", "fair", 13.0, 12.0, [], [], {}, "",
                      state={"atr5": 22.0})
    plan = Playbook(cfg, IntradayPricer())._directional("vwap_trend", view, ch, now, -1, "t", "thesis.", 25003.0, 24980.0)
    floor = cfg.get("intraday.risk.min_stop_atr5") * 22.0
    assert plan.invalidation == pytest.approx(25000.0 + floor)            # 3 pts → 16.5 pts
    assert 25000.0 - plan.target_underlying >= 1.5 * floor - 1e-9 and "floored" in plan.thesis
    wide = Playbook(cfg, IntradayPricer())._directional("vwap_trend", view, ch, now, -1, "t", "thesis.", 25040.0, 24920.0)
    assert wide.invalidation == 25040.0 and wide.target_underlying == 24920.0      # a sane stop is left alone


def test_intraday_broker_fills_at_quote_plus_ticks(cfg):
    from quantdesk.core.types import Instrument, Order
    br = IntradayBroker(cfg, starting_cash=500000, adverse_ticks=1)
    inst = Instrument.option("NIFTY", dt.date(2026, 10, 6), 25000, "CE", 65)
    f = br.execute(Order(inst, 65, "t", "open"), 120.0, ts("2026-09-28 10:00"))
    assert f.price == pytest.approx(120.05)
    f2 = br.execute(Order(inst, -65, "t", "close"), 130.0, ts("2026-09-28 11:00"))
    assert f2.price == pytest.approx(129.95) and f2.fee_breakdown["stt"] == pytest.approx(65 * 129.95 * 0.0015)


# ---- features: no look-ahead ------------------------------------------------------------------------------
def test_session_state_is_causal(sessions):
    bars, _, days = sessions
    day = days[-1]
    full = bars["NIFTY"]
    for hhmm in ("09:40", "11:05", "14:20"):
        now = ts(f"{day} {hhmm}")
        seen = full[full.index + pd.Timedelta(minutes=1) <= now]
        a = session_state(seen, now)
        b = session_state(seen, now, cache={})
        later = full[full.index + pd.Timedelta(minutes=1) <= now + pd.Timedelta(minutes=30)]
        c = session_state(later[later.index + pd.Timedelta(minutes=1) <= now], now)
        for k, v in a.items():
            if isinstance(v, float):
                assert (np.isnan(v) and np.isnan(b[k])) or v == pytest.approx(b[k]), k
                assert (np.isnan(v) and np.isnan(c[k])) or v == pytest.approx(c[k]), k
        assert a["bars_today"] == int((now - ts(f"{day} 09:15")).total_seconds() // 60)


def test_replay_feed_never_shows_the_forming_bar(sessions):
    bars, _, days = sessions
    f = ReplayFeed(bars, days[-1])
    f.advance(10)
    got = f.poll("NIFTY", None)
    assert len(got) == 10 and got.index[-1] == ts(f"{days[-1]} 09:24")
    assert f.history("NIFTY", 3).index.max() < ts(f"{days[-1]} 09:15")


# ---- engine ------------------------------------------------------------------------------------------------
def test_full_session_replay(cfg, sessions, tmp_path):
    bars, meta, days = sessions
    j = Journal(tmp_path / "j.db")
    br = IntradayBroker(cfg, starting_cash=500000, state_path=tmp_path / "b.json")
    rec = SessionRecorder(tmp_path / "data")
    reviews = tmp_path / "reviews"
    totals = []
    for d in days[-3:]:
        eng = IntradayEngine(cfg, ReplayFeed(bars, d), "model", j, br, rec, say=None, review_dir=reviews)
        run_replay(eng)
        totals.append(len(eng.closed))
        assert not eng.open_trades                                             # flat at the close
    tr = j.trades()
    assert len(tr) == sum(totals) and (tr["status"] == "closed").all()
    th = j.thoughts()
    assert len(th) > 100 and set(th["symbol"]) == {"NIFTY", "BANKNIFTY"}
    assert th["narrative"].str.contains("bias").all()
    ev = json.loads(th.iloc[-1]["evidence"])
    assert ev and {"factor", "direction", "weight", "observation"} <= set(ev[0])
    if len(tr):
        t = tr.iloc[0]
        assert "Trigger:" in t["rationale"] and "Thesis:" in t["rationale"] and "Market read:" in t["rationale"]
        opened = pd.to_datetime(tr["opened_at"])
        assert (opened.dt.time >= dt.time(9, 20)).all() and (opened.dt.time <= dt.time(14, 45)).all()
        assert (pd.to_datetime(tr["closed_at"]).dt.time <= dt.time(15, 16)).all()
        assert tr["grade"].notna().all()
        assert (tr.groupby(tr["opened_at"].str[:10]).size() <= cfg.get("intraday.risk.max_trades_per_day")).all()
    fills = j.df("SELECT * FROM fills")
    assert br.cash() == pytest.approx(500000 - (fills["qty"] * fills["price"]).sum() - fills["fees"].sum(), abs=0.01)
    assert sum(tr["pnl"]) == pytest.approx(br.cash() - 500000, abs=0.01)
    assert sorted(p.stem for p in reviews.glob("*.md")) == [str(d) for d in days[-3:]]
    assert (tmp_path / "data" / str(days[-1]) / "NIFTY_1m.csv").exists()


def test_risk_gates(cfg):
    from quantdesk.intraday.risk import IntradayRisk
    r = IntradayRisk(cfg)
    r.reset(dt.date(2026, 9, 28), 500000)
    assert r.gate(ts("2026-09-28 09:17"), 500000, [], "NIFTY")                         # before 09:20
    assert not r.gate(ts("2026-09-28 10:00"), 500000, [], "NIFTY")
    below = 500000 * (1 - cfg.get("intraday.risk.daily_loss_limit") - 0.005)
    assert not any("daily loss" in x for x in r.gate(ts("2026-09-28 10:00"), 500000 * (1 - cfg.get("intraday.risk.daily_loss_limit") + 0.005), [], "NIFTY"))
    assert any("daily loss" in x for x in r.gate(ts("2026-09-28 10:00"), below, [], "NIFTY"))
    r2 = IntradayRisk(cfg)
    r2.reset(dt.date(2026, 9, 28), 500000)
    r2.on_close(-1000, ts("2026-09-28 10:00"))
    r2.on_close(-1000, ts("2026-09-28 10:10"))
    assert any("cooling off" in x for x in r2.gate(ts("2026-09-28 10:20"), 498000, [], "NIFTY"))
    assert not r2.gate(ts("2026-09-28 10:45"), 498000, [], "NIFTY")
