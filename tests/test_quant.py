"""The quant layer: causal features, a model that must prove itself out of sample, a vol forecast
that scales right, and an EV engine that charges every cost and gives no free lunch."""
import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest

from quantdesk.execution.costs import CostModel
from quantdesk.intraday.chains import IntradayPricer, time_to_expiry
from quantdesk.intraday.playbook import PlanLeg, TradePlan
from quantdesk.intraday.quant import (FEATURES, MIN_PER_YEAR, DirectionModel, EVEngine, VolForecaster, features_5m,
                                      to_5m)

IST = "Asia/Kolkata"


def sessions_5m(n_days=40, seed=1, momentum=0.0):
    """Random-walk 5m sessions; `momentum` makes the next 30 minutes follow the last 30 (a planted edge)."""
    rng = np.random.default_rng(seed)
    frames, px = [], 22000.0
    for d in pd.bdate_range("2026-06-01", periods=n_days):
        idx = pd.date_range(pd.Timestamp(d.date(), tz=IST) + pd.Timedelta(hours=9, minutes=15), periods=75, freq="5min")
        r = rng.normal(0, 8e-4, 75)
        for i in range(6, 75):
            r[i] += momentum * r[i - 6:i].sum() / 6
        c = px * np.exp(np.cumsum(r))
        o = np.r_[px, c[:-1]]
        frames.append(pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.0002, "low": np.minimum(o, c) * 0.9998,
                                    "close": c, "volume": 0.0}, index=idx))
        px = c[-1] * np.exp(rng.normal(0, 3e-3))
    return pd.concat(frames)


def test_features_are_causal():
    df = sessions_5m(3, seed=4)
    full = features_5m(df)
    last_day = df[df.index.date == df.index.date[-1]]
    before = df[df.index.date < df.index.date[-1]]
    for k in (1, 2, 5, 17, 40):
        part = features_5m(pd.concat([before, last_day.iloc[:k]]))
        a, b = part[FEATURES].iloc[-1].to_numpy(), full.loc[part.index[-1], FEATURES].to_numpy()
        assert np.allclose(a, b, atol=1e-9), (k, a - b)          # a row never changes when later bars arrive


def test_model_finds_a_planted_edge_and_refuses_noise():
    edge = DirectionModel()
    d = edge.fit(features_5m(sessions_5m(45, seed=2, momentum=0.6)))
    assert edge.valid and d["auc_oos"] > 0.58, d
    noise = DirectionModel()
    d = noise.fit(features_5m(sessions_5m(45, seed=3, momentum=0.0)))
    assert not noise.valid and d["status"] == "no edge out of sample", d
    few = DirectionModel()
    assert few.fit(features_5m(sessions_5m(5, seed=5)))["status"] == "not enough history"


def test_vol_forecast_scales_and_reacts():
    rng = np.random.default_rng(0)
    idx = pd.date_range(pd.Timestamp("2026-09-28 09:15", tz=IST), periods=375, freq="1min")
    prev = pd.DatetimeIndex([t - pd.Timedelta(days=1) for t in idx])
    calm = np.exp(np.cumsum(rng.normal(0, 3e-4, 750))) * 22000
    bars = pd.DataFrame({"close": calm}, index=prev.append(idx))
    f = VolForecaster().forecast(bars, idx[0].date())
    assert f["sigma_min"] == pytest.approx(3e-4, rel=0.2) and f["sigma_30m"] == pytest.approx(f["sigma_min"] * math.sqrt(30))
    wild = bars.copy()
    wild.iloc[-100:, 0] = wild["close"].iloc[-101] * np.exp(np.cumsum(rng.normal(0, 1.5e-3, 100)))
    assert VolForecaster().forecast(wild, idx[0].date())["sigma_min"] > 1.5 * f["sigma_min"]     # today's burst counts
    with_iv = VolForecaster().forecast(bars, idx[0].date(), atm_iv=30.0)
    assert with_iv["sigma_min"] > f["sigma_min"] and with_iv["source"] == "realised+implied"


@pytest.fixture
def plan_factory(cfg):
    pr = IntradayPricer(0.065, 0.012)
    now = pd.Timestamp("2026-09-29 10:30", tz=IST)
    S, exp = 22780.0, dt.date(2026, 10, 6)
    T = time_to_expiry(now, exp)

    def leg(K, ratio, iv=13.0):
        m = pr.price(K, "CE", S, T, iv / 100)
        half = max(0.5, 0.004 * m)
        return PlanLeg(K, "CE", ratio, m + half if ratio > 0 else m - half, m, iv, pr.greeks(K, "CE", S, T, iv / 100)["delta"])

    def make(legs, hold=45):
        return TradePlan("orb", "NIFTY", 1, "x", exp, legs, 65, "t", "th", 22700, 23050, 0.30, 0.60, hold, "nse", 0.6)
    return pr, now, S, leg, make, EVEngine(cfg, pr, CostModel(cfg), n_paths=4000)


def test_ev_has_no_free_lunch_and_charges_costs(plan_factory):
    pr, now, S, leg, make, ev = plan_factory
    sig = 0.13 / math.sqrt(MIN_PER_YEAR)                          # realised vol = the option's own implied vol
    for legs in ([leg(22950, 1)], [leg(22850, 1), leg(23000, -1)]):
        r = ev.evaluate(make(legs), S, now, sig, 0.5, 240)
        costs = r["fees"] + r["exit_cost"]
        assert -2.2 * costs < r["ev"] < 0                          # no edge → you lose about the costs, not more, not less
    single, spread = make([leg(22950, 1)]), make([leg(22850, 1), leg(23000, -1)])
    evs = [ev.evaluate(single, S, now, sig, p, 240)["ev"] for p in (0.40, 0.50, 0.55, 0.60)]
    assert evs == sorted(evs)                                     # more edge, more EV
    assert ev.evaluate(spread, S, now, sig, 0.5, 240)["fees"] > 1.5 * ev.evaluate(single, S, now, sig, 0.5, 240)["fees"]
    assert ev.evaluate(single, S, now, sig * 1.5, 0.5, 240)["ev"] > ev.evaluate(single, S, now, sig * 0.8, 0.5, 240)["ev"]
    r = ev.evaluate(single, S, now, sig, 0.5, 240)
    assert r["cvar5"] >= -single.net_premium - r["fees"] - 1            # can't lose more than you paid (+ costs)
    assert ev.evaluate(single, S, now, sig, 0.5, 20)["horizon_min"] == 20    # never holds past the square-off


def test_engine_prices_every_plan(cfg, tmp_path):
    from quantdesk.core.calendar import TradingCalendar
    from quantdesk.intraday.engine import IntradayEngine, run_replay
    from quantdesk.intraday.feeds import ReplayFeed
    from quantdesk.intraday.sim import IntradayBroker
    from quantdesk.intraday.synthetic import simulate_sessions
    from quantdesk.journal.journal import Journal
    cal = TradingCalendar(cfg.holidays())
    days = [d.date() for d in cal.trading_days("2026-07-15", "2026-09-28")]
    bars, _ = simulate_sessions(days, seed=5)
    j = Journal(tmp_path / "journal.db")
    eng = IntradayEngine(cfg, ReplayFeed(bars, days[-1]), "model", j, IntradayBroker(cfg, starting_cash=20000), say=None)
    run_replay(eng)
    ev = j.events()
    assert ev["message"].str.contains("direction model").sum() == 2          # trained per underlying, result journaled
    th = j.thoughts(str(days[-1]))
    assert th["narrative"].str.contains("Quant: 30-min σ").all()
    hb = j.get_state("intraday_live")["views"]["NIFTY"]["quant"]
    assert hb["sigma_30m_pct"] > 0 and "model_status" in hb
    dec = j.df("SELECT * FROM decisions")
    tr = j.trades()
    assert len(tr) or dec["detail"].str.contains("EV").any()                 # every plan was priced
    for r in tr.itertuples():
        assert "Quant: EV" in r.rationale and "P(up)" in r.rationale


def test_research_priors_only_use_surviving_edges(plan_factory, tmp_path):
    import json
    from quantdesk.intraday.quant import load_research
    pr, now, S, leg, make, ev = plan_factory
    sig = 0.13 / math.sqrt(MIN_PER_YEAR)
    call = make([leg(22950, 1)])
    base = ev.evaluate(call, S, now, sig, 0.5, 240)["ev"]
    assert ev.evaluate(call, S, now, sig, 0.5, 240, base_drift_min=-5.7 / 1e4 / 375)["ev"] < base   # a bearish drift costs a call
    (tmp_path / "edges.json").write_text(json.dumps([
        {"id": "D1", "symbol": "NIFTY", "verdict": "EDGE", "effect_bps": -5.7, "t": -3.44, "n": 4669, "effect_holdout_bps": -4.6},
        {"id": "D1", "symbol": "BANKNIFTY", "verdict": "NO EDGE", "effect_bps": -5.0},
        {"id": "V1", "symbol": "NIFTY", "verdict": "NEEDS MARGIN", "effect_bps": 301}]))
    r = load_research(tmp_path / "edges.json")
    assert set(r) == {"NIFTY"} and r["NIFTY"]["drift"]["per_min"] == pytest.approx(-5.7 / 1e4 / 375)
    assert load_research(tmp_path / "missing.json") == {}


def test_calibration_table():
    import json
    from quantdesk.web.intraday_api import IntradayAPI
    rows = []
    for i in range(20):
        up = i % 4 != 0                                            # the index went up 75% of the time
        rows.append({"direction": 1, "entry_underlying": 100.0, "exit_underlying": 101.0 if up else 99.0, "pnl": 50.0 if up else -60.0,
                     "meta": json.dumps({"quant": {"p_up": 0.55, "p_source": "prior tilt from the analyst's score (unvalidated)"}})})
    rows.append({"direction": 1, "entry_underlying": 100.0, "exit_underlying": float("nan"), "pnl": 0.0, "meta": "{}"})
    cal = IntradayAPI._calibration(pd.DataFrame(rows))
    assert len(cal) == 1 and cal[0]["trades"] == 20 and cal[0]["bucket"] == "53–56%"
    assert cal[0]["realised"] == pytest.approx(0.75) and cal[0]["source"] == "prior tilt from the analyst's score"


def test_quant_failure_degrades_to_standing_aside(cfg, tmp_path):
    """A broken quant layer must not stop the desk thinking; it just won't trade on missing numbers."""
    from quantdesk.core.calendar import TradingCalendar
    from quantdesk.intraday.engine import IntradayEngine, run_replay
    from quantdesk.intraday.feeds import ReplayFeed
    from quantdesk.intraday.sim import IntradayBroker
    from quantdesk.intraday.synthetic import simulate_sessions
    from quantdesk.journal.journal import Journal
    cal = TradingCalendar(cfg.holidays())
    days = [d.date() for d in cal.trading_days("2026-08-20", "2026-09-28")]
    bars, _ = simulate_sessions(days, seed=5)
    j = Journal()
    eng = IntradayEngine(cfg, ReplayFeed(bars, days[-1]), "model", j, IntradayBroker(cfg, starting_cash=20000), say=None)
    eng.volf = None

    def boom(*a, **k):
        raise RuntimeError("synthetic failure")
    eng._select_by_ev = boom
    eng._quant_state = boom
    run_replay(eng)
    th = j.thoughts(str(days[-1]))
    assert len(th) > 100 and not len(j.trades())                                    # kept thinking, didn't trade blind
    errs = j.events(level="ERROR")
    assert errs["message"].str.contains("quant state failed").sum() == 2               # once per underlying, not every minute
