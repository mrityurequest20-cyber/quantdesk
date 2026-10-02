"""A small account: the configured capital, sizing that respects what one lot really costs,
and an account that only resets itself while it has no history."""
import datetime as dt

import pandas as pd
import pytest

from quantdesk.intraday.account import ensure_account, reset_account
from quantdesk.intraday.playbook import PlanLeg, TradePlan
from quantdesk.intraday.risk import IntradayRisk
from quantdesk.journal.journal import Journal


def plan(symbol, lot, legs, conviction=0.8):
    return TradePlan("orb", symbol, 1, "bull_call_spread", dt.date(2026, 10, 6), legs, lot, "t", "th", None, None,
                     0.30, 0.60, 45, "nse", conviction)


def test_small_account_sizing(cfg):
    assert cfg.get("intraday.capital") == 20000
    r = IntradayRisk(cfg)
    # NIFTY 22850/23000 call spread for ~61 → ₹3,955 a lot, ~₹1.2k to the stop: one lot at good conviction
    nifty = plan("NIFTY", 65, [PlanLeg(22850, "CE", 1, 120.0, 119, 13, 0.45), PlanLeg(23000, "CE", -1, 59.2, 60, 13, 0.30)])
    lots, notes = r.size(nifty, 20000, 20000)
    assert lots == 1, notes
    # low conviction: the 8% budget × 0.6 scale can't cover a lot's risk → no trade
    assert r.size(plan("NIFTY", 65, nifty.legs, conviction=0.0), 20000, 20000)[0] == 0
    # BANKNIFTY monthly spread ~₹9.4k a lot and ~₹2.8k to the stop: priced out of a ₹20k account
    bnf = plan("BANKNIFTY", 30, [PlanLeg(51000, "CE", 1, 700.0, 700, 15, 0.45), PlanLeg(51800, "CE", -1, 387.0, 387, 15, 0.30)])
    assert r.size(bnf, 20000, 20000)[0] == 0
    # never pay more than the cash on hand
    lots, notes = r.size(nifty, 20000, 3000)
    assert lots == 0 and "binding: cash" in notes[-1]
    # credit spreads need SPAN-sized margin, not just their max loss
    credit = TradePlan("range_sell", "NIFTY", 0, "iron_condor", dt.date(2026, 10, 6),
                       [PlanLeg(23100, "CE", -1, 40.0, 40, 13, 0.2), PlanLeg(23200, "CE", 1, 20.0, 20, 13, 0.1)],
                       65, "t", "th", None, None, 0.8, 0.3, 120, "nse", 0.8)
    lots, notes = r.size(credit, 20000, 20000)
    assert lots == 0 and "binding: margin" in notes[-1]


def test_account_resets_only_without_history(cfg, tmp_path):
    j = Journal(tmp_path / "journal.db")
    broker = tmp_path / "broker.json"
    broker.write_text('{"cash": 500000}')
    assert ensure_account(cfg, j, broker, say=None) == 20000 and not broker.exists()   # no trades: take the config
    assert j.get_state("intraday_account")["capital"] == 20000
    # after trading, a config change doesn't silently rewrite history
    j.db.execute("INSERT INTO trades (id, status, opened_at) VALUES ('T1', 'closed', '2026-09-29 10:00')")
    j.commit()
    cfg2 = cfg.__class__(dict(cfg.data, intraday=dict(cfg.data["intraday"], capital=100000)))
    msgs = []
    assert ensure_account(cfg2, j, broker, say=msgs.append) == 20000 and "reset-account" in msgs[0]
    # an explicit reset archives it
    base = tmp_path
    moved = reset_account(base)
    assert moved and (moved / "journal.db").exists() and not (base / "journal.db").exists()


def test_restating_a_mispriced_exit_moves_pnl_cash_and_leaves_a_trail(cfg, tmp_path):
    """29 Sep 2026: an exit marked at ₹88.62 (NSE's printed IV) is restated at ₹81.20 (the quote's own IV)."""
    import json

    from quantdesk.core.types import Instrument
    from quantdesk.intraday.account import restate_trade
    from quantdesk.intraday.sim import IntradayBroker
    from quantdesk.journal.journal import trade_from_dict
    inst = Instrument.option("NIFTY", dt.date(2026, 10, 6), 22400.0, "PE", 65)
    costs = IntradayBroker(cfg).costs
    fin, bin_ = costs.fees(inst, 65, 85.10)
    fout, bout = costs.fees(inst, -65, 88.62)
    t = trade_from_dict({"id": "T1", "strategy": "vwap_trend", "family": "intraday", "symbol": "NIFTY", "direction": -1,
                         "kind": "options", "units": 1, "opened_at": "2026-09-29 11:08:04+05:30", "entry_underlying": 22606.7,
                         "initial_risk": 1658.475, "legs": [{"instrument": inst.to_dict(), "qty": 65, "entry_price": 85.10,
                                                             "exit_price": 88.62}]})
    j = Journal(tmp_path / "journal.db")
    j.open_trade(t)
    j.fill("2026-09-29 11:08:04+05:30", "T1", inst.symbol, 65, 85.10, fin, bin_)
    t.closed_at, t.exit_underlying, t.exit_reason, t.exit_note = pd.Timestamp("2026-09-29 11:10:04+05:30"), 22616.8, "invalidation", "x"
    t.fees = fin + fout
    t.pnl = (88.62 - 85.10) * 65 - t.fees
    j.close_trade(t)
    j.fill("2026-09-29 11:10:04+05:30", "T1", inst.symbol, -65, 88.62, fout, bout)
    j.set_state("intraday_live", {"day": "2026-09-29", "equity": 20000 + t.pnl, "day_pnl": t.pnl})
    j.commit()
    (tmp_path / "broker.json").write_text(json.dumps({"cash": 20000 + t.pnl, "positions": {}, "fees_paid": t.fees, "fee_breakdown": {}}))
    old = t.pnl
    r = restate_trade(cfg, tmp_path, "T1", {inst.symbol: 81.20}, "exit re-priced at the quote's own IV")
    fnew = costs.fees(inst, -65, 81.20)[0]
    assert r["new_pnl"] == pytest.approx((81.20 - 85.10) * 65 - fin - fnew) and r["new_pnl"] < 0 < old
    j2 = Journal(tmp_path / "journal.db")
    row = j2.df("SELECT pnl, r_multiple, exit_note FROM trades WHERE id='T1'").iloc[0]
    assert row["pnl"] == pytest.approx(r["new_pnl"]) and row["r_multiple"] == pytest.approx(r["new_pnl"] / 1658.475)
    assert "₹88.62 → ₹81.20" in row["exit_note"]                                   # the old price stays on the record
    assert json.loads((tmp_path / "broker.json").read_text())["cash"] == pytest.approx(20000 + r["new_pnl"])
    assert j2.get_state("intraday_live")["equity"] == pytest.approx(20000 + r["new_pnl"])
    assert j2.df("SELECT * FROM fills WHERE qty < 0")["price"].iloc[0] == 81.20
    assert "restated" in j2.df("SELECT message FROM events WHERE category='restatement'")["message"].iloc[0]
    with pytest.raises(KeyError):
        restate_trade(cfg, tmp_path, "T1", {"NOPE": 1.0}, "wrong leg")


def test_fill_audit_compares_paper_fills_with_recorded_quotes(cfg, tmp_path):
    """deploy/audit_fills.py: a fill at the recorded ask + a tick is fair; an exit above the recorded bid is flagged."""
    import importlib.util
    import json
    from pathlib import Path

    from quantdesk.core.calendar import TradingCalendar
    from quantdesk.core.types import Instrument
    from quantdesk.intraday.chains import ModelOptionChain
    from quantdesk.intraday.recorder import SessionRecorder
    from quantdesk.journal.journal import trade_from_dict
    spec = importlib.util.spec_from_file_location("audit_fills", Path(__file__).resolve().parent.parent / "deploy" / "audit_fills.py")
    af = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(af)
    cal = TradingCalendar(cfg.holidays())
    t_in, t_out = pd.Timestamp("2026-09-29 11:08", tz="Asia/Kolkata"), pd.Timestamp("2026-09-29 11:10", tz="Asia/Kolkata")
    mc = ModelOptionChain(cfg, cal, lambda u, t: (22600.0, 0.14))
    exp = mc.expiries("NIFTY", t_in)[1]
    rec = SessionRecorder(tmp_path / "data")
    for ts in (t_in - pd.Timedelta(minutes=1), t_out - pd.Timedelta(minutes=1)):
        rec.record_chain(mc.chain("NIFTY", exp, ts=ts))
    snap_in, snap_out = (mc.chain("NIFTY", exp, ts=ts) for ts in (t_in - pd.Timedelta(minutes=1), t_out - pd.Timedelta(minutes=1)))
    K = 22400.0
    ask_in, bid_out = float(snap_in.at[K, "pe_ask"]), float(snap_out.at[K, "pe_bid"])
    inst = Instrument.option("NIFTY", exp, K, "PE", 65)
    j = Journal(tmp_path / "intraday" / "journal.db")
    t = trade_from_dict({"id": "T1", "strategy": "vwap_trend", "family": "intraday", "symbol": "NIFTY", "direction": -1,
                         "kind": "options", "units": 1, "opened_at": str(t_in), "entry_underlying": 22600.0, "initial_risk": 1000.0,
                         "legs": [{"instrument": inst.to_dict(), "qty": 65, "entry_price": ask_in + 0.05, "exit_price": bid_out + 7}]})
    j.open_trade(t)
    j.fill(str(t_in), "T1", inst.symbol, 65, ask_in + 0.05, 25.0, {})
    t.closed_at, t.exit_underlying, t.exit_reason, t.pnl, t.fees = t_out, 22610.0, "invalidation", 999.0, 50.0
    j.close_trade(t)
    j.fill(str(t_out), "T1", inst.symbol, -65, bid_out + 7, 25.0, {})          # marked 7 above the market's bid
    j.commit()
    [r] = af.audit(tmp_path, [tmp_path / "data"])
    entry, exit_ = r["fills"]
    assert entry["better_than_market"] == pytest.approx(0, abs=0.01)
    assert exit_["better_than_market"] == pytest.approx((7 + 0.05) * 65, abs=0.5)
    assert r["pnl_at_quotes"] < 0 < r["pnl_booked"]            # bought at the ask, sold at the bid a minute later: a loss
