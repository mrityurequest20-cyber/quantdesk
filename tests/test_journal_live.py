import datetime as dt

import pandas as pd
import pytest

from quantdesk.cli import build_parser
from quantdesk.core.types import Instrument, Trade, TradeLeg
from quantdesk.engine.live import LiveRunner
from quantdesk.execution.broker import PaperBroker
from quantdesk.journal.review import review_trade


def _trade(pnl, mfe, mae, reason, risk=10_000.0, bars=10):
    t = Trade(id="x", strategy="trend_rider", family="trend", symbol="INFY", direction=1, kind="linear",
              legs=[TradeLeg(Instrument.equity("INFY"), 100, 1500.0, 1500.0 + pnl / 100)], units=100,
              opened_at=pd.Timestamp("2026-01-05"), entry_underlying=1500.0, initial_risk=risk, stop=1400.0,
              context={"regime": "trending_up"})
    t.pnl, t.mfe, t.mae, t.exit_reason, t.bars_held = pnl, mfe, mae, reason, bars
    return t


def test_review_rewards_process_over_outcome():
    planned_loss = review_trade(_trade(-10_000, 1_000, -10_000, "stop"))
    lucky_forced = review_trade(_trade(15_000, 16_000, -9_500, "manual"))
    assert planned_loss["grade"] in ("B", "C")
    assert planned_loss["process_score"] > lucky_forced["process_score"]
    assert any("forced" in l for l in lucky_forced["lessons"])


def test_review_lessons():
    rv = review_trade(_trade(-2_000, 15_000, -4_000, "trend_reversal"), regime_at_exit="range")
    text = " ".join(rv["lessons"])
    assert "Round-tripped" in text and "Regime changed" in text
    quick = review_trade(_trade(-10_000, 0, -10_000, "stop", bars=1))
    assert any("within 2 bars" in l for l in quick["lessons"])


def test_paper_runner_daily_cycle_is_idempotent(market, cfg, tmp_path):
    prov, data = market
    pcfg = cfg.with_overrides({"data": {"history_years": 3}})
    bench = data["NIFTY"].index
    days = [d.date() for d in bench[-6:]]

    def runner():
        return LiveRunner(pcfg, broker=PaperBroker(pcfg, state_path=tmp_path / "b.json"),
                          journal_path=tmp_path / "j.db", provider=prov)

    r = runner()
    for d in days:
        r.engine = None
        pre = r.premarket(d)
        assert {c.name for c in pre} >= {"data_freshness", "reconciliation", "kill_switch", "risk_state"}
        out = r.run_eod(d)
        assert out["processed"] == [str(d)]
        assert next(c for c in out["post"] if c.name == "reconciliation").status == "PASS"
    again = runner().run_eod(days[-1])
    assert again["processed"] == []                                    # never processes a day twice
    status = runner().load(days[-1])
    assert len(status.open_trades) == out["open_trades"]
    review = r.review(7, days[-1])
    assert review.startswith("# 7-day review")


def test_cli_parser_and_schedule(capsys, cfg):
    from quantdesk.cli import cmd_schedule
    a = build_parser().parse_args(["--source", "synthetic", "backtest", "--strategies", "pairs", "--mc"])
    assert a.cmd == "backtest" and a.strategies == "pairs" and a.mc
    cmd_schedule(cfg, a)
    out = capsys.readouterr().out
    assert "CRON_TZ=Asia/Kolkata" in out and "paper premarket" in out and "paper run" in out


def test_live_broker_refuses_without_explicit_opt_in(cfg):
    from quantdesk.execution.kite import KiteBroker, LiveTradingDisabled
    with pytest.raises(LiveTradingDisabled):
        KiteBroker(cfg, confirm_live=True)                              # config still says paper
    live = cfg.with_overrides({"account": {"mode": "live"}})
    with pytest.raises(LiveTradingDisabled):
        KiteBroker(live, confirm_live=False)                            # no --live flag
