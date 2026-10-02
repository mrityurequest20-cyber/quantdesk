"""Engine semantics on hand-built bars: decisions at the close fill at the next open,
stops fill intrabar at min(open, stop), gap-throughs cancel entries, and the books
reconcile to the rupee on a full multi-strategy run."""
import numpy as np
import pandas as pd
import pytest

from quantdesk.backtest.runner import run_backtest
from quantdesk.engine.engine import Engine
from quantdesk.execution.broker import PaperBroker
from quantdesk.journal.journal import Journal
from quantdesk.strategies.base import Strategy


class Probe(Strategy):
    """Buys AAA once, at the close of `day`, with a fixed stop."""
    family = "trend"

    def __init__(self, cfg, day, stop):
        super().__init__("probe", {}, cfg)
        self.day, self.stop = pd.Timestamp(day), stop

    def universe(self):
        return ["AAA"]

    def entries(self, ctx):
        if ctx.ts == self.day and not ctx.trades_for(self.name):
            return [self.linear_intent(ctx, "AAA", +1, self.stop, "probe entry", {"why": "test"}, confidence=0.5)]
        return []


def _bars(dates, closes, opens=None, lows=None, highs=None):
    c = np.asarray(closes, dtype=float)
    o = np.asarray(opens if opens is not None else c, dtype=float)
    lo = np.asarray(lows if lows is not None else np.minimum(o, c) - 0.5, dtype=float)
    hi = np.asarray(highs if highs is not None else np.maximum(o, c) + 0.5, dtype=float)
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": 1e6}, index=dates)


def _setup(cfg, aaa: pd.DataFrame):
    dates = aaa.index
    small = cfg.with_overrides({"universe": {"indices": ["NIFTY"], "equities": ["AAA"], "pairs": []}})
    data = {"NIFTY": _bars(dates, np.full(len(dates), 20000.0)), "AAA": aaa,
            "INDIAVIX": _bars(dates, np.full(len(dates), 14.0))}
    return small, data


def _run(cfg, data, strat):
    eng = Engine(cfg, data, [strat], PaperBroker(cfg), Journal(":memory:"), warmup_bars=0)
    eng.broker.costs.eq_bps = 0.0
    eng.broker.costs.atr_frac = 0.0
    eng.run(flatten_at_end=False)
    return eng


def test_signal_at_close_fills_at_next_open(cfg):
    dates = pd.bdate_range("2026-01-05", periods=30)
    closes = np.full(30, 100.0)
    opens = closes.copy()
    opens[11] = 101.0
    s, data = _setup(cfg, _bars(dates, closes, opens))
    eng = _run(s, data, Probe(s, dates[10], stop=95.0))
    t = eng.open_trades[0]
    assert t.opened_at == dates[11]                   # never on the signal bar itself
    assert t.legs[0].entry_price == pytest.approx(101.0)
    assert t.meta["decided_at"] == str(dates[10])


def test_stop_fills_at_worse_of_open_and_stop(cfg):
    dates = pd.bdate_range("2026-01-05", periods=30)
    closes = np.full(30, 100.0)
    opens, lows = closes.copy(), closes - 0.5
    opens[14], lows[14] = 93.0, 92.0                   # gaps below the 95 stop
    s, data = _setup(cfg, _bars(dates, closes, opens, lows))
    eng = _run(s, data, Probe(s, dates[10], stop=95.0))
    t = eng.closed_trades[0]
    assert t.exit_reason == "stop" and t.closed_at == dates[14]
    assert t.legs[0].exit_price == pytest.approx(93.0)   # gap: filled at the open, not the stop
    assert t.pnl == pytest.approx(t.legs[0].qty * (93.0 - 100.0) - t.fees)


def test_gap_through_stop_cancels_entry(cfg):
    dates = pd.bdate_range("2026-01-05", periods=30)
    closes = np.full(30, 100.0)
    opens = closes.copy()
    opens[11] = 94.0
    s, data = _setup(cfg, _bars(dates, closes, opens))
    eng = _run(s, data, Probe(s, dates[10], stop=95.0))
    assert not eng.open_trades and not eng.closed_trades
    d = eng.journal.decisions()
    assert (d["action"] == "cancelled").any()


def test_books_reconcile_and_runs_are_deterministic(market, aux, cfg):
    prov, data = market
    a = run_backtest(cfg, data, start="2021-01-01", aux=aux)
    b = run_backtest(cfg, data, start="2021-01-01", aux=aux)
    start = cfg.get("account.starting_capital")
    assert a.trade_stats["trades"] > 50
    assert not a.engine.broker.positions()                                    # flattened at the end
    assert sum(t.pnl for t in a.engine.closed_trades) == pytest.approx(a.engine.broker.cash() - start, abs=0.01)
    assert a.engine.broker.fees_paid == pytest.approx(sum(t.fees for t in a.engine.closed_trades), abs=0.01)
    assert a.equity["equity"].iloc[-1] == pytest.approx(b.equity["equity"].iloc[-1], abs=1e-6)
    assert len(a.trades) == len(b.trades)
    families = set(a.trades["strategy"])
    assert {"vrp_condor", "pairs", "mean_reversion"} <= families
    assert a.trades["grade"].notna().all()


def test_engine_state_roundtrip(market, aux, cfg):
    prov, data = market
    j = Journal(":memory:")
    from quantdesk.strategies import build_strategies
    eng = Engine(cfg, data, build_strategies(cfg), PaperBroker(cfg), j, aux=aux)
    eng.run(start="2022-01-01", end="2023-06-30", flatten_at_end=False)
    eng.save_state()
    again = Engine(cfg, data, build_strategies(cfg), eng.broker, j, aux=aux)
    assert again.load_state()
    assert [t.id for t in again.open_trades] == [t.id for t in eng.open_trades]
    assert len(again.pending_entries) == len(eng.pending_entries)
    assert again.risk.peak == pytest.approx(eng.risk.peak)
