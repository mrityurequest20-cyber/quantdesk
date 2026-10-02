"""Paper / live operation on daily bars.

`LiveRunner.run_eod()` is idempotent and catches up: it processes every trading day
since the last processed one, each exactly like one backtest bar (execute queued orders
at that day's open, then the close routine). Run it once a day after the close (cron),
and the paper account evolves exactly as the backtest says it would — which makes
paper-vs-backtest drift a meaningful health signal rather than noise.

With the Kite broker (live), `run_open()` executes the queued orders at 09:20 using live
quotes, and `run_eod()` then only runs the close routine for the day.
"""
from __future__ import annotations

import datetime as dt
import logging
from types import SimpleNamespace

import pandas as pd

from ..data import make_provider
from ..execution.broker import PaperBroker
from ..journal.journal import Journal
from ..journal.review import period_review
from ..ops.checks import Routines
from ..strategies import build_strategies
from .engine import Engine

log = logging.getLogger(__name__)


class LiveRunner:
    def __init__(self, cfg, source: str | None = None, broker=None, journal_path=None, provider=None):
        self.cfg = cfg
        rt = cfg.runtime_dir
        self.journal = Journal(journal_path or rt / "journal.db")
        self.broker = broker or PaperBroker(cfg, state_path=rt / "paper_broker.json")
        self.provider = provider or make_provider(cfg, source)
        self.engine: Engine | None = None
        self.data: dict[str, pd.DataFrame] = {}

    def load(self, asof: dt.date | None = None) -> Engine:
        years = self.cfg.get("data.history_years", 8)
        end = pd.Timestamp(asof or dt.date.today())
        start = end - pd.DateOffset(years=years)
        self.data = self.provider.universe(self.cfg.all_symbols(), start, end)
        eng = Engine(self.cfg, self.data, build_strategies(self.cfg), self.broker, self.journal, persist_marks=True)
        eng.load_state()
        closed = self.journal.df("SELECT strategy, r_multiple FROM trades WHERE status='closed' ORDER BY closed_at")
        eng.closed_trades = [SimpleNamespace(strategy=r.strategy, r_multiple=r.r_multiple) for r in closed.itertuples()]
        if not eng.curve:
            eng.curve = [{"ts": None, "equity": self.broker.cash()}]
        self.engine = eng
        return eng

    def premarket(self, today: dt.date | None = None) -> list:
        today = today or dt.date.today()
        eng = self.engine or self.load(today)
        if eng.timeline is not None and len(eng.timeline):
            eng.ctx.ts = eng.timeline[-1]
        r = Routines(self.cfg, eng, self.data, self.journal)
        res = r.premarket(today)
        r.record("premarket", res, ts=pd.Timestamp(today) + pd.Timedelta(hours=8, minutes=40))
        blocking = r.blocking(res)
        if blocking:
            self.journal.set_state("entry_block", {"date": str(today), "reasons": [b.detail for b in blocking]})
        return res

    def run_open(self, today: dt.date | None = None) -> dict:
        """Live only: execute queued exits/entries at ~09:20 against live quotes."""
        if not getattr(self.broker, "live", False):
            raise RuntimeError("paper mode executes queued orders inside run_eod; run_open is for a live broker")
        today = today or dt.date.today()
        eng = self.load(today)
        from ..core.types import Instrument
        need = [l.instrument for pe in eng.pending_entries for l in pe["intent"].legs]
        need += [l.instrument for t in eng.open_trades for l in t.legs]
        need += [Instrument(symbol=pe["intent"].symbol, kind="index", underlying=pe["intent"].symbol)
                 for pe in eng.pending_entries if pe["intent"].symbol in self.cfg.symbols("indices")]
        eng.quote_override = self.broker.quotes(need)
        ts = pd.Timestamp(today)
        eng.on_open(ts)
        eng.quote_override = {}
        eng.save_state()
        self.journal.set_state("opened", str(today))
        return {"executed_for": str(today), "open_trades": len(eng.open_trades)}

    def run_eod(self, asof: dt.date | None = None) -> dict:
        """Process every unprocessed trading day up to `asof` (inclusive)."""
        asof = asof or dt.date.today()
        eng = self.load(asof)
        last = self.journal.get_state("last_processed")
        tl = eng.timeline
        todo = [k for k, ts in enumerate(tl) if ts.date() <= asof and (last is None or ts > pd.Timestamp(last))]
        if last is None:
            todo = todo[-1:]                      # first run: start trading from the latest bar, no replay
        processed = []
        block = self.journal.get_state("entry_block") or {}
        opened_live = self.journal.get_state("opened")
        for k in todo:
            ts = tl[k]
            if (last is not None or processed) and opened_live != str(ts.date()):
                eng.on_open(ts)
            else:
                eng.ctx.ts = ts
                eng.risk.new_bar(ts, self.broker.cash())
            if block.get("date") == str(ts.date()):
                eng.risk.entries_blocked, eng.risk.block_reason = True, "pre-market checks failed: " + "; ".join(block["reasons"])
            snap = eng.on_close(ts, k)
            processed.append(str(ts.date()))
            self.journal.set_state("last_processed", str(ts))
        eng.save_state()
        r = Routines(self.cfg, eng, self.data, self.journal)
        post = r.postmarket(asof)
        r.record("postmarket", post, ts=pd.Timestamp(asof) + pd.Timedelta(hours=16, minutes=45))
        self.journal.commit()
        return {"processed": processed, "post": post, "equity": eng.curve[-1]["equity"] if eng.curve else None,
                "open_trades": len(eng.open_trades), "queued": len(eng.pending_entries)}

    # ---- manual trading from the chart (paper account only) ------------------------------------
    def _guard_manual(self):
        if getattr(self.broker, "live", False):
            raise PermissionError("chart orders are paper-only; live orders go through the routines")
        if (self.cfg.runtime_dir / "KILL").exists():
            raise PermissionError("kill switch engaged")
        eng = self.engine or self.load()
        ok, why = eng.risk.can_enter()
        return eng, ok, why

    def manual_order(self, symbol: str, side: str, qty: int, stop: float | None = None,
                     target: float | None = None, note: str = "") -> dict:
        """Market order at the latest price. qty = shares for equities, lots for index futures."""
        from ..core.types import LINEAR, Instrument, Trade, TradeLeg, new_trade_id
        eng, ok, why = self._guard_manual()
        if not ok:
            raise PermissionError(why)
        if symbol not in eng.bars or qty <= 0:
            raise ValueError(f"unknown symbol or bad quantity: {symbol} x{qty}")
        d = 1 if side.lower().startswith("b") else -1
        spec = self.cfg.instrument_spec(symbol)
        inst = Instrument.future(symbol, int(spec.get("lot_size", 1))) if spec.get("kind") == "index" else Instrument.equity(symbol)
        ts = eng.timeline[-1]
        px = float(eng.bars[symbol].at(ts, "close"))
        units = int(qty)
        risk = abs(px - stop) * units * inst.lot_size if stop else 0.02 * px * units * inst.lot_size
        t = Trade(id=new_trade_id("M"), strategy="manual", family="manual", symbol=symbol, direction=d, kind=LINEAR,
                  legs=[], units=units, opened_at=ts, entry_underlying=px, initial_risk=risk, stop=stop, target=target,
                  rationale=note or "Manual order placed from the chart.", context={"regime": eng.ctx.regime()})
        fill = eng._fill(t, inst, d * units * inst.lot_size, px, ts, "open")
        if fill is None:
            raise RuntimeError("broker rejected the order")
        t.legs.append(TradeLeg(inst, fill.qty, fill.price))
        t.fees, t.pnl = fill.fees, -fill.fees
        t.last_mark = {inst.symbol: fill.price}
        eng.open_trades.append(t)
        self.journal.open_trade(t, ["manual order from the chart; sized by the user, gated by the kill switch and risk state"])
        eng.save_state()
        return {"trade_id": t.id, "price": fill.price, "qty": fill.qty, "fees": fill.fees}

    def manual_close(self, trade_id: str) -> dict:
        """Always allowed (even with the kill switch on): closing reduces risk."""
        if getattr(self.broker, "live", False):
            raise PermissionError("chart orders are paper-only")
        eng = self.engine or self.load()
        t = next((x for x in eng.open_trades if x.id == trade_id.split(":")[0]), None)
        if t is None:
            raise KeyError(f"no open trade {trade_id}")
        ts = eng.timeline[-1]
        eng._close(t, ts, "manual", "closed from the chart", field="close")
        eng.save_state()
        return {"trade_id": t.id, "pnl": t.pnl}

    def manual_modify(self, trade_id: str, stop: float | None = None, target: float | None = None) -> dict:
        eng = self.engine or self.load()
        t = next((x for x in eng.open_trades if x.id == trade_id.split(":")[0]), None)
        if t is None:
            raise KeyError(f"no open trade {trade_id}")
        if stop is not None:
            t.stop = float(stop)
        if target is not None:
            t.target = float(target)
        self.journal.update_open_trade(t)
        self.journal.event(pd.Timestamp.now(), "INFO", "manual", f"{t.id} stop/target set to {t.stop}/{t.target} from the chart")
        eng.save_state()
        return {"trade_id": t.id, "stop": t.stop, "target": t.target}

    def cancel_queued(self, key: str) -> dict:
        eng = self.engine or self.load()
        before = len(eng.pending_entries)
        eng.pending_entries = [pe for pe in eng.pending_entries
                               if f"Q-{pe['intent'].strategy}-{pe['intent'].symbol}" != key]
        if len(eng.pending_entries) == before:
            raise KeyError(f"no queued order {key}")
        self.journal.event(pd.Timestamp.now(), "INFO", "manual", f"queued order {key} cancelled from the chart")
        eng.save_state()
        return {"cancelled": key}

    def review(self, days: int = 1, asof: dt.date | None = None) -> str:
        asof = asof or dt.date.today()
        start = (pd.Timestamp(asof) - pd.Timedelta(days=days - 1)).date()
        title = "Daily review" if days == 1 else f"{days}-day review"
        return period_review(self.journal, str(start), str(asof), title)
