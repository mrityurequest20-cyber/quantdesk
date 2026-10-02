"""The trading engine: one code path for backtests, paper trading and live trading.

A trading day has two halves:

  on_open(ts)   execute yesterday's decisions at today's open (entries and exits),
                then check intrabar stops/targets for linear trades;
  on_close(ts)  mark to market, settle expiries, run the risk state machine, let each
                strategy manage its trades, collect new trade ideas, size them through
                the allocator + risk manager, queue them for the next open, snapshot.

Decisions made at the close of bar t are only executed at the open of t+1, so a
backtest cannot trade on information it would not have had.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from ..analytics.indicators import atr as atr_ind
from ..analytics.regime import regime_frame
from ..analytics.volatility import close_to_close, iv_percentile, iv_rank
from ..core.calendar import TradingCalendar
from ..core.context import FeatureTable, MarketContext
from ..core.types import FUTURE, LINEAR, OPTION, ExitDecision, Order, Trade, TradeLeg, new_trade_id
from ..execution.broker import Broker, PaperBroker
from ..execution.costs import CostModel
from ..journal.journal import Journal, trade_from_dict, trade_to_dict
from ..options.chain import OptionPricer
from ..risk.allocator import Allocator
from ..risk.manager import RiskManager

log = logging.getLogger(__name__)


class Engine:
    def __init__(self, cfg, data: dict[str, pd.DataFrame], strategies: list, broker: Broker | None = None,
                 journal: Journal | None = None, aux: dict | None = None, warmup_bars: int | None = None,
                 persist_marks: bool = False):
        self.cfg = cfg
        self.calendar = TradingCalendar(cfg.holidays())
        self.pricer = OptionPricer.from_config(cfg)
        self.costs = CostModel(cfg)
        self.risk = RiskManager(cfg)
        self.alloc = Allocator(cfg)
        self.broker = broker or PaperBroker(cfg)
        self.journal = journal or Journal(":memory:")
        self.strategies = {s.name: s for s in strategies}
        self.open_trades: list[Trade] = []
        self.closed_trades: list[Trade] = []
        self.pending_entries: list[dict] = []
        self.pending_exits: dict[str, ExitDecision] = {}
        self.curve: list[dict] = []
        self.warmup = cfg.get("backtest.warmup_bars", 260) if warmup_bars is None else warmup_bars
        self.persist_marks = persist_marks
        self.quote_override: dict[str, float] = {}      # live quotes used for 'open' fills (Kite)
        self._prepare(data, aux)

    # ---- setup ---------------------------------------------------------------------------------
    @staticmethod
    def build_aux(cfg, data: dict[str, pd.DataFrame]) -> dict:
        """IV and regime frames for each index. Expensive (GARCH/HMM), so reusable across runs."""
        vix_df = data.get(cfg.get("universe.volatility_index", "INDIAVIX"))
        iv, regimes = {}, {}
        for s in cfg.symbols("indices"):
            if s not in data:
                continue
            df = data[s]
            beta = float(cfg.instrument_spec(s).get("iv_beta", 1.0))
            if vix_df is not None:
                v = vix_df["close"].reindex(df.index).ffill()
                vo = vix_df["open"].reindex(df.index).ffill()
            else:  # no VIX feed: realised vol plus a typical premium as the IV proxy
                v = close_to_close(df["close"], 21) * 1.15
                vo = v.shift(1)
            f = pd.DataFrame({"vix": v, "iv": v * beta / 100, "iv_open": vo * beta / 100}, index=df.index)
            f["iv_rank"] = iv_rank(f["iv"])
            f["iv_pct"] = iv_percentile(f["iv"])
            iv[s] = f
            regimes[s] = regime_frame(df, v, cfg)
        return {"iv": iv, "regime": regimes}

    def _prepare(self, data, aux):
        self.data = {s: df for s, df in data.items() if not s.startswith("_")}
        self.aux = aux or self.build_aux(self.cfg, self.data)
        self.bars = {s: FeatureTable(df.assign(atr=atr_ind(df, 14))) for s, df in self.data.items()}
        self.iv_tables = {s: FeatureTable(f) for s, f in self.aux["iv"].items()}
        reg_tables = {s: FeatureTable(f) for s, f in self.aux["regime"].items()}
        for s in self.strategies.values():
            s.prepare(self.data, self.aux)
        self.ctx = MarketContext(self.cfg, self.bars, reg_tables, self.iv_tables, self.pricer, self.calendar, self.costs)
        bench = self.cfg.get("universe.benchmark", "NIFTY")
        self.timeline = self.data[bench].index if bench in self.data else sorted(set().union(*[d.index for d in self.data.values()]))

    # ---- prices -------------------------------------------------------------------------------------
    def mark(self, inst, ts, field: str = "close") -> float:
        if field == "open" and inst.symbol in self.quote_override:
            return float(self.quote_override[inst.symbol])
        if inst.kind == OPTION:
            bt, it = self.bars.get(inst.underlying), self.iv_tables.get(inst.underlying)
            if bt is None or it is None:
                return float("nan")
            S = bt.at(ts, field)
            iv = it.at(ts, "iv_open" if field == "open" else "iv")
            if not (S == S and iv == iv):
                return float("nan")
            return self.pricer.price(inst, float(S), ts.date(), float(iv))
        sym = inst.underlying if inst.kind == FUTURE else inst.symbol
        t = self.bars.get(sym)
        return float(t.at(ts, field)) if t else float("nan")

    def _atr(self, inst, ts):
        if inst.kind == OPTION:
            return None
        t = self.bars.get(inst.underlying if inst.kind == FUTURE else inst.symbol)
        a = t.at(ts, "atr") if t else float("nan")
        return float(a) if a == a else None

    def trade_marks(self, trade: Trade, ts, field="close") -> dict[str, float]:
        out = {}
        for l in trade.legs:
            m = self.mark(l.instrument, ts, field)
            out[l.instrument.symbol] = m if m == m else trade.last_mark.get(l.instrument.symbol, l.entry_price)
        return out

    def equity(self, marks: dict[str, float]) -> float:
        pos = self.broker.positions()
        return self.broker.cash() + sum(p["qty"] * marks.get(sym, p["avg_price"]) for sym, p in pos.items())

    # ---- execution -----------------------------------------------------------------------------------
    def _fill(self, trade: Trade, inst, qty: int, ref: float, ts, purpose: str):
        order = Order(inst, qty, trade.id, purpose, ref_price=ref, created_at=ts)
        fill = self.broker.execute(order, ref, ts, self._atr(inst, ts))
        if fill is not None:
            self.risk.orders_today += 1
            self.journal.fill(ts, trade.id, inst.symbol, qty, fill.price, fill.fees, fill.fee_breakdown)
        return fill

    def _open(self, pe: dict, ts) -> None:
        intent, units = pe["intent"], pe["units"]
        refs = {}
        for leg in intent.legs:
            ref = self.mark(leg.instrument, ts, "open")
            if not (ref == ref) or ref <= 0:
                self.journal.decision(ts, intent.strategy, intent.symbol, "cancelled", f"no open price for {leg.instrument.symbol}")
                return
            refs[leg.instrument.symbol] = ref
        if intent.kind == LINEAR and intent.stop is not None and len(intent.legs) == 1:
            o = refs[intent.legs[0].instrument.symbol]
            if (intent.direction > 0 and o <= intent.stop) or (intent.direction < 0 and o >= intent.stop):
                self.journal.decision(ts, intent.strategy, intent.symbol, "cancelled",
                                      f"gapped through the stop before entry (open {o:.2f}, stop {intent.stop:.2f})")
                return
        und = self.bars.get(intent.symbol)
        entry_und = self.quote_override.get(intent.symbol) or (float(und.at(ts, "open")) if und else float(intent.entry_ref))
        t = Trade(id=new_trade_id(), strategy=intent.strategy, family=intent.family, symbol=intent.symbol,
                  direction=intent.direction, kind=intent.kind, legs=[], units=units, opened_at=ts,
                  entry_underlying=entry_und, initial_risk=intent.risk_per_unit * units, stop=intent.stop,
                  target=intent.target, exit_rules=dict(intent.exit_rules), rationale=intent.rationale,
                  context=dict(intent.context), meta=dict(intent.meta))
        t.meta["decided_at"] = str(pe["decided_at"])
        t.meta["weight_note"] = pe.get("weight_note", "")
        for leg in intent.legs:
            qty = leg.ratio * units * leg.instrument.lot_size
            fill = self._fill(t, leg.instrument, qty, refs[leg.instrument.symbol], ts, "open")
            if fill is None:
                for done in t.legs:                                  # unwind a partial structure
                    self._fill(t, done.instrument, -done.qty, refs[done.instrument.symbol], ts, "unwind")
                self.journal.event(ts, "ERROR", "execution", f"leg {leg.instrument.symbol} rejected; {t.id} unwound")
                return
            t.legs.append(TradeLeg(leg.instrument, qty, fill.price))
            t.fees += fill.fees
        if t.kind == LINEAR and t.stop is not None and len(t.legs) == 1:
            l = t.legs[0]
            t.initial_risk = abs(l.entry_price - t.stop) * abs(l.qty) + 2 * t.fees
        if t.kind == "options":
            t.meta["entry_premium"] = t.entry_cost
            planned = intent.meta.get("planned_risk_per_unit")
            if planned:                                   # R is measured against the planned stop, not max loss
                t.initial_risk = planned * units + 2 * t.fees
        t.last_mark = {l.instrument.symbol: l.entry_price for l in t.legs}
        t.pnl = -t.fees
        self.open_trades.append(t)
        self.journal.open_trade(t, pe.get("notes"))

    def _close(self, t: Trade, ts, reason: str, note: str = "", refs: dict | None = None,
               field: str = "open", settle: bool = False) -> None:
        for l in t.legs:
            ref = (refs or {}).get(l.instrument.symbol)
            if ref is None:
                ref = self.mark(l.instrument, ts, field)
            if not (ref == ref):
                ref = t.last_mark.get(l.instrument.symbol, l.entry_price)
            if settle:
                fill = self.broker.settle_expiry(l.instrument, ref, ts) if hasattr(self.broker, "settle_expiry") else None
                l.exit_price = ref
                if fill is not None:
                    self.journal.fill(ts, t.id, l.instrument.symbol, fill.qty, ref, 0.0, {"settlement": True})
            else:
                fill = self._fill(t, l.instrument, -l.qty, ref, ts, "close")
                if fill is None:
                    self.journal.event(ts, "ERROR", "execution", f"exit leg {l.instrument.symbol} of {t.id} rejected")
                    return
                l.exit_price = fill.price
                t.fees += fill.fees
        t.pnl = sum(l.qty * (l.exit_price - l.entry_price) for l in t.legs) - t.fees
        t.mae, t.mfe = min(t.mae, t.pnl), max(t.mfe, t.pnl)
        t.status, t.closed_at, t.exit_reason, t.exit_note = "closed", ts, reason, note
        und = self.bars.get(t.symbol)
        t.exit_underlying = float(und.at(ts, field if not settle else "close")) if und else t.meta.get("last_z")
        self.open_trades.remove(t)
        self.pending_exits.pop(t.id, None)
        self.closed_trades.append(t)
        self.journal.close_trade(t, self.ctx.regime(t.symbol if t.symbol in self.iv_tables else None))

    # ---- the trading day ---------------------------------------------------------------------------------
    def on_open(self, ts) -> None:
        self.ctx.ts = ts
        eq_guess = self.curve[-1]["equity"] if self.curve else self.broker.cash()
        self.risk.new_bar(ts, eq_guess)
        for tid, dec in list(self.pending_exits.items()):
            t = next((x for x in self.open_trades if x.id == tid), None)
            if t is None:
                self.pending_exits.pop(tid, None)
                continue
            self._close(t, ts, dec.reason, dec.note, field="open")
        entries, self.pending_entries = self.pending_entries, []
        for pe in entries:
            self._open(pe, ts)
        self._intrabar(ts)

    def _intrabar(self, ts) -> None:
        for t in list(self.open_trades):
            if t.kind != LINEAR or len(t.legs) != 1 or t.id in self.pending_exits:
                continue
            inst = t.legs[0].instrument
            b = self.bars.get(inst.underlying if inst.kind == FUTURE else inst.symbol)
            bar = b.row(ts) if b else None
            if bar is None:
                continue
            o, h, lo = bar["open"], bar["high"], bar["low"]
            d = 1 if t.legs[0].qty > 0 else -1
            hit_stop = t.stop is not None and ((d > 0 and lo <= t.stop) or (d < 0 and h >= t.stop))
            hit_tgt = t.target is not None and ((d > 0 and h >= t.target) or (d < 0 and lo <= t.target))
            if hit_stop:                                    # stop assumed first if both touched (conservative)
                px = min(o, t.stop) if d > 0 else max(o, t.stop)
                self._close(t, ts, "stop", f"stop {t.stop:.2f} hit (fill ref {px:.2f})", refs={inst.symbol: px})
            elif hit_tgt:
                px = max(o, t.target) if d > 0 else min(o, t.target)
                self._close(t, ts, "target", f"target {t.target:.2f} hit", refs={inst.symbol: px})

    def on_close(self, ts, i: int) -> dict:
        ctx = self.ctx.at(ts, i, 0.0, self.open_trades, self.closed_trades)
        marks: dict[str, float] = {}
        for t in self.open_trades:
            m = self.trade_marks(t, ts)
            t.update_excursions(m)
            t.bars_held += 1
            marks.update(m)
        self._settle_expiries(ts)
        equity = self.equity(marks)
        ctx.equity = equity
        for ev in self.risk.end_of_bar(equity):
            if ev == "HALT":
                self.journal.event(ts, "CRITICAL", "risk", self.risk.halt_reason)
                for t in self.open_trades:
                    self.pending_exits[t.id] = ExitDecision("risk_halt", self.risk.halt_reason)
            elif ev == "DAILY_LIMIT":
                self.journal.event(ts, "WARN", "risk", self.risk.block_reason)

        for t in list(self.open_trades):
            if t.id in self.pending_exits:
                continue
            s = self.strategies.get(t.strategy)
            dec = s.manage(t, ctx) if s else None
            if dec is None and t.exit_rules.get("time_stop") and t.bars_held >= t.exit_rules["time_stop"]:
                dec = ExitDecision("time_stop", f"held {t.bars_held} bars")
            if dec is not None:
                self.pending_exits[t.id] = dec
                self.journal.decision(ts, t.strategy, t.symbol, "exit_signal", f"{dec.reason}: {dec.note}")
            elif self.persist_marks:
                self.journal.update_open_trade(t)

        self._collect_entries(ts, i, equity)
        return self._snapshot(ts, equity)

    def _settle_expiries(self, ts) -> None:
        for t in list(self.open_trades):
            exp = t.nearest_expiry()
            if exp is None or exp > ts.date():
                continue
            S = self.bars[t.symbol].at(ts, "close")
            refs = {l.instrument.symbol: (max(S - l.instrument.strike, 0.0) if l.instrument.right == "CE"
                                          else max(l.instrument.strike - S, 0.0)) for l in t.legs}
            self._close(t, ts, "expiry", f"settled at {S:,.2f}", refs=refs, field="close", settle=True)

    def _collect_entries(self, ts, i: int, equity: float) -> None:
        ok, _ = self.risk.can_enter()
        if not ok or i < self.warmup:
            return
        intents = []
        for s in self.strategies.values():
            try:
                intents += s.entries(self.ctx)
            except Exception as exc:  # one broken strategy must not stop the book
                log.exception("strategy %s failed", s.name)
                self.journal.event(ts, "ERROR", "strategy", f"{s.name}.entries raised {exc!r}")
        intents.sort(key=lambda x: -x.confidence)
        pending_trades: list[Trade] = []
        pending_keys = {(pe["intent"].strategy, pe["intent"].symbol) for pe in self.pending_entries}
        for it in intents:
            if (it.strategy, it.symbol) in pending_keys:
                continue
            w, wnote = self.alloc.weight(it, self.ctx)
            if w <= 0:
                self.journal.decision(ts, it.strategy, it.symbol, "rejected", wnote, 0, it.context)
                continue
            dec = self.risk.size(it, equity, w, self.open_trades + pending_trades, self.ctx)
            if not dec.approved:
                self.journal.decision(ts, it.strategy, it.symbol, "rejected", " | ".join([wnote] + dec.reasons), 0, it.context)
                continue
            self.pending_entries.append({"intent": it, "units": dec.units, "notes": [wnote] + dec.reasons,
                                         "weight_note": wnote, "decided_at": ts})
            pending_keys.add((it.strategy, it.symbol))
            pending_trades.append(self._pseudo_trade(it, dec.units, ts))
            self.journal.decision(ts, it.strategy, it.symbol, "approved", " | ".join([wnote] + dec.reasons), dec.units, it.context)

    def _pseudo_trade(self, it, units, ts) -> Trade:
        legs = [TradeLeg(l.instrument, l.ratio * units * l.instrument.lot_size,
                         it.est_prices.get(l.instrument.symbol, self.mark(l.instrument, ts))) for l in it.legs]
        return Trade(id="pending", strategy=it.strategy, family=it.family, symbol=it.symbol, direction=it.direction,
                     kind=it.kind, legs=legs, units=units, opened_at=ts, entry_underlying=it.entry_ref,
                     initial_risk=it.risk_per_unit * units, stop=it.stop, meta=dict(it.meta))

    def _snapshot(self, ts, equity: float) -> dict:
        port = self.risk.portfolio(self.open_trades, self.ctx) if self.open_trades else \
            {"gross": 0.0, "net_delta": 0.0, "vega": 0.0, "open_risk": 0.0}
        row = {"ts": ts, "equity": equity, "cash": self.broker.cash(), "drawdown": self.risk.drawdown(equity),
               "open_trades": len(self.open_trades), "gross": port["gross"], "net_delta": port["net_delta"],
               "vega": port["vega"], "open_risk": port["open_risk"], "regime": self.ctx.regime()}
        self.curve.append(row)
        self.journal.snapshot(**row)
        return row

    # ---- drivers ---------------------------------------------------------------------------------------
    def run(self, start=None, end=None, flatten_at_end: bool = True, progress=None) -> None:
        tl = self.timeline
        idx = [k for k, ts in enumerate(tl) if (start is None or ts >= pd.Timestamp(start))
               and (end is None or ts <= pd.Timestamp(end))]
        if not idx:
            raise ValueError("no bars in the requested window")
        self._first_ts = tl[max(idx[0], 0)]
        for n, k in enumerate(idx):
            ts = tl[k]
            if n > 0:
                self.on_open(ts)
            else:
                self.ctx.ts = ts
                self.risk.new_bar(ts, self.broker.cash())
            self.on_close(ts, k)
            if progress and n % 250 == 0:
                progress(n, len(idx), ts)
        if flatten_at_end:
            last = tl[idx[-1]]
            for t in list(self.open_trades):
                self._close(t, last, "end_of_backtest", "flattened at the last close", field="close")
            if self.curve:
                self.curve[-1]["equity"] = self.equity({})
        self.journal.commit()

    # ---- persistence for paper/live continuity ---------------------------------------------------------------
    def save_state(self) -> None:
        self.journal.set_state("engine", {
            "open_trades": [trade_to_dict(t) | {"last_mark": t.last_mark} for t in self.open_trades],
            "pending_exits": {k: [v.reason, v.note] for k, v in self.pending_exits.items()},
            "pending_entries": [self._serialise_pe(pe) for pe in self.pending_entries],
            "risk": {"peak": self.risk.peak, "halted": self.risk.halted, "halt_reason": self.risk.halt_reason},
            "last_curve": {k: (str(v) if k == "ts" else v) for k, v in (self.curve[-1] if self.curve else {}).items()},
        })

    @staticmethod
    def _serialise_pe(pe: dict) -> dict:
        it = pe["intent"]
        return {"units": pe["units"], "notes": pe["notes"], "weight_note": pe.get("weight_note", ""),
                "decided_at": str(pe["decided_at"]),
                "intent": {"strategy": it.strategy, "family": it.family, "symbol": it.symbol, "direction": it.direction,
                           "kind": it.kind, "entry_ref": it.entry_ref, "risk_per_unit": it.risk_per_unit, "stop": it.stop,
                           "target": it.target, "confidence": it.confidence, "rationale": it.rationale,
                           "context": it.context, "exit_rules": it.exit_rules, "meta": it.meta,
                           "est_prices": it.est_prices, "margin_per_unit": it.margin_per_unit,
                           "legs": [{"instrument": l.instrument.to_dict(), "ratio": l.ratio} for l in it.legs]}}

    def load_state(self) -> bool:
        from ..core.types import Instrument, LegSpec, TradeIntent
        st = self.journal.get_state("engine")
        if not st:
            return False
        self.open_trades = []
        for d in st["open_trades"]:
            t = trade_from_dict(d)
            t.last_mark = d.get("last_mark", {})
            self.open_trades.append(t)
        self.pending_exits = {k: ExitDecision(*v) for k, v in st.get("pending_exits", {}).items()}
        self.pending_entries = []
        for pe in st.get("pending_entries", []):
            i = dict(pe["intent"])
            legs = [LegSpec(Instrument.from_dict(l["instrument"]), int(l["ratio"])) for l in i.pop("legs")]
            self.pending_entries.append({"intent": TradeIntent(legs=legs, **i), "units": pe["units"], "notes": pe["notes"],
                                         "weight_note": pe.get("weight_note", ""), "decided_at": pd.Timestamp(pe["decided_at"])})
        r = st.get("risk", {})
        self.risk.peak, self.risk.halted, self.risk.halt_reason = r.get("peak"), r.get("halted", False), r.get("halt_reason", "")
        lc = st.get("last_curve") or {}
        if lc:
            lc["ts"] = pd.Timestamp(lc["ts"])
            self.curve = [lc]
        return True
