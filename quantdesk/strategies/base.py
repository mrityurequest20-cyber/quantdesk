"""Strategy contract.

A strategy (1) precomputes *causal* features once per data refresh in `features()`,
(2) proposes trades in `entries()` from the bar-t view in the MarketContext, and
(3) manages its own open trades in `manage()` (trailing stops, targets, time and
regime exits). It never sizes positions or sends orders — that is the risk manager's
and the engine's job — so every trade passes the same gate."""
from __future__ import annotations

import abc

import numpy as np
import pandas as pd

from ..core.context import FeatureTable, MarketContext
from ..core.types import LINEAR, OPTIONS, ExitDecision, Instrument, LegSpec, Trade, TradeIntent
from ..options.structures import Structure


class Strategy(abc.ABC):
    family = "trend"
    kind = LINEAR
    description = ""

    def __init__(self, name: str, params: dict, cfg):
        self.name = name
        self.p = dict(params)
        self.cfg = cfg
        self.tables: dict[str, FeatureTable] = {}

    def universe(self) -> list[str]:
        return self.cfg.symbols(self.p.get("symbols", "equities"))

    def prepare(self, data: dict[str, pd.DataFrame], aux: dict) -> None:
        self.tables = {}
        for s in self.universe():
            if s in data and len(data[s]) > 30:
                self.tables[s] = FeatureTable(self.features(data[s], s, aux))

    def features(self, df: pd.DataFrame, symbol: str, aux: dict) -> pd.DataFrame:
        return df

    @abc.abstractmethod
    def entries(self, ctx: MarketContext) -> list[TradeIntent]:
        ...

    def manage(self, trade: Trade, ctx: MarketContext) -> ExitDecision | None:
        return None

    # ---- helpers ---------------------------------------------------------------------------
    def row(self, sym: str, ctx: MarketContext) -> dict | None:
        t = self.tables.get(sym)
        r = t.row(ctx.ts) if t else None
        if r is None or any(isinstance(v, float) and np.isnan(v) for k, v in r.items() if k in self.required):
            return None
        return r

    required: tuple = ()

    def instrument_for(self, sym: str) -> Instrument:
        spec = self.cfg.instrument_spec(sym)
        if spec.get("kind") == "index":
            return Instrument.future(sym, int(spec.get("lot_size", 1)))
        return Instrument.equity(sym)

    def linear_intent(self, ctx: MarketContext, sym: str, direction: int, stop: float, rationale: str,
                      context: dict, target: float | None = None, exit_rules: dict | None = None,
                      confidence: float = 0.5, atr: float | None = None) -> TradeIntent | None:
        price = ctx.price(sym)
        inst = self.instrument_for(sym)
        per_share = abs(price - stop)
        if per_share <= 0:
            return None
        cost = ctx.costs.round_trip_estimate(inst, inst.lot_size, price) / inst.lot_size
        return TradeIntent(
            strategy=self.name, family=self.family, symbol=sym, direction=direction, kind=LINEAR,
            legs=[LegSpec(inst, direction)], entry_ref=price,
            risk_per_unit=(per_share + cost) * inst.lot_size,
            stop=stop, target=target, confidence=float(np.clip(confidence, 0.05, 1.0)),
            rationale=rationale, context={**context, "regime": ctx.regime(sym)},
            exit_rules=exit_rules or {}, meta={"atr": atr},
        )

    def options_intent(self, ctx: MarketContext, st: Structure, direction: int, rationale: str,
                       context: dict, exit_rules: dict, confidence: float = 0.5) -> TradeIntent | None:
        max_loss = st.max_loss()
        if not np.isfinite(max_loss) or max_loss <= 0:
            return None
        prem = st.net_premium()
        cost = sum(ctx.costs.round_trip_estimate(l.instrument, l.instrument.lot_size, st.prices[l.instrument.symbol])
                   for l in st.legs)
        meta = {"structure": st.name, "premium": prem, "max_loss": max_loss, "max_profit": st.max_profit(),
                "breakevens": st.breakevens(), "expiry": str(st.expiry),
                "strikes": {l.instrument.symbol: (l.ratio, l.instrument.strike, l.instrument.right) for l in st.legs}}
        return TradeIntent(
            strategy=self.name, family=self.family, symbol=st.underlying, direction=direction, kind=OPTIONS,
            legs=list(st.legs), entry_ref=st.spot, risk_per_unit=max_loss + cost,
            confidence=float(np.clip(confidence, 0.05, 1.0)), rationale=rationale,
            context={**context, "regime": ctx.regime(st.underlying), "structure": st.describe()},
            exit_rules=exit_rules, meta=meta, est_prices=dict(st.prices), margin_per_unit=max_loss,
        )

    # Options P&L helpers used by the options strategies' manage()
    @staticmethod
    def structure_marks(trade: Trade, ctx: MarketContext) -> dict[str, float]:
        S = ctx.price(trade.symbol)
        iv = ctx.atm_iv(trade.symbol)
        return {l.instrument.symbol: ctx.pricer.price(l.instrument, S, ctx.date, iv) for l in trade.legs}
