"""Pre-trade risk: position sizing and portfolio limits, drawdown de-risking, daily loss
limit and the drawdown kill switch. Every intent passes through `size()`; nothing
reaches the broker without an approved, non-zero size."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core.types import EQUITY, FUTURE, OPTION, Trade, TradeIntent


@dataclass
class SizingDecision:
    units: int
    approved: bool
    budget: float
    weight: float
    reasons: list[str] = field(default_factory=list)
    binding: str = ""


class RiskManager:
    def __init__(self, cfg):
        r = cfg.get("risk", {})
        self.cfg = cfg
        self.risk_per_trade = r.get("risk_per_trade", 0.0075)
        self.opt_max_loss = r.get("options_max_loss_per_trade", 0.02)
        self.max_open = r.get("max_open_trades", 10)
        self.max_gross = r.get("max_gross_exposure", 2.0)
        self.max_symbol = r.get("max_symbol_exposure", 0.4)
        self.max_strat_risk = r.get("max_strategy_open_risk", 0.03)
        self.max_total_risk = r.get("max_total_open_risk", 0.08)
        self.daily_loss_limit = r.get("daily_loss_limit", 0.02)
        self.dd_start = r.get("drawdown_derisk_start", 0.06)
        self.dd_halt = r.get("max_drawdown_halt", 0.15)
        self.max_vega = r.get("max_net_vega_pct", 0.0035)
        self.max_delta = r.get("max_net_delta", 1.5)
        self.max_orders = r.get("max_orders_per_day", 40)
        self.max_margin = r.get("max_margin_utilisation", 0.95)
        self.fut_margin = r.get("futures_margin_pct", 0.15)
        self.peak = None
        self.day = None
        self.day_start_equity = None
        self.orders_today = 0
        self.halted = False
        self.halt_reason = ""
        self.entries_blocked = False
        self.block_reason = ""

    # ---- state machine -----------------------------------------------------------------------
    def new_bar(self, ts, equity: float) -> list[str]:
        events = []
        d = ts.date()
        if d != self.day:
            self.day, self.day_start_equity, self.orders_today = d, equity, 0
            self.entries_blocked, self.block_reason = False, ""
        self.peak = equity if self.peak is None else max(self.peak, equity)
        return events

    def end_of_bar(self, equity: float) -> list[str]:
        events = []
        self.peak = max(self.peak or equity, equity)
        dd = self.drawdown(equity)
        if not self.halted and dd >= self.dd_halt:
            self.halted = True
            self.halt_reason = f"drawdown {dd:.1%} hit the {self.dd_halt:.0%} kill switch: flatten and stop"
            events.append("HALT")
        day_ret = equity / self.day_start_equity - 1 if self.day_start_equity else 0.0
        if not self.entries_blocked and day_ret <= -self.daily_loss_limit:
            self.entries_blocked = True
            self.block_reason = f"daily loss {day_ret:.2%} breached the {self.daily_loss_limit:.0%} limit"
            events.append("DAILY_LIMIT")
        return events

    def drawdown(self, equity: float) -> float:
        return 0.0 if not self.peak else max(0.0, 1 - equity / self.peak)

    def dd_scale(self, equity: float) -> float:
        dd = self.drawdown(equity)
        if dd <= self.dd_start:
            return 1.0
        if dd >= self.dd_halt:
            return 0.0
        return 1.0 - (dd - self.dd_start) / (self.dd_halt - self.dd_start)

    def can_enter(self) -> tuple[bool, str]:
        if self.halted:
            return False, self.halt_reason
        if self.entries_blocked:
            return False, self.block_reason
        if self.orders_today >= self.max_orders:
            return False, f"max {self.max_orders} orders/day reached"
        return True, ""

    def reset_halt(self, equity: float) -> None:
        """Manual re-arm after a post-mortem (CLI: `quantdesk risk reset`)."""
        self.halted, self.halt_reason, self.peak = False, "", equity

    # ---- exposures ----------------------------------------------------------------------------
    @staticmethod
    def leg_price(inst, ctx) -> float:
        if inst.kind == OPTION:
            return ctx.pricer.price(inst, ctx.price(inst.underlying), ctx.date, ctx.atm_iv(inst.underlying))
        return ctx.price(inst.underlying if inst.kind == FUTURE else inst.symbol)

    def trade_exposure(self, trade_legs, ctx) -> dict:
        """Delta-notional per underlying, net vega, margin, for (instrument, qty) pairs."""
        by_sym: dict[str, float] = {}
        vega = 0.0
        margin = 0.0
        for inst, qty in trade_legs:
            S = ctx.price(inst.underlying) if ctx.has(inst.underlying) else np.nan
            if inst.kind == OPTION:
                g = ctx.pricer.greeks(inst, S, ctx.date, ctx.atm_iv(inst.underlying))
                by_sym[inst.underlying] = by_sym.get(inst.underlying, 0.0) + qty * g["delta"] * S
                vega += qty * g["vega"]
            else:
                px = S if inst.kind == FUTURE else ctx.price(inst.symbol)
                by_sym[inst.underlying] = by_sym.get(inst.underlying, 0.0) + qty * px
                margin += abs(qty) * px * (self.fut_margin if inst.kind == FUTURE else 1.0)
        return {"delta_by_symbol": by_sym, "vega": vega, "margin": margin}

    def portfolio(self, trades: list[Trade], ctx) -> dict:
        legs = [(l.instrument, l.qty) for t in trades for l in t.legs]
        ex = self.trade_exposure(legs, ctx)
        ex["margin"] += sum(t.meta.get("max_loss", 0.0) * t.units for t in trades if t.kind == "options")
        ex["gross"] = sum(abs(v) for v in ex["delta_by_symbol"].values())
        ex["net_delta"] = sum(ex["delta_by_symbol"].values())
        ex["open_risk"] = sum(self.open_risk(t, ctx) for t in trades)
        ex["risk_by_strategy"] = {}
        for t in trades:
            ex["risk_by_strategy"][t.strategy] = ex["risk_by_strategy"].get(t.strategy, 0.0) + self.open_risk(t, ctx)
        return ex

    @staticmethod
    def open_risk(t: Trade, ctx) -> float:
        """Money still at risk: distance to stop for linear trades (0 once the stop is past
        breakeven), max-loss-based for option structures, initial risk otherwise."""
        if t.kind == "linear" and t.stop is not None and len(t.legs) == 1 and ctx.has(t.symbol):
            px = ctx.price(t.symbol)
            leg = t.legs[0]
            return max(0.0, (px - t.stop) * leg.qty) if leg.qty > 0 else max(0.0, (t.stop - px) * -leg.qty)
        if t.kind == "options" and t.meta.get("max_loss"):
            return t.meta["max_loss"] * t.units       # worst case, not the planned stop
        return t.initial_risk

    # ---- sizing ---------------------------------------------------------------------------------
    def size(self, intent: TradeIntent, equity: float, weight: float, open_trades: list[Trade], ctx) -> SizingDecision:
        ok, why = self.can_enter()
        if not ok:
            return SizingDecision(0, False, 0.0, weight, [why], "gate")
        if len(open_trades) >= self.max_open:
            return SizingDecision(0, False, 0.0, weight, [f"{self.max_open} trades already open"], "max_open")
        scale = self.dd_scale(equity)
        conf_mult = 0.75 + 0.5 * intent.confidence
        base = self.opt_max_loss if intent.kind == "options" else self.risk_per_trade
        budget = equity * base * weight * scale * conf_mult
        reasons = [f"budget ₹{budget:,.0f} = equity ₹{equity:,.0f} × {base:.2%} × weight {weight:.2f} × "
                   f"DD scale {scale:.2f} × confidence {conf_mult:.2f}"]
        if intent.risk_per_unit <= 0:
            return SizingDecision(0, False, budget, weight, reasons + ["non-positive risk per unit"], "invalid")
        caps = {"risk_budget": math.floor(budget / intent.risk_per_unit)}

        port = self.portfolio(open_trades, ctx)
        one = self.trade_exposure([(l.instrument, l.ratio * l.instrument.lot_size) for l in intent.legs], ctx)
        if intent.kind == "options":
            one["margin"] += intent.margin_per_unit
        per_unit_risk = intent.risk_per_unit

        def cap(name, room, per_unit):
            if per_unit > 1e-9:
                caps[name] = max(0, math.floor(room / per_unit))

        cap("total_open_risk", self.max_total_risk * equity - port["open_risk"], per_unit_risk)
        cap("strategy_open_risk", self.max_strat_risk * equity - port["risk_by_strategy"].get(intent.strategy, 0.0),
            per_unit_risk)
        cap("gross_exposure", self.max_gross * equity - port["gross"], sum(abs(v) for v in one["delta_by_symbol"].values()))
        for sym, d in one["delta_by_symbol"].items():
            cur = port["delta_by_symbol"].get(sym, 0.0)
            if abs(d) > 1e-9 and np.sign(d) == np.sign(cur or d):
                cap(f"symbol_exposure:{sym}", self.max_symbol * equity - abs(cur), abs(d))
        if abs(one["vega"]) > 1e-9 and np.sign(one["vega"]) == np.sign(port["vega"] or one["vega"]):
            cap("net_vega", self.max_vega * equity - abs(port["vega"]), abs(one["vega"]))
        nd = sum(one["delta_by_symbol"].values())
        if abs(nd) > 1e-9 and np.sign(nd) == np.sign(port["net_delta"] or nd):
            cap("net_delta", self.max_delta * equity - abs(port["net_delta"]), abs(nd))
        cap("margin", self.max_margin * equity - port["margin"], one["margin"])

        binding = min(caps, key=caps.get)
        units = int(max(0, caps[binding]))
        if units < 1:
            reasons.append(f"size < 1 unit (binding limit: {binding}; risk/unit ₹{per_unit_risk:,.0f})")
            return SizingDecision(0, False, budget, weight, reasons, binding)
        reasons.append(f"{units} unit(s); binding limit: {binding}")
        return SizingDecision(units, True, budget, weight, reasons, binding)
