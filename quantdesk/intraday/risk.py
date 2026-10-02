"""Intraday risk: per-trade sizing from the plan's own stop, daily limits, cooldowns, and
time windows. Returns (lots, reasons) — reasons are journaled whether it trades or not."""
from __future__ import annotations

import datetime as dt
import math

import pandas as pd


def _t(s: str) -> dt.time:
    h, m = s.split(":")
    return dt.time(int(h), int(m))


class IntradayRisk:
    def __init__(self, cfg):
        r = cfg.get("intraday.risk", {}) or {}
        self.risk_per_trade = r.get("risk_per_trade", 0.005)
        self.max_trades = r.get("max_trades_per_day", 6)
        self.max_open = r.get("max_open", 2)
        self.daily_loss = r.get("daily_loss_limit", 0.015)
        self.cool_n = r.get("cooldown_after_losses", 2)
        self.cool_min = r.get("cooldown_min", 30)
        self.no_before = _t(r.get("no_entry_before", "09:20"))
        self.no_after = _t(r.get("no_entry_after", "14:45"))
        self.square_off = _t(r.get("square_off", "15:15"))
        self.max_outlay = r.get("max_premium_outlay", 0.25)
        self.max_margin = r.get("max_margin", 0.5)
        self.credit_margin = r.get("credit_margin_per_lot", 0.0)
        self.max_lots = r.get("max_lots", 10)
        self.reset(None, 0.0)

    def reset(self, day, equity: float):
        self.day, self.start_equity = day, equity
        self.trades_today, self.consec_losses, self.cool_until, self.halted = 0, 0, None, False

    def on_close(self, pnl: float, now):
        self.consec_losses = self.consec_losses + 1 if pnl < 0 else 0
        if self.consec_losses >= self.cool_n:
            self.cool_until = now + pd.Timedelta(minutes=self.cool_min)

    def gate(self, now, equity: float, open_trades: list, symbol: str) -> list[str]:
        why = []
        t = now.time()
        if t < self.no_before or t > self.no_after:
            why.append(f"outside entry window {self.no_before:%H:%M}–{self.no_after:%H:%M}")
        if self.trades_today >= self.max_trades:
            why.append(f"{self.max_trades} trades already today")
        if len(open_trades) >= self.max_open:
            why.append(f"{self.max_open} positions already open")
        if any(tr.symbol == symbol for tr in open_trades):
            why.append(f"already holding a {symbol} position")
        dd = equity / self.start_equity - 1 if self.start_equity else 0
        if dd <= -self.daily_loss or self.halted:
            self.halted = True
            why.append(f"daily loss limit hit ({dd:.2%} ≤ -{self.daily_loss:.1%}): done for the day")
        if self.cool_until is not None and now < self.cool_until:
            why.append(f"cooling off after {self.consec_losses} straight losses until {self.cool_until:%H:%M}")
        return why

    def size(self, plan, equity: float, cash: float | None = None) -> tuple[int, list[str]]:
        scale = 0.6 + 0.6 * min(max(plan.conviction, 0), 1)
        budget = equity * self.risk_per_trade * scale
        per_lot = plan.planned_risk_per_lot()
        notes = [f"risk budget ₹{budget:,.0f} = equity ₹{equity:,.0f} × {self.risk_per_trade:.2%} × conviction scale {scale:.2f}",
                 f"planned risk/lot ₹{per_lot:,.0f} ({'credit ×' if plan.is_credit else 'debit ×'}{plan.premium_stop})"]
        if not (per_lot > 0 and math.isfinite(per_lot)):
            return 0, notes + ["undefined risk: refused"]
        lots = math.floor(budget / per_lot)
        caps = {"risk": lots, "max_lots": self.max_lots}
        if plan.is_credit:
            # the broker blocks SPAN + exposure margin, not just the spread's max loss
            margin = max(plan.max_loss_per_lot(), self.credit_margin)
            caps["margin"] = math.floor(self.max_margin * equity / max(margin, 1))
        else:
            caps["premium_outlay"] = math.floor(self.max_outlay * equity / max(plan.net_premium, 1))
            if cash is not None:
                caps["cash"] = math.floor(max(cash, 0) / max(plan.net_premium * 1.01, 1))   # can't pay what isn't there
        binding = min(caps, key=caps.get)
        lots = max(0, caps[binding])
        notes.append(f"{lots} lot(s), binding: {binding}")
        return lots, notes
