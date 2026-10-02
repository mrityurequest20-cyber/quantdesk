"""Broker interface and the paper broker.

The engine talks to a `Broker` only through `execute()` and read-only state, so the
same engine drives a backtest, a paper account and (with the Kite adapter) a live one."""
from __future__ import annotations

import abc
import json
from pathlib import Path

import pandas as pd

from ..core.types import Fill, Instrument, Order
from .costs import CostModel


class Broker(abc.ABC):
    name = "base"
    live = False

    @abc.abstractmethod
    def execute(self, order: Order, ref_price: float, ts: pd.Timestamp, atr: float | None = None) -> Fill | None:
        """Execute `order` around `ref_price`; return the fill (None if rejected)."""

    @abc.abstractmethod
    def positions(self) -> dict[str, dict]:
        ...

    @abc.abstractmethod
    def cash(self) -> float:
        ...

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash() + sum(p["qty"] * prices.get(sym, p["avg_price"]) for sym, p in self.positions().items())

    def healthcheck(self) -> tuple[bool, str]:
        return True, f"{self.name} ok"


class PaperBroker(Broker):
    """Simulated fills: reference price ± slippage, full Indian cost stack, cash + positions."""
    name = "paper"

    def __init__(self, cfg, starting_cash: float | None = None, state_path: Path | None = None):
        self.costs = CostModel(cfg)
        self.state_path = state_path
        self._cash = float(starting_cash if starting_cash is not None else cfg.get("account.starting_capital", 1e6))
        self._pos: dict[str, dict] = {}
        self.fills: list[Fill] = []
        self.fees_paid = 0.0
        self.fee_breakdown: dict[str, float] = {}
        if state_path and Path(state_path).exists():
            self.load()

    def execute(self, order: Order, ref_price: float, ts, atr=None) -> Fill | None:
        if order.qty == 0 or not (ref_price == ref_price):  # NaN guard
            order.status = "REJECTED"
            return None
        px = self.costs.fill_price(order.instrument, order.qty, ref_price, atr)
        if order.order_type == "LIMIT" and order.limit_price is not None:
            if (order.qty > 0 and px > order.limit_price) or (order.qty < 0 and px < order.limit_price):
                order.status = "REJECTED"
                return None
        fees, br = self.costs.fees(order.instrument, order.qty, px)
        self._cash -= order.qty * px + fees
        self.fees_paid += fees
        for k, v in br.items():
            self.fee_breakdown[k] = self.fee_breakdown.get(k, 0.0) + v
        sym = order.instrument.symbol
        p = self._pos.get(sym)
        if p is None:
            self._pos[sym] = {"instrument": order.instrument.to_dict(), "qty": order.qty, "avg_price": px}
        else:
            old_qty = p["qty"]
            new_qty = old_qty + order.qty
            if new_qty == 0:
                del self._pos[sym]
            elif (old_qty > 0) == (order.qty > 0):          # adding to the position
                p["avg_price"] = (p["avg_price"] * old_qty + px * order.qty) / new_qty
                p["qty"] = new_qty
            else:                                             # reducing, or flipping through zero
                p["qty"] = new_qty
                if (new_qty > 0) != (old_qty > 0):
                    p["avg_price"] = px
        order.status = "FILLED"
        fill = Fill(order.id, order.instrument, order.qty, px, fees, pd.Timestamp(ts), br)
        self.fills.append(fill)
        if self.state_path:
            self.save()
        return fill

    def settle_expiry(self, inst: Instrument, settle_price: float, ts) -> Fill | None:
        """Cash-settle an expiring option at intrinsic value (no brokerage; STT on ITM exercise is ignored)."""
        p = self._pos.get(inst.symbol)
        if not p:
            return None
        qty = -p["qty"]
        self._cash -= qty * settle_price
        del self._pos[inst.symbol]
        fill = Fill("expiry", inst, qty, settle_price, 0.0, pd.Timestamp(ts), {})
        self.fills.append(fill)
        if self.state_path:
            self.save()
        return fill

    def positions(self) -> dict[str, dict]:
        return self._pos

    def cash(self) -> float:
        return self._cash

    # ---- persistence (paper trading across days) -----------------------------------------
    def save(self) -> None:
        Path(self.state_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w") as fh:
            json.dump({"cash": self._cash, "positions": self._pos, "fees_paid": self.fees_paid,
                       "fee_breakdown": self.fee_breakdown}, fh, indent=1, default=str)

    def load(self) -> None:
        with open(self.state_path) as fh:
            st = json.load(fh)
        self._cash = st["cash"]
        self._pos = st["positions"]
        self.fees_paid = st.get("fees_paid", 0.0)
        self.fee_breakdown = st.get("fee_breakdown", {})
