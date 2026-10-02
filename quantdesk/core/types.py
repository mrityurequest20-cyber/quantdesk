"""Domain objects shared by every layer: instruments, trade intents, trades, orders, fills."""
from __future__ import annotations

import datetime as dt
import itertools
import uuid
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

EQUITY, INDEX, FUTURE, OPTION, VOL_INDEX = "equity", "index", "future", "option", "vol_index"
LINEAR, OPTIONS = "linear", "options"


@dataclass(frozen=True)
class Instrument:
    symbol: str
    kind: str
    underlying: str
    lot_size: int = 1
    strike: float | None = None
    expiry: dt.date | None = None
    right: str | None = None  # "CE" | "PE"

    @property
    def is_option(self) -> bool:
        return self.kind == OPTION

    @classmethod
    def equity(cls, symbol: str) -> "Instrument":
        return cls(symbol=symbol, kind=EQUITY, underlying=symbol, lot_size=1)

    @classmethod
    def future(cls, underlying: str, lot_size: int) -> "Instrument":
        # A rolling front-month proxy: priced off spot, basis and roll cost ignored.
        return cls(symbol=f"{underlying}-FUT", kind=FUTURE, underlying=underlying, lot_size=lot_size)

    @classmethod
    def option(cls, underlying: str, expiry: dt.date, strike: float, right: str, lot_size: int) -> "Instrument":
        right = right.upper()
        if right not in ("CE", "PE"):
            raise ValueError(f"option right must be CE or PE, got {right!r}")
        sym = f"{underlying}{expiry:%d%b%y}{int(round(strike))}{right}".upper()
        return cls(symbol=sym, kind=OPTION, underlying=underlying, lot_size=lot_size,
                   strike=float(strike), expiry=expiry, right=right)

    def cost_segment(self, intraday: bool = False) -> str:
        if self.kind == OPTION:
            return "options"
        if self.kind == FUTURE:
            return "futures"
        return "equity_intraday" if intraday else "equity_delivery"

    def to_dict(self) -> dict:
        return {"symbol": self.symbol, "kind": self.kind, "underlying": self.underlying,
                "lot_size": self.lot_size, "strike": self.strike,
                "expiry": self.expiry.isoformat() if self.expiry else None, "right": self.right}

    @classmethod
    def from_dict(cls, d: dict) -> "Instrument":
        exp = d.get("expiry")
        return cls(symbol=d["symbol"], kind=d["kind"], underlying=d["underlying"],
                   lot_size=int(d.get("lot_size", 1)), strike=d.get("strike"),
                   expiry=dt.date.fromisoformat(exp) if exp else None, right=d.get("right"))


@dataclass
class LegSpec:
    """One leg of a proposed position. qty = ratio * units * instrument.lot_size."""
    instrument: Instrument
    ratio: int


@dataclass
class TradeIntent:
    """What a strategy wants to do. The risk manager decides `units` (the size)."""
    strategy: str
    family: str                     # allocator bucket: trend | mean_reversion | momentum | pairs | short_vol | long_vol
    symbol: str                     # primary underlying (for pairs: "A/B")
    direction: int                  # +1 long, -1 short, 0 market-neutral
    kind: str                       # linear | options
    legs: list[LegSpec]
    entry_ref: float                # underlying price at decision time
    risk_per_unit: float            # INR lost per unit if the plan's stop is hit (sizing input)
    stop: float | None = None       # underlying stop (linear trades)
    target: float | None = None
    confidence: float = 0.5
    rationale: str = ""
    context: dict = field(default_factory=dict)
    exit_rules: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    est_prices: dict = field(default_factory=dict)  # symbol -> model/quote price at decision (options)
    margin_per_unit: float = 0.0


@dataclass
class ExitDecision:
    reason: str
    note: str = ""


@dataclass
class TradeLeg:
    instrument: Instrument
    qty: int                       # signed units (shares / contracts * lot)
    entry_price: float = 0.0
    exit_price: float | None = None


_trade_counter = itertools.count(1)


def new_trade_id(prefix: str = "T") -> str:
    return f"{prefix}{dt.datetime.now():%y%m%d}-{next(_trade_counter):04d}-{uuid.uuid4().hex[:4]}"


@dataclass
class Trade:
    id: str
    strategy: str
    family: str
    symbol: str
    direction: int
    kind: str
    legs: list[TradeLeg]
    units: int
    opened_at: pd.Timestamp
    entry_underlying: float
    initial_risk: float
    stop: float | None = None
    target: float | None = None
    exit_rules: dict = field(default_factory=dict)
    rationale: str = ""
    context: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    fees: float = 0.0
    status: str = "open"
    closed_at: pd.Timestamp | None = None
    exit_reason: str | None = None
    exit_note: str = ""
    exit_underlying: float | None = None
    pnl: float = 0.0               # net of all fees once closed; running MTM while open
    mae: float = 0.0               # worst open P&L seen (INR, <= 0)
    mfe: float = 0.0               # best open P&L seen (INR, >= 0)
    bars_held: int = 0
    last_mark: dict = field(default_factory=dict)

    @property
    def entry_cost(self) -> float:
        """Signed cash paid to open (negative = credit received)."""
        return sum(l.qty * l.entry_price for l in self.legs)

    def value(self, prices: dict[str, float]) -> float:
        return sum(l.qty * prices.get(l.instrument.symbol, l.entry_price) for l in self.legs)

    def open_pnl(self, prices: dict[str, float]) -> float:
        return self.value(prices) - self.entry_cost - self.fees

    def update_excursions(self, prices: dict[str, float]) -> float:
        p = self.open_pnl(prices)
        self.mae = min(self.mae, p)
        self.mfe = max(self.mfe, p)
        self.pnl = p
        self.last_mark = dict(prices)
        return p

    @property
    def r_multiple(self) -> float:
        return self.pnl / self.initial_risk if self.initial_risk > 0 else 0.0

    def symbols(self) -> list[str]:
        return [l.instrument.symbol for l in self.legs]

    def nearest_expiry(self) -> dt.date | None:
        exps = [l.instrument.expiry for l in self.legs if l.instrument.expiry]
        return min(exps) if exps else None

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id, "strategy": self.strategy, "family": self.family, "symbol": self.symbol,
            "direction": self.direction, "kind": self.kind, "units": self.units,
            "opened_at": str(self.opened_at), "closed_at": str(self.closed_at) if self.closed_at is not None else None,
            "entry_underlying": self.entry_underlying, "exit_underlying": self.exit_underlying,
            "stop": self.stop, "target": self.target, "initial_risk": self.initial_risk,
            "pnl": self.pnl, "fees": self.fees, "r_multiple": self.r_multiple,
            "mae": self.mae, "mfe": self.mfe, "bars_held": self.bars_held,
            "exit_reason": self.exit_reason, "status": self.status,
        }


@dataclass
class Order:
    instrument: Instrument
    qty: int                      # signed: + buy, - sell
    trade_id: str
    purpose: str                  # open | close | adjust
    order_type: str = "MARKET"
    limit_price: float | None = None
    ref_price: float | None = None
    created_at: pd.Timestamp | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "NEW"
    broker_id: str | None = None


@dataclass
class Fill:
    order_id: str
    instrument: Instrument
    qty: int
    price: float
    fees: float
    timestamp: pd.Timestamp
    fee_breakdown: dict = field(default_factory=dict)
