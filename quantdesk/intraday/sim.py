"""Simulated execution and marking for intraday options.

* Fills: buys at the ask, sells at the bid (from the real chain when there is one),
  plus a configurable number of adverse ticks, plus the full Indian cost stack.
* Marking between chain refreshes: each held option keeps the implied vol implied by its
  last real quote; it is repriced with the *current* spot and minute-level time to expiry
  (so delta, gamma and theta all show up in the P&L), and its last quoted half-spread
  gives the bid/ask we could actually exit at.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.types import Instrument
from ..execution.broker import PaperBroker
from .chains import IntradayPricer, mid, time_to_expiry


class QuoteMarker:
    def __init__(self, pricer: IntradayPricer):
        self.pricer = pricer
        self.iv: dict[str, float] = {}             # option symbol -> decimal IV from the last quote
        self.half: dict[str, float] = {}           # option symbol -> half spread (INR)
        self.atm_iv: dict[str, float] = {}         # underlying -> ATM IV fallback (decimal)

    def calibrate(self, chain: pd.DataFrame, lot_size: int) -> None:
        u, exp, S, ts = chain.attrs["underlying"], chain.attrs["expiry"], float(chain.attrs["spot"]), chain.attrs["ts"]
        T = time_to_expiry(ts, exp)
        ks = chain.index.to_numpy(dtype=float)
        for side in ("ce", "pe"):
            right = side.upper()
            bid, ask, ltp, iv = (chain[f"{side}_{f}"].to_numpy(dtype=float) for f in ("bid", "ask", "ltp", "iv"))
            ok = (bid > 0) & (ask >= bid)
            m = np.where(ok, (bid + ask) / 2, ltp)
            half = np.where(ok, (ask - bid) / 2, np.maximum(0.05, 0.01 * np.nan_to_num(m)))
            near = (m > 0) & (np.abs(ks / S - 1) <= 0.08)
            # marks re-price at the IV the quote itself implies (the same one the entry was priced at);
            # the exchange's printed IV is only a fallback (see StrikePicker.rows)
            implied = np.full(len(ks), np.nan)
            implied[near] = self.pricer.implied_many(m[near], ks[near], right, S, T)
            for i, K in enumerate(ks):
                if not near[i]:
                    continue
                sym = Instrument.option(u, exp, float(K), right, lot_size).symbol
                v = implied[i]
                if not (v == v and v > 0) and iv[i] > 0:
                    v = iv[i] / 100
                if v == v and v > 0:
                    self.iv[sym] = v
                self.half[sym] = float(half[i])
        atm_i = int(np.abs(ks - S).argmin())
        ivs = [chain[c].to_numpy(dtype=float)[atm_i] for c in ("ce_iv", "pe_iv")]
        ivs = [x for x in ivs if x > 0]
        if ivs:
            self.atm_iv[u] = sum(ivs) / len(ivs) / 100

    def mid(self, inst: Instrument, S: float, now) -> float:
        T = time_to_expiry(now, inst.expiry)
        iv = self.iv.get(inst.symbol)
        if iv is None:
            atm = self.atm_iv.get(inst.underlying, 0.15)
            iv = self.pricer.iv_for(inst.strike, S, max(T, 1e-6), atm)
        return self.pricer.price(inst.strike, inst.right, S, T, iv)

    def bid_ask(self, inst: Instrument, S: float, now) -> tuple[float, float]:
        m = self.mid(inst, S, now)
        h = self.half.get(inst.symbol, max(0.05, 0.01 * m))
        return max(0.05, m - h), max(0.05, m + h)

    def exit_price(self, inst: Instrument, qty: int, S: float, now) -> float:
        """What closing `qty` (signed position) would fetch: longs sell at bid, shorts buy at ask."""
        b, a = self.bid_ask(inst, S, now)
        return b if qty > 0 else a


class IntradayBroker(PaperBroker):
    """Paper broker whose reference prices are already bid/ask, so no extra half-spread is
    added; `adverse_ticks` models queue/latency slippage on top."""

    def __init__(self, cfg, starting_cash=None, state_path=None, adverse_ticks: int = 1):
        super().__init__(cfg, starting_cash=starting_cash, state_path=state_path)
        self.costs.opt_half = 0.0
        self.costs.opt_min = 0.0
        self.costs.opt_half_ticks = adverse_ticks

    def execute(self, order, ref_price, ts, atr=None):
        if order.instrument.is_option:
            ref_price = ref_price + (1 if order.qty > 0 else -1) * self.costs.opt_half_ticks * self.costs.tick
            ref_price = max(ref_price, self.costs.tick)
        return super().execute(order, ref_price, ts, atr)
