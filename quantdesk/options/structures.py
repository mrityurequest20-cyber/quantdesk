"""Multi-leg option structures: builders (delta-targeted strikes) and analysis
(payoff, max profit/loss, breakevens, net Greeks, probability of profit, and the
expected P&L under *your* volatility forecast — the quantity that says whether the
market price is rich or cheap)."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from ..core.calendar import year_fraction
from ..core.types import Instrument, LegSpec
from .chain import OptionPricer
from .pricing import strike_for_delta


@dataclass
class Structure:
    name: str
    underlying: str
    expiry: dt.date
    legs: list[LegSpec]
    prices: dict[str, float]            # model / quote mid per leg symbol at construction
    spot: float
    asof: object
    atm_iv: float
    notes: dict = field(default_factory=dict)

    @property
    def lot_size(self) -> int:
        return self.legs[0].instrument.lot_size

    def net_premium(self, prices: dict[str, float] | None = None) -> float:
        """Per structure unit in INR. Positive = debit paid, negative = credit received."""
        prices = prices or self.prices
        return sum(l.ratio * l.instrument.lot_size * prices[l.instrument.symbol] for l in self.legs)

    def payoff(self, S_T) -> np.ndarray:
        """Expiry value per unit (excluding the premium)."""
        S_T = np.asarray(S_T, dtype=float)
        v = np.zeros_like(S_T)
        for l in self.legs:
            i = l.instrument
            intr = np.maximum(S_T - i.strike, 0) if i.right == "CE" else np.maximum(i.strike - S_T, 0)
            v += l.ratio * i.lot_size * intr
        return v

    def pnl_at_expiry(self, S_T) -> np.ndarray:
        return self.payoff(S_T) - self.net_premium()

    def _grid(self) -> np.ndarray:
        ks = [l.instrument.strike for l in self.legs]
        lo, hi = min(ks + [self.spot]) * 0.6, max(ks + [self.spot]) * 1.4
        return np.unique(np.concatenate([np.linspace(lo, hi, 4001), ks]))

    def max_profit(self) -> float:
        g = self._grid()
        tail_slope = self.pnl_at_expiry([g[-1] * 2])[0] - self.pnl_at_expiry([g[-1]])[0]
        return float("inf") if tail_slope > 1e-6 else float(self.pnl_at_expiry(g).max())

    def max_loss(self) -> float:
        """Positive number: the most you can lose per unit at expiry."""
        g = self._grid()
        tail_up = self.pnl_at_expiry([g[-1] * 2])[0] - self.pnl_at_expiry([g[-1]])[0]
        if tail_up < -1e-6:
            return float("inf")
        low_end = self.pnl_at_expiry([1e-6])[0]
        return float(max(0.0, -min(self.pnl_at_expiry(g).min(), low_end)))

    def breakevens(self) -> list[float]:
        g = self._grid()
        p = self.pnl_at_expiry(g)
        s = np.sign(p)
        idx = np.where(np.diff(s) != 0)[0]
        return [float(g[i] - p[i] * (g[i + 1] - g[i]) / (p[i + 1] - p[i])) for i in idx]

    def greeks(self, pricer: OptionPricer, S: float, asof, atm_iv: float) -> dict:
        tot = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
        for l in self.legs:
            g = pricer.greeks(l.instrument, S, asof, atm_iv)
            for k in tot:
                tot[k] += l.ratio * l.instrument.lot_size * g[k]
        return tot

    def _density(self, vol: float, mu: float, T: float):
        g = self._grid()
        sd = vol * np.sqrt(max(T, 1e-9))
        m = np.log(self.spot) + (mu - 0.5 * vol ** 2) * T
        x = np.log(g)
        cdf = norm.cdf((x - m) / sd)
        w = np.diff(cdf, prepend=0.0)
        w[-1] += 1 - cdf[-1]
        return g, w

    def prob_profit(self, vol: float, mu: float = 0.0) -> float:
        T = year_fraction(self.asof, self.expiry)
        g, w = self._density(vol, mu, T)
        return float(w[self.pnl_at_expiry(g) > 0].sum())

    def expected_pnl(self, vol: float, mu: float = 0.0) -> float:
        """E[P&L at expiry] per unit if the underlying is lognormal with `vol` (decimal)."""
        T = year_fraction(self.asof, self.expiry)
        g, w = self._density(vol, mu, T)
        return float((self.pnl_at_expiry(g) * w).sum())

    def describe(self) -> str:
        legs = ", ".join(f"{'+' if l.ratio > 0 else ''}{l.ratio} {l.instrument.strike:g}{l.instrument.right}"
                         f"@{self.prices[l.instrument.symbol]:.2f}" for l in self.legs)
        prem = self.net_premium()
        kind = "debit" if prem > 0 else "credit"
        money = lambda x: "unlimited" if not np.isfinite(x) else f"₹{x:,.0f}"
        return (f"{self.name} {self.underlying} exp {self.expiry}: {legs} | {kind} ₹{abs(prem):,.0f}/unit, "
                f"max loss {money(self.max_loss())}, max profit {money(self.max_profit())}, "
                f"BE {', '.join(f'{b:,.0f}' for b in self.breakevens())}")


class StructureBuilder:
    def __init__(self, pricer: OptionPricer, underlying: str, lot_size: int, strike_step: float):
        self.pricer = pricer
        self.u = underlying
        self.lot = lot_size
        self.step = strike_step

    def _round(self, K: float) -> float:
        return round(K / self.step) * self.step

    def strike(self, S, asof, expiry, atm_iv, delta, right) -> float:
        T = max(year_fraction(asof, expiry), 1 / 365)
        K = strike_for_delta(S, T, self.pricer.r, self.pricer.q, delta, right,
                             vol_fn=lambda k: self.pricer.iv(k, S, T, atm_iv))
        return self._round(K)

    def _make(self, name, S, asof, expiry, atm_iv, spec: list[tuple[float, str, int]], **notes) -> Structure:
        legs, prices = [], {}
        for K, right, ratio in spec:
            inst = Instrument.option(self.u, expiry, K, right, self.lot)
            legs.append(LegSpec(inst, ratio))
            prices[inst.symbol] = self.pricer.price(inst, S, asof, atm_iv)
        return Structure(name, self.u, expiry, legs, prices, S, asof, atm_iv, notes)

    def iron_condor(self, S, asof, expiry, atm_iv, short_delta=0.16, wing_delta=0.05) -> Structure:
        sp = self.strike(S, asof, expiry, atm_iv, short_delta, "PE")
        sc = self.strike(S, asof, expiry, atm_iv, short_delta, "CE")
        lp = min(self.strike(S, asof, expiry, atm_iv, wing_delta, "PE"), sp - self.step)
        lc = max(self.strike(S, asof, expiry, atm_iv, wing_delta, "CE"), sc + self.step)
        return self._make("iron_condor", S, asof, expiry, atm_iv,
                          [(lp, "PE", 1), (sp, "PE", -1), (sc, "CE", -1), (lc, "CE", 1)])

    def iron_fly(self, S, asof, expiry, atm_iv, wing_delta=0.10) -> Structure:
        atm = self._round(S)
        lp = min(self.strike(S, asof, expiry, atm_iv, wing_delta, "PE"), atm - self.step)
        lc = max(self.strike(S, asof, expiry, atm_iv, wing_delta, "CE"), atm + self.step)
        return self._make("iron_fly", S, asof, expiry, atm_iv,
                          [(lp, "PE", 1), (atm, "PE", -1), (atm, "CE", -1), (lc, "CE", 1)])

    def bull_call_spread(self, S, asof, expiry, atm_iv, long_delta=0.55, short_delta=0.25) -> Structure:
        kl = self.strike(S, asof, expiry, atm_iv, long_delta, "CE")
        ks = max(self.strike(S, asof, expiry, atm_iv, short_delta, "CE"), kl + self.step)
        return self._make("bull_call_spread", S, asof, expiry, atm_iv, [(kl, "CE", 1), (ks, "CE", -1)])

    def bear_put_spread(self, S, asof, expiry, atm_iv, long_delta=0.55, short_delta=0.25) -> Structure:
        kl = self.strike(S, asof, expiry, atm_iv, long_delta, "PE")
        ks = min(self.strike(S, asof, expiry, atm_iv, short_delta, "PE"), kl - self.step)
        return self._make("bear_put_spread", S, asof, expiry, atm_iv, [(kl, "PE", 1), (ks, "PE", -1)])

    def long_straddle(self, S, asof, expiry, atm_iv) -> Structure:
        k = self._round(S)
        return self._make("long_straddle", S, asof, expiry, atm_iv, [(k, "CE", 1), (k, "PE", 1)])

    def long_strangle(self, S, asof, expiry, atm_iv, delta=0.30) -> Structure:
        return self._make("long_strangle", S, asof, expiry, atm_iv,
                          [(self.strike(S, asof, expiry, atm_iv, delta, "CE"), "CE", 1),
                           (self.strike(S, asof, expiry, atm_iv, delta, "PE"), "PE", 1)])

    def short_strangle(self, S, asof, expiry, atm_iv, delta=0.16) -> Structure:
        """Undefined risk. Available for analysis; the default strategies never sell naked."""
        return self._make("short_strangle", S, asof, expiry, atm_iv,
                          [(self.strike(S, asof, expiry, atm_iv, delta, "CE"), "CE", -1),
                           (self.strike(S, asof, expiry, atm_iv, delta, "PE"), "PE", -1)])
