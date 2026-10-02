"""Volatility smile models.

* `SkewModel` — a compact parametric smile in standardised moneyness
  x = ln(K/F) / (atm * sqrt(T)):  iv = atm * (1 + skew*x + curv*x^2), with floors.
  Index options in India show a persistent put skew (skew < 0), which this captures.
  Used when no live chain exists (backtests over history, the synthetic market).
* `SVI` — Gatheral's raw SVI total-variance slice, fitted to a real chain (e.g. from Kite).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass
class SkewModel:
    skew: float = -0.10
    curvature: float = 0.035
    floor: float = 0.55
    x_clip: float = 3.5
    term_slope: float = 0.0     # iv(T) = atm * (1 + term_slope * ln(T / 30d))

    def iv(self, atm: float, K, F: float, T: float):
        T = max(T, 1.0 / 365)
        atm_t = atm * (1 + self.term_slope * np.log(T / (30 / 365)))
        x = np.clip(np.log(np.asarray(K, dtype=float) / F) / (atm_t * np.sqrt(T)), -self.x_clip, self.x_clip)
        out = atm_t * np.maximum(1 + self.skew * x + self.curvature * x ** 2, self.floor)
        return float(out) if np.ndim(out) == 0 else out


@dataclass
class SVI:
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    T: float

    def total_variance(self, k):
        k = np.asarray(k, dtype=float)
        return self.a + self.b * (self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.sigma ** 2))

    def iv(self, K, F: float):
        k = np.log(np.asarray(K, dtype=float) / F)
        return np.sqrt(np.maximum(self.total_variance(k), 1e-10) / self.T)

    @classmethod
    def fit(cls, strikes, ivs, F: float, T: float, weights=None) -> "SVI":
        k = np.log(np.asarray(strikes, dtype=float) / F)
        w = np.asarray(ivs, dtype=float) ** 2 * T
        wts = np.ones_like(w) if weights is None else np.asarray(weights, dtype=float)
        atm_w = float(np.interp(0.0, np.sort(k), w[np.argsort(k)]))

        def resid(p):
            a, b, rho, m, s = p
            model = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s ** 2))
            return (model - w) * wts

        x0 = [atm_w * 0.5, 0.1, -0.4, 0.0, 0.1]
        res = least_squares(resid, x0, bounds=([-1, 1e-6, -0.999, -1, 1e-4], [1, 5, 0.999, 1, 2]))
        a, b, rho, m, s = res.x
        return cls(a, b, rho, m, s, T)

    def sane(self) -> bool:
        """Necessary conditions: Roger Lee's wing bound on total variance and a
        non-negative minimum variance. (Not a full butterfly-arbitrage proof.)"""
        return self.b * (1 + abs(self.rho)) <= 4 and self.a + self.b * self.sigma * np.sqrt(1 - self.rho ** 2) >= 0
