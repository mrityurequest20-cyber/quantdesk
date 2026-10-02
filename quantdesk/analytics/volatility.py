"""Volatility: range-based realised estimators, EWMA, GARCH(1,1) with a causal rolling
forecast, IV rank, and the volatility cone. Vol is quoted in annualised *vol points*
(e.g. 14.2) so it compares directly with India VIX."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

ANN = 252


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def close_to_close(close: pd.Series, n: int = 21) -> pd.Series:
    return log_returns(close).rolling(n, min_periods=n).std() * np.sqrt(ANN) * 100


def parkinson(df: pd.DataFrame, n: int = 21) -> pd.Series:
    hl = np.log(df["high"] / df["low"]) ** 2
    return np.sqrt(hl.rolling(n, min_periods=n).mean() / (4 * np.log(2)) * ANN) * 100


def garman_klass(df: pd.DataFrame, n: int = 21) -> pd.Series:
    hl = np.log(df["high"] / df["low"]) ** 2
    co = np.log(df["close"] / df["open"]) ** 2
    v = (0.5 * hl - (2 * np.log(2) - 1) * co).rolling(n, min_periods=n).mean()
    return np.sqrt(v.clip(lower=0) * ANN) * 100


def rogers_satchell(df: pd.DataFrame, n: int = 21) -> pd.Series:
    h, l, o, c = (np.log(df[k]) for k in ("high", "low", "open", "close"))
    rs = (h - c) * (h - o) + (l - c) * (l - o)
    return np.sqrt(rs.rolling(n, min_periods=n).mean().clip(lower=0) * ANN) * 100


def yang_zhang(df: pd.DataFrame, n: int = 21) -> pd.Series:
    """Drift-independent, handles opening gaps; the most efficient of the OHLC estimators."""
    o, h, l, c = (np.log(df[k]) for k in ("open", "high", "low", "close"))
    overnight = o - c.shift(1)
    open_close = c - o
    rs = (h - c) * (h - o) + (l - c) * (l - o)
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var = (overnight.rolling(n, min_periods=n).var()
           + k * open_close.rolling(n, min_periods=n).var()
           + (1 - k) * rs.rolling(n, min_periods=n).mean())
    return np.sqrt(var.clip(lower=0) * ANN) * 100


def ewma_vol(close: pd.Series, lam: float = 0.94) -> pd.Series:
    r2 = log_returns(close) ** 2
    return np.sqrt(r2.ewm(alpha=1 - lam, adjust=False, min_periods=20).mean() * ANN) * 100


# ---- GARCH(1,1) --------------------------------------------------------------------------
@dataclass
class GarchParams:
    omega: float
    alpha: float
    beta: float

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def long_run_var(self) -> float:
        return self.omega / max(1e-9, 1 - self.persistence)

    @property
    def half_life(self) -> float:
        p = self.persistence
        return np.log(0.5) / np.log(p) if 0 < p < 1 else np.inf


class Garch11:
    """Gaussian (quasi-)maximum-likelihood GARCH(1,1) on percentage returns.

    sigma2[t] = omega + alpha * r[t-1]^2 + beta * sigma2[t-1]
    The recursion is a first-order IIR filter, so it runs vectorised through lfilter.
    """

    def __init__(self, params: GarchParams | None = None):
        self.params = params

    @staticmethod
    def _variance(r: np.ndarray, omega: float, alpha: float, beta: float, s0: float) -> np.ndarray:
        x = np.empty_like(r)
        x[0] = s0
        x[1:] = omega + alpha * r[:-1] ** 2
        return lfilter([1.0], [1.0, -beta], x)

    def fit(self, returns_pct: np.ndarray, start: GarchParams | None = None) -> GarchParams:
        r = np.asarray(returns_pct, dtype=float)
        r = r[np.isfinite(r)]
        r = r - r.mean()
        v = r.var()
        s0 = v

        def nll(theta):
            omega, alpha, beta = theta
            if alpha + beta >= 0.9995:
                return 1e10
            s2 = self._variance(r, omega, alpha, beta, s0)
            s2 = np.maximum(s2, 1e-10)
            return 0.5 * np.sum(np.log(s2) + r ** 2 / s2)

        x0 = [start.omega, start.alpha, start.beta] if start else [v * 0.05, 0.08, 0.87]
        res = minimize(nll, x0, method="L-BFGS-B",
                       bounds=[(1e-8, 10 * v), (1e-6, 0.5), (0.3, 0.9989)])
        self.params = GarchParams(*res.x)
        self._s0 = s0
        return self.params

    def conditional_variance(self, returns_pct: np.ndarray) -> np.ndarray:
        p = self.params
        r = np.asarray(returns_pct, dtype=float)
        s0 = getattr(self, "_s0", np.nanvar(r))
        return self._variance(np.nan_to_num(r), p.omega, p.alpha, p.beta, s0)

    def next_variance(self, returns_pct: np.ndarray) -> float:
        """One-step-ahead variance after observing the last return."""
        p = self.params
        s2 = self.conditional_variance(returns_pct)
        return p.omega + p.alpha * returns_pct[-1] ** 2 + p.beta * s2[-1]

    def term_structure(self, next_var: float, horizon: int) -> np.ndarray:
        p = self.params
        k = np.arange(horizon)
        return p.long_run_var + p.persistence ** k * (next_var - p.long_run_var)

    def forecast_vol(self, returns_pct: np.ndarray, horizon: int = 21) -> float:
        """Annualised vol points expected over the next `horizon` bars."""
        path = self.term_structure(self.next_variance(returns_pct), horizon)
        return float(np.sqrt(path.mean() * ANN))


def rolling_garch_forecast(close: pd.Series, horizon: int = 21, refit_every: int = 63,
                           min_obs: int = 500, window: int = 1500) -> pd.Series:
    """Causal GARCH forecast: params refit every `refit_every` bars on trailing data only;
    between refits the variance is filtered forward with fixed params."""
    r = (log_returns(close) * 100).to_numpy()
    out = np.full(len(r), np.nan)
    model, params = Garch11(), None
    for k in range(min_obs, len(r), refit_every):
        lo = max(1, k + 1 - window)
        params = model.fit(r[lo:k + 1], start=params)
        end = min(len(r), k + refit_every)
        seg = r[lo:end]
        s2 = model.conditional_variance(seg)
        p = params
        nxt = p.omega + p.alpha * seg ** 2 + p.beta * s2          # forecast for t+1 made at t
        idx = np.arange(k, end) - lo
        pers = p.persistence
        lr = p.long_run_var
        mult = (1 - pers ** horizon) / (horizon * (1 - pers)) if pers < 1 else 1.0
        avg = lr + (nxt[idx] - lr) * mult
        out[k:end] = np.sqrt(np.maximum(avg, 0) * ANN)
    return pd.Series(out, index=close.index, name="garch_vol")


def iv_rank(iv: pd.Series, n: int = 252) -> pd.Series:
    lo = iv.rolling(n, min_periods=60).min()
    hi = iv.rolling(n, min_periods=60).max()
    return (iv - lo) / (hi - lo).replace(0, np.nan)


def iv_percentile(iv: pd.Series, n: int = 252) -> pd.Series:
    return iv.rolling(n, min_periods=60).rank(pct=True)


def vol_cone(close: pd.Series, windows=(10, 21, 42, 63, 126, 252)) -> pd.DataFrame:
    rows = []
    for w in windows:
        rv = close_to_close(close, w).dropna()
        if rv.empty:
            continue
        rows.append({"window": w, "min": rv.min(), "p10": rv.quantile(0.10), "p25": rv.quantile(0.25),
                     "median": rv.median(), "p75": rv.quantile(0.75), "p90": rv.quantile(0.90),
                     "max": rv.max(), "current": rv.iloc[-1]})
    return pd.DataFrame(rows).set_index("window")
