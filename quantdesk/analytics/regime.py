"""Market regime detection.

Two complementary views:
1. A Gaussian hidden Markov model on daily returns (Baum-Welch, scaled forward-backward),
   run *causally*: parameters are re-estimated every `refit_every` bars on past data and
   only forward-filtered probabilities are used. State 0 = calm, last state = turbulent.
2. A rule-based trend/volatility classifier (ADX, SMA200, EMA slope, vol percentile, VIX).

They combine into one label per bar: trending_up | trending_down | range | stressed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from .stats import rolling_hurst
from .volatility import close_to_close

REGIMES = ("trending_up", "trending_down", "range", "stressed")


class GaussianHMM:
    def __init__(self, n_states: int = 2, n_iter: int = 60, tol: float = 1e-5, seed: int = 0):
        self.k = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.rng = np.random.default_rng(seed)
        self.pi = self.A = self.mu = self.sd = None

    def _emission(self, x: np.ndarray) -> np.ndarray:
        z = (x[:, None] - self.mu[None, :]) / self.sd[None, :]
        return np.exp(-0.5 * z ** 2) / (self.sd[None, :] * np.sqrt(2 * np.pi)) + 1e-300

    def _forward(self, B: np.ndarray):
        T = len(B)
        alpha = np.empty((T, self.k))
        c = np.empty(T)
        a = self.pi * B[0]
        c[0] = a.sum()
        alpha[0] = a / c[0]
        A = self.A
        for t in range(1, T):
            a = (alpha[t - 1] @ A) * B[t]
            c[t] = a.sum()
            alpha[t] = a / c[t]
        return alpha, c

    def _backward(self, B: np.ndarray, c: np.ndarray):
        T = len(B)
        beta = np.empty((T, self.k))
        beta[-1] = 1.0
        A = self.A
        for t in range(T - 2, -1, -1):
            beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
        return beta

    def fit(self, x: np.ndarray, warm: "GaussianHMM | None" = None) -> "GaussianHMM":
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if warm is not None and warm.mu is not None:
            self.pi, self.A, self.mu, self.sd = warm.pi.copy(), warm.A.copy(), warm.mu.copy(), warm.sd.copy()
        else:
            q = np.quantile(np.abs(x - x.mean()), np.linspace(0.3, 0.95, self.k))
            self.mu = np.full(self.k, x.mean())
            self.sd = np.maximum(q, 1e-6)
            self.A = np.full((self.k, self.k), 0.05 / max(1, self.k - 1))
            np.fill_diagonal(self.A, 0.95)
            self.pi = np.full(self.k, 1.0 / self.k)
        prev = -np.inf
        for _ in range(self.n_iter):
            B = self._emission(x)
            alpha, c = self._forward(B)
            beta = self._backward(B, c)
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)
            xi = (alpha[:-1, :, None] * self.A[None] * (B[1:] * beta[1:])[:, None, :]) / c[1:, None, None]
            self.pi = gamma[0]
            self.A = xi.sum(axis=0) / gamma[:-1].sum(axis=0)[:, None]
            self.A /= self.A.sum(axis=1, keepdims=True)
            w = gamma.sum(axis=0)
            self.mu = (gamma * x[:, None]).sum(axis=0) / w
            self.sd = np.sqrt((gamma * (x[:, None] - self.mu) ** 2).sum(axis=0) / w)
            self.sd = np.maximum(self.sd, 1e-6)
            ll = np.log(c).sum()
            if ll - prev < self.tol:
                break
            prev = ll
        order = np.argsort(self.sd)          # state 0 = lowest vol
        self.mu, self.sd, self.pi = self.mu[order], self.sd[order], self.pi[order]
        self.A = self.A[np.ix_(order, order)]
        self.loglik = prev
        return self

    def filter(self, x: np.ndarray) -> np.ndarray:
        """P(state_t | x_0..x_t) — no future information."""
        alpha, _ = self._forward(self._emission(np.nan_to_num(np.asarray(x, dtype=float))))
        return alpha

    def expected_duration(self) -> np.ndarray:
        return 1.0 / np.maximum(1e-9, 1 - np.diag(self.A))


def rolling_hmm_stress(close: pd.Series, n_states: int = 2, refit_every: int = 63,
                       min_history: int = 500, window: int = 1500) -> pd.Series:
    """Causal P(turbulent state) per bar."""
    r = np.log(close).diff().fillna(0.0).to_numpy() * 100
    out = np.full(len(r), np.nan)
    model = None
    for k in range(min_history, len(r), refit_every):
        lo = max(1, k + 1 - window)
        model = GaussianHMM(n_states).fit(r[lo:k + 1], warm=model)
        end = min(len(r), k + refit_every)
        probs = model.filter(r[lo:end])
        out[k:end] = probs[k - lo:end - lo, -1]
    return pd.Series(out, index=close.index, name="p_stress")


def regime_frame(df: pd.DataFrame, vix: pd.Series | None = None, cfg=None) -> pd.DataFrame:
    """Per-bar regime features and label for an index (causal)."""
    get = (lambda k, d: cfg.get(k, d)) if cfg is not None else (lambda k, d: d)
    close = df["close"]
    out = pd.DataFrame(index=df.index)
    out["close"] = close
    out["sma200"] = ind.sma(close, 200)
    out["ema50"] = ind.ema(close, 50)
    out["ema50_slope"] = out["ema50"].pct_change(10)
    out = out.join(ind.adx(df, 14))
    out["er"] = ind.efficiency_ratio(close, 20)
    out["rv21"] = close_to_close(close, 21)
    out["vol_pct"] = ind.pct_rank(out["rv21"], get("regime.vol_lookback", 252))
    out["hurst"] = rolling_hurst(np.log(close), 100)
    out["p_stress"] = rolling_hmm_stress(close, get("regime.hmm_states", 2), get("regime.hmm_refit_every", 63),
                                         get("regime.hmm_min_history", 500))
    if vix is not None:
        out["vix"] = vix.reindex(df.index).ffill()
    vix_extreme = get("checks.vix_extreme", 30)

    stressed = ((out["p_stress"].fillna(0) > 0.7) & (out["vol_pct"].fillna(0) > 0.8))
    if "vix" in out:
        stressed |= out["vix"] > vix_extreme
    up = (close > out["sma200"]) & (out["ema50_slope"] > 0) & (out["adx"] > 18)
    down = (close < out["sma200"]) & (out["ema50_slope"] < 0) & (out["adx"] > 18)
    label = np.where(stressed, "stressed", np.where(up, "trending_up", np.where(down, "trending_down", "range")))
    out["label"] = pd.Series(label, index=df.index).where(out["sma200"].notna(), "range")
    # A single signed score for dashboards: + trending up, - down, magnitude ~ conviction.
    out["trend_score"] = (np.sign(close - out["sma200"]) * (out["adx"] / 40).clip(upper=1.5)
                          * (0.5 + out["er"].fillna(0))).fillna(0)
    return out
