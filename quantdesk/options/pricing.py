"""Black-Scholes-Merton / Black-76 pricing, Greeks, implied volatility, delta-to-strike.

Conventions: sigma as a decimal (0.14), T in years (ACT/365), r and q continuous.
Greeks are per-unit-of-underlying: vega per 1 vol point, theta per calendar day,
rho per 1% rate move. NSE index and stock options are European, so BSM applies directly.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr as _N          # the standard normal CDF as a plain ufunc: same values as
                                               # scipy.stats.norm.cdf, without its per-call overhead (~30× faster)

_SQRT2PI = np.sqrt(2 * np.pi)


def _d1d2(S, K, T, r, q, sigma):
    S, K, T, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, sigma))
    sqT = np.sqrt(np.maximum(T, 1e-12))
    sig = np.maximum(sigma, 1e-8)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig ** 2) * T) / (sig * sqT)
    return d1, d1 - sig * sqT, sqT


def bs_price(S, K, T, r, q, sigma, right: str):
    """Vectorised European option price. right: 'CE'/'C' call, 'PE'/'P' put."""
    call = right.upper().startswith("C")
    S_, K_, T_ = (np.asarray(x, dtype=float) for x in (S, K, T))
    d1, d2, _ = _d1d2(S, K, T, r, q, sigma)
    df_r, df_q = np.exp(-r * T_), np.exp(-q * T_)
    if call:
        px = S_ * df_q * _N(d1) - K_ * df_r * _N(d2)
        intrinsic = np.maximum(S_ - K_, 0.0)
    else:
        px = K_ * df_r * _N(-d2) - S_ * df_q * _N(-d1)
        intrinsic = np.maximum(K_ - S_, 0.0)
    out = np.where(T_ <= 1e-10, intrinsic, np.maximum(px, 0.0))
    return float(out) if np.ndim(out) == 0 else out


def black76_price(F, K, T, r, sigma, right: str):
    """Options on futures: BSM with q = r and S = F."""
    return bs_price(F, K, T, r, r, sigma, right)


def greeks(S, K, T, r, q, sigma, right: str) -> dict:
    call = right.upper().startswith("C")
    S_, K_, T_ = (np.asarray(x, dtype=float) for x in (S, K, T))
    d1, d2, sqT = _d1d2(S, K, T, r, q, sigma)
    sig = np.maximum(np.asarray(sigma, dtype=float), 1e-8)
    df_r, df_q = np.exp(-r * T_), np.exp(-q * T_)
    pdf = np.exp(-0.5 * d1 ** 2) / _SQRT2PI
    gamma = df_q * pdf / (S_ * sig * sqT)
    vega = S_ * df_q * pdf * sqT / 100
    if call:
        delta = df_q * _N(d1)
        theta = (-S_ * df_q * pdf * sig / (2 * sqT) - r * K_ * df_r * _N(d2) + q * S_ * df_q * _N(d1)) / 365
        rho = K_ * T_ * df_r * _N(d2) / 100
    else:
        delta = -df_q * _N(-d1)
        theta = (-S_ * df_q * pdf * sig / (2 * sqT) + r * K_ * df_r * _N(-d2) - q * S_ * df_q * _N(-d1)) / 365
        rho = -K_ * T_ * df_r * _N(-d2) / 100
    expired = T_ <= 1e-10
    if np.any(expired):
        itm = (S_ > K_) if call else (S_ < K_)
        delta = np.where(expired, np.where(itm, 1.0 if call else -1.0, 0.0), delta)
        gamma = np.where(expired, 0.0, gamma)
        vega = np.where(expired, 0.0, vega)
        theta = np.where(expired, 0.0, theta)
        rho = np.where(expired, 0.0, rho)
    res = {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
    return {k: float(v) if np.ndim(v) == 0 else v for k, v in res.items()}


def implied_vol(price: float, S: float, K: float, T: float, r: float, q: float, right: str,
                lo: float = 1e-4, hi: float = 5.0) -> float:
    """Newton on vega, falling back to Brent. NaN if the price violates no-arbitrage bounds."""
    if T <= 0:
        return float("nan")
    call = right.upper().startswith("C")
    fwd_intr = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0) if call else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
    upper = S * np.exp(-q * T) if call else K * np.exp(-r * T)
    if not (fwd_intr - 1e-9 <= price <= upper + 1e-9):
        return float("nan")
    if price - fwd_intr < 1e-10:
        return lo
    sig = 0.2
    for _ in range(50):
        diff = bs_price(S, K, T, r, q, sig, right) - price
        if abs(diff) < 1e-8:
            return float(sig)
        v = greeks(S, K, T, r, q, sig, right)["vega"] * 100
        if v < 1e-10:
            break
        step = diff / v
        sig_new = sig - step
        if not (lo < sig_new < hi):
            break
        sig = sig_new
    f = lambda s: bs_price(S, K, T, r, q, s, right) - price
    try:
        return float(brentq(f, lo, hi, xtol=1e-10))
    except ValueError:
        return float("nan")


def implied_vol_vec(prices, S: float, K, T: float, r: float, q: float, right: str,
                    lo: float = 1e-4, hi: float = 5.0) -> np.ndarray:
    """Implied vols for many strikes of one expiry at once: Newton on vega, then bisection for anything
    that didn't converge (the price is monotonic in vol, so bisection always does). NaN where a price
    violates the no-arbitrage bounds. Same answers as `implied_vol`, for a whole chain in one pass."""
    P, K = np.asarray(prices, dtype=float), np.asarray(K, dtype=float)
    out = np.full(P.shape, np.nan)
    if T <= 0 or P.size == 0:
        return out
    call = right.upper().startswith("C")
    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    intr = np.maximum(S * df_q - K * df_r, 0) if call else np.maximum(K * df_r - S * df_q, 0)
    upper = np.full(P.shape, S * df_q) if call else K * df_r
    ok = np.isfinite(P) & (P >= intr - 1e-9) & (P <= upper + 1e-9)
    floor = ok & (P - intr < 1e-10)
    out[floor] = lo
    live = ok & ~floor
    if not live.any():
        return out
    p, k = P[live], K[live]
    sig = np.clip(np.sqrt(2 * np.pi / T) * p / S, 0.05, 2.0)           # Brenner–Subrahmanyam start
    done = np.zeros(p.shape, dtype=bool)
    for _ in range(30):
        px = np.asarray(bs_price(S, k, T, r, q, sig, right), dtype=float)
        diff = px - p
        done |= np.abs(diff) < 1e-8
        d1, _, sqT = _d1d2(S, k, T, r, q, sig)
        vega = S * df_q * np.exp(-0.5 * d1 ** 2) / _SQRT2PI * sqT
        step = np.where(done | (vega < 1e-10), 0.0, diff / np.maximum(vega, 1e-10))
        nxt = sig - step
        bad = (nxt <= lo) | (nxt >= hi) | ~np.isfinite(nxt)
        sig = np.where(bad, sig, nxt)
        stuck = bad & ~done
        if done.all() or stuck.all():
            break
    px = np.asarray(bs_price(S, k, T, r, q, sig, right), dtype=float)
    need = np.abs(px - p) >= 1e-6
    if need.any():                                                     # bisection: slow but certain
        a, b = np.full(need.sum(), lo), np.full(need.sum(), hi)
        pk, kk = p[need], k[need]
        for _ in range(70):
            m = 0.5 * (a + b)
            above = np.asarray(bs_price(S, kk, T, r, q, m, right), dtype=float) > pk
            b, a = np.where(above, m, b), np.where(above, a, m)
        sig[need] = 0.5 * (a + b)
    out[np.flatnonzero(live)] = sig
    return out


def strike_for_delta(S: float, T: float, r: float, q: float, target_delta: float, right: str,
                     vol_fn=None, sigma: float = 0.15) -> float:
    """Strike whose |delta| equals target_delta; `vol_fn(K)` lets the smile move the answer."""
    vol_fn = vol_fn or (lambda K: sigma)
    call = right.upper().startswith("C")
    tgt = abs(target_delta)

    def f(K):
        d = greeks(S, K, T, r, q, vol_fn(K), right)["delta"]
        return abs(d) - tgt

    lo, hi = S * 0.3, S * 3.0
    try:
        return float(brentq(f, lo, hi, xtol=1e-6 * S))
    except ValueError:
        return S * (1.1 if call else 0.9)
