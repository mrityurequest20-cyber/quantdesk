"""Time-series statistics used for strategy selection: ADF / Engle-Granger cointegration
(MacKinnon p-values), OU half-life, Hurst exponent, variance ratio, and a Kalman-filter
dynamic hedge ratio for pairs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

# MacKinnon (1994) response-surface coefficients (as in statsmodels' adfvalues),
# "c" (constant) case, index 0 = plain ADF (N=1), index 1 = Engle-Granger with 2 variables.
_TAU_MAX_C = [2.74, 0.92]
_TAU_MIN_C = [-18.83, -18.86]
_TAU_STAR_C = [-1.61, -2.62]
_TAU_C_SMALLP = [[2.1659, 1.4412, 0.038269], [2.92, 1.5012, 0.039796]]
_TAU_C_LARGEP = [[1.7339, 0.93202 * 1e-1, -0.12745 * 1e-1, -0.010368 * 1e-2],
                 [2.1945, 0.64695 * 1e-1, -0.29198 * 1e-1, -0.042377 * 1e-2]]
CRIT_C = {1: {"1%": -3.43, "5%": -2.86, "10%": -2.57}, 2: {"1%": -3.90, "5%": -3.34, "10%": -3.04}}


def mackinnon_p(stat: float, n_vars: int = 1) -> float:
    i = n_vars - 1
    if stat > _TAU_MAX_C[i]:
        return 1.0
    if stat < _TAU_MIN_C[i]:
        return 0.0
    coefs = _TAU_C_SMALLP[i] if stat <= _TAU_STAR_C[i] else _TAU_C_LARGEP[i]
    return float(norm.cdf(sum(c * stat ** k for k, c in enumerate(coefs))))


def _ols(y: np.ndarray, X: np.ndarray):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(1, len(y) - X.shape[1])
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    return beta, resid, np.sqrt(np.diag(cov)), s2


@dataclass
class ADFResult:
    stat: float
    pvalue: float
    lags: int
    nobs: int
    crit: dict

    @property
    def stationary(self) -> bool:
        return self.pvalue < 0.05


def adf(series, max_lag: int | None = None, constant: bool = True, n_vars: int = 1) -> ADFResult:
    """Augmented Dickey-Fuller with AIC lag selection on a common sample."""
    y = np.asarray(series, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if max_lag is None:
        max_lag = int(np.ceil(12 * (n / 100) ** 0.25))
    max_lag = max(0, min(max_lag, n // 2 - 3))
    dy = np.diff(y)

    def design(p, start):
        rows = range(start, len(dy))
        cols = [y[start:len(dy)]]  # y_{t-1} aligned with dy_t (dy[t] = y[t+1]-y[t])
        for i in range(1, p + 1):
            cols.append(dy[start - i:len(dy) - i])
        X = np.column_stack(cols)
        if constant:
            X = np.column_stack([X, np.ones(len(X))])
        return dy[start:], X, len(rows)

    best, best_aic = 0, np.inf
    for p in range(max_lag + 1):
        yy, X, m = design(p, max_lag)
        _, resid, _, _ = _ols(yy, X)
        aic = m * np.log(resid @ resid / m) + 2 * X.shape[1]
        if aic < best_aic:
            best, best_aic = p, aic
    yy, X, m = design(best, best)
    beta, _, se, _ = _ols(yy, X)
    stat = float(beta[0] / se[0])
    return ADFResult(stat, mackinnon_p(stat, n_vars), best, m, CRIT_C[n_vars])


@dataclass
class CointResult:
    hedge_ratio: float
    intercept: float
    adf: ADFResult
    half_life: float
    spread: pd.Series

    @property
    def pvalue(self) -> float:
        return self.adf.pvalue


def engle_granger(y: pd.Series, x: pd.Series) -> CointResult:
    """Step 1: OLS y = a + b x. Step 2: ADF (no constant) on residuals, EG critical values."""
    df = pd.concat([y, x], axis=1).dropna()
    yy, xx = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    beta, resid, _, _ = _ols(yy, np.column_stack([xx, np.ones(len(xx))]))
    r = adf(resid, constant=False, n_vars=2)
    spread = pd.Series(resid, index=df.index)
    return CointResult(float(beta[0]), float(beta[1]), r, half_life(spread), spread)


def half_life(series) -> float:
    """OU half-life from dy_t = a + lambda * y_{t-1}: HL = -ln 2 / lambda."""
    y = np.asarray(series, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 10:
        return np.inf
    beta, *_ = _ols(np.diff(y), np.column_stack([y[:-1], np.ones(len(y) - 1)]))
    lam = beta[0]
    return float(-np.log(2) / lam) if lam < 0 else np.inf


def hurst(series, lags=range(2, 21)) -> float:
    """Hurst exponent from the scaling of lagged differences: std(x[t+l]-x[t]) ~ l^H.
    H < 0.5 mean-reverting, 0.5 random walk, > 0.5 trending. Use on log prices."""
    x = np.asarray(series, dtype=float)
    lags = [l for l in lags if l < len(x) // 2]
    tau = [np.std(x[l:] - x[:-l]) for l in lags]
    return float(np.polyfit(np.log(lags), np.log(tau), 1)[0])


def rolling_hurst(log_price: pd.Series, window: int = 100, lags=range(2, 21)) -> pd.Series:
    """Vectorised rolling Hurst: per-bar regression of log rolling-std on log lag."""
    lags = np.array(list(lags))
    cols = [np.log(log_price.diff(l).rolling(window, min_periods=window).std()) for l in lags]
    M = np.column_stack(cols)
    xl = np.log(lags)
    xc = xl - xl.mean()
    slope = (M - M.mean(axis=1, keepdims=True)) @ xc / (xc @ xc)
    return pd.Series(slope, index=log_price.index, name="hurst")


def variance_ratio(log_price, q: int = 5) -> tuple[float, float]:
    """Lo-MacKinlay VR(q) with the heteroskedasticity-robust z statistic.
    VR > 1 momentum, VR < 1 mean reversion. Returns (VR, z)."""
    p = np.asarray(log_price, dtype=float)
    r = np.diff(p)
    n = len(r)
    mu = r.mean()
    var1 = ((r - mu) ** 2).sum() / (n - 1)
    rq = p[q:] - p[:-q]
    m = q * (n - q + 1) * (1 - q / n)
    varq = ((rq - q * mu) ** 2).sum() / m
    vr = varq / var1
    dev2 = (r - mu) ** 2
    theta = 0.0
    for j in range(1, q):
        delta = (dev2[j:] * dev2[:-j]).sum() / (dev2.sum() ** 2)
        theta += (2 * (q - j) / q) ** 2 * delta
    theta *= n
    z = (vr - 1) / np.sqrt(theta / n) if theta > 0 else 0.0
    return float(vr), float(z)


def kalman_hedge(y: pd.Series, x: pd.Series, delta: float = 1e-4, obs_var: float = 1e-3) -> pd.DataFrame:
    """Dynamic hedge ratio (random-walk state [beta, alpha]).
    Columns: beta, alpha, e (forecast error made *before* updating), q (its variance), z."""
    df = pd.concat([y, x], axis=1).dropna()
    yy, xx = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    n = len(df)
    theta = np.zeros(2)
    P = np.eye(2)
    Vw = delta / (1 - delta) * np.eye(2)
    out = np.zeros((n, 4))
    for t in range(n):
        F = np.array([xx[t], 1.0])
        R = P + Vw
        yhat = F @ theta
        Q = F @ R @ F + obs_var
        e = yy[t] - yhat
        K = R @ F / Q
        out[t] = (theta[0], theta[1], e, Q)   # beta/alpha known *before* seeing y[t]
        theta = theta + K * e
        P = R - np.outer(K, F) @ R
    res = pd.DataFrame(out, index=df.index, columns=["beta", "alpha", "e", "q"])
    res["z"] = res["e"] / np.sqrt(res["q"])
    return res
