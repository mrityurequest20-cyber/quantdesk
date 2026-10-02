"""Synthetic Indian-market simulator for offline development and demos.

This is NOT evidence of edge. It exists so the whole pipeline (analysis → strategies →
risk → execution → journal → reports) can run end-to-end without network access, on
data that carries the stylised facts real markets have:

* regime switching (calm bull / normal / stressed bear) via a 3-state Markov chain;
* volatility clustering and the leverage effect (GJR-GARCH shocks, fat-tailed t(5));
* crash jumps (Poisson, more frequent in the stressed regime);
* a VIX that tracks forecast volatility plus a variance risk premium (IV > RV on average)
  and spikes on down days;
* a factor model for stocks (market beta + sector + idiosyncratic), slow-moving
  idiosyncratic drift (so momentum exists), mild short-term reversal, and
  cointegrated pairs (shared stochastic trend + OU spread).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.calendar import TradingCalendar
from .base import DataProvider

TRADING_DAYS = 252

START_PRICES = {
    "NIFTY": 10400.0, "BANKNIFTY": 25300.0, "RELIANCE": 910.0, "HDFCBANK": 940.0, "ICICIBANK": 310.0,
    "INFY": 520.0, "TCS": 1380.0, "ITC": 262.0, "LT": 1310.0, "SBIN": 305.0, "BHARTIARTL": 420.0,
    "AXISBANK": 560.0, "KOTAKBANK": 1010.0, "HINDUNILVR": 1330.0,
}
SECTORS = {
    "HDFCBANK": "bank", "ICICIBANK": "bank", "AXISBANK": "bank", "KOTAKBANK": "bank", "SBIN": "bank",
    "INFY": "it", "TCS": "it", "RELIANCE": "energy", "ITC": "fmcg", "HINDUNILVR": "fmcg",
    "LT": "infra", "BHARTIARTL": "telecom",
}
BETAS = {"bank": 1.15, "it": 0.85, "energy": 1.05, "fmcg": 0.65, "infra": 1.1, "telecom": 0.8}


def _student_t(rng, df, size):
    return rng.standard_t(df, size) / np.sqrt(df / (df - 2))


def simulate_market(dates: pd.DatetimeIndex, seed: int = 7, symbols: list[str] | None = None,
                    pairs: list[list[str]] | None = None) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = len(dates)

    # 1) regime chain ----------------------------------------------------------------
    P = np.array([[0.990, 0.008, 0.002],
                  [0.012, 0.978, 0.010],
                  [0.012, 0.040, 0.948]])
    mu = np.array([0.24, 0.09, -0.30]) / TRADING_DAYS
    vol = np.array([0.105, 0.155, 0.300]) / np.sqrt(TRADING_DAYS)
    jump_rate = np.array([0.3, 1.0, 4.0]) / TRADING_DAYS
    states = np.zeros(n, dtype=int)
    states[0] = 1
    u = rng.random(n)
    cum = P.cumsum(axis=1)
    for t in range(1, n):
        states[t] = int(np.searchsorted(cum[states[t - 1]], u[t]))

    # 2) GJR-GARCH unit-variance multiplier h_t (mean ≈ 1) -------------------------------
    a, g, b = 0.05, 0.08, 0.89
    w = 1.0 - a - g / 2 - b
    z = _student_t(rng, 6, n)
    h = np.ones(n)
    for t in range(1, n):
        h[t] = w + (a + g * (z[t - 1] < 0)) * z[t - 1] ** 2 * h[t - 1] + b * h[t - 1]
        h[t] = min(max(h[t], 0.4), 5.0)   # floor: index RV rarely drops below ~7%
    sig = vol[states] * np.sqrt(h)
    jumps = (rng.random(n) < jump_rate[states]) * rng.normal(-0.025, 0.015, n)
    r_mkt = mu[states] + sig * z + jumps

    # 3) VIX = sqrt(expected 30-day variance) * (1 + VRP) + down-move spikes -------------
    pers = a + g / 2 + b
    horizon = 21
    avg_h = 1.0 + (h - 1.0) * (1 - pers ** horizon) / (horizon * (1 - pers))
    exp_vol = np.sqrt(vol[states] ** 2 * avg_h * TRADING_DAYS)
    vrp = np.empty(n)
    vrp[0] = 0.15
    eps = rng.normal(0, 0.03, n)
    for t in range(1, n):
        vrp[t] = 0.15 + 0.9 * (vrp[t - 1] - 0.15) + eps[t]
    spike = np.clip(-r_mkt, 0, None) * 4.0
    vix = 100 * exp_vol * (1 + np.clip(vrp, -0.05, 0.6)) * (1 + spike)
    vix = pd.Series(vix).ewm(span=2).mean().to_numpy()

    out: dict[str, pd.DataFrame] = {}
    out["NIFTY"] = _ohlcv(rng, dates, r_mkt, START_PRICES["NIFTY"], sig, base_volume=2.5e8)
    out["INDIAVIX"] = _vix_frame(rng, dates, vix)

    # 4) BANKNIFTY and stocks ------------------------------------------------------------
    sector_f = {s: rng.normal(0, 0.07 / np.sqrt(TRADING_DAYS), n) for s in sorted(set(SECTORS.values()))}
    r_bank = 1.15 * r_mkt + sector_f["bank"] * 1.2 + _student_t(rng, 5, n) * 0.05 / np.sqrt(TRADING_DAYS)
    out["BANKNIFTY"] = _ohlcv(rng, dates, r_bank, START_PRICES["BANKNIFTY"], sig * 1.2, base_volume=1.5e8)

    symbols = symbols or list(SECTORS)
    pairs = [tuple(p) for p in (pairs or [])]
    paired = {s for p in pairs for s in p}
    idio: dict[str, np.ndarray] = {}
    for s in symbols:
        if s in out or s in paired or s not in SECTORS:
            continue
        idio[s] = _idio_returns(rng, n)
    for a_sym, b_sym in pairs:
        # Shared stochastic trend + an OU spread (half-life ~12 days): log A - log B is stationary.
        common = _idio_returns(rng, n, vol_ann=0.14)
        kappa = np.log(2) / 12
        s_sd = 0.045 * np.sqrt(2 * kappa)
        spread = np.zeros(n)
        shocks = rng.normal(0, s_sd, n)
        for t in range(1, n):
            spread[t] = spread[t - 1] * (1 - kappa) + shocks[t]
        d_spread = np.diff(spread, prepend=0.0)
        idio[a_sym] = common + d_spread / 2
        idio[b_sym] = common - d_spread / 2
    for s, e in idio.items():
        sector = SECTORS.get(s, "infra")
        beta = BETAS[sector]
        r = beta * r_mkt + sector_f[sector] + e
        sd = np.sqrt((beta * sig) ** 2 + (0.2 / np.sqrt(TRADING_DAYS)) ** 2)
        out[s] = _ohlcv(rng, dates, r, START_PRICES.get(s, 500.0), sd, base_volume=6e6)
    for df in out.values():
        df.index.name = "date"
    out["_regimes"] = pd.DataFrame({"state": states, "h": h}, index=dates)
    return out


def _idio_returns(rng, n: int, vol_ann: float = 0.17) -> np.ndarray:
    """Idiosyncratic returns with slow drift (momentum) and mild 1-day reversal."""
    drift = np.zeros(n)
    kappa = np.log(2) / 150
    d_sd = 0.12 / TRADING_DAYS * np.sqrt(2 * kappa)
    ds = rng.normal(0, d_sd, n)
    for t in range(1, n):
        drift[t] = drift[t - 1] * (1 - kappa) + ds[t]
    eps = _student_t(rng, 4, n) * vol_ann / np.sqrt(TRADING_DAYS)
    r = np.empty(n)
    r[0] = eps[0]
    for t in range(1, n):
        r[t] = drift[t] + eps[t] - 0.05 * eps[t - 1]
    return r


def _ohlcv(rng, dates, r: np.ndarray, p0: float, sig: np.ndarray, base_volume: float) -> pd.DataFrame:
    close = p0 * np.exp(np.cumsum(r))
    prev = np.concatenate([[p0], close[:-1]])
    share = np.clip(rng.normal(0.30, 0.15, len(r)), 0, 0.8)
    open_ = prev * np.exp(r * share + rng.normal(0, 0.15, len(r)) * sig)
    hi_ext = np.abs(rng.normal(0, 1, len(r))) * sig * 0.55
    lo_ext = np.abs(rng.normal(0, 1, len(r))) * sig * 0.55
    high = np.maximum(open_, close) * np.exp(hi_ext)
    low = np.minimum(open_, close) * np.exp(-lo_ext)
    volume = base_volume * np.exp(0.6 * np.abs(r) / np.maximum(sig, 1e-9) * 0.5 + rng.normal(0, 0.25, len(r)))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=dates)


def _vix_frame(rng, dates, vix: np.ndarray) -> pd.DataFrame:
    close = vix
    open_ = np.concatenate([[vix[0]], vix[:-1]]) * np.exp(rng.normal(0, 0.02, len(vix)))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.03, len(vix))))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.03, len(vix))))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 0.0}, index=dates)


class SyntheticProvider(DataProvider):
    """Deterministic (seeded) synthetic universe. The market is always simulated over the
    same fixed horizon and then cut at `end`, so a given calendar day has the same bars no
    matter when you run it (paper-trading replays stay consistent from day to day)."""
    name = "synthetic"
    HORIZON_START = "2015-01-01"
    HORIZON_END = "2030-12-31"

    def __init__(self, cfg, start=None, end=None, seed: int | None = None):
        self.cfg = cfg
        cal = TradingCalendar(cfg.holidays())
        self.end = pd.Timestamp(end or pd.Timestamp.today().normalize())
        self.dates = cal.trading_days(start or self.HORIZON_START, self.HORIZON_END)
        self.market = simulate_market(self.dates, seed=seed if seed is not None else cfg.get("data.synthetic_seed", 1),
                                      symbols=cfg.all_symbols(), pairs=cfg.get("universe.pairs", []))

    def history(self, symbol: str, start, end=None) -> pd.DataFrame:
        if symbol not in self.market:
            raise KeyError(f"synthetic market has no {symbol}")
        stop = min(pd.Timestamp(end), self.end) if end is not None else self.end
        return self.market[symbol].loc[pd.Timestamp(start):stop].copy()

    def regimes(self) -> pd.DataFrame:
        return self.market["_regimes"].loc[: self.end]
