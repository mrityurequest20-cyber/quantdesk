"""Synthetic 1-minute sessions for offline development and tests — NOT market data.

Each session is drawn from a day type (trend up/down, range, reversal, volatile) with the
intraday stylised facts: opening gap, U-shaped volatility and volume, trend drift front-
loaded into the morning, mean reversion on balance days, fat tails, BANKNIFTY at beta
~1.25 to NIFTY, and an India VIX that rises when the index falls."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .feeds import IST, OPEN

MINUTES = 375
DAY_TYPES = {"trend_up": 0.18, "trend_down": 0.15, "range": 0.40, "reversal": 0.15, "volatile": 0.12}


def _u_shape() -> np.ndarray:
    t = np.arange(MINUTES)
    w = 1 + 2.2 * np.exp(-t / 18) + 0.9 * np.exp(-(MINUTES - 1 - t) / 25)
    return w / w.sum()


def simulate_sessions(days: list[dt.date], seed: int = 11, nifty0: float = 25000.0, bank0: float = 55000.0,
                      vix0: float = 13.5, force_types: dict | None = None) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    w = _u_shape()
    types, probs = list(DAY_TYPES), np.array(list(DAY_TYPES.values()))
    frames = {"NIFTY": [], "BANKNIFTY": [], "INDIAVIX": []}
    meta = []
    n_close, b_close, vix = nifty0, bank0, vix0
    for day in days:
        dtype = (force_types or {}).get(day) or types[rng.choice(len(types), p=probs)]
        vix = float(np.clip(14 + 0.85 * (vix - 14) + rng.normal(0, 0.7), 9, 40))
        sig_d = vix / 100 / np.sqrt(252) * 0.85 * np.exp(rng.normal(0, 0.2)) * (1.6 if dtype == "volatile" else 1.0)
        sig_t = sig_d * np.sqrt(w)
        eps = rng.standard_t(5, MINUTES) / np.sqrt(5 / 3)
        gap = rng.normal(0, 0.3 * sig_d) + {"trend_up": 0.15, "trend_down": -0.15}.get(dtype, 0.0) * sig_d
        r = np.zeros(MINUTES)
        x = 0.0
        if dtype in ("trend_up", "trend_down"):
            sgn = 1 if dtype == "trend_up" else -1
            drift = sgn * rng.uniform(0.9, 1.7) * sig_d * (w + 0.2 / MINUTES) / (w + 0.2 / MINUTES).sum()
            r = drift + sig_t * eps * 0.8
        elif dtype == "reversal":
            sgn = rng.choice([-1, 1])
            k = int(MINUTES * rng.uniform(0.3, 0.5))
            leg1 = sgn * rng.uniform(0.7, 1.1) * sig_d
            drift = np.where(np.arange(MINUTES) < k, leg1 / k, -1.4 * leg1 / (MINUTES - k))
            r = drift + sig_t * eps * 0.8
        else:
            kappa = 0.035 if dtype == "range" else 0.01
            for t in range(MINUTES):
                r[t] = -kappa * x + sig_t[t] * eps[t]
                x += r[t]
        o0 = n_close * np.exp(gap)
        close = o0 * np.exp(np.cumsum(r))
        opens = np.concatenate([[o0], close[:-1]])
        ext = np.abs(rng.normal(0, 1, (2, MINUTES))) * sig_t * 0.6
        high = np.maximum(opens, close) * np.exp(ext[0])
        low = np.minimum(opens, close) * np.exp(-ext[1])
        vol = 1.2e6 * w * MINUTES * np.exp(rng.normal(0, 0.3, MINUTES)) * (1 + 0.25 * np.abs(r) / sig_t)
        idx = pd.date_range(pd.Timestamp(dt.datetime.combine(day, OPEN), tz=IST), periods=MINUTES, freq="min")
        frames["NIFTY"].append(pd.DataFrame({"open": opens, "high": high, "low": low, "close": close, "volume": vol}, index=idx))

        rb = 1.25 * r + sig_t * rng.standard_normal(MINUTES) * 0.45
        bo = b_close * np.exp(1.2 * gap + rng.normal(0, 0.1 * sig_d))
        bclose = bo * np.exp(np.cumsum(rb))
        bopen = np.concatenate([[bo], bclose[:-1]])
        bext = np.abs(rng.normal(0, 1, (2, MINUTES))) * sig_t * 0.75
        frames["BANKNIFTY"].append(pd.DataFrame({
            "open": bopen, "high": np.maximum(bopen, bclose) * np.exp(bext[0]), "low": np.minimum(bopen, bclose) * np.exp(-bext[1]),
            "close": bclose, "volume": vol * 0.6}, index=idx))

        vpath = vix * np.exp(np.cumsum(-4.0 * r + rng.normal(0, 0.0015, MINUTES)))
        vopen = np.concatenate([[vix], vpath[:-1]])
        frames["INDIAVIX"].append(pd.DataFrame({"open": vopen, "high": np.maximum(vopen, vpath) * 1.001,
                                                "low": np.minimum(vopen, vpath) * 0.999, "close": vpath, "volume": 0.0}, index=idx))
        n_close, b_close, vix = float(close[-1]), float(bclose[-1]), float(vpath[-1])
        meta.append({"day": day, "type": dtype, "sigma_d": sig_d, "return": close[-1] / (o0 / np.exp(gap)) - 1})
    return {k: pd.concat(v) for k, v in frames.items()}, pd.DataFrame(meta).set_index("day")
