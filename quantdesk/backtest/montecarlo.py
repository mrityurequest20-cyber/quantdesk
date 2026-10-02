"""Monte Carlo robustness: how bad could it have been with the same edge and a different
sequence of luck? Trade-order bootstrap and stationary block bootstrap of daily returns."""
from __future__ import annotations

import numpy as np
import pandas as pd


def trade_bootstrap(pnls, start_equity: float, n_sims: int = 5000, seed: int = 0) -> dict:
    """Resample closed-trade P&Ls with replacement (same count), rebuild equity paths."""
    pnl = np.asarray(pnls, dtype=float)
    if len(pnl) < 5:
        return {}
    rng = np.random.default_rng(seed)
    draws = rng.choice(pnl, size=(n_sims, len(pnl)), replace=True)
    paths = start_equity + np.cumsum(draws, axis=1)
    peaks = np.maximum.accumulate(np.concatenate([np.full((n_sims, 1), start_equity), paths], axis=1), axis=1)[:, 1:]
    dd = (paths / peaks - 1).min(axis=1)
    final = paths[:, -1]
    return {
        "final_p5": np.percentile(final, 5), "final_p50": np.percentile(final, 50), "final_p95": np.percentile(final, 95),
        "maxdd_p50": np.percentile(dd, 50), "maxdd_p95": np.percentile(dd, 5), "maxdd_worst": dd.min(),
        "p_loss": float((final < start_equity).mean()), "p_dd_gt_20": float((dd < -0.20).mean()),
        "dd_dist": dd, "final_dist": final,
    }


def block_bootstrap(returns: pd.Series, n_sims: int = 2000, block: int = 20, seed: int = 0) -> dict:
    """Stationary-ish block bootstrap (fixed blocks) that preserves volatility clustering."""
    r = pd.Series(returns).dropna().to_numpy()
    n = len(r)
    if n < block * 3:
        return {}
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block, size=(n_sims, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_sims, -1)[:, :n]
    sims = r[idx]
    eq = np.cumprod(1 + sims, axis=1)
    years = n / 252
    cagr = eq[:, -1] ** (1 / years) - 1
    dd = (eq / np.maximum.accumulate(eq, axis=1) - 1).min(axis=1)
    sharpe = sims.mean(axis=1) / sims.std(axis=1) * np.sqrt(252)
    return {"cagr_p5": np.percentile(cagr, 5), "cagr_p50": np.percentile(cagr, 50), "cagr_p95": np.percentile(cagr, 95),
            "sharpe_p5": np.percentile(sharpe, 5), "sharpe_p50": np.percentile(sharpe, 50),
            "maxdd_p50": np.percentile(dd, 50), "maxdd_p95": np.percentile(dd, 5),
            "p_negative_cagr": float((cagr < 0).mean()), "cagr_dist": cagr, "dd_dist": dd}
