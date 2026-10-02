"""Walk-forward optimisation: choose parameters on a rolling in-sample window, trade them
on the following out-of-sample window, stitch the OOS pieces. The OOS curve is the only
honest estimate; the gap between IS and OOS Sharpe measures overfitting. The Deflated
Sharpe Ratio then discounts the OOS Sharpe for the number of variants tried."""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..risk import metrics as M
from ..strategies import REGISTRY
from .runner import run_backtest


@dataclass
class WFResult:
    folds: pd.DataFrame
    oos_equity: pd.Series
    oos_stats: dict
    dsr: float
    n_trials: int


def param_grid(grid: dict[str, list]) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]


def walk_forward(cfg, data, strategy: str, grid: dict[str, list], train_years: float = 3, test_years: float = 1,
                 start=None, end=None, aux=None, metric: str = "sharpe", progress=None) -> WFResult:
    from ..engine.engine import Engine
    bench = data[cfg.get("universe.benchmark", "NIFTY")].index
    start = pd.Timestamp(start) if start else bench[cfg.get("backtest.warmup_bars", 260)]
    end = pd.Timestamp(end) if end else bench[-1]
    aux = aux or Engine.build_aux(cfg, data)
    base = dict(cfg.get(f"strategies.{strategy}", {}))
    combos = param_grid(grid)
    folds, pieces, all_trial_sharpes = [], [], []
    t0 = start
    while True:
        tr_end = t0 + pd.DateOffset(years=train_years)
        te_end = min(tr_end + pd.DateOffset(years=test_years), end)
        if tr_end >= end:
            break
        best, best_score, scores = None, -np.inf, []
        for p in combos:
            params = {**base, **p}
            strat = REGISTRY[strategy](strategy, params, cfg)
            res = run_backtest(cfg, data, [strat], start=t0, end=tr_end, aux=aux)
            sc = res.stats.get(metric, np.nan) if res.stats else np.nan
            sc = -np.inf if sc != sc else sc
            scores.append(sc)
            if sc > best_score:
                best, best_score = p, sc
        all_trial_sharpes += [s for s in scores if np.isfinite(s)]
        strat = REGISTRY[strategy](strategy, {**base, **best}, cfg)
        oos = run_backtest(cfg, data, [strat], start=tr_end, end=te_end, aux=aux)
        r = oos.equity["equity"].pct_change().dropna()
        pieces.append(r)
        folds.append({"train": f"{t0.date()}→{tr_end.date()}", "test": f"{tr_end.date()}→{te_end.date()}",
                      "best_params": best, "is_" + metric: best_score,
                      "oos_" + metric: oos.stats.get(metric, np.nan) if oos.stats else np.nan,
                      "oos_return": oos.stats.get("total_return", np.nan) if oos.stats else np.nan,
                      "oos_trades": oos.trade_stats.get("trades", 0)})
        if progress:
            progress(folds[-1])
        if te_end >= end:
            break
        t0 = t0 + pd.DateOffset(years=test_years)
    rets = pd.concat(pieces) if pieces else pd.Series(dtype=float)
    rets = rets[~rets.index.duplicated()]
    eq = (1 + rets).cumprod() * float(cfg.get("account.starting_capital", 1e6))
    stats = M.returns_stats(eq) if len(eq) > 2 else {}
    sr_std = float(np.std(all_trial_sharpes) / np.sqrt(252)) if len(all_trial_sharpes) > 1 else 0.0
    dsr = M.deflated_sharpe(rets, max(1, len(combos)), sr_std) if len(rets) > 10 else float("nan")
    return WFResult(pd.DataFrame(folds), eq, stats, dsr, len(combos))
