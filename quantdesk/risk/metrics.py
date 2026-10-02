"""Performance and risk statistics, including the Probabilistic and Deflated Sharpe
Ratios (Bailey & López de Prado) that correct a backtest Sharpe for short samples,
fat tails and the number of strategy variants tried."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

ANN = 252
EULER = 0.5772156649


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def max_dd_duration(equity: pd.Series) -> int:
    under = equity < equity.cummax()
    longest = cur = 0
    for u in under:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    return longest


def probabilistic_sharpe(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """P(true per-period Sharpe > sr_benchmark) given sample length, skew and kurtosis."""
    r = pd.Series(returns).dropna()
    n = len(r)
    if n < 10 or r.std() == 0:
        return float("nan")
    sr = r.mean() / r.std()
    g3, g4 = r.skew(), r.kurt() + 3
    denom = np.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def deflated_sharpe(returns: pd.Series, n_trials: int, sr_std_across_trials: float) -> float:
    """PSR against the Sharpe you'd expect from the best of `n_trials` zero-skill variants."""
    if n_trials <= 1:
        return probabilistic_sharpe(returns, 0.0)
    sr0 = sr_std_across_trials * ((1 - EULER) * norm.ppf(1 - 1 / n_trials) + EULER * norm.ppf(1 - 1 / (n_trials * np.e)))
    return probabilistic_sharpe(returns, sr0)


def cornish_fisher_var(r: pd.Series, alpha: float = 0.05) -> float:
    z = norm.ppf(alpha)
    s, k = r.skew(), r.kurt()
    zcf = z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24 - (2 * z ** 3 - 5 * z) * s ** 2 / 36
    return float(-(r.mean() + zcf * r.std()))


def returns_stats(equity: pd.Series, rf: float = 0.0) -> dict:
    eq = equity.dropna()
    r = eq.pct_change().dropna()
    if len(r) < 2:
        return {}
    years = len(r) / ANN
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / max(years, 1e-9)) - 1
    vol = r.std() * np.sqrt(ANN)
    ex = r - rf / ANN
    sharpe = ex.mean() / r.std() * np.sqrt(ANN) if r.std() > 0 else np.nan
    downside = r[r < 0].std() * np.sqrt(ANN)
    sortino = ex.mean() * ANN / downside if downside > 0 else np.nan
    dd = drawdown_series(eq)
    mdd = dd.min()
    var95 = -np.percentile(r, 5)
    cvar95 = -r[r <= -var95].mean() if (r <= -var95).any() else np.nan
    tail = abs(np.percentile(r, 95) / np.percentile(r, 5)) if np.percentile(r, 5) != 0 else np.nan
    return {
        "start": str(eq.index[0].date()), "end": str(eq.index[-1].date()), "years": round(years, 2),
        "total_return": eq.iloc[-1] / eq.iloc[0] - 1, "cagr": cagr, "ann_vol": vol, "sharpe": sharpe,
        "sortino": sortino, "max_drawdown": mdd, "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "max_dd_days": max_dd_duration(eq), "var95_1d": var95, "cvar95_1d": cvar95,
        "cf_var95_1d": cornish_fisher_var(r), "skew": r.skew(), "kurtosis": r.kurt(), "tail_ratio": tail,
        "best_day": r.max(), "worst_day": r.min(), "pct_up_days": (r > 0).mean(),
        "psr_vs_0": probabilistic_sharpe(ex),
    }


def trade_stats(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {"trades": 0}
    pnl = trades["pnl"]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    streak = cur = 0
    for p in pnl:
        cur = cur + 1 if p <= 0 else 0
        streak = max(streak, cur)
    return {
        "trades": len(trades), "win_rate": (pnl > 0).mean(), "avg_win": wins.mean() if len(wins) else 0.0,
        "avg_loss": losses.mean() if len(losses) else 0.0,
        "profit_factor": wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf,
        "expectancy": pnl.mean(), "avg_r": trades["r_multiple"].mean(), "median_r": trades["r_multiple"].median(),
        "total_pnl": pnl.sum(), "total_fees": trades["fees"].sum(), "avg_bars": trades["bars_held"].mean(),
        "max_consec_losses": streak,
    }


def by_strategy(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    rows = {s: trade_stats(g) for s, g in trades.groupby("strategy")}
    return pd.DataFrame(rows).T


def monthly_returns(equity: pd.Series) -> pd.DataFrame:
    m = equity.resample("ME").last().pct_change()
    m.iloc[0] = equity.resample("ME").last().iloc[0] / equity.iloc[0] - 1
    t = pd.DataFrame({"year": m.index.year, "month": m.index.month, "ret": m.values})
    tab = t.pivot(index="year", columns="month", values="ret")
    tab["year_total"] = (1 + tab.fillna(0)).prod(axis=1) - 1
    return tab
