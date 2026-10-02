"""The quant decision layer: forecast the move, test for a directional edge, and price each
candidate trade's expected value before any money goes in.

1. VolForecaster: σ per minute, blending today's EWMA realised vol (1m returns), the prior
   sessions' realised vol and the ATM implied vol (IV forecasts realised vol well, with a
   premium). Horizon σ scales with √minutes.

2. DirectionModel: L2 logistic regression predicting whether NIFTY/BANKNIFTY is higher in 30
   minutes, from 5-minute features (momentum at 3 horizons in σ units, distance from VWAP,
   position in the day's range and the opening range, EMA9−EMA21, RSI, time of day, the
   overnight gap). Trained only on sessions before today. It must earn its place with a
   walk-forward test (train on the older 70% of days, score the newest 30%): out-of-sample AUC
   ≥ 0.53 *and* a log-loss better than the base rate. If not, it's switched off and the desk
   says so; it never trades on an unvalidated model.

3. EVEngine: Monte Carlo of the actual option structure over its holding period (normal returns
   with the forecast σ, and the model's drift for its 30-minute horizon only; the fat tails the
   market prices are in each leg's own IV), marked every 3 minutes with
   minute-level theta and each leg's own IV, exiting the way the engine would (invalidation
   level, premium stop, premium target, underlying target, time stop), crossing the bid/ask
   again on exit and paying the full Indian cost stack both ways. Output: EV per lot (₹),
   P(profit), CVaR 5%, expected R. A plan trades only if EV clears a floor: costs, theta and
   spreads must be paid for by the edge, not hoped away.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..core.types import Instrument
from .chains import IntradayPricer, time_to_expiry

MIN_PER_YEAR = 375 * 252
FEATURES = ["r5", "r15", "r30", "vwap_z", "close_loc", "or_pos", "ema_gap", "rsi", "tod", "gap"]
HORIZON_BARS = 6                                   # 6 × 5m = 30 minutes


# ---- volatility ---------------------------------------------------------------------------------------------
def _within_day_returns(close: pd.Series) -> pd.Series:
    r = np.log(close).diff()
    new_day = pd.Series(close.index.date, index=close.index).ne(pd.Series(close.index.date, index=close.index).shift())
    return r[~new_day.to_numpy()].dropna()


class VolForecaster:
    def __init__(self, halflife_min: float = 30, iv_weight: float = 0.3):
        self.lam = 0.5 ** (1 / halflife_min)
        self.iv_weight = iv_weight
        self._prior_day, self._var_prior = None, np.nan

    def forecast(self, bars: pd.DataFrame, day, atm_iv: float | None = None, today_close: np.ndarray | None = None) -> dict:
        """σ per minute (log-return units) from bars up to now. `atm_iv` in % (e.g. 13.5). Pass `today_close`
        (today's 1m closes) on every minute of a live session: the prior sessions' variance is then computed once."""
        if today_close is None or self._prior_day != day:
            c = bars["close"].astype(float)
            prior = c[c.index.date < day]
            r_prior = _within_day_returns(prior.tail(375 * 5)) if len(prior) > 30 else pd.Series(dtype=float)
            self._var_prior = float((r_prior ** 2).mean()) if len(r_prior) > 30 else np.nan
            self._prior_day = day if today_close is not None else None
            if today_close is None:
                today_close = c[c.index.date == day].to_numpy()
        var_prior = self._var_prior
        r_today = np.diff(np.log(today_close)) if len(today_close) > 2 else np.array([])
        if len(r_today) >= 10:
            w = self.lam ** np.arange(len(r_today))[::-1]
            var_today = float(np.sum(w * r_today ** 2) / w.sum())
        else:
            var_today = np.nan
        n = len(r_today)
        k = n / (n + 60)                                           # today's estimate earns weight as the day goes on
        parts = [(k, var_today), (1 - k, var_prior)]
        parts = [(w_, v) for w_, v in parts if v == v]
        var_rv = sum(w_ * v for w_, v in parts) / sum(w_ for w_, _ in parts) if parts else np.nan
        var_iv = (atm_iv / 100) ** 2 / MIN_PER_YEAR if atm_iv and atm_iv > 0 else np.nan
        if var_rv == var_rv and var_iv == var_iv:
            var, src = (1 - self.iv_weight) * var_rv + self.iv_weight * var_iv, "realised+implied"
        elif var_rv == var_rv:
            var, src = var_rv, "realised"
        elif var_iv == var_iv:
            var, src = var_iv, "implied"
        else:
            var, src = (0.15 ** 2) / MIN_PER_YEAR, "default 15%"
        s = math.sqrt(var)
        return {"sigma_min": s, "sigma_30m": s * math.sqrt(30), "rv_ann": math.sqrt(var_rv * MIN_PER_YEAR) * 100 if var_rv == var_rv else None,
                "vol_ann": s * math.sqrt(MIN_PER_YEAR) * 100, "source": src}


# ---- features -----------------------------------------------------------------------------------------------
def to_5m(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return bars
    return bars.resample("5min", label="left", closed="left", origin="start_day", offset="15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["close"])


def _ewm(x: np.ndarray, span: float | None = None, alpha: float | None = None) -> np.ndarray:
    a = alpha if alpha is not None else 2 / (span + 1)
    out = np.empty_like(x)
    acc = x[0]
    for i, v in enumerate(x):
        acc = v if i == 0 else a * v + (1 - a) * acc
        out[i] = acc
    return out


SIG_DEFAULT = 8e-4                                  # a typical 5m σ for NIFTY/BANKNIFTY, used until there's data


def _day_features(o, h, l, c, v, minutes, prev_close: float, prev_sig: float = float("nan")) -> np.ndarray:
    """Feature matrix (n × len(FEATURES)) for one session's 5m bars, numpy only. Row i uses bars ≤ i and the
    previous session, nothing later (tests recompute on truncated data and require identical rows)."""
    n = len(c)
    lr = np.r_[np.nan, np.diff(np.log(c))]
    seed = prev_sig if prev_sig == prev_sig and prev_sig > 0 else SIG_DEFAULT
    rs = pd.Series(lr)
    sig = rs.rolling(12, min_periods=4).std().fillna(rs.expanding(min_periods=3).std()).fillna(seed).clip(lower=2e-4).to_numpy()

    def lag_ret(k):
        out = np.zeros(n)
        if n > k:
            out[k:] = np.log(c[k:] / c[:-k])
        return out
    r5 = np.nan_to_num(lr) / sig
    r15 = lag_ret(3) / (sig * math.sqrt(3))
    r30 = lag_ret(6) / (sig * math.sqrt(6))
    w = v if v.sum() > 0 else np.ones(n)
    tp = (h + l + c) / 3
    vwap = np.cumsum(tp * w) / np.cumsum(w)
    vwap_z = np.clip((c / vwap - 1) / sig, -8, 8)
    hi, lo = np.maximum.accumulate(h), np.minimum.accumulate(l)
    rng = hi - lo
    close_loc = np.where(rng > 0, (c - lo) / np.where(rng > 0, rng, 1), 0.5) - 0.5
    orh, orl = h[:3].max(), l[:3].min()
    or_pos = np.clip((c - (orh + orl) / 2) / max(orh - orl, 1e-9), -4, 4)
    or_pos[:3] = 0.0                                                # the opening range isn't known until 09:30
    ema_gap = np.clip((_ewm(c, span=9) - _ewm(c, span=21)) / c / sig, -8, 8)
    d = np.r_[0.0, np.diff(c)]
    up, dn = _ewm(np.clip(d, 0, None), alpha=1 / 14), _ewm(np.clip(-d, 0, None), alpha=1 / 14)
    rsi = np.where(dn > 0, 100 - 100 / (1 + up / np.where(dn > 0, dn, 1)), 50.0)
    rsi = (rsi - 50) / 50
    tod = minutes / 375
    gap = math.log(o[0] / prev_close) if prev_close == prev_close and prev_close else 0.0
    gapf = np.full(n, np.clip(gap / (seed * math.sqrt(75)), -6, 6))              # scaled by the *previous* session's σ
    return np.column_stack([r5, r15, r30, vwap_z, close_loc, or_pos, ema_gap, rsi, tod, gapf])


def session_sigma(c: np.ndarray) -> float:
    """A session's typical 5m σ (median of the rolling σ), carried into the next session's features."""
    lr = pd.Series(np.r_[np.nan, np.diff(np.log(c))])
    s = lr.rolling(12, min_periods=4).std().median()
    return float(s) if s == s else float("nan")


def features_5m(df5: pd.DataFrame, prev_close: float = float("nan"), prev_sig: float = float("nan")) -> pd.DataFrame:
    """Causal features per completed 5m bar (each row uses bars up to and including itself), plus the
    30-minute-ahead target `y` (NaN where the session ends first). `prev_close`/`prev_sig` describe the
    session before the first one in `df5`."""
    out = []
    if df5 is None or df5.empty:
        return pd.DataFrame(columns=FEATURES + ["y", "day"])
    dates = df5.index.date
    for day in sorted(set(dates)):
        d = df5[dates == day]
        o, h, l, c, v = (d[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close", "volume"))
        minutes = (d.index - pd.Timestamp(day, tz=d.index.tz) - pd.Timedelta(hours=9, minutes=15)).total_seconds().to_numpy() / 60
        X = _day_features(o, h, l, c, v, minutes, prev_close, prev_sig)
        f = pd.DataFrame(X, index=d.index, columns=FEATURES)
        fwd = np.full(len(c), np.nan)
        if len(c) > HORIZON_BARS:
            fwd[:-HORIZON_BARS] = (np.log(c[HORIZON_BARS:] / c[:-HORIZON_BARS]) > 0).astype(float)
        f["y"] = fwd
        f["day"] = str(day)
        out.append(f)
        prev_close, prev_sig = float(c[-1]), session_sigma(c)
    return pd.concat(out)


# ---- direction model ------------------------------------------------------------------------------------------
def _auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(p).rank().to_numpy()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class DirectionModel:
    def __init__(self, l2: float = 3.0, min_auc: float = 0.53, min_test: int = 150):
        self.l2, self.min_auc, self.min_test = l2, min_auc, min_test
        self.w = None
        self.mu = self.sd = None
        self.valid = False
        self.diag: dict = {"status": "untrained"}

    def _fit(self, X: np.ndarray, y: np.ndarray):
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
        w = np.zeros(Z.shape[1])
        reg = np.full(Z.shape[1], self.l2)
        reg[0] = 0.0
        for _ in range(30):                                        # IRLS / Newton on the penalised log-likelihood
            p = 1 / (1 + np.exp(-Z @ w))
            g = Z.T @ (p - y) + reg * w
            H = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(reg + 1e-9)
            step = np.linalg.solve(H, g)
            w -= step
            if np.abs(step).max() < 1e-7:
                break
        return w, mu, sd

    def _predict(self, w, mu, sd, X):
        Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
        return 1 / (1 + np.exp(-Z @ w))

    def fit(self, feats: pd.DataFrame) -> dict:
        d = feats.dropna(subset=["y"])
        days = sorted(d["day"].unique())
        if len(days) < 8 or len(d) < 300:
            self.valid, self.diag = False, {"status": "not enough history", "samples": int(len(d)), "days": len(days)}
            return self.diag
        cut = days[int(len(days) * 0.7)]
        tr, te = d[d["day"] < cut], d[d["day"] >= cut]
        X_tr, y_tr, X_te, y_te = tr[FEATURES].to_numpy(float), tr["y"].to_numpy(float), te[FEATURES].to_numpy(float), te["y"].to_numpy(float)
        w, mu, sd = self._fit(X_tr, y_tr)
        p_te = self._predict(w, mu, sd, X_te)
        auc = _auc(y_te, p_te)
        base = np.clip(y_tr.mean(), 0.05, 0.95)
        skill = 1 - _logloss(y_te, p_te) / _logloss(y_te, np.full(len(y_te), base))
        self.valid = bool(len(te) >= self.min_test and auc == auc and auc >= self.min_auc and skill > 0)
        self.w, self.mu, self.sd = self._fit(d[FEATURES].to_numpy(float), d["y"].to_numpy(float))
        coefs = dict(zip(["bias"] + FEATURES, np.round(self.w, 3).tolist()))
        self.diag = {"status": "validated" if self.valid else "no edge out of sample", "samples": int(len(d)), "days": len(days),
                     "test_samples": int(len(te)), "auc_oos": round(auc, 3), "logloss_skill_oos": round(skill, 4),
                     "base_rate": round(float(d["y"].mean()), 3), "coefs": coefs}
        return self.diag

    def predict(self, row: pd.Series | None) -> float:
        if self.w is None or row is None:
            return 0.5
        x = row[FEATURES].to_numpy(float)[None, :]
        return float(self._predict(self.w, self.mu, self.sd, x)[0])


# ---- expected value of a plan -----------------------------------------------------------------------------------
class EVEngine:
    def __init__(self, cfg, pricer: IntradayPricer, costs, n_paths: int = 2000, step_min: float = 3.0, seed: int = 7,
                 drift_horizon_min: float = 30.0):
        self.cfg, self.pricer, self.costs = cfg, pricer, costs
        self.n, self.step_min, self.seed = n_paths, step_min, seed
        self.drift_h = drift_horizon_min

    def evaluate(self, plan, S: float, now: pd.Timestamp, sigma_min: float, p_up: float, minutes_left: float,
                 base_drift_min: float = 0.0) -> dict:
        """`base_drift_min`: an unconditional drift per minute (log units) that applies for the whole hold, e.g. the
        research-validated intraday drift of the index; the model's conditional drift comes from `p_up`."""
        h = float(max(3.0, min(plan.time_stop_min, minutes_left)))
        k = int(math.ceil(h / self.step_min))
        dtm = h / k
        # drift from the probability of being up over the model's horizon: P(r > 0) = Φ(μ_h / σ_h); applied only
        # for that horizon (a 30-minute forecast says nothing about the hour after). Normal increments: the tails
        # the market charges for are already in each leg's own IV (the chain's skew).
        p = float(np.clip(p_up, 0.2, 0.8))
        mu_min = sigma_min * math.sqrt(self.drift_h) * norm.ppf(p) / self.drift_h
        t_mid = (np.arange(k) + 0.5) * dtm
        mu_step = np.where(t_mid <= self.drift_h, mu_min, 0.0) + base_drift_min
        z = np.random.default_rng(self.seed).standard_normal((self.n, k))    # same draws for every candidate: fair comparison
        steps = (mu_step - 0.5 * sigma_min ** 2) * dtm + sigma_min * math.sqrt(dtm) * z
        paths = S * np.exp(np.cumsum(steps, axis=1))                # n × k
        lot = plan.lot_size
        T0 = time_to_expiry(now, plan.expiry)
        entry = plan.net_premium                                    # per lot, + debit / − credit
        value = np.zeros_like(paths)
        exit_cost = 0.0
        fees = 0.0
        for leg in plan.legs:
            iv = leg.iv / 100 if leg.iv > 1 else leg.iv
            for j in range(k):
                # business time: IV is quoted per calendar year, but the variance arrives in trading minutes, so each
                # trading minute uses up 1/MIN_PER_YEAR of it (≈5.6 calendar minutes). Without this, repricing with
                # calendar theta hands every intraday option buyer a free edge the market doesn't give.
                T = max(T0 - (j + 1) * dtm / MIN_PER_YEAR, 0.0)
                value[:, j] += leg.ratio * lot * self.pricer_vec(leg.strike, leg.right, paths[:, j], T, iv)
            half = max(abs(leg.price - leg.mid), 0.05)              # cross the spread again on the way out
            exit_cost += half * abs(leg.ratio) * lot
            inst = Instrument.option(plan.symbol, plan.expiry, leg.strike, leg.right, lot)
            fees += self.costs.fees(inst, leg.ratio * lot, leg.price)[0] + self.costs.fees(inst, -leg.ratio * lot, max(leg.mid, 0.05))[0]
        gross = value - exit_cost - entry                          # P&L per lot if exited at each step
        prem = abs(entry)
        hit = np.zeros(paths.shape, dtype=bool)
        d = plan.direction
        if plan.invalidation is not None and d != 0:
            hit |= (paths <= plan.invalidation) if d > 0 else (paths >= plan.invalidation)
        if plan.target_underlying is not None and d != 0:
            hit |= (paths >= plan.target_underlying) if d > 0 else (paths <= plan.target_underlying)
        hit |= gross <= -prem * plan.premium_stop
        hit |= gross >= prem * plan.premium_target
        hit[:, -1] = True
        first = hit.argmax(axis=1)
        pnl = gross[np.arange(self.n), first] - fees
        worst = np.sort(pnl)[: max(1, self.n // 20)]
        risk = max(plan.planned_risk_per_lot(), 1.0)
        return {"ev": float(pnl.mean()), "ev_r": float(pnl.mean() / risk), "p_profit": float((pnl > 0).mean()),
                "cvar5": float(worst.mean()), "median": float(np.median(pnl)), "fees": float(fees), "exit_cost": float(exit_cost),
                "horizon_min": round(h), "p_up": p, "sigma_h_pct": sigma_min * math.sqrt(h) * 100,
                "avg_hold_min": float((first + 1).mean() * dtm)}

    def pricer_vec(self, K, right, S, T, iv):
        from ..options.pricing import bs_price
        if T <= 0:
            return np.maximum(S - K, 0.0) if right == "CE" else np.maximum(K - S, 0.0)
        return bs_price(S, K, T, self.pricer.r, self.pricer.q, iv, right)


def load_research(path) -> dict:
    """Research-validated priors per symbol from edges.json (the Edge research workflow's output):
    only results with verdict EDGE are used. Today: the intraday drift (test D1)."""
    import json
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return {}
    try:
        rows = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    out: dict = {}
    for r in rows:
        if r.get("verdict") == "EDGE" and r.get("id") == "D1" and r.get("effect_bps") is not None:
            out.setdefault(r["symbol"], {})["drift"] = {"bps_day": float(r["effect_bps"]), "t": r.get("t"), "n": r.get("n"),
                                                         "holdout_bps": r.get("effect_holdout_bps"),
                                                         "per_min": float(r["effect_bps"]) / 1e4 / 375}
    return out
