"""Statistical arbitrage on cointegrated pairs (Engle-Granger, re-tested monthly on a
rolling formation window). Legs trade as single-stock futures so the short side is
allowed overnight in India (basis and lot rounding are ignored in the backtest)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..analytics.stats import engle_granger
from ..core.context import FeatureTable
from ..core.types import FUTURE, LINEAR, ExitDecision, Instrument, LegSpec, TradeIntent
from .base import Strategy

UNIT_NOTIONAL = 100_000.0   # INR of leg A per structure unit


class Pairs(Strategy):
    family = "pairs"
    description = "Engle-Granger cointegrated pairs, z-score entry/exit, monthly re-test"

    def universe(self):
        return [f"{a}/{b}" for a, b in self.cfg.get("universe.pairs", [])]

    def prepare(self, data, aux):
        self.tables = {}
        for key in self.universe():
            a, b = key.split("/")
            if a in data and b in data:
                self.tables[key] = FeatureTable(self._pair_features(data[a]["close"], data[b]["close"]))

    def _pair_features(self, pa: pd.Series, pb: pd.Series) -> pd.DataFrame:
        p = self.p
        df = pd.concat([pa, pb], axis=1, keys=["pa", "pb"]).dropna()
        la, lb = np.log(df["pa"]), np.log(df["pb"])
        n = len(df)
        cols = {k: np.full(n, np.nan) for k in ("beta", "alpha", "mu", "sd", "pvalue", "hl", "z")}
        form, step = p["formation"], p["retest_every"]
        for k in range(form, n, step):
            res = engle_granger(la.iloc[k - form:k + 1], lb.iloc[k - form:k + 1])
            mu, sd = float(res.spread.mean()), float(res.spread.std())
            end = min(n, k + step)
            s = la.iloc[k:end].to_numpy() - res.hedge_ratio * lb.iloc[k:end].to_numpy() - res.intercept
            cols["beta"][k:end] = res.hedge_ratio
            cols["alpha"][k:end] = res.intercept
            cols["mu"][k:end] = mu
            cols["sd"][k:end] = sd
            cols["pvalue"][k:end] = res.pvalue
            cols["hl"][k:end] = res.half_life
            cols["z"][k:end] = (s - mu) / sd
        out = pd.DataFrame(cols, index=df.index)
        out["pa"], out["pb"] = df["pa"], df["pb"]
        out["active"] = (out["pvalue"] < p["max_pvalue"]) & (out["hl"] < p["max_half_life"])
        return out

    @staticmethod
    def leg(sym: str) -> Instrument:
        return Instrument(symbol=f"{sym}-FUT", kind=FUTURE, underlying=sym, lot_size=1)

    def entries(self, ctx):
        out = []
        p = self.p
        for key, t in self.tables.items():
            r = t.row(ctx.ts)
            if r is None or not r["active"] or np.isnan(r["z"]) or ctx.trades_for(self.name, key):
                continue
            z = r["z"]
            if abs(z) < p["z_entry"]:
                continue
            a, b = key.split("/")
            side = -1 if z > 0 else +1                  # +1 = long the spread (long A, short B)
            n_a = max(1, int(round(UNIT_NOTIONAL / r["pa"])))
            n_b = max(1, int(round(r["beta"] * UNIT_NOTIONAL / r["pb"])))
            legs = [LegSpec(self.leg(a), side * n_a), LegSpec(self.leg(b), -side * n_b)]
            risk = (p["z_stop"] - abs(z)) * r["sd"] * UNIT_NOTIONAL
            cost = sum(ctx.costs.round_trip_estimate(l.instrument, abs(l.ratio), px)
                       for l, px in zip(legs, (r["pa"], r["pb"])))
            why = (f"{a}/{b} cointegrated (EG p={r['pvalue']:.3f}, half-life {r['hl']:.0f}d, hedge {r['beta']:.2f}). "
                   f"Spread z = {z:+.2f} → {'long' if side > 0 else 'short'} {a}, {'short' if side > 0 else 'long'} {b}. "
                   f"Exit |z| < {p['z_exit']}, stop |z| > {p['z_stop']}, time stop {p['time_stop']} bars.")
            meta = {k: float(r[k]) for k in ("beta", "alpha", "mu", "sd", "hl")}
            meta["entry_z"] = float(z)
            out.append(TradeIntent(
                strategy=self.name, family=self.family, symbol=key, direction=side, kind=LINEAR, legs=legs,
                entry_ref=float(z), risk_per_unit=max(risk, 0.01 * UNIT_NOTIONAL) + cost,
                confidence=float(np.clip(0.4 + 0.2 * (abs(z) - p["z_entry"]) + 0.2 * (0.05 - r["pvalue"]) / 0.05, 0.1, 1)),
                rationale=why, context={"z": round(float(z), 3), "pvalue": round(float(r["pvalue"]), 4),
                                        "half_life": round(float(r["hl"]), 1), "regime": ctx.regime()},
                exit_rules={"time_stop": p["time_stop"]}, meta=meta))
        return out

    def current_z(self, trade, ctx) -> float | None:
        a, b = trade.symbol.split("/")
        if not (ctx.has(a) and ctx.has(b)):
            return None
        m = trade.meta
        s = np.log(ctx.price(a)) - m["beta"] * np.log(ctx.price(b)) - m["alpha"]
        return float((s - m["mu"]) / m["sd"])

    def manage(self, trade, ctx):
        z = self.current_z(trade, ctx)
        if z is None:
            return None
        trade.meta["last_z"] = z
        p = self.p
        if abs(z) < p["z_exit"] or np.sign(z) == trade.direction:
            return ExitDecision("converged", f"spread z back to {z:+.2f}")
        if abs(z) > p["z_stop"]:
            return ExitDecision("z_stop", f"spread diverged to z {z:+.2f}")
        if trade.bars_held >= p["time_stop"]:
            return ExitDecision("time_stop", f"no convergence in {p['time_stop']} bars")
        r = self.tables[trade.symbol].row(ctx.ts)
        if r is not None and r["pvalue"] > 0.25:
            return ExitDecision("cointegration_broke", f"re-test p-value {r['pvalue']:.2f}")
        return None
