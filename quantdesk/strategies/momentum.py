"""Cross-sectional 12-1 momentum, volatility-adjusted, with an absolute-momentum filter
(dual momentum). Rebalances monthly with a rank buffer to cut turnover."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..analytics import indicators as ind
from ..core.types import ExitDecision
from .base import Strategy


class Momentum(Strategy):
    family = "momentum"
    description = "12-1 month risk-adjusted momentum, top-N, monthly rebalance"
    required = ("mom", "score", "sma200", "atr")

    def prepare(self, data, aux):
        super().prepare(data, aux)
        self._rank_cache: dict = {}

    def features(self, df, symbol, aux):
        p = self.p
        c = df["close"]
        out = pd.DataFrame({"close": c}, index=df.index)
        out["mom"] = c.shift(p["skip"]) / c.shift(p["lookback"]) - 1
        vol = np.log(c).diff().rolling(126, min_periods=100).std() * np.sqrt(252)
        out["score"] = out["mom"] / vol
        out["sma200"] = ind.sma(c, 200)
        out["atr"] = ind.atr(df, 14)
        return out

    def ranks(self, ctx) -> dict[str, int]:
        if ctx.ts in self._rank_cache:
            return self._rank_cache[ctx.ts]
        scores = {}
        for sym in self.tables:
            r = self.row(sym, ctx)
            if r is not None:
                scores[sym] = r["score"]
        order = sorted(scores, key=lambda s: -scores[s])
        ranks = {s: i + 1 for i, s in enumerate(order)}
        self._rank_cache = {ctx.ts: ranks}
        return ranks

    def _rebalance_day(self, ctx) -> bool:
        return ctx.i % self.p["rebalance_every"] == 0

    def entries(self, ctx):
        if not self._rebalance_day(ctx):
            return []
        ranks = self.ranks(ctx)
        out = []
        for sym, rk in ranks.items():
            if rk > self.p["top_n"] or ctx.trades_for(self.name, sym):
                continue
            r = self.row(sym, ctx)
            if r["mom"] <= self.p["min_momentum"] or r["close"] < r["sma200"]:
                continue
            stop = r["close"] - self.p["atr_stop"] * r["atr"]
            why = (f"Rank {rk}/{len(ranks)} on 12-1 month momentum ({r['mom']:+.1%}, risk-adjusted {r['score']:.2f}); "
                   f"absolute momentum positive and price above SMA200. Held until it drops out of the top "
                   f"{2 * self.p['top_n']} at a monthly rebalance; catastrophe stop {stop:.2f}.")
            ctxd = {"rank": rk, "mom": round(float(r["mom"]), 4), "score": round(float(r["score"]), 3)}
            it = self.linear_intent(ctx, sym, +1, stop, why, ctxd, confidence=0.5 + 0.1 * (self.p["top_n"] - rk + 1),
                                    atr=r["atr"])
            if it:
                out.append(it)
        return out

    def manage(self, trade, ctx):
        r = self.row(trade.symbol, ctx)
        if r is None:
            return None
        if r["close"] < r["sma200"]:
            return ExitDecision("abs_momentum_off", "price fell below SMA200")
        if self._rebalance_day(ctx):
            rk = self.ranks(ctx).get(trade.symbol, 999)
            if rk > 2 * self.p["top_n"] or r["mom"] < 0:
                return ExitDecision("rebalance_out", f"momentum rank fell to {rk}")
        return None
