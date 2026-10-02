"""Short-horizon mean reversion: buy statistically stretched pullbacks inside a long-term
uptrend, only where the rolling Hurst exponent says the stock is currently anti-persistent."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..analytics import indicators as ind
from ..analytics.stats import rolling_hurst
from ..core.types import ExitDecision
from .base import Strategy


class MeanReversion(Strategy):
    family = "mean_reversion"
    description = "z-score / RSI(3) pullback in an uptrend, Hurst filter, time stop"
    required = ("z", "rsi", "trend", "hurst", "atr")

    def features(self, df, symbol, aux):
        p = self.p
        c = df["close"]
        out = pd.DataFrame({"close": c}, index=df.index)
        out["z"] = ind.zscore(c, p["z_window"])
        out["rsi"] = ind.rsi(c, p["rsi_len"])
        out["trend"] = ind.sma(c, p["trend_filter"])
        out["hurst"] = rolling_hurst(np.log(c), 100)
        out["atr"] = ind.atr(df, 14)
        return out

    def entries(self, ctx):
        out = []
        p = self.p
        for sym in self.tables:
            r = self.row(sym, ctx)
            if r is None or ctx.trades_for(self.name, sym):
                continue
            stretched = r["z"] < p["z_entry"] or r["rsi"] < p["rsi_entry"]
            if stretched and r["close"] > r["trend"] and r["hurst"] < p["hurst_max"]:
                stop = r["close"] - p["atr_stop"] * r["atr"]
                trig = []
                if r["z"] < p["z_entry"]:
                    trig.append(f"z-score {r['z']:.2f} < {p['z_entry']}")
                if r["rsi"] < p["rsi_entry"]:
                    trig.append(f"RSI{p['rsi_len']} {r['rsi']:.0f} < {p['rsi_entry']}")
                char = ("anti-persistent (mean-reverting)" if r["hurst"] < 0.45 else
                        "close to a random walk" if r["hurst"] < 0.5 else "mildly persistent, under the cap")
                why = (f"Stretched pullback in an uptrend: {' and '.join(trig)}; price still above "
                       f"SMA{p['trend_filter']}. Rolling Hurst {r['hurst']:.2f} ({char}, cap {p['hurst_max']}). "
                       f"Exit at z > {p['z_exit']} or after {p['time_stop']} bars; disaster stop {stop:.2f}.")
                conf = 0.4 + 0.3 * min(abs(r["z"]) / 3, 1) + 0.3 * max(0.0, (0.5 - r["hurst"]) * 4)
                ctxd = {k: round(float(r[k]), 4) for k in ("close", "z", "rsi", "trend", "hurst", "atr")}
                it = self.linear_intent(ctx, sym, +1, stop, why, ctxd, confidence=conf, atr=r["atr"],
                                        exit_rules={"time_stop": p["time_stop"]})
                if it:
                    out.append(it)
        return out

    def manage(self, trade, ctx):
        r = self.row(trade.symbol, ctx)
        if r is None:
            return None
        if r["z"] > self.p["z_exit"]:
            return ExitDecision("mean_reverted", f"z-score back to {r['z']:.2f}")
        if trade.bars_held >= self.p["time_stop"]:
            return ExitDecision("time_stop", f"no reversion within {self.p['time_stop']} bars")
        return None
