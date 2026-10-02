"""Trend-following family: EMA-cross trend rider and a volatility-squeeze Donchian breakout."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..analytics import indicators as ind
from ..core.types import ExitDecision
from .base import Strategy


def bars_since(flag: pd.Series) -> pd.Series:
    idx = np.arange(len(flag))
    last = np.where(flag.fillna(False).to_numpy(dtype=bool), idx, -1)
    last = np.maximum.accumulate(last)
    out = np.where(last < 0, 9999, idx - last)
    return pd.Series(out, index=flag.index)


class TrendRider(Strategy):
    """Enter fresh EMA(20/50) crosses in the direction of the 200-day trend when ADX
    confirms a trend; ATR stop, Chandelier trailing stop, exit on the opposite cross."""
    family = "trend"
    description = "EMA cross + ADX trend filter, ATR stop, chandelier trail"
    required = ("fast", "slow", "trend", "adx", "atr")

    def features(self, df, symbol, aux):
        p = self.p
        c = df["close"]
        out = pd.DataFrame({"close": c}, index=df.index)
        out["fast"], out["slow"] = ind.ema(c, p["fast"]), ind.ema(c, p["slow"])
        out["trend"] = ind.sma(c, p["trend_filter"])
        a = ind.adx(df, 14)
        out["adx"], out["pdi"], out["mdi"] = a["adx"], a["plus_di"], a["minus_di"]
        out["atr"] = ind.atr(df, 14)
        ch = ind.chandelier(df, 22, p["chandelier_atr"])
        out["chand_long"], out["chand_short"] = ch["long_stop"], ch["short_stop"]
        above = out["fast"] > out["slow"]
        out["since_up"] = bars_since(above & ~above.shift(1, fill_value=True))
        out["since_dn"] = bars_since(~above & above.shift(1, fill_value=False))
        out["er"] = ind.efficiency_ratio(c, 20)
        return out

    def entries(self, ctx):
        out = []
        for sym in self.tables:
            r = self.row(sym, ctx)
            if r is None or ctx.trades_for(self.name, sym):
                continue
            conf = 0.35 + 0.35 * min(r["adx"] / 40, 1) + 0.3 * (r["er"] if r["er"] == r["er"] else 0)
            ctxd = {k: round(float(r[k]), 4) for k in ("close", "fast", "slow", "trend", "adx", "atr", "er")}
            if (r["fast"] > r["slow"] and r["since_up"] <= 10 and r["close"] > r["trend"]
                    and r["adx"] >= self.p["adx_min"] and r["pdi"] > r["mdi"]):
                stop = r["close"] - self.p["atr_stop"] * r["atr"]
                why = (f"EMA{self.p['fast']} crossed above EMA{self.p['slow']} {int(r['since_up'])} bars ago; price above "
                       f"SMA{self.p['trend_filter']} ({r['trend']:.1f}); ADX {r['adx']:.0f} with +DI>-DI confirms a trend. "
                       f"Stop {self.p['atr_stop']}xATR = {stop:.2f}; trail with a {self.p['chandelier_atr']}xATR chandelier.")
                it = self.linear_intent(ctx, sym, +1, stop, why, ctxd, confidence=conf, atr=r["atr"])
                if it:
                    out.append(it)
            elif (not self.p.get("long_only", True) and r["fast"] < r["slow"] and r["since_dn"] <= 10
                  and r["close"] < r["trend"] and r["adx"] >= self.p["adx_min"] and r["mdi"] > r["pdi"]):
                stop = r["close"] + self.p["atr_stop"] * r["atr"]
                why = f"Bearish EMA cross in a downtrend (ADX {r['adx']:.0f}); stop {stop:.2f}."
                it = self.linear_intent(ctx, sym, -1, stop, why, ctxd, confidence=conf, atr=r["atr"])
                if it:
                    out.append(it)
        return out

    def manage(self, trade, ctx):
        r = self.row(trade.symbol, ctx)
        if r is None:
            return None
        if trade.direction > 0:
            if r["chand_long"] == r["chand_long"] and r["chand_long"] > (trade.stop or -np.inf):
                trade.stop = float(r["chand_long"])
            if r["fast"] < r["slow"]:
                return ExitDecision("trend_reversal", f"EMA{self.p['fast']} fell below EMA{self.p['slow']}")
        else:
            if r["chand_short"] == r["chand_short"] and r["chand_short"] < (trade.stop or np.inf):
                trade.stop = float(r["chand_short"])
            if r["fast"] > r["slow"]:
                return ExitDecision("trend_reversal", "bullish cross against the short")
        return None


class Breakout(Strategy):
    """Donchian channel breakout that only fires out of a volatility squeeze and on
    expanding volume (a coiled range resolving); exit on the shorter opposite channel."""
    family = "trend"
    description = "55-day Donchian breakout from a Bollinger squeeze, 20-day channel exit"
    required = ("up", "exit_lo", "atr", "bw_min", "sma200")

    def features(self, df, symbol, aux):
        p = self.p
        out = pd.DataFrame({"close": df["close"]}, index=df.index)
        dc = ind.donchian(df, p["entry_lookback"])
        out["up"], out["dn"] = dc["upper"], dc["lower"]
        ex = ind.donchian(df, p["exit_lookback"])
        out["exit_lo"], out["exit_hi"] = ex["lower"], ex["upper"]
        out["atr"] = ind.atr(df, 14)
        bw = ind.bollinger(df["close"], 20)["bandwidth"]
        out["bw_pct"] = ind.pct_rank(bw, 250)
        out["bw_min"] = out["bw_pct"].rolling(20, min_periods=5).min()
        out["vol_ratio"] = df["volume"] / df["volume"].rolling(20, min_periods=20).mean()
        out["sma200"] = ind.sma(df["close"], 200)
        return out

    def entries(self, ctx):
        out = []
        for sym in self.tables:
            r = self.row(sym, ctx)
            if r is None or ctx.trades_for(self.name, sym):
                continue
            vol_ok = not (r["vol_ratio"] == r["vol_ratio"]) or r["vol_ratio"] > 1.1
            if r["close"] > r["up"] and r["bw_min"] < self.p["squeeze_pct"] and vol_ok and r["close"] > r["sma200"]:
                stop = r["close"] - self.p["atr_stop"] * r["atr"]
                why = (f"Close {r['close']:.2f} broke the {self.p['entry_lookback']}-day high {r['up']:.2f} after a squeeze "
                       f"(Bollinger width fell to the {r['bw_min']:.0%} percentile) on {r['vol_ratio']:.1f}x average volume. "
                       f"Initial stop {stop:.2f}; exit on a close below the {self.p['exit_lookback']}-day low.")
                conf = 0.45 + 0.3 * (1 - r["bw_min"]) + 0.25 * min(max((r["vol_ratio"] or 1) - 1, 0), 1)
                ctxd = {k: round(float(r[k]), 4) for k in ("close", "up", "atr", "bw_min", "vol_ratio")}
                it = self.linear_intent(ctx, sym, +1, stop, why, ctxd, confidence=conf, atr=r["atr"])
                if it:
                    out.append(it)
        return out

    def manage(self, trade, ctx):
        r = self.row(trade.symbol, ctx)
        if r is None:
            return None
        if r["exit_lo"] > (trade.stop or -np.inf):
            trade.stop = float(r["exit_lo"])
        if r["close"] < r["exit_lo"]:
            return ExitDecision("channel_exit", f"close below the {self.p['exit_lookback']}-day low")
        return None
