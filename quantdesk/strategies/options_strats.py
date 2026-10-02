"""Index options strategies.

* VRPCondor   — sell defined-risk iron condors when implied vol is rich versus a GARCH
                forecast of realised vol (the variance risk premium), IV rank is elevated,
                the regime is not stressed and no scheduled event is imminent.
* TrendSpread — express a confirmed index trend with a debit vertical when IV is cheap.
* LongVol     — buy a straddle when implied vol is cheap versus the GARCH forecast or the
                market is coiled (Bollinger squeeze), expecting a volatility expansion.

All positions are defined-risk; max loss is known at entry and drives sizing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..analytics import indicators as ind
from ..analytics.volatility import rolling_garch_forecast
from ..core.calendar import year_fraction
from ..core.types import OPTIONS, ExitDecision
from ..options.structures import StructureBuilder
from .base import Strategy
from .trend import bars_since


class _IndexOptions(Strategy):
    kind = OPTIONS

    def universe(self):
        return self.cfg.symbols(self.p.get("symbols", "indices"))

    def base_features(self, df, symbol, aux) -> pd.DataFrame:
        out = pd.DataFrame({"close": df["close"]}, index=df.index)
        iv = aux["iv"].get(symbol)
        if iv is not None:
            out = out.join(iv[["iv", "iv_rank", "vix"]], how="left")
        else:
            out["iv"] = out["iv_rank"] = out["vix"] = np.nan
        horizon = max(3, int(round(self.p.get("target_dte", 14) * 5 / 7)))
        out["garch"] = rolling_garch_forecast(df["close"], horizon=horizon)
        out["atr"] = ind.atr(df, 14)
        return out

    def builder(self, sym, ctx) -> StructureBuilder:
        spec = self.cfg.instrument_spec(sym)
        return StructureBuilder(ctx.pricer, sym, int(spec["lot_size"]), float(spec["strike_step"]))

    def expiry(self, sym, ctx, target, min_dte=0):
        spec = self.cfg.instrument_spec(sym)
        return ctx.calendar.pick_expiry(ctx.date, target, min_dte, int(spec.get("expiry_weekday", 1)),
                                        bool(spec.get("weekly_expiry", True)))

    def pnl_state(self, trade, ctx) -> tuple[float, int]:
        """(gross P&L in INR at model marks, calendar days to nearest expiry)."""
        marks = self.structure_marks(trade, ctx)
        gross = trade.value(marks) - trade.entry_cost
        dte = (trade.nearest_expiry() - ctx.date).days
        return gross, dte


class VRPCondor(_IndexOptions):
    family = "short_vol"
    description = "Iron condor when IV > GARCH forecast (variance risk premium)"
    required = ("iv", "iv_rank", "garch")

    def features(self, df, symbol, aux):
        out = self.base_features(df, symbol, aux)
        out["vrp"] = out["iv"] * 100 - out["garch"]
        return out

    def entries(self, ctx):
        p = self.p
        out = []
        for sym in self.tables:
            r = self.row(sym, ctx)
            if r is None or ctx.trades_for(self.name, sym):
                continue
            regime = ctx.regime(sym)
            ev = ctx.event_within(int(self.cfg.get("calendar.event_blackout_days", 1)))
            if (r["iv_rank"] < p["min_iv_rank"] or r["vrp"] < p["min_vrp"] or regime == "stressed"
                    or r["vix"] > p["max_vix"] or ev):
                continue
            exp = self.expiry(sym, ctx, p["target_dte"], p["min_dte"])
            st = self.builder(sym, ctx).iron_condor(r["close"], ctx.date, exp, r["iv"], p["short_delta"], p["wing_delta"])
            edge = st.expected_pnl(r["garch"] / 100)
            if edge <= 0:
                continue
            credit = -st.net_premium()
            why = (f"IV {r['iv'] * 100:.1f} vs GARCH-forecast RV {r['garch']:.1f} → variance risk premium "
                   f"{r['vrp']:+.1f} pts; IV rank {r['iv_rank']:.0%}; regime {regime}. Sold {st.describe()}. "
                   f"Model edge at forecast vol ₹{edge:,.0f}/unit, POP {st.prob_profit(r['garch'] / 100):.0%}. "
                   f"Take profit at {p['take_profit']:.0%} of credit, stop at {p['stop_loss']}x credit, exit at "
                   f"{p['exit_dte']} DTE.")
            conf = 0.4 + 0.3 * min(r["vrp"] / 6, 1) + 0.3 * r["iv_rank"]
            ctxd = {"iv": round(r["iv"] * 100, 2), "garch": round(float(r["garch"]), 2), "vrp": round(float(r["vrp"]), 2),
                    "iv_rank": round(float(r["iv_rank"]), 3), "credit": round(credit, 2), "edge": round(edge, 2)}
            it = self.options_intent(ctx, st, 0, why, ctxd, {"take_profit": p["take_profit"], "stop_loss": p["stop_loss"],
                                                             "exit_dte": p["exit_dte"]}, conf)
            if it:
                it.meta["credit"] = credit
                it.meta["planned_risk_per_unit"] = min(st.max_loss(), p["stop_loss"] * credit)
                out.append(it)
        return out

    def manage(self, trade, ctx):
        p = self.p
        gross, dte = self.pnl_state(trade, ctx)
        credit = trade.meta.get("credit", -trade.entry_cost)
        if gross >= p["take_profit"] * credit:
            return ExitDecision("take_profit", f"captured {gross / credit:.0%} of the credit")
        if gross <= -p["stop_loss"] * credit:
            return ExitDecision("stop_loss", f"loss reached {-gross / credit:.1f}x credit")
        if dte <= p["exit_dte"]:
            return ExitDecision("time_exit", f"{dte} DTE: gamma risk outweighs remaining theta")
        if ctx.regime(trade.symbol) == "stressed":
            return ExitDecision("regime_stress", "market regime turned stressed")
        S = ctx.price(trade.symbol)
        shorts = [l.instrument for l in trade.legs if l.qty < 0]
        if any((i.right == "PE" and S < i.strike) or (i.right == "CE" and S > i.strike) for i in shorts):
            return ExitDecision("short_strike_breached", f"spot {S:,.0f} through a short strike")
        return None


class TrendSpread(_IndexOptions):
    family = "trend"
    description = "Debit vertical in the direction of a confirmed index trend, when IV is cheap"
    required = ("fast", "slow", "sma200", "adx", "iv", "iv_rank")

    def features(self, df, symbol, aux):
        out = self.base_features(df, symbol, aux)
        c = df["close"]
        out["fast"], out["slow"] = ind.ema(c, self.p["fast"]), ind.ema(c, self.p["slow"])
        out["sma200"] = ind.sma(c, 200)
        out["adx"] = ind.adx(df, 14)["adx"]
        above = out["fast"] > out["slow"]
        out["since_up"] = bars_since(above & ~above.shift(1, fill_value=True))
        out["since_dn"] = bars_since(~above & above.shift(1, fill_value=False))
        return out

    def entries(self, ctx):
        p = self.p
        out = []
        for sym in self.tables:
            r = self.row(sym, ctx)
            if r is None or ctx.trades_for(self.name, sym) or r["iv_rank"] > p["max_iv_rank"] or r["adx"] < p["adx_min"]:
                continue
            up = r["fast"] > r["slow"] and r["close"] > r["sma200"] and r["since_up"] <= 15
            dn = r["fast"] < r["slow"] and r["close"] < r["sma200"] and r["since_dn"] <= 15
            if not (up or dn):
                continue
            exp = self.expiry(sym, ctx, p["target_dte"], 10)
            b = self.builder(sym, ctx)
            st = (b.bull_call_spread if up else b.bear_put_spread)(r["close"], ctx.date, exp, r["iv"],
                                                                   p["long_delta"], p["short_delta"])
            why = (f"{sym} {'up' if up else 'down'}trend: EMA{p['fast']} {'>' if up else '<'} EMA{p['slow']}, "
                   f"price {'above' if up else 'below'} SMA200, ADX {r['adx']:.0f}. IV rank {r['iv_rank']:.0%} is low "
                   f"so buying premium is cheap: {st.describe()}. TP {p['take_profit']:.0%} of max profit, "
                   f"SL {p['stop_loss']:.0%} of debit, exit at {p['exit_dte']} DTE or on a trend reversal.")
            ctxd = {"adx": round(float(r["adx"]), 1), "iv": round(r["iv"] * 100, 2), "iv_rank": round(float(r["iv_rank"]), 3)}
            it = self.options_intent(ctx, st, +1 if up else -1, why, ctxd,
                                     {"take_profit": p["take_profit"], "stop_loss": p["stop_loss"], "exit_dte": p["exit_dte"]},
                                     0.4 + 0.4 * min(r["adx"] / 40, 1))
            if it:
                it.meta["planned_risk_per_unit"] = p["stop_loss"] * st.net_premium()
                out.append(it)
        return out

    def manage(self, trade, ctx):
        p = self.p
        gross, dte = self.pnl_state(trade, ctx)
        debit = trade.entry_cost
        maxp = trade.meta.get("max_profit", debit)
        if gross >= p["take_profit"] * maxp:
            return ExitDecision("take_profit", f"{gross / maxp:.0%} of max profit")
        if gross <= -p["stop_loss"] * debit:
            return ExitDecision("stop_loss", f"lost {-gross / debit:.0%} of the debit")
        if dte <= p["exit_dte"]:
            return ExitDecision("time_exit", f"{dte} DTE")
        r = self.row(trade.symbol, ctx)
        if r is not None and ((trade.direction > 0 and r["fast"] < r["slow"]) or (trade.direction < 0 and r["fast"] > r["slow"])):
            return ExitDecision("trend_reversal", "EMA cross against the position")
        return None


class LongVol(_IndexOptions):
    family = "long_vol"
    description = "Long straddle when IV is cheap vs GARCH forecast or the index is coiled"
    required = ("iv", "iv_rank", "garch", "bw_pct")

    def features(self, df, symbol, aux):
        out = self.base_features(df, symbol, aux)
        out["edge"] = out["garch"] - out["iv"] * 100
        out["bw_pct"] = ind.pct_rank(ind.bollinger(df["close"], 20)["bandwidth"], 250)
        return out

    def entries(self, ctx):
        p = self.p
        out = []
        for sym in self.tables:
            r = self.row(sym, ctx)
            if r is None or ctx.trades_for(self.name, sym) or r["iv_rank"] > p["max_iv_rank"]:
                continue
            cheap = r["edge"] > p["min_edge"]
            coiled = r["bw_pct"] < p["squeeze_pct"]
            if not (cheap or coiled):
                continue
            exp = self.expiry(sym, ctx, p["target_dte"], 10)
            st = self.builder(sym, ctx).long_straddle(r["close"], ctx.date, exp, r["iv"])
            edge_inr = st.expected_pnl(max(r["garch"], 1) / 100)
            reasons = []
            if cheap:
                reasons.append(f"GARCH forecast {r['garch']:.1f} exceeds IV {r['iv'] * 100:.1f} by {r['edge']:.1f} pts")
            if coiled:
                reasons.append(f"Bollinger width at the {r['bw_pct']:.0%} percentile (coiled)")
            why = (f"Long volatility on {sym}: " + "; ".join(reasons) + f"; IV rank {r['iv_rank']:.0%}. {st.describe()}. "
                   f"Model EV at forecast vol ₹{edge_inr:,.0f}/unit. TP +{p['take_profit']:.0%}, SL -{p['stop_loss']:.0%} "
                   f"of debit, exit at {p['exit_dte']} DTE.")
            ctxd = {"iv": round(r["iv"] * 100, 2), "garch": round(float(r["garch"]), 2), "bw_pct": round(float(r["bw_pct"]), 3)}
            it = self.options_intent(ctx, st, 0, why, ctxd,
                                     {"take_profit": p["take_profit"], "stop_loss": p["stop_loss"], "exit_dte": p["exit_dte"]},
                                     0.35 + 0.15 * cheap + 0.15 * coiled)
            if it:
                it.meta["planned_risk_per_unit"] = p["stop_loss"] * st.net_premium()
                out.append(it)
        return out

    def manage(self, trade, ctx):
        p = self.p
        gross, dte = self.pnl_state(trade, ctx)
        debit = trade.entry_cost
        if gross >= p["take_profit"] * debit:
            return ExitDecision("take_profit", f"+{gross / debit:.0%} on the debit")
        if gross <= -p["stop_loss"] * debit:
            return ExitDecision("stop_loss", f"-{-gross / debit:.0%} on the debit")
        if dte <= p["exit_dte"]:
            return ExitDecision("time_exit", f"{dte} DTE: theta decay accelerating")
        return None
