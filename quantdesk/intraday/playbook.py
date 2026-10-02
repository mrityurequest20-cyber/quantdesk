"""Intraday options playbook: setups that turn an analyst view into a concrete, costed plan.

Every plan carries its own invalidation (an underlying level that proves the idea wrong),
targets, premium stop, and time stop — decided *before* entry and journaled with it.

  orb           opening-range breakout in the direction of the evidence
  vwap_trend    trend-day pullback to VWAP that resumes
  va_reversion  failed auction at the value-area edge on a balance day → back to POC
  range_sell    balance day + rich premium → defined-risk iron fly, theta harvest

Structure choice follows the vol view: premium cheap/fair → buy the option outright;
premium rich → debit vertical (sells some of the rich vol back); balance + rich → iron fly.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .analyst import MarketView
from .chains import IntradayPricer, mid, time_to_expiry


@dataclass
class PlanLeg:
    strike: float
    right: str
    ratio: int                     # +1 long, -1 short (per lot)
    price: float                   # fill estimate: ask for buys, bid for sells
    mid: float
    iv: float
    delta: float


@dataclass
class TradePlan:
    setup: str
    symbol: str
    direction: int
    structure: str
    expiry: dt.date
    legs: list[PlanLeg]
    lot_size: int
    trigger: str
    thesis: str
    invalidation: float | None
    target_underlying: float | None
    premium_stop: float            # debit: fraction of debit lost; credit: multiple of credit lost
    premium_target: float          # debit: fraction gained; credit: fraction of credit captured
    time_stop_min: int
    quote_source: str
    conviction: float
    notes: dict = field(default_factory=dict)

    @property
    def net_premium(self) -> float:
        """Per lot, INR. + debit / − credit."""
        return sum(l.ratio * l.price for l in self.legs) * self.lot_size

    @property
    def is_credit(self) -> bool:
        return self.net_premium < 0

    def max_loss_per_lot(self) -> float:
        prem = self.net_premium
        rights = {l.right for l in self.legs}
        if not self.is_credit:
            return prem
        width = 0.0
        for r in rights:
            ks = sorted(l.strike for l in self.legs if l.right == r)
            if len(ks) >= 2:
                width = max(width, ks[-1] - ks[0])
        return width * self.lot_size + prem if width else float("inf")

    def planned_risk_per_lot(self) -> float:
        prem = abs(self.net_premium)
        risk = prem * self.premium_stop
        return min(risk, self.max_loss_per_lot())

    def describe(self) -> str:
        legs = ", ".join(f"{'+' if l.ratio > 0 else '−'}{l.strike:g}{l.right}@{l.price:.2f} (Δ{l.delta:+.2f}, IV {l.iv:.1f})"
                         for l in self.legs)
        kind = "credit" if self.is_credit else "debit"
        return (f"{self.structure} {self.symbol} {self.expiry:%d-%b}: {legs}; {kind} ₹{abs(self.net_premium):,.0f}/lot, "
                f"max loss ₹{self.max_loss_per_lot():,.0f}/lot")


class StrikePicker:
    def __init__(self, pricer: IntradayPricer):
        self.pricer = pricer

    def rows(self, chain: pd.DataFrame, right: str, now) -> pd.DataFrame:
        S, T = float(chain.attrs["spot"]), time_to_expiry(now, chain.attrs["expiry"])
        side = right.lower()
        near = chain[np.abs(chain.index.to_numpy(dtype=float) / S - 1) <= 0.08]      # nothing traded is further out
        mids = np.array([mid(row, side) for _, row in near.iterrows()], dtype=float)
        # the IV each price implies under *our* pricer, never the exchange's printed IV: NSE computes its figure
        # from the last trade with its own rate/day-count, and on 29 Sep 2026 its 14.7% against the 14.05% the
        # quote implied made a put bought at ₹85.05 "worth" ₹92.28 the moment it was bought (phantom EV and P&L)
        implied = self.pricer.implied_many(np.where(mids > 0.05, mids, np.nan), near.index.to_numpy(dtype=float), right, S, T) \
            if len(near) else np.array([])
        out = []
        for (K, row), m, iv in zip(near.iterrows(), mids, implied):
            bid, ask = row.get(f"{side}_bid"), row.get(f"{side}_ask")
            quoted = row.get(f"{side}_iv")
            if not (m == m and m > 0.05):
                continue
            if not (iv == iv and iv > 0):
                iv = quoted / 100 if quoted == quoted and quoted and quoted > 0 else float("nan")
            if not (iv == iv and iv > 0):
                continue
            d = self.pricer.greeks(float(K), right, S, T, iv)["delta"]
            spread = (ask - bid) / m if bid == bid and ask == ask and ask >= bid > 0 else np.nan
            out.append({"strike": float(K), "bid": bid, "ask": ask, "mid": m, "iv": iv * 100, "delta": d, "spread": spread})
        return pd.DataFrame(out)

    def by_delta(self, chain, right, target: float, now) -> dict | None:
        r = self.rows(chain, right, now)
        if r.empty:
            return None
        return r.iloc[int(np.argmin(np.abs(r["delta"].abs() - abs(target))))].to_dict()

    def atm(self, chain, right, now) -> dict | None:
        r = self.rows(chain, right, now)
        if r.empty:
            return None
        S = float(chain.attrs["spot"])
        return r.iloc[int(np.argmin(np.abs(r["strike"] - S)))].to_dict()

    def at_strike(self, chain, right, K, now) -> dict | None:
        r = self.rows(chain, right, now)
        m = r[np.isclose(r["strike"], K)] if not r.empty else r
        return None if m.empty else m.iloc[0].to_dict()


def _leg(q: dict, right: str, ratio: int) -> PlanLeg:
    buy = ratio > 0
    px = q["ask"] if buy else q["bid"]
    if not (px == px and px and px > 0):                      # no quote on that side: mid ± 1% as a guess
        px = q["mid"] * (1.01 if buy else 0.99)
    return PlanLeg(q["strike"], right, ratio, float(px), float(q["mid"]), float(q["iv"]), float(q["delta"]))


class Playbook:
    def __init__(self, cfg, pricer: IntradayPricer):
        self.cfg = cfg
        self.p = cfg.get("intraday.setups", {}) or {}
        self.min_conv = cfg.get("intraday.analyst.min_conviction", 0.45)
        self.picker = StrikePicker(pricer)

    def on(self, name: str) -> dict | None:
        p = self.p.get(name, {})
        return p if p.get("enabled", True) else None

    # --- directional builder ------------------------------------------------------------------------
    def _directional(self, setup, view: MarketView, chain, now, direction: int, trigger, thesis, inval, target,
                     conviction: float | None = None) -> TradePlan | None:
        p = self.p.get(setup, {})
        # A structural level a few points from the entry puts the stop inside one minute's noise (29 Sep 2026: a
        # stop 9.5 pts away, σ ≈ 7.6 pts a minute, hit in 2 minutes). Floor the distance at a fraction of the 5m ATR,
        # keep the target at least 1.5× the risk, and say so in the thesis.
        S0, atr5 = float(view.spot), float((view.state or {}).get("atr5") or 0)
        floor = max(float(self.cfg.get("intraday.risk.min_stop_atr5", 0.75)) * atr5, 0.0004 * S0)
        if inval is not None and direction and abs(S0 - inval) < floor:
            inval = S0 - direction * floor
            if target is not None:
                target = S0 + direction * max(abs(target - S0), 1.5 * floor)
            thesis += (f" Stop floored to {inval:,.2f} ({floor:,.0f} pts, the noise floor): the structural level was "
                       f"closer than one minute's noise.")
        right = "CE" if direction > 0 else "PE"
        lot = int(self.cfg.instrument_spec(view.symbol).get("lot_size", 1))
        long_q = self.picker.by_delta(chain, right, p.get("long_delta", 0.55), now)
        if long_q is None:
            return None
        legs = [_leg(long_q, right, +1)]
        structure = "long_call" if direction > 0 else "long_put"
        # a debit spread when IV is rich (sell some of the expensive vol) or always (small accounts)
        if p.get("always_spread", False) or (view.vol_view == "rich" and p.get("spread_when_rich", True)):
            short_q = self.picker.by_delta(chain, right, p.get("short_delta", 0.25), now)
            if short_q is not None and short_q["strike"] != long_q["strike"]:
                legs.append(_leg(short_q, right, -1))
                structure = "bull_call_spread" if direction > 0 else "bear_put_spread"
        return TradePlan(setup, view.symbol, direction, structure, chain.attrs["expiry"], legs, lot, trigger, thesis,
                         inval, target, p.get("premium_stop", 0.35), p.get("premium_target", 0.60),
                         p.get("time_stop_min", 45), chain.attrs.get("source", "?"),
                         view.conviction if conviction is None else conviction)

    # --- setups -----------------------------------------------------------------------------------------
    def orb(self, view: MarketView, s: dict, chain, now) -> TradePlan | None:
        p = self.on("orb")
        if not p or not s["or_done"] or not (15 <= s["minutes"] <= p.get("until_min", 120)) or "last5_close" not in s:
            return None
        mid_or = (s["or_high"] + s["or_low"]) / 2
        for d, lvl, prev_ok in ((1, s["or_high"], s["prev5_high"] <= s["or_high"] * 1.0005),
                                (-1, s["or_low"], s["prev5_low"] >= s["or_low"] * 0.9995)):
            broke = (s["last5_close"] > lvl) if d > 0 else (s["last5_close"] < lvl)
            if not (broke and prev_ok and np.sign(view.score) == d and view.conviction >= self.min_conv):
                continue
            if s.get("rel_volume", 1) == s.get("rel_volume", 1) and s.get("rel_volume", 1) < p.get("min_rel_volume", 0.8):
                continue
            inval = max(mid_or, s["vwap"]) if d > 0 else min(mid_or, s["vwap"])
            risk = abs(s["last"] - inval)
            target = s["last"] + d * max(1.5 * risk, (s["or_high"] - s["or_low"]))
            trig = (f"5m close {s['last5_close']:,.2f} {'above OR high' if d > 0 else 'below OR low'} {lvl:,.2f} "
                    f"(fresh break), relative volume {s.get('rel_volume', float('nan')):.2f}")
            thesis = (f"Opening-range {'breakout' if d > 0 else 'breakdown'} with the evidence behind it: expect range "
                      f"extension ~{abs(target - s['last']):,.0f} pts. Wrong if price gets back to "
                      f"{inval:,.2f} (OR mid / VWAP).")
            return self._directional("orb", view, chain, now, d, trig, thesis, inval, target)
        return None

    def vwap_trend(self, view: MarketView, s: dict, chain, now) -> TradePlan | None:
        p = self.on("vwap_trend")
        if not p or s["minutes"] < p.get("after_min", 45) or "last5_close" not in s or view.day_type not in ("trend", "undetermined"):
            return None
        if view.conviction < self.min_conv or not (s.get("touched_vwap_5") or s.get("touched_ema21_5")):
            return None
        d = int(np.sign(view.score))
        vw, atr5 = s["vwap"], s.get("atr5", 0) or 0
        level_name, level = ("VWAP", vw) if s.get("touched_vwap_5") else ("5m EMA21", s["ema21"])
        resumed = (s["last5_close"] > vw and s["last5_close"] > s["prev5_high"]) if d > 0 else \
                  (s["last5_close"] < vw and s["last5_close"] < s["prev5_low"])
        if not resumed or (d > 0 and s.get("ema9", 0) < s.get("ema21", 0)) or (d < 0 and s.get("ema9", 0) > s.get("ema21", 0)):
            return None
        inval = min(level, vw) - 0.35 * atr5 if d > 0 else max(level, vw) + 0.35 * atr5
        target = s["last"] + d * 2 * abs(s["last"] - inval)
        trig = (f"pullback tagged the {level_name} {level:,.2f} and the 5m bar closed back {'above' if d > 0 else 'below'} "
                f"the prior bar's {'high' if d > 0 else 'low'}")
        thesis = (f"{view.day_type.capitalize()} day continuation: {'buyers' if d > 0 else 'sellers'} defended the "
                  f"{level_name}. Wrong {'below' if d > 0 else 'above'} {inval:,.2f} (level ∓ 0.35 ATR5).")
        return self._directional("vwap_trend", view, chain, now, d, trig, thesis, inval, target)

    def trend_break(self, view: MarketView, s: dict, chain, now) -> TradePlan | None:
        """Trend continuation out of a 30-minute consolidation (a bull/bear flag)."""
        p = self.on("trend_break")
        if not p or "range30_high" not in s or not (p.get("after_min", 60) <= s["minutes"] <= p.get("until_min", 330)):
            return None
        if view.conviction < max(self.min_conv, 0.55) or view.day_type not in ("trend", "undetermined"):
            return None
        d = int(np.sign(view.score))
        hi, lo = s["range30_high"], s["range30_low"]
        width = hi - lo
        atr5 = s.get("atr5", 0) or 0
        if width <= 0 or width > 2.5 * atr5 * 3:                   # not a consolidation
            return None
        broke = s["last5_close"] > hi if d > 0 else s["last5_close"] < lo
        right_side = s["last"] > s["vwap"] if d > 0 else s["last"] < s["vwap"]
        if not (broke and right_side):
            return None
        inval = (hi + lo) / 2
        target = s["last"] + d * max(2 * abs(s["last"] - inval), width)
        trig = (f"5m close {s['last5_close']:,.2f} broke the 30-min {'high' if d > 0 else 'low'} "
                f"{hi if d > 0 else lo:,.2f} after a {width:,.0f}-pt consolidation, on the {'bull' if d > 0 else 'bear'} side of VWAP")
        thesis = (f"Flag breakout in a {view.bias} tape (conviction {view.conviction:.2f}): continuation toward "
                  f"{target:,.0f}. Wrong back inside the flag below/above its midpoint {inval:,.2f}.")
        return self._directional("trend_break", view, chain, now, d, trig, thesis, inval, target)

    def va_reversion(self, view: MarketView, s: dict, chain, now) -> TradePlan | None:
        p = self.on("va_reversion")
        if not p or view.day_type not in ("balance", "undetermined") or "vah" not in s or "rsi5" not in s:
            return None
        if not (p.get("after_min", 75) <= s["minutes"] <= p.get("until_min", 300)):
            return None
        if abs(view.score) > p.get("max_abs_score", 0.4):          # don't fade a directional tape
            return None
        last = s["last"]
        if s["day_high"] > s["vah"] and s["val"] < last < s["vah"] and s["rsi5"] >= p.get("rsi_hi", 60):
            d, edge = -1, s["day_high"]
        elif s["day_low"] < s["val"] and s["val"] < last < s["vah"] and s["rsi5"] <= p.get("rsi_lo", 40):
            d, edge = 1, s["day_low"]
        else:
            return None
        if abs(last - s["poc"]) < 0.25 * (s["vah"] - s["val"]):
            return None
        atr5 = s.get("atr5", 0) or 0
        inval = edge + (0.2 * atr5 if d < 0 else -0.2 * atr5)
        trig = (f"probe {'above VAH' if d < 0 else 'below VAL'} {s['vah'] if d < 0 else s['val']:,.2f} failed; price back "
                f"inside value at {last:,.2f}, 5m RSI {s['rsi5']:.0f}")
        thesis = (f"Balance day: auctions outside value get rejected; rotation back to POC {s['poc']:,.2f}. "
                  f"Wrong beyond the day's {'high' if d < 0 else 'low'} {inval:,.2f}.")
        quality = float(np.clip(0.45 + abs(s["rsi5"] - 50) / 100 + 0.1 * (s.get("value_pos") == "inside value"), 0, 1))
        return self._directional("va_reversion", view, chain, now, d, trig, thesis, inval, s["poc"], quality)

    def range_sell(self, view: MarketView, s: dict, chain, now) -> TradePlan | None:
        p = self.on("range_sell")
        if not p or view.day_type != "balance" or view.vol_view != "rich" or abs(view.score) > p.get("max_abs_score", 0.3):
            return None
        if not (p.get("after_min", 75) <= s["minutes"] <= p.get("until_min", 255)):
            return None
        lot = int(self.cfg.instrument_spec(view.symbol).get("lot_size", 1))
        S = float(chain.attrs["spot"])
        c_atm, p_atm = self.picker.atm(chain, "CE", now), self.picker.atm(chain, "PE", now)
        if c_atm is None or p_atm is None:
            return None
        straddle = c_atm["mid"] + p_atm["mid"]
        width = max(p.get("wing_mult", 1.0) * straddle, 2 * float(self.cfg.instrument_spec(view.symbol).get("strike_step", 50)))
        wc = self.picker.at_strike(chain, "CE", self._nearest(chain, c_atm["strike"] + width), now)
        wp = self.picker.at_strike(chain, "PE", self._nearest(chain, p_atm["strike"] - width), now)
        if wc is None or wp is None:
            return None
        legs = [_leg(wp, "PE", +1), _leg(p_atm, "PE", -1), _leg(c_atm, "CE", -1), _leg(wc, "CE", +1)]
        trig = f"balance day inside value {s['val']:,.2f}–{s['vah']:,.2f}; IV {view.iv:.1f} vs RV {view.rv:.1f}"
        thesis = (f"Two-sided trade and rich premium: sell the ATM straddle (₹{straddle:,.1f}) with wings "
                  f"~{width:,.0f} pts away for defined risk; theta and a vol crush pay if the range holds. "
                  f"Wrong on a close outside value ({s['val']:,.2f}–{s['vah']:,.2f}).")
        return TradePlan("range_sell", view.symbol, 0, "iron_fly", chain.attrs["expiry"], legs, lot, trig, thesis,
                         None, None, p.get("stop_mult", 1.0), p.get("take_profit", 0.30), p.get("time_stop_min", 120),
                         chain.attrs.get("source", "?"), view.conviction,
                         {"range": [s["val"], s["vah"]], "short_strike": c_atm["strike"], "wings": [wp["strike"], wc["strike"]]})

    @staticmethod
    def _nearest(chain, K):
        ks = chain.index.to_numpy(dtype=float)
        return float(ks[np.argmin(np.abs(ks - K))])

    def alternatives(self, plan: TradePlan, chain, now, long_deltas=(0.30, 0.40),
                     spreads=((0.45, 0.30), (0.50, 0.20))) -> list[TradePlan]:
        """Other ways to express a directional plan's view: single long options and debit spreads at other
        deltas. The quant layer prices them all and keeps the best expected value per rupee of risk."""
        if plan.direction == 0:
            return []
        right = "CE" if plan.direction > 0 else "PE"
        base = "call" if right == "CE" else "put"
        out, seen = [], {tuple((l.strike, l.ratio) for l in plan.legs)}

        def add(legs, structure):
            key = tuple((l.strike, l.ratio) for l in legs)
            if key in seen or not legs:
                return
            seen.add(key)
            out.append(TradePlan(plan.setup, plan.symbol, plan.direction, structure, plan.expiry, legs, plan.lot_size,
                                 plan.trigger, plan.thesis, plan.invalidation, plan.target_underlying, plan.premium_stop,
                                 plan.premium_target, plan.time_stop_min, plan.quote_source, plan.conviction, dict(plan.notes)))
        for d in long_deltas:
            q = self.picker.by_delta(chain, right, d, now)
            if q is not None:
                add([_leg(q, right, +1)], f"long_{base}")
        for ld, sd in spreads:
            lq, sq = self.picker.by_delta(chain, right, ld, now), self.picker.by_delta(chain, right, sd, now)
            if lq is not None and sq is not None and lq["strike"] != sq["strike"]:
                add([_leg(lq, right, +1), _leg(sq, right, -1)], "bull_call_spread" if right == "CE" else "bear_put_spread")
        return out

    def scan(self, view: MarketView, s: dict, chain, now) -> list[TradePlan]:
        out = []
        for name in ("orb", "vwap_trend", "trend_break", "va_reversion", "range_sell"):
            try:
                pl = getattr(self, name)(view, s, chain, now)
            except (KeyError, ValueError, IndexError):
                pl = None
            if pl is not None:
                out.append(pl)
        return out
