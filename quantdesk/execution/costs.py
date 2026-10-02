"""Indian transaction costs (brokerage, STT, exchange, SEBI, stamp duty, GST) and
slippage. Costs are the silent killer of short-horizon F&O strategies, so every fill in
the backtest and paper broker goes through here."""
from __future__ import annotations

from ..core.types import Instrument


class CostModel:
    def __init__(self, cfg):
        c = cfg.get("costs", {})
        self.per_order = float(c.get("brokerage_per_order", 20.0))
        self.pct_cap = float(c.get("brokerage_pct_cap", 0.0003))
        self.delivery_brokerage = float(c.get("delivery_brokerage", 0.0))
        self.gst = float(c.get("gst", 0.18))
        self.sebi = float(c.get("sebi_per_crore", 10.0)) / 1e7
        self.segments = c.get("segments", {})
        s = cfg.get("slippage", {})
        self.eq_bps = float(s.get("equity_bps", 4)) / 1e4
        self.idx_bps = float(s.get("index_bps", 1.5)) / 1e4
        self.atr_frac = float(s.get("atr_fraction", 0.02))
        self.opt_half = float(s.get("option_half_spread_pct", 0.012))
        self.opt_min = float(s.get("option_min_half_spread", 0.10))
        self.tick = float(s.get("option_tick", 0.05))

    def fees(self, inst: Instrument, qty: int, price: float, intraday: bool = False) -> tuple[float, dict]:
        """Total statutory + brokerage charges for one executed order (qty signed: + buy)."""
        seg = inst.cost_segment(intraday)
        rates = self.segments.get(seg, {})
        turnover = abs(qty) * price
        buy = qty > 0
        if seg == "equity_delivery":
            brokerage = self.delivery_brokerage
        else:
            brokerage = min(self.per_order, self.pct_cap * turnover) if seg != "options" else self.per_order
        stt = turnover * float(rates.get("stt_buy" if buy else "stt_sell", 0.0))
        exchange = turnover * float(rates.get("exchange", 0.0))
        sebi = turnover * self.sebi
        stamp = turnover * float(rates.get("stamp_buy", 0.0)) if buy else 0.0
        gst = self.gst * (brokerage + exchange + sebi)
        br = {"brokerage": brokerage, "stt": stt, "exchange": exchange, "sebi": sebi, "stamp": stamp, "gst": gst}
        return float(sum(br.values())), br

    def slippage(self, inst: Instrument, price: float, atr: float | None = None) -> float:
        """Adverse price move per unit, always >= 0."""
        if inst.is_option:
            half = max(self.opt_half * price, self.opt_min)
            return round(half / self.tick) * self.tick
        bps = self.idx_bps if inst.kind == "future" and inst.underlying in ("NIFTY", "BANKNIFTY") else self.eq_bps
        extra = self.atr_frac * atr if atr else 0.0
        return price * bps + extra

    def fill_price(self, inst: Instrument, qty: int, ref: float, atr: float | None = None) -> float:
        slip = self.slippage(inst, ref, atr)
        px = ref + slip if qty > 0 else ref - slip
        return max(px, self.tick if inst.is_option else 0.01)

    def round_trip_estimate(self, inst: Instrument, qty: int, price: float) -> float:
        f1, _ = self.fees(inst, abs(qty), price)
        f2, _ = self.fees(inst, -abs(qty), price)
        return f1 + f2 + 2 * abs(qty) * self.slippage(inst, price)
