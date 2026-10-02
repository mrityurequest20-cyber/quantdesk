"""The analyst: turns the intraday state (+ option chain, + order flow) into a view.

It thinks like a discretionary desk, but every step is explicit and logged:
  1. gather evidence — each item is an observation, a signed direction in [-1, 1], a weight
     and a category (trend, structure, momentum, flow, options, volatility);
  2. weigh it — bias score = Σ w·d / Σ w, conviction = |score| × agreement;
  3. classify the day — trend / balance / volatile / undetermined;
  4. read volatility — implied vs realised → premium rich / fair / cheap;
  5. list reasons NOT to trade (vetoes);
  6. write the narrative that goes into the journal.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

DEFAULT_WEIGHTS = {
    "vwap": 1.0, "ema_5m": 0.8, "htf": 0.6, "orb": 1.0, "value": 0.6, "prev_day": 0.5, "cpr": 0.4,
    "supertrend": 0.4, "rsi": 0.4, "flow": 0.6, "divergence": 0.5, "pcr": 0.3, "oi_walls": 0.4,
    "vix": 0.3, "max_pain": 0.3, "iv_move": 0.3, "news": 0.5, "model": 0.8, "research": 0.3,
}


@dataclass
class Evidence:
    factor: str
    category: str
    direction: float
    weight: float
    observation: str

    def signed(self) -> float:
        return self.direction * self.weight


@dataclass
class MarketView:
    symbol: str
    ts: object
    spot: float
    bias: str
    score: float
    conviction: float
    day_type: str
    vol_view: str
    iv: float | None
    rv: float | None
    evidence: list[Evidence]
    vetoes: list[str]
    levels: dict
    narrative: str
    state: dict = field(default_factory=dict)
    chain: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        d = asdict(self)
        d["ts"] = str(self.ts)
        d.pop("state", None)
        return d


def _fmt(x, nd=2):
    return f"{x:,.{nd}f}" if x is not None and x == x else "—"


class Analyst:
    def __init__(self, cfg):
        a = cfg.get("intraday.analyst", {}) or {}
        self.w = {**DEFAULT_WEIGHTS, **(a.get("weights") or {})}
        self.rich = a.get("iv_rv_rich", 1.15)
        self.cheap = a.get("iv_rv_cheap", 0.90)
        self.events_days = cfg.get("calendar.event_blackout_days", 1)
        self.cfg = cfg

    def assess(self, symbol: str, s: dict, chain: dict | None = None, vix: dict | None = None,
               is_expiry_day: bool = False, event: str | None = None, flow: dict | None = None,
               news: dict | None = None, quant: dict | None = None, brain: dict | None = None) -> MarketView:
        ev: list[Evidence] = []
        w = self.w
        last = s["last"]

        def add(f, cat, d, obs, weight=None):
            ev.append(Evidence(f, cat, float(np.clip(d, -1, 1)), w.get(f, 0.5) if weight is None else weight, obs))

        # --- trend ---------------------------------------------------------------------------------
        vw, slope = s["vwap"], s["vwap_slope"]
        d = (1 if last > vw else -1) * (1.0 if np.sign(slope) == np.sign(last - vw) else 0.5)
        add("vwap", "trend", d, f"price {_fmt(last)} {'above' if last > vw else 'below'} {s['vwap_kind'].upper()} {_fmt(vw)} "
                                f"({s['vwap_z']:+.1f}σ), VWAP {'rising' if slope > 0 else 'falling'} {slope:+.2%}/15m")
        if "ema9" in s:
            d = 1 if s["ema9"] > s["ema21"] else -1
            add("ema_5m", "trend", d, f"5m EMA9 {_fmt(s['ema9'])} {'>' if d > 0 else '<'} EMA21 {_fmt(s['ema21'])}")
            add("supertrend", "trend", s["st5"], f"5m Supertrend {'long' if s['st5'] > 0 else 'short'}")
        if "htf_slope" in s:
            add("htf", "trend", np.sign(s["htf_slope"]) * min(abs(s["htf_slope"]) / 0.002, 1),
                f"15m EMA20 slope {s['htf_slope']:+.2%} ({'up' if s['htf_slope'] > 0 else 'down'})")

        # --- structure -----------------------------------------------------------------------------
        if s["or_done"]:
            if last > s["or_high"]:
                add("orb", "structure", 1, f"above the opening range high {_fmt(s['or_high'])}")
            elif last < s["or_low"]:
                add("orb", "structure", -1, f"below the opening range low {_fmt(s['or_low'])}")
            else:
                add("orb", "structure", 0, f"inside the opening range {_fmt(s['or_low'])}–{_fmt(s['or_high'])}")
        if "vah" in s:
            pos = s["value_pos"]
            add("value", "structure", 1 if pos == "above value" else -1 if pos == "below value" else 0,
                f"{pos} (session POC {_fmt(s['poc'])}, VA {_fmt(s['val'])}–{_fmt(s['vah'])})")
        if "pdh" in s:
            d = 1 if last > s["pdh"] else -1 if last < s["pdl"] else 0
            add("prev_day", "structure", d, f"prior day H/L {_fmt(s['pdh'])}/{_fmt(s['pdl'])}: "
                                             f"{'broke above' if d > 0 else 'broke below' if d < 0 else 'inside'}")
            d = 1 if last > s["cpr_tc"] else -1 if last < s["cpr_bc"] else 0
            add("cpr", "structure", d, f"CPR {_fmt(s['cpr_bc'])}–{_fmt(s['cpr_tc'])} (width {s['cpr_width']:.2%}); price "
                                       f"{'above' if d > 0 else 'below' if d < 0 else 'inside'}")

        # --- momentum ------------------------------------------------------------------------------
        if "rsi5" in s:
            r = s["rsi5"]
            d = (r - 50) / 20
            note = " — stretched" if r > 72 or r < 28 else ""
            add("rsi", "momentum", d, f"5m RSI {r:.0f}{note}")

        # --- flow ------------------------------------------------------------------------------------
        if flow and flow.get("source") == "ticks":
            add("flow", "flow", np.tanh(flow["delta_30"] / max(flow["volume_30"], 1) * 5),
                f"tick delta last 30m {flow['delta_30']:+,.0f} of {flow['volume_30']:,.0f} (stacked imbalances "
                f"buy {flow.get('stacked_buy', 0)}, sell {flow.get('stacked_sell', 0)})")
        elif "cvd_slope_30" in s:
            add("flow", "flow", np.tanh(s["cvd_slope_30"] * 3), f"approx. CVD (close-location) 30m slope "
                                                                 f"{s['cvd_slope_30']:+.2f}", weight=w["flow"] * 0.5)
        if s.get("cvd_divergence"):
            dv = s["cvd_divergence"]
            add("divergence", "flow", dv, f"{'bullish' if dv > 0 else 'bearish'} price/CVD divergence (15-bar swing)")

        # --- options positioning ---------------------------------------------------------------------
        c = chain or {}
        if c.get("pcr_oi") == c.get("pcr_oi") and c.get("pcr_oi") is not None and c.get("source") != "model":
            p = c["pcr_oi"]
            add("pcr", "options", np.clip((p - 1.0) / 0.4, -1, 1), f"PCR (OI) {p:.2f}"
                + (f", PCR of today's OI change {c['pcr_doi']:.2f}" if c.get("pcr_doi") == c.get("pcr_doi") and c.get("pcr_doi") else ""))
        if c.get("call_wall") == c.get("call_wall") and c.get("call_wall"):
            cw, pw = c["call_wall"], c.get("put_wall")
            dc, dp = (cw - last) / last, (last - pw) / last if pw else np.inf
            if dc < 0.003:
                add("oi_walls", "options", -0.7, f"just under the call-OI wall {cw:,.0f} ({dc:.2%} away): resistance")
            elif dp < 0.003:
                add("oi_walls", "options", 0.7, f"just above the put-OI wall {pw:,.0f} ({dp:.2%} away): support")
            else:
                add("oi_walls", "options", 0, f"between put wall {pw:,.0f} and call wall {cw:,.0f}")
        if is_expiry_day and c.get("max_pain") and s["minutes"] > 180:
            mp = c["max_pain"]
            add("max_pain", "options", np.clip((mp - last) / (last * 0.004), -1, 1), f"expiry day: max pain {mp:,.0f} "
                                                                                     f"({(mp / last - 1):+.2%} from spot)")

        # --- news (headline tone is noisy: modest weight, scaled by how many stories back it) ---------------
        if news and news.get("n"):
            tone = news["tone"]
            add("news", "news", tone * (0.4 + news.get("confidence", 0.5)),
                f"news tone {tone:+.2f} over the last 2h ({news['n']} {'story' if news['n'] == 1 else 'stories'}); latest "
                f"\u201c{news['latest'][:90]}\u201d ({news.get('latest_age_min', 0):.0f} min ago)")

        # --- the direction model (only once it has passed its walk-forward test) ------------------------
        if quant and quant.get("valid") and quant.get("p_model") is not None:
            pm = quant["p_model"]
            add("model", "quant", (pm - 0.5) * 6, f"direction model: P(up in 30m) {pm:.2f} (walk-forward AUC "
                                                   f"{quant.get('auc', float('nan')):.3f}, {quant.get('samples', 0):,} samples)")

        # --- the brain: global links that survived research (weight from their t-stat) ---------------------
        for e in (brain or {}).get("evidence", []):
            add(e["factor"], "global", e["direction"], e["observation"], weight=e["weight"])

        # --- research priors (only what survived the weekly edge research on years of real data) --------
        if quant and quant.get("research_drift"):
            rd = quant["research_drift"]
            add("research", "quant", float(np.sign(rd["bps_day"])) * 0.5,
                f"{rd.get('n', 0):,}-day intraday drift {rd['bps_day']:+.1f} bps/day open→close (t {rd.get('t', 0):+.2f}, "
                f"holds out of sample): a mild {'long' if rd['bps_day'] > 0 else 'short'} lean")

        # --- volatility --------------------------------------------------------------------------------
        if vix:
            ch = vix.get("chg", 0.0)
            add("vix", "volatility", -np.clip(ch / 0.06, -1, 1), f"India VIX {vix.get('last', float('nan')):.2f} ({ch:+.1%} today)")
        iv = c.get("atm_iv") if c else None
        rv_parts = [x for x in (s.get("rv_intraday"), s.get("rv_5d")) if x == x and x is not None]
        rv = float(np.mean(rv_parts)) if rv_parts else None
        if iv and rv:
            ratio = iv / rv
            vol_view = "rich" if ratio >= self.rich else "cheap" if ratio <= self.cheap else "fair"
        else:
            ratio, vol_view = None, "unknown"

        # --- weigh ------------------------------------------------------------------------------------------
        tot_w = sum(e.weight for e in ev if e.direction != 0) or 1.0
        score = sum(e.signed() for e in ev) / sum(e.weight for e in ev) if ev else 0.0
        agree = sum(e.weight for e in ev if np.sign(e.direction) == np.sign(score) and e.direction != 0) / tot_w
        conviction = float(abs(score) * agree)
        bias = "bullish" if score > 0.15 else "bearish" if score < -0.15 else "neutral"

        # --- day type -------------------------------------------------------------------------------------
        adx = s.get("adx5", 0) or 0
        if s["minutes"] < 45:
            day_type = "forming"
        elif (s["ib_ext"] > 0.5 or (s["or_done"] and (last > s["or_high"] or last < s["or_low"]))) and adx > 22 \
                and (s["close_loc"] > 0.75 or s["close_loc"] < 0.25) and s.get("value_pos") != "inside value":
            day_type = "trend"
        elif s.get("range_vs_avg", 1) == s.get("range_vs_avg", 1) and (s.get("range_vs_avg") or 1) > 1.4 and 0.3 < s["close_loc"] < 0.7:
            day_type = "volatile"
        elif s.get("value_pos") == "inside value" and adx < 22:
            day_type = "balance"
        else:
            day_type = "undetermined"

        # --- vetoes ------------------------------------------------------------------------------------------
        vetoes = []
        if s["minutes"] < 5:
            vetoes.append("first 5 minutes: price discovery, spreads wide")
        if event:
            vetoes.append(f"scheduled event: {event}")
        if news and news.get("breaking"):
            b = news["breaking"]
            vetoes.append(f"breaking news {b['age_min']:.0f} min ago \u201c{b['title'][:90]}\u201d ({b['source']}): "
                          f"letting the market digest it")
        if c.get("atm_spread_pct") and c["atm_spread_pct"] > self.cfg.get("intraday.risk.max_spread_pct", 0.06):
            vetoes.append(f"ATM option spread {c['atm_spread_pct']:.1%} too wide")
        if "rsi5" in s and (s["rsi5"] > 80 or s["rsi5"] < 20):
            vetoes.append(f"5m RSI {s['rsi5']:.0f}: too stretched to chase")

        levels = {k: s[k] for k in ("vwap", "or_high", "or_low", "ib_high", "ib_low", "pdh", "pdl", "poc", "vah", "val",
                                    "cpr_tc", "cpr_bc", "day_high", "day_low") if k in s}
        if c.get("call_wall") == c.get("call_wall") and c.get("call_wall"):
            levels["call_wall"], levels["put_wall"] = c["call_wall"], c.get("put_wall")

        pro = sorted([e for e in ev if np.sign(e.direction) == np.sign(score) and e.direction], key=lambda e: -abs(e.signed()))[:4]
        con = sorted([e for e in ev if np.sign(e.direction) == -np.sign(score) and e.direction], key=lambda e: -abs(e.signed()))[:3]
        head = (f"{symbol} {_fmt(last)} ({s.get('chg', 0):+.2%} on the day, {s['phase']}): {day_type} day; "
                f"bias {bias} (score {score:+.2f}, conviction {conviction:.2f}).")
        parts = [head]
        if pro:
            parts.append("For: " + "; ".join(e.observation for e in pro) + ".")
        if con:
            parts.append("Against: " + "; ".join(e.observation for e in con) + ".")
        if ratio:
            parts.append(f"Vol: ATM IV {iv:.1f} vs realised {rv:.1f} (×{ratio:.2f}) → premium {vol_view}.")
        if brain and brain.get("narrative"):
            parts.append(brain["narrative"])
        if quant and quant.get("sigma_30m_pct"):
            src = (f"model AUC {quant['auc']:.3f}" if quant.get("valid") else
                   f"model off: {quant.get('model_status', 'untrained')}")
            parts.append(f"Quant: 30-min σ {quant['sigma_30m_pct']:.2f}% ({quant.get('vol_source')}); {src}.")
        if vetoes:
            parts.append("No-trade flags: " + "; ".join(vetoes) + ".")
        return MarketView(symbol, s["ts"], last, bias, float(score), conviction, day_type, vol_view, iv, rv, ev, vetoes,
                          levels, " ".join(parts), s, c)
