"""The brain: one picture that connects global markets, news, the Indian tape, the option chain,
the quant layer and the desk's decision.

GlobalFeed      18 global markets (US, Asia, Europe, the dollar, the rupee, crude, gold, US yields,
                US VIX) polled every few minutes: the last *completed* session before India opened,
                the move since 09:15 and in the last 30 minutes, each in σ units of its own history.
Drivers         markets roll up into drivers (US equities, Asia, Europe, Dollar, Rupee, Crude,
                US rates, Fear, Gold), each signed the way it leans on Indian equities.
Links           each driver → NIFTY/BANKNIFTY carries measured weights from the edge research
                (links.json / edges.json): *explanatory* (β to the opening gap, same-5-minute
                correlation) and *predictive* (only a lead that survived holdout + FDR). The brain
                explains with the first and only bets on the second.
News            headlines are mapped to the drivers they're about, so price and story can be
                checked against each other.
Risk overlay    when global stress is high (big σ-moves, US VIX up), size shrinks: volatility
                targeting, not a directional bet.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.global_universe import GLOBAL

IST = "Asia/Kolkata"

DRIVERS = {
    "us":     {"name": "US equities", "members": ["ES", "NQ", "SPX", "NASDAQ", "DJI"], "sign": 1},
    "asia":   {"name": "Asia", "members": ["N225", "HSI", "KOSPI", "SSE"], "sign": 1},
    "europe": {"name": "Europe", "members": ["STOXX", "DAX", "FTSE"], "sign": 1},
    "dollar": {"name": "Dollar (DXY)", "members": ["DXY"], "sign": -1},
    "rupee":  {"name": "USD/INR", "members": ["USDINR"], "sign": -1},
    "crude":  {"name": "Crude (Brent)", "members": ["BRENT"], "sign": -1},
    "rates":  {"name": "US 10Y yield", "members": ["UST10"], "sign": -1},
    "fear":   {"name": "US VIX", "members": ["USVIX"], "sign": -1},
    "gold":   {"name": "Gold", "members": ["GOLD"], "sign": 0},
}
# prior weights of each driver in the descriptive risk regime (not a trading signal)
REGIME_W = {"us": 1.0, "asia": 0.6, "europe": 0.5, "fear": 0.8, "dollar": 0.4, "rupee": 0.4, "crude": 0.4, "rates": 0.3}

NEWS_DRIVERS = {
    "us": r"\b(wall street|s&p|nasdaq|dow jones|us stocks|u\.s\. stocks|us markets)\b",
    "rates": r"\b(fed|fomc|federal reserve|powell|treasury|us yields|bond yields|rate cut|rate hike)\b",
    "asia": r"\b(china|chinese|hang seng|nikkei|japan|asian markets|asia|kospi|shanghai)\b",
    "europe": r"\b(europe|european|ecb|dax|ftse|euro zone|eurozone)\b",
    "dollar": r"\b(dollar index|dxy|greenback|us dollar)\b",
    "rupee": r"\b(rupee|inr|forex reserves)\b",
    "crude": r"\b(crude|oil prices|brent|opec|wti)\b",
    "fear": r"\b(war|attack|missile|geopolitic\w*|sanctions|tensions|vix)\b",
    "india": r"\b(nifty|sensex|dalal street|rbi|fii|fpi|sebi|indian markets|india inc)\b",
}


def news_drivers(title: str) -> list[str]:
    t = title.lower()
    return [k for k, pat in NEWS_DRIVERS.items() if re.search(pat, t)]


# ---- global market data ---------------------------------------------------------------------------------------
class GlobalFeed:
    """Live global prices (Yahoo), fetched in one batch at most every `refresh_min` minutes. `fetch` is
    pluggable (tests / replays pass frames and a clock; nothing after `now` is ever used)."""

    def __init__(self, cfg=None, fetch=None, refresh_min: float = 5.0, keys: list[str] | None = None):
        gc = (cfg.get("intraday.global", {}) if cfg is not None else {}) or {}
        self.enabled = gc.get("enabled", True)
        self.refresh_min = gc.get("refresh_min", refresh_min)
        self.keys = keys or list(GLOBAL)
        self.fetch = fetch or self._yahoo
        self.intraday: dict[str, pd.DataFrame] = {}       # 5m bars, UTC index (bar start)
        self.daily: dict[str, pd.DataFrame] = {}          # daily bars, exchange-local dates
        self.last_fetch: pd.Timestamp | None = None
        self.health: dict[str, str] = {}
        self._daily_day = None

    def _yahoo(self, now: pd.Timestamp) -> tuple[dict, dict]:
        import yfinance as yf
        intraday, daily = {}, {}
        tick = {GLOBAL[k]["yahoo"]: k for k in self.keys}
        d5 = yf.download(list(tick), period="5d", interval="5m", group_by="ticker", auto_adjust=False, progress=False,
                         threads=True)
        # completed daily sessions don't change during India's day: fetch them once per day (half the requests)
        day = pd.Timestamp(now).tz_convert(IST).date()
        dd = None
        if self._daily_day != day:
            dd = yf.download(list(tick), period="3mo", interval="1d", group_by="ticker", auto_adjust=False, progress=False,
                             threads=True)
            self._daily_day = day
        for t, k in tick.items():
            for src, dst, local in ((d5, intraday, False), (dd, daily, True)):
                if src is None:
                    continue
                try:
                    f = src[t].rename(columns={c: str(c).lower() for c in src[t].columns})[["open", "high", "low", "close"]]
                    f = f.dropna(subset=["close"])
                    if not len(f):
                        continue
                    idx = pd.DatetimeIndex(f.index)
                    if local:
                        f.index = pd.DatetimeIndex(idx.date if idx.tz is not None else idx.normalize())
                    else:
                        f.index = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
                    dst[k] = f.astype(float)
                except (KeyError, TypeError, ValueError):
                    continue
        return intraday, daily

    def refresh(self, now: pd.Timestamp, force: bool = False) -> bool:
        if not self.enabled or (not force and self.last_fetch is not None
                                and now - self.last_fetch < pd.Timedelta(minutes=self.refresh_min)):
            return False
        self.last_fetch = now
        try:
            intraday, daily = self.fetch(now)
        except Exception as exc:
            self.health["global"] = f"fail {str(exc)[:100]}"
            return False
        cut = pd.Timestamp(now).tz_convert("UTC")
        for k, f in intraday.items():                     # a 5m bar is known once it has closed
            self.intraday[k] = f[f.index + pd.Timedelta(minutes=5) <= cut]
        for k, f in daily.items():
            self.daily[k] = f
        self.health["global"] = f"ok {len(self.intraday)} intraday, {len(self.daily)} daily"
        return True

    def market(self, key: str, now: pd.Timestamp) -> dict | None:
        """One market's state at `now`: prior completed session, move since India's open, last 30 minutes."""
        meta = GLOBAL.get(key, {})
        now_ist = pd.Timestamp(now).tz_convert(IST)
        today = now_ist.date()
        out = {"key": key, "name": meta.get("name", key), "region": meta.get("region"), "india": meta.get("india", 0)}
        d = self.daily.get(key)
        if d is not None and len(d) > 25:
            done = d[d.index.date < today]                   # sessions that finished before India's day
            r = np.log(done["close"] / done["close"].shift(1)).dropna()
            if len(r) > 20:
                out["prior_ret"] = float(r.iloc[-1])
                out["prior_z"] = float(r.iloc[-1] / max(r.tail(60).std(), 1e-6))
                out["prior_date"] = str(done.index[-1].date())
                out["last"] = float(done["close"].iloc[-1])
        m = self.intraday.get(key)
        if m is not None and len(m):
            cut = pd.Timestamp(now).tz_convert("UTC")
            m = m[m.index + pd.Timedelta(minutes=5) <= cut]
            if len(m):
                known = m["close"].copy()
                known.index = known.index + pd.Timedelta(minutes=5)
                last_t = known.index[-1]
                out["last"] = float(known.iloc[-1])
                out["last_ts"] = str(last_t.tz_convert(IST))
                out["live"] = bool(cut - last_t <= pd.Timedelta(minutes=15))
                lr = np.log(known).diff().dropna()
                s30 = float(lr.tail(600).std() * math.sqrt(6)) if len(lr) > 30 else float("nan")
                ago = known[known.index <= cut - pd.Timedelta(minutes=30)]
                if out["live"] and len(ago) and cut - ago.index[-1] <= pd.Timedelta(minutes=45):
                    r30 = math.log(known.iloc[-1] / ago.iloc[-1])
                    out["r30"], out["z30"] = r30, (r30 / s30 if s30 == s30 and s30 > 0 else float("nan"))
                open_ts = pd.Timestamp(now_ist.normalize() + pd.Timedelta(hours=9, minutes=15)).tz_convert("UTC")
                at_open = known[known.index <= open_ts]
                if len(at_open) and cut > open_ts:
                    out["since_open"] = math.log(known.iloc[-1] / at_open.iloc[-1])
        return out if len(out) > 4 else None

    def snapshot(self, now) -> dict[str, dict]:
        return {k: m for k in self.keys if (m := self.market(k, now)) is not None}


# ---- the graph -------------------------------------------------------------------------------------------------
@dataclass
class Link:
    driver: str
    target: str
    gap_beta: float = float("nan")         # opening gap per unit prior-session return (research links.json)
    gap_corr: float = float("nan")
    gap_from: str = ""                     # the member market that β was measured on
    co_corr: float = float("nan")          # same-5-minute correlation during Indian hours
    lead_t: float = float("nan")           # t-stat of the best predictive test for this driver
    lead_edge: bool = False                # survived holdout + FDR + costs
    lead_sign: int = 0                     # +1: India follows the hypothesised lean; -1: the measured edge is the reverse
    lead_kind: str = ""                    # "prior" (last completed session) or "30m" (last 30 minutes)


@dataclass
class BrainState:
    ts: str
    regime: str
    regime_score: float
    stress: float
    size_mult: float
    drivers: list = field(default_factory=list)
    markets: dict = field(default_factory=dict)
    gap: dict | None = None
    evidence: list = field(default_factory=list)
    narrative: str = ""


class Brain:
    def __init__(self, cfg, gfeed: GlobalFeed | None, research_links: list | None = None, research_edges: list | None = None):
        bc = (cfg.get("intraday.brain", {}) or {}) if cfg is not None else {}
        self.cfg = cfg
        self.gfeed = gfeed
        self.stress_z = float(bc.get("stress_z", 2.0))
        self.min_size_mult = float(bc.get("min_size_mult", 0.5))
        self.links = self._links(research_links or [], research_edges or [])

    @staticmethod
    def _links(links: list, edges: list) -> dict:
        """(driver, target) → Link, from the research outputs (member market with the strongest evidence wins)."""
        member_of = {m: d for d, v in DRIVERS.items() for m in v["members"]}
        out: dict = {}
        for l in links:
            d = member_of.get(l.get("from"))
            if not d:
                continue
            lk = out.setdefault((d, l["to"]), Link(d, l["to"]))
            if l.get("what") == "opening gap" and not (abs(lk.gap_corr) >= abs(l.get("corr", 0))):
                lk.gap_beta, lk.gap_corr, lk.gap_from = l.get("beta", float("nan")), l.get("corr", float("nan")), l.get("from", "")
            if l.get("what") == "same 5 minutes" and not (abs(lk.co_corr) >= abs(l.get("corr", 0))):
                lk.co_corr = l.get("corr", float("nan"))
        for e in edges:
            rid = str(e.get("id", ""))
            if not (rid.startswith("G-") or rid.startswith("L-")):
                continue
            d = member_of.get(rid[2:])
            if not d:
                continue
            lk = out.setdefault((d, e["symbol"]), Link(d, e["symbol"]))
            t = e.get("t")
            is_edge = e.get("verdict") == "EDGE"
            # the link's predictive fields come from its validated test if it has one, else from its strongest test
            if t is not None and ((is_edge and not lk.lead_edge) or (is_edge == lk.lead_edge and not (abs(lk.lead_t) >= abs(t)))):
                lk.lead_t = t
                lk.lead_sign = int(np.sign(e.get("effect_bps") or 0))
                lk.lead_kind = "prior" if rid.startswith("G-") else "30m"
                lk.lead_edge = lk.lead_edge or is_edge
        return out

    def think(self, target: str, now, news_items: list | None = None, gap: float | None = None) -> BrainState:
        """Global state → drivers → links → evidence and a narrative for `target` (NIFTY/BANKNIFTY)."""
        mk = self.gfeed.snapshot(now) if self.gfeed is not None else {}
        drivers = []
        for d, v in DRIVERS.items():
            ms = [mk[m] for m in v["members"] if m in mk]
            if not ms:
                continue
            # the member that best says what this driver is doing: live with a 30-minute move, else a completed session
            lead = (next((m for m in ms if m.get("live") and m.get("r30") is not None), None)
                    or next((m for m in ms if m.get("prior_ret") is not None), ms[0]))
            prior = [m["prior_z"] for m in ms if m.get("prior_z") == m.get("prior_z") and "prior_z" in m]
            now_z = [m["z30"] for m in ms if m.get("z30") == m.get("z30") and "z30" in m]
            sign = v["sign"]
            lk = self.links.get((d, target)) or Link(d, target)
            item = {"id": d, "name": v["name"], "sign": sign, "live": any(m.get("live") for m in ms),
                    "prior_z": float(np.mean(prior)) if prior else None, "z30": float(np.mean(now_z)) if now_z else None,
                    "lead": {"key": lead["key"], "last": lead.get("last"), "prior_ret": lead.get("prior_ret"),
                             "since_open": lead.get("since_open"), "r30": lead.get("r30")},
                    "gap_beta": _num(lk.gap_beta), "gap_corr": _num(lk.gap_corr), "co_corr": _num(lk.co_corr),
                    "lead_t": _num(lk.lead_t), "validated": bool(lk.lead_edge), "lead_sign": lk.lead_sign,
                    "lead_kind": lk.lead_kind,
                    # the prior-session move of the market the gap β was measured on (else the members' average)
                    "gap_prior_ret": (mk[lk.gap_from]["prior_ret"] if lk.gap_from in mk and "prior_ret" in mk[lk.gap_from]
                                      else _mean([m.get("prior_ret") for m in ms]))}
            # India-signed pressure now: prefer the live 30-minute move, else the prior session
            z = item["z30"] if item["z30"] is not None else item["prior_z"]
            item["pressure"] = float(np.clip(sign * z, -4, 4)) if z is not None and sign else 0.0
            drivers.append(item)
        # news mapped to drivers (recency-weighted tone of the last 2 hours)
        if news_items:
            now_ts = pd.Timestamp(now)
            for item in drivers + [{"id": "india"}]:
                rel = [x for x in news_items if item["id"] in news_drivers(x.title) and x.ts <= now_ts
                       and now_ts - x.ts <= pd.Timedelta(hours=2)]
                if rel and "name" in item:
                    w = [0.5 ** ((now_ts - x.ts).total_seconds() / 60 / 45) for x in rel]
                    item["news_tone"] = float(np.dot(w, [x.sentiment for x in rel]) / sum(w))
                    item["news_n"] = len(rel)
                    item["news_latest"] = max(rel, key=lambda x: x.ts).title
        # descriptive regime and the risk overlay
        num = den = 0.0
        for it in drivers:
            w = REGIME_W.get(it["id"], 0)
            if w and it["pressure"]:
                num += w * np.clip(it["pressure"], -3, 3) / 3
                den += w
        score = float(num / den) if den else 0.0
        regime = "risk-on" if score > 0.25 else "risk-off" if score < -0.25 else "mixed"
        big = [abs(it["z30"]) for it in drivers if it.get("z30") is not None and it["id"] in ("us", "asia", "europe", "fear")]
        fear = next((it for it in drivers if it["id"] == "fear"), None)
        stress = max(big + [abs(fear["prior_z"]) if fear and fear.get("prior_z") is not None else 0.0] or [0.0])
        size_mult = 1.0 if stress < self.stress_z else max(self.min_size_mult, self.stress_z / stress)
        # evidence for the analyst: only validated predictive links carry weight
        evidence = []
        for it in drivers:
            if not it["validated"]:
                continue
            # the measured edge decides the direction, not the hypothesis: e.g. BANKNIFTY *fades* Europe's last session
            z = it["prior_z"] if it["lead_kind"] == "prior" else it["z30"]
            if z is None or not it["sign"] and it["lead_sign"] == 0:
                continue
            direction = float(np.clip(it["lead_sign"] * (it["sign"] or 1) * z / 2, -1, 1))
            if not direction:
                continue
            w = float(min(1.0, abs(it["lead_t"] or 0) / 5))
            what = "last session" if it["lead_kind"] == "prior" else "last 30 minutes"
            evidence.append({"factor": f"global_{it['id']}", "direction": direction, "weight": w,
                             "observation": f"{it['name']} {_fmt_move(it)}: {target} has {'followed' if it['lead_sign'] > 0 else 'faded'} "
                                            f"its {what} (validated, t {it['lead_t']:+.1f}) → {'bullish' if direction > 0 else 'bearish'} lean"})
        # the opening gap, attributed to what happened while India was closed
        gap_info = None
        if gap is not None and gap == gap:
            parts = []
            for it in drivers:
                b, r = it.get("gap_beta"), it.get("gap_prior_ret")
                if b is not None and r is not None and abs(it.get("gap_corr") or 0) >= 0.15:
                    parts.append((it["name"], b * r))
            explained = float(sum(p for _, p in parts))
            gap_info = {"gap": gap, "explained": explained, "parts": sorted(parts, key=lambda x: -abs(x[1]))[:4],
                        "share": float(explained / gap) if abs(gap) > 1e-4 else None}
        st = BrainState(str(now), regime, score, float(stress), float(size_mult), drivers, mk, gap_info, evidence)
        st.narrative = self._narrate(target, st)
        return st

    def _narrate(self, target: str, st: BrainState) -> str:
        movers = sorted([d for d in st.drivers if d["pressure"]], key=lambda d: -abs(d["pressure"]))[:4]
        bits = [f"{d['name']} {_fmt_move(d)}" for d in movers]
        s = f"Global: {st.regime} ({st.regime_score:+.2f})" + (": " + "; ".join(bits) if bits else "") + "."
        if st.gap:
            g = st.gap
            parts = f" ({', '.join(f'{n} {v:+.2%}' for n, v in g['parts'][:3])})" if g["parts"] else ""
            if g["explained"] * g["gap"] < 0 and abs(g["explained"]) > 5e-4:
                s += f" India opened {g['gap']:+.2%}, against the global cue of {g['explained']:+.2%}{parts}."
            else:
                s += f" The {g['gap']:+.2%} gap: global moves while India was shut account for {g['explained']:+.2%}{parts}."
        news = [d for d in st.drivers if d.get("news_n")]
        if news:
            n = max(news, key=lambda d: d["news_n"])
            s += f" News is loudest on {n['name'].lower()} ({n['news_n']} stories, tone {n['news_tone']:+.2f})."
        if st.size_mult < 1:
            s += f" Global stress {st.stress:.1f}σ: size ×{st.size_mult:.2f}."
        val = [d["name"] for d in st.drivers if d["validated"]]
        s += (f" Validated leads for {target}: {', '.join(val)}." if val else
              " No global market has a validated lead on the Indian session yet, so global moves explain but don't vote.")
        return s


def _mean(xs):
    xs = [x for x in xs if x is not None and x == x]
    return float(np.mean(xs)) if xs else None


def _num(x):
    return None if x is None or (isinstance(x, float) and x != x) else float(x)


def _fmt_move(d: dict) -> str:
    L = d.get("lead") or {}
    if L.get("r30") is not None:
        return f"{L['r30']:+.2%} in 30m" + (f", {L['since_open']:+.2%} since 09:15" if L.get("since_open") is not None else "")
    if L.get("prior_ret") is not None:
        return f"{L['prior_ret']:+.2%} last session"
    return "flat"
