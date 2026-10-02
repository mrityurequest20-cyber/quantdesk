"""Live news for the desk: what the market is being told, read the way a desk reads the tape.

Every few minutes the desk pulls Indian market and macro headlines from public RSS feeds
(Economic Times, Moneycontrol, Mint, Business Standard, Google News, RBI), then:

  1. dedupes the same story across outlets (token overlap on the headline);
  2. decides what it is about: the index, the banks (BANKNIFTY), macro, or neither;
  3. scores the headline with a finance word list that knows the subject matters
     ("crude rises" / "inflation rises" / "yields rise" are *bad* for Indian equities,
     "Nifty rises" is good), with negation handling;
  4. rates impact: RBI policy, Fed, budget, CPI/GDP prints, war and crashes are high impact.

What the engine uses: a recency-weighted tone per underlying (half-life 45 min) as one piece
of evidence among many (headline sentiment is noisy, so its weight is modest), and a
stand-aside flag for ~15 minutes after a high-impact story breaks.

Headlines are journaled (table `news`) with their scores, so every read can be audited.
"""
from __future__ import annotations

import datetime as dt
import email.utils
import hashlib
import html
import logging
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import pandas as pd

log = logging.getLogger(__name__)
IST = "Asia/Kolkata"

DEFAULT_SOURCES = [
    {"name": "ET Markets", "url": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"},
    {"name": "ET Stocks", "url": "https://economictimes.indiatimes.com/markets/stocks/news/rssfeeds/2146842.cms"},
    {"name": "Moneycontrol", "url": "https://www.moneycontrol.com/rss/marketreports.xml"},
    {"name": "Moneycontrol latest", "url": "https://www.moneycontrol.com/rss/latestnews.xml"},
    {"name": "Mint Markets", "url": "https://www.livemint.com/rss/markets"},
    {"name": "Business Standard", "url": "https://www.business-standard.com/rss/markets-106.rss"},
    {"name": "Google News India", "url": "https://news.google.com/rss/search?q=Nifty%20OR%20Sensex%20OR%20%22Bank%20Nifty%22"
                                         "%20OR%20RBI%20OR%20%22stock%20market%22%20when%3A1d&hl=en-IN&gl=IN&ceid=IN%3Aen"},
    {"name": "Google News Macro", "url": "https://news.google.com/rss/search?q=%22Federal%20Reserve%22%20OR%20%22crude%20oil%22"
                                         "%20OR%20%22Treasury%20yields%22%20OR%20%22Wall%20Street%22%20when%3A1d&hl=en-IN&gl=IN&ceid=IN%3Aen"},
    {"name": "RBI", "url": "https://www.rbi.org.in/pressreleases_rss.xml"},
    {"name": "Google News World", "url": "https://news.google.com/rss/search?q=%22global%20markets%22%20OR%20%22Asian%20markets%22"
                                         "%20OR%20%22European%20stocks%22%20OR%20%22US%20stocks%22%20OR%20%22oil%20prices%22%20when%3A1d"
                                         "&hl=en-US&gl=US&ceid=US%3Aen"},
]

# ---- vocabulary -----------------------------------------------------------------------------------------------
INDEX_WORDS = {"nifty": 3, "sensex": 3, "bank nifty": 3, "banknifty": 3, "nifty bank": 3, "dalal street": 3, "d-street": 3,
               "stock market": 2, "markets": 1, "equities": 1, "stocks": 1, "fii": 2, "fiis": 2, "fpi": 2, "fpis": 2,
               "dii": 1, "india vix": 2, "f&o": 1, "derivatives": 1, "index": 1, "midcap": 1, "smallcap": 1}
MACRO_WORDS = {"rbi": 3, "repo rate": 3, "monetary policy": 3, "mpc": 2, "inflation": 2, "cpi": 2, "wpi": 1, "gdp": 2,
               "fiscal": 1, "budget": 2, "rupee": 2, "crude": 2, "brent": 2, "oil prices": 2, "fed": 2, "fomc": 3,
               "federal reserve": 3, "powell": 2, "treasury": 1, "bond yields": 2, "us yields": 2, "dollar index": 1,
               "tariff": 2, "tariffs": 2, "war": 2, "geopolitical": 2, "sebi": 2, "wall street": 1, "dow": 1, "nasdaq": 1,
               "recession": 2, "gst": 1, "iip": 1, "trade deficit": 1, "china": 1}
BANK_WORDS = {"bank": 1, "banks": 2, "banking": 2, "hdfc bank": 3, "icici bank": 3, "sbi": 3, "state bank": 3, "kotak": 2,
              "axis bank": 3, "indusind": 3, "bank of baroda": 2, "pnb": 2, "psu bank": 3, "npa": 2, "credit growth": 2,
              "deposit": 1, "lending": 1, "nbfc": 1}
NIFTY_HEAVY = {"reliance": 2, "infosys": 2, "tcs": 2, "hdfc bank": 2, "icici bank": 2, "itc": 1, "larsen": 1, "l&t": 1,
               "bharti airtel": 1, "airtel": 1, "hul": 1, "kotak": 1, "sbi": 1, "adani": 1, "tata motors": 1, "m&m": 1}

POS = {"rally": 1.0, "rallies": 1.0, "rallied": 1.0, "surge": 1.0, "surges": 1.0, "surged": 1.0, "soar": 1.0, "soars": 1.0,
       "jump": 0.8, "jumps": 0.8, "jumped": 0.8, "gain": 0.6, "gains": 0.6, "gained": 0.6, "climb": 0.6, "climbs": 0.6,
       "rise": 0.5, "rises": 0.5, "rose": 0.5, "rising": 0.4, "up": 0.3, "higher": 0.5, "high": 0.2, "record": 0.1,
       "rebound": 0.8, "rebounds": 0.8, "recover": 0.7, "recovers": 0.7, "recovery": 0.6, "bullish": 1.0, "bulls": 0.7,
       "upgrade": 0.8, "upgrades": 0.8, "beats": 0.8, "beat": 0.6, "strong": 0.6, "robust": 0.6, "optimism": 0.8,
       "boost": 0.7, "boosts": 0.7, "inflows": 0.8, "buying": 0.6, "outperform": 0.6, "positive": 0.5, "green": 0.4,
       "eases": 0.5, "easing": 0.5, "cools": 0.5, "relief": 0.7, "ceasefire": 0.9, "deal": 0.4,
       "stimulus": 0.7, "extends gains": 1.0, "all-time high": 1.0, "record high": 1.0, "rate cut": 0.9}
NEG = {"fall": 0.6, "falls": 0.6, "fell": 0.6, "falling": 0.5, "drop": 0.6, "drops": 0.6, "dropped": 0.6, "decline": 0.6,
       "declines": 0.6, "slide": 0.7, "slides": 0.7, "slip": 0.5, "slips": 0.5, "tumble": 0.9, "tumbles": 0.9, "plunge": 1.0,
       "plunges": 1.0, "crash": 1.0, "crashes": 1.0, "slump": 0.9, "slumps": 0.9, "sink": 0.8, "sinks": 0.8, "down": 0.3,
       "lower": 0.5, "low": 0.2, "selloff": 1.0, "sell-off": 1.0, "selling": 0.6, "bearish": 1.0, "bears": 0.7,
       "downgrade": 0.8, "downgrades": 0.8, "misses": 0.8, "miss": 0.6, "weak": 0.6, "weakness": 0.6, "fears": 0.7,
       "fear": 0.7, "worries": 0.6, "concerns": 0.5, "concern": 0.5, "outflows": 0.8, "losses": 0.5, "loss": 0.4,
       "red": 0.4, "war": 0.9, "attack": 0.9, "tension": 0.7, "tensions": 0.7, "recession": 0.9, "default": 0.8,
       "fraud": 0.8, "probe": 0.5, "ban": 0.5, "hike": 0.6, "hikes": 0.6, "volatile": 0.4, "turmoil": 0.9, "panic": 1.0,
       "lower circuit": 1.0, "rate hike": 0.9, "profit booking": 0.6, "profit-taking": 0.6, "extends losses": 1.0}
# idioms and policy moves, matched first (regex → tone); the words they use are then not counted again
PHRASES = [
    (r"\bsnaps?\b[\w\s-]{0,20}\blosing (streak|run)\b", 0.8), (r"\bsnaps?\b[\w\s-]{0,20}\bwinning (streak|run)\b", -0.8),
    (r"\blosing (streak|run)\b", -0.6), (r"\bwinning (streak|run)\b", 0.6),
    (r"\b(cuts?|lowers?|slash(es)?) (the )?(repo |policy |key |benchmark |interest )*rates?\b", 0.9),
    (r"\brate cuts?\b", 0.8), (r"\b(hikes?|raises?) (the )?(repo |policy |key |benchmark |interest )*rates?\b", -0.9),
    (r"\brate hikes?\b", -0.8), (r"\bhigher for longer\b", -0.8), (r"\b(record|all[- ]time|\d+-(year|month)) lows?\b", -0.9),
    (r"\b(record|all[- ]time|\d+-(year|month)) highs?\b", 0.9), (r"\bprofit[- ](booking|taking)\b", -0.6),
    (r"\bextends? (gains|rally)\b", 1.0), (r"\bextends? (losses|fall|decline)\b", -1.0),
    (r"\blower circuit\b", -1.0), (r"\bupper circuit\b", 0.8), (r"\bsell[- ]off\b", -1.0),
    (r"\b(fii|fpi)s? (turn )?(net )?(sellers?|selling|outflows?)\b", -0.8),
    (r"\b(fii|fpi)s? (turn )?(net )?(buyers?|buying|inflows?)\b", 0.8),
]
# words that only say which way something moved: for "inverse" subjects the market reads them backwards
MOVEMENT = {"rise", "rises", "rose", "rising", "up", "higher", "high", "jump", "jumps", "jumped", "surge", "surges", "surged",
            "soar", "soars", "climb", "climbs", "gain", "gains", "gained", "spike", "spikes", "fall", "falls", "fell", "falling",
            "drop", "drops", "dropped", "decline", "declines", "slide", "slides", "slip", "slips", "tumble", "tumbles",
            "plunge", "plunges", "sink", "sinks", "down", "lower", "low"}
# for these subjects a rise is bad news for Indian equities (and a fall is good)
INVERSE_SUBJECTS = ("inflation", "crude", "oil", "brent", "yields", "yield", "dollar", "vix", "cpi", "wpi", "deficit",
                    "fii selling", "fpi selling", "outflow", "unemployment", "jobless", "bond yields")
NEGATORS = {"not", "no", "never", "without", "despite", "fails", "halts", "snaps", "ends"}
HIGH_IMPACT = ("repo rate", "monetary policy", "rbi policy", "mpc decision", "fomc", "fed decision", "fed raises", "fed cuts",
               "rate decision", "union budget", r"budget 20\d\d", "gdp data", "cpi data", "inflation data", "election result",
               "war", "attack", "missile", "emergency", "circuit breaker", "sebi bans", "ceasefire", "sanctions")
HIGH_RE = re.compile(r"\b(" + "|".join(k if "\\" in k else re.escape(k) for k in HIGH_IMPACT) + r")\b")
# A recap of the market's own move ("Stock market crash: Sensex tumbles 700 points") is not news to the desk: the
# move is already on its tape, and the explainer usually arrives after it. On 29 Sep 2026 five "market crash"
# recaps in 70 minutes kept the desk out of the morning's sell-off; so a recap never counts as high impact (it
# can't make the desk stand aside) and weighs less in the tone. The first report of a shock ("RBI cuts repo rate",
# "missile attack") is not a recap and still does both.
RECAP_SUBJECT = re.compile(r"\b(sensex|nifty\w*|bank ?nifty|stock ?markets?|share ?markets?|equity markets?|markets|dalal street|"
                           r"d-street|indices|benchmark|stocks|investors)\b", re.I)
RECAP_MOVE = re.compile(r"\b(crash\w*|tumbl\w*|plung\w*|slump\w*|sink\w*|sank|slid\w*|slip\w*|tank\w*|fall\w*|fell|drop\w*|"
                        r"declin\w*|los[et]\w*|bleed\w*|rall\w*|surg\w*|soar\w*|jump\w*|zoom\w*|rebound\w*|recover\w*|"
                        r"climb\w*|gain\w*|rise|rises|rose|selloff|sell-off|bloodbath|carnage|wiped? (?:out|off)|bear grip|"
                        r"bull run|in the red|in the green|\d[\d,]* (?:points|pts)|lakh crore)\b", re.I)


def is_recap(text: str) -> bool:
    return bool(RECAP_SUBJECT.search(text) and RECAP_MOVE.search(text))


# a preview is not the event: "ahead of RBI policy" shouldn't make the desk stand aside
PREVIEW = re.compile(r"\b(ahead of|preview|what to expect|expected to|likely to|may |could |live updates|week ahead|"
                     r"things to know|to watch|before the|explained)\b", re.I)
MEDIUM_IMPACT = ("rbi", "fed", "inflation", "gdp", "crude", "rupee", "fii", "fpi", "tariff", "sebi", "results", "earnings",
                 "guidance", "downgrade", "upgrade")

_TOKEN = re.compile(r"[a-z0-9&\-']+")


def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html.unescape(text or ""))
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _hits(text_l: str, vocab: dict) -> float:
    total = 0.0
    for k, w in vocab.items():
        if " " in k or "&" in k or "-" in k:
            if k in text_l:
                total += w
        elif re.search(rf"\b{re.escape(k)}\b", text_l):
            total += w
    return total


def sentiment(text: str) -> float:
    """Headline tone for Indian equities in [-1, 1]."""
    t = text.lower()
    score, mass = 0.0, 0.0
    for pat, v in PHRASES:
        if re.search(pat, t):
            score += v
            mass += abs(v)
            t = re.sub(pat, " ", t)
    for ph in [k for k in list(POS) + list(NEG) if " " in k or "-" in k]:
        if ph in t:
            v = POS.get(ph, 0.0) - NEG.get(ph, 0.0)
            score += v
            mass += abs(v)
            t = t.replace(ph, " ")
    toks = _tokens(t)
    for i, tok in enumerate(toks):
        v = POS.get(tok, 0.0) - NEG.get(tok, 0.0)
        if not v:
            continue
        if any(x in NEGATORS for x in toks[max(0, i - 2):i]):
            v = -v
        # "crude rises", "yields jump", "inflation falls": the subject decides how the market takes the move
        if tok in MOVEMENT and any(sub in " ".join(toks[max(0, i - 4):i]) for sub in INVERSE_SUBJECTS):
            v = -v
        score += v
        mass += abs(v)
    if mass == 0:
        return 0.0
    return float(max(-1.0, min(1.0, score / max(mass, 1.0))))


def impact(text: str) -> str:
    t = text.lower()
    # whole words only: "war" must not fire on "toward", "forward", "award", "software"
    if HIGH_RE.search(t) and not is_recap(t):
        return "high"
    if any(re.search(rf"\b{re.escape(k)}\b", t) for k in MEDIUM_IMPACT):
        return "medium"
    return "low"


@dataclass
class NewsItem:
    ts: pd.Timestamp
    source: str
    title: str
    link: str = ""
    summary: str = ""
    sources: list = field(default_factory=list)
    id: str = ""
    sentiment: float = 0.0
    impact: str = "low"
    recap: bool = False                             # the market's own move, retold (not new information)
    about: dict = field(default_factory=dict)       # {"NIFTY": relevance, "BANKNIFTY": relevance, "macro": x}

    def to_record(self) -> dict:
        return {"id": self.id, "ts": str(self.ts), "source": self.source, "sources": self.sources, "title": self.title,
                "link": self.link, "summary": self.summary, "sentiment": round(self.sentiment, 3), "impact": self.impact,
                "about": self.about}


def classify(item: NewsItem) -> NewsItem:
    text = f"{item.title}. {item.summary[:240]}"
    tl = text.lower()
    idx, mac, bank, heavy = _hits(tl, INDEX_WORDS), _hits(tl, MACRO_WORDS), _hits(tl, BANK_WORDS), _hits(tl, NIFTY_HEAVY)
    item.about = {"NIFTY": round(idx + 0.6 * mac + 0.5 * heavy, 2), "BANKNIFTY": round(idx + 0.6 * mac + bank, 2), "macro": round(mac, 2)}
    item.sentiment = sentiment(item.title) if item.title else 0.0
    if abs(item.sentiment) < 0.05 and item.summary:
        item.sentiment = 0.5 * sentiment(item.summary[:300])
    item.recap = is_recap(item.title)
    item.impact = impact(text) if not item.recap else min(impact(text), "medium", key=["low", "medium", "high"].index)
    return item


def parse_feed(xml_text: str, source: str, now: pd.Timestamp | None = None) -> list[NewsItem]:
    """RSS 2.0 or Atom → items (IST timestamps; undated items get the fetch time)."""
    now = now or pd.Timestamp.now(tz=IST)
    try:
        root = ET.fromstring(xml_text.encode("utf-8") if isinstance(xml_text, str) else xml_text)
    except ET.ParseError:
        return []
    out = []
    ns_atom = "{http://www.w3.org/2005/Atom}"
    entries = root.findall(".//item") or root.findall(f".//{ns_atom}entry")
    for e in entries[:60]:
        def txt(tag):
            el = e.find(tag)
            if el is None:
                el = e.find(ns_atom + tag)
            return (el.text or "") if el is not None else ""
        title = _clean(txt("title"))
        if not title:
            continue
        link = txt("link").strip()
        if not link:
            el = e.find(ns_atom + "link")
            link = el.get("href", "") if el is not None else ""
        raw_ts = txt("pubDate") or txt("published") or txt("updated")
        ts = now
        if raw_ts:
            try:
                d = email.utils.parsedate_to_datetime(raw_ts)
                ts = pd.Timestamp(d)
            except (TypeError, ValueError):
                try:
                    ts = pd.Timestamp(raw_ts)
                except ValueError:
                    ts = now
            ts = ts.tz_localize(IST) if ts.tzinfo is None else ts.tz_convert(IST)
        src = source
        if "news.google" in source.lower() or source.startswith("Google"):
            m = re.match(r"^(.*) - ([^-]{2,60})$", title)          # Google News appends " - Outlet"
            if m:
                title, src = m.group(1).strip(), f"{m.group(2).strip()} (via Google News)"
        summary = _clean(txt("description") or txt("summary"))[:400]
        if ts > now + pd.Timedelta(minutes=5):
            continue                                   # not published yet on this clock (replays must not see the future)
        item = NewsItem(min(ts, now), src, title, link, summary, [src])
        item.id = hashlib.sha1(re.sub(r"\W+", "", title.lower()).encode()).hexdigest()[:16]
        out.append(classify(item))
    return out


def _similar(a: str, b: str) -> bool:
    ta, tb = set(_tokens(a)) - {"the", "a", "to", "of", "in", "on", "for", "and", "as", "at", "is"}, set(_tokens(b))
    tb -= {"the", "a", "to", "of", "in", "on", "for", "and", "as", "at", "is"}
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= 0.55


class NewsDesk:
    """Holds the day's headlines, fetches new ones, and summarises them per underlying."""

    def __init__(self, cfg, fetch=None, sources: list[dict] | None = None):
        nc = cfg.get("intraday.news", {}) or {}
        self.enabled = nc.get("enabled", True)
        self.sources = sources if sources is not None else (nc.get("sources") or DEFAULT_SOURCES)
        self.refresh_min = nc.get("refresh_min", 4)
        self.half_life = nc.get("half_life_min", 45)
        self.window = nc.get("window_min", 120)
        self.min_relevance = nc.get("min_relevance", 2.0)
        self.breaking_min = nc.get("breaking_stand_aside_min", 15)
        self.fetch = fetch or self._http
        self.items: dict[str, NewsItem] = {}
        self.last_fetch: pd.Timestamp | None = None
        self.health: dict[str, str] = {}

    def _http(self, url: str) -> str:
        import requests
        r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0 (QuantDesk news reader)",
                                                 "Accept": "application/rss+xml, application/xml, text/xml, */*"})
        r.raise_for_status()
        return r.text

    def refresh(self, now: pd.Timestamp, force: bool = False) -> list[NewsItem]:
        """Fetch every source (at most every `refresh_min`); returns the stories that are new."""
        if not self.enabled or (not force and self.last_fetch is not None
                                and now - self.last_fetch < pd.Timedelta(minutes=self.refresh_min)):
            return []
        self.last_fetch = now
        new: list[NewsItem] = []
        from concurrent.futures import ThreadPoolExecutor

        def one(src):
            try:
                return src, parse_feed(self.fetch(src["url"]), src["name"], now), None
            except Exception as exc:                                    # one dead feed must not stop the rest
                return src, [], exc
        with ThreadPoolExecutor(max_workers=min(8, len(self.sources) or 1)) as pool:   # all feeds at once: ~one timeout, not nine
            results = list(pool.map(one, self.sources))
        for src, items, exc in results:
            self.health[src["name"]] = f"fail {str(exc)[:80]}" if exc else f"ok {len(items)}"
            for it in items:
                if now - it.ts > pd.Timedelta(hours=24):
                    continue
                dup = self.items.get(it.id) or next((x for x in self.items.values() if _similar(x.title, it.title)), None)
                if dup:
                    if it.source not in dup.sources:
                        dup.sources.append(it.source)
                    continue
                self.items[it.id] = it
                new.append(it)
        return sorted(new, key=lambda x: x.ts)

    def add(self, items: list[NewsItem]) -> None:
        for it in items:
            self.items.setdefault(it.id, it)

    def relevant(self, symbol: str, now: pd.Timestamp, minutes: int | None = None) -> list[NewsItem]:
        lo = now - pd.Timedelta(minutes=minutes or self.window)
        return sorted([x for x in self.items.values() if lo <= x.ts <= now and x.about.get(symbol, 0) >= self.min_relevance],
                      key=lambda x: x.ts, reverse=True)

    def state(self, symbol: str, now: pd.Timestamp) -> dict | None:
        """Recency-weighted tone and the stand-aside flag, from stories published up to `now`."""
        rel = self.relevant(symbol, now)
        if not rel:
            return None
        num = den = 0.0
        for x in rel:
            age = (now - x.ts).total_seconds() / 60
            w = 0.5 ** (age / self.half_life) * min(x.about.get(symbol, 0) / 4, 1.5) * (1 + 0.5 * (len(x.sources) - 1)) \
                * {"high": 2.0, "medium": 1.3, "low": 1.0}[x.impact] * (0.4 if x.recap else 1.0)
            num += w * x.sentiment
            den += w
        tone = num / den if den else 0.0
        conf = len(rel) / (len(rel) + 3)
        breaking = next((x for x in rel if x.impact == "high" and not x.recap and not PREVIEW.search(x.title)
                         and (now - x.ts) <= pd.Timedelta(minutes=self.breaking_min)), None)
        return {"tone": float(tone), "confidence": float(conf), "n": len(rel), "latest": rel[0].title,
                "latest_age_min": float((now - rel[0].ts).total_seconds() / 60),
                "breaking": ({"title": breaking.title, "age_min": float((now - breaking.ts).total_seconds() / 60),
                              "source": breaking.source} if breaking else None)}
