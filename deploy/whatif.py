#!/usr/bin/env python3
"""What if the desk had traded a recorded day differently? Replays one real session under variants and
prints what each one would have done: the trades, their P&L, and what kept it out the rest of the time.

    python deploy/whatif.py --date 2026-09-29 --until 12:20 --variants as_run,no_news,no_rsi_veto
    PYTHONPATH=<an older checkout> python deploy/whatif.py ...     # replay with the code as it was on the day

Inputs, the same ones the live desk had: the day's recorded 1m bars and the headlines it read (both from
the journal: `deploy/journal.sh restore`), plus the prior sessions from Yahoo (1m for the last week, 5m
before that) for prior-day levels, the vol forecast and the direction model. Headlines are replayed from
the minute the desk fetched them, never earlier. Options are priced off the model chain (India VIX +
skew): the real chain snapshots aren't kept, so fills are close to, not identical with, the live ones.

Runs on a GitHub runner (`whatif.yml`), which can reach Yahoo; the journal is only read."""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
import sqlite3
import sys
import tempfile
from pathlib import Path

import pandas as pd

from quantdesk.config import DEFAULT_CONFIG, Config
from quantdesk.intraday import analyst as analyst_mod
from quantdesk.intraday.engine import IntradayEngine, run_replay
from quantdesk.intraday.feeds import ReplayFeed, normalise_bars
from quantdesk.intraday.news import NewsDesk, NewsItem, classify
from quantdesk.intraday.recorder import SessionRecorder
from quantdesk.intraday.sim import IntradayBroker
from quantdesk.journal.journal import Journal

IST = "Asia/Kolkata"

# name → (config overrides, options). Every variant runs on whatever code is on PYTHONPATH.
VARIANTS = {
    "as_run": ({}, {}),
    "no_news": ({}, {"news": False}),
    "no_rsi_veto": ({}, {"rsi_veto": False}),
    "no_news_no_rsi": ({}, {"news": False, "rsi_veto": False}),
    "conviction_40": ({"intraday": {"analyst": {"min_conviction": 0.40}}}, {}),
    "no_stop_floor": ({"intraday": {"risk": {"min_stop_atr5": 0.0}}}, {}),
    "three_trades": ({"intraday": {"risk": {"max_trades_per_day": 3}}}, {}),
    "ev_gate_off": ({"intraday": {"quant": {"min_ev_inr": -1e9, "min_ev_r": -1e9}}}, {}),
}


def yahoo_history(cfg, symbols: list[str], day: dt.date) -> dict[str, pd.DataFrame]:
    """Prior sessions the live desk would have had: 1m for the last 7 days, 5m for the 59 before."""
    import yfinance as yf
    out = {}
    for s in symbols:
        tk = yf.Ticker(cfg.instrument_spec(s).get("yahoo", s))
        got = {}
        for period, interval in (("59d", "5m"), ("7d", "1m")):
            try:
                raw = tk.history(period=period, interval=interval, auto_adjust=False, prepost=False)
            except Exception as exc:                          # a missing series degrades the replay, it doesn't stop it
                print(f"yahoo {s} {interval}: {exc}", file=sys.stderr)
                continue
            if raw is not None and not raw.empty:
                got[interval] = normalise_bars(raw)
        if not got:
            continue
        five, one = got.get("5m", pd.DataFrame()), got.get("1m", pd.DataFrame())
        if len(one) and len(five):                            # 1m where we have it, 5m before that
            five = five[five.index < one.index.min().normalize()]
        df = pd.concat([x for x in (five, one) if len(x)]).sort_index()
        out[s] = df[df.index.date < day]
    return out


class ReplayNews(NewsDesk):
    """The day's headlines as the desk saw them: each one only from the minute it was fetched."""

    def __init__(self, cfg, journal_db: Path, day: dt.date):
        super().__init__(cfg, fetch=lambda url: "", sources=[])
        con = sqlite3.connect(journal_db)
        rows = pd.read_sql("SELECT * FROM news WHERE ts >= ? AND ts < ?", con, params=(str(day), str(day + dt.timedelta(days=1))))
        self.pending = []
        for r in rows.itertuples():
            it = NewsItem(pd.Timestamp(r.ts).tz_convert(IST), r.source, r.title, r.link or "", r.summary or "",
                          json.loads(r.sources or "[]"), r.id, r.sentiment, r.impact, json.loads(r.about or "{}"))
            self.pending.append((pd.Timestamp(r.seen_at).tz_convert(IST), classify(it)))   # this code's reading of it

    def refresh(self, now, force=False):
        new = [it for seen, it in self.pending if seen <= now and it.id not in self.items]
        self.add(new)
        return new


def run_variant(name: str, runtime: Path, day: dt.date, until: str, history: dict, out: Path) -> dict:
    overrides, opt = VARIANTS[name]
    cfg = Config.load(DEFAULT_CONFIG, overrides=overrides)
    syms = cfg.get("intraday.underlyings") + [cfg.get("universe.volatility_index")]
    today = SessionRecorder(runtime / "intraday" / "data").load_bars(syms, upto=day)
    cut = pd.Timestamp(f"{day} {until}", tz=IST)
    bars = {}
    for s in syms:
        t = today.get(s)
        t = t[(t.index.date == day) & (t.index <= cut)] if t is not None else None
        parts = [x for x in (history.get(s), t) if x is not None and len(x)]
        if parts:
            bars[s] = pd.concat(parts).sort_index()
    j = Journal(out / f"{name}.db")
    br = IntradayBroker(cfg, starting_cash=cfg.get("intraday.capital"), state_path=out / f"{name}.json",
                        adverse_ticks=cfg.get("intraday.adverse_ticks", 1))
    news = ReplayNews(cfg, runtime / "intraday" / "journal.db", day) if opt.get("news", True) else None
    orig = analyst_mod.Analyst.assess
    if opt.get("rsi_veto", True) is False:
        def assess(self, *a, **k):
            v = orig(self, *a, **k)
            v.vetoes = [x for x in v.vetoes if "too stretched" not in x]
            return v
        analyst_mod.Analyst.assess = assess
    try:
        eng = IntradayEngine(cfg, ReplayFeed(bars, day), "model", j, br, None, lambda *a, **k: None, None,
                             out / f"reviews_{name}", news=news)
        run_replay(eng)
    finally:
        analyst_mod.Analyst.assess = orig
    j.commit()
    t = j.df("SELECT symbol, strategy, direction, opened_at, closed_at, pnl, r_multiple, exit_reason, meta FROM trades ORDER BY opened_at")
    th = j.df("SELECT symbol, action FROM thoughts")
    why = collections.Counter(re.sub(r"\d[\d,.:]*", "#", str(a)).split(" “")[0][:70] for a in th["action"]
                              if not str(a).startswith(("ENTER", "EXIT")))
    return {"variant": name, "trades": int(len(t)), "net": round(float(t["pnl"].sum()) if len(t) else 0.0, 2),
            "list": [f"{r.symbol} {r.strategy} {'long' if r.direction > 0 else 'short'} {str(r.opened_at)[11:16]}→{str(r.closed_at)[11:16]} "
                     f"{json.loads(r.meta).get('structure')} ₹{r.pnl:+,.0f} ({r.r_multiple:+.2f}R, {r.exit_reason})" for r in t.itertuples()],
            "kept_out_by": dict(why.most_common(6))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--date", required=True)
    ap.add_argument("--until", default="15:30", help="replay up to this IST time (e.g. the hand-over)")
    ap.add_argument("--variants", default="as_run,no_news,no_rsi_veto,conviction_40")
    ap.add_argument("--runtime", default="runtime")
    ap.add_argument("--label", default="", help="tag for the output (e.g. the code version)")
    ap.add_argument("--json", help="append results as JSON lines here")
    a = ap.parse_args()
    day = dt.date.fromisoformat(a.date)
    cfg = Config.load(DEFAULT_CONFIG)
    syms = cfg.get("intraday.underlyings") + [cfg.get("universe.volatility_index")]
    history = yahoo_history(cfg, syms, day)
    print(f"history: " + ", ".join(f"{s} {len(v)} bars over {len(set(v.index.date))} sessions" for s, v in history.items()), flush=True)
    out = Path(tempfile.mkdtemp(prefix="whatif-"))
    for name in a.variants.split(","):
        r = run_variant(name.strip(), Path(a.runtime), day, a.until, history, out)
        r["code"] = a.label
        print(json.dumps(r, ensure_ascii=False), flush=True)
        if a.json:
            with open(a.json, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
