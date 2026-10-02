"""The intraday engine — one loop for live paper trading and for replays.

Every completed minute:
  1. pull new 1m bars (and ticks, if the feed has them) and record them;
  2. refresh the option chain every few minutes; recalibrate option marks from it;
  3. compute the session state and let the analyst form a view (evidence, bias, day type,
     vol view, vetoes, narrative);
  4. manage open positions: invalidation level, premium stop/target, underlying target,
     breakeven trail, time stop, 15:15 square-off;
  5. scan the playbook; size through intraday risk; execute on the sim broker at bid/ask;
  6. journal the thought (every few minutes, on a bias change, and on every trade event).
At the close: square off, write the session review.
"""
from __future__ import annotations

import datetime as dt
import logging
import math
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.calendar import TradingCalendar
from ..core.types import OPTIONS, Instrument, Order, Trade, TradeLeg, new_trade_id
from ..journal.journal import Journal, trade_from_dict, trade_to_dict
from .analyst import Analyst, MarketView
from .chains import ChainSource, IntradayPricer, ModelOptionChain, chain_analytics, fill_iv
from .features import session_state
from .feeds import IST, IntradayFeed, ReplayFeed, session_bounds
from .orderflow import FootprintBuilder
from .playbook import Playbook, TradePlan
from .quant import DirectionModel, EVEngine, VolForecaster, features_5m, load_research, session_sigma, to_5m
from .risk import IntradayRisk
from .sim import IntradayBroker, QuoteMarker

log = logging.getLogger(__name__)


class IntradayEngine:
    def __init__(self, cfg, feed: IntradayFeed, chains: ChainSource | str | None, journal: Journal,
                 broker: IntradayBroker, recorder=None, say=print, underlyings: list[str] | None = None,
                 review_dir: Path | None = None, news=None, brain=None):
        ic = cfg.get("intraday", {}) or {}
        self.cfg, self.feed, self.journal, self.broker, self.recorder = cfg, feed, journal, broker, recorder
        self.say = say or (lambda *_: None)
        self.underlyings = underlyings or ic.get("underlyings", ["NIFTY", "BANKNIFTY"])
        self.vix = cfg.get("universe.volatility_index", "INDIAVIX")
        self.cal = TradingCalendar(cfg.holidays())
        self.pricer = IntradayPricer(cfg.get("backtest.risk_free", 0.065), cfg.get("backtest.dividend_yield", 0.012))
        self.model_chain = ModelOptionChain(cfg, self.cal, self.model_state, self.pricer)
        self.chains = self.model_chain if chains in (None, "model") else chains
        self.analyst, self.playbook, self.risk = Analyst(cfg), Playbook(cfg, self.pricer), IntradayRisk(cfg)
        self.marker = QuoteMarker(self.pricer)
        self.refresh_min = getattr(self.chains, "refresh_min", None) or ic.get("chain_refresh_min", 3)
        self.max_entry_slip = float(ic.get("max_entry_slip", 0.15))
        self.live_fails = 0
        self.think_every = ic.get("thought_every_min", 5)
        self.stale_min = ic.get("chain_stale_min", 12)
        self.history_days = ic.get("history_days", 6)
        self.expiry_min_days = ic.get("expiry_min_days", 1)
        self.review_dir = review_dir
        self.news = news                                  # NewsDesk (live headlines) or None
        self.brain = brain                                # Brain (global markets ↔ news ↔ India ↔ decision) or None
        self.brain_state: dict = {}
        self.last_action: dict[str, str] = {}
        qc = cfg.get("intraday.quant", {}) or {}
        self.qc = qc
        self.quant_on = bool(qc.get("enabled", True))
        self.volf = VolForecaster()
        self.ev = EVEngine(cfg, self.pricer, broker.costs, n_paths=int(qc.get("n_paths", 2000)))
        self.models: dict[str, DirectionModel] = {}
        self.hist5: dict[str, pd.DataFrame] = {}
        self.qstate: dict[str, dict] = {}
        self._pcache: dict[str, tuple] = {}
        self._qc_cache: dict[str, dict] = {}
        self._qerrors: set = set()
        self.research = load_research(Path(cfg.runtime_dir) / "research" / "edges.json") if self.quant_on else {}
        self.bars: dict[str, pd.DataFrame] = {}
        self.last_ts: dict[str, pd.Timestamp] = {}
        self.chain_df: dict[str, pd.DataFrame] = {}
        self.chain_an: dict[str, dict] = {}
        self.chain_at: dict[str, pd.Timestamp] = {}
        self.chain_fail: dict[str, int] = {}
        self.chain_tried: dict[str, pd.Timestamp] = {}
        self.expiry: dict[str, dt.date] = {}
        self.open_trades: list[Trade] = []
        self.closed: list[Trade] = []
        self.views: dict[str, MarketView] = {}
        self.last_thought: dict[str, pd.Timestamp] = {}
        self.last_bias: dict[str, str] = {}
        self.flow = {u: FootprintBuilder(float(cfg.instrument_spec(u).get("tick", 0.05))) for u in self.underlyings}
        self.feature_cache: dict[str, dict] = {u: {} for u in self.underlyings}
        self.day: dt.date | None = None
        self.day_start_equity = broker.cash()
        self.paused = bool(journal.get_state("intraday_paused", False))
        self.events_today: list[str] = []
        from .events import EventBook, from_config
        self.eventbook = EventBook(from_config(cfg))
        self.warehouse_dir = Path(cfg.runtime_dir) / "warehouse"
        self.gift_source = None                           # callable → parse_gift frame (live only); None = skip
        self.gift: dict | None = None
        self._gift_at = None
        self._gift_fails = 0

    # ---- helpers ----------------------------------------------------------------------------------
    def lot(self, u: str) -> int:
        return int(self.cfg.instrument_spec(u).get("lot_size", 1))

    def model_state(self, u: str, ts) -> tuple[float, float]:
        """Spot and ATM IV for the model chain: last 1m close, India VIX × iv_beta."""
        df = self.bars[u]
        S = float(df.loc[:ts]["close"].iloc[-1])
        v = self.bars.get(self.vix)
        vix = float(v.loc[:ts]["close"].iloc[-1]) if v is not None and len(v.loc[:ts]) else 14.0
        return S, vix / 100 * float(self.cfg.instrument_spec(u).get("iv_beta", 1.0))

    def pick_expiry(self, u: str, today: dt.date) -> dt.date:
        try:
            exps = self.chains.expiries(u, today) if isinstance(self.chains, ModelOptionChain) else self.chains.expiries(u)
        except Exception as exc:
            spec = self.cfg.instrument_spec(u)
            exps = self.cal.expiries(today, 70, int(spec.get("expiry_weekday", 1)), bool(spec.get("weekly_expiry", True)))
            self.journal.event(pd.Timestamp.now(tz=IST), "WARN", "chain", f"{u} expiries from calendar ({exc})")
        exps = [e for e in exps if (e - today).days >= self.expiry_min_days]
        return exps[0]

    def equity(self, now) -> float:
        val = 0.0
        for t in self.open_trades:
            S = self.spot(t.symbol)
            val += sum(l.qty * self.marker.mid(l.instrument, S, now) for l in t.legs)
        return self.broker.cash() + val

    def spot(self, u: str) -> float:
        return float(self.bars[u]["close"].iloc[-1])

    # ---- session lifecycle ------------------------------------------------------------------------------
    def start_session(self, day: dt.date) -> None:
        self.day = day
        for sym in self.underlyings + [self.vix]:
            h = self.feed.history(sym, self.history_days)
            self.bars[sym] = h.tail(375 * self.history_days) if h is not None else \
                pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            self.last_ts[sym] = h.index[-1] if h is not None and len(h) else None
        self.expiry = {u: self.pick_expiry(u, day) for u in self.underlyings}
        self._load_heavyweights()
        self.events_today = self.eventbook.describe(day, self.cal.prev_trading_day(day))
        saved = self._restore(day)
        if saved.get("day_start_equity"):
            self.day_start_equity = float(saved["day_start_equity"])
            self.risk.reset(day, self.day_start_equity)
            r = saved.get("risk") or {}
            self.risk.trades_today = int(r.get("trades_today", len(self.closed) + len(self.open_trades)))
            self.risk.consec_losses, self.risk.halted = int(r.get("consec_losses", 0)), bool(r.get("halted", False))
            self.risk.cool_until = pd.Timestamp(r["cool_until"]) if r.get("cool_until") else None
        else:
            self.day_start_equity = self.broker.cash() + sum(t.entry_cost for t in self.open_trades)
            self.risk.reset(day, self.day_start_equity)
            self.risk.trades_today = len([t for t in self.closed if t.opened_at.date() == day]) + len(self.open_trades)
        if self.quant_on:
            self._train_models(day)
        exp = ", ".join(f"{u} {e:%d-%b}" for u, e in self.expiry.items())
        self.say(f"── session {day} · capital ₹{self.day_start_equity:,.0f} · expiries {exp} · chain {self.chains.name} "
                 f"· feed {self.feed.name}{' · events: ' + ', '.join(self.events_today) if self.events_today else ''}")
        self.journal.event(session_bounds(day)[0], "INFO", "session", f"session start; expiries {exp}; chain {self.chains.name}")

    def step(self) -> bool:
        now = self.feed.now()
        self._commands(now)
        got = False
        for sym in self.underlyings + [self.vix]:
            new = self.feed.poll(sym, self.last_ts.get(sym))
            if new is None or new.empty:
                continue
            self.bars[sym] = pd.concat([self.bars.get(sym), new]) if sym in self.bars and len(self.bars[sym]) else new
            self.last_ts[sym] = new.index[-1]
            if self.recorder:
                self.recorder.record_bars(sym, new)
            got = got or sym in self.underlyings
        if self.feed.has_ticks:
            for u in self.underlyings:
                for t in self.feed.trades(u):
                    self.flow[u].add(t)
        if not got:
            return False
        self._refresh_news(now)
        if self.brain is not None and self.brain.gfeed is not None:
            self._guarded("global", now, "global refresh", self.brain.gfeed.refresh, now)
        for u in self.underlyings:
            if u not in self.bars or self.bars[u].empty or self.bars[u].index[-1].date() != self.day:
                continue
            self._refresh_chain(u, now)
            s = session_state(self.bars[u], now, cache=self.feature_cache[u])
            if s is None:
                continue
            q = self._guarded(u, now, "quant state", self._quant_state, u, now) if self.quant_on else None
            b = self._guarded(u, now, "brain", self._think_globally, u, now) if self.brain is not None else None
            blk = self.eventbook.blocking(now)                   # only an announcement inside the session blocks
            view = self.analyst.assess(u, s, self.chain_an.get(u), self._vix_state(), self.expiry[u] == self.day,
                                       blk.label() if blk else None, self._flow_state(u, now),
                                       self.news.state(u, now) if self.news is not None else None, q,
                                       {"evidence": b.evidence, "narrative": b.narrative} if b is not None else None)
            self.views[u] = view
            exits = self._manage(u, view, now)
            action = self._maybe_enter(u, view, s, now)
            self._think(u, view, now, "; ".join(exits + [action]) if exits else action, force=bool(exits))
        if now.time() >= self.risk.square_off:
            for t in list(self.open_trades):
                self._close(t, now, "square_off", f"intraday square-off at {self.risk.square_off:%H:%M}")
        self._snapshot(now)
        self._persist()
        self._heartbeat(now)
        self.journal.commit()
        return True

    def end_session(self, reason: str = "square_off", note: str = "end of session") -> str:
        now = self.feed.now()
        for u in {t.symbol for t in self.open_trades} - set(self.chain_df):
            self._refresh_chain(u, now)                 # exits priced off a calibrated chain, never a default IV
        for t in list(self.open_trades):
            self._close(t, now, reason, note)
        review = self.session_review()
        self.journal.event(now, "INFO", "session_review", review[:2000])
        self._persist(ended=True)
        self.journal.commit()
        if self.review_dir:
            self.review_dir.mkdir(parents=True, exist_ok=True)
            (self.review_dir / f"{self.day}.md").write_text(review, encoding="utf-8")
        return review

    # ---- data -------------------------------------------------------------------------------------------
    def _refresh_chain(self, u: str, now) -> None:
        at = self.chain_at.get(u)
        if at is not None and (now - at) < pd.Timedelta(minutes=self.refresh_min):
            return
        try:
            ch = None
            fails = self.chain_fail.get(u, 0)
            tried = self.chain_tried.get(u)
            # after 3 straight failures, only retry the real chain every 15 minutes (each try can block ~20s)
            if self.chains is not self.model_chain and (fails < 3 or tried is None or now - tried >= pd.Timedelta(minutes=15)):
                self.chain_tried[u] = now
                try:
                    ch = self.chains.chain(u, self.expiry[u], spot=self.spot(u), ts=now)
                    if fails >= 3:
                        self.journal.event(now, "INFO", "chain", f"{u} {self.chains.name} chain is back")
                    self.chain_fail[u] = 0
                except Exception as exc:
                    # NSE blocks many cloud IPs, throttles, or is down: the desk must not stop because of it.
                    # Price off the model chain (India VIX + skew) and say so in the journal.
                    n = self.chain_fail[u] = fails + 1
                    if n == 1 or n % 10 == 0:
                        self.journal.event(now, "WARN", "chain", f"{u} {self.chains.name} chain unavailable ({exc!s:.160}); "
                                                               f"pricing off the model chain (India VIX)"
                                                               f"{f' — {n} failures in a row' if n > 1 else ''}")
            if ch is None:
                ch = self.model_chain.chain(u, self.expiry[u], spot=self.spot(u), ts=now)
            if not ch.attrs.get("spot") or ch.attrs["spot"] != ch.attrs["spot"]:
                ch.attrs["spot"] = self.spot(u)
            ch = fill_iv(ch, self.pricer)
            self.chain_df[u] = ch
            self.chain_an[u] = chain_analytics(ch, self.pricer)
            self.chain_at[u] = now
            self.marker.calibrate(ch, self.lot(u))
            if self.recorder and ch.attrs.get("source") != "model":
                self.recorder.record_chain(ch)
        except Exception as exc:
            self.chain_at[u] = now
            self.journal.event(now, "WARN", "chain", f"{u} chain refresh failed: {exc}")

    def _refresh_news(self, now) -> None:
        if self.news is None:
            return
        try:
            fresh = self.news.refresh(now)
        except Exception as exc:                          # the desk trades without news rather than not at all
            self.journal.event(now, "WARN", "news", f"news refresh failed: {exc!s:.160}")
            return
        if fresh:
            self.journal.news_add(fresh, now)
            hot = [x for x in fresh if x.impact == "high" and max(x.about.values() or [0]) >= self.news.min_relevance]
            for x in hot:
                self.say(f"  {now:%H:%M} NEWS ({x.source}) {x.title} [tone {x.sentiment:+.2f}]")

    def _think_globally(self, u: str, now):
        df = self.bars[u]
        today = df[df.index.date == self.day]
        prior = df[df.index.date < self.day]
        gap = float(np.log(today["open"].iloc[0] / prior["close"].iloc[-1])) if len(today) and len(prior) else None
        items = list(self.news.items.values()) if self.news is not None else None
        st = self.brain.think(u, now, items, gap)
        self.brain_state[u] = st
        return st

    def _vix_state(self) -> dict | None:
        v = self.bars.get(self.vix)
        if v is None or v.empty:
            return None
        today = v[v.index.date == self.day]
        prev = v[v.index.date < self.day]
        if today.empty:
            return None
        base = float(prev["close"].iloc[-1]) if len(prev) else float(today["open"].iloc[0])
        return {"last": float(today["close"].iloc[-1]), "chg": float(today["close"].iloc[-1]) / base - 1}

    def _flow_state(self, u: str, now) -> dict | None:
        fb = self.flow.get(u)
        if not self.feed.has_ticks or fb is None or not fb.bars:
            return None
        recent = [b for b in fb.bars if b.start >= now - pd.Timedelta(minutes=30)]
        if not recent:
            return None
        st = recent[-1].stacked
        return {"source": "ticks", "delta_30": sum(b.delta for b in recent), "volume_30": sum(b.volume for b in recent),
                "stacked_buy": st["buy"], "stacked_sell": st["sell"]}

    # ---- trading ------------------------------------------------------------------------------------------
    def _maybe_enter(self, u: str, view: MarketView, s: dict, now) -> str:
        if self.paused:
            return "standing aside: new entries paused from the app"
        if view.vetoes:
            return f"standing aside: {view.vetoes[0]}"
        if u not in self.chain_df:
            return "standing aside: no option chain"
        age = now - self.chain_at.get(u, now)
        if self.chain_df[u].attrs.get("source") != "model" and age > pd.Timedelta(minutes=self.stale_min):
            return f"standing aside: option chain {age.seconds // 60} min old"
        eq = self.equity(now)
        gate = self.risk.gate(now, eq, self.open_trades, u)
        if gate:
            return f"standing aside: {gate[0]}"
        plans = self.playbook.scan(view, s, self.chain_df[u], now)
        if not plans:
            return "watching: no setup has triggered"
        if self.quant_on:
            pick = self._guarded(u, now, "EV selection", self._select_by_ev, u, plans, view, eq, now)
            if pick is None:
                return "standing aside: the quant layer failed on this minute (see events)"
            if isinstance(pick, str):
                return pick
            plan, lots, notes = pick
            return self._open(plan, lots, notes, view, s, now)
        plan = max(plans, key=lambda p: p.conviction)
        lots, notes = self.risk.size(plan, eq, self.broker.cash())
        if lots < 1:
            self.journal.decision(now, plan.setup, u, "rejected", " | ".join(notes), 0, {"plan": plan.describe()})
            return f"setup {plan.setup} found but sized to 0 lots ({notes[-1]})"
        return self._open(plan, lots, notes, view, s, now)

    # ---- quant layer --------------------------------------------------------------------------------------------
    def preopen(self, now) -> None:
        """Before the open, every few minutes: GIFT Nifty (NSE IX's NIFTY future, trading since 06:30 IST), the
        overnight cue in one number. Recorded with the session; context for the read, not evidence (untested)."""
        if self.gift_source is None or (self._gift_at is not None and now - self._gift_at < pd.Timedelta(minutes=4)):
            return
        self._gift_at = now
        from ..data.nse import gift_implied_gap
        try:
            g = self.gift_source()
        except Exception as exc:
            self._gift_fails += 1
            if self._gift_fails == 1:
                self.journal.event(now, "WARN", "gift", f"GIFT Nifty unavailable ({exc!s:.160})")
            return
        if g is None or g.empty:
            return
        r = g.iloc[0]
        gap = gift_implied_gap(float(r["last"]), float(r["nifty_close"]), r["expiry"], now.date(),
                               self.cfg.get("backtest.risk_free", 0.065), self.cfg.get("backtest.dividend_yield", 0.012))
        first = self.gift is None
        self.gift = {"ts": str(r["ts"]), "last": float(r["last"]), "pct": float(r["pct"]),
                     "nifty_close": float(r["nifty_close"]), "implied_gap": gap, "taken": str(now)}
        self.journal.set_state("intraday_gift", self.gift)
        if self.recorder is not None:
            path = self.recorder.day_dir(now.date()) / "gift.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([self.gift]).to_csv(path, mode="a", header=not path.exists(), index=False)
        if first:
            self.journal.event(now, "INFO", "gift", f"GIFT Nifty {r['last']:,.1f} ({pd.Timestamp(r['ts']):%H:%M}): "
                                                   f"implies a {gap:+.2%} open for NIFTY after carry")

    def _load_heavyweights(self) -> None:
        """Results dates for the index heavyweights from the warehouse's copy of NSE's event calendar (live.yml pulls
        it); context only. Silently nothing when the file isn't there."""
        from .events import EventBook, from_config, heavyweight_results, load_corp_events
        try:
            corp = load_corp_events(self.warehouse_dir)
        except Exception:
            corp = None
        heavy = sorted({s for u in self.underlyings for s in (self.cfg.get(f"intraday.heavyweights.{u}") or [])})
        self.eventbook = EventBook(from_config(self.cfg) + heavyweight_results(corp, heavy))

    def _live(self, insts, now) -> dict[str, tuple[float, float]]:
        """Bid/ask right now from the broker's book when the chain source has one; {} otherwise (or on an error),
        and the plan's chain prices / the marks stand in."""
        try:
            out = self.chains.live_quotes(insts) if hasattr(self.chains, "live_quotes") else {}
            self.live_fails = 0
            return out or {}
        except Exception as exc:
            self.live_fails += 1
            if self.live_fails == 1 or self.live_fails % 10 == 0:
                self.journal.event(now, "WARN", "quotes", f"live quotes unavailable ({exc!s:.160}); "
                                                          f"using the last chain{f' — {self.live_fails} in a row' if self.live_fails > 1 else ''}")
            return {}

    def _guarded(self, u: str, now, what: str, fn, *args):
        """Run a quant step; on an error, journal it (once per session per kind) and return None so the desk keeps
        thinking and simply doesn't trade on numbers it couldn't compute."""
        try:
            return fn(*args)
        except Exception as exc:
            key = (self.day, u, what)
            if key not in self._qerrors:
                self._qerrors.add(key)
                log.exception("%s failed", what)
                self.journal.event(now, "ERROR", "quant", f"{u} {what} failed: {exc!r:.200}", {"where": _where(exc)})
            return None

    def _train_models(self, day) -> None:
        """Fit each underlying's direction model on sessions before `day` only, and say how it tested."""
        for u in self.underlyings:
            try:
                h = self.feed.history_bars(u, int(self.qc.get("train_days", 55)))
                h = h[h.index.date < day] if h is not None and len(h) else pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            except Exception as exc:
                h = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
                self.journal.event(session_bounds(day)[0], "WARN", "quant", f"{u}: no 5m history for the model ({exc!s:.120})")
            self.hist5[u] = h
            m = DirectionModel(min_auc=float(self.qc.get("min_auc", 0.53)))
            diag = m.fit(features_5m(h)) if len(h) else m.diag
            self.models[u] = m
            msg = (f"{u} direction model: {diag.get('status')}"
                   + (f" (walk-forward AUC {diag['auc_oos']:.3f}, log-loss skill {diag['logloss_skill_oos']:+.2%}, "
                      f"{diag['samples']:,} samples over {diag['days']} days)" if "auc_oos" in diag else
                      f" ({diag.get('samples', 0)} samples, {diag.get('days', 0)} days)"))
            self.journal.event(session_bounds(day)[0], "INFO", "quant", msg)
            self.say("  " + msg)

    def _quant_state(self, u: str, now) -> dict:
        df = self.bars[u]
        c = self._qc_cache.get(u)
        if c is None or c["day"] != self.day:                      # locate today's rows once per session
            prior = df[df.index.date < self.day]
            h5 = self.hist5.get(u)
            last5 = h5[h5.index.date == h5.index.date[-1]] if h5 is not None and len(h5) else None
            c = self._qc_cache[u] = {"day": self.day, "start": len(prior),
                                     "prev_close": float(prior["close"].iloc[-1]) if len(prior) else float("nan"),
                                     "prev_sig": session_sigma(last5["close"].to_numpy(dtype=float)) if last5 is not None else float("nan"),
                                     "volf": VolForecaster()}
        today = df.iloc[c["start"]:]
        m = self.models.get(u)
        p_model = None
        n5 = len(today) // 5                                        # completed 5m bars (features change only then)
        if m is not None and m.w is not None and n5 > 0:
            cached = self._pcache.get(u)
            if cached and cached[0] == (self.day, n5):
                p_model = cached[1]
            else:
                five = to_5m(today.iloc[:n5 * 5])
                f = features_5m(five, c["prev_close"], c["prev_sig"])
                p_model = m.predict(f.iloc[-1])
                self._pcache[u] = ((self.day, n5), p_model)
        an = self.chain_an.get(u) or {}
        vol = c["volf"].forecast(df, self.day, an.get("atm_iv"), today_close=today["close"].to_numpy(dtype=float))
        d = (m.diag if m is not None else {}) or {}
        rd = (self.research.get(u) or {}).get("drift")
        q = {"sigma_min": vol["sigma_min"], "sigma_30m_pct": vol["sigma_30m"] * 100, "vol_ann": vol["vol_ann"], "research_drift": rd,
             "rv_ann": vol["rv_ann"], "vol_source": vol["source"], "p_model": p_model, "valid": bool(m is not None and m.valid),
             "auc": d.get("auc_oos"), "samples": d.get("samples"), "model_status": d.get("status")}
        self.qstate[u] = q
        return q

    def _p_up(self, u: str, view) -> tuple[float, str]:
        q = self.qstate.get(u) or {}
        if q.get("valid") and q.get("p_model") is not None:
            return float(np.clip(q["p_model"], 0.35, 0.65)), f"direction model (AUC {q['auc']:.3f})"
        tilt = float(self.qc.get("prior_tilt", 0.10))
        return float(0.5 + np.clip(tilt * view.score, -tilt, tilt)), "prior tilt from the analyst's score (unvalidated)"

    def _select_by_ev(self, u: str, plans: list, view, eq: float, now):
        """Price every plan and its alternative structures by Monte Carlo; trade the best EV per rupee of risk
        that the account can hold and that clears the EV floor. Otherwise say why not."""
        q = self.qstate.get(u) or {}
        if not q.get("sigma_min"):
            return "standing aside: no volatility forecast yet"
        p_up, p_src = self._p_up(u, view)
        close = pd.Timestamp(dt.datetime.combine(self.day, self.risk.square_off), tz=IST)
        minutes_left = (close - now).total_seconds() / 60
        floor_inr, floor_r = float(self.qc.get("min_ev_inr", 40)), float(self.qc.get("min_ev_r", 0.05))
        cands = []
        for plan in plans:
            variants = [plan] + self.playbook.alternatives(plan, self.chain_df[u], now, self.qc.get("long_deltas", (0.30, 0.40)),
                                                            [tuple(x) for x in self.qc.get("spreads", ((0.45, 0.30), (0.50, 0.20)))])
            for v in variants:
                drift = ((self.research.get(u) or {}).get("drift") or {}).get("per_min", 0.0)
                e = self.ev.evaluate(v, view.spot, now, q["sigma_min"], p_up if v.direction != 0 else 0.5, minutes_left, drift)
                lots, notes = self.risk.size(v, eq, self.broker.cash())
                cands.append((v, e, lots, notes))
        fits = [c for c in cands if c[2] >= 1]
        ok = [c for c in fits if c[1]["ev"] >= max(floor_inr, floor_r * c[0].planned_risk_per_lot())]
        if not ok:
            best = max(fits or cands, key=lambda c: c[1]["ev_r"])
            v, e = best[0], best[1]
            why = (f"best of {len(cands)} structures is {v.structure} with EV ₹{e['ev']:+,.0f}/lot ({e['ev_r']:+.2f}R, "
                   f"P(profit) {e['p_profit']:.0%}) at P(up) {p_up:.2f} [{p_src}]"
                   + ("" if fits else "; none fits the account"))
            self.journal.decision(now, v.setup, u, "rejected", f"EV below the floor: {why}", 0,
                                  {"plan": v.describe(), "ev": {k: round(x, 3) if isinstance(x, float) else x for k, x in e.items()}})
            return f"setup {v.setup} found but not worth it after costs: {why}"
        plan, e, lots, notes = max(ok, key=lambda c: c[1]["ev_r"])
        bst = self.brain_state.get(u)
        if bst is not None and bst.size_mult < 1:
            # global stress: volatility targeting. With one-lot sizes this can mean no trade at all, by design.
            scaled = int(math.floor(lots * bst.size_mult + 0.49))   # 1 lot survives mild stress; ×0.5 (≈4σ) means aside
            notes = notes + [f"global stress {bst.stress:.1f}σ: size ×{bst.size_mult:.2f} → {scaled} lot(s)"]
            if scaled < 1:
                self.journal.decision(now, plan.setup, u, "rejected", notes[-1], 0, {"plan": plan.describe()})
                return f"setup {plan.setup} passed the EV gate but global stress is {bst.stress:.1f}σ: standing aside"
            lots = scaled
        others = sorted([c for c in cands if c[0] is not plan], key=lambda c: -c[1]["ev_r"])[:3]
        text = (f"EV ₹{e['ev']:+,.0f}/lot ({e['ev_r']:+.2f}R), P(profit) {e['p_profit']:.0%}, CVaR5 ₹{e['cvar5']:,.0f}, costs "
                f"₹{e['fees'] + e['exit_cost']:,.0f}/lot, σ over {e['horizon_min']}m {e['sigma_h_pct']:.2f}%, P(up) {p_up:.2f} [{p_src}]; "
                f"picked over " + ", ".join(f"{c[0].structure} {c[1]['ev_r']:+.2f}R" for c in others))
        plan.notes["quant"] = {k: (round(x, 4) if isinstance(x, float) else x) for k, x in e.items()}
        plan.notes["quant"]["p_source"] = p_src
        plan.notes["quant_text"] = text
        return plan, lots, notes + [f"quant: {text}"]

    def _open(self, plan: TradePlan, lots: int, notes: list[str], view: MarketView, s: dict, now) -> str:
        u = plan.symbol
        t = Trade(id=new_trade_id("I"), strategy=plan.setup, family="intraday", symbol=u, direction=plan.direction,
                  kind=OPTIONS, legs=[], units=lots, opened_at=now, entry_underlying=view.spot,
                  initial_risk=plan.planned_risk_per_lot() * lots, stop=plan.invalidation, target=plan.target_underlying,
                  exit_rules={"premium_stop": plan.premium_stop, "premium_target": plan.premium_target,
                              "time_stop_min": plan.time_stop_min, "credit": plan.is_credit},
                  rationale=(f"[{plan.setup}] Trigger: {plan.trigger}. Thesis: {plan.thesis} "
                             f"Structure: {plan.describe()} (vol view {view.vol_view}). "
                             + (f"Quant: {plan.notes['quant_text']}. " if plan.notes.get("quant_text") else "")
                             + f"Market read: {view.narrative}"),
                  context={"regime": view.day_type, "bias": view.bias, "score": round(view.score, 3),
                           "conviction": round(view.conviction, 3), "vol_view": view.vol_view,
                           "evidence": [(e.factor, round(e.direction, 2), e.observation) for e in view.evidence],
                           "levels": {k: round(float(v), 2) for k, v in view.levels.items() if v == v and v is not None},
                           "chain": {k: v for k, v in view.chain.items() if isinstance(v, (int, float, str))}},
                  meta={"structure": plan.structure, "expiry": str(plan.expiry), "quote_source": plan.quote_source,
                        "entry_net_premium_per_lot": plan.net_premium, "max_loss": plan.max_loss_per_lot(),
                        "legs_plan": [(l.strike, l.right, l.ratio, round(l.price, 2), round(l.iv, 2), round(l.delta, 3))
                                      for l in plan.legs], **plan.notes})
        insts = [Instrument.option(u, plan.expiry, l.strike, l.right, plan.lot_size) for l in plan.legs]
        live = self._live(insts, now)
        px = {i.symbol: (live[i.symbol][1] if l.ratio > 0 else live[i.symbol][0]) if i.symbol in live else l.price
              for i, l in zip(insts, plan.legs)}
        if live:
            # a real limit order at the planned prices wouldn't fill if the book has moved away: skip, don't chase
            planned = sum(l.ratio * l.price for l in plan.legs)
            now_net = sum(l.ratio * px[i.symbol] for i, l in zip(insts, plan.legs))
            worse = now_net - planned                       # + = paying more (debit) or collecting less (credit)
            if abs(planned) > 0 and worse > self.max_entry_slip * abs(planned):
                msg = (f"skipped {plan.setup}: the live book moved away (planned ₹{planned * plan.lot_size:,.0f}/lot, "
                       f"now ₹{now_net * plan.lot_size:,.0f}/lot)")
                self.journal.decision(now, plan.setup, u, "rejected", msg, 0, {"plan": plan.describe()})
                return msg
        t.meta["fill_quotes"] = f"live {self.chains.name} book" if len(live) == len(insts) else "option chain"
        for inst, leg in zip(insts, plan.legs):
            qty = leg.ratio * lots * plan.lot_size
            fill = self.broker.execute(Order(inst, qty, t.id, "open"), px[inst.symbol], now)
            if fill is None:
                for done in t.legs:
                    self.broker.execute(Order(done.instrument, -done.qty, t.id, "unwind"), done.entry_price, now)
                return f"order for {inst.symbol} rejected; nothing opened"
            self.journal.fill(now, t.id, inst.symbol, qty, fill.price, fill.fees, fill.fee_breakdown)
            t.legs.append(TradeLeg(inst, qty, fill.price))
            t.fees += fill.fees
        t.pnl = -t.fees
        t.last_mark = {l.instrument.symbol: l.entry_price for l in t.legs}
        self.open_trades.append(t)
        self.risk.trades_today += 1
        self.journal.open_trade(t, notes)
        msg = f"ENTER {plan.setup} {lots}×{plan.describe()} | stop {plan.invalidation or '—'} | {plan.trigger}"
        self.say(f"  {now:%H:%M} {u:<9} ▲ {msg}")
        return msg

    def _manage(self, u: str, view: MarketView, now) -> list[str]:
        out = []
        S = view.spot
        for t in [x for x in self.open_trades if x.symbol == u]:
            marks = {l.instrument.symbol: self.marker.mid(l.instrument, S, now) for l in t.legs}
            t.update_excursions(marks)
            t.bars_held = int((now - t.opened_at).total_seconds() // 60)
            gross = t.value(marks) - t.entry_cost
            prem = abs(t.entry_cost)
            r = t.exit_rules
            credit = r.get("credit", t.entry_cost < 0)
            reason = note = None
            if t.stop is not None and ((t.direction > 0 and S <= t.stop) or (t.direction < 0 and S >= t.stop)):
                reason, note = "invalidation", f"underlying {S:,.2f} through {t.stop:,.2f}"
            elif t.meta.get("range") and not (t.meta["range"][0] <= S <= t.meta["range"][1]):
                reason, note = "range_break", f"underlying {S:,.2f} left the value area {t.meta['range']}"
            elif gross <= -prem * r["premium_stop"]:
                reason, note = "premium_stop", f"P&L ₹{gross:,.0f} hit the {'credit ×' if credit else ''}{r['premium_stop']} stop"
            elif gross >= prem * r["premium_target"]:
                reason, note = "premium_target", f"P&L ₹{gross:,.0f} reached the {r['premium_target']:.0%} target"
            elif t.target is not None and ((t.direction > 0 and S >= t.target) or (t.direction < 0 and S <= t.target)):
                reason, note = "underlying_target", f"underlying reached {t.target:,.2f}"
            elif t.mfe > 0.5 * prem * r["premium_target"] and gross <= 0:
                reason, note = "breakeven_stop", "gave back a half-target open profit: out at breakeven"
            elif t.bars_held >= r["time_stop_min"] and gross < 0.1 * prem:
                reason, note = "time_exit", f"no progress in {t.bars_held} min"
            if reason:
                out.append(self._close(t, now, reason, note))
        return out

    def _close(self, t: Trade, now, reason: str, note: str) -> str:
        S = self.spot(t.symbol)
        live = self._live([l.instrument for l in t.legs], now)
        t.meta["exit_quotes"] = f"live {self.chains.name} book" if len(live) == len(t.legs) else "marked"
        for l in t.legs:
            q = live.get(l.instrument.symbol)
            px = (q[0] if l.qty > 0 else q[1]) if q else self.marker.exit_price(l.instrument, l.qty, S, now)
            fill = self.broker.execute(Order(l.instrument, -l.qty, t.id, "close"), px, now)
            if fill is None:
                self.journal.event(now, "ERROR", "execution", f"exit {l.instrument.symbol} rejected for {t.id}")
                continue
            self.journal.fill(now, t.id, l.instrument.symbol, -l.qty, fill.price, fill.fees, fill.fee_breakdown)
            l.exit_price = fill.price
            t.fees += fill.fees
        t.pnl = sum(l.qty * ((l.exit_price or l.entry_price) - l.entry_price) for l in t.legs) - t.fees
        t.mae, t.mfe = min(t.mae, t.pnl), max(t.mfe, t.pnl)
        t.status, t.closed_at, t.exit_reason, t.exit_note, t.exit_underlying = "closed", now, reason, note, S
        t.bars_held = int((now - t.opened_at).total_seconds() // 60)
        self.open_trades.remove(t)
        self.closed.append(t)
        v = self.views.get(t.symbol)
        rv = self.journal.close_trade(t, v.day_type if v else None)
        self.risk.on_close(t.pnl, now)
        msg = f"EXIT {t.strategy} {reason}: ₹{t.pnl:,.0f} ({t.r_multiple:+.2f}R, grade {rv['grade']}) — {note}"
        self.say(f"  {now:%H:%M} {t.symbol:<9} ▼ {msg}")
        return msg

    # ---- remote control (the app writes commands; the engine executes them) ----------------------------------
    def _commands(self, now) -> None:
        cmds = self.journal.get_state("intraday_cmds") or []
        done = set(self.journal.get_state("intraday_cmds_done") or [])
        todo = [c for c in cmds if c.get("id") not in done]
        for c in todo:
            cmd, arg = c.get("cmd"), c.get("arg")
            if cmd == "pause":
                self.paused = True
            elif cmd == "resume":
                self.paused = False
            elif cmd in ("flatten", "close"):
                for t in list(self.open_trades):
                    if cmd == "flatten" or t.id == arg:
                        self._close(t, now, "manual", f"{'flattened' if cmd == 'flatten' else 'closed'} from the app")
                if cmd == "flatten":
                    self.paused = True
            done.add(c.get("id"))
            self.journal.event(now, "WARN", "remote", f"{cmd}{' ' + str(arg) if arg else ''} from the app")
            self.say(f"  {now:%H:%M} remote command: {cmd} {arg or ''}")
        if todo:
            self.journal.set_state("intraday_cmds_done", sorted(done))
            self.journal.set_state("intraday_paused", self.paused)

    def _heartbeat(self, now) -> None:
        """Everything the app's Live screen needs, in one small state row updated every minute."""
        eq = self.equity(now)
        views = {}
        for u, v in self.views.items():
            ns = self.news.state(u, now) if self.news is not None else None
            views[u] = {"spot": v.spot, "bias": v.bias, "score": v.score, "conviction": v.conviction, "day_type": v.day_type,
                        "vol_view": v.vol_view, "iv": v.iv, "rv": v.rv, "narrative": v.narrative, "vetoes": v.vetoes,
                        "levels": {k: float(x) for k, x in v.levels.items() if x is not None and x == x},
                        "chg": v.state.get("chg"), "vwap": v.state.get("vwap"), "phase": v.state.get("phase"),
                        "expiry": str(self.expiry.get(u)), "news": ns,
                        "quant": {k: x for k, x in (self.qstate.get(u) or {}).items() if k != "sigma_min"} or None,
                        "action": self.last_action.get(u),
                        "brain": _brain_view(self.brain_state.get(u)),
                        "evidence": [{"factor": e.factor, "category": e.category, "direction": e.direction,
                                      "weight": e.weight, "observation": e.observation} for e in v.evidence]}
        positions = []
        for t in self.open_trades:
            S = self.spot(t.symbol)
            marks = {l.instrument.symbol: self.marker.mid(l.instrument, S, now) for l in t.legs}
            positions.append({"id": t.id, "setup": t.strategy, "symbol": t.symbol, "structure": t.meta.get("structure"),
                              "lots": t.units, "opened": str(t.opened_at), "pnl": t.value(marks) - t.entry_cost - t.fees,
                              "stop": t.stop, "target": t.target, "entry_underlying": t.entry_underlying, "spot": S,
                              "legs": [{"symbol": l.instrument.symbol, "qty": l.qty, "entry": l.entry_price,
                                        "mark": marks[l.instrument.symbol]} for l in t.legs],
                              "rationale": t.rationale[:600]})
        self.journal.set_state("intraday_live", {
            "ts": str(now), "day": str(self.day), "equity": eq, "day_start_equity": self.day_start_equity,
            "day_pnl": eq - self.day_start_equity, "paused": self.paused, "halted": self.risk.halted,
            "trades_today": self.risk.trades_today, "feed": self.feed.name, "chain": self.chain_name(),
            "views": views, "positions": positions,
            "news_health": dict(self.news.health) if self.news is not None else None,
            "global": _global_view(self.brain, now), "gift": self.gift, "events": self.events_today})

    # ---- journaling ------------------------------------------------------------------------------------------
    def _think(self, u: str, view: MarketView, now, action: str, force: bool = False) -> None:
        self.last_action[u] = action
        last = self.last_thought.get(u)
        changed = self.last_bias.get(u) != view.bias
        trade_event = action.startswith(("ENTER", "EXIT")) or force
        if last is None or trade_event or changed or now - last >= pd.Timedelta(minutes=self.think_every):
            self.journal.thought(view, action)
            self.last_thought[u], self.last_bias[u] = now, view.bias
            if not trade_event:
                self.say(f"  {now:%H:%M} {u:<9} · {view.bias:<8} {view.score:+.2f} c{view.conviction:.2f} "
                         f"{view.day_type:<12} IV/RV {view.vol_view:<7} | {action}")

    def _snapshot(self, now) -> None:
        if now.minute % 5:
            return
        eq = self.equity(now)
        self.journal.snapshot(now, equity=eq, cash=self.broker.cash(), drawdown=min(0.0, eq / self.day_start_equity - 1),
                              open_trades=len(self.open_trades), gross=0.0, net_delta=0.0, vega=0.0,
                              open_risk=sum(t.initial_risk for t in self.open_trades),
                              regime=",".join(f"{u}:{v.day_type}" for u, v in self.views.items()))

    def _persist(self, ended: bool = False) -> None:
        r = self.risk
        self.journal.set_state("intraday_open", {
            "day": str(self.day), "ended": ended, "trades": [trade_to_dict(t) for t in self.open_trades],
            "closed": [trade_to_dict(t) for t in self.closed if t.opened_at.date() == self.day],
            "day_start_equity": self.day_start_equity,
            "risk": {"trades_today": r.trades_today, "consec_losses": r.consec_losses, "halted": r.halted,
                     "cool_until": str(r.cool_until) if r.cool_until is not None else None}})

    def _restore(self, day: dt.date) -> dict:
        """Pick up today's session after a restart: open positions, the trades already closed, the day's
        starting equity and the risk state (trade count, loss streak, cooldown, halt)."""
        st = self.journal.get_state("intraday_open") or {}
        if st.get("day") != str(day):
            if st.get("trades"):
                self.journal.event(pd.Timestamp.now(tz=IST), "WARN", "session",
                                   f"{len(st['trades'])} position(s) from {st.get('day')} were never squared off; ignored")
            return {}
        self.open_trades = [trade_from_dict(d) for d in st.get("trades") or []]
        self.closed = [trade_from_dict(d) for d in st.get("closed") or []]
        if self.open_trades or self.closed or st.get("day_start_equity"):
            self.say(f"  resumed today's session: {len(self.open_trades)} open, {len(self.closed)} closed")
        return st

    def chain_name(self) -> str:
        """The chain actually in use: the configured source, or what is standing in for it."""
        srcs = {str(ch.attrs.get("source")) for ch in self.chain_df.values()}
        name = self.chains.name
        if self.chains is self.model_chain or not srcs or srcs == {name}:
            return name
        if name not in srcs:
            return f"{'+'.join(sorted(srcs))} (no {name})"
        return "+".join([name] + sorted(srcs - {name}))

    # ---- review ------------------------------------------------------------------------------------------------
    def session_review(self) -> str:
        day = self.day
        trades = [t for t in self.closed if t.opened_at.date() == day]
        th = self.journal.thoughts(str(day))
        end_eq = self.broker.cash()
        L = [f"# Intraday session review — {day}", "",
             f"Capital ₹{self.day_start_equity:,.0f} → ₹{end_eq:,.0f} (**{end_eq / self.day_start_equity - 1:+.2%}**, "
             f"₹{end_eq - self.day_start_equity:+,.0f}); {len(trades)} trade(s); chain source {self.chain_name()}; "
             f"feed {self.feed.name}.", ""]
        if self.events_today:
            L += [f"Events: {'; '.join(self.events_today)}.", ""]
        if self.gift and "NIFTY" in self.bars:
            b = self.bars["NIFTY"]
            today, prior = b[b.index.date == day], b[b.index.date < day]
            if len(today) and len(prior):
                gap = today["open"].iloc[0] / prior["close"].iloc[-1] - 1
                L += [f"GIFT Nifty before the open implied **{self.gift['implied_gap']:+.2%}** for NIFTY (after carry); "
                      f"it opened **{gap:+.2%}**.", ""]
        for u in self.underlyings:
            tu = th[th["symbol"] == u] if not th.empty else th
            if tu.empty:
                continue
            first, lastr = tu.iloc[0], tu.iloc[-1]
            L += [f"## {u}", f"- Opened read ({str(first.ts)[11:16]}): {first.narrative}",
                  f"- Closing read ({str(lastr.ts)[11:16]}): {lastr.narrative}"]
            flips = tu[tu["bias"] != tu["bias"].shift()]
            if len(flips) > 1:
                L.append("- Bias path: " + " → ".join(f"{str(r.ts)[11:16]} {r.bias}" for r in flips.itertuples()))
            types = tu["day_type"].value_counts()
            L.append(f"- Day type (share of reads): " + ", ".join(f"{k} {v / len(tu):.0%}" for k, v in types.items()))
            L.append("")
        if trades:
            L += ["## Trades", "", "| Time | Setup | Structure | Lots | P&L ₹ | R | Exit | Grade |", "|---|---|---|---:|---:|---:|---|---|"]
            for t in trades:
                L.append(f"| {t.opened_at:%H:%M}–{t.closed_at:%H:%M} | {t.strategy} {t.symbol} | {t.meta.get('structure')} | "
                         f"{t.units} | {t.pnl:,.0f} | {t.r_multiple:+.2f} | {t.exit_reason} | "
                         f"{self._grade(t.id)} |")
            L.append("")
            for t in trades:
                L += [f"**{t.id} · {t.strategy} {t.symbol}** — {t.rationale}", f"Exit: {t.exit_reason} — {t.exit_note}.", ""]
            wins = sum(t.pnl > 0 for t in trades)
            L.append(f"Win rate {wins}/{len(trades)}, net ₹{sum(t.pnl for t in trades):,.0f}, costs ₹{sum(t.fees for t in trades):,.0f}.")
        else:
            reasons = th["action"].str.extract(r"^(standing aside|watching)[^:]*: (.*)$")[1].dropna().value_counts().head(4) \
                if not th.empty else pd.Series(dtype=int)
            L.append("No trades. Most common reasons: " + "; ".join(f"{k} (x{v})" for k, v in reasons.items()))
        return "\n".join(L)

    def _grade(self, tid: str) -> str:
        r = self.journal.df("SELECT grade FROM trades WHERE id=?", (tid,))
        return r["grade"].iloc[0] if not r.empty else "?"


def _brain_view(st) -> dict | None:
    if st is None:
        return None
    return {"regime": st.regime, "regime_score": st.regime_score, "stress": st.stress, "size_mult": st.size_mult,
            "narrative": st.narrative, "gap": st.gap, "evidence": st.evidence,
            "drivers": [{k: d.get(k) for k in ("id", "name", "sign", "live", "prior_z", "z30", "pressure", "gap_beta", "gap_corr",
                                                "co_corr", "lead_t", "validated", "lead_sign", "news_tone", "news_n", "news_latest")}
                        | {"move": d.get("lead")} for d in st.drivers]}


def _global_view(brain, now) -> dict | None:
    if brain is None or brain.gfeed is None:
        return None
    mk = brain.gfeed.snapshot(now)
    return {"markets": {k: {x: m.get(x) for x in ("name", "region", "india", "last", "prior_ret", "prior_z", "since_open", "r30",
                                                   "z30", "live", "last_ts", "prior_date")} for k, m in mk.items()},
            "health": dict(brain.gfeed.health)}


def _where(exc: BaseException, frames: int = 6) -> list[str]:
    """The last few frames of a failure, so a journaled error can be traced from the journal alone
    (29 Sep 2026: a one-off TypeError on the live desk was journaled as a bare repr, untraceable)."""
    tb = traceback.extract_tb(exc.__traceback__)[-frames:]
    return [f"{Path(f.filename).name}:{f.lineno} {f.name}: {(f.line or '').strip()[:120]}" for f in tb]


# ---- drivers ------------------------------------------------------------------------------------------
def run_replay(engine: IntradayEngine) -> str:
    feed = engine.feed
    assert isinstance(feed, ReplayFeed)
    engine.start_session(feed.day)
    while feed.advance():
        engine.step()
    return engine.end_session()


def run_live(engine: IntradayEngine, stop_at: dt.time | None = None, handover: bool = False) -> str:
    """Wall-clock loop: waits for the open, steps a few seconds after each minute closes.

    stop_at + handover: stop at that time *without* squaring off; the next run (same journal and
    broker state) resumes the session where this one left off. That is how two back-to-back
    runners cover a full 6¼-hour session when each is capped at 6 hours."""
    now = engine.feed.now()
    day = now.date()
    if not engine.cal.is_trading_day(day):
        return f"{day} is not an NSE trading day"
    open_ts, close_ts = session_bounds(day)
    end = pd.Timestamp(dt.datetime.combine(day, stop_at), tz=IST) if stop_at else close_ts
    if now >= close_ts:
        st = engine.journal.get_state("intraday_open") or {}
        if st.get("day") == str(day) and not st.get("ended") and (st.get("trades") or st.get("closed")):
            engine.start_session(day)                   # a handed-over session nobody closed: close it now
            return engine.end_session()
        return f"the {day} session is over"
    if handover and now >= end:
        return f"past the hand-over time {stop_at:%H:%M}; nothing to do"
    if now < open_ts:
        engine.say(f"waiting for the open ({open_ts:%H:%M})…")
        # read the news while waiting, so the opening read (and the site's News tab) already know the overnight stories
        while engine.feed.now() < open_ts:
            n = engine.feed.now()
            if engine.news is not None:
                engine._refresh_news(n)
            engine.preopen(n)
            engine.journal.commit()
            time.sleep(max(1.0, min(240.0, (open_ts - n).total_seconds() + 5)))
    engine.start_session(day)
    while engine.feed.now() < end + pd.Timedelta(seconds=30):
        try:
            engine.step()
        except Exception as exc:                        # keep the loop alive; journal the failure
            log.exception("step failed")
            engine.journal.event(engine.feed.now(), "ERROR", "engine", repr(exc), {"where": _where(exc)})
        n = engine.feed.now()
        time.sleep(max(1.0, 60 - n.second + 4))
    if handover and end < close_ts:
        engine._persist()
        engine.journal.event(engine.feed.now(), "INFO", "session", f"handed over at {stop_at:%H:%M} with "
                             f"{len(engine.open_trades)} open position(s)")
        engine.journal.commit()
        eq = engine.equity(engine.feed.now())
        return (f"handed over at {stop_at:%H:%M}: {len(engine.open_trades)} open, {len(engine.closed)} closed, "
                f"day P&L ₹{eq - engine.day_start_equity:+,.0f}")
    return engine.end_session()


def close_out(engine: IntradayEngine, note: str = "stopped by the operator") -> str:
    """Square off today's open positions now, at current prices, and close the session: what
    cancelling the day's run (the kill switch) does, so no paper position is left dangling."""
    day = engine.feed.now().date()
    st = engine.journal.get_state("intraday_open") or {}
    if st.get("day") != str(day) or st.get("ended") or not st.get("trades"):
        return "nothing open to close"
    engine.start_session(day)
    return engine.end_session("manual", note)
