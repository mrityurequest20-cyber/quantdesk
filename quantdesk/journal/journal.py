"""The trading journal: an SQLite record of every decision the system makes and why.

Tables
  trades     one row per trade: plan (rationale, context, stop/target), execution, outcome,
             and an automatic post-trade review (grade, process/outcome scores, lessons)
  decisions  every proposal, including the ones risk rejected, with the reason
  fills      every execution with its statutory cost breakdown
  events     halts, limit breaches, data problems, routine notes
  equity     end-of-day account snapshot (equity, drawdown, exposures, regime)
  checks     results of the pre-market / intraday / post-market routines
  state      engine state for paper/live continuity between runs
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.types import Instrument, Trade, TradeLeg
from .review import review_trade

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
  id TEXT PRIMARY KEY, strategy TEXT, family TEXT, symbol TEXT, direction INTEGER, kind TEXT, units INTEGER,
  opened_at TEXT, closed_at TEXT, entry_underlying REAL, exit_underlying REAL, stop REAL, target REAL,
  initial_risk REAL, pnl REAL, fees REAL, r_multiple REAL, mae REAL, mfe REAL, bars_held INTEGER,
  exit_reason TEXT, exit_note TEXT, status TEXT, legs TEXT, rationale TEXT, context TEXT, meta TEXT,
  sizing TEXT, grade TEXT, process_score REAL, outcome_score REAL, lessons TEXT, review TEXT
);
CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, strategy TEXT, symbol TEXT, action TEXT, units INTEGER,
  detail TEXT, context TEXT
);
CREATE TABLE IF NOT EXISTS fills (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, trade_id TEXT, symbol TEXT, qty INTEGER, price REAL,
  fees REAL, breakdown TEXT
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, level TEXT, category TEXT, message TEXT, data TEXT
);
CREATE TABLE IF NOT EXISTS equity (
  ts TEXT PRIMARY KEY, equity REAL, cash REAL, drawdown REAL, open_trades INTEGER, gross REAL,
  net_delta REAL, vega REAL, open_risk REAL, regime TEXT
);
CREATE TABLE IF NOT EXISTS checks (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, routine TEXT, name TEXT, status TEXT, detail TEXT
);
CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS thoughts (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, symbol TEXT, bias TEXT, score REAL, conviction REAL,
  day_type TEXT, vol_view TEXT, spot REAL, action TEXT, narrative TEXT, evidence TEXT, vetoes TEXT, levels TEXT, chain TEXT
);
CREATE INDEX IF NOT EXISTS ix_thoughts_ts ON thoughts(ts);
CREATE TABLE IF NOT EXISTS news (
  id TEXT PRIMARY KEY, ts TEXT, seen_at TEXT, source TEXT, sources TEXT, title TEXT, link TEXT, summary TEXT,
  sentiment REAL, impact TEXT, about TEXT);
CREATE INDEX IF NOT EXISTS ix_news_ts ON news(ts);
CREATE INDEX IF NOT EXISTS ix_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS ix_decisions_ts ON decisions(ts);
"""


def _json(x) -> str:
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, (pd.Timestamp,)):
            return str(o)
        return str(o)
    return json.dumps(x, default=conv)


def trade_to_dict(t: Trade) -> dict:
    d = t.to_record()
    d.update({"legs": [{"instrument": l.instrument.to_dict(), "qty": l.qty, "entry_price": l.entry_price,
                        "exit_price": l.exit_price} for l in t.legs],
              "exit_rules": t.exit_rules, "rationale": t.rationale, "context": t.context, "meta": t.meta,
              "exit_note": t.exit_note})
    return d


def trade_from_dict(d: dict) -> Trade:
    legs = [TradeLeg(Instrument.from_dict(l["instrument"]), int(l["qty"]), float(l["entry_price"]),
                     l.get("exit_price")) for l in d["legs"]]
    t = Trade(id=d["id"], strategy=d["strategy"], family=d["family"], symbol=d["symbol"], direction=int(d["direction"]),
              kind=d["kind"], legs=legs, units=int(d["units"]), opened_at=pd.Timestamp(d["opened_at"]),
              entry_underlying=float(d["entry_underlying"]), initial_risk=float(d["initial_risk"]),
              stop=d.get("stop"), target=d.get("target"), exit_rules=d.get("exit_rules", {}),
              rationale=d.get("rationale", ""), context=d.get("context", {}), meta=d.get("meta", {}),
              fees=float(d.get("fees", 0)), status=d.get("status", "open"))
    t.mae, t.mfe, t.bars_held, t.pnl = float(d.get("mae", 0)), float(d.get("mfe", 0)), int(d.get("bars_held", 0)), float(d.get("pnl", 0))
    if d.get("closed_at"):
        t.closed_at = pd.Timestamp(d["closed_at"])
    t.exit_reason, t.exit_note, t.exit_underlying = d.get("exit_reason"), d.get("exit_note", ""), d.get("exit_underlying")
    return t


class Journal:
    def __init__(self, path: str | Path = ":memory:", autocommit_every: int = 200):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # One connection, serialised by a lock, so the desk server may call from its own thread.
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.lock = threading.RLock()
        self.db.executescript(SCHEMA)
        self._pending = 0
        self._every = autocommit_every

    def _exec(self, sql: str, params=()):
        with self.lock:
            self.db.execute(sql, params)
            self._pending += 1
            if self._pending >= self._every:
                self.commit()

    def commit(self):
        with self.lock:
            self.db.commit()
            self._pending = 0

    def close(self):
        self.commit()
        self.db.close()

    # ---- writes -------------------------------------------------------------------------------
    def open_trade(self, t: Trade, sizing: list[str] | None = None) -> None:
        self._exec("""INSERT OR REPLACE INTO trades (id, strategy, family, symbol, direction, kind, units, opened_at,
                      entry_underlying, stop, target, initial_risk, pnl, fees, status, legs, rationale, context, meta, sizing)
                      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                   (t.id, t.strategy, t.family, t.symbol, t.direction, t.kind, t.units, str(t.opened_at),
                    t.entry_underlying, t.stop, t.target, t.initial_risk, t.pnl, t.fees, "open",
                    _json(trade_to_dict(t)["legs"]), t.rationale, _json(t.context), _json(t.meta), _json(sizing or [])))

    def update_open_trade(self, t: Trade) -> None:
        self._exec("UPDATE trades SET stop=?, pnl=?, fees=?, mae=?, mfe=?, bars_held=?, meta=? WHERE id=?",
                   (t.stop, t.pnl, t.fees, t.mae, t.mfe, t.bars_held, _json(t.meta), t.id))

    def close_trade(self, t: Trade, regime_at_exit: str | None = None) -> dict:
        rv = review_trade(t, regime_at_exit)
        self._exec("""UPDATE trades SET closed_at=?, exit_underlying=?, pnl=?, fees=?, r_multiple=?, mae=?, mfe=?,
                      bars_held=?, exit_reason=?, exit_note=?, status='closed', legs=?, meta=?, grade=?,
                      process_score=?, outcome_score=?, lessons=?, review=? WHERE id=?""",
                   (str(t.closed_at), t.exit_underlying, t.pnl, t.fees, t.r_multiple, t.mae, t.mfe, t.bars_held,
                    t.exit_reason, t.exit_note, _json(trade_to_dict(t)["legs"]), _json(t.meta), rv["grade"],
                    rv["process_score"], rv["outcome_score"], _json(rv["lessons"]), rv["summary"], t.id))
        return rv

    def decision(self, ts, strategy: str, symbol: str, action: str, detail: str, units: int = 0, context=None):
        self._exec("INSERT INTO decisions (ts, strategy, symbol, action, units, detail, context) VALUES (?,?,?,?,?,?,?)",
                   (str(ts), strategy, symbol, action, units, detail, _json(context or {})))

    def fill(self, ts, trade_id: str, symbol: str, qty: int, price: float, fees: float, breakdown: dict):
        self._exec("INSERT INTO fills (ts, trade_id, symbol, qty, price, fees, breakdown) VALUES (?,?,?,?,?,?,?)",
                   (str(ts), trade_id, symbol, qty, price, fees, _json(breakdown)))

    def event(self, ts, level: str, category: str, message: str, data=None):
        self._exec("INSERT INTO events (ts, level, category, message, data) VALUES (?,?,?,?,?)",
                   (str(ts), level, category, message, _json(data or {})))

    def snapshot(self, ts, **row):
        cols = ["equity", "cash", "drawdown", "open_trades", "gross", "net_delta", "vega", "open_risk", "regime"]
        self._exec(f"INSERT OR REPLACE INTO equity (ts, {', '.join(cols)}) VALUES (?{',?' * len(cols)})",
                   (str(ts), *[row.get(c) for c in cols]))

    def check(self, ts, routine: str, name: str, status: str, detail: str):
        self._exec("INSERT INTO checks (ts, routine, name, status, detail) VALUES (?,?,?,?,?)",
                   (str(ts), routine, name, status, detail))

    def thought(self, view, action: str = "") -> None:
        """The analyst's read at one moment (intraday), whether or not it traded."""
        self._exec("INSERT INTO thoughts (ts, symbol, bias, score, conviction, day_type, vol_view, spot, action, narrative, "
                   "evidence, vetoes, levels, chain) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (str(view.ts), view.symbol, view.bias, view.score, view.conviction, view.day_type, view.vol_view,
                    view.spot, action, view.narrative,
                    _json([{"factor": e.factor, "category": e.category, "direction": e.direction, "weight": e.weight,
                            "observation": e.observation} for e in view.evidence]),
                    _json(view.vetoes), _json(view.levels), _json(view.chain)))

    def news_add(self, items, seen_at) -> None:
        """Headlines as the desk saw them (with its own sentiment / impact / relevance scores)."""
        for it in items:
            r = it.to_record()
            self._exec("INSERT OR REPLACE INTO news (id, ts, seen_at, source, sources, title, link, summary, sentiment, impact, about) "
                       "VALUES (?,?,?,?,?,?,?,?,?,?,?)", (r["id"], r["ts"], str(seen_at), r["source"], _json(r["sources"]),
                                                          r["title"], r["link"], r["summary"], r["sentiment"], r["impact"],
                                                          _json(r["about"])))

    def news(self, since: str | None = None, n: int = 200) -> pd.DataFrame:
        q, p = "SELECT * FROM news", []
        if since:
            q += " WHERE ts >= ?"
            p.append(since)
        return self.df(q + " ORDER BY ts DESC LIMIT ?", (*p, int(n)))

    def thoughts(self, since: str | None = None, symbol: str | None = None) -> pd.DataFrame:
        q, p = "SELECT * FROM thoughts WHERE 1=1", []
        if since:
            q += " AND ts >= ?"
            p.append(since)
        if symbol:
            q += " AND symbol = ?"
            p.append(symbol)
        return self.df(q + " ORDER BY id", tuple(p))

    def set_state(self, key: str, value) -> None:
        self._exec("INSERT OR REPLACE INTO state (key, value) VALUES (?,?)", (key, _json(value)))
        self.commit()

    def get_state(self, key: str, default=None):
        with self.lock:
            row = self.db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    # ---- reads --------------------------------------------------------------------------------
    def df(self, sql: str, params=()) -> pd.DataFrame:
        with self.lock:
            self.commit()
            return pd.read_sql_query(sql, self.db, params=params)

    def trades(self, status: str | None = None) -> pd.DataFrame:
        q = "SELECT * FROM trades" + (" WHERE status=?" if status else "") + " ORDER BY opened_at"
        return self.df(q, (status,) if status else ())

    def equity(self) -> pd.DataFrame:
        d = self.df("SELECT * FROM equity ORDER BY ts")
        if not d.empty:
            d["ts"] = pd.to_datetime(d["ts"])
            d = d.set_index("ts")
        return d

    def events(self, since: str | None = None, level: str | None = None) -> pd.DataFrame:
        q, p = "SELECT * FROM events WHERE 1=1", []
        if since:
            q += " AND ts >= ?"
            p.append(since)
        if level:
            q += " AND level = ?"
            p.append(level)
        return self.df(q + " ORDER BY id", tuple(p))

    def decisions(self, since: str | None = None) -> pd.DataFrame:
        return self.df("SELECT * FROM decisions" + (" WHERE ts >= ?" if since else "") + " ORDER BY id",
                       (since,) if since else ())

    def checks(self, since: str | None = None) -> pd.DataFrame:
        return self.df("SELECT * FROM checks" + (" WHERE ts >= ?" if since else "") + " ORDER BY id",
                       (since,) if since else ())
