"""Read-mostly API over the intraday desk's journal(s), for the web/mobile app.

The intraday engine runs as its own process (`quantdesk intraday live`) and writes to its
journal every minute; this side only reads, except for remote-control commands, which it
appends to a queue the engine executes on its next step (pause / resume / close / flatten)."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from ..intraday.feeds import normalise_bars
from ..journal.journal import Journal

COMMANDS = {"pause", "resume", "flatten", "close"}


class IntradayAPI:
    def __init__(self, cfg):
        self.cfg = cfg
        self.base = cfg.runtime_dir / "intraday"
        self._j: dict[str, Journal] = {}

    # ---- accounts / plumbing -------------------------------------------------------------------------
    def _dir(self, account: str) -> Path:
        if account in (None, "", "live"):
            return self.base
        if not account.replace("_", "").replace("-", "").isalnum():
            raise KeyError("bad account name")
        return self.base / account

    def accounts(self) -> list[dict]:
        out = []
        if (self.base / "journal.db").exists():
            out.append({"id": "live", "label": "Live paper"})
        if self.base.exists():
            for p in sorted(self.base.iterdir()):
                if p.is_dir() and (p / "journal.db").exists():
                    out.append({"id": p.name, "label": p.name.capitalize()})
        return out

    def j(self, account: str | None) -> Journal:
        a = account or "live"
        path = self._dir(a) / "journal.db"
        if not path.exists():
            raise KeyError(f"no intraday account {a!r} yet")
        if a not in self._j:
            self._j[a] = Journal(path)
        return self._j[a]

    def _data_dir(self, account: str | None) -> Path:
        d = self._dir(account) / "data"
        return d if d.exists() else self.base / "data"

    def _cash(self, account) -> float | None:
        p = self._dir(account) / "broker.json"
        return json.loads(p.read_text())["cash"] if p.exists() else None

    def capital(self, account) -> float:
        """The account's own starting capital (not whatever the config says today)."""
        acct = self.j(account).get_state("intraday_account") or {}
        return float(acct.get("capital") or self.cfg.get("intraday.capital", 20000))

    def limits(self) -> dict:
        """The risk limits the desk trades under (the app shows how much of each is used today)."""
        r = self.cfg.get("intraday.risk", {}) or {}
        return {k: r.get(k) for k in ("risk_per_trade", "max_trades_per_day", "max_open", "daily_loss_limit")}

    # ---- screens -----------------------------------------------------------------------------------------
    def state(self, account=None) -> dict:
        j = self.j(account)
        hb = j.get_state("intraday_live") or {}
        cash = self._cash(account)
        cap = self.capital(account)
        day = hb.get("day")
        closed = j.df("SELECT id, strategy, symbol, pnl, r_multiple, grade, exit_reason, opened_at, closed_at "
                      "FROM trades WHERE status='closed' AND opened_at >= ? ORDER BY closed_at DESC", (day or "9999",))
        tot = j.df("SELECT COALESCE(SUM(pnl),0) AS pnl, COUNT(*) AS n FROM trades WHERE status='closed'").iloc[0]
        age = None
        if hb.get("ts"):
            age = (pd.Timestamp.now(tz="Asia/Kolkata") - pd.Timestamp(hb["ts"])).total_seconds()
        return {"heartbeat": hb, "age_sec": age, "cash": cash, "capital": cap,
                "equity": hb.get("equity", cash if cash is not None else cap), "total_pnl": float(tot["pnl"]), "total_trades": int(tot["n"]),
                "paused": bool(j.get_state("intraday_paused", False)), "closed_today": closed.to_dict("records"),
                "limits": self.limits(),
                "pending_commands": len([c for c in (j.get_state("intraday_cmds") or [])
                                         if c["id"] not in set(j.get_state("intraday_cmds_done") or [])])}

    def thoughts(self, account=None, symbol=None, n=40, before=None) -> list[dict]:
        j = self.j(account)
        q, p = "SELECT * FROM thoughts WHERE 1=1", []
        if symbol:
            q += " AND symbol = ?"
            p.append(symbol)
        if before:
            q += " AND id < ?"
            p.append(int(before))
        rows = j.df(q + " ORDER BY id DESC LIMIT ?", (*p, int(n)))
        out = []
        for r in rows.to_dict("records"):
            for k in ("evidence", "vetoes", "levels", "chain"):
                r[k] = json.loads(r[k]) if r.get(k) else None
            out.append(r)
        return out

    def news(self, account=None, n=120) -> list[dict]:
        rows = self.j(account).news(n=int(n))
        out = []
        for r in rows.to_dict("records"):
            for k in ("sources", "about"):
                r[k] = json.loads(r[k]) if r.get(k) else None
            out.append(r)
        return out

    def trades(self, account=None, n=100) -> list[dict]:
        t = self.j(account).df("SELECT id, strategy, symbol, direction, units, opened_at, closed_at, pnl, fees, r_multiple, "
                               "grade, exit_reason, status, meta FROM trades ORDER BY opened_at DESC LIMIT ?", (int(n),))
        recs = t.to_dict("records")
        for r in recs:
            m = json.loads(r.pop("meta") or "{}")
            r["structure"], r["expiry"] = m.get("structure"), m.get("expiry")
        return recs

    def trade(self, account=None, tid=None) -> dict:
        t = self.j(account).df("SELECT * FROM trades WHERE id=?", (tid,))
        if t.empty:
            raise KeyError(f"no trade {tid}")
        r = t.iloc[0].to_dict()
        for k in ("legs", "context", "meta", "sizing", "lessons"):
            r[k] = json.loads(r[k]) if r.get(k) else None
        r["fills"] = self.j(account).df("SELECT ts, symbol, qty, price, fees FROM fills WHERE trade_id=? ORDER BY id",
                                        (tid,)).to_dict("records")
        return r

    def reviews(self, account=None) -> list[str]:
        d = self._dir(account) / "reviews"
        return sorted((p.stem for p in d.glob("*.md")), reverse=True) if d.exists() else []

    def review(self, account=None, date=None) -> dict:
        p = self._dir(account) / "reviews" / f"{date}.md"
        if not p.exists() or not date or "/" in date:
            raise KeyError("no such review")
        return {"date": date, "markdown": p.read_text(encoding="utf-8")}

    def stats(self, account=None) -> dict:
        t = self.j(account).trades("closed")
        cap = self.capital(account)
        if t.empty:
            return {"trades": 0, "capital": cap}
        t["day"] = t["opened_at"].str[:10]
        t["structure"] = t["meta"].map(lambda m: json.loads(m).get("structure"))
        t["day_type"] = t["context"].map(lambda c: json.loads(c).get("regime"))
        t["hour"] = t["opened_at"].str[11:13] + ":00"
        daily = t.groupby("day")["pnl"].sum()
        eq = cap + daily.cumsum()
        wins, losses = t[t["pnl"] > 0]["pnl"], t[t["pnl"] <= 0]["pnl"]

        def by(col):
            g = t.groupby(col)
            return [{"key": str(k), "trades": int(len(x)), "win": float((x["pnl"] > 0).mean()), "avg_r": float(x["r_multiple"].mean()),
                     "pnl": float(x["pnl"].sum())} for k, x in g]

        return {"capital": cap, "trades": int(len(t)), "sessions": int(len(daily)), "net": float(t["pnl"].sum()),
                "win_rate": float((t["pnl"] > 0).mean()), "avg_r": float(t["r_multiple"].mean()),
                "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else None,
                "fees": float(t["fees"].sum()), "green_days": float((daily > 0).mean()), "worst_day": float(daily.min()),
                "best_day": float(daily.max()), "max_dd": float((eq / eq.cummax() - 1).min()),
                "equity": [{"day": d, "equity": float(v)} for d, v in eq.items()],
                "by_setup": by("strategy"), "by_structure": by("structure"), "by_day_type": by("day_type"),
                "by_exit": by("exit_reason"), "by_hour": by("hour"), "by_symbol": by("symbol"),
                "grades": {str(k): int(v) for k, v in t["grade"].value_counts().sort_index().items()},
                "calibration": self._calibration(t)}

    @staticmethod
    def _calibration(t: pd.DataFrame) -> list[dict]:
        """Was the probability the desk traded on any good? For trades priced by the quant layer: the P(right
        direction) it assumed vs how often the underlying actually moved its way by the exit."""
        rows = []
        for r in t.itertuples():
            q = (json.loads(r.meta or "{}").get("quant") or {})
            if "p_up" not in q or not r.direction or pd.isna(r.exit_underlying) or pd.isna(r.entry_underlying):
                continue
            p_right = q["p_up"] if r.direction > 0 else 1 - q["p_up"]
            right = (r.exit_underlying - r.entry_underlying) * r.direction > 0
            rows.append((p_right, right, str(q.get("p_source", "?")).split(" (")[0], r.pnl))
        if not rows:
            return []
        df = pd.DataFrame(rows, columns=["p", "right", "source", "pnl"])
        df["bucket"] = pd.cut(df["p"], [0, 0.5, 0.53, 0.56, 0.6, 1.0], labels=["≤50%", "50–53%", "53–56%", "56–60%", ">60%"])
        out = []
        for (src, b), g in df.groupby(["source", "bucket"], observed=True):
            out.append({"source": src, "bucket": str(b), "trades": int(len(g)), "assumed": float(g["p"].mean()),
                        "realised": float(g["right"].mean()), "pnl": float(g["pnl"].sum())})
        return out

    # ---- chart data --------------------------------------------------------------------------------------
    def _bars(self, account, symbol: str, days: int = 1, date: str | None = None) -> pd.DataFrame:
        d = self._data_dir(account)
        if not d.exists():
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        dates = sorted(p.name for p in d.iterdir() if (p / f"{symbol}_1m.csv").exists())
        if date:
            dates = [x for x in dates if x <= date]
        frames = [normalise_bars(pd.read_csv(d / x / f"{symbol}_1m.csv", index_col=0, parse_dates=True)) for x in dates[-days:]]
        return pd.concat(frames) if frames else pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    def chart(self, account=None, symbol="NIFTY", date=None, interval="1m") -> dict:
        symbol = symbol.upper()
        df = self._bars(account, symbol, 1, date)
        if df.empty:
            return {"symbol": symbol, "bars": None}
        day = str(df.index[-1].date())
        if interval in ("3m", "5m", "15m"):
            df = df.resample(interval.replace("m", "min"), label="left", closed="left", origin="start_day",
                             offset="15min").agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                                  "volume": "sum"}).dropna(subset=["close"])
        vol = df["volume"].to_numpy(dtype=float)
        vol = vol if vol.sum() > 0 else np.ones(len(df))
        tp = ((df["high"] + df["low"] + df["close"]) / 3).to_numpy()
        vwap = np.cumsum(tp * vol) / np.cumsum(vol)
        j = self.j(account)
        tr = j.df("SELECT id, strategy, direction, opened_at, closed_at, entry_underlying, exit_underlying, pnl, "
                  "exit_reason, stop, target, meta FROM trades WHERE symbol=? AND opened_at >= ? AND opened_at < ?",
                  (symbol, day, day + " 99"))
        markers = []
        for r in tr.itertuples():
            m = json.loads(r.meta or "{}")
            markers.append({"t": int(pd.Timestamp(r.opened_at).timestamp()), "kind": "entry", "price": r.entry_underlying,
                            "dir": r.direction, "text": f"{r.strategy} · {m.get('structure')}"})
            if r.closed_at:
                markers.append({"t": int(pd.Timestamp(r.closed_at).timestamp()), "kind": "exit", "price": r.exit_underlying,
                                "dir": r.direction, "text": f"{r.exit_reason} ₹{r.pnl:,.0f}"})
        th = j.df("SELECT levels FROM thoughts WHERE symbol=? AND ts >= ? AND ts < ? ORDER BY id DESC LIMIT 1",
                  (symbol, day, day + " 99"))
        levels = json.loads(th.iloc[0]["levels"]) if not th.empty and th.iloc[0]["levels"] else {}
        return {"symbol": symbol, "day": day, "interval": interval,
                "bars": {"t": [int(x.timestamp()) for x in df.index], "o": df["open"].round(2).tolist(),
                         "h": df["high"].round(2).tolist(), "l": df["low"].round(2).tolist(), "c": df["close"].round(2).tolist(),
                         "v": df["volume"].round(0).tolist()},
                "vwap": [round(float(x), 2) for x in vwap], "markers": markers,
                "levels": {k: v for k, v in levels.items() if v is not None}}

    def udf(self, account=None, symbol="NIFTY", interval="1m", frm=None, to=None, countback=None) -> dict:
        """UDF bars for the GoCharting datafeed (intraday resolutions)."""
        df = self._bars(account, symbol.split(":")[-1].upper(), 30)
        if interval in ("3m", "5m", "15m", "30m", "1h"):
            rule = interval.replace("m", "min") if interval.endswith("m") else "60min"
            df = df.resample(rule, label="left", closed="left", origin="start_day", offset="15min").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["close"])
        t = np.array([int(x.timestamp()) for x in df.index], dtype=np.int64)
        m = np.ones(len(t), dtype=bool)
        if to is not None:
            m &= t <= int(float(to))
        if frm is not None:
            idx = np.where(m & (t >= int(float(frm))))[0]
            if countback and len(idx) < int(countback):
                idx = np.where(m)[0][-int(countback):]
        else:
            idx = np.where(m)[0][-int(countback):] if countback else np.where(m)[0]
        if not len(idx):
            return {"s": "no_data", "nextTime": None}
        d = df.iloc[idx]
        return {"s": "ok", "t": t[idx].tolist(), "o": d["open"].tolist(), "h": d["high"].tolist(), "l": d["low"].tolist(),
                "c": d["close"].tolist(), "v": d["volume"].tolist()}

    # ---- remote control ---------------------------------------------------------------------------------
    def command(self, account, body: dict) -> dict:
        cmd = str(body.get("cmd", ""))
        if cmd not in COMMANDS:
            raise ValueError(f"unknown command {cmd!r}; use one of {sorted(COMMANDS)}")
        arg = body.get("arg")
        if cmd == "close" and not arg:
            raise ValueError("close needs the trade id")
        j = self.j(account)
        cmds = j.get_state("intraday_cmds") or []
        c = {"id": uuid.uuid4().hex[:10], "cmd": cmd, "arg": arg, "ts": str(pd.Timestamp.now(tz="Asia/Kolkata"))}
        cmds.append(c)
        j.set_state("intraday_cmds", cmds[-200:])
        if cmd in ("pause", "resume"):                 # reflect immediately even before the engine's next minute
            j.set_state("intraday_paused", cmd == "pause")
        return {"queued": c, "note": "the engine applies it on its next minute"}
