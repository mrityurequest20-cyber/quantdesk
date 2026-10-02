"""The desk UI: a small JSON API over the paper engine + journal, and a static web app that
charts with the GoCharting SDK (falling back to a built-in chart when the SDK can't load).

Standard library only. Single-threaded on purpose: one SQLite connection, one engine,
requests handled in order. Binds to localhost by default.

  GET  /api/config                      SDK url + license key + universe
  GET  /api/symbols                     watchlist with last price, change, regime
  GET  /api/history?symbol=&from=&to=   UDF bars {s,t,o,h,l,c,v} (t = unix seconds)
  GET  /api/last?symbol=                latest price (tick polling)
  GET  /api/overlays?symbol=            SMA50/200 + open-trade stops/targets (fallback chart)
  GET  /api/markers?symbol=             journal entries/exits for the symbol
  GET  /api/broker?symbol=              GoCharting setBrokerAccounts() payload
  GET  /api/status                      account, positions, queue, checks, recent reviews
  GET  /api/analysis?symbol=            chart + quant + options read (text)
  POST /api/order|close|modify|cancel   trade from the chart (paper account only)
"""
from __future__ import annotations

import json
import logging
import math
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from ..analytics import indicators as ind
from ..engine.live import LiveRunner
from ..reporting.market import analyze_symbol

log = logging.getLogger(__name__)
STATIC = Path(__file__).resolve().parent / "static"
TYPES = {".html": "text/html; charset=utf-8", ".js": "application/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".json": "application/json",
         ".png": "image/png", ".webmanifest": "application/manifest+json"}


def epoch_seconds(idx: pd.DatetimeIndex) -> np.ndarray:
    """Unit-safe (pandas 3 may store us, not ns): daily bars stamped 00:00 UTC of their date."""
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return idx.as_unit("s").asi8.astype(np.int64)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return o


class DeskAPI:
    """Everything the handler needs; separated from HTTP so it is unit-testable."""

    def __init__(self, cfg, runner: LiveRunner):
        self.cfg = cfg
        self.runner = runner
        self.eng = runner.engine                  # the daily desk loads lazily, on first use
        self._loaded_marker = runner.journal.get_state("last_processed")
        self._analysis_cache: dict[str, str] = {}

    def _fresh(self):
        """Reload if a cron EOD run moved the account forward since we loaded."""
        marker = self.runner.journal.get_state("last_processed")
        if self.eng is None or marker != self._loaded_marker:
            self.runner.engine = None
            self.eng = self.runner.load()
            self._loaded_marker = marker
            self._analysis_cache.clear()
        return self.eng

    # ---- symbols / bars --------------------------------------------------------------------------
    def segment(self, sym: str) -> str:
        return "INDEX" if self.cfg.instrument_spec(sym).get("kind") in ("index", "vol_index") else "EQUITY"

    def key(self, sym: str) -> str:
        return f"NSE:{self.segment(sym)}:{sym}"

    def config(self) -> dict:
        g = self.cfg.get("web.gocharting", {}) or {}
        key = os.environ.get("GOCHARTING_LICENSE_KEY") or g.get("license_key", "")
        return {"gocharting": {"enabled": bool(g.get("enabled", True)) and bool(key), "licenseKey": key,
                               "sdkUrl": str(g.get("sdk_url", "")).replace("{key}", key), "theme": g.get("theme", "dark")},
                "symbols": [self.key(s) for s in (self.eng.bars if self.eng else self.cfg.all_symbols())], "default": self.key(self.cfg.get("universe.benchmark", "NIFTY")),
                "paperOnly": not getattr(self.runner.broker, "live", False)}

    def symbols(self) -> list[dict]:
        eng = self._fresh()
        out = []
        for s, t in eng.bars.items():
            df = t.df
            if len(df) < 2:
                continue
            spec = self.cfg.instrument_spec(s)
            reg = eng.aux["regime"].get(s)
            out.append({"key": self.key(s), "symbol": s, "exchange": "NSE", "segment": self.segment(s),
                        "description": s, "last": df["close"].iloc[-1], "change": df["close"].iloc[-1] / df["close"].iloc[-2] - 1,
                        "date": str(df.index[-1].date()), "lot_size": int(spec.get("lot_size", 1)),
                        "tradeable": spec.get("kind") != "vol_index",
                        "regime": str(reg["label"].iloc[-1]) if reg is not None else None})
        return out

    def _sym(self, s: str) -> str:
        s = (s or "").split(":")[-1].upper()
        if s not in self._fresh().bars:
            raise KeyError(f"unknown symbol {s}")
        return s

    def history(self, symbol, frm=None, to=None, countback=None) -> dict:
        eng = self._fresh()
        df = eng.bars[self._sym(symbol)].df
        t = epoch_seconds(df.index)
        upto = t <= int(float(to)) if to is not None else np.ones(len(t), dtype=bool)
        if frm is not None:
            idx = np.where(upto & (t >= int(float(frm))))[0]
            if countback and len(idx) < int(countback):          # at least countBack bars ending at `to`
                idx = np.where(upto)[0][-int(countback):]
        elif countback:
            idx = np.where(upto)[0][-int(countback):]
        else:
            idx = np.where(upto)[0]
        if not len(idx):
            return {"s": "no_data", "nextTime": None}
        d = df.iloc[idx]
        return {"s": "ok", "t": t[idx].tolist(), "o": d["open"].round(4).tolist(), "h": d["high"].round(4).tolist(),
                "l": d["low"].round(4).tolist(), "c": d["close"].round(4).tolist(), "v": d["volume"].fillna(0).round(0).tolist()}

    def last(self, symbol) -> dict:
        df = self._fresh().bars[self._sym(symbol)].df
        return {"price": df["close"].iloc[-1], "time": int(df.index[-1].tz_localize("UTC").timestamp())}

    def overlays(self, symbol) -> dict:
        eng = self._fresh()
        s = self._sym(symbol)
        c = eng.bars[s].df["close"]
        lines = {"SMA50": ind.sma(c, 50), "SMA200": ind.sma(c, 200)}
        t = epoch_seconds(c.index).tolist()
        levels = []
        for tr in eng.open_trades:
            if tr.symbol != s:
                continue
            if tr.stop is not None:
                levels.append({"price": tr.stop, "kind": "stop", "label": f"stop {tr.stop:,.2f} · {tr.strategy}"})
            if tr.target is not None:
                levels.append({"price": tr.target, "kind": "target", "label": f"target {tr.target:,.2f} · {tr.strategy}"})
            levels.append({"price": tr.entry_underlying, "kind": "entry", "label": f"entry {tr.entry_underlying:,.2f} · {tr.strategy}"})
        return {"t": t, "lines": {k: [None if not np.isfinite(v) else round(float(v), 4) for v in ser] for k, ser in lines.items()},
                "levels": levels}

    def markers(self, symbol) -> list[dict]:
        s = self._sym(symbol)
        tr = self.runner.journal.df("SELECT * FROM trades WHERE symbol=? ORDER BY opened_at", (s,))
        out = []
        for r in tr.itertuples():
            side = "buy" if r.direction >= 0 else "sell"
            out.append({"time": str(r.opened_at)[:10], "kind": "entry", "side": side, "price": r.entry_underlying,
                        "trade_id": r.id, "text": f"{r.strategy} {side} — {r.rationale[:160] if r.rationale else ''}"})
            if r.status == "closed" and r.closed_at:
                out.append({"time": str(r.closed_at)[:10], "kind": "exit", "side": side, "price": r.exit_underlying,
                            "trade_id": r.id, "text": f"{r.strategy} exit ({r.exit_reason}) P&L ₹{r.pnl:,.0f}, "
                                                      f"{r.r_multiple:+.2f}R, grade {r.grade}"})
        return out

    # ---- account ------------------------------------------------------------------------------------
    def _security(self, sym: str, lot: int) -> dict:
        return {"symbol": sym, "exchange": "NSE", "segment": self.segment(sym), "tick_size": 0.05, "lot_size": lot,
                "quote_currency": "INR"}

    def broker(self, symbol: str | None = None) -> dict:
        """Shape expected by GoCharting's chartInstance.setBrokerAccounts()."""
        eng = self._fresh()
        s = self._sym(symbol) if symbol else None
        eq = eng.curve[-1]["equity"] if eng.curve and eng.curve[-1].get("equity") else self.runner.broker.cash()
        positions, orders, trades = [], [], []
        for t in eng.open_trades:
            for i, l in enumerate(t.legs):
                und = l.instrument.underlying
                if s and und != s:
                    continue
                last = t.last_mark.get(l.instrument.symbol, l.entry_price)
                pid = t.id if len(t.legs) == 1 else f"{t.id}:{i}"
                positions.append({
                    "id": pid, "productId": self.key(und), "symbol": und, "size": l.qty, "price": l.entry_price,
                    "side": "buy" if l.qty > 0 else "sell", "pnl": l.qty * (last - l.entry_price), "unPnl": l.qty * (last - l.entry_price),
                    "rPnl": 0.0, "exchange": "NSE", "broker": "quantdesk", "productType": l.instrument.kind.upper(),
                    "underlying": und, "security": self._security(und, l.instrument.lot_size), "key": f"qd-{pid}",
                    "showStopLossButton": len(t.legs) == 1, "showTakeProfitButton": len(t.legs) == 1,
                    "stopLoss": t.stop, "takeProfit": t.target, "pnlMultiplier": 1,
                    "label": f"{t.strategy} · {l.instrument.symbol}"})
        for pe in eng.pending_entries:
            it = pe["intent"]
            if s and it.symbol != s:
                continue
            key = f"Q-{it.strategy}-{it.symbol}"
            size = sum(abs(l.ratio) * pe["units"] * l.instrument.lot_size for l in it.legs[:1])
            orders.append({"orderId": key, "productId": self.key(it.symbol), "symbol": it.symbol, "side": "buy" if it.direction >= 0 else "sell",
                           "orderType": "market", "size": size, "remainingSize": size, "price": it.entry_ref, "status": "open",
                           "exchange": "NSE", "broker": "quantdesk", "security": self._security(it.symbol, it.legs[0].instrument.lot_size),
                           "key": f"qd-{key}", "validity": "DAY", "stopLoss": it.stop, "takeProfit": it.target,
                           "productType": it.legs[0].instrument.kind.upper(), "filledSize": 0,
                           "label": f"{it.strategy}: fills at next open"})
        fills = self.runner.journal.df("SELECT * FROM fills ORDER BY id DESC LIMIT 200")
        for f in fills.itertuples():
            und = f.symbol.split("-")[0]
            if s and und != s and not f.symbol.startswith(s):
                continue
            trades.append({"tradeId": str(f.id), "orderId": f.trade_id, "productId": self.key(und) if und in eng.bars else f.symbol,
                           "side": "buy" if f.qty > 0 else "sell", "price": f.price, "tradeSize": abs(f.qty), "status": "filled",
                           "broker": "quantdesk", "time": f.ts, "security": {"symbol": f.symbol, "exchange": "NSE"}})
        acct = {"account_id": "QD-PAPER", "AccountID": "QD-PAPER", "AccountType": "Paper", "label": "QuantDesk paper account",
                "currency": "INR", "balance": self.runner.broker.cash(), "equity": eq,
                "margin": sum(t.meta.get("max_loss", 0) * t.units for t in eng.open_trades if t.kind == "options"),
                "freeMargin": self.runner.broker.cash()}
        return {"accountList": [acct], "orderBook": orders, "tradeBook": trades, "positions": positions}

    def status(self) -> dict:
        eng = self._fresh()
        j = self.runner.journal
        eq = eng.curve[-1]["equity"] if eng.curve and eng.curve[-1].get("equity") else self.runner.broker.cash()
        recent = j.df("SELECT id, strategy, symbol, closed_at, exit_reason, pnl, r_multiple, grade, review, lessons "
                      "FROM trades WHERE status='closed' ORDER BY closed_at DESC, rowid DESC LIMIT 12")
        checks = j.df("SELECT ts, routine, name, status, detail FROM checks ORDER BY id DESC LIMIT 40")
        return {
            "equity": eq, "cash": self.runner.broker.cash(), "peak": eng.risk.peak,
            "drawdown": eng.risk.drawdown(eq) if eng.risk.peak else 0.0, "halted": eng.risk.halted,
            "halt_reason": eng.risk.halt_reason, "kill_switch": (self.cfg.runtime_dir / "KILL").exists(),
            "last_processed": j.get_state("last_processed"), "regime": eng.ctx.regime() if eng.ctx.ts is not None else None,
            "open": [{"id": t.id, "strategy": t.strategy, "symbol": t.symbol, "kind": t.kind, "opened": str(t.opened_at)[:10],
                      "pnl": t.pnl, "stop": t.stop, "target": t.target, "structure": t.meta.get("structure"),
                      "rationale": t.rationale} for t in eng.open_trades],
            "queued": [{"key": f"Q-{pe['intent'].strategy}-{pe['intent'].symbol}", "strategy": pe["intent"].strategy,
                        "symbol": pe["intent"].symbol, "units": pe["units"], "rationale": pe["intent"].rationale}
                       for pe in eng.pending_entries],
            "recent": recent.to_dict("records"), "checks": checks.to_dict("records"),
        }

    def analysis(self, symbol) -> dict:
        s = self._sym(symbol)
        eng = self._fresh()
        if s not in self._analysis_cache:
            try:
                self._analysis_cache[s] = analyze_symbol(self.cfg, s, {k: t.df for k, t in eng.bars.items()}, eng.aux)
            except Exception as exc:  # thin histories (e.g. INDIAVIX) can't support every statistic
                self._analysis_cache[s] = f"analysis unavailable for {s}: {exc}"
        return {"symbol": s, "text": self._analysis_cache[s]}

    # ---- trading from the chart --------------------------------------------------------------------
    def order(self, body: dict) -> dict:
        sym = self._sym(body.get("symbol", ""))
        qty = int(float(body.get("quantity") or body.get("size") or 0))
        stop = body.get("stopLoss")
        tgt = body.get("takeProfit")
        res = self.runner.manual_order(sym, str(body.get("side", "buy")), abs(qty),
                                       float(stop) if stop not in (None, "") else None,
                                       float(tgt) if tgt not in (None, "") else None, str(body.get("note", "")))
        self.eng = self.runner.engine
        return res

    def close(self, body: dict) -> dict:
        res = self.runner.manual_close(str(body.get("id") or body.get("positionId") or body.get("trade_id")))
        self.eng = self.runner.engine
        return res

    def modify(self, body: dict) -> dict:
        pid = str(body.get("id") or body.get("positionId") or body.get("trade_id"))
        g = lambda k: float(body[k]) if body.get(k) not in (None, "") else None
        return self.runner.manual_modify(pid, g("stopLoss"), g("takeProfit"))

    def cancel(self, body: dict) -> dict:
        return self.runner.cancel_queued(str(body.get("orderId") or body.get("id")))


def make_handler(api: DeskAPI, iapi=None, token: str | None = None):
    import hmac

    class Handler(BaseHTTPRequestHandler):
        server_version = "QuantDesk/0.2"

        def log_message(self, fmt, *args):
            log.info("%s " + fmt, self.address_string(), *args)

        def _send(self, code: int, body, ctype="application/json", extra_headers=None):
            data = body if isinstance(body, bytes) else json.dumps(_clean(body)).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            for k, v in (extra_headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        # ---- auth: required whenever a token is configured (i.e. when reachable beyond localhost)
        def _authed(self, q) -> tuple[bool, dict]:
            if not token:
                return True, {}
            cookies = dict(c.strip().split("=", 1) for c in (self.headers.get("Cookie") or "").split(";") if "=" in c)
            given = self.headers.get("X-QD-Token") or cookies.get("qd_token") or q.get("token") or ""
            ok = hmac.compare_digest(given.encode(), token.encode())
            extra = {}
            if ok and q.get("token"):
                extra["Set-Cookie"] = f"qd_token={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age=31536000"
            return ok, extra

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[-1] for k, v in parse_qs(u.query).items()}
            if u.path in ("/manifest.webmanifest", "/icon-192.png", "/icon-512.png", "/apple-touch-icon.png"):
                return self._static("app/" + u.path.lstrip("/"))
            ok, extra = self._authed(q)
            if not ok:
                if u.path.startswith("/api/"):
                    return self._send(401, {"error": "token required"})
                return self._static("app/login.html", 401)
            try:
                if u.path in ("/", "/index.html", "/app"):
                    return self._static("app/index.html", extra_headers=extra)
                if u.path in ("/daily", "/daily.html"):
                    return self._static("index.html", extra_headers=extra)
                if u.path.startswith("/static/"):
                    return self._static(u.path[len("/static/"):])
                acct = q.get("account")
                routes = {
                    "/api/config": lambda: api.config(), "/api/symbols": lambda: api.symbols(),
                    "/api/history": lambda: api.history(q.get("symbol"), q.get("from"), q.get("to"), q.get("countback")),
                    "/api/last": lambda: api.last(q.get("symbol")), "/api/overlays": lambda: api.overlays(q.get("symbol")),
                    "/api/markers": lambda: api.markers(q.get("symbol")), "/api/broker": lambda: api.broker(q.get("symbol")),
                    "/api/status": lambda: api.status(), "/api/analysis": lambda: api.analysis(q.get("symbol")),
                }
                if iapi is not None:
                    routes.update({
                        "/api/i/accounts": lambda: iapi.accounts(), "/api/i/state": lambda: iapi.state(acct),
                        "/api/i/thoughts": lambda: iapi.thoughts(acct, q.get("symbol"), q.get("n", 40), q.get("before")),
                        "/api/i/trades": lambda: iapi.trades(acct, q.get("n", 100)),
                        "/api/i/news": lambda: iapi.news(acct, q.get("n", 120)),
                        "/api/i/trade": lambda: iapi.trade(acct, q.get("id")),
                        "/api/i/reviews": lambda: iapi.reviews(acct), "/api/i/review": lambda: iapi.review(acct, q.get("date")),
                        "/api/i/stats": lambda: iapi.stats(acct),
                        "/api/i/chart": lambda: iapi.chart(acct, q.get("symbol", "NIFTY"), q.get("date"), q.get("interval", "1m")),
                        "/api/i/udf": lambda: iapi.udf(acct, q.get("symbol", "NIFTY"), q.get("interval", "1m"), q.get("from"),
                                                       q.get("to"), q.get("countback")),
                    })
                if u.path not in routes:
                    return self._send(404, {"error": "not found"})
                return self._send(200, routes[u.path](), extra_headers=extra)
            except KeyError as exc:
                return self._send(404, {"error": str(exc).strip("'")})
            except Exception as exc:  # surface, don't crash the desk
                log.exception("GET %s failed", u.path)
                return self._send(500, {"error": repr(exc)})

        def do_POST(self):
            u = urlparse(self.path)
            q = {k: v[-1] for k, v in parse_qs(u.query).items()}
            ok, _ = self._authed({k: v for k, v in q.items() if k != "token"})
            if not ok:
                return self._send(401, {"error": "token required"})
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                routes = {"/api/order": api.order, "/api/close": api.close, "/api/modify": api.modify, "/api/cancel": api.cancel}
                if iapi is not None:
                    routes["/api/i/command"] = lambda b: iapi.command(q.get("account"), b)
                fn = routes.get(u.path)
                if fn is None:
                    return self._send(404, {"error": "not found"})
                return self._send(200, fn(body))
            except PermissionError as exc:
                return self._send(403, {"error": str(exc)})
            except (KeyError, ValueError) as exc:
                return self._send(400, {"error": str(exc).strip("'")})
            except Exception as exc:
                log.exception("POST %s failed", u.path)
                return self._send(500, {"error": repr(exc)})

        def _static(self, name: str, code: int = 200, extra_headers=None):
            path = (STATIC / name).resolve()
            if STATIC not in path.parents or not path.is_file():
                return self._send(404, {"error": "not found"})
            return self._send(code, path.read_bytes(), TYPES.get(path.suffix, "application/octet-stream"), extra_headers)

    return Handler


def lan_ip() -> str:
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def serve(cfg, runner: LiveRunner, host: str | None = None, port: int | None = None, token: str | None = None) -> None:
    import secrets

    from .intraday_api import IntradayAPI
    api = DeskAPI(cfg, runner)
    iapi = IntradayAPI(cfg)
    host = host or cfg.get("web.host", "127.0.0.1")
    port = int(port or cfg.get("web.port", 8765))
    local = host in ("127.0.0.1", "localhost", "::1")
    token = token or os.environ.get("QUANTDESK_TOKEN") or (None if local else secrets.token_urlsafe(18))
    httpd = HTTPServer((host, port), make_handler(api, iapi, token))
    if local:
        print(f"QuantDesk on http://127.0.0.1:{port}  (this machine only; use --host 0.0.0.0 for your phone)")
    else:
        ip = lan_ip() if host in ("0.0.0.0", "::") else host
        print(f"QuantDesk on http://{ip}:{port}/?token={token}\n"
              f"Open that link once on your phone (same Wi-Fi, or over Tailscale); it remembers the token.\n"
              f"Keep the token private: anyone with it can pause the desk or close paper positions.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
