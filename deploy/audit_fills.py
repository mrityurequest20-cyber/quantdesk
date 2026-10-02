#!/usr/bin/env python3
"""Check every paper fill against the real quotes the desk recorded: was each fill a price the market
actually showed? For each fill, the last option-chain snapshot at or before it gives that contract's
bid and ask; a buy should be near the ask, a sell near the bid. Each closed trade also gets a "quoted"
P&L: its fills replaced by the recorded bid/ask (plus the broker's one-tick slippage) with the same costs.

The paper broker marks between snapshots with a model; this is the check that the model didn't drift
from the market (29 Sep 2026: exits marked at NSE's printed IV instead of the quote's own IV).

    python deploy/audit_fills.py --runtime runtime --data <dir with YYYY-MM-DD/chains/*.csv> [--date 2026-09-29]

audit.yml runs it on GitHub, where the session-data artifacts (the chain snapshots) can be downloaded."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
from pathlib import Path

import pandas as pd

from quantdesk.config import DEFAULT_CONFIG, Config
from quantdesk.core.types import Instrument
from quantdesk.intraday.recorder import SessionRecorder
from quantdesk.intraday.sim import IntradayBroker

IST = "Asia/Kolkata"
TICK = 0.05


def snapshots(data_dirs: list[Path], day: dt.date) -> dict[str, list[pd.DataFrame]]:
    out: dict[str, list] = {}
    for d in data_dirs:
        for u, chs in SessionRecorder(d).load_chains(day).items():
            out.setdefault(u, []).extend(chs)
    for u in out:
        out[u].sort(key=lambda c: pd.Timestamp(c.attrs["ts"]))
    return out


def quote_at(chains: list[pd.DataFrame], expiry: str, strike: float, right: str, ts: pd.Timestamp):
    """(bid, ask, snapshot time) of a contract in the last snapshot at or before `ts` for that expiry."""
    best = None
    for c in chains:
        if str(c.attrs["expiry"]) != str(expiry) or pd.Timestamp(c.attrs["ts"]) > ts:
            continue
        best = c
    if best is None or strike not in best.index:
        return None
    side = right.lower()
    bid, ask = float(best.at[strike, f"{side}_bid"]), float(best.at[strike, f"{side}_ask"])
    if not (bid > 0 and ask >= bid):
        return None
    return bid, ask, pd.Timestamp(best.attrs["ts"])


def audit(runtime: Path, data_dirs: list[Path], only: dt.date | None = None) -> list[dict]:
    cfg = Config.load(DEFAULT_CONFIG)
    costs = IntradayBroker(cfg).costs
    con = sqlite3.connect(runtime / "intraday" / "journal.db")
    trades = pd.read_sql("SELECT id, symbol, strategy, opened_at, closed_at, pnl, legs, exit_note FROM trades "
                         "WHERE status='closed' ORDER BY opened_at", con)
    fills = pd.read_sql("SELECT trade_id, ts, symbol, qty, price, fees FROM fills ORDER BY id", con)
    out = []
    for t in trades.itertuples():
        day = pd.Timestamp(t.opened_at).date()
        if only and day != only:
            continue
        chains = snapshots(data_dirs, day).get(t.symbol, [])
        legs = {l["instrument"]["symbol"]: l["instrument"] for l in json.loads(t.legs)}
        rows, quoted_gross, quoted_fees, complete = [], 0.0, 0.0, bool(chains)
        for f in fills[fills.trade_id == t.id].itertuples():
            inst = legs[f.symbol]
            ts = pd.Timestamp(f.ts)
            q = quote_at(chains, inst["expiry"], float(inst["strike"]), inst["right"], ts)
            if q is None:
                complete = False
                rows.append({"fill": f"{ts:%H:%M} {f.symbol} {f.qty:+d} @ {f.price:,.2f}", "quote": "no snapshot"})
                continue
            bid, ask, at = q
            fair = ask + TICK if f.qty > 0 else bid - TICK         # what the broker should have filled at
            off = (fair - f.price) if f.qty > 0 else (f.price - fair)   # + = the paper fill was better than the market
            age = (ts - at).total_seconds() / 60
            rows.append({"fill": f"{ts:%H:%M} {f.symbol} {f.qty:+d} @ {f.price:,.2f}",
                         "quote": f"{bid:,.2f} / {ask:,.2f} ({age:.0f} min old)", "fair": round(fair, 2),
                         "better_than_market": round(off * abs(f.qty), 2)})
            quoted_gross -= fair * f.qty
            quoted_fees += costs.fees(Instrument.from_dict(inst), int(f.qty), fair)[0]
        out.append({"trade": t.id, "symbol": t.symbol, "strategy": t.strategy, "opened": str(t.opened_at)[:16],
                    "pnl_booked": round(float(t.pnl), 2),
                    "pnl_at_quotes": round(quoted_gross - quoted_fees, 2) if complete else None,
                    "fills": rows, "restated": "Restated" in (t.exit_note or "")})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runtime", default="runtime")
    ap.add_argument("--data", action="append", default=[], help="folder(s) holding YYYY-MM-DD/chains/ (repeatable)")
    ap.add_argument("--date")
    ap.add_argument("--json")
    a = ap.parse_args()
    dirs = [Path(d) for d in a.data] or [Path(a.runtime) / "intraday" / "data"]
    res = audit(Path(a.runtime), dirs, dt.date.fromisoformat(a.date) if a.date else None)
    for r in res:
        print(json.dumps(r, ensure_ascii=False))
    if a.json:
        Path(a.json).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in res), encoding="utf-8")


if __name__ == "__main__":
    main()
