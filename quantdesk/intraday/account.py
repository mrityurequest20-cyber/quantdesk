"""The paper account's identity: its starting capital and when it started. The capital in the
config is only a *request*; the account's real balance lives in broker.json and the journal.

* A new account, or one that hasn't traded yet, simply takes the configured capital.
* Once it has trades, changing the config does nothing on its own (history would stop adding
  up); `quantdesk intraday reset-account` archives the old account and starts a fresh one.
"""
from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path

import pandas as pd

from ..journal.journal import Journal

IST = "Asia/Kolkata"


def ensure_account(cfg, journal: Journal, broker_path: Path, say=print) -> float:
    """Returns the account's starting capital, (re)initialising it when that's safe."""
    cap = float(cfg.get("intraday.capital", 20000))
    acct = journal.get_state("intraday_account")
    n_trades = int(journal.df("SELECT COUNT(*) AS n FROM trades")["n"].iloc[0])
    now = pd.Timestamp.now(tz=IST)
    if n_trades == 0 and (acct is None or float(acct.get("capital", 0)) != cap):
        if Path(broker_path).exists():
            Path(broker_path).unlink()
        journal.set_state("intraday_account", {"capital": cap, "since": str(now.date())})
        if acct is not None:
            journal.event(now, "WARN", "account", f"no trades yet: account reset from ₹{acct['capital']:,.0f} to ₹{cap:,.0f}")
            say(f"account reset to ₹{cap:,.0f} (it had no trades)")
        journal.commit()
        return cap
    if acct is None:                                   # an older account that predates this record
        acct = {"capital": cap, "since": None}
        journal.set_state("intraday_account", acct)
        journal.commit()
    if float(acct["capital"]) != cap:
        say(f"note: config capital ₹{cap:,.0f} but this account started with ₹{acct['capital']:,.0f} and has "
            f"{n_trades} trades; run `quantdesk intraday reset-account` to start over at ₹{cap:,.0f}")
    return float(acct["capital"])


def reset_account(base: Path) -> Path | None:
    """Move journal, broker, reviews and state aside (runtime/intraday/archive/<stamp>/)."""
    base = Path(base)
    items = [base / n for n in ("journal.db", "broker.json", "reviews") if (base / n).exists()]
    if not items:
        return None
    dest = base / "archive" / pd.Timestamp.now(tz=IST).strftime("%Y-%m-%d_%H%M%S")
    dest.mkdir(parents=True, exist_ok=True)
    for it in items:
        shutil.move(str(it), str(dest / it.name))
    return dest


def restate_trade(cfg, base: Path, trade_id: str, exits: dict[str, float], reason: str) -> dict:
    """Correct a closed paper trade's exit prices after the fact, on the record.

    For when the paper broker's price was wrong (29 Sep 2026: an exit marked at NSE's printed IV instead
    of the IV the quote implied). The exit fills get the new prices and the fees those prices imply; the
    trade's P&L, fees and R follow; paper cash moves by the difference; the day's heartbeat and resume
    state follow too. Nothing is deleted: the old prices go into the trade's exit note and review, and
    an event records the restatement with both versions."""
    import json

    from ..core.types import Instrument
    from .sim import IntradayBroker
    j = Journal(base / "journal.db")
    rows = j.df("SELECT * FROM trades WHERE id=?", (trade_id,))
    if rows.empty:
        raise KeyError(f"no trade {trade_id}")
    t = rows.iloc[0]
    if t["status"] != "closed":
        raise ValueError(f"trade {trade_id} is {t['status']}: only a closed trade can be restated")
    legs = json.loads(t["legs"])
    unknown = set(exits) - {l["instrument"]["symbol"] for l in legs}
    if unknown:
        raise KeyError(f"no leg {', '.join(sorted(unknown))} in trade {trade_id}")
    fills = j.df("SELECT * FROM fills WHERE trade_id=? ORDER BY id", (trade_id,))
    costs = IntradayBroker(cfg).costs
    old_pnl, old_fees = float(t["pnl"]), float(t["fees"])
    fee_delta, breakdown_delta, changes = 0.0, {}, []
    for leg in legs:
        sym = leg["instrument"]["symbol"]
        if sym not in exits:
            continue
        new_px = float(exits[sym])
        closing = fills[(fills["symbol"] == sym) & (fills["qty"] == -int(leg["qty"]))]
        if closing.empty:
            raise ValueError(f"no exit fill for {sym}")
        f = closing.iloc[-1]
        inst = Instrument.from_dict(leg["instrument"])
        fee, bd = costs.fees(inst, int(f["qty"]), new_px)
        old_bd = json.loads(f["breakdown"] or "{}")
        for k in set(bd) | set(old_bd):
            breakdown_delta[k] = breakdown_delta.get(k, 0.0) + bd.get(k, 0.0) - old_bd.get(k, 0.0)
        fee_delta += fee - float(f["fees"])
        changes.append({"symbol": sym, "qty": int(f["qty"]), "old_price": float(f["price"]), "new_price": new_px,
                        "old_fees": float(f["fees"]), "new_fees": fee})
        j._exec("UPDATE fills SET price=?, fees=?, breakdown=? WHERE id=?", (new_px, fee, json.dumps(bd), int(f["id"])))
        leg["exit_price"] = new_px
    gross = sum((float(l["exit_price"]) - float(l["entry_price"])) * int(l["qty"]) for l in legs)
    new_fees = old_fees + fee_delta
    new_pnl = gross - new_fees
    delta = new_pnl - old_pnl
    risk = float(t["initial_risk"] or 0)
    new_r = new_pnl / risk if risk else None
    now = pd.Timestamp.now(tz=IST)
    moved = ", ".join(f"{c['symbol']} ₹{c['old_price']:,.2f} → ₹{c['new_price']:,.2f}" for c in changes)
    stamp = f" [Restated {now:%d %b %Y}: exit {moved}; P&L ₹{old_pnl:+,.0f} → ₹{new_pnl:+,.0f}. {reason}]"
    j._exec("UPDATE trades SET pnl=?, fees=?, r_multiple=?, legs=?, exit_note=?, review=? WHERE id=?",
            (new_pnl, new_fees, new_r, json.dumps(legs), (t["exit_note"] or "") + stamp, (t["review"] or "") + stamp, trade_id))
    # paper cash and the fee ledger move by the difference
    bp = base / "broker.json"
    if bp.exists():
        b = json.loads(bp.read_text())
        b["cash"] = float(b["cash"]) + delta
        b["fees_paid"] = float(b.get("fees_paid", 0.0)) + fee_delta
        fb = b.setdefault("fee_breakdown", {})
        for k, v in breakdown_delta.items():
            fb[k] = float(fb.get(k, 0.0)) + v
        bp.write_text(json.dumps(b, indent=2))
    # the last heartbeat (what the app shows) and the session's resume state
    day = str(t["opened_at"])[:10]
    hb = j.get_state("intraday_live")
    if hb:
        hb["equity"] = float(hb.get("equity", 0.0)) + delta
        if hb.get("day") == day:
            hb["day_pnl"] = float(hb.get("day_pnl", 0.0)) + delta
        j.set_state("intraday_live", hb)
    st = j.get_state("intraday_open")
    if st:
        for d in st.get("closed") or []:
            if d.get("id") == trade_id:
                d.update({"pnl": new_pnl, "fees": new_fees, "r_multiple": new_r, "legs": legs})
        j.set_state("intraday_open", st)
    summary = {"trade": trade_id, "old_pnl": old_pnl, "new_pnl": new_pnl, "delta": delta, "fills": changes, "reason": reason}
    j.event(now, "WARN", "restatement", f"trade {trade_id} restated: P&L ₹{old_pnl:+,.0f} → ₹{new_pnl:+,.0f} ({reason})", summary)
    j.commit()
    return summary
