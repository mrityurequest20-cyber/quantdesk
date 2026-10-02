"""Routine checks — the desk's pre-flight, in-flight and post-flight checklists.

Each check returns PASS / WARN / FAIL with a one-line detail and is written to the
journal. A FAIL in the pre-market routine blocks new entries for the day (exits and
stops still run: never block risk-reducing actions)."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.validation import audit

PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


def _res(name, ok, detail, warn=False) -> CheckResult:
    return CheckResult(name, PASS if ok else (WARN if warn else FAIL), detail)


class Routines:
    def __init__(self, cfg, engine, data: dict[str, pd.DataFrame], journal):
        self.cfg, self.engine, self.data, self.journal = cfg, engine, data, journal

    # ---- pre-market -----------------------------------------------------------------------------
    def premarket(self, today: dt.date) -> list[CheckResult]:
        cal, eng, cfg = self.engine.calendar, self.engine, self.cfg
        out = []
        out.append(CheckResult("trading_day", PASS if cal.is_trading_day(today) else WARN,
                               "NSE open today" if cal.is_trading_day(today) else f"{today} is not a trading day"))
        prev = cal.prev_trading_day(today)
        stale = []
        issues = []
        for s, df in self.data.items():
            last = df.index[-1].date()
            if (prev - last).days > cfg.get("checks.max_data_staleness_days", 4):
                stale.append(f"{s}@{last}")
            lim = 0.6 if cfg.instrument_spec(s).get("kind") == "vol_index" else cfg.get("checks.max_abs_return", 0.2)
            issues += [i for i in audit(s, df.tail(60), lim) if i.severity == "WARN"]
        out.append(_res("data_freshness", not stale, "all series current" if not stale else "stale: " + ", ".join(stale)))
        out.append(_res("data_quality", not issues, "no anomalies in the last 60 bars" if not issues else
                        "; ".join(f"{i.symbol}: {i.message}" for i in issues[:5]), warn=True))
        ok, msg = eng.broker.healthcheck()
        out.append(_res("broker", ok, msg))
        kill = cfg.runtime_dir / "KILL"
        out.append(_res("kill_switch", not kill.exists(), "not engaged" if not kill.exists() else f"{kill} present: no orders"))
        eq = eng.curve[-1]["equity"] if eng.curve else eng.broker.cash()
        dd = eng.risk.drawdown(eq) if eng.risk.peak else 0.0
        if eng.risk.halted:
            out.append(CheckResult("risk_state", FAIL, eng.risk.halt_reason))
        else:
            st = WARN if dd >= eng.risk.dd_start else PASS
            out.append(CheckResult("risk_state", st, f"equity ₹{eq:,.0f}, drawdown {dd:.2%}, "
                                                     f"risk scale {eng.risk.dd_scale(eq):.2f}"))
        out.append(self._reconcile())
        out += self._position_checks(today)
        ev = [f"{n} ({d})" for d, n in cfg.events() if 0 <= (d - today).days <= 2]
        out.append(_res("event_risk", not ev, "no scheduled events within 2 days" if not ev else
                        "upcoming: " + ", ".join(ev) + " — short-vol entries blocked", warn=True))
        vix_sym = cfg.get("universe.volatility_index")
        if vix_sym in self.data:
            v = float(self.data[vix_sym]["close"].iloc[-1])
            st = FAIL if v >= cfg.get("checks.vix_extreme", 30) else WARN if v >= cfg.get("checks.vix_warn", 22) else PASS
            out.append(CheckResult("vix", st, f"India VIX {v:.1f}; regime {eng.ctx.regime() if eng.ctx.ts is not None else '?'}"))
        pend = [f"{pe['intent'].strategy}:{pe['intent'].symbol} x{pe['units']}" for pe in eng.pending_entries]
        exits = [f"{tid}:{d.reason}" for tid, d in eng.pending_exits.items()]
        out.append(CheckResult("queued_orders", INFO, f"entries: {', '.join(pend) or 'none'}; exits: {', '.join(exits) or 'none'}"))
        return out

    def _reconcile(self) -> CheckResult:
        """Engine's view of positions must match the broker's, leg by leg."""
        want: dict[str, int] = {}
        for t in self.engine.open_trades:
            for l in t.legs:
                want[l.instrument.symbol] = want.get(l.instrument.symbol, 0) + l.qty
        have = {s: int(p["qty"]) for s, p in self.engine.broker.positions().items()}
        diff = {s: (want.get(s, 0), have.get(s, 0)) for s in set(want) | set(have) if want.get(s, 0) != have.get(s, 0)}
        if not diff:
            return CheckResult("reconciliation", PASS, f"{len(want)} positions match the broker")
        return CheckResult("reconciliation", FAIL, "mismatch (engine, broker): " +
                           ", ".join(f"{s} {a}/{b}" for s, (a, b) in sorted(diff.items())[:6]))

    def _position_checks(self, today: dt.date) -> list[CheckResult]:
        out = []
        no_stop, near_exp = [], []
        for t in self.engine.open_trades:
            if t.kind == "linear" and len(t.legs) == 1 and t.stop is None:
                no_stop.append(t.symbol)
            exp = t.nearest_expiry()
            if exp and (exp - today).days <= 1:
                near_exp.append(f"{t.symbol} {t.meta.get('structure', '')} exp {exp}")
        out.append(_res("stops_in_place", not no_stop, "every linear trade has a stop" if not no_stop else
                        "no stop on: " + ", ".join(no_stop)))
        out.append(_res("expiry_watch", not near_exp, "no option expiring within a day" if not near_exp else
                        "; ".join(near_exp), warn=True))
        return out

    # ---- intraday ------------------------------------------------------------------------------
    def intraday(self, quotes: dict[str, float]) -> list[CheckResult]:
        eng = self.engine
        out = []
        breached, near = [], []
        for t in eng.open_trades:
            if t.kind != "linear" or len(t.legs) != 1 or t.stop is None or t.symbol not in quotes:
                continue
            px, d = quotes[t.symbol], (1 if t.legs[0].qty > 0 else -1)
            atr = eng.bars[t.symbol].at(eng.ctx.ts, "atr") if t.symbol in eng.bars else np.nan
            if (d > 0 and px <= t.stop) or (d < 0 and px >= t.stop):
                breached.append(f"{t.symbol} {px:.2f} vs stop {t.stop:.2f}")
            elif atr == atr and abs(px - t.stop) < 0.5 * atr:
                near.append(f"{t.symbol} within 0.5 ATR of stop")
        out.append(_res("stops", not breached, "no stop breached" if not breached else "BREACHED: " + "; ".join(breached)))
        out.append(_res("stop_proximity", not near, "nothing close to its stop" if not near else "; ".join(near), warn=True))
        if eng.curve:
            eq0 = eng.risk.day_start_equity or eng.curve[-1]["equity"]
            marks = dict(quotes)
            eq = eng.equity(marks)
            day = eq / eq0 - 1
            st = FAIL if day <= -eng.risk.daily_loss_limit else WARN if day <= -eng.risk.daily_loss_limit / 2 else PASS
            out.append(CheckResult("day_pnl", st, f"intraday P&L {day:+.2%} (limit -{eng.risk.daily_loss_limit:.0%})"))
        return out

    # ---- post-market ---------------------------------------------------------------------------
    def postmarket(self, today: dt.date) -> list[CheckResult]:
        eng, out = self.engine, []
        bench = self.cfg.get("universe.benchmark", "NIFTY")
        got = bench in self.data and self.data[bench].index[-1].date() >= today
        out.append(_res("eod_data", got, f"{bench} bar for {today} received" if got else f"no {bench} bar for {today} yet", warn=True))
        out.append(self._reconcile())
        tr = self.journal.trades()
        unreviewed = tr[(tr["status"] == "closed") & tr["grade"].isna()] if not tr.empty else tr
        no_why = tr[(tr["status"] == "open") & (tr["rationale"].fillna("") == "")] if not tr.empty else tr
        out.append(_res("journal_complete", unreviewed.empty and no_why.empty,
                        "every closed trade reviewed, every open trade has a rationale" if unreviewed.empty and no_why.empty
                        else f"{len(unreviewed)} unreviewed, {len(no_why)} without rationale"))
        if eng.curve:
            c = eng.curve[-1]
            out.append(CheckResult("eod_snapshot", PASS, f"equity ₹{c['equity']:,.0f}, DD {c['drawdown']:.2%}, "
                                                         f"{c['open_trades']} open, net Δ ₹{c['net_delta']:,.0f}, "
                                                         f"vega ₹{c['vega']:,.0f}/pt"))
        return out

    def record(self, routine: str, results: list[CheckResult], ts=None) -> None:
        ts = ts or pd.Timestamp.now()
        for r in results:
            self.journal.check(ts, routine, r.name, r.status, r.detail)
        self.journal.commit()

    @staticmethod
    def blocking(results: list[CheckResult]) -> list[CheckResult]:
        hard = {"data_freshness", "broker", "kill_switch", "risk_state", "reconciliation", "vix"}
        return [r for r in results if r.status == FAIL and r.name in hard]

    @staticmethod
    def format(results: list[CheckResult]) -> str:
        icon = {PASS: "✔", WARN: "!", FAIL: "✘", INFO: "·"}
        return "\n".join(f"  {icon.get(r.status, '?')} {r.status:<4} {r.name:<16} {r.detail}" for r in results)
