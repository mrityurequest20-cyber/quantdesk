"""Automatic post-trade review and periodic (daily / weekly) journal reviews.

The grading deliberately weights *process* over *outcome* (60/40): a planned stop-out
for -1R is a good trade; a lucky +2R that ignored the plan is not. That's the habit a
journal is supposed to build."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

PLANNED_EXITS = {"stop", "target", "trend_reversal", "channel_exit", "mean_reverted", "time_stop", "rebalance_out",
                 "abs_momentum_off", "converged", "z_stop", "take_profit", "stop_loss", "time_exit", "expiry",
                 "short_strike_breached", "regime_stress", "cointegration_broke",
                 # intraday options exits
                 "invalidation", "premium_stop", "premium_target", "underlying_target", "breakeven_stop", "square_off",
                 "range_break"}
FORCED_EXITS = {"risk_halt", "manual", "data_missing", "end_of_backtest"}


def review_trade(t, regime_at_exit: str | None = None) -> dict:
    R = t.r_multiple
    risk = t.initial_risk if t.initial_risk > 0 else max(abs(t.pnl), 1.0)
    mfe_r, mae_r = t.mfe / risk, -t.mae / risk
    capture = t.pnl / t.mfe if t.mfe > 0 else (1.0 if t.pnl >= 0 else 0.0)
    lessons: list[str] = []

    # ---- process score -------------------------------------------------------------------------
    process = 50.0
    if t.exit_reason in PLANNED_EXITS or (t.strategy == "manual" and t.exit_reason == "manual"):
        process += 25
    elif t.exit_reason in FORCED_EXITS:
        process -= 10
        lessons.append(f"Exit was forced ({t.exit_reason}), not part of the trade plan.")
    if t.kind == "linear" and t.stop is not None:
        process += 10                                   # had a hard stop from the first bar
    if mae_r <= 0.5:
        process += 15                                   # entry timing: little heat taken
    elif mae_r > 0.9 and t.exit_reason != "stop":
        process -= 10
        lessons.append(f"Took {mae_r:.0%} of planned risk as heat before the exit; entry timing was poor.")
    process = float(np.clip(process, 0, 100))

    # ---- outcome score ---------------------------------------------------------------------------
    outcome = float(np.clip(50 + 25 * R, 0, 100))

    # ---- lessons ------------------------------------------------------------------------------------
    if mfe_r >= 1.0 and t.pnl <= 0:
        lessons.append(f"Round-tripped a {mfe_r:.1f}R open profit into a loss: trail or scale out earlier.")
    elif t.mfe > 0 and mfe_r > 0.5 and capture < 0.35:
        lessons.append(f"Captured only {capture:.0%} of the best open profit ({mfe_r:.1f}R).")
    if t.exit_reason == "stop" and t.bars_held <= 2:
        lessons.append("Stopped out within 2 bars: the stop may sit inside normal noise (check the ATR multiple).")
    if t.exit_reason in ("time_stop", "time_exit") and abs(R) < 0.3:
        lessons.append("Thesis did not play out in the allotted time: dead money, consider tighter entry filters.")
    if t.exit_reason == "short_strike_breached":
        lessons.append("Short strike breached: consider wider short deltas or adjusting when a short's delta doubles.")
    if t.exit_reason == "z_stop":
        lessons.append("Spread kept diverging: the cointegration relationship may be breaking; check news/fundamentals.")
    entry_regime = (t.context or {}).get("regime")
    if regime_at_exit and entry_regime and regime_at_exit != entry_regime:
        lessons.append(f"Regime changed during the trade ({entry_regime} → {regime_at_exit}).")
    if R >= 2:
        lessons.append(f"Big winner ({R:.1f}R): note what the setup looked like and look for more of it.")
    fee_share = t.fees / max(abs(t.pnl + t.fees), 1e-9) if t.pnl + t.fees > 0 else None
    if fee_share is not None and fee_share > 0.3:
        lessons.append(f"Costs ate {fee_share:.0%} of the gross profit: trade size or frequency is too small for this edge.")

    score = 0.6 * process + 0.4 * outcome
    grade = "A" if score >= 80 else "B" if score >= 65 else "C" if score >= 50 else "D" if score >= 35 else "F"
    cap_txt = f", captured {capture:.0%} of it" if t.pnl > 0 and mfe_r > 0.1 else ""
    summary = (f"{t.strategy} {t.symbol}: {t.exit_reason} after {t.bars_held} bars, P&L ₹{t.pnl:,.0f} ({R:+.2f}R); "
               f"best open profit {mfe_r:.2f}R{cap_txt}, worst heat {mae_r:.2f}R. Grade {grade} "
               f"(process {process:.0f}, outcome {outcome:.0f}).")
    return {"grade": grade, "process_score": process, "outcome_score": outcome, "lessons": lessons,
            "summary": summary, "mfe_r": mfe_r, "mae_r": mae_r, "capture": capture}


def period_review(journal, start: str, end: str | None = None, title: str = "Review") -> str:
    """Markdown review of a period: P&L, trades, per-strategy stats, grades, lessons, risk events."""
    end = end or "9999"
    trades = journal.df("SELECT * FROM trades WHERE status='closed' AND closed_at >= ? AND closed_at <= ?", (start, end + " 99"))
    opened = journal.df("SELECT * FROM trades WHERE opened_at >= ? AND opened_at <= ?", (start, end + " 99"))
    eq = journal.df("SELECT * FROM equity WHERE ts >= ? AND ts <= ? ORDER BY ts", (start, end + " 99"))
    ev = journal.df("SELECT * FROM events WHERE ts >= ? AND ts <= ? AND level IN ('WARN','ERROR','CRITICAL') ORDER BY id",
                    (start, end + " 99"))
    ck = journal.df("SELECT * FROM checks WHERE ts >= ? AND ts <= ? AND status IN ('WARN','FAIL') ORDER BY id", (start, end + " 99"))
    open_now = journal.trades("open")
    L = [f"# {title}: {start}" + (f" → {end}" if end != "9999" else ""), ""]
    if not eq.empty:
        e0, e1 = eq["equity"].iloc[0], eq["equity"].iloc[-1]
        L += [f"**Equity** ₹{e1:,.0f} ({e1 / e0 - 1:+.2%} over the period), drawdown {eq['drawdown'].iloc[-1]:.2%}, "
              f"regime **{eq['regime'].iloc[-1]}**.", ""]
    L += [f"**Trades:** {len(opened)} opened, {len(trades)} closed, {len(open_now)} open now.", ""]
    if not trades.empty:
        wins = (trades["pnl"] > 0).mean()
        L += [f"Closed P&L ₹{trades['pnl'].sum():,.0f}, win rate {wins:.0%}, avg {trades['r_multiple'].mean():+.2f}R, "
              f"fees ₹{trades['fees'].sum():,.0f}.", "", "| Strategy | Trades | Win % | Avg R | P&L ₹ |", "|---|---:|---:|---:|---:|"]
        for s, g in trades.groupby("strategy"):
            L.append(f"| {s} | {len(g)} | {(g['pnl'] > 0).mean():.0%} | {g['r_multiple'].mean():+.2f} | {g['pnl'].sum():,.0f} |")
        L += ["", "**Grades:** " + ", ".join(f"{k}: {v}" for k, v in sorted(Counter(trades["grade"]).items())), ""]
        best, worst = trades.loc[trades["pnl"].idxmax()], trades.loc[trades["pnl"].idxmin()]
        L += [f"Best: {best['strategy']} {best['symbol']} ₹{best['pnl']:,.0f} ({best['r_multiple']:+.2f}R, {best['exit_reason']})",
              f"Worst: {worst['strategy']} {worst['symbol']} ₹{worst['pnl']:,.0f} ({worst['r_multiple']:+.2f}R, {worst['exit_reason']})", ""]
        lessons = Counter(l.split(":")[0] for ls in trades["lessons"].dropna() for l in json.loads(ls))
        if lessons:
            L += ["**Recurring lessons:**"] + [f"- {k} (x{v})" for k, v in lessons.most_common(6)] + [""]
    if not open_now.empty:
        L += ["**Open positions:**"] + [f"- {r.strategy} {r.symbol} since {str(r.opened_at)[:10]}, open P&L ₹{r.pnl:,.0f}, "
                                        f"stop {f'{r.stop:,.2f}' if r.stop is not None and r.stop == r.stop else '—'}"
                                        for r in open_now.itertuples()] + [""]
    if not ev.empty:
        L += ["**Risk / system events:**"] + [f"- {r.ts[:10]} [{r.level}] {r.category}: {r.message}" for r in ev.tail(12).itertuples()] + [""]
    if not ck.empty:
        L += ["**Checks not passing:**"] + [f"- {r.ts[:16]} {r.routine}/{r.name}: {r.status} — {r.detail}" for r in ck.tail(12).itertuples()] + [""]
    return "\n".join(L)
