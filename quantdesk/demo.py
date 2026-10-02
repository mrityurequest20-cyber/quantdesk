"""End-to-end offline demo on the synthetic market: backtest + Monte Carlo + tearsheet,
market analysis + scan, then a paper-trading replay of the last N sessions with the full
daily routine (pre-market checks → EOD cycle → post-market checks → journal review)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from .backtest.montecarlo import block_bootstrap, trade_bootstrap
from .backtest.runner import run_backtest
from .data.synthetic import SyntheticProvider
from .engine.engine import Engine
from .engine.live import LiveRunner
from .execution.broker import PaperBroker
from .journal.journal import Journal
from .ops.checks import Routines
from .reporting.html import backtest_report, desk_report
from .reporting.market import analyze_symbol, scan


def _say(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def run_demo(cfg, out: Path, paper_days: int = 15) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.today().normalize()
    prov = SyntheticProvider(cfg, end=today)
    data = prov.universe(cfg.all_symbols(), SyntheticProvider.HORIZON_START)
    bench = data[cfg.get("universe.benchmark", "NIFTY")].index
    paper_start = bench[-paper_days]
    bt_end = bench[-paper_days - 1]

    _say("1/4 building regime + volatility models (GARCH, HMM) …")
    aux = Engine.build_aux(cfg, data)

    _say(f"2/4 backtest {cfg.get('backtest.start')} → {bt_end.date()} …")
    bt_journal = Journal(out / "backtest_journal.db")
    res = run_backtest(cfg, data, start=cfg.get("backtest.start"), end=bt_end, aux=aux, journal=bt_journal,
                       progress=lambda n, N, ts: _say(f"    {ts.date()}  ({n}/{N})"))
    mc = trade_bootstrap(res.trades["pnl"].to_numpy(), cfg.get("account.starting_capital"))
    bb = block_bootstrap(res.equity["equity"].pct_change())
    (out / "backtest_report.html").write_text(backtest_report(res, "QuantDesk backtest (synthetic market)", synthetic=True, mc=mc),
                                              encoding="utf-8")
    res.trades.to_csv(out / "backtest_trades.csv", index=False)
    summary = res.summary()
    summary += (f"\nBlock bootstrap of daily returns: CAGR p5/p50/p95 {bb['cagr_p5']:+.1%} / {bb['cagr_p50']:+.1%} / "
                f"{bb['cagr_p95']:+.1%}; max DD p50 {bb['maxdd_p50']:.1%}, p95 {bb['maxdd_p95']:.1%}; "
                f"P(CAGR<0) {bb['p_negative_cagr']:.1%}")
    (out / "backtest_summary.txt").write_text(summary + "\n\n" + res.per_strategy.round(3).to_string(), encoding="utf-8")
    _say(summary)

    _say("3/4 market analysis …")
    upto = {s: df.loc[:bt_end] for s, df in data.items()}
    analysis = "\n\n".join(analyze_symbol(cfg, s, upto, None) for s in ("NIFTY", "BANKNIFTY", "RELIANCE"))
    sc = scan(cfg, upto)
    (out / "market_analysis.txt").write_text(analysis + "\n\nSCAN\n" + sc.round(3).to_string(), encoding="utf-8")

    _say(f"4/4 paper trading replay, {paper_days} sessions from {paper_start.date()} …")
    pcfg = cfg.with_overrides({"data": {"history_years": 3}})
    pdir = out / "paper"
    if pdir.exists():
        for f in pdir.iterdir():
            f.unlink()
    pdir.mkdir(parents=True, exist_ok=True)
    runner = LiveRunner(pcfg, broker=PaperBroker(pcfg, state_path=pdir / "paper_broker.json"),
                        journal_path=pdir / "journal.db", provider=prov)
    daily = []
    last_pre = []
    for ts in bench[-paper_days:]:
        d = ts.date()
        runner.engine = None
        last_pre = runner.premarket(d)
        outd = runner.run_eod(d)
        rev = runner.review(1, d)
        daily.append(f"## {d}\n" + Routines.format(last_pre) + f"\n\nEOD: equity ₹{outd['equity']:,.0f}, "
                     f"{outd['open_trades']} open, {outd['queued']} queued\n" + Routines.format(outd["post"]) + "\n\n" + rev)
        _say(f"    {d}: equity ₹{outd['equity']:,.0f}  open {outd['open_trades']}  queued {outd['queued']}")
    weekly = runner.review(paper_days + 5, bench[-1].date())
    (out / "paper_daily_log.md").write_text("\n\n".join(daily), encoding="utf-8")
    (out / "paper_review.md").write_text(weekly, encoding="utf-8")
    desk = desk_report(f"QuantDesk daily desk report · {bench[-1].date()} (synthetic market)",
                       [("Paper account review", weekly), ("Market analysis (as of backtest end)", analysis)], last_pre)
    (out / "desk_report.html").write_text(desk, encoding="utf-8")
    _say(f"\ndone → {out}")
    for f in sorted(out.iterdir()):
        if f.is_file():
            _say(f"  {f.name}")
    return {"backtest": res, "out": out}
