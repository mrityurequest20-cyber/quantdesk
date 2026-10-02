"""QuantDesk command line.

  python -m quantdesk analyze NIFTY RELIANCE      chart + quant + options read
  python -m quantdesk scan                        universe dashboard
  python -m quantdesk options NIFTY               chain, expected moves, costed structures
  python -m quantdesk backtest --report bt.html   full multi-strategy backtest
  python -m quantdesk walkforward --strategy trend_rider --grid "fast=10,20;slow=50,100"
  python -m quantdesk paper premarket|run|intraday|review|status
  python -m quantdesk journal trades|events|decisions|export
  python -m quantdesk demo                        everything, offline, on synthetic data
  python -m quantdesk schedule                    cron lines for the daily routine
  python -m quantdesk serve                       desk UI with GoCharting charts on http://127.0.0.1:8765
  python -m quantdesk intraday live               real-time intraday options desk (paper), thinking out loud
  python -m quantdesk intraday replay --synthetic 20   offline: 20 synthetic sessions through the same engine

Global: --source yahoo|csv|synthetic, --config extra.yaml (repeatable), --live (Kite; real money).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from .config import DEFAULT_CONFIG, Config


def _load_data(cfg, source: str | None, start=None, end=None):
    from .data import make_provider
    source = source or cfg.get("data.source", "yahoo")
    end = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    start = pd.Timestamp(start) if start else end - pd.DateOffset(years=cfg.get("data.history_years", 8))
    kw = {"end": end} if source == "synthetic" else {}
    prov = make_provider(cfg, source, **kw)
    data = prov.universe(cfg.all_symbols(), start, end)
    if not data:
        sys.exit(f"no data from source {source!r}; try --source synthetic for an offline run")
    return data


def _print_df(df: pd.DataFrame, pct_cols=(), fmt="{:,.2f}"):
    d = df.copy()
    for c in d.columns:
        if c in pct_cols:
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v * 100:+.1f}%")
        elif pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else fmt.format(v))
    print(d.to_string())


# ---- commands --------------------------------------------------------------------------------
def cmd_analyze(cfg, a):
    from .engine.engine import Engine
    from .reporting.market import analyze_symbol
    data = _load_data(cfg, a.source)
    aux = Engine.build_aux(cfg, data) if any(s in cfg.symbols("indices") for s in a.symbols) else None
    for s in a.symbols:
        if s not in data:
            print(f"{s}: no data")
            continue
        print(analyze_symbol(cfg, s, data, aux))
        print()


def cmd_scan(cfg, a):
    from .reporting.market import scan
    data = _load_data(cfg, a.source)
    _print_df(scan(cfg, data), pct_cols=("1d", "1m", "12-1m", "vs_sma200"))


def cmd_options(cfg, a):
    from .reporting.market import analyze_symbol
    data = _load_data(cfg, a.source)
    txt = analyze_symbol(cfg, a.symbol, data)
    start = txt.find("Implied vol:")
    print(txt[start:] if start >= 0 else txt)


def cmd_backtest(cfg, a):
    from .backtest.montecarlo import trade_bootstrap
    from .backtest.runner import run_backtest
    from .journal.journal import Journal
    from .reporting.html import backtest_report
    from .strategies import build_strategies
    data = _load_data(cfg, a.source, end=a.end)
    strats = build_strategies(cfg, a.strategies.split(",") if a.strategies else None)
    journal = Journal(a.journal) if a.journal else None
    print(f"backtesting {', '.join(s.name for s in strats)} on {len(data)} series …", file=sys.stderr)
    res = run_backtest(cfg, data, strats, start=a.start or cfg.get("backtest.start"), end=a.end, journal=journal,
                       progress=lambda n, N, ts: print(f"  {ts.date()} ({n}/{N})", file=sys.stderr))
    print(res.summary())
    print()
    _print_df(res.per_strategy[["trades", "win_rate", "profit_factor", "avg_r", "total_pnl", "total_fees"]],
              pct_cols=("win_rate",))
    mc = trade_bootstrap(res.trades["pnl"].to_numpy(), cfg.get("account.starting_capital")) if a.mc or a.report else None
    if mc:
        print(f"\nMonte Carlo (trade bootstrap): median max DD {mc['maxdd_p50']:.1%}, 95th pct {mc['maxdd_p95']:.1%}, "
              f"P(loss) {mc['p_loss']:.1%}")
    if a.report:
        Path(a.report).write_text(backtest_report(res, synthetic=(a.source or cfg.get("data.source")) == "synthetic", mc=mc),
                                  encoding="utf-8")
        print(f"\nreport → {a.report}")
    if a.trades_csv:
        res.trades.to_csv(a.trades_csv, index=False)


def cmd_walkforward(cfg, a):
    from .backtest.walkforward import walk_forward
    grid = {}
    for part in a.grid.split(";"):
        k, v = part.split("=")
        grid[k.strip()] = [json.loads(x) for x in v.split(",")]
    data = _load_data(cfg, a.source)
    wf = walk_forward(cfg, data, a.strategy, grid, a.train, a.test,
                      progress=lambda f: print(f"  fold {f['test']}: best {f['best_params']} IS {f['is_sharpe']:.2f} "
                                               f"OOS {f['oos_sharpe']:.2f}", file=sys.stderr))
    print(wf.folds.to_string())
    s = wf.oos_stats
    if s:
        print(f"\nStitched OOS: CAGR {s['cagr']:+.2%}, Sharpe {s['sharpe']:.2f}, max DD {s['max_drawdown']:.2%}; "
              f"Deflated Sharpe vs {wf.n_trials} variants: {wf.dsr:.1%}")


def _runner(cfg, a):
    from .engine.live import LiveRunner
    broker = None
    if a.live:
        from .execution.kite import KiteBroker
        broker = KiteBroker(cfg, confirm_live=True)
    return LiveRunner(cfg, a.source, broker=broker)


def cmd_paper(cfg, a):
    from .ops.checks import Routines
    runner = _runner(cfg, a)
    asof = dt.date.fromisoformat(a.asof) if a.asof else None
    if a.action == "premarket":
        res = runner.premarket(asof)
        print(Routines.format(res))
        blocking = Routines.blocking(res)
        print("\nENTRIES BLOCKED TODAY: " + "; ".join(b.detail for b in blocking) if blocking else "\nclear to trade")
    elif a.action == "open":
        print(runner.run_open(asof))
    elif a.action == "run":
        out = runner.run_eod(asof)
        print(f"processed {out['processed'] or 'nothing new'}; equity ₹{out['equity']:,.0f}; "
              f"{out['open_trades']} open, {out['queued']} queued for next open")
        print(Routines.format(out["post"]))
    elif a.action == "intraday":
        eng = runner.load(asof)
        if a.live:
            quotes = runner.broker.ltp(list(runner.data))
        else:
            quotes = {s: float(df["close"].iloc[-1]) for s, df in runner.data.items()}
        eng.ctx.ts = eng.timeline[-1]
        r = Routines(cfg, eng, runner.data, runner.journal)
        res = r.intraday(quotes)
        r.record("intraday", res, ts=pd.Timestamp.now())
        print(Routines.format(res))
    elif a.action == "review":
        print(runner.review(a.days, asof))
    elif a.action == "status":
        eng = runner.load(asof)
        c = eng.curve[-1] if eng.curve else {}
        print(f"equity ₹{c.get('equity', runner.broker.cash()):,.0f} · cash ₹{runner.broker.cash():,.0f} · "
              f"halted: {eng.risk.halted} · last processed: {runner.journal.get_state('last_processed')}")
        for t in eng.open_trades:
            print(f"  {t.id} {t.strategy:<15} {t.symbol:<18} since {str(t.opened_at)[:10]} P&L ₹{t.pnl:,.0f} "
                  f"stop {f'{t.stop:,.2f}' if t.stop is not None else '—'}")
        for pe in eng.pending_entries:
            print(f"  queued: {pe['intent'].strategy} {pe['intent'].symbol} x{pe['units']}")


def cmd_journal(cfg, a):
    from .journal.journal import Journal
    j = Journal(a.db or cfg.runtime_dir / "journal.db")
    if a.what == "trades":
        t = j.trades("open" if a.open else None)
        cols = ["id", "strategy", "symbol", "opened_at", "closed_at", "exit_reason", "pnl", "r_multiple", "grade"]
        print(t[cols].tail(a.n).to_string(index=False) if not t.empty else "no trades")
    elif a.what == "events":
        print(j.events().tail(a.n).to_string(index=False))
    elif a.what == "decisions":
        print(j.decisions().tail(a.n)[["ts", "strategy", "symbol", "action", "units", "detail"]].to_string(index=False))
    elif a.what == "show":
        t = j.df("SELECT * FROM trades WHERE id=?", (a.id,))
        if t.empty:
            print("no such trade")
        for k, v in t.iloc[0].items():
            print(f"{k:>15}: {v}")
    elif a.what == "export":
        out = Path(a.out or cfg.runtime_dir / "export")
        out.mkdir(parents=True, exist_ok=True)
        for name in ("trades", "decisions", "fills", "events", "equity", "checks"):
            j.df(f"SELECT * FROM {name}").to_csv(out / f"{name}.csv", index=False)
        print(f"exported → {out}")


def cmd_risk(cfg, a):
    from .engine.live import LiveRunner
    r = LiveRunner(cfg, a.source)
    eng = r.load()
    eq = eng.curve[-1]["equity"] if eng.curve else r.broker.cash()
    eng.risk.reset_halt(eq)
    eng.save_state()
    r.journal.event(pd.Timestamp.now(), "WARN", "risk", f"kill switch manually re-armed at equity ₹{eq:,.0f}")
    print("risk halt cleared; peak reset to current equity")


def cmd_schedule(cfg, a):
    root = Path(__file__).resolve().parent.parent
    py = sys.executable
    print(f"""# QuantDesk daily routine (IST). Install with `crontab -e`; server clock assumed Asia/Kolkata.
CRON_TZ=Asia/Kolkata
40 8 * * 1-5  cd {root} && {py} -m quantdesk paper premarket >> runtime/cron.log 2>&1
# live only: execute queued orders against live quotes shortly after the open
# 20 9 * * 1-5  cd {root} && {py} -m quantdesk --live paper open >> runtime/cron.log 2>&1
*/30 10-15 * * 1-5 cd {root} && {py} -m quantdesk paper intraday >> runtime/cron.log 2>&1
45 16 * * 1-5 cd {root} && {py} -m quantdesk paper run >> runtime/cron.log 2>&1
50 16 * * 1-5 cd {root} && {py} -m quantdesk paper review >> runtime/cron.log 2>&1
0 18 * * 5    cd {root} && {py} -m quantdesk paper review --days 7 >> runtime/cron.log 2>&1
# intraday options desk (paper): waits for 09:15, trades to 15:30, squares off, writes the review
55 8 * * 1-5  cd {root} && {py} -m quantdesk intraday live --quiet >> runtime/intraday.log 2>&1
# the phone app (token-protected); or run `quantdesk intraday live --forever` as a service instead
@reboot       cd {root} && {py} -m quantdesk serve --host 0.0.0.0 >> runtime/web.log 2>&1""")


def cmd_serve(cfg, a):
    from .engine.live import LiveRunner
    from .execution.broker import PaperBroker
    from .web.server import serve
    rt = cfg.runtime_dir
    broker = PaperBroker(cfg, state_path=Path(a.broker_state) if a.broker_state else rt / "paper_broker.json")
    runner = LiveRunner(cfg, a.source, broker=broker, journal_path=a.journal or rt / "journal.db")
    serve(cfg, runner, a.host, a.port, a.token)


def cmd_demo(cfg, a):
    from .demo import run_demo
    run_demo(cfg, Path(a.out) if a.out else cfg.runtime_dir / "demo", paper_days=a.days)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quantdesk", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", action="append", help="extra YAML merged over the default config (repeatable)")
    p.add_argument("--source", choices=["yahoo", "csv", "synthetic"], help="data source (default from config)")
    p.add_argument("--live", action="store_true", help="use the Kite broker (REAL MONEY; also needs account.mode: live)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("analyze", help="chart + quant + options analysis")
    s.add_argument("symbols", nargs="+")
    s.set_defaults(fn=cmd_analyze)
    s = sub.add_parser("scan", help="universe dashboard")
    s.set_defaults(fn=cmd_scan)
    s = sub.add_parser("options", help="option chain, expected moves and costed structures for an index")
    s.add_argument("symbol")
    s.set_defaults(fn=cmd_options)
    s = sub.add_parser("backtest", help="multi-strategy backtest")
    s.add_argument("--strategies", help="comma list (default: all enabled)")
    s.add_argument("--start")
    s.add_argument("--end")
    s.add_argument("--report", help="write an HTML tearsheet here")
    s.add_argument("--journal", help="persist the backtest journal to this SQLite file")
    s.add_argument("--trades-csv")
    s.add_argument("--mc", action="store_true", help="Monte Carlo trade bootstrap")
    s.set_defaults(fn=cmd_backtest)
    s = sub.add_parser("walkforward", help="walk-forward optimisation of one strategy")
    s.add_argument("--strategy", required=True)
    s.add_argument("--grid", required=True, help='e.g. "fast=10,20;slow=50,100"')
    s.add_argument("--train", type=float, default=3)
    s.add_argument("--test", type=float, default=1)
    s.set_defaults(fn=cmd_walkforward)
    s = sub.add_parser("paper", help="paper/live daily routines")
    s.add_argument("action", choices=["premarket", "open", "run", "intraday", "review", "status"])
    s.add_argument("--asof", help="YYYY-MM-DD (default today)")
    s.add_argument("--days", type=int, default=1)
    s.set_defaults(fn=cmd_paper)
    s = sub.add_parser("journal", help="read the trading journal")
    s.add_argument("what", choices=["trades", "events", "decisions", "show", "export"])
    s.add_argument("--db")
    s.add_argument("--open", action="store_true")
    s.add_argument("--id")
    s.add_argument("--out")
    s.add_argument("-n", type=int, default=30)
    s.set_defaults(fn=cmd_journal)
    s = sub.add_parser("risk", help="risk admin")
    s.add_argument("action", choices=["reset"])
    s.set_defaults(fn=cmd_risk)
    s = sub.add_parser("schedule", help="print cron lines for the daily routine")
    s.set_defaults(fn=cmd_schedule)
    s = sub.add_parser("serve", help="desk UI: GoCharting charts + positions, queue, journal, checks")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.add_argument("--journal", help="journal DB to serve (default runtime/journal.db)")
    s.add_argument("--broker-state", help="paper broker state JSON (default runtime/paper_broker.json)")
    s.add_argument("--token", help="access token (auto-generated when --host is not localhost)")
    s.set_defaults(fn=cmd_serve)
    from .intraday.cli import register as register_intraday
    register_intraday(sub)
    s = sub.add_parser("demo", help="end-to-end offline demo on synthetic data")
    s.add_argument("--out")
    s.add_argument("--days", type=int, default=15, help="days of paper trading to replay")
    s.set_defaults(fn=cmd_demo)
    from .data.cli import register as register_data
    register_data(sub)
    s = sub.add_parser("research", help="test pre-registered edge hypotheses on real NIFTY/BANKNIFTY data")
    s.add_argument("--out", default="research", help="folder for edge_report.md and edges.json")
    s.add_argument("--warehouse", help="the data warehouse folder: also test the volatility premium on real option "
                                       "prices and FII positioning (data_report.md)")
    s.set_defaults(fn=cmd_research)
    return p


def cmd_research(cfg, a):
    import pandas as pd
    from .research.edges import links_json, load_global, load_yahoo, report, run, to_json
    data = load_yahoo()
    data["global"] = load_global()
    res = run(data)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    md = report(res, data, f"{pd.Timestamp.now(tz='Asia/Kolkata'):%Y-%m-%d %H:%M} IST")
    (out / "edge_report.md").write_text(md, encoding="utf-8")
    (out / "edges.json").write_text(to_json(res), encoding="utf-8")
    (out / "links.json").write_text(links_json(data), encoding="utf-8")
    print(md)
    wh = Path(a.warehouse) if a.warehouse else None
    if wh and any(wh.glob("fo_bhav_*.parquet")):
        from .research import warehouse_research as W
        res = W.run_all(wh, data["daily"], cfg)
        md2 = W.report(res, f"{pd.Timestamp.now(tz='Asia/Kolkata'):%Y-%m-%d %H:%M} IST")
        (out / "data_report.md").write_text(md2, encoding="utf-8")
        (out / "vrp_positioning.json").write_text(W.to_json(res), encoding="utf-8")
        if len(res.get("trade_rows", [])):
            res["trade_rows"].assign(strikes=res["trade_rows"]["strikes"].astype(str)).to_csv(
                out / "vrp_trades.csv.gz", index=False)
        print(md2)
    elif wh:
        print(f"no warehouse files in {wh}: skipping the option-price and positioning research")


def main(argv=None):
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cfg = Config.load(DEFAULT_CONFIG, *(a.config or []))
    a.fn(cfg, a)


if __name__ == "__main__":
    main()
