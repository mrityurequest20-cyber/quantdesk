"""Backtest driver and result object."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..engine.engine import Engine
from ..execution.broker import PaperBroker
from ..journal.journal import Journal
from ..risk import metrics as M
from ..strategies import build_strategies


@dataclass
class BacktestResult:
    equity: pd.DataFrame                 # per-bar snapshot (equity, drawdown, exposures, regime)
    trades: pd.DataFrame                 # closed trades with review columns
    stats: dict
    trade_stats: dict
    per_strategy: pd.DataFrame
    benchmark: pd.Series | None
    fees: dict
    journal: Journal
    engine: Engine
    meta: dict = field(default_factory=dict)

    def summary(self) -> str:
        s, t = self.stats, self.trade_stats
        if not s:
            return "no results"
        lines = [
            f"Period {s['start']} → {s['end']} ({s['years']} y)",
            f"CAGR {s['cagr']:+.2%} | vol {s['ann_vol']:.2%} | Sharpe {s['sharpe']:.2f} | Sortino {s['sortino']:.2f} | "
            f"max DD {s['max_drawdown']:.2%} ({s['max_dd_days']} d) | Calmar {s['calmar']:.2f}",
            f"PSR(SR>0) {s['psr_vs_0']:.2%} | VaR95 {s['var95_1d']:.2%} | CVaR95 {s['cvar95_1d']:.2%} | "
            f"skew {s['skew']:+.2f} | kurt {s['kurtosis']:.1f}",
        ]
        if t.get("trades"):
            lines.append(f"Trades {t['trades']} | win {t['win_rate']:.1%} | PF {t['profit_factor']:.2f} | "
                         f"avg {t['avg_r']:+.2f}R | expectancy ₹{t['expectancy']:,.0f} | fees ₹{t['total_fees']:,.0f}")
        if self.benchmark is not None and len(self.benchmark) > 2:
            b = M.returns_stats(self.benchmark)
            lines.append(f"Benchmark buy&hold: CAGR {b['cagr']:+.2%}, Sharpe {b['sharpe']:.2f}, max DD {b['max_drawdown']:.2%}")
        return "\n".join(lines)


def run_backtest(cfg, data: dict[str, pd.DataFrame], strategies=None, start=None, end=None,
                 journal: Journal | None = None, aux: dict | None = None, progress=None,
                 warmup_bars: int | None = None) -> BacktestResult:
    strategies = strategies if strategies is not None else build_strategies(cfg)
    broker = PaperBroker(cfg)
    eng = Engine(cfg, data, strategies, broker, journal or Journal(":memory:"), aux=aux, warmup_bars=warmup_bars)
    eng.run(start=start, end=end, progress=progress)
    eq = pd.DataFrame(eng.curve).set_index("ts")
    trades = eng.journal.trades("closed")
    rf = cfg.get("backtest.risk_free", 0.0)
    stats = M.returns_stats(eq["equity"], rf)
    closed = pd.DataFrame([t.to_record() for t in eng.closed_trades])
    tstats = M.trade_stats(closed)
    bench_sym = cfg.get("universe.benchmark", "NIFTY")
    bench = None
    if bench_sym in data:
        b = data[bench_sym]["close"].reindex(eq.index).ffill()
        bench = b / b.iloc[0] * eq["equity"].iloc[0]
    return BacktestResult(eq, trades, stats, tstats, M.by_strategy(closed), bench,
                          {"total": broker.fees_paid, **broker.fee_breakdown}, eng.journal, eng,
                          {"strategies": [s.name for s in strategies]})
