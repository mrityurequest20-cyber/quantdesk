"""Regime-aware meta-allocator: scales each strategy's risk budget by how well its
family suits the current regime and (shrunk) evidence from its own recent trades."""
from __future__ import annotations

import numpy as np


class Allocator:
    def __init__(self, cfg):
        a = cfg.get("allocator", {})
        self.lookback = a.get("lookback_trades", 20)
        self.tilt = a.get("perf_tilt", 0.5)
        self.min_w = a.get("min_weight", 0.25)
        self.max_w = a.get("max_weight", 1.5)
        self.table = a.get("regime_weights", {})
        self.index_syms = set(cfg.get("universe.indices", []))

    def weight(self, intent, ctx) -> tuple[float, str]:
        regime_sym = intent.symbol if intent.symbol in self.index_syms else None
        label = ctx.regime(regime_sym)
        w_reg = float(self.table.get(intent.family, {}).get(label, 1.0))
        if w_reg <= 0:
            return 0.0, f"{intent.family} is switched off in a {label} regime"
        recent = [t.r_multiple for t in ctx.closed_trades if t.strategy == intent.strategy][-self.lookback:]
        n = len(recent)
        tilt = 1.0
        if n >= 5:
            m = float(np.mean(recent))
            shrink = n / (n + 10)
            tilt = 1 + self.tilt * np.tanh(m) * shrink
        w = float(np.clip(w_reg * tilt, self.min_w, self.max_w))
        note = f"regime {label} → {w_reg:.2f}"
        if n >= 5:
            note += f"; last {n} trades avg {np.mean(recent):+.2f}R → tilt {tilt:.2f}"
        return w, note
