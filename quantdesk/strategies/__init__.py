from __future__ import annotations

from .base import Strategy
from .mean_reversion import MeanReversion
from .momentum import Momentum
from .options_strats import LongVol, TrendSpread, VRPCondor
from .pairs import Pairs
from .trend import Breakout, TrendRider

REGISTRY: dict[str, type[Strategy]] = {
    "trend_rider": TrendRider,
    "breakout": Breakout,
    "mean_reversion": MeanReversion,
    "momentum": Momentum,
    "pairs": Pairs,
    "vrp_condor": VRPCondor,
    "trend_spread": TrendSpread,
    "long_vol": LongVol,
}


def build_strategies(cfg, only: list[str] | None = None) -> list[Strategy]:
    out = []
    for name, params in (cfg.get("strategies", {}) or {}).items():
        if only and name not in only:
            continue
        if not only and not params.get("enabled", True):
            continue
        cls = REGISTRY.get(name)
        if cls is None:
            raise KeyError(f"unknown strategy {name!r}; known: {sorted(REGISTRY)}")
        out.append(cls(name, params, cfg))
    return out


__all__ = ["Strategy", "REGISTRY", "build_strategies"]
