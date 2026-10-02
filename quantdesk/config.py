"""Configuration loading: YAML files deep-merged in order, with dotted-path access."""
from __future__ import annotations

import copy
import datetime as dt
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "quantdesk.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class Config:
    """Thin wrapper over the merged YAML dict.

    ``cfg.get("risk.risk_per_trade")`` reads a dotted path; ``cfg["risk"]`` a section.
    """

    def __init__(self, data: dict, root: Path | None = None):
        self.data = data
        self.root = root or PROJECT_ROOT

    @classmethod
    def load(cls, *paths: str | Path, overrides: dict | None = None) -> "Config":
        files = [Path(p) for p in paths] or [DEFAULT_CONFIG]
        merged: dict = {}
        for p in files:
            with open(p, encoding="utf-8") as fh:
                merged = deep_merge(merged, yaml.safe_load(fh) or {})
        if overrides:
            merged = deep_merge(merged, overrides)
        return cls(merged, root=PROJECT_ROOT)

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def with_overrides(self, overrides: dict) -> "Config":
        return Config(deep_merge(self.data, overrides), self.root)

    # ---- convenience -------------------------------------------------------------
    @property
    def runtime_dir(self) -> Path:
        p = Path(self.get("runtime.dir", "runtime"))
        p = p if p.is_absolute() else self.root / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    def instrument_spec(self, symbol: str) -> dict:
        specs = self.get("instruments", {})
        if symbol in specs:
            return dict(specs[symbol])
        spec = dict(specs.get("_equity_default", {"kind": "equity", "lot_size": 1}))
        spec.setdefault("yahoo", f"{symbol}{spec.pop('yahoo_suffix', '.NS')}")
        spec.pop("yahoo_suffix", None)
        return spec

    def symbols(self, group: str) -> list[str]:
        """Resolve a strategy's `symbols:` value: a group name or an explicit list."""
        if isinstance(group, list):
            return list(group)
        if group == "indices":
            return list(self.get("universe.indices", []))
        if group == "equities":
            return list(self.get("universe.equities", []))
        if group == "all":
            return self.symbols("indices") + self.symbols("equities")
        return [group]

    def all_symbols(self) -> list[str]:
        syms = self.symbols("all")
        vix = self.get("universe.volatility_index")
        if vix:
            syms.append(vix)
        for a, b in self.get("universe.pairs", []) or []:
            syms += [a, b]
        return list(dict.fromkeys(syms))

    def holidays(self) -> set[dt.date]:
        out = set()
        for h in self.get("calendar.holidays", []) or []:
            out.add(h if isinstance(h, dt.date) else dt.date.fromisoformat(str(h)))
        return out

    def events(self) -> list[tuple[dt.date, str]]:
        out = []
        for ev in self.get("calendar.events", []) or []:
            d = ev["date"]
            out.append((d if isinstance(d, dt.date) else dt.date.fromisoformat(str(d)), ev["name"]))
        return sorted(out)
