"""Zerodha Kite Connect adapter (LIVE money).

Guard rails, all of which must pass before a single order leaves the machine:
  1. `account.mode: live` in config AND the `--live` CLI flag (`confirm_live=True`);
  2. KITE_API_KEY / KITE_ACCESS_TOKEN in the environment (daily token; see Kite docs);
  3. no kill-switch file at runtime/KILL (touch it to stop all order flow instantly);
  4. every order is a marketable LIMIT with a price-protection band, never a bare MARKET;
  5. per-order notional cap (`live.max_order_value`).

Requires `pip install kiteconnect`. Untested against the real API from this repo's CI —
paper-trade first, then go live with one lot.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd

from ..core.types import Fill, Instrument, Order
from .broker import Broker
from .costs import CostModel

log = logging.getLogger(__name__)

INDEX_LTP = {"NIFTY": "NSE:NIFTY 50", "BANKNIFTY": "NSE:NIFTY BANK", "INDIAVIX": "NSE:INDIA VIX"}


class LiveTradingDisabled(RuntimeError):
    pass


class KiteBroker(Broker):
    name = "kite"
    live = True

    def __init__(self, cfg, confirm_live: bool = False):
        if cfg.get("account.mode") != "live" or not confirm_live:
            raise LiveTradingDisabled("live trading needs account.mode=live in config AND the --live flag")
        try:
            from kiteconnect import KiteConnect
        except ImportError as exc:
            raise LiveTradingDisabled("pip install kiteconnect") from exc
        key, token = os.environ.get("KITE_API_KEY"), os.environ.get("KITE_ACCESS_TOKEN")
        if not key or not token:
            raise LiveTradingDisabled("set KITE_API_KEY and KITE_ACCESS_TOKEN")
        self.kite = KiteConnect(api_key=key)
        self.kite.set_access_token(token)
        self.cfg = cfg
        self.costs = CostModel(cfg)
        self.kill_file = cfg.runtime_dir / "KILL"
        self.protection = float(cfg.get("live.price_protection", 0.01))
        self.max_order_value = float(cfg.get("live.max_order_value", 500000))
        self.fill_timeout = float(cfg.get("live.fill_timeout_sec", 20))
        self._nfo: pd.DataFrame | None = None

    # ---- symbology ---------------------------------------------------------------------
    def _nfo_instruments(self) -> pd.DataFrame:
        if self._nfo is None:
            self._nfo = pd.DataFrame(self.kite.instruments("NFO"))
        return self._nfo

    def tradingsymbol(self, inst: Instrument) -> tuple[str, str]:
        if inst.kind == "equity":
            return "NSE", inst.symbol
        nfo = self._nfo_instruments()
        if inst.kind == "option":
            m = nfo[(nfo["name"] == inst.underlying) & (nfo["instrument_type"] == inst.right)
                    & (nfo["strike"] == inst.strike) & (pd.to_datetime(nfo["expiry"]).dt.date == inst.expiry)]
        else:  # futures: nearest expiry
            m = nfo[(nfo["name"] == inst.underlying) & (nfo["instrument_type"] == "FUT")].sort_values("expiry").head(1)
        if m.empty:
            raise ValueError(f"no NFO contract for {inst.symbol}")
        return "NFO", str(m.iloc[0]["tradingsymbol"])

    # ---- Broker API ----------------------------------------------------------------------
    def execute(self, order: Order, ref_price: float, ts, atr=None) -> Fill | None:
        if Path(self.kill_file).exists():
            log.error("kill switch present (%s): order %s blocked", self.kill_file, order.id)
            order.status = "REJECTED"
            return None
        if abs(order.qty) * ref_price > self.max_order_value:
            log.error("order %s notional %.0f exceeds live.max_order_value", order.id, abs(order.qty) * ref_price)
            order.status = "REJECTED"
            return None
        exch, tsym = self.tradingsymbol(order.instrument)
        k = self.kite
        side = k.TRANSACTION_TYPE_BUY if order.qty > 0 else k.TRANSACTION_TYPE_SELL
        tick = 0.05
        band = ref_price * (1 + self.protection) if order.qty > 0 else ref_price * (1 - self.protection)
        limit = round(round(band / tick) * tick, 2)
        product = k.PRODUCT_CNC if order.instrument.kind == "equity" else k.PRODUCT_NRML
        broker_id = k.place_order(variety=k.VARIETY_REGULAR, exchange=exch, tradingsymbol=tsym,
                                  transaction_type=side, quantity=abs(order.qty), product=product,
                                  order_type=k.ORDER_TYPE_LIMIT, price=limit, tag="quantdesk")
        order.broker_id = str(broker_id)
        deadline = time.time() + self.fill_timeout
        while time.time() < deadline:
            hist = k.order_history(broker_id)
            last = hist[-1] if hist else {}
            status = last.get("status")
            if status == "COMPLETE":
                px = float(last["average_price"])
                fees, br = self.costs.fees(order.instrument, order.qty, px)
                order.status = "FILLED"
                return Fill(order.id, order.instrument, order.qty, px, fees, pd.Timestamp.now(), br)
            if status in ("REJECTED", "CANCELLED"):
                log.error("order %s %s: %s", broker_id, status, last.get("status_message"))
                order.status = status
                return None
            time.sleep(1.0)
        k.cancel_order(variety=k.VARIETY_REGULAR, order_id=broker_id)
        log.warning("order %s not filled within %.0fs; cancelled", broker_id, self.fill_timeout)
        order.status = "CANCELLED"
        return None

    def positions(self) -> dict[str, dict]:
        out = {}
        for p in self.kite.positions().get("net", []):
            if p["quantity"]:
                out[p["tradingsymbol"]] = {"qty": p["quantity"], "avg_price": p["average_price"],
                                           "last_price": p["last_price"], "pnl": p.get("pnl")}
        return out

    def cash(self) -> float:
        m = self.kite.margins("equity")
        return float(m["available"]["live_balance"]) if "live_balance" in m.get("available", {}) else float(m["net"])

    def ltp(self, symbols: list[str]) -> dict[str, float]:
        keys = [INDEX_LTP.get(s, f"NSE:{s}") for s in symbols]
        q = self.kite.ltp(keys)
        return {s: float(q[k]["last_price"]) for s, k in zip(symbols, keys) if k in q}

    def quotes(self, instruments: list[Instrument]) -> dict[str, float]:
        """LTP for any mix of equities, index futures and NFO options, keyed by our symbol."""
        keys = {}
        for inst in instruments:
            if inst.kind in ("equity", "index") and inst.symbol in INDEX_LTP:
                keys[inst.symbol] = INDEX_LTP[inst.symbol]
            else:
                exch, tsym = self.tradingsymbol(inst)
                keys[inst.symbol] = f"{exch}:{tsym}"
        q = self.kite.ltp(list(keys.values()))
        return {sym: float(q[k]["last_price"]) for sym, k in keys.items() if k in q}

    def healthcheck(self) -> tuple[bool, str]:
        try:
            prof = self.kite.profile()
            return True, f"kite session ok for {prof.get('user_id')}"
        except Exception as exc:  # token expired, network, etc.
            return False, f"kite session failed: {exc}"
