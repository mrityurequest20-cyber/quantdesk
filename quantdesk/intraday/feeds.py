"""Intraday bar feeds. Every feed yields *completed* 1-minute bars (tz-aware IST index,
columns open/high/low/close/volume) and owns the clock, so the same engine runs live
(wall clock) or on a replay (simulated clock) without knowing which.

  YahooIntradayFeed  free; 1m bars for the last ~7 days, polled each minute. Good for
                     starting out; not exchange-grade (can lag, index volume is sparse).
  KiteIntradayFeed   Kite Connect: minute history + KiteTicker full-mode ticks aggregated
                     into bars; also emits trades for order flow (footprint / CVD).
  ReplayFeed         recorded (or synthetic) sessions stepped minute by minute.
"""
from __future__ import annotations

import abc
import datetime as dt
import logging
import threading
from collections import defaultdict

import numpy as np
import pandas as pd

from .orderflow import KiteSnapshotAdapter, Trade

log = logging.getLogger(__name__)
IST = "Asia/Kolkata"
OPEN, CLOSE = dt.time(9, 15), dt.time(15, 30)
BAR = pd.Timedelta(minutes=1)


def ist(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize(IST) if ts.tzinfo is None else ts.tz_convert(IST)


def session_bounds(day: dt.date) -> tuple[pd.Timestamp, pd.Timestamp]:
    return (pd.Timestamp(dt.datetime.combine(day, OPEN), tz=IST), pd.Timestamp(dt.datetime.combine(day, CLOSE), tz=IST))


def normalise_bars(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: str(c).lower() for c in df.columns})
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize(IST) if idx.tz is None else idx.tz_convert(IST)
    df.index = idx.floor("min")
    if "volume" not in df:
        df["volume"] = 0.0
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    t = df.index.time
    return df[(t >= OPEN) & (t < CLOSE)]


class IntradayFeed(abc.ABC):
    name = "base"
    realtime = True
    has_ticks = False

    def now(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz=IST)

    @abc.abstractmethod
    def history(self, symbol: str, days: int = 5) -> pd.DataFrame:
        """Completed 1m bars for the last `days` sessions."""

    @abc.abstractmethod
    def poll(self, symbol: str, since: pd.Timestamp | None) -> pd.DataFrame:
        """Completed 1m bars strictly after `since`."""

    def trades(self, symbol: str) -> list[Trade]:
        """Trades since the last call (order-flow feeds only)."""
        return []

    def history_bars(self, symbol: str, days: int = 55) -> pd.DataFrame:
        """Longer 5m history for model training (default: the 1m history resampled)."""
        from .quant import to_5m
        return to_5m(self.history(symbol, days))

    def completed(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop the still-forming bar: a bar is complete once its minute has ended."""
        return df[df.index + BAR <= self.now()]


class YahooIntradayFeed(IntradayFeed):
    name = "yahoo"

    def __init__(self, cfg):
        self.cfg = cfg

    def _ticker(self, symbol: str) -> str:
        return self.cfg.instrument_spec(symbol).get("yahoo", f"{symbol}.NS")

    def _fetch(self, symbol: str, period: str) -> pd.DataFrame:
        import yfinance as yf
        raw = yf.Ticker(self._ticker(symbol)).history(period=period, interval="1m", auto_adjust=False, prepost=False)
        if raw is None or raw.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        return self.completed(normalise_bars(raw))

    def history(self, symbol, days=5):
        return self._fetch(symbol, f"{min(max(days, 1), 7)}d")

    def poll(self, symbol, since):
        df = self._fetch(symbol, "1d")
        return df if since is None else df[df.index > since]

    def history_bars(self, symbol, days=55):
        import yfinance as yf
        from .quant import to_5m
        raw = yf.Ticker(self._ticker(symbol)).history(period=f"{min(max(days, 5), 59)}d", interval="5m", auto_adjust=False,
                                                      prepost=False)
        if raw is None or raw.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        return to_5m(self.completed(normalise_bars(raw)))


class BarAggregator:
    """Tick → 1m OHLCV. Volume from cumulative day volume differences when available."""

    def __init__(self):
        self.cur: dict[str, dict] = {}
        self.done: dict[str, list] = defaultdict(list)
        self.cum: dict[str, float] = {}

    def add(self, symbol: str, ts: pd.Timestamp, price: float, cum_volume: float | None = None, size: float = 0.0):
        m = ist(ts).floor("min")
        if cum_volume is not None:
            prev = self.cum.get(symbol)
            self.cum[symbol] = cum_volume
            size = max(cum_volume - prev, 0.0) if prev is not None else 0.0
        c = self.cur.get(symbol)
        if c is not None and m > c["t"]:
            self.done[symbol].append(c)
            c = None
        if c is None:
            c = self.cur[symbol] = {"t": m, "open": price, "high": price, "low": price, "close": price, "volume": 0.0}
        c["high"], c["low"], c["close"] = max(c["high"], price), min(c["low"], price), price
        c["volume"] += size

    def roll(self, symbol: str, now: pd.Timestamp) -> None:
        c = self.cur.get(symbol)
        if c is not None and ist(now).floor("min") > c["t"]:
            self.done[symbol].append(c)
            self.cur.pop(symbol)

    def take(self, symbol: str) -> pd.DataFrame:
        rows, self.done[symbol] = self.done[symbol], []
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        return pd.DataFrame(rows).set_index("t")[["open", "high", "low", "close", "volume"]]


class KiteIntradayFeed(IntradayFeed):
    """Needs KITE_API_KEY / KITE_ACCESS_TOKEN and `pip install kiteconnect`."""
    name = "kite"
    has_ticks = True
    INDEX_TOKENS = {"NIFTY": "NSE:NIFTY 50", "BANKNIFTY": "NSE:NIFTY BANK", "INDIAVIX": "NSE:INDIA VIX"}

    def __init__(self, cfg, symbols: list[str]):
        import os

        from kiteconnect import KiteConnect, KiteTicker
        key, token = os.environ["KITE_API_KEY"], os.environ["KITE_ACCESS_TOKEN"]
        self.kite = KiteConnect(api_key=key)
        self.kite.set_access_token(token)
        keys = {s: self.INDEX_TOKENS.get(s, f"NSE:{s}") for s in symbols}
        ltp = self.kite.ltp(list(keys.values()))
        self.tokens = {s: ltp[k]["instrument_token"] for s, k in keys.items() if k in ltp}
        self.by_token = {v: k for k, v in self.tokens.items()}
        self.agg = BarAggregator()
        self.adapter = KiteSnapshotAdapter()
        self._trades: dict[str, list[Trade]] = defaultdict(list)
        self._lock = threading.Lock()
        self.ticker = KiteTicker(key, token)
        self.ticker.on_ticks = self._on_ticks
        self.ticker.on_connect = lambda ws, resp: (ws.subscribe(list(self.tokens.values())),
                                                    ws.set_mode(ws.MODE_FULL, list(self.tokens.values())))
        self.ticker.connect(threaded=True)

    def _on_ticks(self, ws, ticks):
        with self._lock:
            for t in ticks:
                sym = self.by_token.get(t.get("instrument_token"))
                if sym is None:
                    continue
                ts = t.get("exchange_timestamp") or t.get("last_trade_time") or pd.Timestamp.now(tz=IST)
                self.agg.add(sym, ts, float(t["last_price"]), cum_volume=t.get("volume_traded"))
                self._trades[sym] += self.adapter.to_trades(t)

    def history(self, symbol, days=5):
        now = self.now()
        rows = self.kite.historical_data(self.tokens[symbol], (now - pd.Timedelta(days=days + 4)).to_pydatetime(),
                                         now.to_pydatetime(), "minute")
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return self.completed(normalise_bars(df.set_index("date")))

    def poll(self, symbol, since):
        with self._lock:
            self.agg.roll(symbol, self.now())
            df = self.agg.take(symbol)
        if df.empty:
            return df
        df.index = pd.DatetimeIndex(df.index).tz_convert(IST) if pd.DatetimeIndex(df.index).tz else pd.DatetimeIndex(df.index).tz_localize(IST)
        return df if since is None else df[df.index > since]

    def trades(self, symbol):
        with self._lock:
            out, self._trades[symbol] = self._trades[symbol], []
        return out


class ReplayFeed(IntradayFeed):
    """Steps a simulated clock through recorded or synthetic 1m sessions.
    `history()` returns only sessions *before* the replay day; `poll()` only bars whose
    minute has closed on the simulated clock — the engine cannot see the future."""
    name = "replay"
    realtime = False

    def __init__(self, bars: dict[str, pd.DataFrame], day: dt.date, trades: dict[str, list[Trade]] | None = None):
        self.bars = {s: normalise_bars(df) for s, df in bars.items()}
        self.day = day
        self.open_ts, self.close_ts = session_bounds(day)
        self.clock = self.open_ts
        self._trades = {s: sorted(v, key=lambda t: t.ts) for s, v in (trades or {}).items()}
        self._tpos = defaultdict(int)
        self.has_ticks = bool(trades)

    def now(self):
        return self.clock

    def advance(self, minutes: int = 1) -> bool:
        self.clock = self.clock + pd.Timedelta(minutes=minutes)
        return self.clock <= self.close_ts

    def history(self, symbol, days=5):
        df = self.bars.get(symbol)
        if df is None:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        prior = df[df.index < self.open_ts]
        days_kept = sorted(set(prior.index.date))[-days:]
        return prior[np.isin(prior.index.date, days_kept)]

    def poll(self, symbol, since):
        df = self.bars.get(symbol)
        if df is None:
            return df
        day = df[(df.index >= self.open_ts) & (df.index + BAR <= self.clock)]
        return day if since is None else day[day.index > since]

    def trades(self, symbol):
        lst = self._trades.get(symbol, [])
        i = self._tpos[symbol]
        j = i
        while j < len(lst) and lst[j].ts < self.clock:
            j += 1
        self._tpos[symbol] = j
        return lst[i:j]

    @staticmethod
    def sessions(bars: pd.DataFrame) -> list[dt.date]:
        return sorted(set(normalise_bars(bars).index.date))
