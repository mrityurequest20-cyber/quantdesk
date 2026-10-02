"""Option chains for intraday work, normalised to one frame shape, plus chain analytics.

Sources
  NSEOptionChain    free, no account: nseindia.com v3 API (cookie warm-up on /option-chain,
                    expiries from /api/option-chain-contract-info, chain from
                    /api/option-chain-v3). Best effort: NSE throttles and changes endpoints.
  KiteOptionChain   real quotes (bid/ask/OI/volume) through a Kite Connect session.
  KotakOptionChain  real quotes through Kotak Neo's consumer-key endpoints (kotak.py).
  FallbackChain     the first source that answers (e.g. Kotak, then NSE).
  ModelOptionChain  priced from the real spot + an ATM IV (India VIX × beta) + skew, with a
                    modelled spread. Always labelled source='model'.
  RecordedChains    replays chain snapshots saved by the session recorder.

Frame: index = strike; columns ce_/pe_ × ltp, bid, ask, iv (%), oi, doi, vol;
attrs: underlying, spot, expiry (date), ts, source.
"""
from __future__ import annotations

import abc
import datetime as dt
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.types import Instrument
from ..options.pricing import bs_price, greeks, implied_vol, implied_vol_vec
from ..options.surface import SkewModel

log = logging.getLogger(__name__)
IST = "Asia/Kolkata"
SIDES = ("ce", "pe")
FIELDS = ("ltp", "bid", "ask", "iv", "oi", "doi", "vol")
COLUMNS = [f"{s}_{f}" for s in SIDES for f in FIELDS]


def expiry_close(expiry: dt.date) -> pd.Timestamp:
    return pd.Timestamp(dt.datetime.combine(expiry, dt.time(15, 30)), tz=IST)


def time_to_expiry(now: pd.Timestamp, expiry: dt.date) -> float:
    """ACT/365 in years, to the minute. Intraday theta needs this; date-level T is too coarse."""
    now = pd.Timestamp(now)
    now = now.tz_localize(IST) if now.tzinfo is None else now.tz_convert(IST)
    return max((expiry_close(expiry) - now).total_seconds(), 0.0) / (365 * 86400)


class IntradayPricer:
    """BSM with minute-level time to expiry and a skew model around an ATM IV."""

    def __init__(self, r: float = 0.065, q: float = 0.012, skew: SkewModel | None = None):
        self.r, self.q, self.skew = r, q, skew or SkewModel()

    def iv_for(self, K: float, S: float, T: float, atm_iv: float) -> float:
        F = S * np.exp((self.r - self.q) * T)
        return float(self.skew.iv(atm_iv, K, F, max(T, 1 / 365 / 24)))

    def price(self, K: float, right: str, S: float, T: float, iv: float) -> float:
        if T <= 0:
            return max(S - K, 0.0) if right == "CE" else max(K - S, 0.0)
        return float(bs_price(S, K, T, self.r, self.q, iv, right))

    def greeks(self, K: float, right: str, S: float, T: float, iv: float) -> dict:
        return greeks(S, K, max(T, 1e-6), self.r, self.q, iv, right)

    def implied(self, price: float, K: float, right: str, S: float, T: float) -> float:
        return implied_vol(price, S, K, T, self.r, self.q, right) if T > 0 and price > 0 else float("nan")

    def implied_many(self, prices, strikes, right: str, S: float, T: float) -> np.ndarray:
        """`implied` for a whole chain side in one pass (NaN where there's no usable price)."""
        p = np.asarray(prices, dtype=float)
        return implied_vol_vec(np.where(p > 0, p, np.nan), S, strikes, T, self.r, self.q, right)


def empty_chain(underlying: str, spot: float, expiry: dt.date, ts, source: str) -> pd.DataFrame:
    df = pd.DataFrame(columns=COLUMNS, dtype=float)
    df.index.name = "strike"
    df.attrs.update({"underlying": underlying, "spot": spot, "expiry": expiry, "ts": ts, "source": source})
    return df


class ChainSource(abc.ABC):
    name = "base"
    refresh_min: float | None = None           # how often the engine may ask (None: intraday.chain_refresh_min)

    @abc.abstractmethod
    def expiries(self, underlying: str) -> list[dt.date]:
        ...

    @abc.abstractmethod
    def chain(self, underlying: str, expiry: dt.date, spot: float | None = None, ts=None) -> pd.DataFrame:
        ...

    def live_quotes(self, instruments) -> dict[str, tuple[float, float]]:
        """{symbol: (bid, ask)} right now, for sources with a live book; {} otherwise."""
        return {}


class FallbackChain(ChainSource):
    """The first source that answers: the broker's live book, then NSE's public chain. The second source is asked
    at most every `secondary_min` minutes (NSE throttles); in between, its last chain is handed back unchanged,
    timestamp and all, so the engine's staleness rule still sees how old it is."""

    def __init__(self, primary: ChainSource, secondary: ChainSource, secondary_min: float = 3):
        self.primary, self.secondary, self.secondary_min = primary, secondary, secondary_min
        self.name = primary.name
        self.refresh_min = primary.refresh_min
        self.error: dict[str, str] = {}
        self._held: dict[str, tuple[pd.Timestamp, pd.DataFrame]] = {}

    def expiries(self, underlying: str) -> list[dt.date]:
        try:
            return self.primary.expiries(underlying)
        except Exception as exc:
            self.error[underlying] = f"{self.primary.name}: {exc!s:.160}"
            return self.secondary.expiries(underlying)

    def chain(self, underlying, expiry, spot=None, ts=None) -> pd.DataFrame:
        try:
            ch = self.primary.chain(underlying, expiry, spot=spot, ts=ts)
            self.error.pop(underlying, None)
            return ch
        except Exception as exc:
            self.error[underlying] = f"{self.primary.name}: {exc!s:.160}"
        now = pd.Timestamp.now(tz=IST)
        held = self._held.get(underlying)
        if held and held[1].attrs.get("expiry") == expiry and now - held[0] < pd.Timedelta(minutes=self.secondary_min):
            return held[1]
        try:
            ch = self.secondary.chain(underlying, expiry, spot=spot, ts=ts)
        except Exception as exc:
            raise RuntimeError(f"{self.error[underlying]}; {self.secondary.name}: {exc!s:.160}") from exc
        self._held[underlying] = (now, ch)
        return ch

    def live_quotes(self, instruments) -> dict[str, tuple[float, float]]:
        return self.primary.live_quotes(instruments)


# ---- NSE (free) ---------------------------------------------------------------------------------
class NSEOptionChain(ChainSource):
    name = "nse"
    BASE = "https://www.nseindia.com"
    HEADERS = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                             "Chrome/130.0.0.0 Safari/537.36",
               "accept-language": "en-US,en;q=0.9", "accept-encoding": "gzip, deflate",
               "referer": "https://www.nseindia.com/option-chain"}
    INDICES = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"}

    def __init__(self, min_interval_sec: float = 3.0):
        import requests
        self.s = requests.Session()
        self.s.headers.update(self.HEADERS)
        self.min_interval = min_interval_sec
        self._last_call = 0.0
        self._warm = False

    def _get(self, url: str) -> dict:
        wait = self.min_interval - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(3):
            if not self._warm:
                self.s.get(f"{self.BASE}/option-chain", timeout=10)
                self._warm = True
            r = self.s.get(url, timeout=10)
            self._last_call = time.time()
            if r.status_code == 200 and r.text.strip().startswith("{"):
                data = r.json()
                if data:
                    return data
            self._warm = False                       # cookies expired or blocked: re-warm and retry
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"NSE returned no data for {url}")

    def expiries(self, underlying: str) -> list[dt.date]:
        d = self._get(f"{self.BASE}/api/option-chain-contract-info?symbol={underlying}")
        raw = d.get("expiryDates") or d.get("records", {}).get("expiryDates", [])
        return sorted(dt.datetime.strptime(x, "%d-%b-%Y").date() for x in raw)

    def chain(self, underlying: str, expiry: dt.date, spot=None, ts=None) -> pd.DataFrame:
        kind = "Indices" if underlying in self.INDICES else "Equity"
        e = expiry.strftime("%d-%b-%Y")
        d = self._get(f"{self.BASE}/api/option-chain-v3?type={kind}&symbol={underlying}&expiry={e}")
        return self.parse(d, underlying, expiry)

    @staticmethod
    def parse(d: dict, underlying: str, expiry: dt.date) -> pd.DataFrame:
        rec = d.get("records", d)
        rows = {}
        spot = rec.get("underlyingValue")
        for item in rec.get("data", []):
            exp_s = item.get("expiryDates") or item.get("expiryDate")
            if exp_s and dt.datetime.strptime(exp_s, "%d-%b-%Y").date() != expiry:
                continue
            k = float(item.get("strikePrice"))
            row = rows.setdefault(k, {c: np.nan for c in COLUMNS})
            for side in SIDES:
                o = item.get(side.upper())
                if not o:
                    continue
                spot = spot or o.get("underlyingValue")
                row[f"{side}_ltp"] = o.get("lastPrice")
                row[f"{side}_bid"] = o.get("buyPrice1", o.get("bidprice"))
                row[f"{side}_ask"] = o.get("sellPrice1", o.get("askPrice"))
                row[f"{side}_iv"] = o.get("impliedVolatility")
                row[f"{side}_oi"] = o.get("openInterest")
                row[f"{side}_doi"] = o.get("changeinOpenInterest")
                row[f"{side}_vol"] = o.get("totalTradedVolume")
        df = pd.DataFrame.from_dict(rows, orient="index", columns=COLUMNS).astype(float).sort_index()
        df.index.name = "strike"
        ts = rec.get("timestamp")
        try:
            ts = pd.Timestamp(dt.datetime.strptime(ts, "%d-%b-%Y %H:%M:%S"), tz=IST) if ts else pd.Timestamp.now(tz=IST)
        except (TypeError, ValueError):
            ts = pd.Timestamp.now(tz=IST)
        for c in [c for c in df.columns if c.endswith(("_bid", "_ask", "_iv"))]:
            df[c] = df[c].where(df[c] > 0)                     # NSE uses 0 for "no quote"
        df.attrs.update({"underlying": underlying, "spot": float(spot) if spot else np.nan, "expiry": expiry,
                         "ts": ts, "source": "nse"})
        return df


# ---- Kite (real quotes via a broker session) ---------------------------------------------------
class KiteOptionChain(ChainSource):
    name = "kite"

    def __init__(self, kite, strikes_each_side: int = 15):
        self.kite = kite
        self.n = strikes_each_side
        self._nfo = None

    def _instruments(self) -> pd.DataFrame:
        if self._nfo is None:
            self._nfo = pd.DataFrame(self.kite.instruments("NFO"))
            self._nfo["expiry"] = pd.to_datetime(self._nfo["expiry"]).dt.date
        return self._nfo

    def expiries(self, underlying: str) -> list[dt.date]:
        nfo = self._instruments()
        return sorted(set(nfo[(nfo["name"] == underlying) & nfo["instrument_type"].isin(["CE", "PE"])]["expiry"]))

    def chain(self, underlying, expiry, spot=None, ts=None) -> pd.DataFrame:
        nfo = self._instruments()
        sub = nfo[(nfo["name"] == underlying) & (nfo["expiry"] == expiry) & nfo["instrument_type"].isin(["CE", "PE"])]
        strikes = np.sort(sub["strike"].unique())
        if spot is not None and len(strikes):
            i = int(np.argmin(np.abs(strikes - spot)))
            strikes = strikes[max(0, i - self.n): i + self.n + 1]
        sub = sub[sub["strike"].isin(strikes)]
        keys = [f"NFO:{s}" for s in sub["tradingsymbol"]]
        quotes = {}
        for i in range(0, len(keys), 400):
            quotes.update(self.kite.quote(keys[i:i + 400]))
        rows = {}
        for r in sub.itertuples():
            q = quotes.get(f"NFO:{r.tradingsymbol}")
            if not q:
                continue
            side = r.instrument_type.lower()
            row = rows.setdefault(float(r.strike), {c: np.nan for c in COLUMNS})
            depth = q.get("depth", {})
            row[f"{side}_ltp"] = q.get("last_price")
            row[f"{side}_bid"] = (depth.get("buy") or [{}])[0].get("price") or np.nan
            row[f"{side}_ask"] = (depth.get("sell") or [{}])[0].get("price") or np.nan
            row[f"{side}_oi"] = q.get("oi")
            row[f"{side}_vol"] = q.get("volume")
            row[f"{side}_doi"] = (q.get("oi") or 0) - (q.get("oi_day_low") or q.get("oi") or 0)
        df = pd.DataFrame.from_dict(rows, orient="index", columns=COLUMNS).astype(float).sort_index()
        df.index.name = "strike"
        df.attrs.update({"underlying": underlying, "spot": spot, "expiry": expiry,
                         "ts": pd.Timestamp.now(tz=IST), "source": "kite"})
        return fill_iv(df, IntradayPricer())


# ---- model (fallback / offline) -----------------------------------------------------------------
class ModelOptionChain(ChainSource):
    """Chain priced from spot + ATM IV. `state_fn(underlying, ts) -> (spot, atm_iv)` supplies
    live inputs (e.g. last 1m close and India VIX × beta). Spreads and OI are modelled."""
    name = "model"

    def __init__(self, cfg, calendar, state_fn, pricer: IntradayPricer | None = None, n_strikes: int = 20):
        self.cfg, self.cal, self.state_fn = cfg, calendar, state_fn
        self.pricer = pricer or IntradayPricer(cfg.get("backtest.risk_free", 0.065), cfg.get("backtest.dividend_yield", 0.012))
        self.n = n_strikes

    def expiries(self, underlying: str, asof=None) -> list[dt.date]:
        spec = self.cfg.instrument_spec(underlying)
        asof = pd.Timestamp(asof or pd.Timestamp.now(tz=IST)).date()
        return self.cal.expiries(asof, 70, int(spec.get("expiry_weekday", 1)), bool(spec.get("weekly_expiry", True)))

    def chain(self, underlying, expiry, spot=None, ts=None) -> pd.DataFrame:
        ts = pd.Timestamp(ts or pd.Timestamp.now(tz=IST))
        S, atm = self.state_fn(underlying, ts)
        spot = spot or S
        step = float(self.cfg.instrument_spec(underlying).get("strike_step", 50))
        T = time_to_expiry(ts, expiry)
        atm_k = round(spot / step) * step
        K = atm_k + step * np.arange(-self.n, self.n + 1)
        F = spot * np.exp((self.pricer.r - self.pricer.q) * T)
        iv = np.asarray(self.pricer.skew.iv(atm, K, F, max(T, 1 / 365 / 24)), dtype=float)
        cols = {}
        for side in SIDES:
            right = side.upper()
            m = np.asarray(bs_price(spot, K, max(T, 1e-9), self.pricer.r, self.pricer.q, iv, right), dtype=float)
            half = np.maximum(0.05, 0.004 * m + 0.05 * (1 + np.abs(K - spot) / (spot * 0.02)))
            dist = (K - spot) / (spot * max(atm, 0.05) * np.sqrt(max(T, 1 / 365)))
            oi = 5e6 * np.exp(-0.5 * (dist - (0.8 if right == "CE" else -0.8)) ** 2) * np.where(K % (step * 10) == 0, 1.8, 1.0)
            cols.update({f"{side}_ltp": np.round(m / 0.05) * 0.05, f"{side}_bid": np.maximum(0.05, np.round((m - half) / 0.05) * 0.05),
                         f"{side}_ask": np.round((m + half) / 0.05) * 0.05, f"{side}_iv": iv * 100, f"{side}_oi": oi,
                         f"{side}_doi": np.zeros(len(K)), f"{side}_vol": oi * 4})
        rows = pd.DataFrame(cols, index=K.astype(float))[COLUMNS].to_dict("index")
        df = pd.DataFrame.from_dict(rows, orient="index", columns=COLUMNS).astype(float)
        df.index.name = "strike"
        df.attrs.update({"underlying": underlying, "spot": spot, "expiry": expiry, "ts": ts, "source": "model"})
        return df


class RecordedChains(ChainSource):
    """Chain snapshots saved by the recorder; returns the latest snapshot at or before `ts`."""
    name = "recorded"

    def __init__(self, snapshots: dict[str, list[pd.DataFrame]]):
        self.snaps = {u: sorted(v, key=lambda d: d.attrs["ts"]) for u, v in snapshots.items()}

    def expiries(self, underlying: str) -> list[dt.date]:
        return sorted({d.attrs["expiry"] for d in self.snaps.get(underlying, [])})

    def chain(self, underlying, expiry, spot=None, ts=None) -> pd.DataFrame:
        cands = [d for d in self.snaps.get(underlying, []) if d.attrs["expiry"] == expiry
                 and (ts is None or d.attrs["ts"] <= pd.Timestamp(ts))]
        if not cands:
            raise LookupError(f"no recorded chain for {underlying} {expiry} at {ts}")
        return cands[-1]


# ---- analytics ------------------------------------------------------------------------------------
def fill_iv(df: pd.DataFrame, pricer: IntradayPricer) -> pd.DataFrame:
    """Fill missing IVs from the quote mid (or LTP) so every strike has an IV (%)."""
    S, exp, ts = df.attrs.get("spot"), df.attrs.get("expiry"), df.attrs.get("ts")
    if not S or exp is None or df.empty:
        return df
    T = time_to_expiry(ts, exp)
    for side in SIDES:
        iv = df[f"{side}_iv"].to_numpy(dtype=float, copy=True)
        missing = ~(iv > 0)
        if not missing.any():
            continue
        bid, ask, ltp = (df[f"{side}_{f}"].to_numpy(dtype=float) for f in ("bid", "ask", "ltp"))
        px = np.where((bid > 0) & (ask > bid), (bid + ask) / 2, ltp)
        for i in np.flatnonzero(missing & (px > 0)):
            iv[i] = pricer.implied(float(px[i]), float(df.index[i]), side.upper(), float(S), T) * 100
        df[f"{side}_iv"] = iv
    return df


def mid(row, side: str) -> float:
    b, a, l = row.get(f"{side}_bid"), row.get(f"{side}_ask"), row.get(f"{side}_ltp")
    if pd.notna(b) and pd.notna(a) and a >= b > 0:
        return (a + b) / 2
    return float(l) if pd.notna(l) else float("nan")


def chain_analytics(df: pd.DataFrame, pricer: IntradayPricer | None = None) -> dict:
    """ATM IV, implied move, PCR, max pain, OI walls, OI build-up, skew, ATM spread."""
    if df is None or df.empty:
        return {}
    pricer = pricer or IntradayPricer()
    S, exp, ts = float(df.attrs["spot"]), df.attrs["expiry"], df.attrs["ts"]
    T = time_to_expiry(ts, exp)
    ks = df.index.to_numpy(dtype=float)
    atm = float(ks[np.argmin(np.abs(ks - S))])
    row = df.loc[atm]
    ce_m, pe_m = mid(row, "ce"), mid(row, "pe")
    ivs = [v for v in (row.get("ce_iv"), row.get("pe_iv")) if pd.notna(v) and v > 0]
    atm_iv = float(np.mean(ivs)) if ivs else float("nan")
    straddle = ce_m + pe_m
    out = {"spot": S, "expiry": str(exp), "dte_days": T * 365, "atm_strike": atm, "atm_iv": atm_iv,
           "straddle": straddle, "implied_move": straddle / S if S else np.nan, "source": df.attrs.get("source")}
    if pd.notna(row.get("ce_bid")) and pd.notna(row.get("ce_ask")) and ce_m > 0:
        out["atm_spread_pct"] = float((row["ce_ask"] - row["ce_bid"]) / ce_m)
    ce_oi, pe_oi = df["ce_oi"].fillna(0), df["pe_oi"].fillna(0)
    if ce_oi.sum() > 0:
        out["pcr_oi"] = float(pe_oi.sum() / ce_oi.sum())
        d_ce, d_pe = df["ce_doi"].fillna(0).sum(), df["pe_doi"].fillna(0).sum()
        out["pcr_doi"] = float(d_pe / d_ce) if d_ce > 0 else np.nan
        above, below = df[df.index > S], df[df.index < S]
        out["call_wall"] = float(above["ce_oi"].idxmax()) if len(above) and above["ce_oi"].max() > 0 else np.nan
        out["put_wall"] = float(below["pe_oi"].idxmax()) if len(below) and below["pe_oi"].max() > 0 else np.nan
        grid = ks[:, None]
        pain = (np.maximum(grid - ks[None, :], 0) * ce_oi.to_numpy()[None, :]).sum(1) + \
               (np.maximum(ks[None, :] - grid, 0) * pe_oi.to_numpy()[None, :]).sum(1)
        out["max_pain"] = float(ks[int(np.argmin(pain))])
        cd, pdoi = df["ce_doi"].fillna(0), df["pe_doi"].fillna(0)
        out["top_call_adds"] = [float(k) for k in cd[cd > 0].nlargest(2).index]
        out["top_put_adds"] = [float(k) for k in pdoi[pdoi > 0].nlargest(2).index]
    try:                                                # 25-delta risk reversal (put IV − call IV)
        sig = atm_iv / 100 if atm_iv == atm_iv else 0.15
        from ..options.pricing import strike_for_delta
        kp = strike_for_delta(S, max(T, 1 / 365), pricer.r, pricer.q, 0.25, "PE", sigma=sig)
        kc = strike_for_delta(S, max(T, 1 / 365), pricer.r, pricer.q, 0.25, "CE", sigma=sig)
        ivp = float(np.interp(kp, ks, df["pe_iv"].interpolate().bfill().ffill()))
        ivc = float(np.interp(kc, ks, df["ce_iv"].interpolate().bfill().ffill()))
        out["skew_25d"] = ivp - ivc
    except Exception:  # sparse chains
        out["skew_25d"] = np.nan
    return out


def option_instrument(underlying: str, expiry: dt.date, strike: float, right: str, lot: int) -> Instrument:
    return Instrument.option(underlying, expiry, strike, right, lot)


def save_chain(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for k in ("underlying", "spot", "expiry", "ts", "source"):
        out[f"_{k}"] = str(df.attrs.get(k))
    out.to_csv(path)


def load_chain(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, index_col=0)
    df = raw[COLUMNS].astype(float)
    df.index.name = "strike"
    a = raw.iloc[0]
    df.attrs.update({"underlying": a["_underlying"], "spot": float(a["_spot"]),
                     "expiry": dt.date.fromisoformat(str(a["_expiry"])), "ts": pd.Timestamp(a["_ts"]), "source": a["_source"]})
    return df
