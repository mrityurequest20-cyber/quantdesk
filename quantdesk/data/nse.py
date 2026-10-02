"""NSE's public end-of-day files and a few public API endpoints, fetched politely and parsed into tidy frames.

Formats were taken from the real files (deploy/probe_nse.py, 2 Oct 2026):

  F&O bhavcopy, UDiFF (from 8 Jul 2024)  /content/fo/BhavCopy_NSE_FO_0_0_0_YYYYMMDD_F_0000.csv.zip
      TradDt, FinInstrmTp (IDO index option, IDF index future, STO/STF stock), TckrSymb, XpryDt, StrkPric, OptnTp,
      Opn/Hgh/Lw/Cls/LastPric, PrvsClsgPric, UndrlygPric, SttlmPric, OpnIntrst (units), ChngInOpnIntrst,
      TtlTradgVol (contracts), TtlTrfVal (₹, notional), NewBrdLotQty (lot)
  F&O bhavcopy, old format (until 5 Jul 2024)  /content/historical/DERIVATIVES/YYYY/MON/foDDMONYYYYbhav.csv.zip
      INSTRUMENT (OPTIDX/FUTIDX/…), SYMBOL, EXPIRY_DT, STRIKE_PR, OPTION_TYP, OPEN…CLOSE, SETTLE_PR, CONTRACTS,
      VAL_INLAKH, OPEN_INT, CHG_IN_OI, TIMESTAMP — no underlying price and no lot size
  Participant-wise OI / volume  /content/nsccl/fao_participant_{oi,vol}_DDMMYYYY.csv
      a title line, then: Client Type, Future Index Long/Short, Future Stock Long/Short, Option Index Call/Put
      Long/Short, Option Stock …, Total Long/Short Contracts; rows Client, DII, FII, Pro, TOTAL
  /api/fiidiiTradeReact   the latest day's FII/FPI and DII cash buy/sell/net (₹ crore); no history
  /api/marketStatus       includes "giftnifty" (the NSE IX NIFTY future: last, change, expiry, timestamp)
  /api/event-calendar     upcoming board meetings (results etc.)
  /api/holiday-master?type=trading   the exchange's holiday list per segment

Only index derivatives are kept from the bhavcopy (NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, NIFTYNXT50): the
stock contracts are 90% of the file and nothing here uses them."""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import time
import zipfile

import numpy as np
import pandas as pd

ARCHIVE = "https://nsearchives.nseindia.com"
SITE = "https://www.nseindia.com"
INDEX_SYMBOLS = ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50")
UDIFF_FROM = dt.date(2024, 7, 8)                 # first day NSE published the F&O bhavcopy in UDiFF
UA = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/130.0.0.0 Safari/537.36",
      "accept-language": "en-US,en;q=0.9", "accept": "*/*", "referer": "https://www.nseindia.com/"}
BHAV_COLS = ["date", "symbol", "kind", "expiry", "strike", "open", "high", "low", "close", "last", "prev_close",
             "settle", "underlying", "oi", "chg_oi", "contracts", "lot", "src"]


class NotPublished(Exception):
    """NSE has no file for that day (a holiday, a weekend, or not published yet)."""


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---- the client ----------------------------------------------------------------------------------------------------
class NSE:
    """A polite session: a browser user agent, ≥`gap` seconds between requests, retries with backoff, and the
    cookie warm-up the API endpoints need. Returns (bytes, url)."""

    def __init__(self, session=None, gap: float = 0.4, tries: int = 3):
        if session is None:
            import requests
            session = requests.Session()
        self.s = session
        self.s.headers.update(UA)
        self.gap, self.tries = gap, tries
        self._last = 0.0
        self._warm = False
        self.calls = 0

    def _get(self, url: str, timeout: float = 30):
        for attempt in range(self.tries):
            wait = self.gap - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                r = self.s.get(url, timeout=timeout)
            except Exception:
                if attempt == self.tries - 1:
                    raise
                time.sleep(2 * (attempt + 1))
                continue
            finally:
                self._last = time.monotonic()
                self.calls += 1
            if r.status_code == 404:
                raise NotPublished(url)
            if r.status_code == 200:
                return r
            if attempt < self.tries - 1:
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"NSE answered HTTP {r.status_code} for {url}")

    def archive(self, path: str) -> tuple[bytes, str]:
        url = f"{ARCHIVE}{path}"
        r = self._get(url)
        if r.content[:15].lower().startswith((b"<!doctype", b"<html")):
            raise NotPublished(url)                  # NSE serves an HTML error page with HTTP 200 sometimes
        return r.content, url

    def api(self, path: str):
        if not self._warm:
            for p in ("/", "/option-chain"):
                try:
                    self.s.get(f"{SITE}{p}", timeout=20)
                except Exception:
                    pass
            self._warm = True
        url = f"{SITE}{path}"
        r = self._get(url)
        try:
            return r.json(), r.content, url
        except ValueError:
            self._warm = False
            raise RuntimeError(f"NSE API {path} didn't return JSON (blocked or changed)") from None

    # ---- end-of-day files -----------------------------------------------------------------------------------------
    def fo_bhav(self, day: dt.date) -> tuple[bytes, str]:
        if day >= UDIFF_FROM:
            return self.archive(f"/content/fo/BhavCopy_NSE_FO_0_0_0_{day:%Y%m%d}_F_0000.csv.zip")
        mon = day.strftime("%b").upper()
        return self.archive(f"/content/historical/DERIVATIVES/{day:%Y}/{mon}/fo{day:%d}{mon}{day:%Y}bhav.csv.zip")

    def participant(self, kind: str, day: dt.date) -> tuple[bytes, str]:
        assert kind in ("oi", "vol")
        return self.archive(f"/content/nsccl/fao_participant_{kind}_{day:%d%m%Y}.csv")


# ---- parsers --------------------------------------------------------------------------------------------------------
def _unzip_csv(b: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        return pd.read_csv(z.open(name), low_memory=False)


def parse_fo_bhav(b: bytes, day: dt.date, symbols=INDEX_SYMBOLS) -> pd.DataFrame:
    """Index futures and options from either bhavcopy format → BHAV_COLS (one row per contract)."""
    raw = _unzip_csv(b)
    raw.columns = [str(c).strip() for c in raw.columns]
    if "FinInstrmTp" in raw.columns:                                         # UDiFF
        d = raw[raw["FinInstrmTp"].isin(["IDO", "IDF"]) & raw["TckrSymb"].isin(symbols)]
        kind = np.where(d["FinInstrmTp"] == "IDF", "FUT", d["OptnTp"].astype(str).str.strip())
        out = pd.DataFrame({
            "date": pd.to_datetime(d["TradDt"]).dt.date, "symbol": d["TckrSymb"].values, "kind": kind,
            "expiry": pd.to_datetime(d["XpryDt"]).dt.date, "strike": pd.to_numeric(d["StrkPric"], errors="coerce"),
            "open": d["OpnPric"], "high": d["HghPric"], "low": d["LwPric"], "close": d["ClsPric"], "last": d["LastPric"],
            "prev_close": d["PrvsClsgPric"], "settle": d["SttlmPric"], "underlying": d["UndrlygPric"],
            "oi": d["OpnIntrst"], "chg_oi": d["ChngInOpnIntrst"], "contracts": d["TtlTradgVol"],
            "lot": d["NewBrdLotQty"], "src": "udiff"})
    elif "INSTRUMENT" in raw.columns:                                        # the old format
        d = raw[raw["INSTRUMENT"].astype(str).str.strip().isin(["OPTIDX", "FUTIDX"])
                & raw["SYMBOL"].astype(str).str.strip().isin(symbols)]
        kind = np.where(d["INSTRUMENT"].str.strip() == "FUTIDX", "FUT", d["OPTION_TYP"].astype(str).str.strip())
        out = pd.DataFrame({
            "date": pd.to_datetime(d["TIMESTAMP"], format="%d-%b-%Y").dt.date, "symbol": d["SYMBOL"].str.strip().values,
            "kind": kind, "expiry": pd.to_datetime(d["EXPIRY_DT"], format="%d-%b-%Y").dt.date,
            "strike": pd.to_numeric(d["STRIKE_PR"], errors="coerce"),
            "open": d["OPEN"], "high": d["HIGH"], "low": d["LOW"], "close": d["CLOSE"], "last": np.nan,
            "prev_close": np.nan, "settle": d["SETTLE_PR"], "underlying": np.nan, "oi": d["OPEN_INT"],
            "chg_oi": d["CHG_IN_OI"], "contracts": d["CONTRACTS"], "lot": np.nan, "src": "old"})
    else:
        raise ValueError(f"unrecognised F&O bhavcopy columns: {list(raw.columns)[:8]}")
    out.loc[out["kind"] == "FUT", "strike"] = np.nan
    for c in ("open", "high", "low", "close", "last", "prev_close", "settle", "underlying", "oi", "chg_oi", "contracts",
              "lot", "strike"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    if len(out) and (out["date"] != day).any():
        raise ValueError(f"bhavcopy for {day} carries trade dates {sorted(set(out['date']))[:3]}")
    return out[BHAV_COLS].reset_index(drop=True)


PARTICIPANT_COLS = {
    "future index long": "fut_idx_long", "future index short": "fut_idx_short",
    "future stock long": "fut_stk_long", "future stock short": "fut_stk_short",
    "option index call long": "opt_idx_call_long", "option index put long": "opt_idx_put_long",
    "option index call short": "opt_idx_call_short", "option index put short": "opt_idx_put_short",
    "option stock call long": "opt_stk_call_long", "option stock put long": "opt_stk_put_long",
    "option stock call short": "opt_stk_call_short", "option stock put short": "opt_stk_put_short",
    "total long contracts": "total_long", "total short contracts": "total_short"}


def parse_participant(b: bytes, day: dt.date) -> pd.DataFrame:
    """Participant-wise OI or volume CSV → one row per participant (Client, DII, FII, Pro, TOTAL), contracts."""
    lines = b.decode("utf-8-sig", "replace").splitlines()
    head = next(i for i, ln in enumerate(lines) if ln.lower().lstrip('"').startswith("client type"))
    df = pd.read_csv(io.StringIO("\n".join(lines[head:])))
    df.columns = [" ".join(str(c).replace("\t", " ").split()).lower() for c in df.columns]
    df = df.rename(columns={"client type": "participant", **PARTICIPANT_COLS})
    missing = set(PARTICIPANT_COLS.values()) - set(df.columns)
    if missing:
        raise ValueError(f"participant file for {day}: missing columns {sorted(missing)}")
    df["participant"] = df["participant"].astype(str).str.strip()
    df = df[df["participant"].isin(["Client", "DII", "FII", "Pro", "TOTAL"])]
    for c in PARTICIPANT_COLS.values():
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "").str.strip(), errors="coerce").astype("float64")
    df.insert(0, "date", day)
    return df[["date", "participant", *PARTICIPANT_COLS.values()]].reset_index(drop=True)


def parse_fii_dii(rows) -> pd.DataFrame:
    """/api/fiidiiTradeReact → date, category (FII/FPI, DII), buy, sell, net (₹ crore, provisional)."""
    out = pd.DataFrame([{"date": dt.datetime.strptime(r["date"], "%d-%b-%Y").date(),
                         "category": "FII" if "FII" in r["category"] else str(r["category"]),
                         "buy": float(r["buyValue"]), "sell": float(r["sellValue"]), "net": float(r["netValue"])}
                        for r in rows if isinstance(r, dict) and r.get("date")])
    return out if len(out) else pd.DataFrame(columns=["date", "category", "buy", "sell", "net"])


def parse_gift(status: dict, fetched: pd.Timestamp | None = None) -> pd.DataFrame:
    """/api/marketStatus → one GIFT Nifty print: ts (IST), last, change, pct, expiry, contracts; plus NIFTY's last
    close from the same response, for the implied gap."""
    g = (status or {}).get("giftnifty") or {}
    if not g.get("LASTPRICE"):
        return pd.DataFrame()
    ts = pd.Timestamp(dt.datetime.strptime(g["TIMESTMP"], "%d-%b-%Y %H:%M"), tz="Asia/Kolkata")
    nifty = next((m for m in status.get("marketState", []) if m.get("index") == "NIFTY 50"), {})
    ind = status.get("indicativenifty50") or {}
    close = ind.get("finalClosingValue") or ind.get("closingValue") or nifty.get("last")
    return pd.DataFrame([{"ts": ts, "last": float(g["LASTPRICE"]), "change": float(g.get("DAYCHANGE") or 0),
                          "pct": float(g.get("PERCHANGE") or 0),
                          "expiry": dt.datetime.strptime(g["EXPIRYDATE"], "%d-%b-%Y").date(),
                          "contracts": float(g.get("CONTRACTSTRADED") or 0),
                          "nifty_close": float(close) if close not in (None, "") else np.nan,
                          "fetched": fetched if fetched is not None else pd.Timestamp.now(tz="Asia/Kolkata")}])


def gift_implied_gap(last: float, nifty_close: float, expiry: dt.date, on: dt.date, r: float = 0.065,
                     q: float = 0.012) -> float:
    """The opening gap GIFT Nifty implies: its premium over NIFTY's close minus the fair carry to its expiry
    (it is a future, so it trades above spot by about (r − q)·T even when nothing has happened)."""
    T = max((expiry - on).days, 0) / 365
    fair = nifty_close * np.exp((r - q) * T)
    return float(last / fair - 1)


def parse_events(rows) -> pd.DataFrame:
    """/api/event-calendar → date, symbol, company, purpose, desc."""
    out = pd.DataFrame([{"date": dt.datetime.strptime(r["date"], "%d-%b-%Y").date(), "symbol": r.get("symbol", ""),
                         "company": r.get("company", ""), "purpose": r.get("purpose", ""),
                         "desc": str(r.get("bm_desc", ""))[:300]} for r in rows if isinstance(r, dict) and r.get("date")])
    return out if len(out) else pd.DataFrame(columns=["date", "symbol", "company", "purpose", "desc"])


def parse_holidays(d: dict, segment: str = "FO") -> list[tuple[dt.date, str]]:
    return sorted((dt.datetime.strptime(x["tradingDate"], "%d-%b-%Y").date(), x.get("description", ""))
                  for x in (d or {}).get(segment, []) if x.get("tradingDate"))


def to_json_bytes(x) -> bytes:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"), default=str).encode()
