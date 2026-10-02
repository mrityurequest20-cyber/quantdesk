"""Kotak Neo data: the consumer-key endpoints (quotes, option chain, expiries, candles), the chain built from them,
the fallback to NSE, bars with a Yahoo fallback, and paper fills priced off the live book. Response shapes are the
ones in Kotak's official SDK docs (kotak-neo-python 3.0.x, docs/functions/market_data)."""
import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

import quantdesk.intraday.kotak as kotak_mod
from quantdesk.core.calendar import TradingCalendar
from quantdesk.core.types import Instrument
from quantdesk.intraday.chains import FallbackChain
from quantdesk.intraday.engine import IntradayEngine, run_replay
from quantdesk.intraday.feeds import IST, ReplayFeed, YahooIntradayFeed
from quantdesk.intraday.kotak import (KotakClient, KotakError, KotakIntradayFeed, KotakOptionChain, _error, best,
                                      check)
from quantdesk.intraday.sim import IntradayBroker
from quantdesk.intraday.synthetic import simulate_sessions
from quantdesk.journal.journal import Journal

EXP = dt.date(2026, 10, 6)


class Resp:
    def __init__(self, body, status=200, headers=None):
        self.body, self.status_code, self.headers = body, status, headers or {}
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        if isinstance(self.body, str):
            raise ValueError("not json")
        return self.body


class FakeKotak:
    """Routes GETs like Kotak's backend: a NIFTY chain of 5 strikes around 24,850 with 5-level depth. `shape`:
    "live" is what the API actually sent on 2 Oct 2026 (inst/strkPrc, oi cur/prev/chg with chg = the price
    change); "docs" is the shape in Kotak's SDK docs."""

    def __init__(self, strikes=(24750, 24800, 24850, 24900, 24950), spot=24852.35, quoted=True, shape="live"):
        self.headers, self.log = {}, []
        self.strikes, self.spot, self.quoted, self.shape = strikes, spot, quoted, shape
        self.tok = {}
        for i, k in enumerate(strikes):
            self.tok[(k, "CE")] = str(71000 + 2 * i)
            self.tok[(k, "PE")] = str(71001 + 2 * i)

    def price(self, k, right):
        intrinsic = max(self.spot - k, 0) if right == "CE" else max(k - self.spot, 0)
        return round(intrinsic + 95 - abs(k - self.spot) * 0.3, 2)

    def get(self, url, params=None, timeout=None):
        self.log.append((url, params))
        path = url.split("mis.kotaksecurities.com/")[1]
        if path.startswith("market-data/1.0/watchlist/expiries"):
            return Resp({"exchange": "nse_fo", "underlying": params["underlying"],
                         "expiries": ["2026-10-06", "2026-10-13", "2026-10-27"]})
        if path.startswith("market-data/1.0/watchlist/option-chain"):
            def leg(k, r):
                if self.shape == "docs":
                    return {"instrument": {"neoSymbol": f"nse_fo|{self.tok[(k, r)]}", "symbol": f"NIFTY26OCT{k}{r}",
                                           "optionType": r, "strikePrice": str(k), "moneyness": "ATM"},
                            "quote": {"ltp": f"{self.price(k, r):.4f}", "volume": 1000 + k % 7},
                            "openInterest": {"current": 500000, "previous": 450000, "change": 50000, "changePct": 11.1}}
                return {"inst": {"neoSymbol": f"nse_fo|{self.tok[(k, r)]}", "symbol": f"NIFTY26O06{k}{r}", "optType": r,
                                 "strkPrc": str(k), "exp": "2026-10-06", "moneyness": "atm"},
                        "quote": {"ltp": f"{self.price(k, r)}", "o": "1", "h": "1", "l": "1", "c": "1", "pc": "1",
                                  "vol": str(1000 + k % 7)},
                        "oi": {"cur": "500000", "prev": "450000", "chg": "-391.85", "chgPct": "-26.77"}}
            d = {"common_data": {"mktLot": "65", "multiplier": "1", "expiryDt": params.get("expiry"), "unlSymbol": "NIFTY",
                                 "exSeg": "nse_fo"},
                 "call": [leg(k, "CE") for k in self.strikes], "put": [leg(k, "PE") for k in self.strikes]}
            return Resp({"data": d} if self.shape == "docs" else d)
        if path.startswith("script-details/1.0/quotes/neosymbol/"):
            syms = path.split("/neosymbol/")[1].rsplit("/", 1)[0].split(",")
            out = []
            for s in syms:
                seg, tok = s.split("|", 1)
                if seg == "nse_cm":
                    out.append({"exchange_token": tok.replace("%20", " "), "display_symbol": "NIFTY 50", "exchange": "nse_cm",
                                "ltp": f"{self.spot:.4f}", "lstup_time": str(int(pd.Timestamp.now().timestamp())),
                                "depth": {"buy": [], "sell": []}})
                    continue
                k, r = next(key for key, v in self.tok.items() if v == tok)
                p = self.price(k, r)
                depth = ({"buy": [{"price": f"{p - 0.4:.4f}", "quantity": "650", "orders": "3"}],
                          "sell": [{"price": f"{p + 0.4:.4f}", "quantity": "325", "orders": "2"}]}
                         if self.quoted else {"buy": [{"price": "0.0000"}], "sell": [{"price": "0.0000"}]})
                out.append({"exchange_token": tok, "display_symbol": f"NIFTY26OCT{k}{r}", "exchange": "nse_fo",
                            "ltp": f"{p:.4f}", "last_volume": "123450", "open_int": "520000", "depth": depth})
            return Resp(out)
        if path.startswith("market-data/1.0/historical/details"):
            day = params["fromdate"]
            rows = [[f"{day}T09:{15 + i:02d}:00+0530", 24800 + i, 24805 + i, 24795 + i, 24801 + i, 0, 0] for i in range(10)]
            return Resp({"status": "success", "interval": "1min", "data": {"candles": rows}})
        return Resp({"code": 404, "message": "no such route"}, 404)


@pytest.fixture
def fake():
    return FakeKotak()


@pytest.fixture
def client(fake):
    return KotakClient("ck-token", session=fake, min_gap=0)


def test_needs_a_consumer_key_and_sends_it_as_is():
    with pytest.raises(KotakError):
        KotakClient.from_env({})
    f = FakeKotak()
    KotakClient.from_env({"KOTAK_CONSUMER_KEY": " abc "}, session=f)
    assert f.headers["Authorization"] == "abc"                                   # no "Bearer", trimmed


def test_quotes_go_25_a_call_with_literal_delimiters(client, fake):
    insts = [("nse_fo", fake.tok[(24850, "CE")])] * 120 + [("nse_cm", "Nifty 50")]
    qs = client.quotes(insts)
    calls = [u for u, _ in fake.log if "/neosymbol/" in u]
    assert len(calls) == 5 and len(qs) == 121                                   # 25 a call
    assert "nse_fo|71004,nse_fo|71004" in calls[0] and "nse_cm|Nifty%2050" in calls[4]


def test_error_shapes_raise():
    assert _error({"code": 400, "message": "Failed to call upstream quote API"})
    assert _error({"status": "ERROR", "fault": {"code": 400, "message": "Invalid interval value"}}) == "Invalid interval value"
    assert _error({"Error Message": "Complete the 2fa process before accessing this application"})
    assert _error([{"ltp": "1"}]) is None and _error({"data": {"candles": []}}) is None

    class Upstream(FakeKotak):
        def get(self, url, params=None, timeout=None):
            return Resp({"code": 400, "message": "Failed to call upstream quote API"}, 400)
    with pytest.raises(KotakError, match="upstream"):
        KotakClient("k", session=Upstream(), min_gap=0).quotes([("nse_cm", "Nifty 50")])

    class Html(FakeKotak):
        def get(self, url, params=None, timeout=None):
            return Resp("<html>gateway timeout</html>", 504)
    with pytest.raises(KotakError, match="not JSON"):
        KotakClient("k", session=Html(), min_gap=0).expiries("NIFTY")


def test_quote_batches_shrink_when_kotak_says_max_value(fake):
    class Strict(FakeKotak):
        def get(self, url, params=None, timeout=None):
            if "/neosymbol/" in url and url.split("/neosymbol/")[1].count("|") > 20:
                return Resp({"fault": {"code": "400", "message": "Please set the Neo symbol max value to 50."}}, 400)
            return super().get(url, params, timeout)
    s = Strict()
    k = KotakClient("k", session=s, min_gap=0)
    qs = k.quotes([("nse_fo", s.tok[(24850, "CE")])] * 83)
    assert len(qs) == 83 and k.batch == 12                                      # 25 → 12, and it stays 12


def test_a_429_is_retried_once(monkeypatch):
    monkeypatch.setattr(kotak_mod.time, "sleep", lambda s: None)

    class Busy(FakeKotak):
        n = 0

        def get(self, url, params=None, timeout=None):
            self.n += 1
            if self.n == 1:
                return Resp({"code": 429, "message": "Too many requests"}, 429, {"Retry-After": "1"})
            return super().get(url, params, timeout)
    b = Busy()
    assert KotakClient("k", session=b, min_gap=0).expiries("NIFTY")[0] == EXP and b.n == 2


@pytest.mark.parametrize("shape", ["live", "docs"])
def test_chain_from_the_live_book(shape):
    fake = FakeKotak(shape=shape)
    client = KotakClient("ck-token", session=fake, min_gap=0)
    ch = KotakOptionChain(client, strikes=17)
    assert ch.count == 20                                                       # each side; a multiple of 10
    assert ch.expiries("NIFTY") == [EXP, dt.date(2026, 10, 13), dt.date(2026, 10, 27)]
    df = ch.chain("NIFTY", EXP)
    assert list(df.index) == [24750.0, 24800.0, 24850.0, 24900.0, 24950.0]
    assert df.attrs["source"] == "kotak" and df.attrs["spot"] == pytest.approx(24852.35) and df.attrs["lot"] == 65
    r = df.loc[24850.0]
    p = fake.price(24850, "CE")
    assert (r.ce_bid, r.ce_ask, r.ce_ltp) == (pytest.approx(p - 0.4), pytest.approx(p + 0.4), pytest.approx(p))
    assert r.ce_oi == 520000 and r.ce_doi == 50000 and r.ce_vol == 123450       # OI change = cur − prev, not "chg"
    assert (df[["ce_iv", "pe_iv"]] > 0).all().all()                             # IVs from the quote mids
    n_quote_calls = sum("/neosymbol/" in u for u, _ in fake.log)
    assert n_quote_calls == 1                                                    # 10 contracts + the index: one call
    ch.expiries("NIFTY")
    assert sum("expiries" in u for u, _ in fake.log) == 1                       # cached for the day


def test_an_empty_book_is_an_error_not_a_chain():
    ch = KotakOptionChain(KotakClient("k", session=FakeKotak(quoted=False), min_gap=0))
    with pytest.raises(KotakError, match="no bid/ask"):
        ch.chain("NIFTY", EXP)
    df = ch.chain("NIFTY", EXP, require_quotes=False)                           # after hours: LTPs, tokens, OI
    assert df.attrs["quoted"] == 0 and (df["ce_ltp"] > 0).all() and len(ch.tokens) == 10


def test_live_quotes_for_held_contracts(client, fake):
    ch = KotakOptionChain(client)
    assert ch.live_quotes([Instrument.option("NIFTY", EXP, 24850, "PE", 65)]) == {}   # no token known yet
    ch.chain("NIFTY", EXP)
    held = [Instrument.option("NIFTY", EXP, 24850, "PE", 65), Instrument.option("NIFTY", EXP, 24950, "CE", 65)]
    live = ch.live_quotes(held)
    p = fake.price(24850, "PE")
    assert live[held[0].symbol] == (pytest.approx(p - 0.4), pytest.approx(p + 0.4)) and len(live) == 2


class Fixed:
    name = "nse"
    refresh_min = None

    def __init__(self):
        self.calls = 0

    def expiries(self, u):
        return [EXP]

    def chain(self, u, expiry, spot=None, ts=None):
        self.calls += 1
        df = pd.DataFrame({"ce_bid": [1.0]}, index=[24850.0])
        df.attrs.update({"source": "nse", "expiry": expiry, "ts": pd.Timestamp.now(tz=IST)})
        return df

    def live_quotes(self, insts):
        return {}


def test_fallback_to_nse_without_hammering_it(client):
    class Down(KotakOptionChain):
        def chain(self, *a, **k):
            raise KotakError("503 Trade API service is unavailable")
    nse = Fixed()
    fb = FallbackChain(Down(client), nse, secondary_min=3)
    assert fb.name == "kotak" and fb.refresh_min == 1
    a = fb.chain("NIFTY", EXP)
    b = fb.chain("NIFTY", EXP)
    assert a.attrs["source"] == "nse" and b is a and nse.calls == 1            # NSE asked once in 3 minutes
    assert "503" in fb.error["NIFTY"]
    ok = FallbackChain(KotakOptionChain(client), nse)
    assert ok.chain("NIFTY", EXP).attrs["source"] == "kotak" and nse.calls == 1


def test_bars_from_kotak_with_yahoo_behind(cfg, client, monkeypatch):
    feed = KotakIntradayFeed(cfg, client)
    day = dt.date(2026, 10, 5)
    monkeypatch.setattr(feed, "now", lambda: pd.Timestamp(f"{day} 09:25:30", tz=IST))
    bars = feed.poll("NIFTY", None)
    assert len(bars) == 10 and bars.index[-1] == pd.Timestamp(f"{day} 09:24", tz=IST)
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"] and feed.served["kotak"] == 1
    assert feed.name == "kotak"
    assert len(feed.poll("NIFTY", pd.Timestamp(f"{day} 09:22", tz=IST))) == 2

    yahoo = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [0.0]},
                         index=[pd.Timestamp(f"{day} 09:24", tz=IST)])
    monkeypatch.setattr(YahooIntradayFeed, "poll", lambda self, s, since: yahoo)
    monkeypatch.setattr(feed, "kotak_bars", lambda s, d: (_ for _ in ()).throw(KotakError("down")))
    for _ in range(5):
        assert feed.poll("NIFTY", None) is yahoo
    assert feed.rest_until is not None and feed.served["yahoo"] == 5            # rested after 5 failures
    assert feed.name == "kotak+yahoo (5 of 7 polls from Yahoo)"


def test_check_reports_what_works(client):
    lines = []
    assert check(client, ["NIFTY"], say=lines.append)
    text = "\n".join(lines)
    assert "ltp" in text and "5 strikes" in text and "lot 65" in text and "one-minute bars" in text
    assert "ck-token" not in text                                               # never prints the key


# ---- the desk on a live book --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sessions():
    from quantdesk.config import DEFAULT_CONFIG, Config
    cal = TradingCalendar(Config.load(DEFAULT_CONFIG).holidays())
    days = [d.date() for d in cal.trading_days("2026-08-17", "2026-09-28")]
    bars, _ = simulate_sessions(days, seed=5)
    return bars, days


class LiveBook:
    """A broker book over the model chain: every contract quoted ₹1 either side of the desk's own mark, times
    `away` (1 = fair; 1.5 = the book ran 50% away from the plan between the chain and the order)."""
    name = "kotak"
    refresh_min = None

    def __init__(self, away=1.0):
        self.eng, self.away, self.quoted = None, away, {}

    def expiries(self, u):
        return self.eng.model_chain.expiries(u, self.eng.day)

    def chain(self, u, expiry, spot=None, ts=None):
        ch = self.eng.model_chain.chain(u, expiry, spot=spot, ts=ts)
        ch.attrs["source"] = "kotak"
        return ch

    def live_quotes(self, insts):
        now, out = self.eng.feed.now(), {}
        for i in insts:
            m = self.eng.marker.mid(i, self.eng.spot(i.underlying), now) * self.away
            out[i.symbol] = (round(max(0.05, m - 1.0), 2), round(m + 1.0, 2))
            self.quoted[(now, i.symbol)] = out[i.symbol]
        return out


def _replay(cfg, bars, day, book):
    eng = IntradayEngine(cfg, ReplayFeed(bars, day), book, Journal(), IntradayBroker(cfg, starting_cash=500000), say=None)
    book.eng = eng
    run_replay(eng)
    return eng


@pytest.fixture(scope="module")
def traded(sessions):
    """The first recent day the desk trades on a fair live book: (cfg, day, engine, book)."""
    from quantdesk.config import DEFAULT_CONFIG, Config
    cfg = Config.load(DEFAULT_CONFIG)
    bars, days = sessions
    for d in days[-8:]:
        book = LiveBook()
        eng = _replay(cfg, bars, d, book)
        if eng.closed:
            return cfg, d, eng, book
    pytest.skip("no trade in the sample")


def test_fills_come_from_the_live_book(traded):
    cfg, day, eng, book = traded
    fills = eng.journal.df("SELECT * FROM fills")
    assert len(fills) >= 2
    for f in fills.itertuples():
        b, a = book.quoted[(pd.Timestamp(f.ts).tz_convert(IST), f.symbol)]
        assert f.price == pytest.approx(a + 0.05 if f.qty > 0 else b - 0.05)    # the book, plus one adverse tick
    assert all(t.meta["fill_quotes"] == "live kotak book" for t in eng.closed)
    assert all(t.meta["exit_quotes"] == "live kotak book" for t in eng.closed)
    assert eng.chain_name() == "kotak"


def test_entries_dont_chase_a_book_that_ran_away(traded, sessions):
    cfg, day, _, _ = traded
    eng = _replay(cfg, sessions[0], day, LiveBook(away=1.5))
    dec = eng.journal.df("SELECT * FROM decisions")
    moved = dec[dec.astype(str).apply(lambda r: r.str.contains("moved away")).any(axis=1)]
    assert len(moved) >= 1 and not eng.closed and not eng.open_trades


def test_chain_name_says_what_stands_in(cfg, sessions):
    bars, days = sessions
    eng = IntradayEngine(cfg, ReplayFeed(bars, days[-1]), Fixed(), Journal(), IntradayBroker(cfg, starting_cash=500000),
                         say=None)
    eng.chains.name = "kotak"
    for src, want in (({"kotak"}, "kotak"), ({"nse"}, "nse (no kotak)"), ({"model"}, "model (no kotak)"),
                      ({"kotak", "model"}, "kotak+model")):
        eng.chain_df = {f"U{i}": pd.DataFrame().pipe(lambda d, s=s: (d.attrs.update(source=s), d)[1])
                        for i, s in enumerate(sorted(src))}
        assert eng.chain_name() == want
