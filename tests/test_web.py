"""Desk UI: the HTTP API (real server on an ephemeral port) and the GoCharting datafeed
adapter (executed with Node against that server, when Node is available)."""
import json
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import pytest

from quantdesk.engine.live import LiveRunner
from quantdesk.execution.broker import PaperBroker
from quantdesk.web.server import DeskAPI, make_handler

STATIC = Path(__file__).resolve().parent.parent / "quantdesk" / "web" / "static"


@pytest.fixture(scope="module")
def desk(market, tmp_path_factory):
    from quantdesk.config import DEFAULT_CONFIG, Config
    tmp = tmp_path_factory.mktemp("desk")
    cfg = Config.load(DEFAULT_CONFIG, overrides={"runtime": {"dir": str(tmp / "rt")}, "data": {"history_years": 3}})
    prov, data = market
    runner = LiveRunner(cfg, broker=PaperBroker(cfg, state_path=tmp / "b.json"), journal_path=tmp / "j.db", provider=prov)
    bench = data["NIFTY"].index
    for d in bench[-4:]:                       # a few real EOD cycles so there is state to show
        runner.engine = None
        runner.run_eod(d.date())
    api = DeskAPI(cfg, runner)
    httpd = HTTPServer(("127.0.0.1", 0), make_handler(api))
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{httpd.server_port}", cfg
    httpd.shutdown()


def get(base, path):
    with urllib.request.urlopen(base + path) as r:
        return json.loads(r.read())


def post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_config_and_symbols(desk):
    base, _ = desk
    c = get(base, "/api/config")
    assert c["gocharting"]["licenseKey"] and "{key}" not in c["gocharting"]["sdkUrl"]
    assert c["default"] == "NSE:INDEX:NIFTY" and c["paperOnly"] is True
    syms = get(base, "/api/symbols")
    nifty = next(s for s in syms if s["symbol"] == "NIFTY")
    assert nifty["segment"] == "INDEX" and nifty["lot_size"] == 65 and nifty["regime"]


def test_history_is_udf_in_unix_seconds(desk):
    base, _ = desk
    h = get(base, "/api/history?symbol=NSE:EQUITY:RELIANCE")
    assert h["s"] == "ok" and len(h["t"]) == len(h["c"]) > 500
    assert all(1.4e9 < t < 2.2e9 for t in h["t"][:5])                   # seconds, not ms/us
    assert h["t"] == sorted(h["t"])
    to = h["t"][-10]
    cb = get(base, f"/api/history?symbol=RELIANCE&to={to}&countback=5")
    assert cb["t"] == h["t"][-14:-9]
    rng = get(base, f"/api/history?symbol=RELIANCE&from={h['t'][-3]}&to={h['t'][-1]}")
    assert rng["t"] == h["t"][-3:]
    assert get(base, "/api/history?symbol=RELIANCE&from=1&to=2")["s"] == "no_data"


def test_broker_payload_matches_gocharting_shape(desk):
    base, _ = desk
    b = get(base, "/api/broker")
    assert set(b) == {"accountList", "orderBook", "tradeBook", "positions"}
    assert b["accountList"][0]["currency"] == "INR"
    for p in b["positions"]:
        assert {"id", "productId", "symbol", "size", "price", "side", "security", "stopLoss", "takeProfit"} <= set(p)


def test_trade_from_chart_roundtrip(desk):
    base, cfg = desk
    code, o = post(base, "/api/order", {"symbol": "NSE:EQUITY:INFY", "side": "buy", "quantity": 10, "stopLoss": 1.0})
    assert code == 200 and o["qty"] == 10
    pos = [p for p in get(base, "/api/broker?symbol=INFY")["positions"] if p["id"] == o["trade_id"]]
    assert pos and pos[0]["size"] == 10 and pos[0]["stopLoss"] == 1.0
    code, m = post(base, "/api/modify", {"id": o["trade_id"], "stopLoss": 2.0, "takeProfit": 99999})
    assert code == 200 and m["stop"] == 2.0
    assert any(l["kind"] == "target" for l in get(base, "/api/overlays?symbol=INFY")["levels"])
    code, c = post(base, "/api/close", {"id": o["trade_id"]})
    assert code == 200 and c["pnl"] < 0                                   # round-trip costs
    st = get(base, "/api/status")
    assert st["recent"][0]["id"] == o["trade_id"] and st["recent"][0]["grade"]


def test_kill_switch_blocks_new_orders_but_not_closes(desk):
    base, cfg = desk
    code, o = post(base, "/api/order", {"symbol": "TCS", "side": "buy", "quantity": 1})
    assert code == 200
    kill = cfg.runtime_dir / "KILL"
    kill.write_text("stop")
    try:
        assert post(base, "/api/order", {"symbol": "TCS", "side": "buy", "quantity": 1})[0] == 403
        assert post(base, "/api/close", {"id": o["trade_id"]})[0] == 200
        assert get(base, "/api/status")["kill_switch"] is True
    finally:
        kill.unlink()


def test_errors_and_static(desk):
    base, _ = desk
    assert post(base, "/api/order", {"symbol": "NOPE", "side": "buy", "quantity": 1})[0] == 400
    assert post(base, "/api/order", {"symbol": "TCS", "side": "buy", "quantity": 0})[0] == 400
    with urllib.request.urlopen(base + "/") as r:
        assert b"QuantDesk" in r.read()
    for bad in ("/static/../server.py", "/static/%2e%2e/server.py", "/api/nothing"):
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + bad)
        assert e.value.code == 404


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_gocharting_datafeed_adapter_against_server(desk):
    base, _ = desk
    script = f"""
const {{ createQuantDeskDatafeed, toSec }} = require({json.dumps(str(STATIC / "datafeed.js"))});
const df = createQuantDeskDatafeed({json.dumps(base)}, {{ pollMs: 50 }});
(async () => {{
  const info = await new Promise((res, rej) => df.resolveSymbol("NSE:INDEX:NIFTY", res, rej));
  const bars = await df.getBars(info, "1D", {{ from: new Date(0), to: new Date(), countBack: 30 }});
  const search = await new Promise((res) => df.searchSymbols("BANK", "", "", res));
  const err = await new Promise((res) => df.resolveSymbol("NSE:EQUITY:NOPE", () => res("resolved?"), res));
  const tick = await new Promise((res) => df.subscribeTicks(info, "1D", res, "u1"));
  df.unsubscribeTicks("u1"); df.destroy();
  console.log(JSON.stringify({{ info, n: bars.t.length, s: bars.s, t0: bars.t[0], search: search.items.map(i => i.key),
                               err, tick, toSec: [toSec(new Date(1e12)), toSec(1e12), toSec(5)] }}));
}})().catch((e) => {{ console.error(e); process.exit(1); }});
"""
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    r = json.loads(out.stdout)
    assert r["info"]["full_name"] == "NSE:INDEX:NIFTY" and r["info"]["segment"] == "INDEX"
    assert r["info"]["exchange_info"]["zone"] == "Asia/Kolkata" and r["info"]["exchange_info"]["valid_intervals"] == ["1D"]
    assert r["s"] == "ok" and r["n"] > 30 and r["t0"] > 1.4e9
    assert set(r["search"]) >= {"NSE:INDEX:BANKNIFTY", "NSE:EQUITY:HDFCBANK"}
    assert "NOPE" in r["err"]
    assert r["tick"]["type"] == "trade" and r["tick"]["price"] > 0
    assert r["toSec"] == [1e9, 1e9, 5]
