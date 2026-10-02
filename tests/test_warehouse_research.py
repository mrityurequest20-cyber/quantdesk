"""Warehouse research: the option-trade P&L arithmetic, that a rich-IV world shows a harvestable premium and a fair one
doesn't, and that a planted positioning signal is found while noise isn't."""
import datetime as dt
import math

import numpy as np
import pandas as pd

from quantdesk.options.pricing import bs_price
from quantdesk.research import warehouse_research as W


def bdays(n, start="2024-01-01"):
    return [d.date() for d in pd.bdate_range(start, periods=n)]


def chain_rows(day, expiry, S, sigma, strikes, T=None):
    T = (max((expiry - day).days, 0) / 365 + 1e-9) if T is None else T
    rows = []
    for K in strikes:
        for right in ("CE", "PE"):
            rows.append({"date": day, "kind": right, "expiry": expiry, "strike": float(K),
                         "close": round(max(bs_price(S, K, T, W.R, W.Q, sigma, right), 0.05), 2), "contracts": 100.0})
    return rows


def test_short_straddle_pnl_by_hand():
    days = bdays(15)
    spot = pd.Series(20000.0, index=days)
    spot[days[-1]] = 20100.0                                                    # settles 100 points above the strike
    e = days[-1]
    opts = pd.DataFrame(chain_rows(days[-2], e, 20000.0, 0.15, range(19500, 20550, 50)))
    tr = W.build_trades(opts, spot, "NIFTY")
    t = tr[(tr.strategy == "short_straddle") & (tr.k == 1)].iloc[0]
    c = opts[(opts.kind == "CE") & (opts.strike == 20000)].close.iloc[0]
    p = opts[(opts.kind == "PE") & (opts.strike == 20000)].close.iloc[0]
    want = (c - W.leg_cost(c)) + (p - W.leg_cost(p)) - 100.0
    assert math.isclose(t.pnl_pts, want, abs_tol=1e-9) and t.strikes == [20000.0, 20000.0]
    assert math.isclose(t.credit_pts, c + p)
    fly = tr[tr.strategy == "iron_fly"].iloc[0]
    assert fly.max_loss_pts > 0 and len(fly.strikes) == 4                      # defined risk, four legs


def _world(iv, rv, n_exp=200, seed=1):
    """Weekly expiries; the index follows risk-neutral GBM at `rv` on trading days, and options are priced at `iv` over
    the trading days left (so iv == rv is a genuinely fair market: no weekend or drift gift to either side)."""
    rng = np.random.default_rng(seed)
    days = bdays(n_exp * 5 + 20)
    mu = (W.R - W.Q - rv * rv / 2) / 252
    S = 20000 * np.exp(np.cumsum(rng.normal(mu, rv / math.sqrt(252), len(days))))
    spot = pd.Series(S, index=days)
    rows = []
    for j in range(4, n_exp * 5 + 15, 5):
        if j + 1 >= len(days):
            break
        e = days[j + 1]
        for k in (1, 3, 5):
            d = days[j + 1 - k]
            s0 = float(spot[d])
            atm = round(s0 / 50) * 50
            rows += chain_rows(d, e, s0, iv, range(int(atm - 1000), int(atm + 1050), 50), T=k / 252)
    return pd.DataFrame(rows), spot


def test_rich_implied_vol_shows_a_premium_and_fair_pricing_does_not():
    opts, spot = _world(iv=0.20, rv=0.11)
    res = {(r.strategy, r.k): r for r in W.evaluate_vrp(W.build_trades(opts, spot, "NIFTY"))}
    ss = res[("short_straddle", 1)]
    assert ss.mean_pts > 0 and ss.t > 3 and ss.verdict.startswith("EDGE")
    assert "needs" in ss.verdict                                                # a naked straddle needs ~₹1.5L+, not ₹20k
    opts, spot = _world(iv=0.12, rv=0.12, seed=2)
    fair = {(r.strategy, r.k): r for r in W.evaluate_vrp(W.build_trades(opts, spot, "NIFTY"))}
    assert not any(r.verdict.startswith("EDGE") for r in fair.values())        # costs eat a fairly priced premium


def _participants(n=900, planted=True, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n)
    net = np.cumsum(rng.normal(0, 5000, n))
    rows = []
    for d, x in zip(dates, net):
        long_ = 200000 + max(x, -150000)
        rows.append({"date": d, "participant": "FII", "fut_idx_long": long_, "fut_idx_short": 200000.0,
                     "opt_idx_call_long": 1e5 + rng.normal(0, 1e3), "opt_idx_call_short": 1e5, "opt_idx_put_long": 1e5,
                     "opt_idx_put_short": 1e5 + rng.normal(0, 1e3)})
        rows.append({"date": d, "participant": "Client", "fut_idx_long": 3e5 + rng.normal(0, 1e4), "fut_idx_short": 3e5,
                     "opt_idx_call_long": 1, "opt_idx_call_short": 1, "opt_idx_put_long": 1, "opt_idx_put_short": 1})
    part = pd.DataFrame(rows)
    d1 = np.sign(np.diff(net, prepend=net[0]))
    oc = rng.normal(0, 0.008, n)
    if planted:                                                                  # the next session follows FII's change
        oc[1:] += 0.004 * d1[:-1]
    o = 20000 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    daily = pd.DataFrame({"open": o, "high": o, "low": o, "close": o * np.exp(oc)}, index=dates)
    return part, {"NIFTY": daily}


def test_positioning_finds_a_planted_signal_and_not_noise():
    part, daily = _participants(planted=True)
    res = {r.id: r for r in W.run_positioning(part, daily)}
    assert res["P1"].verdict == "EDGE" and res["P1"].effect_bps > 20
    part, daily = _participants(planted=False, seed=4)
    assert not any(r.verdict == "EDGE" for r in W.run_positioning(part, daily))
