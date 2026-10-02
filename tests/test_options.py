import datetime as dt

import numpy as np
import pytest

from quantdesk.core.types import Instrument
from quantdesk.options import SVI, OptionPricer, StructureBuilder, bs_price, greeks, implied_vol, strike_for_delta

S, r, q = 25000.0, 0.065, 0.012


@pytest.mark.parametrize("K", [22000, 24500, 25000, 25600, 28000])
@pytest.mark.parametrize("T", [2 / 365, 14 / 365, 0.5])
def test_put_call_parity(K, T):
    c, p = bs_price(S, K, T, r, q, 0.16, "CE"), bs_price(S, K, T, r, q, 0.16, "PE")
    assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-6)


@pytest.mark.parametrize("sigma", [0.08, 0.14, 0.35, 0.8])
@pytest.mark.parametrize("K,right", [(23500, "PE"), (25000, "CE"), (26500, "CE"), (24000, "CE")])
def test_implied_vol_roundtrip(sigma, K, right):
    T = 21 / 365
    px = bs_price(S, K, T, r, q, sigma, right)
    if px < 0.01:
        pytest.skip("price too small to invert meaningfully")
    assert implied_vol(px, S, K, T, r, q, right) == pytest.approx(sigma, abs=1e-5)


def test_implied_vol_rejects_arbitrage():
    assert np.isnan(implied_vol(0.0001, S, 20000, 0.1, r, q, "CE"))   # below intrinsic
    assert np.isnan(implied_vol(S * 2, S, 25000, 0.1, r, q, "CE"))     # above the upper bound


@pytest.mark.parametrize("right", ["CE", "PE"])
def test_greeks_match_finite_differences(right):
    K, T, s = 25300, 30 / 365, 0.15
    g = greeks(S, K, T, r, q, s, right)
    h = 0.5
    up, dn, mid = (bs_price(x, K, T, r, q, s, right) for x in (S + h, S - h, S))
    assert g["delta"] == pytest.approx((up - dn) / (2 * h), rel=1e-4)
    assert g["gamma"] == pytest.approx((up - 2 * mid + dn) / h ** 2, rel=1e-3)
    dv = (bs_price(S, K, T, r, q, s + 1e-4, right) - bs_price(S, K, T, r, q, s - 1e-4, right)) / 2e-4
    assert g["vega"] == pytest.approx(dv / 100, rel=1e-4)
    dt_ = 1e-4
    theta_fd = (bs_price(S, K, T - dt_, r, q, s, right) - bs_price(S, K, T + dt_, r, q, s, right)) / (2 * dt_) / 365
    assert g["theta"] == pytest.approx(theta_fd, rel=1e-3)


def test_expired_option_is_intrinsic():
    assert bs_price(25100, 25000, 0.0, r, q, 0.2, "CE") == pytest.approx(100)
    assert bs_price(25100, 25000, 0.0, r, q, 0.2, "PE") == 0.0


def test_strike_for_delta():
    T, s = 14 / 365, 0.14
    for d in (0.1, 0.25, 0.5):
        K = strike_for_delta(S, T, r, q, d, "PE", sigma=s)
        assert abs(greeks(S, K, T, r, q, s, "PE")["delta"]) == pytest.approx(d, abs=1e-4)


def test_iron_condor_payoff_and_risk():
    b = StructureBuilder(OptionPricer(), "NIFTY", 65, 50)
    asof, exp = dt.date(2026, 9, 28), dt.date(2026, 10, 13)
    ic = b.iron_condor(S, asof, exp, 0.14, 0.16, 0.05)
    ks = sorted(l.instrument.strike for l in ic.legs)
    credit = -ic.net_premium()
    assert credit > 0
    width = max(ks[1] - ks[0], ks[3] - ks[2])
    assert ic.max_loss() == pytest.approx(width * 65 - credit, rel=1e-6)
    assert ic.max_profit() == pytest.approx(credit, rel=1e-6)
    be = ic.breakevens()
    assert len(be) == 2 and ks[1] > be[0] > ks[0] and ks[2] < be[1] < ks[3]
    g = ic.greeks(OptionPricer(), S, asof, 0.14)
    assert g["vega"] < 0 and g["theta"] > 0          # short volatility, collects decay
    assert 0.5 < ic.prob_profit(0.14) < 0.95
    # the variance risk premium: EV is higher if realised vol comes in below implied
    assert ic.expected_pnl(0.10) > ic.expected_pnl(0.14) > ic.expected_pnl(0.20)


def test_debit_spread_and_straddle_bounds():
    b = StructureBuilder(OptionPricer(), "NIFTY", 65, 50)
    asof, exp = dt.date(2026, 9, 28), dt.date(2026, 10, 27)
    bc = b.bull_call_spread(S, asof, exp, 0.14)
    assert bc.net_premium() > 0 and bc.max_loss() == pytest.approx(bc.net_premium())
    st = b.long_straddle(S, asof, exp, 0.14)
    assert st.max_profit() == float("inf")
    assert st.max_loss() == pytest.approx(st.net_premium())


def test_option_symbol_and_pricer():
    inst = Instrument.option("NIFTY", dt.date(2026, 10, 27), 25000, "ce", 65)
    assert inst.symbol == "NIFTY27OCT2625000CE" and inst.right == "CE"
    p = OptionPricer()
    assert p.price(inst, 25200, dt.date(2026, 10, 27), 0.14) == pytest.approx(200)
    with pytest.raises(ValueError):
        Instrument.option("NIFTY", dt.date(2026, 10, 27), 25000, "XX", 65)


def test_svi_fit_recovers_smile():
    truth = SVI(a=0.0004, b=0.02, rho=-0.5, m=0.01, sigma=0.08, T=30 / 365)
    F = 25000
    K = np.linspace(21000, 29000, 41)
    fit = SVI.fit(K, truth.iv(K, F), F, truth.T)
    assert np.max(np.abs(fit.iv(K, F) - truth.iv(K, F))) < 5e-4
    assert fit.sane()
