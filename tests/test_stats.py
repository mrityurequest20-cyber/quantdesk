import numpy as np
import pandas as pd
import pytest

from quantdesk.analytics import stats as S
from quantdesk.analytics import volatility as V
from quantdesk.analytics.regime import GaussianHMM


def test_mackinnon_pvalues_at_critical_values():
    assert S.mackinnon_p(-2.86, 1) == pytest.approx(0.05, abs=0.006)
    assert S.mackinnon_p(-3.43, 1) == pytest.approx(0.01, abs=0.003)
    assert S.mackinnon_p(-3.34, 2) == pytest.approx(0.05, abs=0.006)


def test_adf_separates_random_walk_from_ar1():
    rng = np.random.default_rng(3)
    rw = np.cumsum(rng.normal(size=1500))
    ar = np.zeros(1500)
    for i in range(1, 1500):
        ar[i] = 0.8 * ar[i - 1] + rng.normal()
    assert S.adf(rw).pvalue > 0.1
    assert S.adf(ar).pvalue < 0.01


def test_engle_granger_and_half_life():
    rng = np.random.default_rng(5)
    n = 1200
    x = np.cumsum(rng.normal(0, 0.01, n)) + 5
    kappa = np.log(2) / 10
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = e[i - 1] * (1 - kappa) + rng.normal(0, 0.01)
    y = 0.3 + 1.5 * x + e
    res = S.engle_granger(pd.Series(y), pd.Series(x))
    assert res.pvalue < 0.01
    assert res.hedge_ratio == pytest.approx(1.5, abs=0.05)
    assert res.half_life == pytest.approx(10, rel=0.35)
    unrelated = S.engle_granger(pd.Series(np.cumsum(rng.normal(size=n))), pd.Series(x))
    assert unrelated.pvalue > 0.05


def test_hurst_and_variance_ratio():
    rng = np.random.default_rng(1)
    rw = np.cumsum(rng.normal(size=3000))
    mr = np.zeros(3000)
    for i in range(1, 3000):
        mr[i] = 0.3 * mr[i - 1] + rng.normal()
    assert S.hurst(rw) == pytest.approx(0.5, abs=0.08)
    assert S.hurst(mr) < 0.3
    vr_rw, z_rw = S.variance_ratio(rw, 5)
    vr_mr, z_mr = S.variance_ratio(mr, 5)
    assert abs(z_rw) < 2.5 and vr_mr < 0.6 and z_mr < -5
    roll = S.rolling_hurst(pd.Series(rw), 200)
    assert roll.dropna().between(0.2, 0.8).mean() > 0.95


def test_kalman_hedge_tracks_beta():
    rng = np.random.default_rng(2)
    x = pd.Series(np.cumsum(rng.normal(0, 0.01, 1500)) + 5)
    y = 1.2 * x + 0.5 + rng.normal(0, 0.005, 1500)
    k = S.kalman_hedge(y, x)
    assert k["beta"].iloc[-200:].mean() == pytest.approx(1.2, abs=0.1)
    assert abs(k["z"].iloc[-500:].mean()) < 0.3


def test_garch_recovers_parameters():
    rng = np.random.default_rng(11)
    n, w, a, b = 6000, 0.02, 0.08, 0.90
    r = np.zeros(n)
    s2 = w / (1 - a - b)
    for t in range(n):
        r[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = w + a * r[t] ** 2 + b * s2
    p = V.Garch11().fit(r)
    assert p.alpha == pytest.approx(a, abs=0.03)
    assert p.beta == pytest.approx(b, abs=0.04)
    assert p.persistence == pytest.approx(a + b, abs=0.02)


def test_range_estimators_agree_on_gbm():
    rng = np.random.default_rng(4)
    n, steps, sig = 400, 78, 0.20
    dt = 1 / 252 / steps
    path = np.exp(np.cumsum(rng.normal(0, sig * np.sqrt(dt), n * steps))) * 100
    bars = path.reshape(n, steps)
    df = pd.DataFrame({"open": bars[:, 0], "high": bars.max(1), "low": bars.min(1), "close": bars[:, -1]},
                      index=pd.bdate_range("2020-01-01", periods=n))
    for est in (V.parkinson, V.garman_klass, V.rogers_satchell):
        assert est(df, 252).iloc[-1] == pytest.approx(20, rel=0.15)
    assert V.close_to_close(df["close"], 252).iloc[-1] == pytest.approx(20, rel=0.2)


def test_hmm_finds_two_volatility_states():
    rng = np.random.default_rng(9)
    states = np.repeat([0, 1, 0, 1, 0], 300)
    x = rng.normal(0, np.where(states == 0, 0.6, 2.0))
    m = GaussianHMM(2).fit(x)
    assert m.sd[0] == pytest.approx(0.6, rel=0.2) and m.sd[1] == pytest.approx(2.0, rel=0.2)
    p = m.filter(x)[:, 1]
    assert (p[states == 1] > 0.5).mean() > 0.85 and (p[states == 0] < 0.5).mean() > 0.85
