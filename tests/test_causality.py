"""No look-ahead: every feature computed on data up to bar k must equal the same feature
computed on the full history, at bar k. If a function peeks at the future (centred
windows, backfill, full-sample normalisation, a model fitted on all data), this fails."""
import numpy as np
import pandas as pd
import pytest

from quantdesk.analytics import indicators as ind
from quantdesk.analytics import volatility as V
from quantdesk.analytics.regime import regime_frame, rolling_hmm_stress
from quantdesk.analytics.stats import rolling_hurst
from quantdesk.strategies import build_strategies

CUTS = (700, 1100, 1400)


def _same(a, b, label):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    both_nan = np.isnan(a) & np.isnan(b)
    ok = both_nan | np.isclose(a, b, rtol=1e-7, atol=1e-9)
    assert ok.all(), f"{label}: look-ahead detected ({a[~ok][:3]} vs {b[~ok][:3]})"


def _check_frame(fn, df, label):
    full = fn(df)
    for k in CUTS:
        part = fn(df.iloc[:k])
        a = part.iloc[-1] if isinstance(part, (pd.Series, pd.DataFrame)) else part
        b = full.iloc[k - 1]
        if isinstance(a, pd.Series):
            num = [c for c in a.index if pd.api.types.is_number(a[c]) and not isinstance(a[c], bool)]
            _same(a[num].to_numpy(dtype=float), b[num].to_numpy(dtype=float), f"{label}@{k}")
            obj = [c for c in a.index if c not in num]
            for c in obj:
                assert a[c] == b[c], f"{label}@{k}:{c}"
        else:
            _same([a], [b], f"{label}@{k}")


INDICATORS = {
    "sma": lambda d: ind.sma(d["close"], 20), "ema": lambda d: ind.ema(d["close"], 50),
    "rsi": lambda d: ind.rsi(d["close"], 14), "macd": lambda d: ind.macd(d["close"]),
    "bollinger": lambda d: ind.bollinger(d["close"]), "atr": lambda d: ind.atr(d),
    "adx": lambda d: ind.adx(d), "supertrend": lambda d: ind.supertrend(d), "donchian": lambda d: ind.donchian(d, 55),
    "keltner": lambda d: ind.keltner(d), "stoch": lambda d: ind.stochastic(d), "obv": lambda d: ind.obv(d),
    "vwap": lambda d: ind.rolling_vwap(d), "z": lambda d: ind.zscore(d["close"], 20),
    "pct_rank": lambda d: ind.pct_rank(d["close"], 250), "er": lambda d: ind.efficiency_ratio(d["close"]),
    "chandelier": lambda d: ind.chandelier(d), "yang_zhang": lambda d: V.yang_zhang(d),
    "garman_klass": lambda d: V.garman_klass(d), "ewma": lambda d: V.ewma_vol(d["close"]),
    "iv_rank": lambda d: V.iv_rank(d["close"]), "hurst": lambda d: rolling_hurst(np.log(d["close"]), 100),
}


@pytest.mark.parametrize("name", sorted(INDICATORS))
def test_indicator_is_causal(market, name):
    _check_frame(INDICATORS[name], market[1]["RELIANCE"], name)


def test_garch_forecast_is_causal(market):
    _check_frame(lambda d: V.rolling_garch_forecast(d["close"], 10), market[1]["NIFTY"], "garch")


def test_hmm_stress_is_causal(market):
    _check_frame(lambda d: rolling_hmm_stress(d["close"]), market[1]["NIFTY"], "hmm")


def test_regime_frame_is_causal(market, cfg):
    vix = market[1]["INDIAVIX"]["close"]
    _check_frame(lambda d: regime_frame(d, vix.reindex(d.index), cfg).drop(columns=["close"]), market[1]["NIFTY"], "regime")


def test_every_strategy_feature_is_causal(market, cfg):
    data = market[1]
    bench = data["NIFTY"].index
    for k in CUTS:
        cut = bench[k - 1]
        part = {s: df.loc[:cut] for s, df in data.items()}
        for full_s, part_s in zip(build_strategies(cfg), build_strategies(cfg)):
            full_s.prepare(data, _aux(cfg, data))
            part_s.prepare(part, _aux(cfg, part))
            for sym, t in part_s.tables.items():
                a, b = t.row(cut), full_s.tables[sym].row(cut)
                if a is None:
                    continue
                num = [c for c, v in a.items() if isinstance(v, (float, int, np.floating, np.integer)) and not isinstance(v, bool)]
                _same([a[c] for c in num], [b[c] for c in num], f"{full_s.name}:{sym}@{k}")


_AUX_CACHE = {}


def _aux(cfg, data):
    from quantdesk.engine.engine import Engine
    key = (len(data["NIFTY"]),)
    if key not in _AUX_CACHE:
        _AUX_CACHE[key] = Engine.build_aux(cfg, data)
    return _AUX_CACHE[key]
