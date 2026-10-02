"""Market analysis: the desk's morning read of an index or stock, from chart structure to
volatility and the options surface, ending in concrete, costed structure ideas."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ..analytics import indicators as ind
from ..analytics import volatility as vol
from ..analytics.chart import analyze_chart
from ..analytics.regime import regime_frame
from ..analytics.stats import adf, hurst, variance_ratio
from ..core.calendar import TradingCalendar
from ..options.chain import OptionPricer, build_chain, expected_move, straddle_move
from ..options.structures import StructureBuilder


def analyze_symbol(cfg, symbol: str, data: dict[str, pd.DataFrame], aux: dict | None = None) -> str:
    df = data[symbol]
    rep = analyze_chart(symbol, df)
    L = [f"== {symbol} · {rep.date.date()} =="]
    L += rep.narrative
    close = df["close"]
    lp = np.log(close.tail(250))
    H = hurst(lp.to_numpy())
    vr, z = variance_ratio(lp.to_numpy(), 5)
    a = adf(lp.to_numpy())
    L.append(f"Statistical character (last 250 bars): Hurst {H:.2f}, variance ratio VR(5) {vr:.2f} (z {z:+.1f}), "
             f"ADF p {a.pvalue:.2f} → " + ("trending/persistent" if H > 0.55 and vr > 1 else
                                            "mean-reverting" if H < 0.45 or (vr < 1 and z < -2) else "close to a random walk") + ".")
    L.append("Realised vol (21d): close-to-close {:.1f}, Parkinson {:.1f}, Garman-Klass {:.1f}, Yang-Zhang {:.1f}, EWMA {:.1f}.".format(
        vol.close_to_close(close).iloc[-1], vol.parkinson(df).iloc[-1], vol.garman_klass(df).iloc[-1],
        vol.yang_zhang(df).iloc[-1], vol.ewma_vol(close).iloc[-1]))
    g = vol.Garch11()
    r = (np.log(close).diff().dropna() * 100).to_numpy()[-1500:]
    p = g.fit(r)
    L.append(f"GARCH(1,1): α {p.alpha:.3f}, β {p.beta:.3f}, persistence {p.persistence:.3f} (shock half-life "
             f"{p.half_life:.0f} d), long-run vol {np.sqrt(p.long_run_var * 252):.1f}; forecast next 5d "
             f"{g.forecast_vol(r, 5):.1f}, next 21d {g.forecast_vol(r, 21):.1f}.")
    cone = vol.vol_cone(close.tail(252 * 5))
    if not cone.empty and 21 in cone.index:
        c = cone.loc[21]
        L.append(f"Vol cone (21d, 5y): current {c['current']:.1f} vs median {c['median']:.1f} "
                 f"[p10 {c['p10']:.1f} – p90 {c['p90']:.1f}].")
    spec = cfg.instrument_spec(symbol)
    if spec.get("kind") == "index":
        L += _index_section(cfg, symbol, df, data, aux, g, r)
    return "\n".join(L)


def _index_section(cfg, symbol, df, data, aux, g, r) -> list[str]:
    L = []
    vix_sym = cfg.get("universe.volatility_index")
    vix = data[vix_sym]["close"] if vix_sym in data else None
    reg = (aux or {}).get("regime", {}).get(symbol)
    if reg is None:
        reg = regime_frame(df, vix, cfg)
    rr = reg.iloc[-1]
    L.append(f"Regime: {rr['label']} (HMM P(turbulent) {rr['p_stress']:.0%}, vol percentile {rr['vol_pct']:.0%}, "
             f"ADX {rr['adx']:.0f}, efficiency {rr['er']:.2f}).")
    if vix is None:
        return L
    spec = cfg.instrument_spec(symbol)
    beta = float(spec.get("iv_beta", 1.0))
    iv = float(vix.iloc[-1]) * beta / 100
    ivr = float(vol.iv_rank(vix).iloc[-1])
    S = float(df["close"].iloc[-1])
    asof = df.index[-1].date()
    cal = TradingCalendar(cfg.holidays())
    pricer = OptionPricer.from_config(cfg)
    exps = cal.expiries(asof, 60, int(spec.get("expiry_weekday", 1)), bool(spec.get("weekly_expiry", True)))[:4]
    g5 = g.forecast_vol(r, 5)
    L.append(f"Implied vol: India VIX {vix.iloc[-1]:.1f} → {symbol} ATM IV ≈ {iv * 100:.1f} (β {beta}); IV rank (1y) "
             f"{ivr:.0%}; variance risk premium vs GARCH {iv * 100 - g.forecast_vol(r, 10):+.1f} pts.")
    L.append("Expected moves (1σ): " + "; ".join(f"{e} ({(e - asof).days}d) ±{expected_move(S, iv, (e - asof).days):,.0f}"
                                                  for e in exps if (e - asof).days > 0))
    b = StructureBuilder(pricer, symbol, int(spec["lot_size"]), float(spec["strike_step"]))
    e14 = cal.pick_expiry(asof, 14, 6, int(spec.get("expiry_weekday", 1)), bool(spec.get("weekly_expiry", True)))
    e28 = cal.pick_expiry(asof, 28, 10, int(spec.get("expiry_weekday", 1)), bool(spec.get("weekly_expiry", True)))
    ch = build_chain(pricer, symbol, S, asof, e14, iv, float(spec["strike_step"]), int(spec["lot_size"]), 5)
    L.append(f"Model chain ({e14}, model prices from VIX + skew; use live quotes before trading):")
    L.append(ch[["CE_ltp", "CE_delta", "CE_iv", "PE_ltp", "PE_delta", "PE_iv"]].round(2).to_string())
    L.append(f"ATM straddle ₹{straddle_move(ch):,.1f} ≈ {straddle_move(ch) / S:.2%} of spot.")
    fv = g.forecast_vol(r, max(3, (e14 - asof).days * 5 // 7)) / 100
    ideas = [b.iron_condor(S, asof, e14, iv, 0.16, 0.06), b.long_straddle(S, asof, e28, iv),
             b.bull_call_spread(S, asof, e28, iv), b.bear_put_spread(S, asof, e28, iv)]
    L.append(f"Structure ideas (EV and POP at the GARCH forecast vol {fv * 100:.1f}, zero drift; per 1 lot):")
    for st in ideas:
        L.append(f"  - {st.describe()} | POP {st.prob_profit(fv):.0%}, EV ₹{st.expected_pnl(fv):,.0f}")
    return L


def scan(cfg, data: dict[str, pd.DataFrame], aux: dict | None = None) -> pd.DataFrame:
    rows = []
    for s in cfg.symbols("all"):
        if s not in data or len(data[s]) < 260:
            continue
        df = data[s]
        c = df["close"]
        a = ind.adx(df).iloc[-1]
        rows.append({
            "symbol": s, "close": c.iloc[-1], "1d": c.pct_change().iloc[-1], "1m": c.pct_change(21).iloc[-1],
            "12-1m": c.shift(21).iloc[-1] / c.shift(252).iloc[-1] - 1,
            "vs_sma200": c.iloc[-1] / ind.sma(c, 200).iloc[-1] - 1, "adx": a["adx"],
            "rsi14": ind.rsi(c, 14).iloc[-1], "z20": ind.zscore(c, 20).iloc[-1],
            "rv21": vol.close_to_close(c).iloc[-1], "yz21": vol.yang_zhang(df).iloc[-1],
            "trend": "up" if c.iloc[-1] > ind.sma(c, 200).iloc[-1] and a["plus_di"] > a["minus_di"] else
                     "down" if c.iloc[-1] < ind.sma(c, 200).iloc[-1] and a["minus_di"] > a["plus_di"] else "mixed",
        })
    out = pd.DataFrame(rows).set_index("symbol")
    if not out.empty:
        out["mom_rank"] = out["12-1m"].rank(ascending=False).astype(int)
    return out
