"""Edge research: a fixed list of hypotheses about NIFTY and BANKNIFTY, tested on real data with the
statistics that stop a researcher fooling themselves.

Rules (fixed before looking at the data, so the list can't be tuned to what happened to work):
  * every hypothesis is stated as "trade in direction D over window W": the effect is the average
    signed index return per trade, in basis points and in index points;
  * t-statistics are Newey-West (HAC), so autocorrelated or overlapping observations don't inflate them;
  * discovery = the older 2/3 of the sample, holdout = the newest 1/3; an edge must keep its sign
    in the holdout (one-sided p < 0.10);
  * Benjamini-Hochberg false-discovery control (q = 0.10) across every test run, on the *discovery*
    p-values (with the full sample's p the holdout sits inside the evidence it is meant to confirm,
    and chance results confirm themselves: 3 false "edges" in 20 noise worlds, against 1 in 40 this way);
  * the cost hurdle: what one lot of a 0.35Δ index option costs to get in and out (brokerage, STT,
    exchange, GST, stamp, the bid/ask) expressed in index points. Under fair (business-time) option
    pricing theta is paid for by gamma, so the directional edge must beat the costs.

Verdicts: EDGE (survives all of it), REAL BUT BELOW COSTS, NEEDS MARGIN (real, but only a premium
seller can harvest it, which a ₹20k account can't), NO EDGE.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

IST = "Asia/Kolkata"
# round trip for one lot of a 0.35Δ weekly/monthly option, in index points (₹96 NIFTY / ₹100 BANKNIFTY, see EVEngine)
COST_POINTS = {"NIFTY": 96 / (65 * 0.35), "BANKNIFTY": 100 / (30 * 0.35)}
LOT = {"NIFTY": 65, "BANKNIFTY": 30}
DELTA = 0.35


@dataclass
class Result:
    id: str
    symbol: str
    hypothesis: str
    data: str
    n: int
    effect_bps: float
    effect_pts: float
    t: float
    p: float
    effect_holdout_bps: float = float("nan")
    p_holdout: float = float("nan")
    hurdle_pts: float = float("nan")
    kind: str = "directional"             # directional | premium (needs a seller)
    sd_pts: float = float("nan")          # per-trade σ of the index move, points
    capital_half_kelly: float = float("nan")   # account size at which ONE lot of a 0.35Δ option is a half-Kelly bet
    bh_pass: bool = False
    verdict: str = ""
    note: str = ""
    params: dict = field(default_factory=dict)


# ---- statistics ---------------------------------------------------------------------------------------------------
def hac_mean(x: np.ndarray, lags: int | None = None) -> tuple[float, float, float]:
    """Mean, Newey-West t and two-sided p for the mean of x."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        return float("nan"), float("nan"), float("nan")
    lags = int(lags if lags is not None else max(1, math.floor(4 * (n / 100) ** (2 / 9))))
    m = x.mean()
    e = x - m
    s = e @ e / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1 - k / (lags + 1)
        s += 2 * w * (e[k:] @ e[:-k]) / n
    se = math.sqrt(max(s, 1e-30) / n)
    t = m / se
    return float(m), float(t), float(2 * stats.t.sf(abs(t), n - 1))


def benjamini_hochberg(pvals: list[float], q: float = 0.10) -> list[bool]:
    p = np.array([x if x == x else 1.0 for x in pvals])
    order = np.argsort(p)
    m = len(p)
    thresh = q * (np.arange(1, m + 1)) / m
    passed = p[order] <= thresh
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    out = np.zeros(m, dtype=bool)
    out[order[:k]] = True
    return out.tolist()


def _split(x: pd.Series):
    x = x.dropna()
    cut = int(len(x) * 2 / 3)
    return x.iloc[:cut], x.iloc[cut:]


def evaluate(rid, symbol, hypothesis, data, signed_returns: pd.Series, price: float, lags=None, kind="directional",
             note="", params=None) -> Result:
    """`signed_returns`: per-trade log returns already signed by the trade direction (a series indexed by time)."""
    x = signed_returns.dropna()
    m, t, p = hac_mean(x.to_numpy(), lags)
    disc, hold = _split(x)
    md, _, pdisc = hac_mean(disc.to_numpy(), lags)
    mh, th, ph2 = hac_mean(hold.to_numpy(), lags)
    one_sided = ph2 / 2 if (mh == mh and md == md and np.sign(mh) == np.sign(md)) else 1 - (ph2 / 2 if ph2 == ph2 else 0)
    r = Result(rid, symbol, hypothesis, data, int(len(x)), m * 1e4, m * price, t, p, mh * 1e4, one_sided,
               COST_POINTS.get(symbol, float("nan")), kind, note=note, params={**(params or {}), "p_discovery": pdisc})
    r.sd_pts = float(x.std() * price)
    net = abs(r.effect_pts) - r.hurdle_pts
    if kind == "directional" and net > 0 and symbol in LOT:
        # one lot of a 0.35Δ option moves ≈ Δ × lot rupees per index point; Kelly fraction f* = μ/σ² of the account
        mu, sd = net * DELTA * LOT[symbol], r.sd_pts * DELTA * LOT[symbol]
        r.capital_half_kelly = float(2 * sd * sd / mu)
    return r


# ---- data ---------------------------------------------------------------------------------------------------------
def _ist(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: str(c).lower() for c in df.columns})
    idx = pd.DatetimeIndex(df.index)
    df.index = idx.tz_localize(IST) if idx.tz is None else idx.tz_convert(IST)
    return df[["open", "high", "low", "close"] + (["volume"] if "volume" in df else [])].astype(float)


def load_yahoo(symbols=("NIFTY", "BANKNIFTY")) -> dict:
    import yfinance as yf
    tick = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "INDIAVIX": "^INDIAVIX"}
    out = {"daily": {}, "hourly": {}, "m5": {}}
    for s in list(symbols) + ["INDIAVIX"]:
        t = yf.Ticker(tick[s])
        out["daily"][s] = _ist(t.history(period="max", interval="1d", auto_adjust=False)).dropna(subset=["close"])
        if s == "INDIAVIX":
            continue
        out["hourly"][s] = _ist(t.history(period="730d", interval="1h", auto_adjust=False)).dropna(subset=["close"])
        out["m5"][s] = _ist(t.history(period="60d", interval="5m", auto_adjust=False)).dropna(subset=["close"])
    return out


def _local_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """Daily bars keyed by the exchange's *own* calendar date. Converting an Asian or US daily bar to IST
    would shift its date and let a session that runs during India's day pose as a 'prior' session."""
    df = raw.rename(columns={c: str(c).lower() for c in raw.columns})
    idx = pd.DatetimeIndex(df.index)
    df.index = pd.DatetimeIndex(idx.date if idx.tz is not None else idx.normalize())
    return df[["open", "high", "low", "close"]].astype(float).dropna(subset=["close"])


def load_global(daily: bool = True, intraday: bool = True) -> dict:
    import yfinance as yf
    from ..data.global_universe import GLOBAL
    out = {"daily": {}, "m5": {}}
    for key, g in GLOBAL.items():
        t = yf.Ticker(g["yahoo"])
        try:
            if daily:
                out["daily"][key] = _local_daily(t.history(period="max", interval="1d", auto_adjust=False))
            if intraday and g.get("intraday"):
                m = t.history(period="60d", interval="5m", auto_adjust=False)
                if m is not None and len(m):
                    m = m.rename(columns={c: str(c).lower() for c in m.columns})
                    m.index = pd.DatetimeIndex(m.index).tz_convert("UTC")
                    out["m5"][key] = m[["open", "high", "low", "close"]].astype(float).dropna(subset=["close"])
        except Exception as exc:                                     # one missing market must not stop the study
            out.setdefault("errors", {})[key] = str(exc)[:120]
    return out


def _session(df: pd.DataFrame) -> pd.DataFrame:
    t = df.index.time
    return df[(t >= dt.time(9, 15)) & (t < dt.time(15, 30))]


# ---- the hypotheses -----------------------------------------------------------------------------------------------
def daily_tests(sym: str, d: pd.DataFrame, vix: pd.DataFrame | None) -> list[Result]:
    d = d[d["open"] > 0].copy()
    d["r_oc"] = np.log(d["close"] / d["open"])                       # the part an intraday trader can hold
    d["gap"] = np.log(d["open"] / d["close"].shift(1))
    d["r_cc"] = np.log(d["close"] / d["close"].shift(1))
    px = float(d["close"].iloc[-1])
    out = [evaluate("D1", sym, "Intraday drift: long from the open to the close, every day", "daily", d["r_oc"], px)]
    g = d[d["gap"].abs() > 0.003]
    out.append(evaluate("D2", sym, "Gap continuation: after a gap > 0.3%, trade in the gap's direction open→close",
                        "daily", np.sign(g["gap"]) * g["r_oc"], px, params={"min_gap": 0.003}))
    big = d["r_cc"].shift(1) < -0.015
    out.append(evaluate("D3", sym, "Rebound: the day after a close-to-close fall > 1.5%, long open→close", "daily",
                        d.loc[big, "r_oc"], px, params={"fall": -0.015}))
    tue = d.index.dayofweek == 1
    out.append(evaluate("D4", sym, "Tuesday (weekly expiry): long open→close on Tuesdays", "daily", d.loc[tue, "r_oc"], px))
    ym = d.index.tz_localize(None).to_period("M")
    pos = pd.Series(range(len(d)), index=d.index).groupby(ym).rank(method="first")
    size = pd.Series(1, index=d.index).groupby(ym).transform("size")
    tom = (pos <= 3) | (pos == size)
    out.append(evaluate("D5", sym, "Turn of the month (last day + first 3): long open→close", "daily", d.loc[tom, "r_oc"], px))
    if vix is not None and len(vix):
        v = vix["close"].reindex(d.index).ffill()
        spike = np.log(v / v.shift(1)) > 0.10
        out.append(evaluate("D6", sym, "After an India VIX jump > 10%: long the next day open→close", "daily",
                            d["r_oc"][spike.shift(1, fill_value=False).to_numpy()], px))
        if sym == "NIFTY":
            rv_fwd = d["r_cc"].rolling(21).std().shift(-21) * math.sqrt(252)
            vrp = (v / 100 - rv_fwd).dropna()
            r = evaluate("V1", sym, "Volatility risk premium: India VIX minus the next 21 days' realised vol", "daily",
                         vrp, px, lags=25, kind="premium",
                         note="in vol points ×100, not a return; harvested by selling options (margin)")
            r.effect_pts = float("nan")
            r.params = {"share_positive": round(float((vrp > 0).mean()), 3), "mean_vix": round(float(v.mean()), 2)}
            out.append(r)
    return out


def hourly_tests(sym: str, h: pd.DataFrame, daily: pd.DataFrame) -> list[Result]:
    h = _session(h)
    if h.empty:
        return []
    px = float(h["close"].iloc[-1])
    rows = []
    prev_close = daily["close"].copy()
    prev_close.index = prev_close.index.date
    pc = prev_close.shift(1)
    for day, g in h.groupby(h.index.date):
        if len(g) < 6:
            continue
        op = g["open"].iloc[0]
        first = g[g.index.time < dt.time(10, 15)]
        mid = g[(g.index.time >= dt.time(10, 15))]
        last = g[g.index.time >= dt.time(14, 15)]
        if first.empty or mid.empty or last.empty:
            continue
        c1015 = first["close"].iloc[-1]
        c1415 = g[g.index.time < dt.time(14, 15)]["close"].iloc[-1]
        close = g["close"].iloc[-1]
        prev = pc.get(day, np.nan)
        rows.append({"day": pd.Timestamp(day), "r_first": math.log(c1015 / op), "r_rest": math.log(close / c1015),
                     "r_on_first": math.log(c1015 / prev) if prev == prev else np.nan,
                     "r_day_to_1415": math.log(c1415 / op), "r_last": math.log(close / c1415)})
    x = pd.DataFrame(rows).set_index("day")
    return [
        evaluate("H1", sym, "First-hour momentum: trade the 09:15→10:15 direction from 10:15 to the close", "hourly",
                 np.sign(x["r_first"]) * x["r_rest"], px),
        evaluate("H2", sym, "Intraday momentum (Gao et al.): overnight + first hour predicts the last hour (14:15→close)",
                 "hourly", np.sign(x["r_on_first"]) * x["r_last"], px),
        evaluate("H3", sym, "Late-day trend: trade the open→14:15 direction into the close", "hourly",
                 np.sign(x["r_day_to_1415"]) * x["r_last"], px),
    ]


def m5_tests(sym: str, m: pd.DataFrame) -> list[Result]:
    m = _session(m)
    if m.empty:
        return []
    px = float(m["close"].iloc[-1])
    orb, vwap_rev, mom = [], [], []
    for day, g in m.groupby(m.index.date):
        if len(g) < 60:
            continue
        c, hi, lo = g["close"].to_numpy(), g["high"].to_numpy(), g["low"].to_numpy()
        orh, orl = hi[:6].max(), lo[:6].min()                              # 30-minute opening range
        exit_i = min(len(c) - 1, 71)                                        # ~15:15
        for i in range(6, exit_i):
            if c[i] > orh or c[i] < orl:
                d_ = 1 if c[i] > orh else -1
                orb.append((pd.Timestamp(g.index[i]), d_ * math.log(c[exit_i] / c[i])))
                break
        tp = (hi + lo + c) / 3
        vw = np.cumsum(tp) / np.arange(1, len(c) + 1)
        lr = np.r_[0, np.diff(np.log(c))]
        sd = pd.Series(lr).rolling(12, min_periods=6).std().to_numpy()
        for i in range(12, len(c) - 6, 6):                                 # non-overlapping 30-minute steps
            z = (c[i] / vw[i] - 1) / (sd[i] * math.sqrt(6)) if sd[i] and sd[i] == sd[i] else 0
            fwd = math.log(c[i + 6] / c[i])
            if abs(z) > 2:
                vwap_rev.append((pd.Timestamp(g.index[i]), -np.sign(z) * fwd))
            prev30 = math.log(c[i] / c[i - 6])
            mom.append((pd.Timestamp(g.index[i]), np.sign(prev30) * fwd))
    ser = lambda rows: pd.Series([r for _, r in rows], index=[t for t, _ in rows], dtype=float)
    return [
        evaluate("M1", sym, "Opening-range breakout: first 5m close outside the 30-min range, hold to 15:15", "5m", ser(orb), px),
        evaluate("M2", sym, "VWAP reversion: > 2σ from VWAP, fade it for 30 minutes", "5m", ser(vwap_rev), px),
        evaluate("M3", sym, "30-minute momentum: trade the last 30 minutes' direction for the next 30", "5m", ser(mom), px),
    ]


def _prior_session_returns(g: pd.DataFrame, india_dates: pd.DatetimeIndex) -> pd.Series:
    """For each Indian trading date D: the global market's last *completed* session return with a local date
    strictly before D (so it finished before NIFTY opened)."""
    r = np.log(g["close"] / g["close"].shift(1)).dropna()
    left = pd.DataFrame({"d": pd.DatetimeIndex(india_dates.date).as_unit("ns")})
    right = pd.DataFrame({"d": pd.DatetimeIndex(r.index).as_unit("ns"), "r": r.to_numpy()}).sort_values("d")
    m = pd.merge_asof(left.sort_values("d"), right, on="d", direction="backward", allow_exact_matches=False)
    return pd.Series(m["r"].to_numpy(), index=india_dates)


def global_daily_tests(sym: str, d: pd.DataFrame, gdaily: dict) -> tuple[list[Result], list[dict]]:
    """Trades: open→close in the direction the prior global session implies for India. Links: how much of
    the opening gap each global market explains (informational: the gap happens before the desk can trade)."""
    from ..data.global_universe import GLOBAL
    d = d[d["open"] > 0].copy()
    r_oc = np.log(d["close"] / d["open"])
    gap = np.log(d["open"] / d["close"].shift(1))
    px = float(d["close"].iloc[-1])
    res, links = [], []
    for key, g in gdaily.items():
        meta = GLOBAL.get(key, {})
        if g is None or len(g) < 300:
            continue
        x = _prior_session_returns(g, d.index)
        ok = x.notna() & r_oc.notna() & (x != 0)
        if ok.sum() < 250:
            continue
        sign = meta.get("india", 0) or 1
        res.append(evaluate(f"G-{key}", sym, f"{meta.get('name', key)} prior session → trade NIFTY-side "
                            f"{'with' if sign > 0 else 'against'} it, open→close" if sym == "NIFTY" else
                            f"{meta.get('name', key)} prior session → trade {'with' if sign > 0 else 'against'} it, open→close",
                            "daily+global", np.sign(sign * x[ok]) * r_oc[ok], px))
        okg = x.notna() & gap.notna()
        if okg.sum() > 250:
            xs, ys = x[okg].to_numpy(), gap[okg].to_numpy()
            beta = float(np.cov(xs, ys)[0, 1] / np.var(xs))
            resid = ys - beta * xs
            se = math.sqrt(np.var(resid) / (np.var(xs) * len(xs)))
            links.append({"from": key, "name": meta.get("name", key), "to": sym, "what": "opening gap", "n": int(okg.sum()),
                          "beta": round(beta, 3), "t": round(beta / se, 2), "corr": round(float(np.corrcoef(xs, ys)[0, 1]), 3),
                          "r2": round(float(np.corrcoef(xs, ys)[0, 1] ** 2), 3)})
    return res, links


def _asof(df5: pd.DataFrame, times: pd.DatetimeIndex, tol: str = "10min") -> np.ndarray:
    """Price known at each time: a 5m bar's close is known at its start + 5 minutes; stale beyond `tol` → NaN."""
    s = df5["close"].copy()
    s.index = pd.DatetimeIndex(s.index).tz_convert("UTC") + pd.Timedelta(minutes=5)
    right = pd.DataFrame({"t": s.index.as_unit("ns"), "px": s.to_numpy()}).sort_values("t")
    left = pd.DataFrame({"t": pd.DatetimeIndex(times).tz_convert("UTC").as_unit("ns")})
    m = pd.merge_asof(left, right, on="t", direction="backward", tolerance=pd.Timedelta(tol))
    return m["px"].to_numpy()


def global_intraday_tests(sym: str, m5: pd.DataFrame, gm5: dict) -> tuple[list[Result], list[dict]]:
    """Lead-lag: does a global market's last 30 minutes predict NIFTY's next 30 (non-overlapping blocks)?
    Links: contemporaneous 5m correlation during Indian hours (moves together, not predictive)."""
    from ..data.global_universe import GLOBAL
    m = _session(m5)
    if m.empty:
        return [], []
    px = float(m["close"].iloc[-1])
    grid = []
    for day in sorted(set(m.index.date)):
        base = pd.Timestamp(day, tz=IST) + pd.Timedelta(hours=9, minutes=45)
        grid += [base + pd.Timedelta(minutes=30 * k) for k in range(11)]          # 09:45 … 14:45, outcome to +30m
    grid = pd.DatetimeIndex(grid)
    n_now, n_next = _asof(m, grid), _asof(m, grid + pd.Timedelta(minutes=30))
    outcome = np.log(n_next / n_now)
    res, links = [], []
    nm = m.copy()
    nm.index = pd.DatetimeIndex(nm.index).tz_convert("UTC")
    n_ret5 = np.log(nm["close"]).diff()
    for key, g in gm5.items():
        meta = GLOBAL.get(key, {})
        g_now, g_prev = _asof(g, grid), _asof(g, grid - pd.Timedelta(minutes=30))
        sig = np.log(g_now / g_prev)
        ok = np.isfinite(sig) & np.isfinite(outcome) & (sig != 0)
        if ok.sum() < 100:
            continue
        sign = meta.get("india", 0) or 1
        res.append(evaluate(f"L-{key}", sym, f"{meta.get('name', key)} last 30 min → {sym} next 30 min "
                            f"({'with' if sign > 0 else 'against'} it)", "5m+global",
                            pd.Series(np.sign(sign * sig[ok]) * outcome[ok], index=grid[ok]), px))
        gg = g.copy()
        gg.index = pd.DatetimeIndex(gg.index).tz_convert("UTC")
        g_ret5 = np.log(gg["close"]).diff()
        j = pd.concat([n_ret5.rename("n"), g_ret5.rename("g")], axis=1, join="inner").dropna()
        j = j[(j["n"] != 0) & (j["g"] != 0)]
        if len(j) > 200:
            c = float(j["n"].corr(j["g"]))
            links.append({"from": key, "name": meta.get("name", key), "to": sym, "what": "same 5 minutes", "n": int(len(j)),
                          "beta": round(float(np.cov(j["g"], j["n"])[0, 1] / np.var(j["g"])), 3),
                          "t": round(c * math.sqrt((len(j) - 2) / max(1 - c * c, 1e-9)), 2), "corr": round(c, 3), "r2": round(c * c, 3)})
    return res, links


def run(data: dict, symbols=("NIFTY", "BANKNIFTY"), q: float = 0.10) -> list[Result]:
    res: list[Result] = []
    vix = data["daily"].get("INDIAVIX")
    for s in symbols:
        if s in data["daily"]:
            res += daily_tests(s, data["daily"][s], vix)
        if s in data.get("hourly", {}):
            res += hourly_tests(s, data["hourly"][s], data["daily"][s])
        if s in data.get("m5", {}):
            res += m5_tests(s, data["m5"][s])
        glob = data.get("global") or {}
        if glob.get("daily") and s in data["daily"]:
            r_, l_ = global_daily_tests(s, data["daily"][s], glob["daily"])
            res += r_
            data.setdefault("links", []).extend(l_)
        if glob.get("m5") and s in data.get("m5", {}):
            r_, l_ = global_intraday_tests(s, data["m5"][s], glob["m5"])
            res += r_
            data.setdefault("links", []).extend(l_)
    passed = benjamini_hochberg([r.params.get("p_discovery", r.p) for r in res], q)
    for r, ok in zip(res, passed):
        r.bh_pass = bool(ok)
        holds = r.p_holdout == r.p_holdout and r.p_holdout < 0.10
        if not (ok and holds):
            r.verdict = "NO EDGE"
        elif r.kind == "premium":
            r.verdict = "NEEDS MARGIN" if r.effect_bps > 0 else "NO EDGE"
        elif abs(r.effect_pts) < r.hurdle_pts:
            r.verdict = "REAL BUT BELOW COSTS"
        else:
            r.verdict = "EDGE"
    return res


def report(res: list[Result], data: dict, generated: str) -> str:
    span = {k: {s: f"{df.index[0]:%d-%b-%Y} → {df.index[-1]:%d-%b-%Y} ({len(df):,} bars)" for s, df in v.items()}
            for k, v in data.items() if k in ("daily", "hourly", "m5")}
    glob = data.get("global") or {}
    gspan = {f"global {k}": f"{len(v)} markets" for k, v in glob.items() if k in ("daily", "m5") and v}
    L = [f"# Edge research — NIFTY & BANKNIFTY, with global markets", "", f"Generated {generated}. Data: Yahoo Finance.", ""]
    for k, v in span.items():
        for s, t in v.items():
            L.append(f"- {k} {s}: {t}")
    for k, t in gspan.items():
        L.append(f"- {k}: {t}" + (f" (missing: {', '.join(glob.get('errors', {}))})" if glob.get("errors") else ""))
    L += ["", f"{len(res)} pre-registered tests · Benjamini–Hochberg q = 0.10 · holdout = newest third · cost hurdle "
          f"NIFTY {COST_POINTS['NIFTY']:.1f} pts, BANKNIFTY {COST_POINTS['BANKNIFTY']:.1f} pts per round trip", "",
          "Effect/trade is for the side the test states; a negative effect means the edge is the *opposite* side.", "",
          "| Verdict | ID | Market | Hypothesis | N | Effect/trade | t | p | Holdout | BH |",
          "|---|---|---|---|---:|---:|---:|---:|---:|:-:|"]
    order = {"EDGE": 0, "NEEDS MARGIN": 1, "REAL BUT BELOW COSTS": 2, "NO EDGE": 3}
    for r in sorted(res, key=lambda r: (order.get(r.verdict, 9), r.p)):
        eff = (f"{r.effect_bps:+.1f} bps ({r.effect_pts:+.1f} pts)" if r.kind == "directional"
               else f"{r.effect_bps / 100:+.2f} vol pts")
        hold = f"{r.effect_holdout_bps:+.1f} bps, p {r.p_holdout:.2f}" if r.kind == "directional" else f"p {r.p_holdout:.2f}"
        L.append(f"| **{r.verdict}** | {r.id} | {r.symbol} | {r.hypothesis} | {r.n:,} | {eff} | {r.t:+.2f} | {r.p:.3f} | "
                 f"{hold} | {'✓' if r.bh_pass else '·'} |")
    notes = [r for r in res if r.note or r.params]
    if notes:
        L += ["", "Notes:"]
        for r in notes:
            L.append(f"- {r.id} {r.symbol}: {r.note} {json.dumps(r.params) if r.params else ''}".strip())
    links = data.get("links") or []
    if links:
        L += ["", "## Global links (how markets move together; not trades)", "",
              "The opening gap happens before the desk can trade, and same-5-minute co-movement isn't a forecast; they "
              "explain *why* NIFTY is where it is, which is what the brain uses them for.", "",
              "| From | To | What | N | β | corr | R² | t |", "|---|---|---|---:|---:|---:|---:|---:|"]
        for l in sorted(links, key=lambda l: (-abs(l["t"]) if l["t"] == l["t"] else 0)):
            L.append(f"| {l['name']} | {l['to']} | {l['what']} | {l['n']:,} | {l['beta']:+.3f} | {l['corr']:+.3f} | "
                     f"{l['r2']:.3f} | {l['t']:+.1f} |")
    edges = [r for r in res if r.verdict == "EDGE"]
    L += ["", "## Verdict", ""]
    if edges:
        L.append("Survived discovery, holdout, false-discovery control and the cost hurdle:")
        for r in edges:
            side = "as stated" if r.effect_pts > 0 else "**the opposite side** (the effect is negative)"
            L.append(f"- **{r.id} {r.symbol}**: {r.hypothesis}. Trade {side}: {abs(r.effect_pts):.1f} pts/trade vs "
                     f"{r.hurdle_pts:.1f} pts of costs (t {r.t:+.2f}, holdout {r.effect_holdout_bps:+.1f} bps). "
                     f"Per-trade σ {r.sd_pts:,.0f} pts, so one lot of a 0.35Δ option is a half-Kelly bet only on an account of "
                     f"about ₹{r.capital_half_kelly:,.0f}; smaller accounts are over-betting it.")
    else:
        L.append("Nothing survived every test. Trading any of these would be trading noise; the desk won't.")
    return "\n".join(L) + "\n"


def links_json(data: dict) -> str:
    return json.dumps(data.get("links") or [], indent=1)


def to_json(res: list[Result]) -> str:
    return json.dumps([{k: (None if isinstance(v, float) and v != v else v) for k, v in asdict(r).items()} for r in res], indent=1)
