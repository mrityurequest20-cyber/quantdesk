"""Research on the data warehouse (real NSE data, 2019 →):

1. **The volatility risk premium, traded.** India VIX sits above the volatility NIFTY then delivers most of the
   time — but is that harvestable after real option prices, real spreads and the full Indian cost stack? Each
   strategy below is opened at the close k sessions before an expiry, at the bhavcopy's closing prices (worsened
   by a half-spread and a tick), and held to cash settlement at the index close on expiry day.
2. **Positioning.** Does FII / client index-futures and options positioning (participant-wise OI, published after
   the close) predict the next session(s)? Tested as trades entered at the next open.

Both lists were fixed before any result was seen. Newey-West t-statistics; each series is split in time, the older
2/3 for discovery and the newest 1/3 kept untouched; Benjamini-Hochberg FDR (q = 0.10) runs on the *discovery* p-values
within each family, and a discovery counts only if the holdout then keeps its sign (one-sided p < 0.10). (Taking the
FDR on the full sample, holdout included, let chance results confirm themselves: on simulated noise it "found" edges
in 3 of 20 worlds.)
"""
from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

from ..options.pricing import implied_vol_vec
from .edges import LOT, Result, _split, benjamini_hochberg, evaluate, hac_mean

R, Q = 0.065, 0.012
ACCOUNT = 20_000
STT_EXERCISE = 0.00125            # on the intrinsic value of a long option exercised at expiry
OFFSETS = {"NIFTY": (1, 3, 5), "BANKNIFTY": (1, 5, 10)}


# ---- 1. the volatility premium -------------------------------------------------------------------------------------
# name → (legs as (qty, right, target), filter) where target is "atm", a delta (calls +, puts −), or "atm±wing"
STRATEGIES = {
    "short_straddle":     ([(-1, "CE", "atm"), (-1, "PE", "atm")], None),
    "short_strangle_20d": ([(-1, "CE", 0.20), (-1, "PE", -0.20)], None),
    "iron_fly":           ([(-1, "CE", "atm"), (-1, "PE", "atm"), (1, "CE", "atm+wing"), (1, "PE", "atm-wing")], None),
    "iron_condor_20_10":  ([(-1, "CE", 0.20), (-1, "PE", -0.20), (1, "CE", 0.10), (1, "PE", -0.10)], None),
    "bull_put_30_15":     ([(-1, "PE", -0.30), (1, "PE", -0.15)], None),
    "bear_call_30_15":    ([(-1, "CE", 0.30), (1, "CE", 0.15)], None),
    "short_straddle_ivrv": ([(-1, "CE", "atm"), (-1, "PE", "atm")], "iv>rv"),
    "iron_fly_ivrv":      ([(-1, "CE", "atm"), (-1, "PE", "atm"), (1, "CE", "atm+wing"), (1, "PE", "atm-wing")], "iv>rv"),
}
NAKED = {"short_straddle", "short_strangle_20d", "short_straddle_ivrv"}


def leg_cost(price):
    """Execution cost per unit, index points: a tick of slippage plus a half-spread (≥ a tick, 0.2% of premium).
    The desk's recorded NSE/Kotak quotes show 0.03–0.15% half-spreads near the money; far wings are wider."""
    return 0.05 + np.maximum(0.05, 0.002 * np.asarray(price, dtype=float))


def _delta(S, K, T, sigma, right):
    d1 = (np.log(S / K) + (R - Q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return np.exp(-Q * T) * (ndtr(d1) if right == "CE" else ndtr(d1) - 1)


def _year_frac(d: dt.date, e: dt.date) -> float:
    return max((e - d).days, 0) / 365 + 1e-9


@dataclass
class VRPResult:
    symbol: str
    strategy: str
    k: int
    n: int
    mean_pts: float
    t: float
    p: float
    mean_holdout_pts: float
    p_holdout: float
    win_rate: float
    worst_rs: float
    mean_rs: float
    total_rs: float
    max_dd_rs: float
    sharpe: float
    capital_rs: float
    capital_note: str
    first: str
    last: str
    bh_pass: bool = False
    verdict: str = ""
    params: dict = field(default_factory=dict)


def build_trades(opts: pd.DataFrame, spot: pd.Series, symbol: str, fees=None, lot: int | None = None) -> pd.DataFrame:
    """Every (expiry, offset, strategy) trade the data allows → one row each: entry/expiry dates, P&L in index points
    per unit (after spreads, slippage, fees and exercise STT), credit, max loss, ATM IV, 10-day realised vol.

    `opts`: date, kind (CE/PE), expiry, strike, close, contracts. `spot`: index close by date.
    `fees(right, qty, price) → ₹` for one lot (the desk's cost model); None = no fees."""
    lot = lot or LOT.get(symbol, 1)
    spot = spot.dropna()
    days = sorted(spot.index)                                  # the trading calendar: every day the index closed
    pos = {d: i for i, d in enumerate(days)}
    rv10 = np.log(spot.reindex(days)).diff().rolling(10).std() * math.sqrt(252)
    expiries = sorted(set(opts["expiry"]))
    plan = []                                              # (entry day, expiry, k, settlement day)
    for e in expiries:
        settle = max((d for d in days if d <= e), default=None)
        if settle is None or settle < e - dt.timedelta(days=4) or settle not in pos:
            continue                                                       # data ends before this expiry
        for k in OFFSETS.get(symbol, (1, 3, 5)):
            i = pos[settle] - k
            if i >= 10 and not any(days[i] < x < e for x in expiries):    # trade the nearest expiry only
                plan.append((days[i], e, k, settle))
    if not plan:
        return pd.DataFrame()
    keys = pd.DataFrame({"date": [x[0] for x in plan], "expiry": [x[1] for x in plan]}).drop_duplicates()
    live = opts[(opts["close"] > 0) & (opts["contracts"] > 0)].merge(keys, on=["date", "expiry"])
    chains = {k: g for k, g in live.groupby(["date", "expiry"])}
    out = []
    for d, e, k, settle in plan:
        S_T = float(spot[settle])
        ch = chains.get((d, e))
        if ch is None or ch.empty:
            continue
        S, T = float(spot[d]), _year_frac(d, e)
        sides = {}
        for right in ("CE", "PE"):
            c = ch[ch["kind"] == right].drop_duplicates("strike").sort_values("strike")
            K, P = c["strike"].to_numpy(float), c["close"].to_numpy(float)
            iv = implied_vol_vec(P, S, K, T, R, Q, right)
            sides[right] = pd.DataFrame({"K": K, "P": P, "iv": iv, "delta": _delta(S, K, T, np.where(iv > 0, iv, np.nan), right)})
        both = set(sides["CE"]["K"]) & set(sides["PE"]["K"])
        if not both:
            continue
        K0 = min(both, key=lambda x: abs(x - S))
        ce0 = sides["CE"].set_index("K").loc[K0]
        pe0 = sides["PE"].set_index("K").loc[K0]
        atm_iv = float(np.nanmean([ce0["iv"], pe0["iv"]]))
        wing = float(ce0["P"] + pe0["P"])
        rv = float(rv10.get(d, np.nan))
        for name, (legs, filt) in STRATEGIES.items():
            if filt == "iv>rv" and not (atm_iv == atm_iv and rv == rv and atm_iv > rv):
                continue
            picked = []
            for qty, right, target in legs:
                s = sides[right]
                if target == "atm":
                    row = s[s["K"] == K0]
                elif isinstance(target, str):                               # atm±wing: the straddle's width out
                    goal = K0 + wing if target.endswith("+wing") else K0 - wing
                    row = s.iloc[[int(np.abs(s["K"] - goal).argmin())]] if len(s) else s
                else:
                    ok = s.dropna(subset=["delta"])
                    ok = ok[(ok["K"] > S) if right == "CE" else (ok["K"] < S)]
                    if ok.empty:
                        break
                    j = int(np.abs(ok["delta"] - target).argmin())
                    if abs(ok["delta"].iloc[j] - target) > 0.08:
                        break
                    row = ok.iloc[[j]]
                if row.empty:
                    break
                picked.append((qty, right, float(row["K"].iloc[0]), float(row["P"].iloc[0])))
            else:
                if len({(r_, k_) for _, r_, k_, _ in picked}) < len(picked):
                    continue                                             # a wing collapsed onto a short strike
                pnl, credit, fee_pts = 0.0, 0.0, 0.0
                for qty, right, K, P in picked:
                    intr = max(S_T - K, 0.0) if right == "CE" else max(K - S_T, 0.0)
                    c = float(leg_cost(P))
                    fill = P - c if qty < 0 else P + c
                    pnl += -qty * fill + qty * intr                              # sell: +fill − payout; buy: −fill + payout
                    credit += -qty * P
                    if fees is not None:
                        fee_pts += fees(right, qty * lot, fill) / lot
                    if qty > 0 and intr > 0:
                        fee_pts += STT_EXERCISE * intr
                pnl -= fee_pts
                strikes = sorted(k_ for _, _, k_, _ in picked)
                defined = name not in NAKED
                if defined:
                    calls = [k_ for q_, r_, k_, _ in picked if r_ == "CE"]
                    puts = [k_ for q_, r_, k_, _ in picked if r_ == "PE"]
                    width = max((max(calls) - min(calls)) if len(calls) > 1 else 0, (max(puts) - min(puts)) if len(puts) > 1 else 0)
                    max_loss = max(width - credit, 0.0) + fee_pts
                else:
                    max_loss = float("nan")
                out.append({"symbol": symbol, "strategy": name, "k": k, "entry": d, "expiry": e, "S": S, "S_T": S_T,
                            "pnl_pts": pnl, "credit_pts": credit, "max_loss_pts": max_loss, "atm_iv": atm_iv, "rv10": rv,
                            "strikes": strikes})
    return pd.DataFrame(out)


def evaluate_vrp(trades: pd.DataFrame, lot_of=LOT, account: float = ACCOUNT, q: float = 0.10) -> list[VRPResult]:
    res = []
    if trades.empty:
        return res
    for (sym, name, k), g in trades.sort_values("entry").groupby(["symbol", "strategy", "k"]):
        x = g.set_index(pd.to_datetime(g["entry"]))["pnl_pts"]
        if len(x) < 20:
            continue
        lot = lot_of.get(sym, 1)
        m, t, p = hac_mean(x.to_numpy())
        disc, hold = _split(x)
        md, _, pd_ = hac_mean(disc.to_numpy())
        mh, _, ph = hac_mean(hold.to_numpy())
        p1 = ph / 2 if (mh == mh and md == md and np.sign(mh) == np.sign(md)) else 1.0
        rs = x * lot
        eq = rs.cumsum()
        per_year = len(x) / max((x.index[-1] - x.index[0]).days / 365.25, 0.25)
        sd = float(x.std())
        if name in NAKED:
            cap = 0.12 * float(g["S"].iloc[-1]) * lot
            note = "≈12% of notional (SPAN + exposure, naked short)"
        else:
            cap = float(g["max_loss_pts"].quantile(0.9)) * lot
            note = "90th-percentile max loss per lot (defined risk)"
        res.append(VRPResult(sym, name, int(k), len(x), m, t, p, mh, p1, float((x > 0).mean()), float(rs.min()),
                             float(rs.mean()), float(rs.sum()), float((eq - eq.cummax()).min()),
                             float(m / sd * math.sqrt(per_year)) if sd > 0 else float("nan"), cap, note,
                             str(x.index[0].date()), str(x.index[-1].date()),
                             params={"p_discovery": pd_, "median_credit_pts": float(g["credit_pts"].median()),
                                     "median_atm_iv": float(g["atm_iv"].median()),
                                     "median_rv10": float(g["rv10"].median())}))
    for r, ok in zip(res, benjamini_hochberg([r.params["p_discovery"] for r in res], q)):
        r.bh_pass = bool(ok)
        holds = r.p_holdout < 0.10
        if not (ok and holds):
            r.verdict = "NO EDGE"
        elif r.mean_pts > 0:
            r.verdict = "EDGE (fits ₹20k)" if r.capital_rs <= account else f"EDGE, needs ~₹{r.capital_rs / 1e3:,.0f}k"
        else:
            r.verdict = "RELIABLY LOSES"
    return res


# ---- 2. positioning ------------------------------------------------------------------------------------------------
POSITIONING = [
    ("P1", "FII index-futures net change → next session, open→close", "fii_fut_d1", "oc1", 1),
    ("P2", "FII long ratio at a 1-yr extreme → fade it over 5 sessions", "fii_ratio_extreme", "oc5", 1),
    ("P3", "Client index-futures net change, contrarian → next session", "client_fut_d1", "oc1", -1),
    ("P4", "FII index-options net (calls − puts) change → next session", "fii_opt_d1", "oc1", 1),
    ("P5", "FII futures net 5-day change → next 5 sessions", "fii_fut_d5", "oc5", 1),
]


def positioning_features(part: pd.DataFrame) -> pd.DataFrame:
    """Participant OI (long rows: date, participant, …) → one row per date of signals, known after that date's close."""
    p = part.copy()
    p["date"] = pd.to_datetime(p["date"])
    fii = p[p["participant"] == "FII"].set_index("date").sort_index()
    cli = p[p["participant"] == "Client"].set_index("date").sort_index()
    f = pd.DataFrame(index=fii.index)
    net = fii["fut_idx_long"] - fii["fut_idx_short"]
    ratio = fii["fut_idx_long"] / (fii["fut_idx_long"] + fii["fut_idx_short"])
    opt = (fii["opt_idx_call_long"] - fii["opt_idx_call_short"]) - (fii["opt_idx_put_long"] - fii["opt_idx_put_short"])
    f["fii_fut_d1"] = np.sign(net.diff())
    f["fii_fut_d5"] = np.sign(net - net.shift(5))
    f["fii_opt_d1"] = np.sign(opt.diff())
    lo, hi = ratio.rolling(250, min_periods=120).quantile(0.2), ratio.rolling(250, min_periods=120).quantile(0.8)
    f["fii_ratio_extreme"] = np.where(ratio <= lo, 1.0, np.where(ratio >= hi, -1.0, np.nan))   # fade the extreme
    cnet = cli["fut_idx_long"] - cli["fut_idx_short"]
    f["client_fut_d1"] = np.sign(cnet.reindex(f.index).diff())
    f["fii_ratio"] = ratio
    return f


def positioning_targets(daily: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """For each feature date d: the next session's open→close (oc1) and next-open → close 5 sessions on (oc5), logs."""
    d = daily.copy()
    d.index = pd.DatetimeIndex([pd.Timestamp(x).tz_localize(None).normalize() if pd.Timestamp(x).tzinfo else
                                pd.Timestamp(x).normalize() for x in d.index])
    d = d[~d.index.duplicated()].sort_index()
    nxt = d.index.searchsorted(dates, side="right")
    oc1, oc5 = np.full(len(dates), np.nan), np.full(len(dates), np.nan)
    for j, i in enumerate(nxt):
        if i < len(d):
            oc1[j] = math.log(d["close"].iloc[i] / d["open"].iloc[i])
        if i + 4 < len(d):
            oc5[j] = math.log(d["close"].iloc[i + 4] / d["open"].iloc[i])
    return pd.DataFrame({"oc1": oc1, "oc5": oc5}, index=dates)


def run_positioning(part: pd.DataFrame, daily: dict, q: float = 0.10) -> list[Result]:
    f = positioning_features(part)
    res = []
    for sym in ("NIFTY", "BANKNIFTY"):
        if sym not in daily:
            continue
        tgt = positioning_targets(daily[sym], f.index)
        price = float(daily[sym]["close"].iloc[-1])
        for rid, hyp, feat, horizon, sign in POSITIONING:
            sig = (f[feat] * sign).replace(0, np.nan).dropna()
            if horizon == "oc5":
                # positions open on consecutive signal days overlap: count each 5-session bet once, on the
                # calendar of all participant dates (so a gap in the signal doesn't shorten the spacing)
                cal = pd.Series(np.arange(len(f.index)), index=f.index)
                spaced, last = [], -5
                for d in sig.index:
                    if cal[d] - last >= 5:
                        spaced.append(d)
                        last = cal[d]
                sig = sig.loc[spaced]
            x = (sig * tgt[horizon].reindex(sig.index)).dropna()
            if len(x) < 40:
                continue
            r = evaluate(rid, sym, hyp, f"participant OI {f.index[0]:%b %Y}–{f.index[-1]:%b %Y}", x, price)
            r.params["p_discovery"] = hac_mean(_split(x)[0].to_numpy())[2]
            res.append(r)
    for r, ok in zip(res, benjamini_hochberg([r.params["p_discovery"] for r in res], q)):
        r.bh_pass = bool(ok)
        holds = r.p_holdout == r.p_holdout and r.p_holdout < 0.10
        r.verdict = ("NO EDGE" if not (ok and holds) else
                     "REAL BUT BELOW COSTS" if abs(r.effect_pts) < r.hurdle_pts else "EDGE")
    return res


# ---- loading and reporting ------------------------------------------------------------------------------------------
def load_options(folder: Path, symbols=("NIFTY", "BANKNIFTY")) -> pd.DataFrame:
    import pyarrow.parquet as pq
    parts = []
    for p in sorted(Path(folder).glob("fo_bhav_*.parquet")):
        t = pq.read_table(p, columns=["date", "symbol", "kind", "expiry", "strike", "close", "underlying", "contracts"],
                          filters=[("symbol", "in", list(symbols))])
        parts.append(t.to_pandas())
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    return df


def spot_series(opts: pd.DataFrame, symbol: str, daily: pd.DataFrame | None) -> pd.Series:
    """The index close by date: the bhavcopy's own underlying price where it has one (UDiFF), else Yahoo's close."""
    s = opts[opts["symbol"] == symbol].dropna(subset=["underlying"]).groupby("date")["underlying"].median()
    if daily is not None and len(daily):
        y = daily["close"].copy()
        y.index = [pd.Timestamp(x).date() for x in y.index]
        y = y[~pd.Index(y.index).duplicated(keep="last")]
        s = s.combine_first(y)
    return s.sort_index()


def run_all(folder: Path, daily: dict, cfg=None) -> dict:
    out = {"vrp": [], "positioning": [], "trades": 0, "span": {}}
    opts = load_options(folder)
    if cfg is not None:
        from ..core.types import Instrument
        from ..intraday.sim import IntradayBroker
        costs = IntradayBroker(cfg).costs
        exp = dt.date(2030, 1, 1)

        def fees_for(sym):
            return lambda right, qty, price: costs.fees(Instrument.option(sym, exp, 1.0, right, LOT[sym]), int(qty), float(price))[0]
    trades = []
    for sym in ("NIFTY", "BANKNIFTY"):
        o = opts[(opts["symbol"] == sym) & opts["kind"].isin(["CE", "PE"])] if len(opts) else opts
        if o.empty:
            continue
        sp = spot_series(opts, sym, daily.get(sym))
        tr = build_trades(o, sp, sym, fees_for(sym) if cfg is not None else None)
        trades.append(tr)
        out["span"][sym] = f"{min(o['date'])} → {max(o['date'])}"
    tr = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    out["trades"] = len(tr)
    out["vrp"] = evaluate_vrp(tr)
    out["trade_rows"] = tr
    part_files = sorted(Path(folder).glob("participant_oi_*.parquet"))
    if part_files:
        part = pd.concat([pd.read_parquet(p) for p in part_files], ignore_index=True)
        out["positioning"] = run_positioning(part, daily)
    return out


def report(res: dict, generated: str) -> str:
    L = ["# Warehouse research — real NSE option prices and positioning", "",
         f"Generated {generated}. Data: NSE F&O bhavcopy and participant-wise OI (the `warehouse` release), "
         f"index closes from the bhavcopy or Yahoo. {res['trades']:,} option trades simulated.", "",
         "## 1. The volatility premium, traded on real prices", "",
         "Opened at the close k sessions before expiry at bhavcopy closing prices worsened by a half-spread and a tick, "
         "plus brokerage, STT, exchange, GST, stamp and exercise STT; held to cash settlement. ₹ figures are per lot "
         "at today's lot sizes (NIFTY 65, BANKNIFTY 30). Benjamini-Hochberg across all rows below.", "",
         "| verdict | strategy | index | k | trades | mean ₹/lot | t | holdout ₹/lot | win | worst ₹ | max DD ₹ | Sharpe | capital |",
         "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    order = lambda v: (0 if v.startswith("EDGE (fits") else 1 if v.startswith("EDGE") else 2 if v == "RELIABLY LOSES" else 3)  # noqa: E731
    for r in sorted(res["vrp"], key=lambda r: (order(r.verdict), -abs(r.t))):
        lot = LOT.get(r.symbol, 1)
        L.append(f"| **{r.verdict}** | {r.strategy} | {r.symbol} | {r.k} | {r.n} | {r.mean_pts * lot:+,.0f} | {r.t:+.2f} | "
                 f"{r.mean_holdout_pts * lot:+,.0f} | {r.win_rate:.0%} | {r.worst_rs:,.0f} | {r.max_dd_rs:,.0f} | "
                 f"{r.sharpe:.2f} | ₹{r.capital_rs:,.0f} |")
    L += ["", "## 2. Positioning (participant-wise OI)", "",
          "Signals known after the close, traded from the next open. Effect in basis points of the index per trade; "
          "the cost hurdle is one lot of a 0.35Δ option in and out.", "",
          "| verdict | id | index | hypothesis | n | effect bps | t | p | holdout bps |", "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for r in sorted(res["positioning"], key=lambda r: (r.verdict != "EDGE", r.p)):
        L.append(f"| **{r.verdict}** | {r.id} | {r.symbol} | {r.hypothesis} | {r.n:,} | {r.effect_bps:+.1f} | {r.t:+.2f} | "
                 f"{r.p:.3f} | {r.effect_holdout_bps:+.1f} |")
    return "\n".join(L) + "\n"


def to_json(res: dict) -> str:
    def clean(d):
        return {k: (None if isinstance(v, float) and v != v else v) for k, v in d.items()}
    return json.dumps({"vrp": [clean(asdict(r)) for r in res["vrp"]],
                       "positioning": [clean(asdict(r)) for r in res["positioning"]]}, indent=1, default=str)
