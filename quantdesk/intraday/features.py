"""Intraday market state for one underlying at one moment, computed only from bars that
have closed by `now` (the engine passes nothing later; tests enforce it).

Everything that depends only on *previous* sessions (prior-day levels and value area, CPR,
5-day range, relative-volume baselines, warmed-up 5m/15m history) is computed once per day
and cached; each minute only today's bars are reprocessed."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ..analytics import indicators as ind
from .orderflow import approx_delta, delta_divergence, profile_from_bars

OR_MIN, IB_MIN = 15, 60
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return df
    return df.resample(rule, label="left", closed="left", origin="start_day", offset="15min").agg(AGG).dropna(subset=["close"])


def _completed(agg: pd.DataFrame, rule: str, now) -> pd.DataFrame:
    return agg[agg.index + pd.Timedelta(rule) <= now]


def phase(minutes: float) -> str:
    if minutes < 30:
        return "opening"
    if minutes < 135:
        return "morning"
    if minutes < 255:
        return "midday"
    if minutes < 330:
        return "afternoon"
    return "closing"


def _day_groups(prior: pd.DataFrame) -> list[pd.DataFrame]:
    keys = prior.index.normalize()
    cut = np.flatnonzero(keys[1:] != keys[:-1]) + 1
    bounds = np.concatenate([[0], cut, [len(prior)]])
    return [prior.iloc[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]


def prior_context(prior: pd.DataFrame) -> dict:
    """Everything derived from completed sessions — computed once per day."""
    c: dict = {}
    if prior.empty:
        return c
    days = _day_groups(prior)
    last = days[-1]
    H, L, C = float(last["high"].max()), float(last["low"].min()), float(last["close"].iloc[-1])
    P, BC = (H + L + C) / 3, (H + L) / 2
    TC = 2 * P - BC
    c.update({"pdh": H, "pdl": L, "pdc": C, "cpr_p": P, "cpr_bc": min(BC, TC), "cpr_tc": max(BC, TC),
              "cpr_width": abs(TC - BC) / P})
    prof = profile_from_bars(last)
    if prof:
        c.update({"p_poc": prof.poc, "p_vah": prof.vah, "p_val": prof.val})
    c["avg_range_5d"] = float(np.mean([d["high"].max() / d["low"].min() - 1 for d in days[-5:]]))
    r = np.log(prior["close"]).diff().dropna().tail(375 * 5)
    c["rv_5d"] = float(r.std() * np.sqrt(375 * 252) * 100) if len(r) > 30 else np.nan
    curves = [d["volume"].cumsum().to_numpy() for d in days[-10:] if d["volume"].sum() > 0]
    if curves:
        n = max(len(x) for x in curves)
        padded = np.array([np.pad(x, (0, n - len(x)), mode="edge") for x in curves])
        c["vol_curve"] = padded.mean(axis=0)
    tail = pd.concat(days[-3:])
    c["b5_prior"] = _resample(tail, "5min")
    c["b15_prior"] = _resample(pd.concat(days[-6:]), "15min")
    return c


def session_state(bars: pd.DataFrame, now: pd.Timestamp, tick: float = 0.05, cache: dict | None = None) -> dict | None:
    """`bars`: 1m bars across several sessions, all closed by `now`. Returns a flat dict."""
    if bars is None or bars.empty:
        return None
    day_start = pd.Timestamp(dt.datetime.combine(now.date(), dt.time(9, 15)), tz=now.tz)
    i0 = bars.index.searchsorted(day_start)
    day, prior = bars.iloc[i0:], bars.iloc[:i0]
    if day.empty:
        return None
    key = (now.date(), len(prior))
    if cache is not None and cache.get("key") == key:
        pc = cache["prior"]
    else:
        pc = prior_context(prior)
        if cache is not None:
            cache.update({"key": key, "prior": pc})

    o, last = float(day["open"].iloc[0]), float(day["close"].iloc[-1])
    minutes = (now - day.index[0]).total_seconds() / 60
    s: dict = {"ts": now, "last": last, "open": o, "day_high": float(day["high"].max()), "day_low": float(day["low"].min()),
               "minutes": minutes, "phase": phase(minutes), "bars_today": len(day)}
    s.update({k: v for k, v in pc.items() if not k.startswith(("b5", "b15", "vol_curve"))})
    if "pdc" in pc:
        s["gap"], s["chg"] = o / pc["pdc"] - 1, last / pc["pdc"] - 1
        if "p_vah" in pc:
            s["open_vs_pva"] = "above" if o > pc["p_vah"] else "below" if o < pc["p_val"] else "inside"
    else:
        s["gap"], s["chg"] = 0.0, last / o - 1

    # VWAP (TWAP when the feed has no volume, e.g. Yahoo index bars) and its sigma bands
    vol = day["volume"].to_numpy(dtype=float)
    s["vwap_kind"] = "vwap" if vol.sum() > 0 else "twap"
    if vol.sum() <= 0:
        vol = np.ones(len(day))
    tp = ((day["high"] + day["low"] + day["close"]) / 3).to_numpy()
    cv = np.cumsum(vol)
    vwap = np.cumsum(tp * vol) / cv
    sd = np.sqrt(np.maximum(np.cumsum(vol * (tp - vwap) ** 2) / cv, 0))
    s.update({"vwap": float(vwap[-1]), "vwap_sd": float(sd[-1]),
              "vwap_z": float((last - vwap[-1]) / sd[-1]) if sd[-1] > 0 else 0.0,
              "vwap_slope": float((vwap[-1] / vwap[-16] - 1) if len(vwap) > 15 else 0.0)})
    lo5, hi5 = day["low"].to_numpy()[-5:], day["high"].to_numpy()[-5:]
    s["touched_vwap_5"] = bool(((lo5 <= vwap[-5:]) & (hi5 >= vwap[-5:])).any())

    # opening range and initial balance
    s.update({"or_high": float(day["high"].iloc[:OR_MIN].max()), "or_low": float(day["low"].iloc[:OR_MIN].min()),
              "or_done": minutes >= OR_MIN, "ib_high": float(day["high"].iloc[:IB_MIN].max()),
              "ib_low": float(day["low"].iloc[:IB_MIN].min()), "ib_done": minutes >= IB_MIN})
    ibr = s["ib_high"] - s["ib_low"]
    s["ib_ext"] = ((max(s["day_high"] - s["ib_high"], 0) + max(s["ib_low"] - s["day_low"], 0)) / ibr) if ibr > 0 and s["ib_done"] else 0.0

    # 5m / 15m structure: recomputed only when a new 5m (15m) bar completes
    n5, n15 = int(minutes // 5), int(minutes // 15)
    c5 = cache.setdefault("f5", {}) if cache is not None else {}
    if c5.get("key") != (key, n5):
        f5: dict = {}
        b5 = pd.concat([pc.get("b5_prior", pd.DataFrame()), _completed(_resample(day, "5min"), "5min", now)])
        if len(b5) >= 30:
            cl = b5["close"]
            f5.update({"ema9": float(ind.ema(cl, 9).iloc[-1]), "ema21": float(ind.ema(cl, 21).iloc[-1]),
                       "rsi5": float(ind.rsi(cl, 14).iloc[-1]), "adx5": float(ind.adx(b5.tail(120), 14)["adx"].iloc[-1]),
                       "st5": int(ind.supertrend(b5.tail(120), 10, 3)["direction"].iloc[-1]),
                       "atr5": float(ind.atr(b5.tail(120), 14).iloc[-1]),
                       "last5_close": float(cl.iloc[-1]), "last5_open": float(b5["open"].iloc[-1]),
                       "prev5_high": float(b5["high"].iloc[-2]), "prev5_low": float(b5["low"].iloc[-2]),
                       "range30_high": float(b5["high"].iloc[-7:-1].max()), "range30_low": float(b5["low"].iloc[-7:-1].min())})
        c5.update({"key": (key, n5), "f": f5})
    s.update(c5["f"])
    if "ema21" in s:
        s["touched_ema21_5"] = bool(((lo5 <= s["ema21"]) & (hi5 >= s["ema21"])).any())
    c15 = cache.setdefault("f15", {}) if cache is not None else {}
    if c15.get("key") != (key, n15):
        f15: dict = {}
        b15 = pd.concat([pc.get("b15_prior", pd.DataFrame()), _completed(_resample(day, "15min"), "15min", now)])
        if len(b15) >= 25:
            e = ind.ema(b15["close"], 20)
            f15["htf_slope"] = float(e.iloc[-1] / e.iloc[-4] - 1)
        c15.update({"key": (key, n15), "f": f15})
    s.update(c15["f"])

    # intraday realised vol (Parkinson on the last 30 one-minute bars, annualised)
    tail = day.tail(30)
    hl = np.log(tail["high"] / tail["low"]) ** 2
    s["rv_intraday"] = float(np.sqrt(hl.mean() / (4 * np.log(2)) * 375 * 252) * 100) if len(tail) >= 10 else np.nan

    # relative volume vs the same minute on previous sessions
    curve = pc.get("vol_curve")
    if curve is not None and day["volume"].sum() > 0:
        k = min(len(day), len(curve)) - 1
        s["rel_volume"] = float(day["volume"].sum() / curve[k]) if curve[k] > 0 else np.nan

    # order flow from bars (approximate) and the developing profile
    if day["volume"].sum() > 0:
        cvd = approx_delta(day).cumsum()
        s["cvd"] = float(cvd.iloc[-1])
        s["cvd_slope_30"] = float(cvd.iloc[-1] - cvd.iloc[-min(30, len(cvd))]) / max(day["volume"].tail(30).sum(), 1)
        s["cvd_divergence"] = delta_divergence(day["close"], cvd, 15)
    prof = profile_from_bars(day, tick=max(tick, (s["day_high"] - s["day_low"]) / 60 or tick))
    if prof:
        s.update({"poc": prof.poc, "vah": prof.vah, "val": prof.val, "value_pos": prof.position(last)})

    rng = s["day_high"] - s["day_low"]
    s["close_loc"] = (last - s["day_low"]) / rng if rng > 0 else 0.5
    s["range_vs_avg"] = (rng / o) / s["avg_range_5d"] if s.get("avg_range_5d") else np.nan
    return s
