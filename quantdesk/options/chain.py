"""Option pricer (any contract, any date) and option-chain construction / chain analytics."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ..core.calendar import year_fraction
from ..core.types import Instrument
from .pricing import bs_price, greeks
from .surface import SkewModel


class OptionPricer:
    """Model prices from spot + an ATM implied vol (e.g. India VIX scaled by `iv_beta`)
    + a smile. This is what the backtester and paper broker use when no live quotes exist."""

    def __init__(self, r: float = 0.065, q: float = 0.012, skew: SkewModel | None = None):
        self.r, self.q = r, q
        self.skew = skew or SkewModel()

    @classmethod
    def from_config(cls, cfg) -> "OptionPricer":
        return cls(cfg.get("backtest.risk_free", 0.065), cfg.get("backtest.dividend_yield", 0.012))

    def forward(self, S: float, T: float) -> float:
        return S * np.exp((self.r - self.q) * T)

    def iv(self, K, S: float, T: float, atm_iv: float):
        return self.skew.iv(atm_iv, K, self.forward(S, T), T)

    def price(self, inst: Instrument, S: float, asof, atm_iv: float) -> float:
        T = year_fraction(asof, inst.expiry)
        if T <= 0:
            return max(S - inst.strike, 0.0) if inst.right == "CE" else max(inst.strike - S, 0.0)
        return float(bs_price(S, inst.strike, T, self.r, self.q, self.iv(inst.strike, S, T, atm_iv), inst.right))

    def greeks(self, inst: Instrument, S: float, asof, atm_iv: float) -> dict:
        T = year_fraction(asof, inst.expiry)
        return greeks(S, inst.strike, T, self.r, self.q, self.iv(inst.strike, S, T, atm_iv), inst.right)


def build_chain(pricer: OptionPricer, underlying: str, spot: float, asof, expiry: dt.date,
                atm_iv: float, strike_step: float, lot_size: int, n_strikes: int = 15) -> pd.DataFrame:
    atm = round(spot / strike_step) * strike_step
    strikes = atm + strike_step * np.arange(-n_strikes, n_strikes + 1)
    T = year_fraction(asof, expiry)
    rows = []
    for K in strikes:
        row = {"strike": K}
        for right in ("CE", "PE"):
            inst = Instrument.option(underlying, expiry, K, right, lot_size)
            g = pricer.greeks(inst, spot, asof, atm_iv)
            row[f"{right}_ltp"] = pricer.price(inst, spot, asof, atm_iv)
            row[f"{right}_iv"] = float(pricer.iv(K, spot, T, atm_iv)) * 100
            row[f"{right}_delta"] = g["delta"]
            row[f"{right}_theta"] = g["theta"]
        row["gamma"] = g["gamma"]
        row["vega"] = g["vega"]
        rows.append(row)
    df = pd.DataFrame(rows).set_index("strike")
    df.attrs.update({"underlying": underlying, "spot": spot, "expiry": expiry, "asof": str(asof), "T": T})
    return df


def expected_move(spot: float, iv: float, days: float) -> float:
    """One-sigma move over `days` calendar days implied by an IV (decimal)."""
    return spot * iv * np.sqrt(max(days, 0) / 365)


def straddle_move(chain: pd.DataFrame) -> float:
    """ATM straddle price ≈ 0.8 × one-sigma expected move (the trader's rule of thumb)."""
    spot = chain.attrs["spot"]
    k = chain.index[np.argmin(np.abs(chain.index - spot))]
    return float(chain.loc[k, "CE_ltp"] + chain.loc[k, "PE_ltp"])


def put_call_ratio(chain: pd.DataFrame) -> float | None:
    if "CE_oi" not in chain or "PE_oi" not in chain:
        return None
    return float(chain["PE_oi"].sum() / max(chain["CE_oi"].sum(), 1))


def max_pain(chain: pd.DataFrame) -> float | None:
    """Expiry price that minimises total intrinsic value owed to option buyers."""
    if "CE_oi" not in chain or "PE_oi" not in chain:
        return None
    ks = chain.index.to_numpy(dtype=float)
    pain = [(np.maximum(s - ks, 0) * chain["CE_oi"]).sum() + (np.maximum(ks - s, 0) * chain["PE_oi"]).sum() for s in ks]
    return float(ks[int(np.argmin(pain))])
