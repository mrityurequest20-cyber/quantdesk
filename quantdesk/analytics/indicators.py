"""Technical indicators. Every function is *causal*: the value at bar t uses bars <= t only
(tests/test_causality.py enforces this by recomputing on truncated data)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = wilder(d.clip(lower=0), n)
    dn = wilder((-d).clip(lower=0), n)
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(dn != 0, 100.0).where(up.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    upper, lower = mid + k * sd, mid - k * sd
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower,
                         "pct_b": (close - lower) / (upper - lower),
                         "bandwidth": (upper - lower) / mid})


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.where(prev.notna(), df["high"] - df["low"])


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return wilder(true_range(df), n)


def adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = wilder(true_range(df), n)
    plus_di = 100 * wilder(plus_dm, n) / tr
    minus_di = 100 * wilder(minus_dm, n) / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return pd.DataFrame({"adx": wilder(dx, n), "plus_di": plus_di, "minus_di": minus_di})


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0) -> pd.DataFrame:
    a = atr(df, n).to_numpy()
    hl2 = ((df["high"] + df["low"]) / 2).to_numpy()
    close = df["close"].to_numpy()
    upper_b, lower_b = hl2 + mult * a, hl2 - mult * a
    line = np.full(len(df), np.nan)
    direction = np.zeros(len(df))
    fu, fl = np.nan, np.nan
    for i in range(len(df)):
        if np.isnan(a[i]):
            continue
        fu = upper_b[i] if np.isnan(fu) or upper_b[i] < fu or close[i - 1] > fu else fu
        fl = lower_b[i] if np.isnan(fl) or lower_b[i] > fl or close[i - 1] < fl else fl
        prev_dir = direction[i - 1] if i else 1
        if prev_dir >= 0:
            direction[i] = -1 if close[i] < fl else 1
        else:
            direction[i] = 1 if close[i] > fu else -1
        line[i] = fl if direction[i] > 0 else fu
    return pd.DataFrame({"line": line, "direction": direction}, index=df.index)


def donchian(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Channel of the *previous* n bars, so `close > upper` is a genuine breakout."""
    upper = df["high"].rolling(n, min_periods=n).max().shift(1)
    lower = df["low"].rolling(n, min_periods=n).min().shift(1)
    return pd.DataFrame({"upper": upper, "lower": lower, "mid": (upper + lower) / 2})


def keltner(df: pd.DataFrame, n: int = 20, mult: float = 2.0) -> pd.DataFrame:
    mid = ema(df["close"], n)
    a = atr(df, n)
    return pd.DataFrame({"mid": mid, "upper": mid + mult * a, "lower": mid - mult * a})


def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3) -> pd.DataFrame:
    lo = df["low"].rolling(k, min_periods=k).min()
    hi = df["high"].rolling(k, min_periods=k).max()
    pk = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    return pd.DataFrame({"k": pk, "d": pk.rolling(d, min_periods=d).mean()})


def obv(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df["close"].diff()).fillna(0) * df["volume"]).cumsum()


def rolling_vwap(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["volume"]).rolling(n, min_periods=n).sum() / df["volume"].rolling(n, min_periods=n).sum()


def zscore(s: pd.Series, n: int) -> pd.Series:
    m = s.rolling(n, min_periods=n).mean()
    sd = s.rolling(n, min_periods=n).std()
    return (s - m) / sd.replace(0, np.nan)


def roc(s: pd.Series, n: int) -> pd.Series:
    return s / s.shift(n) - 1


def pct_rank(s: pd.Series, n: int) -> pd.Series:
    """Rolling percentile of the current value within the last n (0..1)."""
    return s.rolling(n, min_periods=max(20, n // 4)).rank(pct=True)


def range_rank(s: pd.Series, n: int) -> pd.Series:
    """(x - min) / (max - min) over the last n — the 'IV rank' definition."""
    lo = s.rolling(n, min_periods=max(20, n // 4)).min()
    hi = s.rolling(n, min_periods=max(20, n // 4)).max()
    return (s - lo) / (hi - lo).replace(0, np.nan)


def efficiency_ratio(close: pd.Series, n: int = 20) -> pd.Series:
    """Kaufman: |net move| / sum(|bar moves|). 1 = straight line, ~0 = chop."""
    net = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n, min_periods=n).sum()
    return net / path.replace(0, np.nan)


def chandelier(df: pd.DataFrame, n: int = 22, mult: float = 3.0) -> pd.DataFrame:
    a = atr(df, n)
    return pd.DataFrame({"long_stop": df["high"].rolling(n, min_periods=n).max() - mult * a,
                         "short_stop": df["low"].rolling(n, min_periods=n).min() + mult * a})


def slope(s: pd.Series, n: int) -> pd.Series:
    """Least-squares slope of log(s) over n bars, annualised-ish (per bar)."""
    y = np.log(s)
    x = np.arange(n) - (n - 1) / 2
    denom = (x ** 2).sum()
    return y.rolling(n, min_periods=n).apply(lambda w: float((w * x).sum() / denom), raw=True)
