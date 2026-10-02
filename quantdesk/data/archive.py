"""Keep what the desk records, for good: each session's option-chain snapshots and 1-minute bars (plus any other
table it writes, e.g. GIFT Nifty prints), compacted into Parquet and kept on a yearly `chains-YYYY` release.

The live runners' session data otherwise lives only in 90-day workflow artifacts, and intraday option history is
what costs money to buy: a year of the desk's own minute-by-minute chains is the most valuable dataset it can build.

One release asset per day, part and kind: `2026-10-05_morning-123_chains.parquet`, `…_bars.parquet`. The part names
the job that recorded it (the morning and afternoon runners each hold half a session)."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from ..intraday.chains import load_chain


def compact_day(day_dir: Path) -> dict[str, pd.DataFrame]:
    """A recorder day folder → {"chains": long frame, "bars": long frame, <other csv>: frame}."""
    out: dict[str, pd.DataFrame] = {}
    snaps = []
    for p in sorted((day_dir / "chains").glob("*.csv")):
        ch = load_chain(p)
        df = ch.reset_index()
        a = ch.attrs
        df.insert(0, "ts", pd.Timestamp(a["ts"]))
        df.insert(1, "underlying", a["underlying"])
        df.insert(2, "expiry", pd.Timestamp(a["expiry"]))
        df.insert(3, "spot", float(a["spot"]))
        df.insert(4, "source", str(a["source"]))
        snaps.append(df)
    if snaps:
        out["chains"] = pd.concat(snaps, ignore_index=True)
    bars = []
    for p in sorted(day_dir.glob("*_1m.csv")):
        df = pd.read_csv(p, parse_dates=["ts"])
        df.insert(0, "symbol", p.name[:-len("_1m.csv")])
        bars.append(df)
    if bars:
        out["bars"] = pd.concat(bars, ignore_index=True)
    for p in sorted(day_dir.glob("*.csv")):
        if not p.name.endswith("_1m.csv"):
            out[p.stem] = pd.read_csv(p)
    return out


def day_dirs(data_dir: Path, day: dt.date | None = None) -> list[Path]:
    out = []
    for p in sorted(Path(data_dir).glob("*")):
        try:
            d = dt.date.fromisoformat(p.name)
        except ValueError:
            continue
        if p.is_dir() and (day is None or d == day):
            out.append(p)
    return out


def archive(data_dir: Path, part: str, out_dir: Path, day: dt.date | None = None, store_for=None, say=print) -> list[str]:
    """Compact every recorded day (or one) under `data_dir`, write the Parquet files to `out_dir`, and push them to
    `store_for(year)` (a ReleaseStore) when given. Returns the file names written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[int, list[str]] = {}
    for dd in day_dirs(data_dir, day):
        frames = compact_day(dd)
        if not frames:
            continue
        for kind, df in frames.items():
            name = f"{dd.name}_{part}_{kind}.parquet"
            df.to_parquet(out_dir / name, index=False, compression="zstd")
            written.setdefault(int(dd.name[:4]), []).append(name)
        say(f"  {dd.name}: " + ", ".join(f"{k} {len(v):,} rows" for k, v in frames.items()))
    if store_for:
        for year, names in written.items():
            store_for(year).push(out_dir, names)
            say(f"  pushed {len(names)} file(s) to release chains-{year}")
    return [n for names in written.values() for n in names]


def load_archive(folder: Path, kind: str = "chains", start=None, end=None) -> pd.DataFrame:
    """Read archived files of one kind back (after `gh release download chains-YYYY`)."""
    parts = []
    for p in sorted(Path(folder).glob(f"*_{kind}.parquet")):
        d = dt.date.fromisoformat(p.name[:10])
        if (start and d < pd.Timestamp(start).date()) or (end and d > pd.Timestamp(end).date()):
            continue
        parts.append(pd.read_parquet(p))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    # a live run's artifact can carry earlier days too, so the same snapshot may be archived under two parts
    keys = {"chains": ["ts", "underlying", "expiry", "strike"], "bars": ["symbol", "ts"]}.get(kind)
    return df.drop_duplicates(keys).reset_index(drop=True) if keys else df.drop_duplicates().reset_index(drop=True)
