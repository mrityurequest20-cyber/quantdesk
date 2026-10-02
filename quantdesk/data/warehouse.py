"""The market-data warehouse: tidy Parquet tables built from NSE's public files, kept as assets of the
`warehouse` GitHub release (release assets don't bloat the git history; a branch would).

Tables (one file per period, `{table}_{period}.parquet`):
  fo_bhav           month  every index future and option, daily: OHLC, close, settle, underlying, OI, volume
  participant_oi    year   FII / DII / Pro / Client open interest by product, daily (contracts)
  participant_vol   year   the same for volume
  fii_dii           year   FII/FPI and DII cash-market buy/sell/net, ₹ crore (NSE gives only the latest day:
                           this table grows from the day collection started)
  gift_nifty        year   GIFT Nifty prints with NIFTY's prior close (also from collection start)
  corp_events       year   board meetings / results from NSE's event calendar
  nse_holidays      year   the exchange's F&O holiday list
  manifest          year   provenance: every file fetched, its URL, SHA-256, size, rows, and whether NSE had it

`update()` fills every missing day in a range (so a late or skipped run catches up by itself) and writes each
period once. Raw files aren't stored — they stay downloadable from NSE's archives and the manifest's checksums
say exactly which bytes each row came from."""
from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

import pandas as pd

from . import nse as N

IST = "Asia/Kolkata"
TABLES = {                                   # table → (period, key columns, date column)
    "fo_bhav": ("month", ["date", "symbol", "kind", "expiry", "strike"], "date"),
    "participant_oi": ("year", ["date", "participant"], "date"),
    "participant_vol": ("year", ["date", "participant"], "date"),
    "fii_dii": ("year", ["date", "category"], "date"),
    "gift_nifty": ("year", ["ts"], "ts"),
    "corp_events": ("year", ["date", "symbol", "purpose"], "date"),
    "nse_holidays": ("year", ["date"], "date"),
    "manifest": ("year", ["table", "date"], "date"),
}
DATE_COLS = ("date", "expiry")
DAILY_FILES = ("fo_bhav", "participant_oi", "participant_vol")
SETTLE_DAYS = 3                              # a file still missing this many days later is taken as never coming


class Warehouse:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.dirty: set[str] = set()                                    # file names written since load

    @staticmethod
    def period(table: str, d) -> str:
        d = pd.Timestamp(d)
        return f"{d:%Y-%m}" if TABLES[table][0] == "month" else f"{d:%Y}"

    def path(self, table: str, period: str) -> Path:
        return self.root / f"{table}_{period}.parquet"

    def files(self, table: str) -> list[Path]:
        return sorted(self.root.glob(f"{table}_*.parquet"))

    def read(self, table: str, start=None, end=None) -> pd.DataFrame:
        col = TABLES[table][2]
        parts = []
        for p in self.files(table):
            per = p.stem[len(table) + 1:]
            lo, hi = pd.Period(per).start_time, pd.Period(per).end_time
            if (start is not None and hi < pd.Timestamp(start)) or (end is not None and lo > pd.Timestamp(end)):
                continue
            parts.append(pd.read_parquet(p))
        if not parts:
            return pd.DataFrame()
        df = pd.concat(parts, ignore_index=True)
        t = pd.to_datetime(df[col])
        t = t.dt.tz_localize(None) if getattr(t.dt, "tz", None) is not None else t
        keep = pd.Series(True, index=df.index)
        if start is not None:
            keep &= t >= pd.Timestamp(start)
        if end is not None:
            keep &= t <= pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        return df[keep].reset_index(drop=True)

    def upsert(self, table: str, df: pd.DataFrame) -> int:
        """Add rows (replacing any with the same keys) to the table's period files. Returns rows written."""
        if df is None or df.empty:
            return 0
        _, keys, col = TABLES[table]
        df = df.copy()
        for c in DATE_COLS:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])
        per = pd.to_datetime(df[col]).map(lambda x: self.period(table, x))
        n = 0
        for p, part in df.groupby(per):
            path = self.path(table, p)
            if path.exists():
                old = pd.read_parquet(path)
                part = pd.concat([old, part], ignore_index=True)
            part = part.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)
            part.to_parquet(path, index=False, compression="zstd")
            self.dirty.add(path.name)
            n += len(part)
        return n

    def coverage(self) -> pd.DataFrame:
        """Rows, first/last day and day count per table. The daily-file tables are counted from the manifest, so
        this works with only the small files pulled."""
        rows = []
        man = self.read("manifest")
        for t, (_, _, col) in TABLES.items():
            if t in DAILY_FILES and not man.empty:
                m = man[(man["table"] == t) & (man["status"] == "ok")]
                d = pd.to_datetime(m["date"])
                rows.append({"table": t, "rows": int(m["rows"].sum()), "first": d.min().date() if len(d) else None,
                             "last": d.max().date() if len(d) else None, "days": d.nunique(),
                             "files": len(self.files(t))})
                continue
            df = self.read(t)
            if df.empty:
                rows.append({"table": t, "rows": 0, "first": None, "last": None, "days": 0, "files": 0})
                continue
            d = pd.to_datetime(df[col])
            d = d.dt.tz_localize(None) if getattr(d.dt, "tz", None) is not None else d
            rows.append({"table": t, "rows": len(df), "first": d.min().date(), "last": d.max().date(),
                         "days": d.dt.normalize().nunique(), "files": len(self.files(t))})
        return pd.DataFrame(rows)


# ---- filling it ---------------------------------------------------------------------------------------------------
def _settled(man: pd.DataFrame, table: str, today: dt.date) -> set[dt.date]:
    """Days a table needs nothing more for: fetched, or confirmed absent (NSE still had nothing SETTLE_DAYS later)."""
    if man.empty:
        return set()
    m = man[man["table"] == table]
    d = pd.to_datetime(m["date"]).dt.date
    fetched = pd.to_datetime(m["fetched"], utc=True).dt.tz_convert(IST).dt.date
    ok = m["status"] == "ok"
    gone = (m["status"] == "none") & ((fetched - d).map(lambda x: x.days) >= SETTLE_DAYS)
    return set(d[ok | gone])


def update(wh: Warehouse, nse: N.NSE, start: dt.date, end: dt.date, only: set[str] | None = None,
           holidays: set[dt.date] | None = None, say=print, today: dt.date | None = None,
           checkpoint_every: int = 60, on_checkpoint=None) -> dict:
    """Fetch every missing trading day in [start, end] for the daily files, plus today's API snapshots when the range
    reaches today. Returns counts per table."""
    sim = today is not None
    today = today or pd.Timestamp.now(tz=IST).date()
    only = only or set(DAILY_FILES) | {"fii_dii", "gift_nifty", "corp_events", "nse_holidays"}
    holidays = holidays or set()
    days = [d.date() for d in pd.bdate_range(start, min(end, today)) if d.date() not in holidays]
    man = wh.read("manifest")
    counts: dict[str, int] = {}
    stamp = (lambda: pd.Timestamp(today, tz=IST)) if sim else (lambda: pd.Timestamp.now(tz=IST))   # noqa: E731
    fetchers = {"fo_bhav": (nse.fo_bhav, N.parse_fo_bhav),
                "participant_oi": (lambda d: nse.participant("oi", d), N.parse_participant),
                "participant_vol": (lambda d: nse.participant("vol", d), N.parse_participant)}
    for table in [t for t in DAILY_FILES if t in only]:
        fetch, parse = fetchers[table]
        done = _settled(man, table, today)
        todo = [d for d in days if d not in done]
        if not todo:
            continue
        say(f"  {table}: {len(todo)} day(s) to fetch, {todo[0]} → {todo[-1]}")
        buf, mbuf = [], []
        for i, d in enumerate(todo, 1):
            try:
                b, url = fetch(d)
                df = parse(b, d)
                buf.append(df)
                mbuf.append({"table": table, "date": d, "status": "ok", "url": url, "sha256": N.sha256(b),
                             "bytes": len(b), "rows": len(df), "fetched": stamp()})
            except N.NotPublished as exc:
                mbuf.append({"table": table, "date": d, "status": "none", "url": str(exc), "sha256": "", "bytes": 0,
                             "rows": 0, "fetched": stamp()})
            except Exception as exc:                                       # leave it for the next run
                say(f"    {table} {d}: {exc!s:.200}")
            if i % checkpoint_every == 0 or i == len(todo):
                if buf:
                    wh.upsert(table, pd.concat(buf, ignore_index=True))
                if mbuf:
                    wh.upsert("manifest", pd.DataFrame(mbuf))
                got = sum(len(x) for x in buf)
                counts[table] = counts.get(table, 0) + got
                say(f"    … {i}/{len(todo)} days, {sum(r['status'] == 'ok' for r in mbuf)} files, {got:,} rows")
                buf, mbuf = [], []
                if on_checkpoint:
                    on_checkpoint()
    if end >= today - dt.timedelta(days=1):
        _snapshots(wh, nse, only, counts, say)
    return counts


def _snapshots(wh: Warehouse, nse: N.NSE, only: set[str], counts: dict, say) -> None:
    """The API endpoints only show the present: take today's reading of each."""
    jobs = [("fii_dii", "/api/fiidiiTradeReact", N.parse_fii_dii),
            ("gift_nifty", "/api/marketStatus", lambda j: N.parse_gift(j)),
            ("corp_events", "/api/event-calendar", N.parse_events),
            ("nse_holidays", "/api/holiday-master?type=trading",
             lambda j: pd.DataFrame(N.parse_holidays(j), columns=["date", "description"]))]
    for table, path, parse in jobs:
        if table not in only:
            continue
        try:
            j, raw, url = nse.api(path)
            df = parse(j)
            wh.upsert(table, df)
            counts[table] = len(df)
            wh.upsert("manifest", pd.DataFrame([{"table": table, "date": pd.Timestamp.now(tz=IST).date(), "status": "ok",
                                                 "url": url, "sha256": N.sha256(raw), "bytes": len(raw), "rows": len(df),
                                                 "fetched": pd.Timestamp.now(tz=IST)}]))
            say(f"  {table}: {len(df)} row(s)")
        except Exception as exc:
            say(f"  {table}: FAIL {exc!s:.200}")


def holiday_diff(wh: Warehouse, configured: set[dt.date], year: int) -> tuple[list, list]:
    """(NSE holidays the config lacks, config holidays NSE doesn't list) for one year, weekdays only."""
    h = wh.read("nse_holidays")
    if h.empty:
        return [], []
    nse = {(d.date(), desc) for d, desc in zip(pd.to_datetime(h["date"]), h["description"]) if d.year == year and d.weekday() < 5}
    mine = {d for d in configured if d.year == year and d.weekday() < 5}
    return sorted(x for x in nse if x[0] not in mine), sorted(d for d in mine if d not in {x[0] for x in nse})


# ---- the GitHub release that holds it ------------------------------------------------------------------------------
class ReleaseStore:
    """Pull/push the warehouse's files as assets of one GitHub release, through the `gh` CLI (GH_TOKEN)."""

    def __init__(self, tag: str = "warehouse", repo: str | None = None, run=subprocess.run,
                 title: str = "Market-data warehouse",
                 notes: str = "NSE end-of-day data as Parquet tables, maintained by data.yml "
                              "(quantdesk/data/warehouse.py). Not a software release."):
        self.tag, self.repo, self._run, self.title, self.notes = tag, repo, run, title, notes

    def _gh(self, *args, check=True) -> subprocess.CompletedProcess:
        cmd = ["gh", *args] + (["-R", self.repo] if self.repo else [])
        return self._run(cmd, check=check, capture_output=True, text=True)

    def assets(self) -> list[str]:
        r = self._gh("release", "view", self.tag, "--json", "assets", check=False)
        if r.returncode != 0:
            return []
        return [a["name"] for a in json.loads(r.stdout or "{}").get("assets", [])]

    def ensure(self) -> None:
        if self._gh("release", "view", self.tag, check=False).returncode != 0:
            self._gh("release", "create", self.tag, "--title", self.title, "--latest=false", "--notes", self.notes)

    def pull(self, dest: Path, names: list[str]) -> list[str]:
        have = set(self.assets())
        names = [n for n in names if n in have]
        for n in names:
            self._gh("release", "download", self.tag, "--pattern", n, "--dir", str(dest), "--clobber")
        return names

    def push(self, src: Path, names: list[str]) -> None:
        if not names:
            return
        self.ensure()
        for i in range(0, len(names), 20):
            self._gh("release", "upload", self.tag, *[str(src / n) for n in names[i:i + 20]], "--clobber")


def needed_assets(start: dt.date, end: dt.date, tables=TABLES) -> list[str]:
    """The period files an update over [start, end] reads or writes."""
    out = []
    for t, (per, _, _) in tables.items():
        rng = pd.period_range(start, end, freq="M" if per == "month" else "Y")
        out += [f"{t}_{p}.parquet" for p in rng.astype(str)]
    return out
