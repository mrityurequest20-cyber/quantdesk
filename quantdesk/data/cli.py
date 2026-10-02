"""`quantdesk data …`: the market-data warehouse (quantdesk/data/warehouse.py).

  update   fetch every missing day in a range (default: the last 10 days) and today's API snapshots;
           with --release, pull the files it touches from the GitHub release first and push what changed
  status   what the warehouse holds: rows, first/last day, gaps; NSE's holiday list vs the config's
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd

IST = "Asia/Kolkata"


def _dir(cfg, a) -> Path:
    return Path(a.dir) if a.dir else Path(cfg.runtime_dir) / "warehouse"


def cmd_update(cfg, a):
    from .nse import NSE
    from .warehouse import TABLES, ReleaseStore, Warehouse, needed_assets, update
    today = pd.Timestamp.now(tz=IST).date()
    end = dt.date.fromisoformat(a.to) if a.to else today
    start = dt.date.fromisoformat(a.start) if a.start else end - dt.timedelta(days=10)
    only = set(a.only.split(",")) if a.only else None
    if only and only - set(TABLES):
        sys.exit(f"unknown table(s): {', '.join(sorted(only - set(TABLES)))}")
    wh = Warehouse(_dir(cfg, a))
    store = ReleaseStore(a.release) if a.release else None
    if store:
        names = needed_assets(start, max(end, today))
        got = store.pull(wh.root, names)
        print(f"pulled {len(got)} file(s) from release '{a.release}'", flush=True)
        wh.dirty.clear()
    print(f"warehouse {wh.root} · {start} → {end}", flush=True)
    def checkpoint():                     # a long backfill keeps what it has fetched even if the job dies later
        if store and wh.dirty:
            store.push(wh.root, sorted(wh.dirty))
            print(f"    pushed {len(wh.dirty)} file(s)", flush=True)
            wh.dirty.clear()
    counts = update(wh, NSE(gap=a.gap), start, end, only, holidays=set(cfg.holidays()),
                    say=lambda m: print(m, flush=True), on_checkpoint=checkpoint)
    print("rows added: " + (", ".join(f"{k} {v:,}" for k, v in counts.items()) or "none"), flush=True)
    if store:
        n = len(wh.dirty)
        store.push(wh.root, sorted(wh.dirty))
        print(f"pushed {n} file(s) to release '{a.release}'", flush=True)


def cmd_status(cfg, a):
    from .warehouse import Warehouse, holiday_diff
    wh = Warehouse(_dir(cfg, a))
    if a.release:
        from .warehouse import ReleaseStore
        store = ReleaseStore(a.release)
        small = ("manifest_", "fii_dii_", "gift_nifty_", "corp_events_", "nse_holidays_")
        names = [n for n in store.assets() if n.endswith(".parquet") and (a.all or n.startswith(small))]
        store.pull(wh.root, names)
    cov = wh.coverage()
    print(f"## Data warehouse ({wh.root})\n")
    print("| table | rows | first | last | days | files |\n|---|---:|---|---|---:|---:|")
    for r in cov.itertuples():
        print(f"| {r.table} | {r.rows:,} | {r.first or '—'} | {r.last or '—'} | {r.days:,} | {r.files} |")
    man = wh.read("manifest")
    if not man.empty:
        bad = man[man["status"] == "none"]
        recent = bad[pd.to_datetime(bad["date"]) >= pd.Timestamp.now() - pd.Timedelta(days=30)]
        if len(recent):
            print("\nNot published by NSE in the last 30 days (holidays, or still pending):")
            for t, g in recent.groupby("table"):
                print(f"- {t}: {', '.join(str(pd.Timestamp(d).date()) for d in sorted(g['date']))}")
    year = pd.Timestamp.now(tz=IST).year
    missing, extra = holiday_diff(wh, set(cfg.holidays()), year)
    if missing or extra:
        print(f"\n**Holiday list check ({year}, F&O, weekdays):**")
        for d, desc in missing:
            print(f"- NSE lists {d} ({desc}) as a holiday; config/quantdesk.yaml doesn't")
        for d in extra:
            print(f"- config/quantdesk.yaml has {d} as a holiday; NSE doesn't")
    elif not wh.read("nse_holidays").empty:
        print(f"\nHoliday list check ({year}): config/quantdesk.yaml matches NSE's F&O holidays.")


def cmd_archive(cfg, a):
    from .archive import archive
    from .warehouse import ReleaseStore
    data = Path(a.data) if a.data else Path(cfg.runtime_dir) / "intraday" / "data"
    day = dt.date.fromisoformat(a.day) if a.day else None
    store_for = None
    if a.release_prefix:
        store_for = lambda y: ReleaseStore(f"{a.release_prefix}-{y}", title=f"Option-chain archive {y}",   # noqa: E731
                                           notes="The live desk's recorded option-chain snapshots and 1-minute bars, "
                                                 "one Parquet file per session part (quantdesk/data/archive.py). "
                                                 "Not a software release.")
    names = archive(data, a.part, Path(a.out), day, store_for, say=lambda m: print(m, flush=True))
    print(f"archived {len(names)} file(s) from {data}" if names else f"nothing recorded under {data}"
          + (f" for {day}" if day else ""))


def register(sub):
    s = sub.add_parser("data", help="the market-data warehouse: NSE bhavcopy, participant OI, FII/DII, GIFT Nifty")
    ss = s.add_subparsers(dest="dcmd", required=True)
    x = ss.add_parser("update", help="fetch missing days (default: the last 10) and today's snapshots")
    x.add_argument("--from", dest="start", help="YYYY-MM-DD")
    x.add_argument("--to", help="YYYY-MM-DD (default today)")
    x.add_argument("--only", help="comma-separated tables (fo_bhav, participant_oi, participant_vol, fii_dii, "
                                  "gift_nifty, corp_events, nse_holidays)")
    x.add_argument("--dir", help="warehouse folder (default runtime/warehouse)")
    x.add_argument("--release", help="GitHub release holding the files (e.g. warehouse): pull first, push after")
    x.add_argument("--gap", type=float, default=0.5, help="seconds between NSE requests")
    x.set_defaults(fn=cmd_update)
    x = ss.add_parser("archive-session", help="keep the desk's recorded chains and bars as Parquet (chains-YYYY release)")
    x.add_argument("--part", required=True, help="which job recorded it, e.g. morning-<run id>")
    x.add_argument("--data", help="recorder folder (default runtime/intraday/data)")
    x.add_argument("--day", help="only this day (YYYY-MM-DD)")
    x.add_argument("--out", default="_archive")
    x.add_argument("--release-prefix", default="chains", help="push to <prefix>-<year>; empty to only write files")
    x.set_defaults(fn=cmd_archive)
    x = ss.add_parser("status", help="coverage, gaps, and NSE's holidays vs the config")
    x.add_argument("--dir")
    x.add_argument("--release", help="pull the manifest and the small tables from this release first")
    x.add_argument("--all", action="store_true", help="with --release: pull every table (large)")
    x.set_defaults(fn=cmd_status)
