#!/usr/bin/env python3
"""Fetch one sample of every NSE source the data warehouse reads and print what came back (status, type, size,
the first lines / keys), so the parsers are written against NSE's real formats. Runs on a GitHub runner
(data-probe.yml): NSE isn't reachable from every network."""
from __future__ import annotations

import io
import json
import sys
import time
import zipfile

import requests

UA = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/130.0.0.0 Safari/537.36",
      "accept-language": "en-US,en;q=0.9", "accept": "*/*", "referer": "https://www.nseindia.com/"}
A = "https://nsearchives.nseindia.com"
URLS = [
    ("bhav udiff 2026-10-01", f"{A}/content/fo/BhavCopy_NSE_FO_0_0_0_20261001_F_0000.csv.zip"),
    ("bhav udiff 2024-07-08", f"{A}/content/fo/BhavCopy_NSE_FO_0_0_0_20240708_F_0000.csv.zip"),
    ("bhav udiff 2023-10-03", f"{A}/content/fo/BhavCopy_NSE_FO_0_0_0_20231003_F_0000.csv.zip"),
    ("bhav old 2024-07-05", f"{A}/content/historical/DERIVATIVES/2024/JUL/fo05JUL2024bhav.csv.zip"),
    ("bhav old 2022-10-03", f"{A}/content/historical/DERIVATIVES/2022/OCT/fo03OCT2022bhav.csv.zip"),
    ("participant oi 2026-10-01", f"{A}/content/nsccl/fao_participant_oi_01102026.csv"),
    ("participant oi 2021-10-01", f"{A}/content/nsccl/fao_participant_oi_01102021.csv"),
    ("participant vol 2026-10-01", f"{A}/content/nsccl/fao_participant_vol_01102026.csv"),
    ("fii stats 2026-10-01", f"{A}/content/fo/fii_stats_01-Oct-2026.xls"),
    ("cm bhav udiff 2026-10-01", f"{A}/content/cm/BhavCopy_NSE_CM_0_0_0_20261001_F_0000.csv.zip"),
]
API = [
    ("fii/dii cash", "https://www.nseindia.com/api/fiidiiTradeReact"),
    ("market status (GIFT Nifty)", "https://www.nseindia.com/api/marketStatus"),
    ("event calendar", "https://www.nseindia.com/api/event-calendar"),
    ("holidays", "https://www.nseindia.com/api/holiday-master?type=trading"),
    ("india vix history", "https://www.nseindia.com/api/historical/vixhistory?from=01-09-2026&to=01-10-2026"),
]


def show(label: str, r: requests.Response) -> None:
    ct = r.headers.get("content-type", "")
    print(f"\n=== {label}: HTTP {r.status_code} · {ct} · {len(r.content):,} bytes · {r.url}")
    body = r.content
    if body[:2] == b"PK":
        z = zipfile.ZipFile(io.BytesIO(body))
        for n in z.namelist():
            text = z.read(n).decode("utf-8", "replace")
            lines = text.splitlines()
            print(f"  zip member {n}: {len(lines):,} lines")
            for ln in lines[:4]:
                print("   ", ln[:400])
            idx = [ln for ln in lines if ",NIFTY," in ln or ",NIFTY " in ln][:3]
            for ln in idx:
                print("    NIFTY>", ln[:400])
        return
    if b"json" in ct.encode() or body[:1] in (b"{", b"["):
        try:
            d = json.loads(body)
            if isinstance(d, dict):
                print("  keys:", list(d)[:30])
                for k, v in list(d.items())[:8]:
                    print(f"  {k}: {json.dumps(v)[:600]}")
            else:
                print(f"  list of {len(d)}; first:", json.dumps(d[:2])[:1200])
            return
        except ValueError:
            pass
    if body[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        print("  (legacy .xls binary)")
        return
    text = body.decode("utf-8", "replace")
    for ln in text.splitlines()[:8]:
        print("   ", ln[:400])


def main() -> int:
    s = requests.Session()
    s.headers.update(UA)
    for label, url in URLS:
        try:
            show(label, s.get(url, timeout=30))
        except Exception as exc:
            print(f"\n=== {label}: FAIL {exc!r}")
        time.sleep(1)
    try:
        s.get("https://www.nseindia.com/", timeout=20)
        s.get("https://www.nseindia.com/option-chain", timeout=20)
    except Exception as exc:
        print("warm-up failed:", exc)
    for label, url in API:
        try:
            show(label, s.get(url, timeout=30))
        except Exception as exc:
            print(f"\n=== {label}: FAIL {exc!r}")
        time.sleep(1.5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
