#!/usr/bin/env python3
"""Start the live desk when it should be running and isn't (run by scheduler.yml every few minutes).

Why: GitHub's scheduled events are best-effort. On 29 Sep – 1 Oct 2026 live.yml's 08:52 IST cron was
delivered at 15:27, 15:19 and 15:46 IST: the desk traded 11 minutes in three sessions. This check is
itself scheduled (so each firing can be late too), but it fires every few minutes all day, so some
firings land in the morning, and any one of them starting the desk is enough. It starts the desk by
workflow_dispatch, which GitHub runs at once.

Rules, on an NSE trading day between 08:25 and 14:45 IST:
  * a live.yml run is queued or running            → nothing to do
  * a run today was cancelled                      → the operator stopped the desk (the kill switch): stay stopped
  * a run today succeeded                          → the session is covered
  * a run today failed                             → start it again, at most 3 starts a day
  * no run today                                   → start it
Standard library only; reads the holiday list straight from config/quantdesk.yaml."""
from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
WINDOW = (dt.time(8, 25), dt.time(14, 45))
MAX_STARTS = 3
ACTIVE = {"queued", "in_progress", "waiting", "pending", "requested"}


def holidays(config: Path) -> set[dt.date]:
    out, inside = set(), False
    for line in config.read_text(encoding="utf-8").splitlines():
        if re.match(r"^\s*holidays:\s*$", line):
            inside = True
            continue
        if inside:
            m = re.match(r"^\s*-\s*(\d{4}-\d{2}-\d{2})", line)
            if m:
                out.add(dt.date.fromisoformat(m.group(1)))
            elif line.strip() and not line.strip().startswith("#"):
                break
    return out


def decide(now: dt.datetime, runs: list[dict], hols: set[dt.date]) -> tuple[str, str]:
    """('start' | 'wait', why). `runs`: live.yml runs as `gh run list --json status,conclusion,createdAt,event` gives them."""
    now = now.astimezone(IST)
    day = now.date()
    if day.weekday() >= 5 or day in hols:
        return "wait", f"{day} is not an NSE trading day"
    if not (WINDOW[0] <= now.time() < WINDOW[1]):
        return "wait", f"{now:%H:%M} IST is outside the start window {WINDOW[0]:%H:%M}–{WINDOW[1]:%H:%M}"
    if any(r.get("status") in ACTIVE for r in runs):
        return "wait", "the desk is already queued or running"
    today = [r for r in runs if _ist(r["createdAt"]).date() == day]
    if any(r.get("conclusion") == "cancelled" for r in today):
        return "wait", "a run today was cancelled: the desk was stopped on purpose, so it stays stopped"
    if any(r.get("conclusion") == "success" for r in today):
        return "wait", "today's session is already covered"
    if len(today) >= MAX_STARTS:
        return "wait", f"{len(today)} runs today and none succeeded: not starting again (look at the logs)"
    return "start", "no run today yet" if not today else f"today's run failed ({len(today)} so far): starting again"


def _ist(ts: str) -> dt.datetime:
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(IST)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    hols = holidays(repo_root / "config" / "quantdesk.yaml")
    runs = json.loads(subprocess.run(["gh", "run", "list", "--workflow", "live.yml", "--limit", "30", "--json",
                                      "status,conclusion,createdAt,event"], check=True, capture_output=True,
                                     text=True).stdout or "[]")
    action, why = decide(dt.datetime.now(IST), runs, hols)
    print(f"{action}: {why}")
    if action == "start":
        subprocess.run(["gh", "workflow", "run", "live.yml", "--ref", "main"], check=True)
        print("dispatched live.yml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
