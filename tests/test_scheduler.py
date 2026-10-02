"""The desk scheduler: starts the desk when it should be running and isn't, never against the kill switch."""
import datetime as dt
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("scheduler", ROOT / "deploy" / "scheduler.py")
sch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sch)
IST = sch.IST


def at(s):
    return dt.datetime.fromisoformat(s).replace(tzinfo=IST)


def run(created_ist, status="completed", conclusion="success"):
    utc = at(created_ist).astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"createdAt": utc, "status": status, "conclusion": conclusion, "event": "schedule"}


HOLS = sch.holidays(ROOT / "config" / "quantdesk.yaml")


def test_reads_the_holidays_from_the_config():
    assert dt.date(2026, 10, 2) in HOLS and dt.date(2026, 10, 20) in HOLS and dt.date(2026, 10, 1) not in HOLS
    assert dt.date(2026, 10, 7) not in HOLS                     # an RBI event day is a trading day


def test_starts_a_missing_session_and_only_in_the_window():
    yday = [run("2026-09-30 15:19")]                             # 1 Oct 2026: the cron hadn't arrived yet
    assert sch.decide(at("2026-10-01 08:40"), yday, HOLS)[0] == "start"
    assert sch.decide(at("2026-10-01 13:30"), yday, HOLS)[0] == "start"      # late is better than never
    assert sch.decide(at("2026-10-01 08:00"), yday, HOLS)[0] == "wait"       # too early: the job would time out
    assert sch.decide(at("2026-10-01 15:00"), yday, HOLS)[0] == "wait"       # too late to trade
    assert sch.decide(at("2026-10-02 10:00"), [], HOLS)[0] == "wait"         # Gandhi Jayanti
    assert sch.decide(at("2026-10-03 10:00"), [], HOLS)[0] == "wait"         # Saturday


def test_never_doubles_up_and_respects_the_kill_switch():
    t = at("2026-10-05 10:00")
    assert sch.decide(t, [run("2026-10-05 08:40", status="in_progress", conclusion=None)], HOLS)[0] == "wait"
    assert sch.decide(t, [run("2026-10-05 08:40", status="queued", conclusion=None)], HOLS)[0] == "wait"
    act, why = sch.decide(t, [run("2026-10-05 08:40", conclusion="cancelled")], HOLS)
    assert act == "wait" and "stopped on purpose" in why
    assert sch.decide(t, [run("2026-10-05 08:40", conclusion="success")], HOLS)[0] == "wait"


def test_retries_a_failed_day_a_few_times_then_stops():
    t = at("2026-10-05 11:00")
    assert sch.decide(t, [run("2026-10-05 08:40", conclusion="failure")], HOLS)[0] == "start"
    three = [run(f"2026-10-05 0{h}:40", conclusion="failure") for h in (8, 9)] + [run("2026-10-05 10:10", conclusion="failure")]
    assert sch.decide(t, three, HOLS)[0] == "wait"
