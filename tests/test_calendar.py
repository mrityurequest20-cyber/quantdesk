import datetime as dt

from quantdesk.core.calendar import TradingCalendar, year_fraction


def test_trading_days_skip_weekends_and_holidays(cfg):
    cal = TradingCalendar(cfg.holidays())
    assert not cal.is_trading_day(dt.date(2026, 1, 26))        # Republic Day (Mon)
    assert not cal.is_trading_day(dt.date(2026, 9, 27))        # Sunday
    assert cal.next_trading_day(dt.date(2026, 1, 23)) == dt.date(2026, 1, 27)
    assert cal.prev_trading_day(dt.date(2026, 1, 27)) == dt.date(2026, 1, 23)


def test_weekly_expiry_is_tuesday_and_shifts_before_holidays(cfg):
    cal = TradingCalendar(cfg.holidays())
    exps = cal.expiries(dt.date(2026, 2, 20), 30)
    assert dt.date(2026, 2, 24) in exps                         # plain Tuesday
    assert dt.date(2026, 3, 2) in exps                          # Tue 3-Mar is Holi → Mon 2-Mar
    assert dt.date(2026, 3, 3) not in exps
    assert all(e.weekday() in (0, 1) for e in exps)


def test_monthly_expiry_last_tuesday(cfg):
    cal = TradingCalendar(cfg.holidays())
    assert cal.monthly_expiry(2026, 10) == dt.date(2026, 10, 27)
    assert cal.monthly_expiry(2026, 3) == dt.date(2026, 3, 30)  # last Tue 31-Mar is a holiday
    monthly_only = cal.expiries(dt.date(2026, 9, 1), 70, weekly=False)
    assert monthly_only == [dt.date(2026, 9, 29), dt.date(2026, 10, 27)]


def test_pick_expiry_respects_min_dte(cfg):
    cal = TradingCalendar(cfg.holidays())
    e = cal.pick_expiry(dt.date(2026, 9, 28), target_dte=14, min_dte=6)
    assert (e - dt.date(2026, 9, 28)).days >= 6
    assert e == dt.date(2026, 10, 13)


def test_year_fraction():
    assert year_fraction(dt.date(2026, 1, 1), dt.date(2026, 1, 1)) == 0
    assert abs(year_fraction(dt.date(2026, 1, 1), dt.date(2027, 1, 1)) - 1) < 1e-9
