from datetime import date, timedelta

from jupr_app.domain.recaps.weekly_recap import get_week_bounds


def test_week_bounds_seven_days():
    start = date(2025, 1, 6)
    start_dt, end_dt = get_week_bounds(start, "America/Mazatlan")
    assert end_dt - start_dt == timedelta(days=7) - timedelta(microseconds=1)
