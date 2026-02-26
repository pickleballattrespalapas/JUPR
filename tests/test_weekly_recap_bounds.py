from datetime import date, timedelta

import pytest

from jupr_app.domain.recaps.weekly_recap import _within_bounds, get_date_range_bounds, normalize_date_range


def test_date_range_bounds_inclusive_custom_span():
    start = date(2025, 2, 1)
    end = date(2025, 2, 3)
    start_dt, end_dt = get_date_range_bounds(start, end, "America/Mazatlan")
    assert end_dt - start_dt == timedelta(days=3) - timedelta(microseconds=1)


def test_within_bounds_is_inclusive_for_lower_and_upper_bound():
    start = date(2025, 2, 1)
    end = date(2025, 2, 3)
    assert _within_bounds(start, start, end)
    assert _within_bounds(end, start, end)


def test_normalize_date_range_rejects_inverted_dates():
    with pytest.raises(ValueError):
        normalize_date_range(date(2025, 2, 2), date(2025, 2, 1))
