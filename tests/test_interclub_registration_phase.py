from datetime import datetime, timezone

import pytest

from jupr_app.services import interclub_registration_phase as phase


@pytest.mark.parametrize("instant,status", [
    ("2026-09-21T13:59:59+00:00", "scheduled"),
    ("2026-09-21T14:00:00+00:00", "open"),
    ("2026-09-22T13:59:59+00:00", "open"),
    ("2026-09-22T14:00:00+00:00", "closed"),
])
def test_shared_registration_window_uses_inclusive_open_exclusive_close(instant, status):
    season = {"registration_opens_at": "2026-09-21T07:00:00-07:00", "registration_closes_at": "2026-09-22T14:00:00Z", "registration_revision": 7}
    result = phase.registration_state(season, now=datetime.fromisoformat(instant))
    assert result == {"opens_at": "2026-09-21T14:00:00+00:00", "closes_at": "2026-09-22T14:00:00+00:00", "revision": 7,
                      "status": status, "can_register": status == "open", "meet_planning_open": status == "closed"}


@pytest.mark.parametrize("opens,closes", [
    (None, None), (None, "2026-09-22T14:00:00Z"), ("2026-09-21T14:00:00Z", None),
    ("invalid", "2026-09-22T14:00:00Z"), ("2026-09-21T14:00:00", "2026-09-22T14:00:00Z"),
    ("2026-09-22T14:00:00Z", "2026-09-22T14:00:00Z"), ("2026-09-23T14:00:00Z", "2026-09-22T14:00:00Z"),
])
def test_missing_or_invalid_dates_never_invent_registration_or_meet_access(opens, closes):
    season = {"registration_opens_at": opens, "registration_closes_at": closes}
    result = phase.registration_state(season, now=datetime(2027, 1, 1, tzinfo=timezone.utc))
    assert result["status"] == "unconfigured"
    assert result["revision"] == 0
    assert not result["can_register"] and not result["meet_planning_open"]
    for intake in (False, True):
        with pytest.raises(phase.RegistrationPhaseError, match="commissioner must set"):
            phase.require_registration_phase(season, intake=intake)


def test_serialization_and_checks_preserve_existing_season_and_membership_data():
    season = {"id": "season", "details": {"name": "Coastal League"}, "registration_opens_at": "2000-01-01T00:00:00Z",
              "registration_closes_at": "2001-01-01T00:00:00Z", "registration_revision": 2}
    original = dict(season)
    assert phase.require_registration_phase(season)["meet_planning_open"]
    with pytest.raises(phase.RegistrationPhaseError, match="registration is not open"):
        phase.require_registration_phase(season, intake=True)
    serialized = phase.registration_season(season)
    assert serialized["registration"]["status"] == "closed"
    assert season == original and "registration" not in season


def test_future_or_current_registration_blocks_meet_planning():
    for opens in ("2000-01-01T00:00:00Z", "2998-01-01T00:00:00Z"):
        season = {"registration_opens_at": opens, "registration_closes_at": "2999-01-01T00:00:00Z"}
        with pytest.raises(phase.RegistrationPhaseError, match="after season registration closes"):
            phase.require_registration_phase(season)
