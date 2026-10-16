"""Explicit phase data for synthetic interclub seasons used by contract tests."""


def set_registration_phase(season, phase):
    dates = {
        "unconfigured": (None, None),
        "scheduled": ("2098-01-01T00:00:00Z", "2099-01-01T00:00:00Z"),
        "open": ("2020-01-01T00:00:00Z", "2099-01-01T00:00:00Z"),
        "closed": ("2020-01-01T00:00:00Z", "2021-01-01T00:00:00Z"),
    }
    opens, closes = dates[phase]
    season.update(registration_opens_at=opens, registration_closes_at=closes,
                  registration_revision=0 if phase == "unconfigured" else 1)
    return season
