from jupr_app.domain.gamification.badge_catalog import BADGE_DEFINITIONS
from jupr_app.domain.gamification.requirements import (
    clear_requirements_cache,
    requirement_for,
)


def test_badge_requirements_are_complete() -> None:
    clear_requirements_cache()
    missing = []
    for badge in BADGE_DEFINITIONS:
        req = requirement_for(badge.badge_id)
        cleaned = str(req or "").strip()
        if not cleaned or "requirements tbd" in cleaned.lower():
            missing.append(badge.badge_id)
    assert missing == [], (
        "Expected all badge requirements to be resolved; missing or TBD for: "
        f"{', '.join(missing)}"
    )
