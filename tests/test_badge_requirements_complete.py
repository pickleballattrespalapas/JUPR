from jupr_app.domain.gamification.badge_catalog import BADGE_DEFINITIONS
from jupr_app.domain.gamification.requirements import requirement_for


def test_all_badges_have_requirement_strings():
    missing = []
    for badge in BADGE_DEFINITIONS:
        req = requirement_for(badge.badge_id)
        if req is None or not req.strip() or "requirements tbd" in req.lower():
            missing.append(badge.badge_id)
    assert not missing, f"Missing requirement strings for badges: {', '.join(sorted(missing))}"
