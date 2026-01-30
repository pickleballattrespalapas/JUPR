from jupr_app.domain.gamification.requirements import load_requirements_map, requirement_for


def test_requirements_loader_has_known_badges():
    requirements = load_requirements_map()
    expected = [
        "participant",
        "first_win",
        "weekly_regular",
        "iron_week",
        "marathon_month",
    ]
    for badge_id in expected:
        value = requirements.get(badge_id)
        assert value is not None
        assert value.strip()
        assert value != "Requirements TBD"


def test_requirement_for_missing_badge_returns_fallback():
    assert requirement_for("no_such_badge") == "Requirements TBD"
