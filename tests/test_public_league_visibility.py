from jupr_app.services.public_league_visibility import public_league_view


def test_public_league_visibility_is_fail_closed_by_lifecycle() -> None:
    cases = [
        ({"league_name": "Active", "status": "active", "is_active": True}, "active"),
        ({"league_name": "Ended", "status": "ended", "is_active": False}, "past"),
        ({"league_name": "Draft", "status": "draft", "is_active": False}, None),
        ({"league_name": "Draft mismatch", "status": "draft", "is_active": True}, None),
        ({"league_name": "Paused", "status": "paused", "is_active": False}, None),
        ({"league_name": "Archived", "status": "archived", "is_active": False}, None),
        ({"league_name": "Legacy published", "status": "published", "is_active": True}, None),
        ({"league_name": "Missing lifecycle", "is_active": True}, None),
    ]

    for row, expected in cases:
        assert public_league_view(row) == expected
