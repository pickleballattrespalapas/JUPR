from jupr_app.data.sb_write import _enforce_rating_state_policy


def test_manual_player_rating_edit_blocked_in_production():
    try:
        _enforce_rating_state_policy(
            table="players",
            payload={"rating": 1200.0},
            derived_from_match_history=False,
        )
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert "Manual rating edits are disabled in PRODUCTION_MODE" in str(exc)


def test_manual_league_rating_edit_blocked_in_production():
    try:
        _enforce_rating_state_policy(
            table="league_ratings",
            payload={"rating": 1200.0},
            derived_from_match_history=False,
        )
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert "Manual league rating edits are disabled in PRODUCTION_MODE" in str(exc)


def test_derived_rating_edit_allowed_in_production():
    _enforce_rating_state_policy(
        table="players",
        payload={"rating": 1210.0, "wins": 1},
        derived_from_match_history=True,
    )
