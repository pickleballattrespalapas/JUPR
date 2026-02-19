from jupr_app.data import sb_write


def test_manual_player_rating_edit_blocked_in_production():
    sb_write.PRODUCTION_MODE = True
    try:
        sb_write._enforce_rating_state_policy(
            table="players",
            payload={"rating": 1200.0},
            derived_from_match_history=False,
        )
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert "Manual rating edits are disabled in PRODUCTION_MODE" in str(exc)
        assert "Set derived_from_match_history=True" in str(exc)


def test_manual_league_rating_edit_blocked_in_production():
    sb_write.PRODUCTION_MODE = True
    try:
        sb_write._enforce_rating_state_policy(
            table="league_ratings",
            payload={"rating": 1200.0},
            derived_from_match_history=False,
        )
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert "Manual league rating edits are disabled in PRODUCTION_MODE" in str(exc)


def test_derived_rating_edit_allowed_in_production():
    sb_write.PRODUCTION_MODE = True
    sb_write._enforce_rating_state_policy(
        table="players",
        payload={"rating": 1210.0, "wins": 1},
        derived_from_match_history=True,
    )
