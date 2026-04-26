from jupr_app.ui.public_links import (
    build_public_params,
    build_route_params,
    redact_query_params,
)


def test_build_public_params_skips_none_and_preserves_common_route_keys():
    params = build_public_params(
        "players",
        {
            "pid": "42",
            "league": "OVERALL",
            "club_id": "tres_palapas",
            "tournament_id": "100",
            "event_id": "200",
            "player": None,
        },
    )

    assert params == {
        "page": "players",
        "public": "1",
        "pid": "42",
        "league": "OVERALL",
        "club_id": "tres_palapas",
        "tournament_id": "100",
        "event_id": "200",
    }


def test_build_route_params_public_mode_never_includes_admin_flag():
    params = build_route_params(
        page="leaderboards",
        params={"league": "OVERALL", "admin": "1", "pid": None},
        public_mode=True,
    )

    assert params == {
        "page": "leaderboards",
        "league": "OVERALL",
        "public": "1",
    }
    assert "admin" not in params


def test_build_route_params_admin_mode_never_includes_public_flag():
    params = build_route_params(
        page="tournament_manager",
        params={"tournament_id": "abc", "public": "1", "club_id": "tres_palapas"},
        public_mode=False,
    )

    assert params == {
        "page": "tournament_manager",
        "tournament_id": "abc",
        "club_id": "tres_palapas",
        "admin": "1",
    }
    assert "public" not in params


def test_redact_query_params_preserves_safe_keys_and_redacts_sensitive_keys():
    redacted = redact_query_params(
        {
            "page": "players",
            "pid": "123",
            "league": "OVERALL",
            "tournament_id": "abc",
            "access_token": "secret-value",
            "refresh_token": "refresh-secret",
            "password": "pw",
            "code": "otp",
        }
    )

    assert redacted["page"] == "players"
    assert redacted["pid"] == "123"
    assert redacted["league"] == "OVERALL"
    assert redacted["tournament_id"] == "abc"
    assert redacted["access_token"] == "[REDACTED]"
    assert redacted["refresh_token"] == "[REDACTED]"
    assert redacted["password"] == "[REDACTED]"
    assert redacted["code"] == "[REDACTED]"
