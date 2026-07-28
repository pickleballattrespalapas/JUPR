import pytest

from jupr_app.domain.tournament_team_invitation_tokens import (
    build_tournament_team_invitation_token,
    verify_tournament_team_invitation_token,
)

SECRET = "a-dedicated-test-secret-that-is-at-least-32-bytes"


def _token(**overrides):
    values = {
        "tournament_id": "tournament-1",
        "team_id": "team-1",
        "member_id": "member-1",
        "invited_email": "player@example.com",
        "invitation_version": 2,
        "expires_in_seconds": 60,
        "now": 100,
        "secret": SECRET,
    }
    values.update(overrides)
    return build_tournament_team_invitation_token(**values)


def test_token_is_bound_to_team_email_member_and_version():
    token = _token()
    verified = verify_tournament_team_invitation_token(
        token,
        expected_tournament_id="tournament-1",
        expected_team_id="team-1",
        expected_member_id="member-1",
        expected_invited_email="PLAYER@example.com",
        expected_invitation_version=2,
        now=120,
        secret=SECRET,
    )
    assert verified["team_id"] == "team-1"
    assert "player@example.com" not in str(verified)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expected_team_id", "team-2"),
        ("expected_member_id", "member-2"),
        ("expected_invited_email", "other@example.com"),
        ("expected_invitation_version", 3),
    ],
)
def test_wrong_binding_is_rejected(field, value):
    with pytest.raises(ValueError):
        verify_tournament_team_invitation_token(
            _token(),
            **{field: value},
            now=120,
            secret=SECRET,
        )


def test_tampering_and_expiry_are_rejected():
    token = _token()
    with pytest.raises(ValueError, match="Invalid"):
        verify_tournament_team_invitation_token(
            f"{token[:-1]}x",
            now=120,
            secret=SECRET,
        )
    with pytest.raises(ValueError, match="expired"):
        verify_tournament_team_invitation_token(
            token,
            now=161,
            secret=SECRET,
        )


def test_weak_dedicated_secret_is_rejected():
    with pytest.raises(ValueError, match="dedicated secret"):
        _token(secret="too-short")


def test_existing_registration_edit_secret_is_final_fallback(monkeypatch):
    monkeypatch.delenv(
        "JUPR_TOURNAMENT_TEAM_INVITATION_SECRET",
        raising=False,
    )
    monkeypatch.setenv(
        "JUPR_REGISTRATION_EDIT_SECRET",
        "reviewed-registration-secret-at-least-32-bytes",
    )
    token = _token(secret=None)

    assert verify_tournament_team_invitation_token(
        token,
        expected_team_id="team-1",
        now=120,
        secret=None,
    )["team_id"] == "team-1"


def test_short_configured_fallback_secret_is_rejected(monkeypatch):
    monkeypatch.delenv(
        "JUPR_TOURNAMENT_TEAM_INVITATION_SECRET",
        raising=False,
    )
    monkeypatch.setenv("JUPR_REGISTRATION_EDIT_SECRET", "too-short")

    with pytest.raises(ValueError, match="JUPR_REGISTRATION_EDIT_SECRET"):
        _token(secret=None)
