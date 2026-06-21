import pytest

from jupr_app.domain.tournament_registration_edit_tokens import build_registration_edit_token, verify_registration_edit_token

SECRET = "test-secret"


def test_token_verifies_for_matching_registration():
    token = build_registration_edit_token(tournament_id="t1", registration_id="r1", email="ADA@Example.com", now=100, secret=SECRET)
    payload = verify_registration_edit_token(token, expected_tournament_id="t1", expected_registration_id="r1", expected_email="ada@example.com", now=101, secret=SECRET)
    assert payload["tournament_id"] == "t1"
    assert payload["registration_id"] == "r1"
    assert "ada@example.com" not in token


def test_token_rejects_wrong_email():
    token = build_registration_edit_token(tournament_id="t1", registration_id="r1", email="ada@example.com", now=100, secret=SECRET)
    with pytest.raises(ValueError):
        verify_registration_edit_token(token, expected_email="grace@example.com", now=101, secret=SECRET)


def test_token_rejects_wrong_tournament_id():
    token = build_registration_edit_token(tournament_id="t1", registration_id="r1", email="ada@example.com", now=100, secret=SECRET)
    with pytest.raises(ValueError):
        verify_registration_edit_token(token, expected_tournament_id="t2", now=101, secret=SECRET)


def test_token_rejects_expired_token():
    token = build_registration_edit_token(tournament_id="t1", registration_id="r1", email="ada@example.com", expires_in_seconds=5, now=100, secret=SECRET)
    with pytest.raises(ValueError):
        verify_registration_edit_token(token, now=106, secret=SECRET)


def test_token_rejects_tampered_signature():
    token = build_registration_edit_token(tournament_id="t1", registration_id="r1", email="ada@example.com", now=100, secret=SECRET)
    with pytest.raises(ValueError):
        verify_registration_edit_token(token + "x", now=101, secret=SECRET)
