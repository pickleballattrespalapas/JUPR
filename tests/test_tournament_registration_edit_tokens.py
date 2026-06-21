import importlib

import pytest

from jupr_app.config import get_registration_edit_token_secret
from jupr_app.domain.tournament_registration_edit_tokens import build_registration_edit_token, verify_registration_edit_token

SECRET = "test-secret"


def test_edit_token_module_imports_without_env_secret(monkeypatch):
    monkeypatch.delenv("JUPR_REGISTRATION_EDIT_SECRET", raising=False)

    module = importlib.import_module("jupr_app.domain.tournament_registration_edit_tokens")

    assert module.TOKEN_VERSION == "v1"


def test_build_token_works_with_explicit_secret():
    token = build_registration_edit_token(tournament_id="t1", registration_id="r1", email="ada@example.com", now=100, secret=SECRET)

    assert isinstance(token, str)
    assert token.count(".") == 1
    assert "ada@example.com" not in token


def test_verify_token_works_with_explicit_secret():
    token = build_registration_edit_token(tournament_id="t1", registration_id="r1", email="ADA@Example.com", now=100, secret=SECRET)

    payload = verify_registration_edit_token(token, expected_tournament_id="t1", expected_registration_id="r1", expected_email="ada@example.com", now=101, secret=SECRET)

    assert payload["tournament_id"] == "t1"
    assert payload["registration_id"] == "r1"


def test_build_token_without_explicit_secret_raises_clear_error_when_env_missing(monkeypatch):
    monkeypatch.delenv("JUPR_REGISTRATION_EDIT_SECRET", raising=False)

    with pytest.raises(ValueError, match="JUPR_REGISTRATION_EDIT_SECRET is required"):
        build_registration_edit_token(tournament_id="t1", registration_id="r1", email="ada@example.com", now=100)


def test_get_registration_edit_token_secret_reads_env(monkeypatch):
    monkeypatch.setenv("JUPR_REGISTRATION_EDIT_SECRET", "env-secret")

    assert get_registration_edit_token_secret() == "env-secret"


def test_get_registration_edit_token_secret_raises_clear_error_when_missing(monkeypatch):
    monkeypatch.delenv("JUPR_REGISTRATION_EDIT_SECRET", raising=False)

    with pytest.raises(ValueError, match="JUPR_REGISTRATION_EDIT_SECRET is required"):
        get_registration_edit_token_secret()


def test_token_rejects_wrong_email():
    token = build_registration_edit_token(tournament_id="t1", registration_id="r1", email="ada@example.com", now=100, secret=SECRET)
    with pytest.raises(ValueError):
        verify_registration_edit_token(token, expected_email="grace@example.com", now=101, secret=SECRET)


def test_token_rejects_wrong_tournament_id():
    token = build_registration_edit_token(tournament_id="t1", registration_id="r1", email="ada@example.com", now=100, secret=SECRET)
    with pytest.raises(ValueError):
        verify_registration_edit_token(token, expected_tournament_id="t2", now=101, secret=SECRET)


def test_token_rejects_expired_link():
    token = build_registration_edit_token(tournament_id="t1", registration_id="r1", email="ada@example.com", expires_in_seconds=5, now=100, secret=SECRET)
    with pytest.raises(ValueError, match="expired"):
        verify_registration_edit_token(token, now=106, secret=SECRET)


def test_token_rejects_tampering():
    token = build_registration_edit_token(tournament_id="t1", registration_id="r1", email="ada@example.com", now=100, secret=SECRET)
    with pytest.raises(ValueError):
        verify_registration_edit_token(token + "x", now=101, secret=SECRET)


def test_tournament_registration_edit_page_imports_without_env_secret(monkeypatch):
    monkeypatch.delenv("JUPR_REGISTRATION_EDIT_SECRET", raising=False)

    module = importlib.import_module("jupr_app.ui.pages.tournament_registration_edit")

    assert callable(module.render)
