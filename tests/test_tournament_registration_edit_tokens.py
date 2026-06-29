import importlib

import pytest

from jupr_app.config import get_registration_edit_token_secret
from jupr_app.domain.tournament_registration_edit_tokens import build_registration_edit_token, verify_registration_edit_token

SECRET = "test-secret"


def _clear_implicit_token_secret_env(monkeypatch):
    for key in [
        "JUPR_REGISTRATION_EDIT_SECRET",
        "SUPABASE_SERVICE_ROLE_KEY",
        "SUPABASE_ANON_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)


def test_edit_token_module_imports_without_env_secret(monkeypatch):
    _clear_implicit_token_secret_env(monkeypatch)

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


def test_build_token_without_explicit_secret_raises_clear_error_when_config_missing(monkeypatch):
    _clear_implicit_token_secret_env(monkeypatch)

    with pytest.raises(ValueError, match="JUPR_REGISTRATION_EDIT_SECRET is required"):
        build_registration_edit_token(tournament_id="t1", registration_id="r1", email="ada@example.com", now=100)


def test_get_registration_edit_token_secret_reads_env(monkeypatch):
    monkeypatch.setenv("JUPR_REGISTRATION_EDIT_SECRET", "env-secret")

    assert get_registration_edit_token_secret() == "env-secret"


def test_get_registration_edit_token_secret_can_fall_back_to_supabase_service_role(monkeypatch):
    monkeypatch.delenv("JUPR_REGISTRATION_EDIT_SECRET", raising=False)
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role-secret")

    assert get_registration_edit_token_secret() == "registration-edit-token:service-role-secret"


def test_build_and_verify_token_can_use_supabase_fallback_secret(monkeypatch):
    monkeypatch.delenv("JUPR_REGISTRATION_EDIT_SECRET", raising=False)
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role-secret")

    token = build_registration_edit_token(tournament_id="t1", registration_id="r1", email="ada@example.com", now=100)
    payload = verify_registration_edit_token(token, expected_tournament_id="t1", expected_registration_id="r1", expected_email="ADA@example.com", now=101)

    assert payload["registration_id"] == "r1"


def test_get_registration_edit_token_secret_raises_clear_error_when_missing(monkeypatch):
    _clear_implicit_token_secret_env(monkeypatch)

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
    _clear_implicit_token_secret_env(monkeypatch)

    module = importlib.import_module("jupr_app.ui.pages.tournament_registration_edit")

    assert callable(module.render)
