from __future__ import annotations

import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts import initialize_registration_edit_secret as setup
from jupr_app import config
from jupr_app.domain.tournament_registration_confirmation_tokens import (
    build_registration_confirmation_token,
    verify_registration_confirmation_token,
)


@pytest.fixture
def production(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "production")
    monkeypatch.setenv("FLY_APP_NAME", setup.APP)
    monkeypatch.setenv("SUPABASE_URL", setup.PROJECT_URL)
    monkeypatch.setenv("FLY_API_TOKEN", "fake-deploy-token")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "fake-server-key")
    monkeypatch.delenv(setup.EDIT_KEY, raising=False)
    monkeypatch.delenv(setup.CONFIRMATION_KEY, raising=False)
    monkeypatch.setattr(config, "_streamlit_secret_value", lambda *_: "")
    return {
        "edit_ready": False, "edit_env_present": False, "confirmation_env_present": False,
        "confirmation_fingerprint": hashlib.sha256(config.get_registration_confirmation_token_secret().encode()).hexdigest(),
    }


def test_initialization_preserves_outstanding_confirmation_tokens(production, monkeypatch):
    old_signer = config.get_registration_confirmation_token_secret()
    token = build_registration_confirmation_token(
        tournament_id="fixture-tournament", registration_id="fixture-registration",
        email="participant@example.test", now=1000,
    )
    updates = setup.prepare_missing_keys({"SUPABASE_SERVICE_ROLE_KEY"}, production, "fake-server-key")
    for name, value in updates.items():
        monkeypatch.setenv(name, value)
    assert config.get_registration_confirmation_token_secret() == old_signer
    assert config.get_explicit_registration_edit_token_secret() != old_signer
    assert len(config.get_explicit_registration_edit_token_secret()) >= 32
    assert verify_registration_confirmation_token(
        token, expected_tournament_id="fixture-tournament",
        expected_registration_id="fixture-registration",
        expected_email="participant@example.test", now=1100,
    )["registration_id"] == "fixture-registration"


def test_existing_confirmation_key_is_never_replaced(production):
    production["confirmation_env_present"] = True
    updates = setup.prepare_missing_keys({setup.CONFIRMATION_KEY}, production, "")
    assert set(updates) == {setup.EDIT_KEY}


@pytest.mark.parametrize("existing", ["inventory", "environment", "config"])
def test_existing_edit_keys_are_never_rotated(production, existing):
    names = {setup.EDIT_KEY} if existing == "inventory" else set()
    if existing == "environment":
        production["edit_env_present"] = True
    if existing == "config":
        production["edit_ready"] = True
    with pytest.raises(setup.SetupError, match="never replaces"):
        setup.prepare_missing_keys(names, production, "fake-server-key")


def test_unrecognized_confirmation_fallback_stops_without_generating_key(production, monkeypatch):
    monkeypatch.setattr(setup.secrets, "token_urlsafe", lambda *_: pytest.fail("must stop before generating"))
    with pytest.raises(setup.SetupError, match="differs"):
        setup.prepare_missing_keys(set(), production, "different-server-key")


def test_default_check_cannot_mutate(production, monkeypatch):
    monkeypatch.setattr(setup, "inventory", lambda: {"SUPABASE_SERVICE_ROLE_KEY"})
    monkeypatch.setattr(setup, "probe", lambda: production)
    monkeypatch.setattr(setup, "fly", lambda *_a, **_k: pytest.fail("no import permitted"))
    with pytest.raises(setup.SetupError, match="owner approval"):
        setup.run()


@pytest.mark.parametrize("name,value", [("JUPR_ENV", "staging"), ("FLY_APP_NAME", "juprleagues-api-staging"), ("SUPABASE_URL", "https://wrong.supabase.co")])
def test_wrong_environment_stops_before_access(production, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    monkeypatch.setattr(setup, "fly", lambda *_a, **_k: pytest.fail("no remote access"))
    with pytest.raises(setup.SetupError, match="exact production"):
        setup.run(initialize_missing=True)


def test_configuration_change_stops_before_import(production, monkeypatch):
    names = iter([{"SMTP_HOST"}, {"SMTP_HOST", setup.EDIT_KEY}])
    monkeypatch.setattr(setup, "inventory", lambda: next(names))
    monkeypatch.setattr(setup, "probe", lambda: production)
    monkeypatch.setattr(setup, "fly", lambda *_a, **_k: pytest.fail("configuration changed"))
    with pytest.raises(setup.SetupError, match="changed during"):
        setup.run(initialize_missing=True)


def test_import_is_single_stdin_operation_and_rerun_does_not_rotate(production, monkeypatch):
    current = dict(production)
    names = {"SUPABASE_SERVICE_ROLE_KEY"}
    imports = []
    monkeypatch.setattr(setup, "inventory", lambda: set(names))
    monkeypatch.setattr(setup, "probe", lambda: dict(current))
    def fly(*args, payload=None, **kwargs):
        assert args == ("secrets", "import")
        values = dict(line.split("=", 1) for line in payload.splitlines())
        assert set(values) == {setup.EDIT_KEY, setup.CONFIRMATION_KEY}
        imports.append(values)
        names.update(values)
        current.update(edit_ready=True, edit_env_present=True, confirmation_env_present=True)
        return ""
    monkeypatch.setattr(setup, "fly", fly)
    assert setup.run(initialize_missing=True) == {"registration_edit_ready": True, "confirmation_signer_preserved": True, "changed": True}
    assert setup.run(initialize_missing=True) == {"registration_edit_ready": True, "changed": False}
    assert len(imports) == 1


@pytest.mark.parametrize("failure", ["exit", "timeout"])
def test_provider_failure_never_exposes_secret_output(monkeypatch, capsys, failure):
    secret = "fake-secret-must-not-appear"
    def fail(*args, **kwargs):
        assert secret not in repr(args)
        assert kwargs["input"] == secret
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args[0], 300, output=secret, stderr=secret)
        return SimpleNamespace(returncode=1, stdout=secret, stderr=secret)
    monkeypatch.setattr(setup.subprocess, "run", fail)
    with pytest.raises(setup.SetupError) as caught:
        setup.fly("secrets", "import", payload=secret)
    assert secret not in str(caught.value)
    assert secret not in capsys.readouterr().out + capsys.readouterr().err


@pytest.mark.parametrize("status", ["Staged", "Partial", "Unknown", None])
def test_pending_or_unknown_secrets_block_initialization(monkeypatch, status):
    monkeypatch.setattr(setup, "fly", lambda *_: json.dumps([{"name": "SMTP_HOST", "digest": "fake-digest", "status": status}]))
    with pytest.raises(setup.SetupError, match="fully deployed"):
        setup.inventory()


def test_pinned_flyctl_inventory_accepts_fully_deployed_secrets(monkeypatch):
    # Matches SecretWithStatus JSON tags in flyctl v0.4.49 secrets/list.go.
    monkeypatch.setattr(setup, "fly", lambda *_: json.dumps([
        {"name": "SMTP_HOST", "digest": "fake-digest", "status": "Deployed"},
        {"name": "SUPABASE_SERVICE_ROLE_KEY", "digest": "another-digest", "status": "Deployed"},
    ]))
    assert setup.inventory() == {"SMTP_HOST", "SUPABASE_SERVICE_ROLE_KEY"}
