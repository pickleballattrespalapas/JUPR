from __future__ import annotations

import pytest

from jupr_app.domain.tournament_registration_confirmation_tokens import (
    build_registration_confirmation_token,
    verify_registration_confirmation_token,
)


SECRET = "confirmation-test-secret"


def test_confirmation_token_round_trip_and_domain_binding() -> None:
    token = build_registration_confirmation_token(
        tournament_id="t1",
        registration_id="reg_1",
        email="Ada@Example.com",
        now=100,
        secret=SECRET,
    )

    payload = verify_registration_confirmation_token(
        token,
        expected_tournament_id="t1",
        expected_registration_id="reg_1",
        expected_email="ada@example.com",
        now=101,
        secret=SECRET,
    )

    assert payload["purpose"] == "tournament_registration_confirmation"
    assert payload["registration_id"] == "reg_1"


@pytest.mark.parametrize(
    "change",
    ["tampered", "wrong_tournament", "wrong_registration", "wrong_email", "expired"],
)
def test_confirmation_token_rejects_invalid_access(change: str) -> None:
    token = build_registration_confirmation_token(
        tournament_id="t1",
        registration_id="reg_1",
        email="ada@example.com",
        expires_in_seconds=5,
        now=100,
        secret=SECRET,
    )
    if change == "tampered":
        token = f"{token}x"

    with pytest.raises(ValueError):
        verify_registration_confirmation_token(
            token,
            expected_tournament_id="t2" if change == "wrong_tournament" else "t1",
            expected_registration_id="reg_2" if change == "wrong_registration" else "reg_1",
            expected_email="grace@example.com" if change == "wrong_email" else "ada@example.com",
            now=106 if change == "expired" else 101,
            secret=SECRET,
        )


def test_confirmation_secret_never_falls_back_to_publishable_key(monkeypatch) -> None:
    from jupr_app import config

    for name in (
        "JUPR_REGISTRATION_CONFIRMATION_SECRET",
        "JUPR_REGISTRATION_EDIT_SECRET",
        "SUPABASE_SERVICE_ROLE_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SUPABASE_ANON_KEY", "public-browser-key")
    monkeypatch.setattr(config, "_streamlit_secret_value", lambda *_args: "")

    with pytest.raises(ValueError, match="CONFIRMATION_SECRET"):
        config.get_registration_confirmation_token_secret()
