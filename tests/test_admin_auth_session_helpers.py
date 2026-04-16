from types import SimpleNamespace

import pytest

from jupr_app.ui import admin_auth
from jupr_app.ui.admin_auth import _exchange_auth_code_for_session, _set_auth_session


class _SetSessionAuth:
    def __init__(self, succeed_on: int):
        self.calls = []
        self._succeed_on = succeed_on

    def set_session(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if len(self.calls) < self._succeed_on:
            raise TypeError(f"attempt {len(self.calls)} failed")
        return {"ok": True, "attempt": len(self.calls)}


class _ExchangeCodeAuth:
    def __init__(self, succeed_on: int):
        self.calls = []
        self._succeed_on = succeed_on

    def exchange_code_for_session(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if len(self.calls) < self._succeed_on:
            raise TypeError(f"attempt {len(self.calls)} failed")
        return {"ok": True, "attempt": len(self.calls)}


class _ResetEmailAuth:
    def __init__(self):
        self.calls = []

    def reset_password_email(self, *args, **kwargs):
        self.calls.append(("reset_password_email", args, kwargs))
        if len(args) == 1 and not kwargs:
            return {"ok": True}
        raise TypeError("redirect signature rejected")


class _ResetClient:
    def __init__(self, auth):
        self.auth = auth


def test_set_auth_session_falls_back_in_order():
    auth = _SetSessionAuth(succeed_on=3)
    client = SimpleNamespace(auth=auth)

    response = _set_auth_session(client, "a-token", "r-token")

    assert response["ok"] is True
    assert auth.calls == [
        (("a-token", "r-token"), {}),
        (({"access_token": "a-token", "refresh_token": "r-token"},), {}),
        ((), {"session": {"access_token": "a-token", "refresh_token": "r-token"}}),
    ]


def test_set_auth_session_raises_last_exception_if_all_attempts_fail():
    auth = _SetSessionAuth(succeed_on=999)
    client = SimpleNamespace(auth=auth)

    with pytest.raises(TypeError, match="attempt 3 failed"):
        _set_auth_session(client, "a-token", "r-token")

    assert len(auth.calls) == 3


def test_exchange_auth_code_for_session_falls_back_in_order():
    auth = _ExchangeCodeAuth(succeed_on=3)
    client = SimpleNamespace(auth=auth)

    response = _exchange_auth_code_for_session(client, "abc123")

    assert response["ok"] is True
    assert auth.calls == [
        (({"auth_code": "abc123"},), {}),
        (("abc123",), {}),
        ((), {"code": "abc123"}),
    ]


def test_exchange_auth_code_for_session_raises_last_exception_if_all_attempts_fail():
    auth = _ExchangeCodeAuth(succeed_on=999)
    client = SimpleNamespace(auth=auth)

    with pytest.raises(TypeError, match="attempt 3 failed"):
        _exchange_auth_code_for_session(client, "abc123")

    assert len(auth.calls) == 3


def test_send_password_reset_email_falls_back_to_no_redirect(monkeypatch):
    auth = _ResetEmailAuth()
    client = _ResetClient(auth)
    monkeypatch.setattr(admin_auth, "make_supabase_auth_client", lambda: client)

    admin_auth.send_password_reset_email(
        "Admin@Example.com",
        redirect_to="https://example.com/reset",
    )

    assert auth.calls == [
        (
            "reset_password_email",
            ("admin@example.com", {"redirect_to": "https://example.com/reset"}),
            {},
        ),
        (
            "reset_password_email",
            ("admin@example.com",),
            {"options": {"redirect_to": "https://example.com/reset"}},
        ),
        ("reset_password_email", ("admin@example.com",), {}),
    ]
