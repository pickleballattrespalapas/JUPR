from __future__ import annotations

import pytest

from jupr_app.ui.admin_auth import _exchange_auth_code_for_session, _set_auth_session
from jupr_app.ui.pages.reset_password import _should_wait_for_probe


class _FakeAuth:
    def __init__(self, failures_before_success: int = 0):
        self.failures_before_success = failures_before_success
        self.calls = []

    def set_session(self, *args, **kwargs):
        self.calls.append(("set_session", args, kwargs))
        if len(self.calls) <= self.failures_before_success:
            raise TypeError(f"set_session failure {len(self.calls)}")
        return {"ok": True, "args": args, "kwargs": kwargs}

    def exchange_code_for_session(self, *args, **kwargs):
        self.calls.append(("exchange_code_for_session", args, kwargs))
        if len(self.calls) <= self.failures_before_success:
            raise TypeError(f"exchange failure {len(self.calls)}")
        return {"ok": True, "args": args, "kwargs": kwargs}


class _FakeClient:
    def __init__(self, auth):
        self.auth = auth


def test_set_auth_session_fallback_order_uses_second_signature():
    auth = _FakeAuth(failures_before_success=1)
    client = _FakeClient(auth)

    response = _set_auth_session(client, "a", "r")

    assert response["ok"] is True
    assert auth.calls == [
        ("set_session", ("a", "r"), {}),
        (
            "set_session",
            ({"access_token": "a", "refresh_token": "r"},),
            {},
        ),
    ]


def test_set_auth_session_raises_last_exception_when_all_variants_fail():
    auth = _FakeAuth(failures_before_success=3)
    client = _FakeClient(auth)

    with pytest.raises(TypeError, match="set_session failure 3"):
        _set_auth_session(client, "a", "r")

    assert len(auth.calls) == 3
    assert auth.calls[2] == (
        "set_session",
        (),
        {"session": {"access_token": "a", "refresh_token": "r"}},
    )


def test_exchange_auth_code_for_session_fallback_order_uses_third_signature():
    auth = _FakeAuth(failures_before_success=2)
    client = _FakeClient(auth)

    response = _exchange_auth_code_for_session(client, "abc")

    assert response["ok"] is True
    assert auth.calls == [
        ("exchange_code_for_session", ({"auth_code": "abc"},), {}),
        ("exchange_code_for_session", ("abc",), {}),
        ("exchange_code_for_session", (), {"code": "abc"}),
    ]


def test_exchange_auth_code_for_session_raises_last_exception_when_all_variants_fail():
    auth = _FakeAuth(failures_before_success=3)
    client = _FakeClient(auth)

    with pytest.raises(TypeError, match="exchange failure 3"):
        _exchange_auth_code_for_session(client, "abc")

    assert len(auth.calls) == 3


@pytest.mark.parametrize(
    "has_recovery_query,recovery_probe,expected",
    [
        (False, "", True),
        (False, "0", True),
        (False, "1", False),
        (True, "", False),
        (True, "1", False),
    ],
)
def test_should_wait_for_probe(has_recovery_query, recovery_probe, expected):
    assert _should_wait_for_probe(has_recovery_query, recovery_probe) is expected
