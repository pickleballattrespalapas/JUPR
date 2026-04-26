from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from jupr_app.ui import admin_auth


class _FakeAuthApiError(Exception):
    pass


class _FakeQueryParams(dict):
    pass


class _FakeSt:
    def __init__(self):
        self.session_state = {}
        self.query_params = _FakeQueryParams()


class _InvalidRefreshAuth:
    def set_session(self, *_args, **_kwargs):
        raise _FakeAuthApiError("Invalid Refresh Token: Already Used")


class _SuccessAuth:
    def __init__(self, response):
        self._response = response

    def set_session(self, *_args, **_kwargs):
        return self._response


def test_maybe_restore_invalid_refresh_clears_browser_tokens_now(monkeypatch):
    fake_st = _FakeSt()
    fake_st.session_state["admin_auth_session"] = SimpleNamespace(
        access_token="stale-a", refresh_token="stale-r"
    )
    fake_st.query_params.update(
        {
            "jupr_admin_access_token": "stale-a",
            "jupr_admin_refresh_token": "stale-r",
            "jupr_admin_restore_from_storage": "1",
            "page": "admin_login",
        }
    )

    clear_now_calls = []

    monkeypatch.setattr(admin_auth, "st", fake_st)
    monkeypatch.setattr(
        admin_auth,
        "restore_admin_browser_session",
        lambda: {"access_token": "stale-a", "refresh_token": "stale-r"},
    )
    monkeypatch.setattr(
        admin_auth,
        "make_supabase_auth_client",
        lambda: SimpleNamespace(auth=_InvalidRefreshAuth()),
    )
    monkeypatch.setattr(
        admin_auth,
        "render_admin_browser_session_clear_now",
        lambda: clear_now_calls.append("called"),
    )

    restored = admin_auth.maybe_restore_admin_login_from_browser()

    assert restored is False
    assert clear_now_calls == ["called"]
    assert fake_st.session_state.get("_admin_restore_failed_this_run") is True
    assert fake_st.session_state.get("admin_auth_user") is None
    assert fake_st.session_state.get("admin_auth_session") is None
    assert "jupr_admin_access_token" not in fake_st.query_params
    assert "jupr_admin_refresh_token" not in fake_st.query_params
    assert "jupr_admin_restore_from_storage" not in fake_st.query_params


def test_maybe_restore_success_persists_rotated_tokens(monkeypatch):
    fake_st = _FakeSt()
    persisted_tokens = []

    rotated_session = SimpleNamespace(access_token="rotated-a", refresh_token="rotated-r")
    response = SimpleNamespace(
        session=rotated_session,
        user=SimpleNamespace(email="admin@example.com"),
    )

    monkeypatch.setattr(admin_auth, "st", fake_st)
    monkeypatch.setattr(
        admin_auth,
        "restore_admin_browser_session",
        lambda: {"access_token": "old-a", "refresh_token": "old-r"},
    )
    monkeypatch.setattr(
        admin_auth,
        "make_supabase_auth_client",
        lambda: SimpleNamespace(auth=_SuccessAuth(response)),
    )
    monkeypatch.setattr(admin_auth, "load_admin_allowlist", lambda: {"admin@example.com"})
    monkeypatch.setattr(
        admin_auth,
        "persist_admin_browser_session",
        lambda access, refresh: persisted_tokens.append((access, refresh)),
    )

    restored = admin_auth.maybe_restore_admin_login_from_browser()

    assert restored is True
    assert persisted_tokens == [("rotated-a", "rotated-r")]
    assert fake_st.session_state["admin_auth_user"].email == "admin@example.com"
    assert fake_st.session_state["admin_auth_session"].refresh_token == "rotated-r"


def test_streamlit_app_only_restores_browser_tokens_for_admin_entry():
    app = Path("streamlit_app.py").read_text(encoding="utf-8")

    assert "if admin_entry_requested:\n            maybe_restore_admin_login_from_browser()" in app


def test_append_auth_debug_event_redacts_sensitive_reason_and_query_params(monkeypatch):
    fake_st = _FakeSt()
    fake_st.query_params.update(
        {
            "page": "admin_tools",
            "jupr_admin_access_token": "secret-token",
            "jupr_admin_refresh_token": "refresh-token",
            "public": "1",
        }
    )
    monkeypatch.setattr(admin_auth, "st", fake_st)

    admin_auth._append_auth_debug_event(
        "restore_failed",
        success=False,
        reason="Exception while validating access_token value",
    )

    events = fake_st.session_state.get("jupr_auth_debug_events", [])
    assert len(events) == 1
    event = events[0]
    assert event["event_type"] == "restore_failed"
    assert event["success"] is False
    assert event["reason"] == "[REDACTED]"
    assert event["route_query_params"]["jupr_admin_access_token"] == "[REDACTED]"
    assert event["route_query_params"]["jupr_admin_refresh_token"] == "[REDACTED]"


def test_append_auth_debug_event_keeps_only_latest_50(monkeypatch):
    fake_st = _FakeSt()
    monkeypatch.setattr(admin_auth, "st", fake_st)

    for idx in range(60):
        admin_auth._append_auth_debug_event(f"event_{idx}", success=True, reason="")

    events = fake_st.session_state.get("jupr_auth_debug_events", [])
    assert len(events) == 50
    assert events[0]["event_type"] == "event_10"
    assert events[-1]["event_type"] == "event_59"
