from __future__ import annotations

import pytest
from postgrest.exceptions import APIError

from jupr_app.domain.admin.roles import (
    ROLE_CLUB_OWNER,
    ROLE_READ_ONLY,
    ROLE_SCOREKEEPER,
    ROLE_SUPER_ADMIN,
    can_delete_matches,
    can_enter_scores,
    can_manage_matches,
    can_manage_players,
    can_manage_roles,
    can_manage_subscriptions,
    can_manage_tournaments,
    can_run_replay,
    can_view_audit_log,
    resolve_admin_role,
)


class _Query:
    def __init__(self, response_data=None, error: Exception | None = None):
        self._response_data = response_data
        self._error = error

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def execute(self):
        if self._error is not None:
            raise self._error
        return type("Resp", (), {"data": self._response_data})()


class _Supabase:
    def __init__(self, query: _Query):
        self._query = query

    def table(self, _name: str):
        return self._query


def _api_error(code: str) -> APIError:
    return APIError({"message": "boom", "code": code})


def test_resolve_admin_role_uses_assignment_row():
    supabase = _Supabase(_Query(response_data=[{"role": ROLE_CLUB_OWNER}]))
    result = resolve_admin_role(
        supabase=supabase,
        email="owner@example.com",
        user_id=None,
        allowlist={"joe@example.com"},
    )
    assert result.role == ROLE_CLUB_OWNER
    assert result.source == "admin_role_assignments"


def test_resolve_admin_role_missing_table_falls_back_to_allowlist_super_admin():
    supabase = _Supabase(_Query(error=_api_error("42P01")))
    result = resolve_admin_role(
        supabase=supabase,
        email="joe@example.com",
        user_id=None,
        allowlist={"joe@example.com"},
    )
    assert result.role == ROLE_SUPER_ADMIN
    assert result.table_available is False


def test_resolve_admin_role_defaults_to_read_only_without_assignment():
    supabase = _Supabase(_Query(response_data=[]))
    result = resolve_admin_role(
        supabase=supabase,
        email="viewer@example.com",
        user_id=None,
        allowlist={"joe@example.com"},
    )
    assert result.role == ROLE_READ_ONLY
    assert result.source == "default"


@pytest.mark.parametrize(
    "role,expected",
    [
        (
            ROLE_SUPER_ADMIN,
            {
                "can_manage_roles": True,
                "can_manage_players": True,
                "can_manage_matches": True,
                "can_delete_matches": True,
                "can_run_replay": True,
                "can_manage_tournaments": True,
                "can_manage_subscriptions": True,
                "can_view_audit_log": True,
                "can_enter_scores": True,
            },
        ),
        (
            ROLE_CLUB_OWNER,
            {
                "can_manage_roles": False,
                "can_manage_players": True,
                "can_manage_matches": True,
                "can_delete_matches": True,
                "can_run_replay": False,
                "can_manage_tournaments": True,
                "can_manage_subscriptions": True,
                "can_view_audit_log": True,
                "can_enter_scores": True,
            },
        ),
        (
            ROLE_SCOREKEEPER,
            {
                "can_manage_roles": False,
                "can_manage_players": False,
                "can_manage_matches": False,
                "can_delete_matches": False,
                "can_run_replay": False,
                "can_manage_tournaments": False,
                "can_manage_subscriptions": False,
                "can_view_audit_log": False,
                "can_enter_scores": True,
            },
        ),
        (
            ROLE_READ_ONLY,
            {
                "can_manage_roles": False,
                "can_manage_players": False,
                "can_manage_matches": False,
                "can_delete_matches": False,
                "can_run_replay": False,
                "can_manage_tournaments": False,
                "can_manage_subscriptions": False,
                "can_view_audit_log": True,
                "can_enter_scores": False,
            },
        ),
    ],
)
def test_permission_matrix(role: str, expected: dict[str, bool]):
    actual = {
        "can_manage_roles": can_manage_roles(role),
        "can_manage_players": can_manage_players(role),
        "can_manage_matches": can_manage_matches(role),
        "can_delete_matches": can_delete_matches(role),
        "can_run_replay": can_run_replay(role),
        "can_manage_tournaments": can_manage_tournaments(role),
        "can_manage_subscriptions": can_manage_subscriptions(role),
        "can_view_audit_log": can_view_audit_log(role),
        "can_enter_scores": can_enter_scores(role),
    }
    assert actual == expected
