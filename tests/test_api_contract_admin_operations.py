from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi import HTTPException
from fastapi.testclient import TestClient

from services.api.main import app


class _Query:
    def __init__(self, rows, *, should_fail: bool = False):
        self.rows = list(rows)
        self.should_fail = should_fail
        self.filters: dict[str, object] = {}

    def select(self, _fields):
        return self

    def eq(self, field, value):
        self.filters[str(field)] = value
        return self

    def execute(self):
        if self.should_fail:
            raise RuntimeError("private database detail")
        return SimpleNamespace(
            data=[
                dict(row)
                for row in self.rows
                if all(
                    row.get(field) == value
                    for field, value in self.filters.items()
                )
            ]
        )


class _Supabase:
    def __init__(self, rows, *, should_fail: bool = False):
        self.rows = rows
        self.should_fail = should_fail

    def table(self, table_name):
        assert table_name == "admin_role_assignments"
        return _Query(self.rows, should_fail=self.should_fail)


def _configure(
    monkeypatch,
    rows,
    *,
    user_id: str = "user-1",
    should_fail: bool = False,
):
    monkeypatch.setenv("SUPABASE_URL", "https://staging.example.test")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only")
    monkeypatch.setattr(
        "services.api.main.create_client",
        lambda _url, _key: _Supabase(rows, should_fail=should_fail),
    )
    monkeypatch.setattr(
        "services.api.admin_auth_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(
            email="admin@example.com",
            user_id=user_id,
        ),
    )


def test_admin_operations_status_rejects_anonymous_access_before_building(
    monkeypatch,
):
    called = False

    def _unexpected_build():
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(
        "services.api.admin_operations_routes.build_admin_operations_status",
        _unexpected_build,
    )

    response = TestClient(app).get("/admin/operations/status")

    assert response.status_code == 401
    assert response.json() == {"detail": "missing bearer token"}
    assert called is False


def test_admin_operations_status_contract_requires_bound_assignment(monkeypatch):
    _configure(
        monkeypatch,
        [
            {
                "club_id": "tres_palapas",
                "email": "admin@example.com",
                "role": "club_owner",
                "user_id": "user-1",
            }
        ],
    )
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_WRITE_PILOT", "1")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG", "1")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", "1")
    monkeypatch.setenv("JUPR_REQUIRE_API_AUDIT_LOG", "1")

    response = TestClient(app).get(
        "/admin/operations/status?club_id=tres_palapas",
        headers={"Authorization": "Bearer verified-by-test"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "jupr-api"
    assert payload["environment"] == "staging"
    assert payload["write_pilot_enabled"] is True
    assert payload["strict_audit_required"] is True
    assert "match_log" in payload["enabled_workflows"]
    assert payload["pilot_gates"]
    assert payload["permanent_guardrails"]

    workflows = {workflow["key"]: workflow for workflow in payload["workflows"]}
    assert workflows["match_log"]["enabled"] is True
    assert workflows["match_log"]["apply_enabled"] is True
    assert (
        workflows["match_log"]["apply_env_flag"]
        == "JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY"
    )
    assert (
        workflows["score_entry"]["env_flag"]
        == "JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY"
    )
    assert workflows["admin_tools"]["risk"] == "critical"


def test_admin_operations_status_denies_wrong_club_and_user_binding(monkeypatch):
    rows = [
        {
            "club_id": "club-a",
            "email": "admin@example.com",
            "role": "super_admin",
            "user_id": "user-1",
        }
    ]
    _configure(monkeypatch, rows)
    wrong_club = TestClient(app).get(
        "/admin/operations/status?club_id=club-b",
        headers={"Authorization": "Bearer verified-by-test"},
    )
    assert wrong_club.status_code == 403
    assert wrong_club.json() == {"detail": "admin access denied"}
    assert "club-a" not in wrong_club.text

    _configure(monkeypatch, rows, user_id="different-user")
    wrong_user = TestClient(app).get(
        "/admin/operations/status?club_id=club-a",
        headers={"Authorization": "Bearer verified-by-test"},
    )
    assert wrong_user.status_code == 403
    assert wrong_user.json() == {"detail": "admin access denied"}


def test_admin_operations_status_preserves_generic_auth_and_lookup_errors(
    monkeypatch,
):
    _configure(monkeypatch, [], should_fail=True)
    unavailable = TestClient(app).get(
        "/admin/operations/status",
        headers={"Authorization": "Bearer verified-by-test"},
    )
    assert unavailable.status_code == 503
    assert unavailable.json() == {"detail": "admin access check unavailable"}
    assert "private database detail" not in unavailable.text

    monkeypatch.setattr(
        "services.api.admin_auth_routes.authenticate_bearer",
        lambda _authorization: (_ for _ in ()).throw(
            HTTPException(status_code=401, detail="invalid bearer token")
        ),
    )
    invalid = TestClient(app).get(
        "/admin/operations/status",
        headers={"Authorization": "Bearer expired"},
    )
    assert invalid.status_code == 401
    assert invalid.json() == {"detail": "invalid bearer token"}
