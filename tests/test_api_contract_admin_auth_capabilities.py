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
        rows = [
            dict(row)
            for row in self.rows
            if all(row.get(field) == value for field, value in self.filters.items())
        ]
        return SimpleNamespace(data=rows)


class _Supabase:
    def __init__(self, rows, *, should_fail: bool = False):
        self.rows = rows
        self.should_fail = should_fail

    def table(self, table_name):
        assert table_name == "admin_role_assignments"
        return _Query(self.rows, should_fail=self.should_fail)


def _configure(monkeypatch, rows, *, user_id="user-1", should_fail=False):
    monkeypatch.setenv("SUPABASE_URL", "https://staging.example.test")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only")
    monkeypatch.setattr(
        "services.api.main.create_client",
        lambda _url, _key: _Supabase(rows, should_fail=should_fail),
    )
    monkeypatch.setattr(
        "services.api.admin_auth_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="admin@example.com", user_id=user_id),
    )


def test_capabilities_verify_jwt_assignment_and_requested_club(monkeypatch):
    _configure(
        monkeypatch,
        [
            {"club_id": "club-a", "email": "admin@example.com", "role": "scorekeeper", "user_id": "user-1"},
            {"club_id": "club-b", "email": "admin@example.com", "role": "read_only", "user_id": None},
            {"club_id": "club-a", "email": "other@example.com", "role": "super_admin", "user_id": "other"},
        ],
    )

    response = TestClient(app).get(
        "/admin/auth/capabilities?club_id=club-a",
        headers={"Authorization": "Bearer verified-by-test"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "authorized": True,
        "user": {"email": "admin@example.com"},
        "requested_club_id": "club-a",
        "assignments": [
            {
                "club_id": "club-a",
                "role": "scorekeeper",
                "permissions": ["enter_scores"],
            }
        ],
    }
    assert "user_id" not in response.text


def test_capabilities_deny_wrong_club_without_enumerating_assignment(monkeypatch):
    _configure(
        monkeypatch,
        [{"club_id": "club-a", "email": "admin@example.com", "role": "club_owner", "user_id": "user-1"}],
    )

    response = TestClient(app).get(
        "/admin/auth/capabilities?club_id=club-b",
        headers={"Authorization": "Bearer verified-by-test"},
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "admin access denied"}
    assert "club-a" not in response.text


def test_capabilities_deny_assignment_bound_to_another_auth_user(monkeypatch):
    _configure(
        monkeypatch,
        [{"club_id": "club-a", "email": "admin@example.com", "role": "super_admin", "user_id": "different-user"}],
    )

    response = TestClient(app).get(
        "/admin/auth/capabilities",
        headers={"Authorization": "Bearer verified-by-test"},
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "admin access denied"}


def test_capabilities_preserve_generic_jwt_and_backend_errors(monkeypatch):
    _configure(monkeypatch, [], should_fail=True)
    unavailable = TestClient(app).get(
        "/admin/auth/capabilities",
        headers={"Authorization": "Bearer verified-by-test"},
    )
    assert unavailable.status_code == 503
    assert unavailable.json() == {"detail": "admin access check unavailable"}
    assert "private database detail" not in unavailable.text

    monkeypatch.setattr(
        "services.api.admin_auth_routes.authenticate_bearer",
        lambda _authorization: (_ for _ in ()).throw(HTTPException(status_code=401, detail="invalid bearer token")),
    )
    invalid = TestClient(app).get(
        "/admin/auth/capabilities",
        headers={"Authorization": "Bearer expired"},
    )
    assert invalid.status_code == 401
    assert invalid.json() == {"detail": "invalid bearer token"}
